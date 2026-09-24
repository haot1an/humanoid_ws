"""UBQP-001 的 torch 批量版：与 C++ `src/humanoid_wbc/src/upper_body_qp.cpp` 逐项对应。

    决策变量 x = dq_u（B, nu）
    min Σ w_i (A_i x − b_i)² + w_posture ‖x − k_posture (q_nominal − q_des)‖² + w_reg ‖x‖²
    s.t. max(−dq_max, (q_min − q_des)/dt) ≤ x ≤ min(dq_max, (q_max − q_des)/dt)
    每臂 4 行：手部位置 3 行（反馈 + 基座前馈）+ 前臂俯仰 1 行（J_pitch = (a × e_z)ᵀ J_rot，用户推导实现）
    输出 q_des += x·dt，再 clamp 到 q ± delta_max；dq_des = x

约定：Jacobian 的列与 v 使用同一种广义速度约定（MuJoCo 或 Isaac 由适配层保证），
     upper_dof 给出上肢 nu 个关节在 v 中的列号，顺序即决策变量顺序。
"""
from dataclasses import dataclass

import torch

from .box_qp import solve_box_qp

ROWS_PER_ARM = 4
NUM_ARMS = 2
GRAVITY = 9.81


@dataclass
class UpperBodyQpCfg:
    """默认值与 C++ UpperBodyQpConfig 相同（DESIGN UBQP-001）。"""
    dt: float = 0.01
    k_hand: float = 20.0
    k_pitch: float = 10.0
    w_hand: float = 100.0
    w_pitch: float = 0.0
    w_posture: float = 0.1
    k_posture: float = 2.0
    w_reg: float = 1e-4
    dq_max: float = 5.0
    delta_max: float = 0.2
    feedforward: bool = True


@dataclass
class UpperBodyKinematicsBatch:
    """全部为 world 系；Jacobian 列约定与 v 一致。"""
    hand_pos: torch.Tensor      # (B, 2, 3)
    forearm_axis: torch.Tensor  # (B, 2, 3) 前臂单位向量 a
    J_hand: torch.Tensor        # (B, 2, 3, nv)
    J_rot: torch.Tensor         # (B, 2, 3, nv) elbow_link 角速度
    J_load: torch.Tensor        # (B, 2, 3, nv) 负载作用点线速度
    bias: torch.Tensor          # (B, nv) 纯重力 g(q)，电机需提供的保持力为正（RL-LOCO-001 口径）


class BatchedUpperBodyQp:
    def __init__(self, num_envs, nv, upper_dof, q_min, q_max, cfg: UpperBodyQpCfg, device="cpu",
                 dtype=torch.float64):
        self.cfg = cfg
        self.B, self.nv, self.nu = num_envs, nv, len(upper_dof)
        self.device, self.dtype = device, dtype
        self.upper_dof = torch.as_tensor(upper_dof, dtype=torch.long, device=device)
        if int(self.upper_dof.min()) < 6 or int(self.upper_dof.max()) >= nv:
            raise ValueError("upper_dof 越界或包含浮动基的列")
        self.q_min = torch.as_tensor(q_min, dtype=dtype, device=device).expand(self.B, self.nu).clone()
        self.q_max = torch.as_tensor(q_max, dtype=dtype, device=device).expand(self.B, self.nu).clone()
        if not bool((self.q_min < self.q_max).all()):
            raise ValueError("q_min 必须小于 q_max")
        w = torch.zeros(NUM_ARMS * ROWS_PER_ARM, dtype=dtype, device=device)
        for s in range(NUM_ARMS):
            w[s * ROWS_PER_ARM: s * ROWS_PER_ARM + 3] = cfg.w_hand
            w[s * ROWS_PER_ARM + 3] = cfg.w_pitch
        self.w_task = w
        self.rest_mask = torch.ones(nv, dtype=dtype, device=device)
        self.rest_mask[self.upper_dof] = 0.0  # v_rest = v * rest_mask：上肢列置零，只剩基座（与腿）
        self.q_des = torch.full((self.B, self.nu), float("nan"), dtype=dtype, device=device)
        self.dq_des = torch.zeros((self.B, self.nu), dtype=dtype, device=device)
        self.task_residual = torch.zeros((self.B, NUM_ARMS * ROWS_PER_ARM), dtype=dtype, device=device)
        self.unconverged = 0

    def reset(self, env_ids, q_upper_measured):
        """进入 QP 模式 / episode 重置时：q_des 从给定关节角开始积分。"""
        self.q_des[env_ids] = q_upper_measured.to(self.dtype)
        self.dq_des[env_ids] = 0.0

    def _upper(self, J):
        return J.index_select(-1, self.upper_dof)

    def build_tasks(self, kin: UpperBodyKinematicsBatch, ref_pos, ref_vel, v):
        cfg = self.cfg
        v_rest = v * self.rest_mask
        A = torch.zeros((self.B, NUM_ARMS * ROWS_PER_ARM, self.nu), dtype=self.dtype, device=self.device)
        b = torch.zeros((self.B, NUM_ARMS * ROWS_PER_ARM), dtype=self.dtype, device=self.device)
        e_z = torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        for s in range(NUM_ARMS):
            r = s * ROWS_PER_ARM
            # 手部位置：J_u dq_u = v_ref + k (p_ref − p) − J_hand v_rest
            J = kin.J_hand[:, s]
            v_des = ref_vel[:, s] + cfg.k_hand * (ref_pos[:, s] - kin.hand_pos[:, s])
            ff = (J @ v_rest.unsqueeze(-1)).squeeze(-1) if cfg.feedforward else torch.zeros_like(v_des)
            A[:, r:r + 3] = self._upper(J)
            b[:, r:r + 3] = v_des - ff
            # 前臂俯仰：J_pitch = (a × e_z)ᵀ J_rot；目标 ȧ_z = k_pitch (0 − a_z) − J_pitch v_rest
            if cfg.w_pitch > 0.0:
                a = kin.forearm_axis[:, s]
                c = torch.cross(a, e_z.expand_as(a), dim=-1)
                J_pitch = (c.unsqueeze(-2) @ kin.J_rot[:, s]).squeeze(-2)          # (B, nv)
                az_dot_des = cfg.k_pitch * (0.0 - a[:, 2])
                az_ff = (J_pitch * v_rest).sum(-1) if cfg.feedforward else torch.zeros_like(az_dot_des)
                A[:, r + 3] = self._upper(J_pitch)
                b[:, r + 3] = az_dot_des - az_ff
        return A, b

    def update(self, kin: UpperBodyKinematicsBatch, ref_pos, ref_vel, v, q_upper, q_nominal):
        cfg = self.cfg
        if torch.isnan(self.q_des).any():
            raise RuntimeError("BatchedUpperBodyQp.update: 未 reset，q_des 仍为 NaN")
        A, b = self.build_tasks(kin, ref_pos, ref_vel, v)
        W = self.w_task
        AtW = A.transpose(-1, -2) * W                                                # (B, nu, 8)
        I = torch.eye(self.nu, dtype=self.dtype, device=self.device)
        P = 2.0 * (AtW @ A + (cfg.w_posture + cfg.w_reg) * I)
        dq_posture = cfg.k_posture * (q_nominal - self.q_des)
        g = -2.0 * ((AtW @ b.unsqueeze(-1)).squeeze(-1) + cfg.w_posture * dq_posture)
        lo = ((self.q_min - self.q_des) / cfg.dt).clamp(min=-cfg.dq_max)
        hi = ((self.q_max - self.q_des) / cfg.dt).clamp(max=cfg.dq_max)
        hi = torch.maximum(hi, lo)  # q_des 越界超过一拍可回退量时，只允许朝限位内侧移动
        x, _, self.unconverged = solve_box_qp(P, g, lo, hi)
        self.task_residual = (A @ x.unsqueeze(-1)).squeeze(-1) - b
        self.dq_des = x
        q_des = self.q_des + x * cfg.dt
        self.q_des = torch.minimum(torch.maximum(q_des, q_upper - cfg.delta_max), q_upper + cfg.delta_max)
        return x

    def tau_feedforward(self, kin: UpperBodyKinematicsBatch, payload_mass):
        """重力 g(q) 上肢分量 + 已知负载前馈 Σ J_load(:, upper)ᵀ (0, 0, m/2·g)。payload_mass: (B,)。"""
        f_z = 0.5 * payload_mass.to(self.dtype) * GRAVITY                           # (B,)
        J_load_z = kin.J_load[:, :, 2, :].sum(1)                                    # (B, nv)：两臂 z 行之和
        return (kin.bias + J_load_z * f_z.unsqueeze(-1)).index_select(-1, self.upper_dof)
