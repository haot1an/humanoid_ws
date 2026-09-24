"""上肢 QP action term：不接收策略输出（action_dim = 0），在每个物理步写入 torso + 双臂的
Unitree 五元组目标（q_des, dq_des, τ_ff；kp / kd 由 IdealPD 执行器配置给出）。

    上肢控制频率 100 Hz（DEC-003）：物理步 0.005 s，每 2 个物理步解一次 QP
    reference：名义载体系 = 基座位姿一阶低通（截止 0.5 Hz），手部目标 = 载体系中的固定偏移（RL-LOCO-001 v2）
               一阶低通对匀速运动有稳态滞后 v·τ（实测 0.53 m/s 滞后 165 mm，箱子被拉进躯干，EXP-017 之后发现）。
               carrier_lead = "command"：用【指令】速度前馈超前 τ·v_cmd（同一低通滤过，稳态恰好抵消滞后、指令跳变时
               不产生目标阶跃）。不能用实测速度：部署实测会形成“手前移 → 身体前倾 → 加速”的正反馈（已实测失败）。
    负载：reset 时由 events.reset_payload 采样并写入 elbow_link 质量；控制器侧模型不含负载
          （先从 Isaac 真实 g(q) 中减去负载项，再按“已知负载”加回前馈，避免控制器使用仿真真值）
"""
from dataclasses import MISSING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from ..isaac_kinematics import (NV_REDUCED, UPPER_DOF_REDUCED, UPPER_JOINTS, IsaacUpperBodyAdapter,
                                quat_xyzw_to_matrix)
from ..upper_body_qp import GRAVITY, BatchedUpperBodyQp, UpperBodyKinematicsBatch, UpperBodyQpCfg

# carry15 标称姿态下、骨盆竖直时的手部点相对骨盆原点的位置（scripts/view_carry_poses.py --thetas 15 --spacing 0.36：
# 手 (0.369449, ±0.18, 1.168187)，骨盆 (0, 0, 1.05)）
HAND_OFFSET_B = ((0.369449, 0.18, 0.118187), (0.369449, -0.18, 0.118187))
LIMIT_MARGIN = 0.05


def _yaw_from_quat_xyzw(q):
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class UpperBodyQpAction(ActionTerm):
    cfg: "UpperBodyQpActionCfg"

    def __init__(self, cfg: "UpperBodyQpActionCfg", env):
        super().__init__(cfg, env)
        robot = self._asset
        dev = self.device  # num_envs 由基类 ManagerTermBase 提供（只读属性）
        self.dtype = torch.float64 if cfg.float64 else torch.float32
        self.adapter = IsaacUpperBodyAdapter(list(robot.joint_names), list(robot.body_names), dtype=self.dtype)
        self.joint_ids = self.adapter.upper_joint_idx
        self.elbow_body_ids = self.adapter.elbow_body
        limits = robot.data.joint_pos_limits.torch[0, self.joint_ids].to(self.dtype)
        self.qp = BatchedUpperBodyQp(self.num_envs, NV_REDUCED, UPPER_DOF_REDUCED,
                                     limits[:, 0] + LIMIT_MARGIN, limits[:, 1] - LIMIT_MARGIN,
                                     UpperBodyQpCfg(**cfg.qp), device=dev, dtype=self.dtype)
        self.q_nominal = robot.data.default_joint_pos.torch[:, self.joint_ids].to(self.dtype).clone()
        self.decimation = int(round(self.qp.cfg.dt / env.physics_dt))
        if self.decimation < 1 or abs(self.decimation * env.physics_dt - self.qp.cfg.dt) > 1e-9:
            raise ValueError(f"上肢控制周期 {self.qp.cfg.dt} 不是物理步 {env.physics_dt} 的整数倍")
        tau = 1.0 / (2.0 * torch.pi * cfg.carrier_cutoff_hz)
        self.alpha = 1.0 - torch.exp(torch.tensor(-self.qp.cfg.dt / tau)).item()
        self.hand_offset_b = torch.tensor(HAND_OFFSET_B, dtype=self.dtype, device=dev)          # (2, 3)
        self.carrier_pos = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=dev)
        self.carrier_yaw = torch.zeros(self.num_envs, dtype=self.dtype, device=dev)
        self.carrier_tau = tau
        self.lead_pos = torch.zeros((self.num_envs, 3), dtype=self.dtype, device=dev)                # 指令前馈超前量
        self.lead_yaw = torch.zeros(self.num_envs, dtype=self.dtype, device=dev)
        if cfg.carrier_lead not in ("none", "command"):
            raise ValueError(f"carrier_lead 只能是 none / command，收到 {cfg.carrier_lead}")
        self.ref_prev = torch.zeros((self.num_envs, 2, 3), dtype=self.dtype, device=dev)
        self.payload_mass = torch.zeros(self.num_envs, dtype=self.dtype, device=dev)            # 箱子总质量 [kg]
        self.payload_cap = float(cfg.payload_cap_init)                                           # 课程修改
        self.tau_ff = torch.zeros((self.num_envs, len(self.joint_ids)), dtype=self.dtype, device=dev)
        self.hand_err = torch.zeros((self.num_envs, 2, 3), dtype=self.dtype, device=dev)          # 评估用
        self._step = 0
        self._raw = torch.zeros((self.num_envs, 2 if cfg.ref_delta else 0), device=dev)
        # DEC-008：策略输出的上身参考修正（双手对称内夹 Δin、上下 Δz），经一阶低通后加到跟随坐标系中的手部偏移上
        self.delta_goal = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=dev)
        self.delta = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=dev)
        self.delta_alpha = 1.0 - torch.exp(torch.tensor(-self.qp.cfg.dt / cfg.delta_tau)).item()
        # v1（对照组，RL-LOCO-001 第 5 项）：上肢不接 QP，目标 = carry15 + 随机偏移，每 T ~ U(1,3) s 重采样并一阶平滑
        self.rand_target = self.q_nominal.clone()
        self.rand_goal = self.q_nominal.clone()
        self.rand_timer = torch.zeros(self.num_envs, dtype=self.dtype, device=dev)

    # ---- ActionTerm 接口：策略不控制上肢
    @property
    def action_dim(self) -> int:
        return 2 if self.cfg.ref_delta else 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor):
        if not self.cfg.ref_delta:
            return
        self._raw[:] = actions
        a = actions.to(self.dtype)
        self.delta_goal[:, 0] = torch.clamp(self.cfg.delta_scale * a[:, 0], 0.0, self.cfg.delta_in_max)
        self.delta_goal[:, 1] = torch.clamp(self.cfg.delta_scale * a[:, 1], -self.cfg.delta_z_max, self.cfg.delta_z_max)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        d = self._asset.data
        if self.cfg.mode == "random":
            self.rand_target[env_ids] = self.q_nominal[env_ids]
            self.rand_goal[env_ids] = self.q_nominal[env_ids]
            self.rand_timer[env_ids] = 0.0
        self.carrier_pos[env_ids] = d.root_link_pos_w.torch[env_ids].to(self.dtype)
        self.carrier_yaw[env_ids] = _yaw_from_quat_xyzw(d.root_link_quat_w.torch[env_ids].to(self.dtype))
        self.lead_pos[env_ids] = 0.0
        self.lead_yaw[env_ids] = 0.0
        self.delta_goal[env_ids] = 0.0
        self.delta[env_ids] = 0.0
        self.ref_prev[env_ids] = self._hand_targets()[env_ids]
        self.qp.reset(env_ids, self.q_nominal[env_ids])

    # ---- 每个物理步调用
    def apply_actions(self):
        if self._step % self.decimation == 0:
            self._control_tick()
        self._step += 1
        robot = self._asset
        robot.set_joint_position_target_index(target=self.qp.q_des.float(), joint_ids=self.joint_ids)
        robot.set_joint_velocity_target_index(target=self.qp.dq_des.float(), joint_ids=self.joint_ids)
        robot.set_joint_effort_target_index(target=self.tau_ff.float(), joint_ids=self.joint_ids)

    def _hand_targets(self):
        yaw = self.carrier_yaw + self.lead_yaw
        c, s = torch.cos(yaw), torch.sin(yaw)
        R = torch.zeros((self.num_envs, 3, 3), dtype=self.dtype, device=self.device)
        R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
        off = self.hand_offset_b.unsqueeze(0).expand(self.num_envs, 2, 3).clone()
        if self.cfg.ref_delta:  # 左手 y > 0：向内 = y 减小；右手相反
            off[:, 0, 1] -= self.delta[:, 0]
            off[:, 1, 1] += self.delta[:, 0]
            off[:, :, 2] += self.delta[:, 1:2]
        return (self.carrier_pos + self.lead_pos).unsqueeze(1) + torch.einsum("bij,baj->bai", R, off)

    def _update_carrier(self, root_pos, root_quat):
        if self.cfg.ref_delta:
            self.delta += self.delta_alpha * (self.delta_goal - self.delta)
        """载体系低通 + 指令速度前馈超前。指令 (vx, vy, wz) 在基座 yaw 系，旋到世界系后乘 τ 再经同一低通。"""
        self.carrier_pos += self.alpha * (root_pos - self.carrier_pos)
        dyaw = _yaw_from_quat_xyzw(root_quat.to(self.dtype)) - self.carrier_yaw
        self.carrier_yaw += self.alpha * torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        if self.cfg.carrier_lead != "command":
            return
        cmd = self._env.command_manager.get_command(self.cfg.command_name).to(self.dtype)
        c, s = torch.cos(self.carrier_yaw), torch.sin(self.carrier_yaw)
        goal = torch.zeros_like(self.lead_pos)
        goal[:, 0] = self.carrier_tau * (c * cmd[:, 0] - s * cmd[:, 1])
        goal[:, 1] = self.carrier_tau * (s * cmd[:, 0] + c * cmd[:, 1])
        self.lead_pos += self.alpha * (goal - self.lead_pos)
        self.lead_yaw += self.alpha * (self.carrier_tau * cmd[:, 2] - self.lead_yaw)

    def _random_tick(self):
        """v1：上肢关节目标做平滑随机游走；tau_ff 仍为重力 + 已知负载前馈（与 v2 相同口径）。"""
        d = self._asset.data
        dt = self.qp.cfg.dt
        self.rand_timer -= dt
        due = self.rand_timer <= 0.0
        if bool(due.any()):
            n = int(due.sum())
            amp = self.cfg.random_amplitude
            offset = (torch.rand((n, self.q_nominal.shape[1]), device=self.device, dtype=self.dtype) * 2 - 1) * amp
            self.rand_goal[due] = self.q_nominal[due] + offset
            self.rand_timer[due] = 1.0 + 2.0 * torch.rand(n, device=self.device, dtype=self.dtype)
        tau_lp = 1.0 / (2.0 * torch.pi * self.cfg.random_cutoff_hz)
        alpha = 1.0 - torch.exp(torch.tensor(-dt / tau_lp)).item()
        self.rand_target += alpha * (self.rand_goal - self.rand_target)
        lo, hi = self.qp.q_min, self.qp.q_max
        self.qp.q_des = torch.clamp(self.rand_target, lo, hi)
        self.qp.dq_des = torch.zeros_like(self.qp.q_des)
        # 仅用于日志：手部误差仍相对名义载体系
        root_pos = d.root_link_pos_w.torch.to(self.dtype)
        root_quat = d.root_link_quat_w.torch
        self._update_carrier(root_pos, root_quat)
        kin = self.adapter.kinematics(d.body_link_pos_w.torch, d.body_link_quat_w.torch, d.body_link_jacobian_w.torch,
                                      d.gravity_compensation_forces.torch, root_pos, root_quat)
        g_payload = self._payload_gravity(d, kin)
        self.tau_ff = self.qp.tau_feedforward(
            UpperBodyKinematicsBatch(kin.hand_pos, kin.forearm_axis, kin.J_hand, kin.J_rot, kin.J_load,
                                     kin.bias - g_payload),
            self.payload_mass if self.cfg.payload_feedforward else torch.zeros_like(self.payload_mass))
        self.hand_err = self._hand_targets() - kin.hand_pos

    def _payload_gravity(self, d, kin):
        """Isaac 的 g(q) 含负载（加在 elbow_link 质心），控制器侧模型不应知道它 → 先减去。"""
        g_payload = torch.zeros_like(kin.bias)
        if not self.cfg.payload_in_elbow:  # DEC-007：真实箱子不在 elbow_link 质量里，Isaac g(q) 不含负载
            return g_payload
        com_jac = d.body_com_jacobian_w.torch
        for b in self.elbow_body_ids:
            Jz_upper = com_jac[:, b, 2, self.adapter.upper_cols].to(self.dtype)
            g_payload[:, 6:] += Jz_upper * (0.5 * self.payload_mass * GRAVITY).unsqueeze(-1)
        return g_payload

    def _control_tick(self):
        if self.cfg.mode == "random":
            self._random_tick()
            return
        d = self._asset.data
        dt = self.qp.cfg.dt
        # 名义载体系：基座位姿一阶低通（yaw 走最短角差）
        root_pos = d.root_link_pos_w.torch.to(self.dtype)
        root_quat = d.root_link_quat_w.torch
        self._update_carrier(root_pos, root_quat)
        ref_pos = self._hand_targets()
        ref_vel = (ref_pos - self.ref_prev) / dt
        self.ref_prev = ref_pos

        kin = self.adapter.kinematics(d.body_link_pos_w.torch, d.body_link_quat_w.torch, d.body_link_jacobian_w.torch,
                                      d.gravity_compensation_forces.torch, root_pos, root_quat)
        # 控制器侧模型不含负载：从真实 g(q) 中减去负载项（负载加在 elbow_link 质心上）
        g_payload = self._payload_gravity(d, kin)
        kin_nominal = UpperBodyKinematicsBatch(kin.hand_pos, kin.forearm_axis, kin.J_hand, kin.J_rot, kin.J_load,
                                               kin.bias - g_payload)
        v = self.adapter.reduced_velocity(d.root_link_lin_vel_w.torch, d.root_link_ang_vel_b.torch, d.joint_vel.torch)
        q_upper = d.joint_pos.torch[:, self.joint_ids].to(self.dtype)
        self.qp.update(kin_nominal, ref_pos, ref_vel, v, q_upper, self.q_nominal)
        m_known = self.payload_mass if self.cfg.payload_feedforward else torch.zeros_like(self.payload_mass)
        self.tau_ff = self.qp.tau_feedforward(kin_nominal, m_known)
        self.hand_err = ref_pos - kin.hand_pos


@configclass
class UpperBodyQpActionCfg(ActionTermCfg):
    class_type: type = UpperBodyQpAction
    asset_name: str = MISSING
    qp: dict = {"dt": 0.01, "k_hand": 10.0, "k_pitch": 5.0, "w_pitch": 10.0}  # EXP-007 带负载时 k_hand 20 自激
    carrier_cutoff_hz: float = 0.5
    carrier_lead: str = "none"       # "none" = EXP-011–017 口径（有 v·τ 滞后）；"command" = 指令速度前馈（EXP-018 起）
    command_name: str = "base_velocity"
    payload_cap_init: float = 5.0   # 课程未接入时的负载上限 [kg]
    payload_feedforward: bool = True
    payload_in_elbow: bool = True
    ref_delta: bool = False          # DEC-008：策略额外输出 2 维上身参考修正（v4）
    delta_scale: float = 0.03        # 原始动作 → 米
    delta_in_max: float = 0.06       # 内夹量上限 [m]（只向内）
    delta_z_max: float = 0.05        # 上下修正上限 [m]
    delta_tau: float = 0.05          # 修正量一阶低通时间常数 [s]，避免 50 Hz 阶跃造成参考速度尖峰    # True = 负载写在 elbow_link 质量里（v2）；False = 真实箱子刚体（v3，DEC-007）
    float64: bool = False
    mode: str = "qp"                 # "qp" = v2（QP 在训练回路）；"random" = v1 对照组
    random_amplitude: float = 0.3    # v1 随机偏移幅度 [rad]（RL-LOCO-001 第 7 项课程在 v1 中调制该值）
    random_cutoff_hz: float = 0.5    # v1 目标平滑的一阶低通截止频率
