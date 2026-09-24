"""H1 左右镜像（DEC-009 第 6 项），供 rsl-rl symmetry data augmentation 使用。

镜像 = 关于矢状面（base 系 y → −y）的反射 M = diag(1, −1, 1)：
  - 普通矢量（线速度、重力投影、位置）：(x, y, z) → (x, −y, z)
  - 角速度（伪矢量）：ω → −Mω = (−x, y, −z)
  - 速度指令 (vx, vy, wz) → (vx, −vy, −wz)
  - 关节：左右对调；绕 x / z 轴的关节（hip_roll、hip_yaw、shoulder_roll、shoulder_yaw、torso）取负，
    绕 y 轴的关节（hip_pitch、knee、ankle、shoulder_pitch、elbow）不变。依据：H1 MJCF 左右关节轴定义相同（腿），
    手臂父系绕 x 轴 ±0.436 rad 镜像倾斜；carry15 默认角 roll / yaw 左右异号、pitch / elbow 同号，与此一致。
  - 箱子相对姿态 R → M R M：第 1 列 → M c0，第 2 列 → −M c1
  - 步态时钟：左右互换 = 相位平移 0.5 → (sin, cos) → (−sin, −cos)
  - 上身参考修正 Δ（双手对称内夹、上下）本身左右对称，不变；两臂负载质量对调。
索引按观测项名字与关节名在首次调用时构建，任何未知观测项都会报错，避免静默错位。
"""
import torch
from tensordict import TensorDict

__all__ = ["compute_symmetric_states", "build_maps"]

NEG_KEYS = ("hip_roll", "hip_yaw", "shoulder_roll", "shoulder_yaw", "torso")


def _joint_perm_sign(names):
    perm, sign = [], []
    for n in names:
        m = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        perm.append(names.index(m))
        sign.append(-1.0 if any(k in n for k in NEG_KEYS) else 1.0)
    return perm, sign


def _vec(sx, sy, sz):
    return list(range(3)), [sx, sy, sz]


def build_maps(env):
    """返回 {group: (perm, sign)} 与动作的 (perm, sign)：mirrored[:, i] = sign[i] · x[:, perm[i]]。"""
    u = env.unwrapped
    robot = u.scene["robot"]
    joints = list(robot.joint_names)
    upper = u.action_manager.get_term("upper_body")
    leg_names = list(u.action_manager.get_term("joint_pos")._joint_names)
    upper_names = [joints[i] for i in upper.adapter.upper_joint_idx]
    jp, js = _joint_perm_sign(joints)
    lp, ls = _joint_perm_sign(leg_names)
    up, us = _joint_perm_sign(upper_names)
    act_perm = lp + [len(lp) + i for i in range(u.action_manager.total_action_dim - len(lp))]
    act_sign = ls + [1.0] * (u.action_manager.total_action_dim - len(lp))
    term_map = {
        "base_lin_vel": _vec(1, -1, 1),
        "base_ang_vel": _vec(-1, 1, -1),
        "projected_gravity": _vec(1, -1, 1),
        "velocity_commands": _vec(1, -1, -1),
        "joint_pos": (jp, js),
        "joint_vel": (jp, js),
        "actions": (act_perm, act_sign),
        "payload_mass": ([1, 0], [1.0, 1.0]),
        "upper_targets": (up, us),
        "box_pose": (list(range(9)), [1, -1, 1, 1, -1, 1, -1, 1, -1]),
        "gait_clock": ([0, 1], [-1.0, -1.0]),
    }
    om = u.observation_manager
    maps = {}
    for g in om.active_terms:
        perm, sign, off = [], [], 0
        for name, dim in zip(om.active_terms[g], om.group_obs_term_dim[g]):
            if name not in term_map:
                raise KeyError(f"镜像映射缺少观测项 {g}/{name}")
            p, s = term_map[name]
            if len(p) != dim[0]:
                raise ValueError(f"观测项 {g}/{name} 维度 {dim} 与镜像映射 {len(p)} 不符")
            perm += [off + i for i in p]
            sign += s
            off += dim[0]
        maps[g] = (perm, sign)
    return maps, (act_perm, act_sign)


def _apply(x, ps):
    perm, sign = ps
    return x[..., perm] * torch.as_tensor(sign, dtype=x.dtype, device=x.device)


@torch.no_grad()
def compute_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """rsl-rl 接口：返回 [原始; 左右镜像] 两份。"""
    u = env.unwrapped
    if not hasattr(u, "_h1_mirror_maps"):
        u._h1_mirror_maps = build_maps(env)
    obs_maps, act_map = u._h1_mirror_maps
    obs_aug = None
    if obs is not None:
        b = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for g in obs.keys():
            obs_aug[g][:b] = obs[g]
            obs_aug[g][b:] = _apply(obs[g], obs_maps[g])
    act_aug = None
    if actions is not None:
        act_aug = torch.cat([actions, _apply(actions, act_map)], 0)
    return obs_aug, act_aug
