"""loco-manipulation 专用 MDP 项（事件、观测、奖励、课程）。"""
import torch

from isaaclab.managers import SceneEntityCfg

from ..isaac_kinematics import quat_xyzw_to_matrix

UPPER_TERM = "upper_body"
# DEC-007 真实箱子（深 × 宽 × 高，x 沿前进方向）
BOX_SIZE = (0.20, 0.44, 0.20)
# carry15 标称姿态、骨盆竖直时箱子中心相对骨盆原点的位置（yaw 系）：两前臂负载点中点 + (前臂半径 0.025 + 半高 0.10)·e_z。
# 取自 MuJoCo 模型（sim2sim_carry.py box_pose() 在 reset 后的值），Isaac 端由冒烟测试核对落定后的相对位置。
BOX_OFFSET_B = (0.239047, 0.0, 0.243187)


def _upper(env):
    return env.action_manager.get_term(UPPER_TERM)


# ---------------- 事件
def reset_payload(env, env_ids, asset_cfg: SceneEntityCfg):
    """每个 episode 开始：箱子总质量 m ~ U(0, cap)，两个 elbow_link 各加 m/2（加在质心上，惯量不变）。"""
    term, robot = _upper(env), env.scene[asset_cfg.name]
    if not hasattr(term, "_elbow_nominal_mass"):
        term._elbow_nominal_mass = robot.data.body_mass.torch[:, asset_cfg.body_ids].clone()
    env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    m = torch.rand(len(env_ids), device=env.device) * term.payload_cap
    term.payload_mass[env_ids] = m.to(term.payload_mass.dtype)
    masses = term._elbow_nominal_mass[env_ids] + 0.5 * m.unsqueeze(-1)
    body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int32, device=env.device)
    robot.set_masses_index(masses=masses, body_ids=body_ids, env_ids=env_ids.to(torch.int32))


def _yaw_quat_xyzw(q):
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw


def reset_box(env, env_ids, empty_mass=0.3, spawn_margin=0.02, box_name="box", robot_name="robot"):
    """DEC-007 第 2 / 5 项：箱子质量 = 空箱 + U(0, cap)（均匀密度惯量），并按机器人刚重置的基座位姿把箱子水平放在
    两前臂上方 spawn_margin 处，速度取与基座刚体运动一致的值（reset_base 带随机初速度）。
    必须排在 reset_base / reset_robot_joints 之后（events 按注册顺序执行）；关节 reset 为默认角的 1.0 倍，
    所以前臂位置由基座位姿唯一确定。"""
    term, robot, box = _upper(env), env.scene[robot_name], env.scene[box_name]
    env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    n = len(env_ids)
    m = empty_mass + torch.rand(n, device=env.device) * term.payload_cap
    term.payload_mass[env_ids] = m.to(term.payload_mass.dtype)
    ids32 = env_ids.to(torch.int32)
    box.set_masses_index(masses=m.unsqueeze(-1), env_ids=ids32)
    a, b, c = BOX_SIZE
    inertia = torch.zeros((n, 1, 9), device=env.device)
    inertia[:, 0, 0] = m * (b * b + c * c) / 12.0
    inertia[:, 0, 4] = m * (a * a + c * c) / 12.0
    inertia[:, 0, 8] = m * (a * a + b * b) / 12.0
    box.set_inertias_index(inertias=inertia, env_ids=ids32)
    d = robot.data
    root_pos = d.root_link_pos_w.torch[env_ids]
    yaw = _yaw_quat_xyzw(d.root_link_quat_w.torch[env_ids])
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    ox, oy, oz = BOX_OFFSET_B
    oz = oz + spawn_margin
    pos = root_pos + torch.stack([cy * ox - sy * oy, sy * ox + cy * oy, torch.full_like(yaw, oz)], -1)
    quat = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)], -1)
    box.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], -1), env_ids=ids32)
    w = d.root_link_ang_vel_w.torch[env_ids]
    v = d.root_link_lin_vel_w.torch[env_ids] + torch.cross(w, pos - root_pos, dim=-1)
    box.write_root_velocity_to_sim_index(root_velocity=torch.cat([v, w], -1), env_ids=ids32)


def _load_point_mid(env, robot_name="robot"):
    term, d = _upper(env), env.scene[robot_name].data
    pos = d.body_link_pos_w.torch[:, term.elbow_body_ids]
    R = quat_xyzw_to_matrix(d.body_link_quat_w.torch[:, term.elbow_body_ids])
    load = pos + (R @ term.adapter.load_local.to(pos).unsqueeze(-1)).squeeze(-1)
    return load.mean(1)


# ---------------- 奖励（v4）
def box_slip_l2(env, box_name="box", robot_name="robot"):
    """DEC-008 第 4 项：箱子相对两前臂负载点中点的速度平方 [(m/s)²]。负载点速度 = elbow 线速度 + ω × r。"""
    term, d = _upper(env), env.scene[robot_name].data
    ids = term.elbow_body_ids
    pos = d.body_link_pos_w.torch[:, ids]
    R = quat_xyzw_to_matrix(d.body_link_quat_w.torch[:, ids])
    r = (R @ term.adapter.load_local.to(pos).unsqueeze(-1)).squeeze(-1)
    v_load = d.body_link_lin_vel_w.torch[:, ids] + torch.cross(d.body_link_ang_vel_w.torch[:, ids], r, dim=-1)
    v_rel = env.scene[box_name].data.root_link_lin_vel_w.torch - v_load.mean(1)
    return (v_rel ** 2).sum(-1)


# ---------------- 终止
def box_dropped(env, drop_height=0.15, max_horizontal=0.6, box_name="box", robot_name="robot"):
    """DEC-007 第 6 项：箱子中心低于两前臂负载点中点 drop_height 以上，或与骨盆水平距离超过 max_horizontal。"""
    bp = env.scene[box_name].data.root_link_pos_w.torch
    mid = _load_point_mid(env, robot_name)
    low = bp[:, 2] < mid[:, 2] - drop_height
    far = (bp[:, :2] - env.scene[robot_name].data.root_link_pos_w.torch[:, :2]).norm(dim=-1) > max_horizontal
    return low | far


def torso_ground_contact(env, threshold=1.0, sensor_name="torso_ground"):
    """与默认 base_contact 语义相同（躯干触地 = 摔倒），但只统计躯干与地面之间的力。
    v3 打开自碰撞并加入真实箱子后，默认 contact_forces 的净力会包含箱子 / 手臂压在躯干上的力，被误判为摔倒
    （冒烟测试第 3 步起大量触发）。"""
    f = env.scene.sensors[sensor_name].data.force_matrix_w_history.torch  # (N, H, B, F, 3)
    return (f.norm(dim=-1).amax(dim=(1, 2, 3)) > threshold)


# ---------------- 观测（critic 特权）
def box_pose_in_base(env, box_name="box", robot_name="robot"):
    """DEC-007 第 8 项：箱子相对骨盆的位置(3) + 相对姿态旋转矩阵前两列(6)。"""
    d, bd = env.scene[robot_name].data, env.scene[box_name].data
    Rb = quat_xyzw_to_matrix(d.root_link_quat_w.torch)
    Rx = quat_xyzw_to_matrix(bd.root_link_quat_w.torch)
    p = (Rb.transpose(-1, -2) @ (bd.root_link_pos_w.torch - d.root_link_pos_w.torch).unsqueeze(-1)).squeeze(-1)
    Rrel = Rb.transpose(-1, -2) @ Rx
    return torch.cat([p, Rrel[..., :, 0], Rrel[..., :, 1]], -1)


def payload_mass_per_arm(env):
    m = _upper(env).payload_mass.float()
    return torch.stack([0.5 * m, 0.5 * m], -1)


def upper_joint_targets(env):
    return _upper(env).qp.q_des.float()


# ---------------- 奖励
def upper_task_residual_l2(env):
    """两手手部位置行的 QP 残差平方和 [(m/s)²]：上肢任务没做到的部分，腿可以通过更平稳的步态减小它。"""
    r = _upper(env).qp.task_residual
    return (r[:, [0, 1, 2, 4, 5, 6]] ** 2).sum(-1).float()


# ---------------- 课程
def payload_curriculum(env, env_ids, enabled=False, initial_cap=1.0, step=0.5, max_cap=5.0, ratio_threshold=0.9,
                       ema=0.05):
    """负载上限课程（RL-LOCO-001 第 7 项，规则由用户批准：平均 episode 长度超过 90% 时提高一档）。

    机制（面试要能讲清）：
      1. 课程项在 _reset_idx 里、缓冲区清零【之前】被调用，所以 env.episode_length_buf[env_ids]
         就是刚刚结束的这些 episode 的长度。除以 env.max_episode_length 得到“活满比例”。
      2. 每次只有一部分环境在重置，单次样本噪声大，因此用指数滑动平均（EMA）平滑：
         ratio_ema ← (1 − ema) · ratio_ema + ema · 本批比例。
      3. ratio_ema 超过阈值 0.9 时提高上限，但每次只按“本批环境占总数的比例”提高，
         即所有环境各重置一次才累计一整档 step，避免一次重置就跳一大档。
      4. 上限只升不降，并夹在 [initial_cap, max_cap]；reset_payload 从 U(0, cap) 采样，
         所以轻负载始终会被采到，不会出现“只练重载、忘了轻载”。
    返回当前上限，会记录到训练日志 Curriculum/payload。

    enabled = False 时不做课程，上限保持 action term 的 payload_cap_init（EXP-011 / 015 的固定上限配置）。
    开关必须是【本函数参数】而不是 env cfg 的字段：Hydra 命令行覆盖发生在 __post_init__ 之后，
    在 __post_init__ 里按字段选择函数不会被覆盖触发（EXP-016 因此没有真正启用课程）。
    命令行启用：env.curriculum.payload.params.enabled=true
    """
    term = _upper(env)
    if not enabled:
        return term.payload_cap
    if not hasattr(term, "payload_ratio_ema"):
        term.payload_ratio_ema = 0.0
        term.payload_cap = float(initial_cap)
    if len(env_ids) == 0:
        return term.payload_cap
    ratio = (env.episode_length_buf[env_ids].float() / env.max_episode_length).mean().item()
    term.payload_ratio_ema = (1.0 - ema) * term.payload_ratio_ema + ema * ratio
    if term.payload_ratio_ema > ratio_threshold and term.payload_cap < max_cap:
        term.payload_cap = min(max_cap, term.payload_cap + step * len(env_ids) / env.num_envs)
    return term.payload_cap


def payload_cap_constant(env, env_ids):
    """占位/对照：不做课程，上限保持 payload_cap_init（EXP-011 使用）。"""
    return _upper(env).payload_cap
