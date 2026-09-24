"""EXP-011 策略的 MuJoCo 部署评估：RL 控腿 + 上肢 QP 托箱行走。

    policy 50 Hz（obs 60 维，Isaac 关节顺序）→ 10 个腿部关节 q_des
    上肢 QP 100 Hz（torch 版 UBQP-001，与 C++ 版一致，EXP-009）→ torso + 双臂 q_des / dq_des / τ_ff
    关节 PD 500 Hz（MuJoCo 每个仿真步），力矩限幅取 MJCF ctrlrange（部署端真值）
    箱子：MjSpec 在两个 elbow_link 上各挂 m/2（前臂轴上距肘 0.15 m）；控制器侧模型不含箱子，只按“已知质量”前馈

用法：/home/tt/miniconda3/envs/rl/bin/python scripts/sim2sim_carry.py --policy <policy.pt> --contract <carry_v2.json> --out run01
"""
import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from h1_locomanip.mujoco_kinematics import kinematics_from_data, upper_dof, upper_limits  # noqa: E402
from h1_locomanip.upper_body_qp import BatchedUpperBodyQp, UpperBodyQpCfg  # noqa: E402

SCENE = pathlib.Path(__file__).resolve().parents[2] / "src/humanoid_wbc/model/unitree_h1/scene.xml"
LOAD_DISTANCE = 0.15
GRAVITY_DIR_W = np.array([0.0, 0.0, -1.0])
LIMIT_MARGIN = 0.05
BOX_HALF = np.array([0.10, 0.22, 0.10])   # 箱子半尺寸：深 0.20 × 宽 0.44 × 高 0.20 m；中心在负载点（距肘 0.15），
                                          # 后沿距肘 +0.05 m（静止间隙扫描见 validation/2026-09-23/box_geometry_sweep.txt）
FOREARM_RADIUS = 0.025                    # 前臂碰撞胶囊半径（h1.xml elbow_link capsule size 0.025）


def build_model(payload, box_mode="attached", box_mu=0.75):
    """返回 (仿真模型, 控制器侧无箱模型)。不改 XML，全部用 MjSpec 运行时添加。
    attached（v2）：payload > 0 时在两前臂各挂 m/2 的刚体，箱子只是示意几何。
    free（v3，DEC-007）：箱子是带自由关节的刚体（质量 = payload，即总质量），用显式接触对与机器人全部碰撞体和地面接触，
    摩擦 box_mu（显式 pair 覆盖 MuJoCo 默认的 max 合成规则，与 Isaac 端“有效摩擦 = 箱子 μ”一致）。
    自由箱子放在 worldbody 最后，所以其 qpos / qvel 排在机器人之后，机器人索引不变。"""
    ctrl = mujoco.MjModel.from_xml_path(str(SCENE))
    if box_mode == "free":
        spec = mujoco.MjSpec.from_file(str(SCENE))
        for i, g in enumerate(spec.geoms):
            if not g.name:
                g.name = f"_g{i}"
        robot_geoms = [g.name for g in spec.geoms if g.contype or g.conaffinity]
        box = spec.worldbody.add_body(name="box", pos=[0.0, 0.0, 2.0])
        box.add_freejoint(name="box_free")
        a, b, c = 2 * BOX_HALF
        box.add_geom(name="payload_visual", type=mujoco.mjtGeom.mjGEOM_BOX, size=BOX_HALF.tolist(),
                     contype=0, conaffinity=0, mass=payload, rgba=[0.85, 0.6, 0.3, 1.0])
        for gname in robot_geoms:
            spec.add_pair(geomname1="payload_visual", geomname2=gname,
                          friction=[box_mu, box_mu, 0.005, 1e-4, 1e-4], condim=3)
        return spec.compile(), ctrl
    if payload <= 0.0:
        return ctrl, ctrl
    spec = mujoco.MjSpec.from_file(str(SCENE))
    axis = np.array([0.28, 0.0, -0.015])
    axis = axis / np.linalg.norm(axis)
    for side in ["left", "right"]:
        body = spec.body(f"{side}_elbow_link").add_body(
            name=f"{side}_payload", pos=(axis * LOAD_DISTANCE).tolist(), mass=0.5 * payload,
            inertia=[1e-4, 1e-4, 1e-4], explicitinertial=True)
    # 示意箱子：mocap 体（无碰撞、无质量），每个渲染帧按 box_pose() 摆放，与间隙指标用同一几何（水平放在两前臂上）
    vis = spec.worldbody.add_body(name="box_vis", mocap=True)
    vis.add_geom(name="payload_visual", type=mujoco.mjtGeom.mjGEOM_BOX, size=BOX_HALF.tolist(),
                 contype=0, conaffinity=0, mass=0.0, density=0.0, rgba=[0.85, 0.6, 0.3, 0.8])
    return spec.compile(), ctrl


def obb_obb_signed(c1, R1, h1, c2, R2, h2):
    """两个有向包围盒的分离轴检验：>0 为某一分离轴上的最大分离量（真实距离的下界），<0 为精确穿透深度。"""
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    axes += [np.cross(R1[:, i], R2[:, j]) for i in range(3) for j in range(3)]
    best_overlap, max_sep = np.inf, -np.inf
    for a in axes:
        n = np.linalg.norm(a)
        if n < 1e-9:
            continue
        a = a / n
        r1 = sum(h1[i] * abs(R1[:, i] @ a) for i in range(3))
        r2 = sum(h2[i] * abs(R2[:, i] @ a) for i in range(3))
        sep = abs((c2 - c1) @ a) - (r1 + r2)
        max_sep = max(max_sep, sep)
        best_overlap = min(best_overlap, -sep)
    return max_sep if max_sep > 0 else -best_overlap


def point_obb_signed(p, c, R, h):
    """点到有向盒的有符号距离（外正内负，精确）。"""
    q = np.abs(R.T @ (p - c)) - h
    return np.linalg.norm(np.maximum(q, 0.0)) + min(q.max(), 0.0)


class CarryDeployment:
    def __init__(self, policy_path, contract, payload, ankle_limit=None, carrier_lead=None, box_mu=None):
        self.c = contract
        self.box_free = contract.get("payload_mode", "attached") == "free_box"
        if self.box_free:  # payload 参数 = 加到空箱上的质量；总质量 = 空箱 + payload（与训练端 reset_box 一致）
            payload = contract["box_empty_mass"] + payload
        self.box_mu = box_mu if box_mu is not None else contract.get("box_friction_eval", 0.75)
        self.m, self.mc = build_model(payload, "free" if self.box_free else "attached", self.box_mu)
        # v4（DEC-008）：策略额外输出 2 维上身参考修正 Δ（内夹、上下），actor 观测末尾追加箱子相对骨盆位姿(9)
        self.ref_delta = bool(contract.get("ref_delta", False))
        if self.ref_delta:
            dc = contract["delta_cfg"]
            self.delta_scale, self.delta_in_max, self.delta_z_max = dc["scale"], dc["in_max"], dc["z_max"]
            self.delta_alpha = 1.0 - np.exp(-contract["qp_cfg"]["dt"] / dc["tau"])
        self.delta_override = None
        self.box_qadr = self.m.jnt_qposadr[self.m.joint("box_free").id] if self.box_free else None
        self.box_vadr = self.m.jnt_dofadr[self.m.joint("box_free").id] if self.box_free else None
        self.d, self.dc = mujoco.MjData(self.m), None
        self.dc = mujoco.MjData(self.mc) if self.mc is not self.m else self.d
        self.payload = payload
        self.policy = torch.jit.load(policy_path).eval()
        self.names = contract["joint_names_isaac"]
        self.q_default = np.array(contract["default_joint_pos"])
        self.leg_names = contract["leg_action_joint_names"]
        self.leg_in_isaac = [self.names.index(n) for n in self.leg_names]
        self.scale = contract["leg_action_scale"]
        # 四类索引：按名字查，取到立刻校验
        idx = lambda n: (mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n),
                         mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n))
        self.qadr, self.vadr, self.aid = [], [], []
        for n in self.names:
            jid, aid = idx(n)
            if jid < 0 or aid < 0 or self.m.actuator_trnid[aid, 0] != jid:
                raise RuntimeError(f"关节/执行器映射失败: {n}")
            self.qadr.append(self.m.jnt_qposadr[jid])
            self.vadr.append(self.m.jnt_dofadr[jid])
            self.aid.append(aid)
        self.qadr, self.vadr, self.aid = map(np.array, (self.qadr, self.vadr, self.aid))
        # 每关节有效 kp / kd：按执行器分组组装（显式执行器的增益不在 data 里，见 export_contract.py 注释）
        self.kp = np.zeros(len(self.names))
        self.kd = np.zeros(len(self.names))
        for g, jnames in contract["actuator_joint_names"].items():
            for j, n in enumerate(jnames):
                i = self.names.index(n)
                self.kp[i] = contract["group_stiffness"][g][j]
                self.kd[i] = contract["group_damping"][g][j]
        if not np.all(self.kp > 0):
            raise RuntimeError(f"有关节 kp 为 0: {[n for n, k in zip(self.names, self.kp) if k <= 0]}")
        self.tau_lo = self.m.actuator_ctrlrange[self.aid, 0].copy()
        self.tau_hi = self.m.actuator_ctrlrange[self.aid, 1].copy()
        if ankle_limit is not None:  # 单变量实验：把踝限幅改为 Isaac 训练端的值，检验摔倒是否由饱和造成
            for i, n in enumerate(self.names):
                if n.endswith("_ankle"):
                    self.tau_lo[i], self.tau_hi[i] = -ankle_limit, ankle_limit
                    # MuJoCo 会按模型 ctrlrange 再夹一次，必须同时放宽模型限幅，否则本实验无效果
                    self.m.actuator_ctrlrange[self.aid[i]] = [-ankle_limit, ankle_limit]
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # 步态量化（2026-09-24）：左右脚 = ankle_link；腿部关节 = 腿策略控制的 10 个关节
        self.foot_body = [self.m.body(f"{s}_ankle_link").id for s in ("left", "right")]
        self.leg_idx = np.array(self.leg_in_isaac)
        self.hang = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, "hang")
        # 上肢 QP（torch, batch = 1）：与训练端同配置
        self.upper_names = contract["upper_joint_names"]
        self.upper_in_isaac = [self.names.index(n) for n in self.upper_names]
        q_min, q_max = upper_limits(self.mc, LIMIT_MARGIN)
        cfg = UpperBodyQpCfg(**{k: v for k, v in contract["qp_cfg"].items()})
        self.qp = BatchedUpperBodyQp(1, self.mc.nv, upper_dof(self.mc), q_min, q_max, cfg, dtype=torch.float64)
        self.hand_offset = np.array(contract["hand_offset_b"])
        self.carrier_tau = 1.0 / (2 * np.pi * contract["carrier_cutoff_hz"])
        self.alpha = 1.0 - np.exp(-cfg.dt / self.carrier_tau)
        # 一阶低通对匀速运动有稳态滞后 v·τ。carrier_lead：
        #   none     = 不补偿（EXP-011–017 训练口径）
        #   command  = 指令速度前馈 τ·v_cmd（经同一低通；与训练端 upper_body_action.py 相同，EXP-018 起）
        #   measured = 实测基座速度补偿（仅复现失败实验：速度正反馈，4/9 摔倒）
        # 默认取契约里的训练端设置，保证部署与训练同一 reference
        self.carrier_lead = carrier_lead or contract.get("carrier_lead", "none")
        if self.carrier_lead not in ("none", "command", "measured"):
            raise ValueError(f"未知 carrier_lead: {self.carrier_lead}")
        self.policy_decim = int(round(contract["policy_dt"] / self.m.opt.timestep))
        self.upper_decim = int(round(cfg.dt / self.m.opt.timestep))
        self.q_nominal = torch.as_tensor(self.q_default[self.upper_in_isaac], dtype=torch.float64).unsqueeze(0)
        # 箱体间隙检查（只读指标，不参与物理）：示意箱子与机器人自身碰撞体的最小有符号距离。
        # 排除两侧 elbow_link（箱子本就搭在前臂上，那里的接触是预期支撑）与地面。
        self.box_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "payload_visual")
        excluded = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_elbow_link") for s in ["left", "right"]}
        self.check_geoms = [g for g in range(self.m.ngeom)
                            if self.m.geom_contype[g] and self.m.geom_bodyid[g] not in excluded
                            and self.m.geom_bodyid[g] != 0]

    # ---- 状态 / 观测
    def _sync_ctrl_model(self):
        if self.dc is not self.d:
            self.dc.qpos[:] = self.d.qpos[:self.mc.nq]
            self.dc.qvel[:] = self.d.qvel[:self.mc.nv]
            mujoco.mj_forward(self.mc, self.dc)

    def _obs(self, cmd):
        d = self.d
        R = d.xmat[self.pelvis].reshape(3, 3)
        omega_b = d.qvel[3:6]
        r_com = d.xipos[self.pelvis] - d.xpos[self.pelvis]
        v_com_w = d.qvel[0:3] + np.cross(R @ omega_b, r_com)   # Isaac base_lin_vel = 骨盆质心速度（EXP-004）
        q = d.qpos[self.qadr]
        dq = d.qvel[self.vadr]
        parts = [R.T @ v_com_w, omega_b, R.T @ GRAVITY_DIR_W, cmd, q - self.q_default, dq, self.last_action]
        if self.ref_delta:  # 与 mdp.box_pose_in_base 相同：骨盆系位置 + 相对旋转矩阵前两列
            bc, bR = self.box_pose()
            Rrel = R.T @ bR
            parts += [R.T @ (bc - d.qpos[0:3]), Rrel[:, 0], Rrel[:, 1]]
        return np.concatenate(parts).astype(np.float32)

    def _hand_targets(self):
        c, s = np.cos(self.carrier_yaw + self.lead_yaw), np.sin(self.carrier_yaw + self.lead_yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        off = self.hand_offset.copy()
        if self.ref_delta:
            off[0, 1] -= self.delta[0]
            off[1, 1] += self.delta[0]
            off[:, 2] += self.delta[1]
        return self.carrier_pos + self.lead_pos + off @ Rz.T

    def _upper_tick(self):
        self._sync_ctrl_model()
        d = self.dc
        yaw = np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]),
                         1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
        self.carrier_pos += self.alpha * (d.qpos[0:3] - self.carrier_pos)
        dyaw = yaw - self.carrier_yaw
        self.carrier_yaw += self.alpha * np.arctan2(np.sin(dyaw), np.cos(dyaw))
        if self.ref_delta:
            self.delta += self.delta_alpha * (self.delta_goal - self.delta)
        if self.carrier_lead == "command":
            c, s = np.cos(self.carrier_yaw), np.sin(self.carrier_yaw)
            vx, vy, wz = self.cmd
            goal = self.carrier_tau * np.array([c * vx - s * vy, s * vx + c * vy, 0.0])
            self.lead_pos += self.alpha * (goal - self.lead_pos)
            self.lead_yaw += self.alpha * (self.carrier_tau * wz - self.lead_yaw)
        elif self.carrier_lead == "measured":
            v_xy = np.array([d.qvel[0], d.qvel[1], 0.0])   # 基座原点线速度（world），只补偿水平方向
            self.lead_pos += self.alpha * (self.carrier_tau * v_xy - self.lead_pos)
        ref = self._hand_targets()
        ref_vel = (ref - self.ref_prev) / self.qp.cfg.dt
        self.ref_prev = ref
        kin = kinematics_from_data(self.mc, d, LOAD_DISTANCE)
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64).unsqueeze(0)
        q_upper = T(d.qpos[[self.mc.jnt_qposadr[self.mc.joint(n).id] for n in self.upper_names]])
        self.qp.update(kin, T(ref), T(ref_vel), T(d.qvel), q_upper, self.q_nominal)
        m_known = torch.tensor([self.payload], dtype=torch.float64)
        self.tau_ff_upper = self.qp.tau_feedforward(kin, m_known)[0].numpy()
        self.hand_err = ref - kin.hand_pos[0].numpy()
        # 绝对口径：双手相对【当前】骨盆（仅 yaw）的前向位置 − 标称 0.369 m；负值 = 手被拉回（滞后参考时约 −v·τ）
        rel = kin.hand_pos[0].numpy().mean(0) - d.qpos[0:3]
        self.hand_fwd_err = np.cos(yaw) * rel[0] + np.sin(yaw) * rel[1] - self.hand_offset[:, 0].mean()
        self.forearm_az = kin.forearm_axis[0, :, 2].numpy()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        if self.hang >= 0:
            self.d.eq_active[self.hang] = 0
        self.d.qpos[0:3] = [0.0, 0.0, 1.05]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qpos[self.qadr] = self.q_default
        mujoco.mj_forward(self.m, self.d)
        if self.box_free:  # 与训练端 reset_box 相同：水平放在两前臂标称位置上方 2 cm，静止
            c, _ = self._nominal_box_pose()
            self.d.qpos[self.box_qadr:self.box_qadr + 7] = [*(c + [0.0, 0.0, 0.02]), 1.0, 0.0, 0.0, 0.0]
            self.d.qvel[self.box_vadr:self.box_vadr + 6] = 0.0
            mujoco.mj_forward(self.m, self.d)
        self._sync_ctrl_model()
        self.box_dropped_at = None
        self.box_rel_ref = None
        self.last_action = np.zeros(len(self.leg_names) + (2 if self.ref_delta else 0))
        self.delta = np.zeros(2)
        self.delta_goal = np.zeros(2)
        self.q_des = self.q_default.copy()
        self.carrier_pos = self.d.qpos[0:3].copy()
        self.lead_pos = np.zeros(3)
        self.lead_yaw = 0.0
        self.cmd = np.zeros(3)
        self.carrier_yaw = 0.0
        self.ref_prev = self._hand_targets()
        self.qp.reset(slice(None), self.q_nominal)
        self.tau_ff_upper = np.zeros(len(self.upper_names))
        self.hand_err = np.zeros((2, 3))
        self.hand_fwd_err = 0.0
        self.forearm_az = np.zeros(2)

    def video_writer(self, path, width=960, height=540, fps=50):
        import subprocess
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, width)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, height)
        renderer = mujoco.Renderer(self.m, height, width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid, cam.distance, cam.elevation, cam.azimuth = self.pelvis, 3.2, -12.0, 130.0
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                 "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
        return renderer, cam, proc

    def run(self, cmd, duration, video=None, heading_hold=False):
        self.reset()
        cmd = np.asarray(cmd, float)
        self.cmd = cmd
        n = int(round(duration / self.m.opt.timestep))
        log = {k: [] for k in ["t", "vx_b", "vy_b", "wz", "pelvis_z", "hand_err_l", "hand_err_r", "az_l", "az_r",
                               "box_clearance", "hand_fwd_err", "box_rel_x", "box_rel_y", "box_rel_z",
                               "delta_in", "delta_z"]}
        self.box_worst = (np.inf, None)
        sat = np.zeros(len(self.names), dtype=int)
        tau_max = np.zeros(len(self.names))
        fell_at, steps = None, 0
        gait = {k: [] for k in ["t", "contact_l", "contact_r", "foot_z_l", "foot_z_r", "power_leg", "tau_leg_sq",
                                "vx_w", "vy_b", "pelvis_z", "roll", "pitch"]}
        renderer = cam = proc = None
        if video is not None:
            renderer, cam, proc = self.video_writer(video)
        for step in range(n):
            if step % self.upper_decim == 0:
                self._upper_tick()
            if step % self.policy_decim == 0:
                if heading_hold:
                    # 与 Isaac UniformVelocityCommand(heading_command=True) 相同：wz = clip(0.5·wrap(目标航向 − 航向), ±1)，目标航向 0
                    q = self.d.qpos[3:7]
                    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                    cmd[2] = np.clip(0.5 * np.arctan2(np.sin(-yaw), np.cos(-yaw)), -1.0, 1.0)
                obs = self._obs(cmd)
                with torch.no_grad():
                    a = self.policy(torch.as_tensor(obs).unsqueeze(0))[0].numpy()
                self.last_action = a.astype(float)
                self.q_des = self.q_default.copy()
                self.q_des[self.leg_in_isaac] += self.scale * self.last_action[:len(self.leg_names)]
                if self.ref_delta and self.delta_override is not None:  # 评估用：忽略策略的 Δ 输出（观测里的 last_action 仍为策略原值）
                    self.delta_goal = np.array(self.delta_override)
                elif self.ref_delta:
                    self.delta_goal = np.array([
                        np.clip(self.delta_scale * a[len(self.leg_names)], 0.0, self.delta_in_max),
                        np.clip(self.delta_scale * a[len(self.leg_names) + 1], -self.delta_z_max, self.delta_z_max)])
                log["delta_in"].append(self.delta[0]); log["delta_z"].append(self.delta[1])
                log["t"].append(self.d.time)
                log["vx_b"].append(obs[0]); log["vy_b"].append(obs[1]); log["wz"].append(obs[5])
                log["pelvis_z"].append(self.d.qpos[2])
                log["hand_err_l"].append(np.linalg.norm(self.hand_err[0]))
                log["hand_err_r"].append(np.linalg.norm(self.hand_err[1]))
                log["az_l"].append(self.forearm_az[0]); log["az_r"].append(self.forearm_az[1])
                log["box_clearance"].append(self._box_clearance())
                log["hand_fwd_err"].append(self.hand_fwd_err)
                if self.box_free:
                    dropped, rel = self._box_check()
                    log["box_rel_x"].append(rel[0]); log["box_rel_y"].append(rel[1]); log["box_rel_z"].append(rel[2])
                    if dropped:
                        self.box_dropped_at = self.d.time
                        break
            q = self.d.qpos[self.qadr]
            dq = self.d.qvel[self.vadr]
            q_des = self.q_des.copy()
            dq_des = np.zeros_like(dq)
            q_des[self.upper_in_isaac] = self.qp.q_des[0].numpy()
            dq_des[self.upper_in_isaac] = self.qp.dq_des[0].numpy()
            tau = self.kp * (q_des - q) + self.kd * (dq_des - dq)
            tau[self.upper_in_isaac] += self.tau_ff_upper
            tau_max = np.maximum(tau_max, np.abs(tau))
            sat += (tau < self.tau_lo) | (tau > self.tau_hi)
            self.d.ctrl[self.aid] = np.clip(tau, self.tau_lo, self.tau_hi)
            mujoco.mj_step(self.m, self.d)
            steps = step + 1
            self._log_gait(gait, tau)
            if renderer is not None and step % self.policy_decim == 0 and self.box_free:
                renderer.update_scene(self.d, camera=cam)
                proc.stdin.write(renderer.render().tobytes())
            elif renderer is not None and step % self.policy_decim == 0:
                c, R = self.box_pose()
                mid = self.m.body("box_vis").mocapid[0]
                self.d.mocap_pos[mid] = c
                mujoco.mju_mat2Quat(self.d.mocap_quat[mid], R.flatten())
                mujoco.mj_kinematics(self.m, self.d)
                renderer.update_scene(self.d, camera=cam)
                proc.stdin.write(renderer.render().tobytes())
            if self.d.qpos[2] < 0.5 or self._torso_contact():
                fell_at = self.d.time
                break
        if proc is not None:
            proc.stdin.close()
            proc.wait()
            renderer.close()
        self.gait = {k: np.array(v) for k, v in gait.items()}
        return ({k: np.array(v) for k, v in log.items()}, fell_at, sat / max(steps, 1), tau_max,
                int(self.qp.unconverged))

    def _log_gait(self, gait, tau):
        """每个物理步（500 Hz）记录：脚-地接触、脚高、腿部机械功率与力矩平方、骨盆状态。"""
        d = self.d
        cl = cr = False
        for k in range(d.ncon):
            con = d.contact[k]
            if self.floor not in (con.geom1, con.geom2):
                continue
            b = self.m.geom_bodyid[con.geom2 if con.geom1 == self.floor else con.geom1]
            cl |= b == self.foot_body[0]
            cr |= b == self.foot_body[1]
        tau_leg = np.clip(tau, self.tau_lo, self.tau_hi)[self.leg_idx]
        dq_leg = d.qvel[self.vadr][self.leg_idx]
        q = d.qpos[3:7]
        R = d.xmat[self.pelvis].reshape(3, 3)
        gait["t"].append(d.time)
        gait["contact_l"].append(cl); gait["contact_r"].append(cr)
        gait["foot_z_l"].append(d.xpos[self.foot_body[0]][2]); gait["foot_z_r"].append(d.xpos[self.foot_body[1]][2])
        gait["power_leg"].append(float(np.abs(tau_leg * dq_leg).sum()))
        gait["tau_leg_sq"].append(float((tau_leg ** 2).mean()))
        gait["vx_w"].append(float((R.T @ d.qvel[0:3])[0])); gait["vy_b"].append(float((R.T @ d.qvel[0:3])[1]))
        gait["pelvis_z"].append(d.qpos[2])
        gait["roll"].append(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        gait["pitch"].append(np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1)))

    def box_pose(self):
        """free 模式：自由箱子的真实位姿；attached 模式：见 _nominal_box_pose。"""
        if self.box_free:
            b = self.m.body("box").id
            return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()
        return self._nominal_box_pose()

    def _load_mid(self):
        pts = []
        for side in ["left", "right"]:
            b = self.m.body(f"{side}_elbow_link").id
            R = self.d.xmat[b].reshape(3, 3)
            pts.append(self.d.xpos[b] + R @ (np.array([0.28, 0.0, -0.015]) / np.linalg.norm([0.28, 0.0, -0.015]))
                       * LOAD_DISTANCE)
        return 0.5 * (pts[0] + pts[1])

    def _box_check(self):
        """DEC-007 第 6 / 10 项：与训练端相同的掉箱判据；并返回箱子相对负载点中点的位置（骨盆 yaw 系）用于滑移统计。"""
        c, _ = self.box_pose()
        mid = self._load_mid()
        q = self.d.qpos[3:7]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        dropped = c[2] < mid[2] - 0.15 or np.linalg.norm(c[:2] - self.d.qpos[0:2]) > 0.6
        return dropped, Rz.T @ (c - mid)

    def _nominal_box_pose(self):
        """物理上的箱子位姿：放在两前臂上，受重力保持水平。
        中心 = 两前臂负载点中点 + (前臂半径 + 箱子半高) · e_z；x 轴 = 两前臂方向平均后投影到水平面；z 轴 = 世界竖直。
        （旧做法把箱子刚性挂在左前臂坐标系上，会随前臂绕自身轴的滚转一起转动——那个自由度 QP 并不约束。）"""
        d, m = self.d, self.m
        pts, axes = [], []
        for side in ["left", "right"]:
            b = m.body(f"{side}_elbow_link").id
            R = d.xmat[b].reshape(3, 3)
            a = R @ (np.array([0.28, 0.0, -0.015]) / np.linalg.norm([0.28, 0.0, -0.015]))
            pts.append(d.xpos[b] + a * LOAD_DISTANCE)
            axes.append(a)
        x = axes[0] + axes[1]
        x[2] = 0.0
        x /= np.linalg.norm(x)
        z = np.array([0.0, 0.0, 1.0])
        y = np.cross(z, x)
        center = 0.5 * (pts[0] + pts[1]) + (FOREARM_RADIUS + BOX_HALF[2]) * z
        return center, np.column_stack([x, y, z])

    def _box_clearance(self):
        """箱体到机器人自身碰撞体（不含前臂）的最小有符号距离 [m]，负值 = 穿透。无负载时返回 nan。

        不用 mj_geomDistance：在本模型上它对“箱体-躯干盒”无论是否分离都返回 0.0（合成场景下正确，原因未查明）。
        改为独立几何计算：box-box 用分离轴定理（穿透深度精确）；胶囊 / 圆柱沿轴线 41 点采样，点到箱体
        有符号距离减半径；球为球心到箱体距离减半径。网格等其他类型跳过（本模型需检查的几何体中没有）。
        """
        if self.box_geom < 0:
            return float("nan")
        d, m = self.d, self.m
        bc, bR = self.box_pose()
        bh = BOX_HALF
        best = np.inf
        for g in self.check_geoms:
            gtype, size = m.geom_type[g], m.geom_size[g]
            gc, gR = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                dist = obb_obb_signed(bc, bR, bh, gc, gR, size)
            elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                pts = gc + np.outer(np.linspace(-size[1], size[1], 41), gR[:, 2])
                dist = min(point_obb_signed(p, bc, bR, bh) for p in pts) - size[0]
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                dist = point_obb_signed(gc, bc, bR, bh) - size[0]
            else:
                continue
            if dist < best:
                best = dist
                if dist < self.box_worst[0]:
                    self.box_worst = (dist, m.geom(g).name or f"{m.body(m.geom_bodyid[g]).name}#{g}")
        return best

    def _torso_contact(self):
        for k in range(self.d.ncon):
            con = self.d.contact[k]
            if self.floor in (con.geom1, con.geom2) and self.torso in (self.m.geom_bodyid[con.geom1],
                                                                       self.m.geom_bodyid[con.geom2]):
                return True
        return False


def gait_metrics(g, warmup, total_mass):
    """由 500 Hz 步态记录计算量化指标（t ≥ warmup）。着地 = 接触状态 0→1 的上升沿；
    一个步态周期 = 同一只脚相邻两次着地；抬脚高度 = 每次摆动期脚（ankle_link 原点）最高点 − 该脚支撑期高度中位。"""
    w = g["t"] >= warmup
    if w.sum() < 100:
        return None
    t = g["t"][w]
    dur = t[-1] - t[0]
    out = {}
    stance, cycle, clear, n_td = [], [], [], []
    for side in ("l", "r"):
        c = g[f"contact_{side}"][w].astype(int)
        z = g[f"foot_z_{side}"][w]
        td = np.flatnonzero(np.diff(c) == 1) + 1          # 着地
        lo = np.flatnonzero(np.diff(c) == -1) + 1         # 离地
        n_td.append(len(td))
        cyc = np.diff(t[td]) if len(td) > 1 else np.array([np.nan])
        cycle.append(np.nanmedian(cyc))
        stance.append(c.mean())
        z_st = np.median(z[c == 1]) if (c == 1).any() else np.nan
        peaks = []
        for a in lo:                                     # 每次离地 → 下一次着地之间为一次摆动
            nxt = td[td > a]
            if len(nxt):
                peaks.append(z[a:nxt[0]].max() - z_st)
        clear.append(float(np.median(peaks)) if peaks else float("nan"))
    cl, cr = g["contact_l"][w], g["contact_r"][w]
    si = lambda a, b: float(2 * abs(a - b) / (a + b)) if (a + b) > 0 else float("nan")
    p_mean = g["power_leg"][w].mean()
    v_mean = abs(g["vx_w"][w].mean())
    out.update({
        "step_freq_hz": float((n_td[0] + n_td[1]) / dur),
        "cycle_s_lr": [float(x) for x in cycle],
        "stance_frac_lr": [float(x) for x in stance],
        "stance_symmetry_index": si(stance[0], stance[1]),
        "cycle_symmetry_index": si(cycle[0], cycle[1]),
        "double_support_frac": float((cl & cr).mean()),
        "flight_frac": float((~cl & ~cr).mean()),
        "clearance_mm_lr": [float(1e3 * x) for x in clear],
        "leg_tau_rms_nm": float(np.sqrt(g["tau_leg_sq"][w].mean())),
        "leg_power_w": float(p_mean),
        "cost_of_transport": float(p_mean / (total_mass * 9.81 * v_mean)) if v_mean > 0.05 else None,
        "pelvis_z_mean_m": float(g["pelvis_z"][w].mean()),
        "roll_rms_deg": float(np.degrees(np.sqrt((g["roll"][w] ** 2).mean()))),
        "pitch_mean_deg": float(np.degrees(g["pitch"][w].mean())),
        "pitch_rms_deg": float(np.degrees(np.sqrt(((g["pitch"][w] - g["pitch"][w].mean()) ** 2).mean()))),
        "lateral_vel_rms": float(np.sqrt((g["vy_b"][w] ** 2).mean())),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--ankle_limit", type=float, default=None, help="覆盖踝关节力矩限幅 [N·m]（单变量实验）")
    ap.add_argument("--payloads", type=float, nargs="+", default=[0.0, 2.5, 5.0])
    ap.add_argument("--cases", nargs="+", default=["fwd_0.5", "fwd_1.0", "turn_0.5_0.5"])
    ap.add_argument("--open_loop_heading", action="store_true",
                    help="直行工况给固定 wz = 0、不做航向闭环（2026-09-23 之前的旧协议，仅复现用）。默认：与训练端相同的"
                         "航向闭环 wz = clip(0.5·wrap(0 − yaw), ±1)（训练 rel_heading_envs = 1.0；用户 2026-09-23 批准）")
    ap.add_argument("--video", action="store_true", help="每个工况录制 mp4 到输出目录")
    ap.add_argument("--delta_override", type=float, nargs=2, default=None,
                    help="v4 评估用：把上身参考修正固定为 (Δin, Δz) [m]，忽略策略输出")
    ap.add_argument("--box_mu", type=float, default=None, help="free 模式箱子摩擦系数（默认取契约 box_friction_eval）")
    ap.add_argument("--carrier_lead", choices=["none", "command", "measured"], default=None,
                    help="覆盖契约中的载体系超前方式（默认与训练端一致）")
    args = ap.parse_args()
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)
    contract = json.load(open(args.contract))
    print(f"mujoco {mujoco.__version__} torch {torch.__version__}; policy_decim/upper_decim = "
          f"{int(round(contract['policy_dt'] / 0.002))}/{int(round(contract['qp_cfg']['dt'] / 0.002))}")
    results = []
    all_cases = {"fwd_0.5": [0.5, 0.0, 0.0], "fwd_1.0": [1.0, 0.0, 0.0], "turn_0.5_0.5": [0.5, 0.0, 0.5]}
    for payload in args.payloads:
        dep = CarryDeployment(args.policy, contract, payload, args.ankle_limit, args.carrier_lead, args.box_mu)
        dep.delta_override = args.delta_override
        for name in args.cases:
            cmd = all_cases[name]
            video = (out / f"p{payload:g}_{name}.mp4") if args.video else None
            cmd = list(cmd)
            log, fell, sat, tau_max, unconv = dep.run(cmd, args.duration, video,
                                                      heading_hold=not args.open_loop_heading and name.startswith("fwd"))
            w = log["t"] >= args.warmup
            rms = lambda x: float(np.sqrt((x[w] ** 2).mean())) if w.any() else float("nan")
            r = {
                "payload_kg": payload, "case": name, "cmd": cmd, "ankle_limit": args.ankle_limit,
                "survived_s": float(log["t"][-1]) if len(log["t"]) else 0.0, "fell_at_s": fell,
                "vx_b_mean": float(log["vx_b"][w].mean()) if w.any() else None,
                "wz_mean": float(log["wz"][w].mean()) if w.any() else None,
                "vx_rmse": rms(log["vx_b"] - cmd[0]), "wz_rmse": rms(log["wz"] - cmd[2]),
                "hand_err_rms_mm": [1e3 * rms(log["hand_err_l"]), 1e3 * rms(log["hand_err_r"])],
                "hand_err_max_mm": [float(1e3 * log["hand_err_l"][w].max()), float(1e3 * log["hand_err_r"][w].max())]
                if w.any() else None,
                "az_abs_max": [float(np.abs(log["az_l"][w]).max()), float(np.abs(log["az_r"][w]).max())]
                if w.any() else None,
                "qp_unconverged": unconv,
                "hand_fwd_err_mean_mm": float(1e3 * log["hand_fwd_err"][w].mean()) if w.any() else None,
                "box_clearance_min_mm": (float(1e3 * np.nanmin(log["box_clearance"][w]))
                                         if w.any() and np.isfinite(log["box_clearance"][w]).any() else None),
                "box_penetration_frac": (float((log["box_clearance"][w] < 0).mean())
                                         if w.any() and np.isfinite(log["box_clearance"][w]).any() else None),
                "box_worst_geom": dep.box_worst[1],
                "box_mode": "free" if dep.box_free else "attached",
                "delta_in_mean_mm": float(1e3 * log["delta_in"][w].mean()) if dep.ref_delta and w.any() else None,
                "delta_z_mean_mm": float(1e3 * log["delta_z"][w].mean()) if dep.ref_delta and w.any() else None,
                "box_mu": dep.box_mu if dep.box_free else None,
                "box_dropped_at_s": dep.box_dropped_at,
                # 滑移：箱子相对两前臂负载点中点（骨盆 yaw 系）的位置，相对 t = 0.5 s（落定后）的最大变化
                "box_slip_max_mm": (float(1e3 * np.max(np.linalg.norm(
                    np.stack([log["box_rel_x"], log["box_rel_y"], log["box_rel_z"]], -1)[log["t"] >= 0.5]
                    - np.array([log["box_rel_x"], log["box_rel_y"], log["box_rel_z"]])[:, np.argmax(log["t"] >= 0.5)],
                    axis=-1))) if dep.box_free and (log["t"] >= 0.5).any() else None),
                "box_rel_final_mm": ([float(1e3 * log[k][-1]) for k in ("box_rel_x", "box_rel_y", "box_rel_z")]
                                     if dep.box_free and len(log["box_rel_x"]) else None),
                "gait": gait_metrics(dep.gait, args.warmup, float(dep.m.body_subtreemass[0])),
                "saturated": {n: round(float(s), 4) for n, s in zip(dep.names, sat) if s > 0},
                "tau_max": {n: round(float(t), 1) for n, t in zip(dep.names, tau_max)},
            }
            results.append(r)
            np.savez(out / f"p{payload:g}_{name}.npz", **log)
            np.savez(out / f"p{payload:g}_{name}_gait.npz", **dep.gait)
            if r["vx_b_mean"] is None:
                print(f"[{payload:g} kg {name:12s}] 存活 {r['survived_s']:5.2f}s（未到 warmup {args.warmup}s，无统计）"
                      f"  fell_at {r['fell_at_s']}  掉箱 {r['box_dropped_at_s']}")
            else:
                print(f"[{payload:g} kg {name:12s}] 存活 {r['survived_s']:5.2f}s  vx {r['vx_b_mean']:.3f} "
                      f"(RMSE {r['vx_rmse']:.3f})  wz {r['wz_mean']:+.3f} (RMSE {r['wz_rmse']:.3f})  "
                      f"手部误差 RMS {r['hand_err_rms_mm'][0]:.1f}/{r['hand_err_rms_mm'][1]:.1f} mm  "
                      f"手前向偏差 {r['hand_fwd_err_mean_mm']:+.0f} mm  箱间隙 min {r['box_clearance_min_mm']} mm  "
                      f"max|a_z| {max(r['az_abs_max']):.3f}  掉箱 {r['box_dropped_at_s']}  滑移 max "
                      f"{r['box_slip_max_mm']}  饱和 {r['saturated']}")
    json.dump(results, open(out / "summary.json", "w"), indent=1)
    print("[out]", out / "summary.json")


if __name__ == "__main__":
    main()
