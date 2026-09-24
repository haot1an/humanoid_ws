"""从 MuJoCo 状态构造 UpperBodyKinematicsBatch，与 C++ MujocoUpperBodyKinematics 一致：
手部点 = elbow_link 系 (0.28, 0, −0.015)（h1.xml:178 碰撞球心），a = 该点单位化，负载点 = a · load_distance。
广义速度约定为 MuJoCo qvel：[vel_world(3); omega_body(3); dq_joint]。
"""
import mujoco
import numpy as np
import torch

from .upper_body_qp import UpperBodyKinematicsBatch

HAND_LOCAL = np.array([0.28, 0.0, -0.015])
AXIS_LOCAL = HAND_LOCAL / np.linalg.norm(HAND_LOCAL)
UPPER_JOINTS = ["torso",
                "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
                "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]


def upper_dof(m):
    return [int(m.jnt_dofadr[m.joint(n).id]) for n in UPPER_JOINTS]


def upper_limits(m, margin=0.05):
    r = np.array([m.jnt_range[m.joint(n).id] for n in UPPER_JOINTS])
    return r[:, 0] + margin, r[:, 1] - margin


_gravity_data = {}


def gravity(m, d):
    """零速度偏置力 = g(q)，与 C++ MujocoUpperBodyKinematics / Isaac gravity_compensation_forces 一致。"""
    gd = _gravity_data.setdefault(id(m), mujoco.MjData(m))
    gd.qpos[:] = d.qpos
    gd.qvel[:] = 0.0
    mujoco.mj_forward(m, gd)
    return gd.qfrc_bias.copy()


def kinematics_from_data(m, d, load_distance=0.15):
    """d 需已 mj_forward。返回 batch 大小 1 的 UpperBodyKinematicsBatch（float64）；bias 为纯重力 g(q)。"""
    hand, axis, Jh, Jr, Jl = [], [], [], [], []
    for side in ["left", "right"]:
        b = m.body(f"{side}_elbow_link").id
        R = d.xmat[b].reshape(3, 3)
        p = d.xpos[b] + R @ HAND_LOCAL
        jacp, jacr, jacl = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, jacr, p, b)
        mujoco.mj_jac(m, d, jacl, None, d.xpos[b] + R @ (AXIS_LOCAL * load_distance), b)
        hand.append(p); axis.append(R @ AXIS_LOCAL); Jh.append(jacp); Jr.append(jacr); Jl.append(jacl)
    t = lambda x: torch.as_tensor(np.array(x), dtype=torch.float64).unsqueeze(0)
    return UpperBodyKinematicsBatch(hand_pos=t(hand), forearm_axis=t(axis), J_hand=t(Jh), J_rot=t(Jr),
                                    J_load=t(Jl), bias=t(gravity(m, d)))


def stack(kins):
    """把多个 batch=1 的运动学拼成一个 batch。"""
    return UpperBodyKinematicsBatch(*[torch.cat([getattr(k, f) for k in kins]) for f in
                                      ["hand_pos", "forearm_axis", "J_hand", "J_rot", "J_load", "bias"]])
