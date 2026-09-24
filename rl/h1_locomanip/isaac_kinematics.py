"""Isaac Lab → UpperBodyKinematicsBatch 适配（不 import isaaclab，只处理张量）。

为避开浮动基列约定差异（Isaac：基座 [lin, ang] 均为 world 系；MuJoCo：线速度 world、角速度 body 系），
只取 Isaac Jacobian 的**关节列**，基座列按 MuJoCo 约定自行构造，输出精简广义速度约定：
    v' = [v_base_origin_world(3); omega_base_body(3); dq_upper(9)]，nv' = 15，upper_dof = 6..14
刚体点 p 的基座列（MuJoCo 约定）：线速度 [I | −[p − p_B]× R_B]，角速度 [0 | R_B]。
腿关节不在手的运动链上，省略它们的列不影响手部任务。
"""
import torch

from .upper_body_qp import UpperBodyKinematicsBatch

HAND_LOCAL = torch.tensor([0.28, 0.0, -0.015], dtype=torch.float64)
UPPER_JOINTS = ["torso",
                "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
                "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]
NV_REDUCED = 15
UPPER_DOF_REDUCED = list(range(6, 15))


def quat_xyzw_to_matrix(q):
    """Isaac Lab 6.x 四元数为 xyzw（init_state rot (0,0,0,1) = 单位）。q: (..., 4)。"""
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def skew(r):
    z = torch.zeros_like(r[..., 0])
    return torch.stack([z, -r[..., 2], r[..., 1], r[..., 2], z, -r[..., 0], -r[..., 1], r[..., 0], z],
                       -1).reshape(*r.shape[:-1], 3, 3)


class IsaacUpperBodyAdapter:
    def __init__(self, joint_names, body_names, num_base_dofs=6, load_distance=0.15, dtype=torch.float64):
        self.dtype = dtype
        self.upper_cols = [num_base_dofs + joint_names.index(n) for n in UPPER_JOINTS]  # Jacobian 关节列
        self.upper_joint_idx = [joint_names.index(n) for n in UPPER_JOINTS]
        self.elbow_body = [body_names.index(f"{s}_elbow_link") for s in ["left", "right"]]
        self.hand_local = HAND_LOCAL.to(dtype)
        self.axis_local = self.hand_local / self.hand_local.norm()
        self.load_local = self.axis_local * load_distance

    def reduced_velocity(self, root_lin_vel_w, root_ang_vel_b, joint_vel):
        """root_lin_vel_w 必须是基座**坐标原点**（root_link）的速度。"""
        return torch.cat([root_lin_vel_w, root_ang_vel_b, joint_vel[:, self.upper_joint_idx]], -1).to(self.dtype)

    def _point_jac(self, p, J_link, p_link, p_B, R_B):
        """点 p 的线速度 Jacobian（15 列）；J_link: Isaac 该 body 的 (B, 6, ncol)。"""
        B = p.shape[0]
        J = torch.zeros((B, 3, NV_REDUCED), dtype=self.dtype, device=p.device)
        J[:, :, 0:3] = torch.eye(3, dtype=self.dtype, device=p.device)
        J[:, :, 3:6] = -skew(p - p_B) @ R_B
        J_lin, J_ang = J_link[:, 0:3, self.upper_cols], J_link[:, 3:6, self.upper_cols]
        J[:, :, 6:] = J_lin - skew(p - p_link) @ J_ang  # 连杆原点的列平移到点 p
        return J

    def kinematics(self, body_pos_w, body_quat_w, body_link_jacobian_w, gravity_forces, root_pos_w, root_quat_w):
        """全部为 Isaac 张量（任意 dtype），body 维按 body_names，Jacobian 列按 Isaac 约定。"""
        f = lambda t: t.to(self.dtype)
        R_B, p_B = quat_xyzw_to_matrix(f(root_quat_w)), f(root_pos_w)
        hand, axis, Jh, Jr, Jl = [], [], [], [], []
        for b in self.elbow_body:
            p_link, R = f(body_pos_w[:, b]), quat_xyzw_to_matrix(f(body_quat_w[:, b]))
            J_link = f(body_link_jacobian_w[:, b])
            p_hand = p_link + (R @ self.hand_local.to(R.device))
            p_load = p_link + (R @ self.load_local.to(R.device))
            hand.append(p_hand)
            axis.append(R @ self.axis_local.to(R.device))
            Jh.append(self._point_jac(p_hand, J_link, p_link, p_B, R_B))
            Jl.append(self._point_jac(p_load, J_link, p_link, p_B, R_B))
            Jrot = torch.zeros_like(Jh[-1])
            Jrot[:, :, 3:6] = R_B
            Jrot[:, :, 6:] = J_link[:, 3:6, self.upper_cols]
            Jr.append(Jrot)
        bias = torch.zeros((p_B.shape[0], NV_REDUCED), dtype=self.dtype, device=p_B.device)
        bias[:, 6:] = f(gravity_forces[:, self.upper_cols])  # 只有 g(q)，Isaac 不提供 Coriolis
        st = lambda xs: torch.stack(xs, 1)
        return UpperBodyKinematicsBatch(hand_pos=st(hand), forearm_axis=st(axis), J_hand=st(Jh), J_rot=st(Jr),
                                        J_load=st(Jl), bias=bias)
