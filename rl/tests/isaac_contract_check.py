"""Isaac Lab ↔ MuJoCo 上肢运动学契约核对（在服务器 Isaac 环境中运行）。

对 N 个环境设置随机状态（离地 2 m，避免接触），同一状态分别由
  (a) IsaacUpperBodyAdapter（Isaac Jacobian 关节列 + 自构 MuJoCo 约定基座列）
  (b) mujoco_kinematics（MJCF，Python MuJoCo）
构造上肢运动学，逐项比较；并用 Isaac 自己的 body 速度检验 (a) 的 J·v 是否自洽。
用法（服务器）：cd /data/h1_locomanip/rl && python tests/isaac_contract_check.py --mjcf ../model/unitree_h1/scene.xml
"""
import argparse
import pathlib
import sys

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--mjcf", required=True)
parser.add_argument("--num_envs", type=int, default=8)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from h1_locomanip.isaac_kinematics import (UPPER_JOINTS, IsaacUpperBodyAdapter,  # noqa: E402
                                           quat_xyzw_to_matrix)
from h1_locomanip.mujoco_kinematics import kinematics_from_data, stack, upper_dof  # noqa: E402

TASK = "Isaac-Velocity-Flat-H1-v0"


def P(x):
    return x.torch if hasattr(x, "torch") else x


def main():
    env_cfg, _ = resolve_task_config(TASK, "")
    with launch_simulation(env_cfg, args_cli):
        N = args_cli.num_envs
        env_cfg.scene.num_envs = N
        env_cfg.events.add_base_mass = None
        env_cfg.events.push_robot = None
        env = gym.make(TASK, cfg=env_cfg)
        env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        dev = u.device
        names, bnames = list(robot.joint_names), list(robot.body_names)
        d = robot.data

        g = torch.Generator().manual_seed(0)
        rnd = lambda *s: (torch.rand(*s, generator=g, dtype=torch.float32) * 2 - 1).to(dev)
        lim = P(d.soft_joint_pos_limits)
        jp = torch.clamp(P(d.default_joint_pos) + 0.3 * rnd(N, len(names)), lim[..., 0], lim[..., 1])
        jv = rnd(N, len(names))
        axis_angle = 0.3 * rnd(N, 3)
        ang = axis_angle.norm(dim=-1, keepdim=True)
        quat_xyzw = torch.cat([axis_angle / ang * torch.sin(ang / 2), torch.cos(ang / 2)], -1)
        origins = u.scene.env_origins
        pose = torch.cat([origins + torch.tensor([0.0, 0.0, 2.0], device=dev), quat_xyzw], -1)
        vel = torch.cat([0.5 * rnd(N, 3), rnd(N, 3)], -1)
        robot.write_root_link_pose_to_sim_index(root_pose=pose)
        robot.write_root_link_velocity_to_sim_index(root_velocity=vel)
        robot.write_joint_state_to_sim_index(position=jp, velocity=jv)
        u.sim.forward()
        robot.update(0.0)

        read_jp, read_jv = P(d.joint_pos), P(d.joint_vel)
        print(f"读回检查：|joint_pos − 写入| max {(read_jp - jp).abs().max():.2e}, |joint_vel − 写入| max {(read_jv - jv).abs().max():.2e}")
        root_pos, root_quat = P(d.root_link_pos_w), P(d.root_link_quat_w)
        root_lin_w, root_ang_b = P(d.root_link_lin_vel_w), P(d.root_link_ang_vel_b)
        J = P(d.body_link_jacobian_w)
        print(f"body_link_jacobian_w shape {tuple(J.shape)}, dtype {J.dtype}; gravity_compensation_forces shape {tuple(P(d.gravity_compensation_forces).shape)}")

        ad = IsaacUpperBodyAdapter(names, bnames)
        body_pos = P(d.body_link_pos_w) - origins.unsqueeze(1)
        kin_i = ad.kinematics(body_pos, P(d.body_link_quat_w), J, P(d.gravity_compensation_forces),
                              root_pos - origins, root_quat)
        v_red = ad.reduced_velocity(root_lin_w, root_ang_b, read_jv)

        # (1) Isaac 内部自洽：用适配器的约定在 elbow_link 原点构造 Jacobian，J·v' 应等于 Isaac 的 body 速度
        R_B = quat_xyzw_to_matrix(root_quat.double())
        for s, b in enumerate(ad.elbow_body):
            p_link = body_pos[:, b].double()
            J_pt = ad._point_jac(p_link, J[:, b].double(), p_link, (root_pos - origins).double(), R_B)
            lin_pred = (J_pt @ v_red.unsqueeze(-1)).squeeze(-1)
            ang_pred = (kin_i.J_rot[:, s] @ v_red.unsqueeze(-1)).squeeze(-1)
            err = P(d.body_link_lin_vel_w)[:, b].double() - lin_pred
            omega_w = (R_B @ root_ang_b.double().unsqueeze(-1)).squeeze(-1)
            r_com = (P(d.root_com_pos_w) - P(d.root_link_pos_w)).double()
            print(f"  诊断：误差向量 vs ω×r_com(pelvis) 的最大差 {(err - torch.cross(omega_w, r_com, dim=-1)).abs().max():.2e}; "
                  f"|r_com| = {r_com.norm(dim=-1).max():.4f} m; |root_link_lin_vel_w − root_com_lin_vel_w| max "
                  f"{(P(d.root_link_lin_vel_w) - P(d.root_com_lin_vel_w)).abs().max():.2e}")
            print(f"[自洽] {UPPER_JOINTS[1 + 4 * s].split('_')[0]} elbow_link: |J·v' − body_link_lin_vel_w| max "
                  f"{(lin_pred - P(d.body_link_lin_vel_w)[:, b].double()).abs().max():.2e}, "
                  f"|J_rot·v' − body_link_ang_vel_w| max {(ang_pred - P(d.body_link_ang_vel_w)[:, b].double()).abs().max():.2e}")

        # (1b) 物理仲裁：自由下落 1 个物理步（无接触），手部点 / elbow 原点位置的有限差分 vs 预测（梯形平均，误差 O(dt²)）
        print(f"写入 vs 读回：|root_link_lin_vel_w − 写入线速度| max {(root_lin_w - vel[:, :3]).abs().max():.2e}")
        dt = u.physics_dt
        def snapshot():
            kin = ad.kinematics(P(d.body_link_pos_w) - origins.unsqueeze(1), P(d.body_link_quat_w), P(d.body_link_jacobian_w),
                                P(d.gravity_compensation_forces), P(d.root_link_pos_w) - origins, P(d.root_link_quat_w))
            vr = ad.reduced_velocity(P(d.root_link_lin_vel_w), P(d.root_link_ang_vel_b), P(d.joint_vel))
            pred = (kin.J_hand @ vr[:, None, :, None]).squeeze(-1)
            b_idx = ad.elbow_body
            link_pos = (P(d.body_link_pos_w) - origins.unsqueeze(1))[:, b_idx].double()
            link_vel_isaac = P(d.body_link_lin_vel_w)[:, b_idx].double()
            return kin.hand_pos.clone(), pred.clone(), link_pos.clone(), link_vel_isaac.clone()
        h0, p0, l0, lv0 = snapshot()
        u.sim.step(render=False)
        robot.update(dt)
        h1, p1, l1, lv1 = snapshot()
        fd_hand = (h1 - h0) / dt
        fd_link = (l1 - l0) / dt
        for tag, a_h, a_l in [("步前", p0, lv0), ("步后", p1, lv1), ("梯形", 0.5 * (p0 + p1), 0.5 * (lv0 + lv1))]:
            print(f"[物理仲裁] dt={dt} 用{tag}速度: |FD(hand) − 适配器 J·v'| max {(fd_hand - a_h).abs().max():.2e} m/s；"
                  f"|FD(elbow 原点) − Isaac body_link_lin_vel_w| max {(fd_link - a_l).abs().max():.2e} m/s")

        # (1c) 干净仲裁：关节锁定（PD 目标 = 当前角、关节速度 0），整机刚体自由下落 1 步
        robot.write_root_link_pose_to_sim_index(root_pose=pose)
        robot.write_root_link_velocity_to_sim_index(root_velocity=vel)
        robot.write_joint_state_to_sim_index(position=jp, velocity=torch.zeros_like(jv))
        robot.set_joint_position_target_index(target=jp)
        robot.set_joint_velocity_target_index(target=torch.zeros_like(jv))
        robot.write_data_to_sim()
        u.sim.forward()
        robot.update(0.0)
        h0, p0, l0, lv0 = snapshot()
        u.sim.step(render=False)
        robot.update(dt)
        h1, p1, l1, lv1 = snapshot()
        fd_hand, fd_link = (h1 - h0) / dt, (l1 - l0) / dt
        print(f"[刚体仲裁] 步后关节速度 max {P(d.joint_vel).abs().max():.2e} rad/s")
        for tag, a_h, a_l in [("步前", p0, lv0), ("步后", p1, lv1), ("梯形", 0.5 * (p0 + p1), 0.5 * (lv0 + lv1))]:
            print(f"[刚体仲裁] 用{tag}速度: |FD(hand) − 适配器 J·v'| max {(fd_hand - a_h).abs().max():.2e} m/s；"
                  f"|FD(elbow 原点) − Isaac body_link_lin_vel_w| max {(fd_link - a_l).abs().max():.2e} m/s")

        # (2) 与 MuJoCo（MJCF）对比
        m = mujoco.MjModel.from_xml_path(args_cli.mjcf)
        md = mujoco.MjData(m)
        cols = list(range(6)) + upper_dof(m)
        kins, grav = [], []
        for e in range(N):
            md.qpos[0:3] = (root_pos[e] - origins[e]).cpu().numpy()
            x, y, z, w = root_quat[e].cpu().numpy()
            md.qpos[3:7] = [w, x, y, z]
            md.qvel[0:3] = root_lin_w[e].cpu().numpy()
            md.qvel[3:6] = root_ang_b[e].cpu().numpy()
            for i, n in enumerate(names):
                jid = m.joint(n).id
                md.qpos[m.jnt_qposadr[jid]] = float(read_jp[e, i])
                md.qvel[m.jnt_dofadr[jid]] = float(read_jv[e, i])
            mujoco.mj_forward(m, md)
            k = kinematics_from_data(m, md)
            kins.append(k)
            md.qvel[:] = 0.0
            mujoco.mj_forward(m, md)
            grav.append(md.qfrc_bias[upper_dof(m)].copy())
        kin_m = stack(kins)
        red = lambda Jm: Jm[..., cols]
        report = [
            ("hand_pos [m]", kin_i.hand_pos, kin_m.hand_pos),
            ("forearm_axis", kin_i.forearm_axis, kin_m.forearm_axis),
            ("J_hand (15 列)", kin_i.J_hand, red(kin_m.J_hand)),
            ("J_rot (15 列)", kin_i.J_rot, red(kin_m.J_rot)),
            ("J_load (15 列)", kin_i.J_load, red(kin_m.J_load)),
            ("J_hand·v' [m/s]", (kin_i.J_hand @ v_red[:, None, :, None]).squeeze(-1).cpu(),
             (red(kin_m.J_hand) @ v_red.cpu()[:, None, :, None]).squeeze(-1)),
            ("g(q) 上肢 [N·m]", kin_i.bias[:, 6:], torch.as_tensor(np.array(grav))),
        ]
        print(f"\n[Isaac vs MuJoCo] N={N}（mujoco {mujoco.__version__}）")
        for name, a, b in report:
            diff = (a.detach().double().cpu() - b.detach().double().cpu()).abs()
            print(f"  {name:18s} max|Δ| = {diff.max():.2e}   (量级 {b.abs().max():.2e})")
        env.close()


if __name__ == "__main__":
    main()
