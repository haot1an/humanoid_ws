"""比较端箱候选姿态: A = 上臂竖直 (θ=0), B = 上臂前倾 θ. 纯运动学/静力学, 不做动力学仿真.

每个姿态对每条手臂解 4 个方程 (未知数: 肩 pitch/roll/yaw、肘):
    上臂矢状面前倾角 = θ,  前臂水平 a_z = 0,  前臂朝正前 a_y = 0,  手部点 y = ±spacing/2
torso = 0, pelvis 固定在 (0,0,1.05) 竖直 (与 view_joint_frames.py 相同).

用法 (Python: /home/tt/miniconda3/envs/rl/bin/python):
    view_carry_poses.py --png out.png            # 打印指标 + 渲染 侧视/正视 对比图 (箱子为示意几何, 不参与物理)
    view_carry_poses.py --viewer 15              # 窗口查看 θ=15° 的姿态
    view_carry_poses.py --viewer 15 --spacing 0.36
"""
import argparse
import subprocess

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares

from view_joint_frames import load

HAND_LOCAL = np.array([0.28, 0.0, -0.015])   # elbow_link 系手部点, h1.xml:178
AXIS_LOCAL = HAND_LOCAL / np.linalg.norm(HAND_LOCAL)
SPACING = 0.43                                # 两手间距 [m], 用户 2026-09-18
BOX_MASS = 5.0                                # 示例负载 [kg], 仅用于力矩估算
BOX_D = 0.15                                  # 箱子压在前臂上距肘的距离 [m]
BOX_SIZE = np.array([0.30, 0.50, 0.20])       # 示意箱子 x 深 / y 宽 / z 高 [m]
ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"]


def arm_state(m, d, side):
    eb = m.body(f"{side}_elbow_link").id
    R = d.xmat[eb].reshape(3, 3)
    hand = d.xpos[eb] + R @ HAND_LOCAL
    a = R @ AXIS_LOCAL
    shoulder = d.xanchor[m.joint(f"{side}_shoulder_pitch").id].copy()
    elbow = d.xanchor[m.joint(f"{side}_elbow").id].copy()
    return hand, a, shoulder, elbow


def upper_arm_tilt(shoulder, elbow):
    u = elbow - shoulder
    return np.degrees(np.arctan2(u[0], -u[2]))   # 矢状面内相对竖直向下的前倾角, 前为正


def solve_arm(m, d, side, theta_deg):
    adr = [m.jnt_qposadr[m.joint(f"{side}_{j}").id] for j in ARM_JOINTS]
    lo = np.array([m.jnt_range[m.joint(f"{side}_{j}").id][0] for j in ARM_JOINTS])
    hi = np.array([m.jnt_range[m.joint(f"{side}_{j}").id][1] for j in ARM_JOINTS])
    y_target = SPACING / 2 * (1 if side == "left" else -1)

    def residual(x):
        d.qpos[adr] = x
        mujoco.mj_kinematics(m, d)
        hand, a, sh, el = arm_state(m, d, side)
        return [np.radians(upper_arm_tilt(sh, el) - theta_deg), a[2], a[1], hand[1] - y_target]

    sol = least_squares(residual, np.zeros(4), bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
    d.qpos[adr] = sol.x
    mujoco.mj_forward(m, d)
    return sol.x, np.abs(sol.fun).max(), lo, hi


def metrics(m, d, side, q, lo, hi):
    hand, a, sh, el = arm_state(m, d, side)
    eb = m.body(f"{side}_elbow_link").id
    dofs = [m.jnt_dofadr[m.joint(f"{side}_{j}").id] for j in ARM_JOINTS]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, jacr, hand, eb)
    J_pitch = np.cross(a, [0, 0, 1]) @ jacr
    J_task = np.vstack([jacp[:, dofs], J_pitch[dofs]])        # 4 x 4: 手部位置 3 + 俯仰 1
    sigma_min = np.linalg.svd(J_task, compute_uv=False).min()
    # 静力矩: 模型重力 (qfrc_bias 在零速度下 = g(q)) + 箱子一半重量作用在前臂距肘 BOX_D 处
    p_box = el + BOX_D * a
    jb = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jb, None, p_box, eb)
    tau_box = jb[:, dofs].T @ np.array([0, 0, -BOX_MASS / 2 * 9.81])
    tau = d.qfrc_bias[dofs] - tau_box  # 需要电机提供的保持力矩 = 重力项 + 抵消箱子
    limits = [m.actuator_ctrlrange[m.actuator(f"{side}_{j}").id][1] for j in ARM_JOINTS]
    return {
        "q": q, "limit_margin": np.minimum(q - lo, hi - q), "hand": hand, "a": a, "sh": sh, "el": el,
        "sigma_min": sigma_min, "tau": tau, "tau_ratio": np.abs(tau) / np.array(limits),
    }


def set_pose(m, d, theta_deg):
    d.qpos[m.jnt_qposadr[m.joint("torso").id]] = 0.0
    out = {}
    for side in ["left", "right"]:
        q, res, lo, hi = solve_arm(m, d, side, theta_deg)
        out[side] = metrics(m, d, side, q, lo, hi) | {"residual": res}
    return out


def report(theta_deg, r):
    L, R = r["left"], r["right"]
    print(f"\n=== θ = {theta_deg:.0f}° ({'A: 上臂竖直' if theta_deg == 0 else 'B: 上臂前倾'})   IK 残差 max {max(L['residual'], R['residual']):.1e}")
    print("  左臂关节 [肩pitch, 肩roll, 肩yaw, 肘] rad:", np.round(L["q"], 3),
          f"  肘解剖角 ≈ {90 + np.degrees(L['q'][3]):.0f}°（H1 肘 0 = 90°）")
    print("  离关节限位最小余量 rad:", np.round(L["limit_margin"], 3))
    print(f"  左手 {np.round(L['hand'], 3)}  右手 {np.round(R['hand'], 3)}  间距 {L['hand'][1] - R['hand'][1]:.3f} m")
    print(f"  手在肩前 {L['hand'][0] - L['sh'][0]:.3f} m, 肩下 {L['sh'][2] - L['hand'][2]:.3f} m;  前臂 a = {np.round(L['a'], 3)}")
    print(f"  [手位置3+俯仰1] 4x4 任务 Jacobian 最小奇异值 σ_min = {L['sigma_min']:.4f}  (越小越接近奇异)")
    print(f"  托 {BOX_MASS:.0f} kg 箱 (距肘 {BOX_D} m) 静力矩 N·m [肩pitch, 肩roll, 肩yaw, 肘]:", np.round(L["tau"], 2),
          " 占上限:", np.round(L["tau_ratio"], 2))


def add_box(scene, r):
    """在渲染场景里加一个示意箱子: 底面搭在两前臂距肘 BOX_D 处的上表面 (capsule 半径 0.025)."""
    L, R = r["left"], r["right"]
    c = 0.5 * ((L["el"] + BOX_D * L["a"]) + (R["el"] + BOX_D * R["a"]))
    c[2] += 0.025 + BOX_SIZE[2] / 2
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, BOX_SIZE / 2, c, np.eye(3).flatten(),
                        np.array([0.85, 0.6, 0.3, 0.55], dtype=np.float32))
    scene.ngeom += 1


def render_views(m, d, r, width=640, height=640):
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    renderer = mujoco.Renderer(m, height, width)
    frames = []
    for azimuth in (90.0, 180.0):   # 侧视 / 正视
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.1, 0.0, 1.15]
        cam.distance, cam.elevation, cam.azimuth = 1.8, -5.0, azimuth
        renderer.update_scene(d, camera=cam)
        add_box(renderer.scene, r)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def main():
    global SPACING
    ap = argparse.ArgumentParser()
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.0, 15.0, 20.0])
    ap.add_argument("--png")
    ap.add_argument("--viewer", type=float, help="打开窗口查看该 θ [deg]")
    ap.add_argument("--spacing", type=float, default=SPACING, help="两手间距 [m]")
    args = ap.parse_args()
    SPACING = args.spacing
    m, d, _ = load()

    if args.viewer is not None:
        r = set_pose(m, d, args.viewer)
        report(args.viewer, r)
        with mujoco.viewer.launch_passive(m, d) as viewer:
            add_box(viewer.user_scn, r)
            while viewer.is_running():
                viewer.sync()
        return

    rows = []
    for th in args.thetas:
        r = set_pose(m, d, th)
        report(th, r)
        if args.png:
            rows.append(np.concatenate(render_views(m, d, r), axis=1))
    if args.png:
        img = np.concatenate(rows, axis=0)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{img.shape[1]}x{img.shape[0]}", "-i", "-", args.png],
                       input=img.tobytes(), check=True)
        print("\n[png]", args.png, "  行 = θ", args.thetas, "; 左列侧视, 右列正视")


if __name__ == "__main__":
    main()
