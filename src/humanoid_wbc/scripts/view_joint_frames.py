"""查看 H1 每个关节的位置/转轴, 以及 Jacobian 列与关节转轴的关系. 纯运动学 (mj_forward), 不做动力学.

用法 (Python 环境: /home/tt/miniconda3/envs/rl/bin/python):
    view_joint_frames.py --table              # 打印关节表 + elbow_link 旋转 Jacobian 逐列核对, 不开窗口
    view_joint_frames.py                      # 窗口: 默认姿态, 显示关节轴箭头 + body 坐标系 + 关节名
    view_joint_frames.py --sweep              # 窗口: 逐个关节 ±0.4 rad 摆动, 终端打印当前关节
    view_joint_frames.py --png out.png        # 离屏渲染一张带关节轴的截图

颜色约定 (MuJoCo): body 坐标系 红=x 绿=y 蓝=z; 关节轴为箭头, 方向即 xaxis (右手定则的正转方向).
姿态: pelvis (0,0,1.05) 单位姿态, 关节取 Isaac Lab H1 默认角 (与 EXP-003/004 一致).
"""
import argparse
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

SCENE = pathlib.Path(__file__).resolve().parents[1] / "model/unitree_h1/scene.xml"
DEFAULT_POS = {  # Isaac Lab H1 init_state.joint_pos (EXP-003 params/env.yaml)
    "hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": -0.28, "knee": 0.79, "ankle": -0.52,
    "torso": 0.0, "shoulder_pitch": 0.28, "shoulder_roll": 0.0, "shoulder_yaw": 0.0, "elbow": 0.52,
}


def load():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    hang = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "hang")
    if hang >= 0:
        d.eq_active[hang] = 0
    d.qpos[0:3] = [0.0, 0.0, 1.05]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    joints = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    for j in joints:
        name = m.joint(j).name
        key = name.replace("left_", "").replace("right_", "")
        d.qpos[m.jnt_qposadr[j]] = DEFAULT_POS[key]
    mujoco.mj_forward(m, d)
    return m, d, joints


def print_table(m, d, joints):
    print(f"{'joint':22s} {'dof':>3s} {'parent body':22s} {'anchor_world [m]':>26s} "
          f"{'axis_world':>22s} {'axis_in_body':>18s} {'range [rad]':>16s}")
    for j in joints:
        body = m.body(m.jnt_bodyid[j]).name
        print(f"{m.joint(j).name:22s} {m.jnt_dofadr[j]:3d} {body:22s} "
              f"{np.array2string(d.xanchor[j], precision=3, suppress_small=True):>26s} "
              f"{np.array2string(d.xaxis[j], precision=3, suppress_small=True):>22s} "
              f"{np.array2string(m.jnt_axis[j], precision=0):>18s} "
              f"{np.array2string(m.jnt_range[j], precision=2):>16s}")

    # Jacobian 逐列核对: 旋转 Jacobian 第 k 列 = 只有 qdot_k = 1 时 elbow_link 的角速度 (world 系)
    target = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, jacr, target)
    print(f"\n== left_elbow_link 旋转 Jacobian J_rot (3 x {m.nv}) 逐列 ==")
    R_pelvis = d.xmat[m.body("pelvis").id].reshape(3, 3)
    print("列 0-2 (基座线速度, world 系):", np.abs(jacr[:, 0:3]).max(), "-> 平移不产生转动, 全 0")
    print("列 3-5 (基座角速度, pelvis 系) 与 R_pelvis 的最大差:", np.abs(jacr[:, 3:6] - R_pelvis).max())
    ancestors = set()
    b = target
    while b > 0:
        ancestors.add(b)
        b = m.body_parentid[b]
    for j in joints:
        k = m.jnt_dofadr[j]
        on_chain = m.jnt_bodyid[j] in ancestors
        col = jacr[:, k]
        expect = d.xaxis[j] if on_chain else np.zeros(3)
        print(f"  列 {k:2d} {m.joint(j).name:22s} 在运动链上={str(on_chain):5s} "
              f"J_rot 列={np.array2string(col, precision=3, suppress_small=True):24s} "
              f"|列 - 期望|={np.abs(col - expect).max():.1e}")


def show_options(opt, m):
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    opt.label = mujoco.mjtLabel.mjLABEL_JOINT
    m.vis.scale.jointlength = 5.0   # 以 m.stat.meansize 为单位
    m.vis.scale.jointwidth = 0.25
    m.vis.scale.framelength = 3.0
    m.vis.scale.framewidth = 0.15


def set_camera(cam, m):
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("torso_link").id
    cam.distance = 2.2
    cam.elevation = -10.0
    cam.azimuth = 150.0


def run_window(m, d, joints, sweep):
    with mujoco.viewer.launch_passive(m, d) as viewer:
        show_options(viewer.opt, m)
        set_camera(viewer.cam, m)
        q0 = d.qpos.copy()
        idx, t_start = 0, time.perf_counter()
        if sweep:
            print("逐个关节摆动, 每个 3 s; 正方向 = 关节轴箭头按右手定则的转动方向")
        while viewer.is_running():
            if sweep:
                t = time.perf_counter() - t_start
                if t > 3.0:
                    idx, t_start, t = (idx + 1) % len(joints), time.perf_counter(), 0.0
                    d.qpos[:] = q0
                j = joints[idx]
                if t == 0.0:
                    print(f"  -> {m.joint(j).name}  axis_world={np.round(d.xaxis[j], 3)}")
                lo, hi = m.jnt_range[j]
                a = q0[m.jnt_qposadr[j]]
                d.qpos[m.jnt_qposadr[j]] = np.clip(a + 0.4 * np.sin(2 * np.pi * t / 3.0), lo, hi)
                mujoco.mj_forward(m, d)
            viewer.sync()
            time.sleep(0.02)


def save_png(m, d, path, width=1280, height=960):
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    renderer = mujoco.Renderer(m, height, width)
    opt = mujoco.MjvOption()
    show_options(opt, m)
    cam = mujoco.MjvCamera()
    set_camera(cam, m)
    renderer.update_scene(d, camera=cam, scene_option=opt)
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                    "-i", "-", str(path)], input=renderer.render().tobytes(), check=True)
    print("[png]", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--png")
    args = ap.parse_args()
    m, d, joints = load()
    if args.table:
        print_table(m, d, joints)
    elif args.png:
        save_png(m, d, args.png)
    else:
        run_window(m, d, joints, args.sweep)


if __name__ == "__main__":
    main()
