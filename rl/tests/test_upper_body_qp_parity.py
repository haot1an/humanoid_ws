"""torch 批量版 UBQP-001 与 C++ UpperBodyQp 在同一状态下逐项对比（开环：每拍都用 C++ 记录的 q_des_before）。

输入：upper_body_qp_test --dump_ctrl 导出的 CSV（每行一次 QP 调用）
用法：/home/tt/miniconda3/envs/rl/bin/python tests/test_upper_body_qp_parity.py <dump.csv> --k_hand 10 --k_pitch 5 --w_pitch 10
"""
import argparse
import pathlib
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from h1_locomanip.mujoco_kinematics import kinematics_from_data, stack, upper_dof, upper_limits  # noqa: E402
from h1_locomanip.upper_body_qp import BatchedUpperBodyQp, UpperBodyQpCfg  # noqa: E402

SCENE = pathlib.Path(__file__).resolve().parents[2] / "src/humanoid_wbc/model/unitree_h1/scene.xml"

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--k_hand", type=float, default=20.0)
ap.add_argument("--k_pitch", type=float, default=10.0)
ap.add_argument("--w_pitch", type=float, default=0.0)
ap.add_argument("--feedforward", type=int, default=1)
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(str(SCENE))  # 控制器侧模型：不含负载（与 C++ 的 mc 相同）
d = mujoco.MjData(m)
header = open(args.dump).readline()
nq, nv, nu = [int(x.split("=")[1]) for x in header[1:].split()]
assert (nq, nv) == (m.nq, m.nv), f"dump 的 nq/nv {nq}/{nv} 与模型 {m.nq}/{m.nv} 不一致"
D = np.loadtxt(args.dump, delimiter=",", comments="#")
cols, i = {}, 0
for name, width in [("t", 1), ("qpos", nq), ("qvel", nv), ("q_upper", nu), ("q_nominal", nu), ("q_des_before", nu),
                    ("ref_pos", 6), ("ref_vel", 6), ("ff_mass", 1), ("x", nu), ("q_des_after", nu), ("tau_ff", nu)]:
    cols[name] = D[:, i:i + width]
    i += width
assert i == D.shape[1], f"列数 {D.shape[1]} 与布局 {i} 不符"
N = len(D)

kins = []
for r in range(N):
    d.qpos[:] = cols["qpos"][r]
    d.qvel[:] = cols["qvel"][r]
    mujoco.mj_forward(m, d)
    kins.append(kinematics_from_data(m, d))
kin = stack(kins)

cfg = UpperBodyQpCfg(k_hand=args.k_hand, k_pitch=args.k_pitch, w_pitch=args.w_pitch, feedforward=bool(args.feedforward))
q_min, q_max = upper_limits(m)
qp = BatchedUpperBodyQp(N, nv, upper_dof(m), q_min, q_max, cfg)
T = lambda a: torch.as_tensor(a, dtype=torch.float64)
qp.reset(slice(None), T(cols["q_des_before"]))
x = qp.update(kin, T(cols["ref_pos"]).view(N, 2, 3), T(cols["ref_vel"]).view(N, 2, 3), T(cols["qvel"]),
              T(cols["q_upper"]), T(cols["q_nominal"]))
tau = qp.tau_feedforward(kin, T(cols["ff_mass"][:, 0]))

print(f"mujoco(py) {mujoco.__version__}, torch {torch.__version__}, {N} 个控制拍, cfg={cfg}")
report = {"x = dq [rad/s]": (x, cols["x"]), "q_des_after [rad]": (qp.q_des, cols["q_des_after"]),
          "tau_ff [N·m]": (tau, cols["tau_ff"])}
for name, (a, b) in report.items():
    diff = (a.numpy() - b).__abs__()
    print(f"{name:20s} max|torch − C++| = {diff.max():.2e}   中位数 {np.median(diff.max(1)):.1e}")
print(f"积极集未收敛问题数 {qp.unconverged}")
