#!/usr/bin/env python3
"""
从 MuJoCo 模型的质量矩阵离线标定关节 PD 增益。

原理
----
关节 PD 的闭环是二阶系统   I·ë + Kd·ė + Kp·e = 0
    ωn = sqrt(Kp / I)          自然频率（响应快慢）
    ζ  = Kd / (2·sqrt(Kp·I))   阻尼比

反过来设计：给定目标 (ωn, ζ)，
    Kp = I·ωn²
    Kd = 2ζ·I·ωn − damping     ← 必须减去 MuJoCo 内建的 <joint damping>

这样全身各关节的响应特性一致，调参旋钮从 38 个（19×Kp + 19×Kd）收敛成 2 个。

I 从哪来
-------
mj_fullM 的对角线元素 M[dof,dof] 就是该关节的等效转动惯量，且已包含
<joint armature>（电机转子反射惯量）——那份惯量电机确实要驱动，应当计入。

注意 I 强烈依赖关节位形（髋部可变化 ±60%），因此必须指定一个代表位形。
本脚本用 keyframe "home"。I 与基座位姿无关（实测差异 ~1e-16）。

用法
----
    python3 compute_pd_gains.py
输出末尾是可直接粘贴进 include/humanoid_wbc/h1_config.hpp 的 C++ 代码。
任一自检不通过则以非零状态退出。
"""

import os
import sys

import mujoco
import numpy as np

# ─────────────────────────── 可调参数 ───────────────────────────

OMEGA_N = 20.0      # 目标闭环带宽 [rad/s]  (20 rad/s ≈ 3.2 Hz)
ZETA = 1.0          # 目标阻尼比    (1.0 = 临界阻尼，无超调)
STEP = 0.3          # 力矩预算校核用的阶跃幅度 [rad]
KEYFRAME = "home"   # 计算惯量所用的位形

# 关节顺序必须与 include/humanoid_wbc/h1_config.hpp 的 enum H1Joint 完全一致
JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",

    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",

    "torso",

    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",

    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(_HERE, "..", "model", "unitree_h1", "scene.xml")


def die(msg):
    """自检失败一律以非零状态退出 —— 警告会被忽略，退出码不会。"""
    sys.exit(f"[FAIL] {msg}")


# ─────────────────────── 1. 加载模型，摆到 home ───────────────────────

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 用名字查 keyframe，不硬编码索引
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
if key_id < 0:
    die(f"模型中找不到 keyframe '{KEYFRAME}'")

# 该调用同时把 qvel 清零 —— 这样 qfrc_bias 恰好等于纯重力项 g(q)
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

if not np.allclose(data.qvel, 0.0):
    die("qvel 非零，qfrc_bias 将混入科氏力，不是纯重力")


# ─────────────────────── 2. 取质量矩阵 ───────────────────────

M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M, data.qM)


# ─────────────────────── 3. 逐关节收集 ───────────────────────

if len(JOINT_NAMES) != model.nu:
    die(f"JOINT_NAMES 有 {len(JOINT_NAMES)} 项，但模型 nu={model.nu}")

records = []
for name in JOINT_NAMES:
    # 取值 → 立刻验证 → 才能用
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        die(f"找不到关节 '{name}'")
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
        die(f"'{name}' 不是 hinge 关节")

    dof = model.jnt_dofadr[jid]

    # 反查驱动该关节的 actuator。注意：Python binding 把 actuator_trnid
    # reshape 成 (nu, 2)，所以这里是 [a][0]；C 里必须写 [2*a+0]。
    aid = next(
        (a for a in range(model.nu)
         if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
         and model.actuator_trnid[a][0] == jid),
        -1,
    )
    if aid < 0:
        die(f"'{name}' 没有对应的 actuator")

    I = M[dof, dof]                         # 等效惯量（含 armature）
    damping = model.dof_damping[dof]        # MuJoCo 自动施加的被动阻尼
    g = data.qfrc_bias[dof]                 # qvel=0 → 纯重力力矩
    ctrl_max = model.actuator_ctrlrange[aid][1]

    kp = I * OMEGA_N ** 2
    kd_ideal = 2.0 * ZETA * I * OMEGA_N
    kd_write = kd_ideal - damping           # ← 写进 h1_config 的是这个

    records.append(dict(name=name, jid=jid, dof=dof, aid=aid, I=I,
                        damping=damping, g=g, ctrl_max=ctrl_max,
                        kp=kp, kd_ideal=kd_ideal, kd_write=kd_write))


# ─────────────────────── 4. 自检 ───────────────────────

# (a) Kd 不能为负：模型自带阻尼已超过目标阻尼
for r in records:
    if r["kd_write"] < 0.0:
        die(f"{r['name']}: Kd={r['kd_write']:.2f} < 0，"
            f"内建 damping={r['damping']:.1f} 已超过目标，请提高 OMEGA_N")

# (b) 力矩预算：阶跃瞬间的力矩不得超过 ctrlrange
for r in records:
    r["tau_peak"] = r["kp"] * STEP + abs(r["g"])
    r["budget"] = r["tau_peak"] / r["ctrl_max"]
    if r["budget"] > 1.0:
        die(f"{r['name']}: {STEP} rad 阶跃需 {r['tau_peak']:.1f} N·m，"
            f"超过上限 {r['ctrl_max']:.0f} N·m（{r['budget']*100:.0f}%）")

# (c) 左右对称 —— 不需要任何外部参照就能抓出索引错误
by_name = {r["name"]: r for r in records}
for r in records:
    if not r["name"].startswith("left_"):
        continue
    mate = by_name.get("right_" + r["name"][len("left_"):])
    if mate is None:
        die(f"{r['name']} 找不到对应的右侧关节")
    if abs(r["I"] - mate["I"]) > 1e-9:
        die(f"左右不对称: {r['name']} I={r['I']:.6f} "
            f"vs {mate['name']} I={mate['I']:.6f} —— 索引可能错了")

# (d) actuator 不得重复占用
seen = {}
for r in records:
    if r["aid"] in seen:
        die(f"actuator {r['aid']} 同时被 {seen[r['aid']]} 和 {r['name']} 占用")
    seen[r["aid"]] = r["name"]


# ─────────────────────── 5. 报告 ───────────────────────

print(f"位形: keyframe '{KEYFRAME}'   左膝角 = {data.qpos[10]:.2f} rad")
print(f"目标: omega_n = {OMEGA_N} rad/s   zeta = {ZETA}   阶跃校核 = {STEP} rad")
print()
print(f"{'关节':<22}{'I':>8}{'g(q)':>8}{'Kp':>9}{'Kd理想':>8}"
      f"{'内建':>6}{'Kd写入':>8}{'上限':>7}{'阶跃τ':>8}{'预算':>7}")
print("─" * 93)
for r in records:
    warn = "  <-- 余量薄" if r["budget"] > 0.8 else ""
    print(f"{r['name']:<22}{r['I']:>8.4f}{r['g']:>8.2f}{r['kp']:>9.2f}"
          f"{r['kd_ideal']:>8.2f}{r['damping']:>6.1f}{r['kd_write']:>8.2f}"
          f"{r['ctrl_max']:>7.0f}{r['tau_peak']:>8.1f}{r['budget']*100:>6.0f}%{warn}")

worst = max(records, key=lambda r: r["budget"])
print(f"\n[OK] 全部自检通过。力矩预算最紧: {worst['name']} "
      f"{worst['budget']*100:.0f}%")


# ─────────────────────── 6. 生成 C++ ───────────────────────

header = (f"// 由 scripts/compute_pd_gains.py 生成，请勿手改。\n"
          f"// 位形 keyframe \"{KEYFRAME}\"   omega_n={OMEGA_N} rad/s   zeta={ZETA}\n"
          f"// Kp = I*omega_n^2 ;  Kd = 2*zeta*I*omega_n - <joint damping>\n"
          f"// I 取自 mj_fullM 对角线（含 armature），随关节位形变化，此处为 "
          f"\"{KEYFRAME}\" 位形。")

print("\n" + "=" * 93)
print("以下内容粘贴进 include/humanoid_wbc/h1_config.hpp")
print("=" * 93 + "\n")
print(header)
for field, label in (("kp", "kDefaultKp"), ("kd_write", "kDefaultKd")):
    print(f"inline constexpr std::array<double, kNumJoints> {label} = {{")
    for r in records:
        print(f"    {r[field]:8.2f},   // {r['name']}")
    print("};")
    print(f"static_assert({label}.size() == kNumJoints,"
          f" \"{label} 长度与 enum 不一致！\");")
    print()
