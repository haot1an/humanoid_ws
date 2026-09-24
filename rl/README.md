# rl/ — 训练与部署（Isaac Lab + MuJoCo）

项目总览与结果见仓库根目录 [README](../README.md)。

```text
h1_locomanip/
  box_qp.py              GPU 批量盒约束 QP（积极集 / 投影牛顿），训练端代替 OSQP
  upper_body_qp.py       上肢加权 QP 的 torch 批量版，逐项对应 C++ UpperBodyQp
  isaac_kinematics.py    从 Isaac 状态构造与 MuJoCo 约定一致的上肢运动学（关节列、基座列、xyzw 四元数）
  mujoco_kinematics.py   从 MuJoCo 状态构造上肢运动学（一致性测试 / sim2sim 用）
  tasks/
    __init__.py          任务注册：Carry-v2（附着质量）、CarryBox-v3（真实箱子）、CarryBoxRes-v4（混合方案）、CarryBoxGait-v5（周期步态 + 镜像增强，最终版）
    h1_symmetry.py       左右镜像映射（rsl-rl symmetry data augmentation）
    carry_env_cfg.py     环境配置：执行器、动作、观测（非对称 AC）、奖励、终止、事件、课程
    upper_body_action.py 上肢 QP action term：跟随坐标系 + 指令超前、100 Hz QP、负载前馈、v4 的 Δ 修正
    mdp.py               负载 / 箱子重置、课程、残差与箱子滑动奖励、掉箱与躯干触地终止、箱子位姿观测
scripts/
  sim2sim_carry.py       MuJoCo 部署评估：附着 / 自由箱子、航向闭环、箱体间隙、滑移、步态量化、录像
  export_contract.py     导出训练端契约（关节顺序、分组 PD 增益、限幅、QP 参数）
  smoke_test_env.py      v2 环境冒烟测试
  smoke_test_box.py      v3 / v4 环境冒烟测试（箱子放置、落定、终止项正反例、步态着地占比）
  warmstart_v4.py        从 v2 策略按观测项名字映射权重，热启动 v4
  warmstart_v5.py        从 v4 策略热启动 v5（追加时钟观测列）
tests/
  test_upper_body_qp_parity.py  torch 版 vs C++ 版逐项对比
  isaac_contract_check.py       Isaac vs MuJoCo 运动学 / 重力契约
  test_payload_curriculum.py    课程逻辑离线测试（假环境）
  isaac_symmetry_check.py       镜像映射检查（两次镜像 = 恒等、关节对调 / 取负、默认姿态对称）
contract/                       carry_v2*.json / carry_v3.json / carry_v4.json / carry_v5.json
```

训练入口：`isaaclab.sh train --rl_library rsl_rl --task <任务名> --external_callback h1_locomanip.tasks.register`（`PYTHONPATH` 需包含本目录）。
