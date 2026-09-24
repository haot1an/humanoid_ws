# rl/ — H1 loco-manipulation 的 RL 部分（DEC-004）

与 `src/humanoid_wbc/`（C++ MuJoCo 控制栈）并列；同步到训练服务器 `/data/h1_locomanip/`。
设计见 `docs/DESIGN.md` RL-LOCO-001、UBQP-001；实验见 `docs/EXPERIMENTS.md`。

```text
h1_locomanip/
  box_qp.py            批量盒约束 QP（积极集），训练端替代 OSQP（EXP-008）
  upper_body_qp.py     UBQP-001 的 torch 批量版，逐项对应 C++ UpperBodyQp
  mujoco_kinematics.py 从 MuJoCo 状态构造上肢运动学（一致性测试 / Python sim2sim 用）
tests/
  test_upper_body_qp_parity.py  torch 版与 C++ 版在同一状态下逐项对比
```
