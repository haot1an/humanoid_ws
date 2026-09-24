"""离线单元测试 payload_curriculum 的逻辑（不需要 Isaac Lab：用假 env 与 isaaclab 桩模块）。
用法：/home/tt/miniconda3/envs/rl/bin/python rl/tests/test_payload_curriculum.py
"""
import pathlib
import sys
import types

import torch

# isaaclab 只在 mdp.py 顶部用于 SceneEntityCfg 类型；这里放一个桩，让模块可在本机导入
stub = types.ModuleType("isaaclab.managers")
stub.SceneEntityCfg = object
sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
sys.modules["isaaclab.managers"] = stub
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "h1_locomanip" / "tasks"))
import mdp  # noqa: E402


class FakeEnv:
    def __init__(self, num_envs, cap_init):
        self.num_envs = num_envs
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        term = types.SimpleNamespace(payload_cap=cap_init)
        self.action_manager = types.SimpleNamespace(get_term=lambda name: term)
        self.term = term


def run(enabled, episode_len, batches=400, batch=10, num_envs=100):
    env = FakeEnv(num_envs, cap_init=1.0)
    caps = []
    for k in range(batches):
        ids = torch.arange(k * batch, (k + 1) * batch) % num_envs
        env.episode_length_buf[ids] = episode_len
        caps.append(mdp.payload_curriculum(env, ids, enabled=enabled))
    first_rise = next((i for i, c in enumerate(caps) if c > 1.0), None)
    return caps[-1], first_rise, max(caps)


checks = []
cap, first, mx = run(enabled=False, episode_len=1000)
checks.append(("enabled=False：上限保持 1.0", cap == 1.0 and mx == 1.0, f"末值 {cap}"))
cap, first, mx = run(enabled=True, episode_len=500)
checks.append(("活满比例 0.5 < 0.9：不升档", cap == 1.0, f"末值 {cap}"))
cap, first, mx = run(enabled=True, episode_len=1000)
checks.append(("活满比例 1.0：升档并封顶 5.0", abs(cap - 5.0) < 1e-9 and mx <= 5.0 + 1e-9, f"末值 {cap:.3f}，第 {first} 批首次升档"))
# 升档速度：EMA 从 0 越过 0.9 需要 ln(0.1)/ln(0.95) ≈ 44.9 批；之后每批 +0.5×10/100 = 0.05 kg，4 kg 需 80 批
checks.append(("首次升档出现在第 45 批附近", first is not None and 44 <= first <= 46, f"实测第 {first} 批"))
_, first_full, _ = (None, None, None)
env = FakeEnv(100, 1.0)
for k in range(400):
    ids = torch.arange(k * 10, (k + 1) * 10) % 100
    env.episode_length_buf[ids] = 1000
    c = mdp.payload_curriculum(env, ids, enabled=True)
    if c >= 5.0 - 1e-9:
        first_full = k
        break
checks.append(("封顶约在第 45 + 80 = 125 批", first_full is not None and 120 <= first_full <= 130, f"实测第 {first_full} 批"))

ok = all(c[1] for c in checks)
for name, passed, info in checks:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}：{info}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
