"""DEC-008 第 7 项：从 EXP-018（v2，actor 60 → 10，critic 71 → 1）热启动 v4（actor 71 → 12，critic 82 → 1）。

按观测项名字映射输入列（v4 布局由 smoke_test_box.py 打印核对）：
  v2 actor : lin3 ang3 grav3 cmd3 jpos19 jvel19 actions10                          = 60
  v4 actor : lin3 ang3 grav3 cmd3 jpos19 jvel19 actions12(腿10 + Δ2) box_pose9      = 71
  v2 critic: 同 actor 前 60 + payload2 + upper_targets9                             = 71
  v4 critic: 前 50 + actions12 + payload2 + upper_targets9 + box_pose9              = 82
新增输入列权重置 0（网络初始输出与 EXP-018 完全相同）；actor 新增 2 个输出行置 0（Δ 初始为 0）；
新增 2 维动作的 std 取 new_std。优化器动量丢弃（形状已变），迭代数置 0。
用法（服务器）：python scripts/warmstart_v4.py <exp018 model_1999.pt> <out model_0.pt>
"""
import sys

import torch

NEW_STD = 1.0


def map_cols(w_old, segments, n_new):
    """segments: [(old_start, new_start, length)]，其余新列为 0。"""
    w = torch.zeros(w_old.shape[0], n_new, dtype=w_old.dtype)
    for o, n, k in segments:
        w[:, n:n + k] = w_old[:, o:o + k]
    return w


def main(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    a, c = dict(ck["actor_state_dict"]), dict(ck["critic_state_dict"])
    assert a["mlp.0.weight"].shape == (128, 60) and a["mlp.6.weight"].shape == (10, 128), "源不是 v2 actor"
    assert c["mlp.0.weight"].shape == (128, 71), "源不是 v2 critic"
    # actor 输入：前 60 列原位（actions 的前 10 列即腿），60..70 新增
    a["mlp.0.weight"] = map_cols(a["mlp.0.weight"], [(0, 0, 60)], 71)
    # actor 输出：腿 10 行原样，Δ 2 行置 0
    a["mlp.6.weight"] = torch.cat([a["mlp.6.weight"], torch.zeros(2, 128)], 0)
    a["mlp.6.bias"] = torch.cat([a["mlp.6.bias"], torch.zeros(2)], 0)
    a["distribution.std_param"] = torch.cat([a["distribution.std_param"], torch.full((2,), NEW_STD)], 0)
    # critic 输入：0..59 原位；v2 payload(60..61) + upper_targets(62..70) → v4 62..72；v4 60..61（Δ 动作）与 73..81（箱子）为 0
    c["mlp.0.weight"] = map_cols(c["mlp.0.weight"], [(0, 0, 60), (60, 62, 11)], 82)
    opt = ck["optimizer_state_dict"]
    opt = {"state": {}, "param_groups": opt["param_groups"]}
    torch.save({"actor_state_dict": a, "critic_state_dict": c, "optimizer_state_dict": opt, "iter": 0,
                "infos": {"warmstart_from": src}}, dst)
    print(f"[warmstart] {src} → {dst}；腿部 std {ck['actor_state_dict']['distribution.std_param'].tolist()}，"
          f"Δ std {NEW_STD}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
