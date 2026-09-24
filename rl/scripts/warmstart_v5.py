"""DEC-009 第 7 项：从 EXP-020（v4：actor 71 → 12，critic 82 → 1）热启动 v5（actor 73，critic 84）。
v5 只在两个观测组末尾各追加 gait_clock(2)，其余布局不变（isaac_symmetry_check.py 打印核对）：
新增两列权重置 0（初始行为与 EXP-020 相同），输出层与 std 原样保留；优化器动量丢弃，迭代数置 0。
用法（服务器）：python scripts/warmstart_v5.py <exp020 model_1999.pt> <out model_0.pt>
"""
import sys

import torch


def pad_cols(w, n_new):
    return torch.cat([w, torch.zeros(w.shape[0], n_new - w.shape[1], dtype=w.dtype)], 1)


def main(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    a, c = dict(ck["actor_state_dict"]), dict(ck["critic_state_dict"])
    assert a["mlp.0.weight"].shape == (128, 71) and a["mlp.6.weight"].shape == (12, 128), "源不是 v4 actor"
    assert c["mlp.0.weight"].shape == (128, 82), "源不是 v4 critic"
    a["mlp.0.weight"] = pad_cols(a["mlp.0.weight"], 73)
    c["mlp.0.weight"] = pad_cols(c["mlp.0.weight"], 84)
    opt = {"state": {}, "param_groups": ck["optimizer_state_dict"]["param_groups"]}
    torch.save({"actor_state_dict": a, "critic_state_dict": c, "optimizer_state_dict": opt, "iter": 0,
                "infos": {"warmstart_from": src}}, dst)
    print(f"[warmstart] {src} → {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
