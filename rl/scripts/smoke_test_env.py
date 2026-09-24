"""H1-LocoManip-Carry-v2 冒烟测试：建环境、零动作跑若干步，检查负载写入、QP 数值、各 reward 项量级与耗时。
用法（服务器）：cd /data/h1_locomanip/rl && python scripts/smoke_test_env.py --num_envs 64 --steps 150
"""
import argparse
import pathlib
import sys
import time

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=150)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import h1_locomanip.tasks as tasks  # noqa: E402

tasks.register()


def main():
    env_cfg, _ = resolve_task_config(tasks.TASK_ID, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(tasks.TASK_ID, cfg=env_cfg)
        obs, _ = env.reset()
        u = env.unwrapped
        term = u.action_manager.get_term("upper_body")
        robot = u.scene["robot"]
        print(f"[smoke] action_dim total {u.action_manager.total_action_dim}; obs shapes "
              f"{ {k: tuple(v.shape) for k, v in obs.items()} }; physics_dt {u.physics_dt}, step_dt {u.step_dt}, "
              f"上肢 decimation {term.decimation}, 低通 alpha {term.alpha:.4f}")
        elbow = term.elbow_body_ids
        dm = robot.data.body_mass.torch[:, elbow] - term._elbow_nominal_mass
        print(f"[smoke] 负载：payload_mass 范围 [{term.payload_mass.min():.2f}, {term.payload_mass.max():.2f}] kg；"
              f"elbow 质量增量 − m/2 的最大差 {(dm - 0.5 * term.payload_mass.float().unsqueeze(-1)).abs().max():.2e}")
        print(f"[smoke] 上肢执行器 {type(robot.actuators['upper']).__name__}, 力矩上限 "
              f"{robot.actuators['upper'].effort_limit[0].tolist()}")
        act = torch.zeros((u.num_envs, u.action_manager.total_action_dim), device=u.device)
        sums, t0, nan_steps, unconv = {}, time.perf_counter(), 0, 0
        herr = []
        for i in range(args_cli.steps):
            obs, rew, term_, trunc, info = env.step(act)
            if not all(torch.isfinite(v).all() for v in obs.values()) or not torch.isfinite(rew).all():
                nan_steps += 1
            unconv += term.qp.unconverged
            for name in u.reward_manager.active_terms:
                idx = u.reward_manager.active_terms.index(name)
                sums[name] = sums.get(name, 0.0) + u.reward_manager._step_reward[:, idx].abs().mean().item()
            if i >= 50:
                herr.append(term.hand_err.norm(dim=-1).mean().item())
        dt_wall = time.perf_counter() - t0
        print(f"[smoke] {args_cli.steps} 步 × {u.num_envs} 环境 墙钟 {dt_wall:.1f} s（{args_cli.steps * u.num_envs / dt_wall:.0f} env-steps/s）；"
              f"含 NaN 的步数 {nan_steps}；积极集未收敛累计 {unconv}")
        print(f"[smoke] 手部误差（第 50 步后均值）{1e3 * sum(herr) / max(len(herr), 1):.1f} mm；终止 episode 比例 "
              f"{(u.episode_length_buf < args_cli.steps).float().mean():.2f}")
        if term.cfg.carrier_lead == "command":
            # 指令前馈超前：零动作下机器人很快摔倒重置，所以不等收敛，而是和一阶低通的解析响应比：
            # 指令恒定、yaw 近似不变时 |lead| = τ|v_cmd| · (1 − (1 − α)^n)，n = 重置后上肢 tick 数
            cmd = u.command_manager.get_command(term.cfg.command_name).to(term.dtype)
            c, s_ = torch.cos(term.carrier_yaw), torch.sin(term.carrier_yaw)
            goal = term.carrier_tau * torch.stack([c * cmd[:, 0] - s_ * cmd[:, 1], s_ * cmd[:, 0] + c * cmd[:, 1]], -1)
            n = (u.episode_length_buf.to(term.dtype) * u.cfg.decimation / term.decimation)
            expect = goal * (1 - (1 - term.alpha) ** n).unsqueeze(-1)
            ok = (u.episode_length_buf >= 20) & (goal.norm(dim=-1) > 0.05)
            res = (term.lead_pos[ok, :2] - expect[ok]).norm(dim=-1)
            print(f"[smoke] carrier_lead=command：{int(ok.sum())} 个环境，解析期望 |lead| 中位 "
                  f"{1e3 * expect[ok].norm(dim=-1).median():.1f} mm，|lead − 期望| 中位 {1e3 * res.median():.2f} mm、"
                  f"最大 {1e3 * res.max():.2f} mm（含 yaw 变化引起的偏差）")
        tot = sum(sums.values())
        print("[smoke] 各 reward 项 |每步加权值| 的平均（已乘 dt）：")
        for k, v in sorted(sums.items(), key=lambda kv: -kv[1]):
            print(f"    {k:28s} {v / args_cli.steps:.5f}  占比 {100 * v / tot:5.1f}%")
        env.close()


if __name__ == "__main__":
    main()
