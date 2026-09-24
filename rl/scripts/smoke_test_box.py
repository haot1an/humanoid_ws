"""H1-LocoManip-CarryBox-v3（DEC-007）冒烟测试：真实箱子放置、落定、掉箱判定、自碰撞稳定性、观测维度与吞吐。
用法（服务器）：cd /data/h1_locomanip/rl_next && python scripts/smoke_test_box.py --num_envs 256 --steps 100
零动作下机器人会在 1–3 s 内摔倒，所以落定检查只看前 settle_steps 步。
"""
import argparse
import pathlib
import sys
import time

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--settle_steps", type=int, default=25)
parser.add_argument("--policy", default=None, help="TorchScript 策略（actor 观测与 v2 相同）；不给则零动作")
parser.add_argument("--verbose_steps", type=int, default=0)
parser.add_argument("--task", default="H1-LocoManip-CarryBox-v3")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import h1_locomanip.tasks as tasks  # noqa: E402
from h1_locomanip.tasks import mdp  # noqa: E402

tasks.register()


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=env_cfg)
        obs, _ = env.reset()
        u = env.unwrapped
        robot, box, term = u.scene["robot"], u.scene["box"], u.action_manager.get_term("upper_body")
        print(f"[box] obs shapes { {k: tuple(v.shape) for k, v in obs.items()} }; 自碰撞 "
              f"{u.cfg.scene.robot.spawn.articulation_props.enabled_self_collisions}; "
              f"payload_in_elbow {term.cfg.payload_in_elbow}; carrier_lead {term.cfg.carrier_lead}")
        import omni.usd  # noqa: PLC0415
        from pxr import UsdPhysics  # noqa: PLC0415
        stage = omni.usd.get_context().get_stage()
        ground = [str(p.GetPath()) for p in stage.Traverse()
                  if str(p.GetPath()).startswith("/World/ground") and p.HasAPI(UsdPhysics.CollisionAPI)]
        fm = u.scene.sensors["torso_ground"].data.force_matrix_w.torch
        print(f"[box] /World/ground 下碰撞体 {ground}；torso_ground force_matrix 形状 {tuple(fm.shape)}")
        om = u.observation_manager
        for g in ("policy", "critic"):
            print(f"[box] 观测组 {g}: " + ", ".join(f"{n}{list(d)}" for n, d in zip(om.active_terms[g],
                                                                              om.group_obs_term_dim[g])))
        print(f"[box] 动作项 {u.action_manager.active_terms} 维度 {u.action_manager.action_term_dim}")
        m = box.data.body_mass.torch[:, 0]
        print(f"[box] 箱子质量 [{m.min():.2f}, {m.max():.2f}] kg；与 payload_mass 最大差 "
              f"{(m - term.payload_mass.float()).abs().max():.2e}；elbow_link 质量（应为模型原值）"
              f"{robot.data.body_mass.torch[0, term.elbow_body_ids].tolist()}")

        def rel_b():
            return mdp.box_pose_in_base(u)[:, :3]

        r0 = rel_b()
        print(f"[box] reset 后箱子相对骨盆位置均值 {r0.mean(0).tolist()}（期望 ≈ {mdp.BOX_OFFSET_B} + z 0.02）")
        act = torch.zeros((u.num_envs, u.action_manager.total_action_dim), device=u.device)
        policy = torch.jit.load(args_cli.policy, map_location=u.device).eval() if args_cli.policy else None
        alive = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
        dropped_any = torch.zeros_like(alive)
        t0, nan_steps, max_qd = time.perf_counter(), 0, 0.0
        counts = {n: 0 for n in u.termination_manager.active_terms}
        first_drop_t, drop_rel = [], []
        low_pelvis = 0
        for i in range(args_cli.steps):
            if policy is not None:
                with torch.no_grad():
                    act = policy(obs["policy"])
            obs, rew, term_, trunc, info = env.step(act)
            done = term_ | trunc
            if not all(torch.isfinite(v).all() for v in obs.values()) or not torch.isfinite(rew).all():
                nan_steps += 1
            max_qd = max(max_qd, robot.data.joint_vel.torch[alive].abs().max().item()) if alive.any() else max_qd
            cf_ = u.scene.sensors["contact_forces"]
            if i == 0:
                foot_ids = [cf_.body_names.index(f"{sd}_ankle_link") for sd in ("left", "right")]
                foot_contact = torch.zeros(2, device=u.device)
                foot_n = 0
            if i >= 100:
                fz = cf_.data.net_forces_w.torch[:, foot_ids].norm(dim=-1) > 1.0
                foot_contact += (fz & alive.unsqueeze(-1)).float().sum(0)
                foot_n += int(alive.sum())
            for n in counts:
                counts[n] += int(u.termination_manager.get_term(n).sum())
            low_pelvis += int((robot.data.root_link_pos_w.torch[:, 2] < 0.5).sum())
            newly = u.termination_manager.get_term("box_dropped") & alive & ~dropped_any
            if newly.any():
                first_drop_t += [(i + 1) * u.step_dt] * int(newly.sum())
            dropped_any |= u.termination_manager.get_term("box_dropped") & alive
            if i < args_cli.verbose_steps:
                gap = box.data.root_link_pos_w.torch[:, 2] - mdp._load_point_mid(u)[:, 2]
                print(f"[box] step {i + 1}: 终止 " + ", ".join(
                    f"{n}={int((u.termination_manager.get_term(n) & alive).sum())}"
                    for n in u.termination_manager.active_terms)
                    + f"；箱高差 中位 {1e3 * gap.median():.0f} mm；骨盆高 中位 "
                      f"{robot.data.root_link_pos_w.torch[:, 2].median():.3f}；箱相对骨盆 {rel_b().median(0).values.tolist()}")
            alive &= ~done
            if i + 1 == args_cli.settle_steps and alive.any():
                gap = box.data.root_link_pos_w.torch[:, 2] - mdp._load_point_mid(u)[:, 2]
                r = rel_b()
                a = alive
                print(f"[box] 第 {i + 1} 步（{(i + 1) * u.step_dt:.2f} s）存活 {int(a.sum())}/{u.num_envs}：箱子中心 − 负载点中点"
                      f" 高度 中位 {1e3 * gap[a].median():.1f} mm（胶囊假设 125 mm）；相对骨盆位移 |Δ| 中位 "
                      f"{1e3 * (r[a] - r0[a]).norm(dim=-1).median():.1f} mm、最大 {1e3 * (r[a] - r0[a]).norm(dim=-1).max():.1f} mm")
        cf = u.scene.sensors["contact_forces"]
        names = cf.body_names
        f = cf.data.net_forces_w.torch.norm(dim=-1)  # (N, B)
        touching = {n: float((f[:, j] > 1.0).float().mean()) for j, n in enumerate(names)
                    if any(k in n for k in ("knee", "hip", "pelvis", "torso", "ankle"))}
        z = robot.data.root_link_pos_w.torch[:, 2]
        print(f"[box] 末步骨盆高 分位 10/50/90% {z.quantile(0.1):.3f}/{z.median():.3f}/{z.quantile(0.9):.3f} m；"
              f"各连杆有接触（>1 N）的环境比例 { {k: round(v, 2) for k, v in touching.items()} }")
        if foot_n:
            print(f"[box] 首个 episode 存活段（第 100 步后）左 / 右脚着地占比 "
                  f"{(foot_contact / foot_n).tolist()}")
        dt_wall = time.perf_counter() - t0
        print(f"[box] {args_cli.steps} 步 × {u.num_envs} 环境 墙钟 {dt_wall:.1f} s（"
              f"{args_cli.steps * u.num_envs / dt_wall:.0f} env-steps/s）；NaN 步数 {nan_steps}；"
              f"存活段内最大关节速度 {max_qd:.1f} rad/s；首个 episode 内触发掉箱的环境 {int(dropped_any.sum())}")
        print(f"[box] 全程终止计数 {counts}；骨盆高 < 0.5 m 的环境步数 {low_pelvis}")
        if first_drop_t:
            t = torch.tensor(first_drop_t)
            print(f"[box] 首个 episode 掉箱时刻 [s]：中位 {t.median():.2f}、10% {t.quantile(0.1):.2f}、90% {t.quantile(0.9):.2f}")
        env.close()


if __name__ == "__main__":
    main()
