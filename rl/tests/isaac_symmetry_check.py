"""DEC-009 验证（服务器 Isaac 中运行）：
1. 镜像映射逐项打印（关节对调 / 取负是否符合预期）；
2. 镜像两次 = 恒等（policy / critic 观测与动作）；
3. 默认站姿的镜像一致性：重置到默认关节角、零速度后，关节项镜像应与自身相等（姿态本身左右对称）；
4. 站立时脚 link 高度（核对 FOOT_STAND_HEIGHT）与两个步态奖励的取值范围。
用法：cd /data/h1_locomanip/rl_next && python tests/isaac_symmetry_check.py --num_envs 16
"""
import argparse
import pathlib
import sys

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import h1_locomanip.tasks as tasks  # noqa: E402
from h1_locomanip.tasks import h1_symmetry, mdp  # noqa: E402

tasks.register()


def main():
    env_cfg, _ = resolve_task_config(tasks.GAIT_TASK_ID, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(tasks.GAIT_TASK_ID, cfg=env_cfg)
        obs, _ = env.reset()
        u = env.unwrapped
        maps, act_map = h1_symmetry.build_maps(env)
        joints = list(u.scene["robot"].joint_names)
        jp, js = h1_symmetry._joint_perm_sign(joints)
        print("[sym] 关节镜像：" + ", ".join(f"{n}→{'-' if s < 0 else ''}{joints[p]}" for n, p, s in zip(joints, jp, js)))
        legs = list(u.action_manager.get_term("joint_pos")._joint_names)
        print(f"[sym] 动作镜像 perm {act_map[0]} sign {act_map[1]}（腿 {legs} + Δ 2）")
        ok = True
        for g in obs.keys():
            x = obs[g]
            twice = h1_symmetry._apply(h1_symmetry._apply(x, maps[g]), maps[g])
            err = (twice - x).abs().max().item()
            ok &= err == 0.0
            print(f"[sym] {g}: 维度 {x.shape[1]}，镜像两次误差 {err:.1e}")
        a = torch.randn(u.num_envs, u.action_manager.total_action_dim, device=u.device)
        err = (h1_symmetry._apply(h1_symmetry._apply(a, act_map), act_map) - a).abs().max().item()
        ok &= err == 0.0
        print(f"[sym] 动作镜像两次误差 {err:.1e}")
        from tensordict import TensorDict  # noqa: PLC0415  rsl-rl 传入的是 TensorDict
        oa, aa = h1_symmetry.compute_symmetric_states(env, TensorDict(dict(obs), batch_size=[u.num_envs]), a)
        print(f"[sym] 增强后 batch：obs {oa.batch_size[0]}（原 {u.num_envs}），actions {tuple(aa.shape)}")
        # 默认站姿：关节位置项（joint_pos_rel）在默认角时为 0，镜像必为 0；取策略观测中 joint_pos 段检查镜像后与原值的差
        om = u.observation_manager
        off = 0
        for name, dim in zip(om.active_terms["policy"], om.group_obs_term_dim["policy"]):
            if name == "joint_pos":
                seg = slice(off, off + dim[0])
            off += dim[0]
        robot = u.scene["robot"]
        q = robot.data.joint_pos.torch[:1] - robot.data.default_joint_pos.torch[:1]
        mq = h1_symmetry._apply(q, (jp, js))
        print(f"[sym] 默认角左右对称性：mirror(q_default) 与 q_default 的最大差 "
              f"{(h1_symmetry._apply(robot.data.default_joint_pos.torch[:1], (jp, js)) - robot.data.default_joint_pos.torch[:1]).abs().max():.2e}"
              f"（应为 0；关节段 {seg}）；当前 |q − q_default| 最大 {q.abs().max():.2e}，镜像后 {mq.abs().max():.2e}")
        # 步态奖励与脚高
        act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
        feet = [robot.body_names.index(f) for f in ("left_ankle_link", "right_ankle_link")]
        z0 = robot.data.body_link_pos_w.torch[:, feet, 2]
        print(f"[sym] reset 时脚 link 高度 {z0.mean(0).tolist()}（FOOT_STAND_HEIGHT = {mdp.FOOT_STAND_HEIGHT}）")
        rc, rh = [], []
        for _ in range(40):
            obs, *_ = env.step(act)
            i_c = u.reward_manager.active_terms.index("gait_contact")
            i_h = u.reward_manager.active_terms.index("swing_height")
            rc.append(u.reward_manager._step_reward[:, i_c].mean().item())
            rh.append(u.reward_manager._step_reward[:, i_h].mean().item())
        print(f"[sym] 零动作 40 步：gait_contact 每步加权值 {min(rc):.4f}–{max(rc):.4f}，swing_height {min(rh):.4f}–{max(rh):.4f}"
              f"（已乘 dt）；时钟 {obs['policy'][0, -2:].tolist()}")
        print(f"[sym] 结果：{'PASS' if ok else 'FAIL'}")
        env.close()


if __name__ == "__main__":
    main()
