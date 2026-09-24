"""从 Isaac 训练环境导出部署所需的接口契约（关节顺序、默认角、增益、限幅、action 映射、观测布局）。
用法（服务器）：PYTHONPATH=/data/h1_locomanip/rl python scripts/export_contract.py --out ../contract/carry_v2.json
"""
import argparse
import json
import pathlib
import sys

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402

import h1_locomanip.tasks as tasks  # noqa: E402

tasks.register()


def main():
    env_cfg, _ = resolve_task_config(tasks.TASK_ID, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 2
        env = gym.make(tasks.TASK_ID, cfg=env_cfg)
        env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        leg_term = u.action_manager.get_term("joint_pos")
        upper = u.action_manager.get_term("upper_body")
        act = robot.actuators
        out = {
            "task": tasks.TASK_ID,
            "sim_dt": u.cfg.sim.dt, "decimation": u.cfg.decimation, "policy_dt": u.step_dt,
            "upper_dt": upper.qp.cfg.dt, "carrier_cutoff_hz": upper.cfg.carrier_cutoff_hz,
            "carrier_lead": upper.cfg.carrier_lead,
            "joint_names_isaac": list(robot.joint_names),
            "leg_action_joint_names": list(leg_term._joint_names),
            "leg_action_scale": float(leg_term.cfg.scale), "leg_action_use_default_offset": True,
            "upper_joint_names": list(upper.adapter.upper_joint_idx and
                                      [robot.joint_names[i] for i in upper.adapter.upper_joint_idx]),
            "default_joint_pos": robot.data.default_joint_pos.torch[0].tolist(),
            "hand_offset_b": upper.hand_offset_b.tolist(),
            "qp_cfg": {k: getattr(upper.qp.cfg, k) for k in vars(upper.qp.cfg)},
            "obs_terms_policy": [(n, list(u.observation_manager.group_obs_term_dim["policy"][i]))
                                 for i, n in enumerate(u.observation_manager.active_terms["policy"])],
            # 注意：显式执行器（IdealPD）在 PhysX 侧驱动增益为 0，data.joint_stiffness 读不到真实增益，
            # 因此按执行器分组导出；部署端据此组装每个关节的有效 kp / kd
            "data_joint_stiffness": robot.data.joint_stiffness.torch[0].tolist(),
            "data_joint_damping": robot.data.joint_damping.torch[0].tolist(),
            "group_stiffness": {g: act[g].stiffness[0].tolist() for g in act},
            "group_damping": {g: act[g].damping[0].tolist() for g in act},
            "effort_limit": {g: (act[g].effort_limit[0].tolist() if hasattr(act[g].effort_limit, "tolist")
                                 else act[g].effort_limit) for g in act},
            "actuator_joint_names": {g: [robot.joint_names[i] for i in act[g].joint_indices] for g in act},
            "actuator_types": {g: type(act[g]).__name__ for g in act},
        }
        pathlib.Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args_cli.out, "w"), indent=1)
        print("[contract]", args_cli.out)
        print(json.dumps({k: out[k] for k in ["leg_action_joint_names", "upper_joint_names", "obs_terms_policy",
                                              "policy_dt", "upper_dt"]}, indent=1, ensure_ascii=False))
        env.close()


if __name__ == "__main__":
    main()
