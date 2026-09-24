"""注册 H1 托箱行走任务。训练入口通过 --external_callback h1_locomanip.tasks.register 调用。"""
import sys

import gymnasium as gym

TASK_ID = "H1-LocoManip-Carry-v2"
BOX_TASK_ID = "H1-LocoManip-CarryBox-v3"   # DEC-007：真实箱子接触
RES_TASK_ID = "H1-LocoManip-CarryBoxRes-v4"  # DEC-008：混合方案（RL 输出上身参考修正）


def register():
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.carry_env_cfg:H1CarryEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.carry_env_cfg:H1CarryPPORunnerCfg",
            },
        )
    if BOX_TASK_ID not in gym.registry:
        gym.register(
            id=BOX_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.carry_env_cfg:H1CarryBoxEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.carry_env_cfg:H1CarryBoxPPORunnerCfg",
            },
        )
    if RES_TASK_ID not in gym.registry:
        gym.register(
            id=RES_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.carry_env_cfg:H1CarryBoxResEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.carry_env_cfg:H1CarryBoxResPPORunnerCfg",
            },
        )
    return sys.argv[1:]  # train.py 取与自身剩余参数的交集，全部返回表示本回调不消费任何参数
