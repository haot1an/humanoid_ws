"""H1 托箱行走环境（RL-LOCO-001 v2）：在 Isaac Lab 默认 H1 flat 基础上的改动逐项对应设计表。"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg  # noqa: F401  (类型参考)
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1FlatPPORunnerCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg

from . import h1_symmetry, mdp
from .upper_body_action import UpperBodyQpActionCfg

# carry15 端箱姿态（DEC-003，IK 见 scripts/view_carry_poses.py），右臂 roll / yaw 取反
CARRY15 = {
    "torso": 0.0,
    "left_shoulder_pitch": -0.235528, "left_shoulder_roll": -0.0932, "left_shoulder_yaw": 0.120633,
    "left_elbow": 0.149264,
    "right_shoulder_pitch": -0.235528, "right_shoulder_roll": 0.0932, "right_shoulder_yaw": -0.120633,
    "right_elbow": 0.149264,
}
LEG_JOINTS = [".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", ".*_ankle"]


@configclass
class CriticCfg(ObsGroup):
    """非对称 AC：actor 观测（无噪声）+ 负载质量(2) + 上肢目标角(9)。"""
    base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
    velocity_commands = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
    actions = ObsTerm(func=base_mdp.last_action)
    payload_mass = ObsTerm(func=mdp.payload_mass_per_arm)
    upper_targets = ObsTerm(func=mdp.upper_joint_targets)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class H1CarryEnvCfg(H1FlatEnvCfg):
    # 注意：不要在 __post_init__ 里按本类字段做分支——Hydra 命令行覆盖在 __post_init__ 之后才写入，
    # 分支不会被重新执行（EXP-016 的课程开关因此失效）。可覆盖的开关要落在最终的 term 参数上。
    residual_weight: float = -1.0  # 上肢任务残差惩罚权重；命令行改用 env.rewards.upper_task_residual.weight=...

    def __post_init__(self):
        super().__post_init__()
        robot = self.scene.robot
        # 初始 / 默认关节角：腿沿用默认，上肢为 carry15（joint_pos_rel 以此为零点）
        robot.init_state.joint_pos.update(CARRY15)
        robot.init_state.joint_pos = {k: v for k, v in robot.init_state.joint_pos.items()
                                      if k not in (".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw",
                                                   ".*_elbow")}
        # 执行器：腿 / 脚沿用默认隐式 PD；torso + 双臂改显式 IdealPD（τ = kp e + kd ė + τ_ff 后整体限幅，与 C++ bridge 一致），
        # 力矩上限取 MJCF（DEC-003“4.批准”）
        robot.actuators = {
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee"],
                # DEC-005：统一到 MJCF ctrlrange（髋 ±200、膝 ±300、踝 ±40）。Isaac 默认为髋/膝 300、踝 100，
                # EXP-014 单变量实验证明踝 40 vs 100 是 1.0 m/s 带负载摔倒的直接原因。
                # 复现 EXP-011 旧限幅：env.scene.robot.actuators.legs.effort_limit_sim=300
                #                      env.scene.robot.actuators.feet.effort_limit_sim=100
                effort_limit_sim={".*_hip_yaw": 200.0, ".*_hip_roll": 200.0, ".*_hip_pitch": 200.0,
                                  ".*_knee": 300.0},
                stiffness={".*_hip_yaw": 150.0, ".*_hip_roll": 150.0, ".*_hip_pitch": 200.0, ".*_knee": 200.0},
                damping={".*_hip_yaw": 5.0, ".*_hip_roll": 5.0, ".*_hip_pitch": 5.0, ".*_knee": 5.0},
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle"],
                effort_limit_sim=40.0,
                stiffness={".*_ankle": 20.0}, damping={".*_ankle": 4.0},
            ),
            "upper": IdealPDActuatorCfg(
                joint_names_expr=["torso", ".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
                stiffness={"torso": 200.0, ".*_shoulder_.*": 40.0, ".*_elbow": 40.0},
                damping={"torso": 5.0, ".*_shoulder_.*": 10.0, ".*_elbow": 10.0},
                effort_limit={"torso": 200.0, ".*_shoulder_pitch": 40.0, ".*_shoulder_roll": 40.0,
                              ".*_shoulder_yaw": 18.0, ".*_elbow": 18.0},
                effort_limit_sim={"torso": 200.0, ".*_shoulder_pitch": 40.0, ".*_shoulder_roll": 40.0,
                                  ".*_shoulder_yaw": 18.0, ".*_elbow": 18.0},
            ),
        }
        # 1. Action：只输出 10 个腿部关节；上肢由 QP action term 驱动
        self.actions.joint_pos.joint_names = LEG_JOINTS
        self.actions.upper_body = UpperBodyQpActionCfg(asset_name="robot")
        # 3. Critic 特权观测
        self.observations.critic = CriticCfg()
        # 4. Reward：删除策略不控制的关节偏差项；加上肢任务残差惩罚
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_torso = None
        self.rewards.upper_task_residual = RewTerm(func=mdp.upper_task_residual_l2, weight=self.residual_weight)
        # 6. 负载随机化
        self.events.payload = EventTerm(func=mdp.reset_payload, mode="reset",
                                        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_elbow_link"])})
        # 7. 课程：负载上限（用户批准的规则；Agent 代写实现，用户须能解释机制）。默认关闭（固定上限），
        #    启用：env.curriculum.payload.params.enabled=true env.actions.upper_body.payload_cap_init=1.0
        self.curriculum.payload = CurrTerm(func=mdp.payload_curriculum, params={"enabled": False})


@configclass
class H1CarryBoxEnvCfg(H1CarryEnvCfg):
    """RL-LOCO-001 v3（DEC-007）：箱子为真实刚体，靠前臂摩擦托住；打开机器人自碰撞；掉箱终止。
    其余与 v2 相同（EXP-018 配置：载体系指令超前在本类默认打开；课程仍需命令行启用）。"""

    def __post_init__(self):
        super().__post_init__()
        # 4. 机器人自碰撞（用户 2026-09-23 改为打开）
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        # 碰撞体：H1 flat 默认的 h1_minimal.usd 只有两踝和躯干有碰撞体，箱子会穿过手臂（冒烟测试实测）。
        # 用户 2026-09-23 选择改用完整 h1.usd（所有连杆为网格凸包碰撞体）
        self.scene.robot.spawn.usd_path = self.scene.robot.spawn.usd_path.replace("h1_minimal.usd", "h1.usd")
        assert self.scene.robot.spawn.usd_path.endswith("/H1/h1.usd"), self.scene.robot.spawn.usd_path
        # 1. 箱子：独立刚体；材料 friction combine = min，配合下方把前臂 / 肩 / 躯干材料设为 μ = 1.0，
        #    有效摩擦 = min(μ_box, 1.0) = μ_box（PhysX combine 优先级 average < min < multiply < max）
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=mdp.BOX_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
                    static_friction=0.75, dynamic_friction=0.75, friction_combine_mode="min"),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.6, 0.3)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.24, 0.0, 1.31)),
        )
        # 9. 负载不再写进 elbow_link；QP 前馈仍按已知总质量左右各半
        self.actions.upper_body.payload_in_elbow = False
        self.actions.upper_body.carrier_lead = "command"
        # 2 / 5. 箱子质量与初始放置（替换 v2 的 elbow_link 加质量）；必须在 reset_base / reset_robot_joints 之后
        self.events.payload = EventTerm(func=mdp.reset_box, mode="reset", params={"empty_mass": 0.3})
        # 3. 摩擦：箱子 μ ~ U(0.5, 1.0)（静 / 动分别采样，make_consistent 保证动 ≤ 静），每 episode 重新分配
        self.events.box_friction = EventTerm(
            func=base_mdp.randomize_rigid_body_material, mode="reset",
            params={"asset_cfg": SceneEntityCfg("box"), "static_friction_range": (0.5, 1.0),
                    "dynamic_friction_range": (0.5, 1.0), "restitution_range": (0.0, 0.0), "num_buckets": 64,
                    "make_consistent": True})
        self.events.upper_link_friction = EventTerm(
            func=base_mdp.randomize_rigid_body_material, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", ".*_shoulder_.*_link",
                                                                     ".*_elbow_link"]),
                    "static_friction_range": (1.0, 1.0), "dynamic_friction_range": (1.0, 1.0),
                    "restitution_range": (0.0, 0.0), "num_buckets": 1})
        # 躯干触地终止改为只统计躯干-地面接触力（语义不变，见 mdp.torso_ground_contact）
        self.scene.torso_ground = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/torso_link", history_length=3,
                                                   filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"])
        self.terminations.base_contact = DoneTerm(func=mdp.torso_ground_contact, params={"threshold": 1.0})
        # 6. 掉箱终止
        self.terminations.box_dropped = DoneTerm(func=mdp.box_dropped, params={"drop_height": 0.15,
                                                                               "max_horizontal": 0.6})
        # 8. critic 特权：箱子相对骨盆位姿（质量已由 payload_mass 提供）
        self.observations.critic.box_pose = ObsTerm(func=mdp.box_pose_in_base)


@configclass
class H1CarryBoxResEnvCfg(H1CarryBoxEnvCfg):
    """RL-LOCO-001 v4（DEC-008）：混合方案。策略输出 10 维腿 + 2 维上身参考修正（QP 执行），actor 可见箱子位姿，
    箱子滑动惩罚，骨盆过低终止，关闭机器人自碰撞（箱子与机器人照常碰撞）。"""
    box_slip_weight: float = -1.0  # 冒烟测试后按约占 5% 调整；命令行 env.rewards.box_slip.weight=...

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False       # 6
        self.actions.upper_body.ref_delta = True                                          # 1 / 2
        self.observations.policy.box_pose = ObsTerm(func=mdp.box_pose_in_base)           # 3
        self.rewards.box_slip = RewTerm(func=mdp.box_slip_l2, weight=self.box_slip_weight)  # 4
        self.terminations.pelvis_low = DoneTerm(func=base_mdp.root_height_below_minimum,  # 5
                                                params={"minimum_height": 0.8})


FEET = ["left_ankle_link", "right_ankle_link"]


@configclass
class H1CarryBoxGaitEnvCfg(H1CarryBoxResEnvCfg):
    """RL-LOCO-001 v5（DEC-009）：v4 + 周期步态（时钟观测、着地时序奖励、摆动脚高度跟踪），feet_air_time 权重 0。
    左右镜像增强在 runner 配置里（H1CarryBoxGaitPPORunnerCfg）。"""
    gait_period: float = 0.8

    def __post_init__(self):
        super().__post_init__()
        T = self.gait_period
        self.observations.policy.gait_clock = ObsTerm(func=mdp.gait_clock, params={"period": T})       # 2
        self.observations.critic.gait_clock = ObsTerm(func=mdp.gait_clock, params={"period": T})
        self.rewards.gait_contact = RewTerm(                                                              # 3
            func=mdp.gait_contact_match, weight=1.0,
            params={"period": T, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)})
        self.rewards.swing_height = RewTerm(                                                              # 4
            func=mdp.swing_foot_height, weight=1.0,
            params={"period": T, "target": 0.08, "std": 0.02,
                    "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True)})
        self.rewards.feet_air_time.weight = 0.0                                                           # 5


@configclass
class H1CarryPPORunnerCfg(H1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000
        self.experiment_name = "h1_locomanip_v2"
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}


@configclass
class H1CarryBoxPPORunnerCfg(H1CarryPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "h1_locomanip_v3"


@configclass
class H1CarryBoxResPPORunnerCfg(H1CarryPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "h1_locomanip_v4"


@configclass
class H1CarryBoxGaitPPORunnerCfg(H1CarryPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "h1_locomanip_v5"
        # DEC-009 第 6 项：左右镜像数据增强（原始 + 镜像两份参与 PPO 更新）
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=h1_symmetry.compute_symmetric_states)
