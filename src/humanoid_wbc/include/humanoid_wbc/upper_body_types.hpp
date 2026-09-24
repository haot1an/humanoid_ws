// 上肢 QP 的纯数据接口：不含任何 MuJoCo 类型（AGENTS §5.1 防火墙）
#pragma once
#include <Eigen/Dense>
#include <array>

namespace humanoid_wbc
{
    enum ArmSide
    {
        kLeftArm = 0,
        kRightArm = 1,
        kNumArms = 2
    };

    // 后端（MujocoUpperBodyKinematics）每个上肢控制周期填充一次；向量与 Jacobian 全部是 world 系
    // Jacobian 的列对应广义速度 v = [vel_world(3); omega_body(3); dq_joint(19)]，与 MuJoCo qvel 同约定
    struct UpperBodyKinematics
    {
        std::array<Eigen::Vector3d, kNumArms> hand_pos_world;     // 手部点（前臂末端碰撞球心）[m]
        std::array<Eigen::Vector3d, kNumArms> forearm_axis_world; // 前臂轴单位向量 a（肘 → 手）
        std::array<Eigen::MatrixXd, kNumArms> J_hand_pos;         // 3 x nv，手部点线速度 [m/s]
        std::array<Eigen::MatrixXd, kNumArms> J_forearm_rot;      // 3 x nv，elbow_link 角速度 [rad/s]
        std::array<Eigen::Vector3d, kNumArms> load_point_world;   // 负载作用点：前臂轴上距肘 load_distance 处 [m]
        std::array<Eigen::MatrixXd, kNumArms> J_load_point;       // 3 x nv，负载作用点线速度
        Eigen::VectorXd bias_force;                               // nv，g(q)：零速度偏置力 = 纯重力 [N·m / N]（与 Isaac 对齐，RL-LOCO-001）

        explicit UpperBodyKinematics(int nv)
            : bias_force(Eigen::VectorXd::Zero(nv))
        {
            for (int s = 0; s < kNumArms; ++s)
            {
                hand_pos_world[s].setConstant(std::numeric_limits<double>::quiet_NaN());
                forearm_axis_world[s].setConstant(std::numeric_limits<double>::quiet_NaN());
                J_hand_pos[s] = Eigen::MatrixXd::Zero(3, nv);
                J_forearm_rot[s] = Eigen::MatrixXd::Zero(3, nv);
                load_point_world[s].setConstant(std::numeric_limits<double>::quiet_NaN());
                J_load_point[s] = Eigen::MatrixXd::Zero(3, nv);
            }
        }
    };

    // 手部目标（world 系）。创建者 / 更新规则由调用方（reference generator）负责，控制器只读
    struct UpperBodyReference
    {
        std::array<Eigen::Vector3d, kNumArms> hand_pos_world;
        std::array<Eigen::Vector3d, kNumArms> hand_vel_world;
    };
}
