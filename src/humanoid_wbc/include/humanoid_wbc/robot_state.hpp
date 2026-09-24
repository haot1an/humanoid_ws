#pragma once
#include <Eigen/Dense>        
#include <cassert>

struct RobotState {
    double time = 0.0;

    // ── 编码器直给 ─────────────────────────
    Eigen::VectorXd q_joint;         // 19
    Eigen::VectorXd dq_joint;        // 19

    // ── IMU 直给（yaw 会缓慢漂移）───────────
    Eigen::Quaterniond base_quat_wb {1,0,0,0};   // world ← body
    Eigen::Vector3d    base_omega_body {Eigen::Vector3d::Zero()};

    // ── GROUND TRUTH：真机上必须状态估计 ────
    Eigen::Vector3d base_pos_world {Eigen::Vector3d::Zero()};
    Eigen::Vector3d base_vel_world {Eigen::Vector3d::Zero()};

    explicit RobotState(int njoint)
        : q_joint(Eigen::VectorXd::Zero(njoint)),
          dq_joint(Eigen::VectorXd::Zero(njoint)) {}
};

// 拼成 MuJoCo 约定的 v ∈ R^25 = [vel_world(3); omega_body(3); dq_joint(19)]
inline void toGeneralizedVelocity(const RobotState& s, Eigen::Ref<Eigen::VectorXd> v_out);

inline void toGeneralizedVelocity(const RobotState& s, Eigen::Ref<Eigen::VectorXd> v_out)
{

    assert(v_out.size() == 6 + s.dq_joint.size());

    v_out.segment<3>(0) = s.base_vel_world;
    v_out.segment<3>(3) = s.base_omega_body;
    v_out.segment(6, s.dq_joint.size()) = s.dq_joint;

}


struct JointCommand {
    Eigen::VectorXd q_des;    // 19  目标关节角
    Eigen::VectorXd dq_des;   // 19  目标关节角速度
    Eigen::VectorXd kp;       // 19  比例增益（可在线改）
    Eigen::VectorXd kd;       // 19  微分增益（已减去模型内建 damping）
    Eigen::VectorXd tau_ff;   // 19  前馈力矩（WBC 的输出走这里）

    explicit JointCommand(int njoint)
        : q_des (Eigen::VectorXd::Zero(njoint)),
          dq_des(Eigen::VectorXd::Zero(njoint)),
          kp    (Eigen::VectorXd::Zero(njoint)),
          kd    (Eigen::VectorXd::Zero(njoint)),
          tau_ff(Eigen::VectorXd::Zero(njoint)) {}
};
