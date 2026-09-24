// 纯数据：M, h, tau_passive, J_c, Jdot_c_v, J_com, ...
#pragma once
#include <Eigen/Dense>

namespace humanoid_wbc
{
    struct DynamicsData
    {
        Eigen::MatrixXd M;
        Eigen::VectorXd h;
        Eigen::VectorXd tau_passive;

        Eigen::MatrixXd J_c;
        Eigen::VectorXd Jdot_c_v;
        Eigen::VectorXd contact_pos_world;
        Eigen::VectorXd contact_vel_world;

        Eigen::MatrixXd J_com;
        Eigen::VectorXd Jdot_com_v;

        Eigen::Vector3d com_pos;
        Eigen::Vector3d com_vel;

        Eigen::MatrixXd J_torso_rot;      // 3 x nv，torso 角速度 Jacobian
        Eigen::Vector3d Jdot_torso_rot_v; // Jdot_rot * qdot

        Eigen::Matrix3d torso_R;     // torso 当前旋转矩阵
        Eigen::Vector3d torso_omega; // torso 当前角速度

        explicit DynamicsData(int nv, int nc)
            : M(Eigen::MatrixXd::Zero(nv, nv)),
              h(Eigen::VectorXd::Zero(nv)),
              tau_passive(Eigen::VectorXd::Zero(nv)),
              J_c(Eigen::MatrixXd::Zero(nc, nv)),
              Jdot_c_v(Eigen::VectorXd::Zero(nc)),
              contact_pos_world(Eigen::VectorXd::Zero(nc)),
              contact_vel_world(Eigen::VectorXd::Zero(nc)),
              J_com(Eigen::MatrixXd::Zero(3, nv)),
              Jdot_com_v(Eigen::VectorXd::Zero(3)),
              com_pos(Eigen::Vector3d::Zero()),
              com_vel(Eigen::Vector3d::Zero()),
              J_torso_rot(Eigen::MatrixXd::Zero(3, nv)),
              Jdot_torso_rot_v(Eigen::Vector3d::Zero()),
              torso_R(Eigen::Matrix3d::Identity()),
              torso_omega(Eigen::Vector3d::Zero())
        {
        }
    };
}
