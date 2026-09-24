// 顶层：DynamicsData + RobotState → JointCommand
#pragma once
#include <Eigen/Dense>
#include "humanoid_wbc/dynamics_data.hpp"

namespace humanoid_wbc
{
    class WbcController
    {
    public:
        WbcController(int nv, int nu, int nc);

        void buildProblem(
            const DynamicsData &dyn,
            const Eigen::Vector3d &com_pos_des,
            const Eigen::Matrix3d &torso_R_des,
            const Eigen::VectorXd &q_joint,
            const Eigen::VectorXd &dq_joint,
            const Eigen::VectorXd &q_posture_des,
        const Eigen::VectorXd &contact_ref_world);

        const Eigen::MatrixXd &P() const { return P_; }
        const Eigen::VectorXd &q() const { return q_; }

        const Eigen::MatrixXd &A() const { return A_; }
        const Eigen::VectorXd &l() const { return l_; }
        const Eigen::VectorXd &u() const { return u_; }

        void setTorqueLimits(
            const Eigen::VectorXd &tau_min,
            const Eigen::VectorXd &tau_max);

        Eigen::VectorXd computeTorque(
            const DynamicsData &dyn,
            const Eigen::VectorXd &qdd,
            const Eigen::VectorXd &fc) const;
        int numVariables() const { return nx_; }
        int numConstraints() const { return ncon_; }

    private:
        int nv_;   // 25
        int nu_;   // 19
        int nc_;   // 8
        int nf_;   // 24
        int nx_;   // 49
        int ncon_; // 89

        Eigen::MatrixXd P_;
        Eigen::VectorXd q_;

        Eigen::MatrixXd A_;
        Eigen::VectorXd l_;
        Eigen::VectorXd u_;

        Eigen::VectorXd tau_min_;
        Eigen::VectorXd tau_max_;
    };
}