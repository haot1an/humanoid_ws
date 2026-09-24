#include "humanoid_wbc/wbc_controller.hpp"
#include <limits>
#include <cassert>
#include <stdexcept>
namespace humanoid_wbc
{

    //     nc_   // contact point number = 8
    // nf_   // contact force dimension = 3 * nc = 24
    // nx_   // decision variable dimension = nv + nf = 49
    WbcController::WbcController(int nv, int nu, int nc)
        : nv_(nv),
          nu_(nu),
          nc_(nc),
          nf_(3 * nc),
          nx_(nv + 3 * nc),
          ncon_(3 * nc + (nv - nu) + nu + 5 * nc),
          P_(Eigen::MatrixXd::Zero(nx_, nx_)),
          q_(Eigen::VectorXd::Zero(nx_)),
          A_(Eigen::MatrixXd::Zero(ncon_, nx_)),
          l_(Eigen::VectorXd::Zero(ncon_)),
          u_(Eigen::VectorXd::Zero(ncon_)),
          tau_min_(Eigen::VectorXd::Zero(nu_)),
          tau_max_(Eigen::VectorXd::Zero(nu_))
    {
    }
    void WbcController::setTorqueLimits(
        const Eigen::VectorXd &tau_min,
        const Eigen::VectorXd &tau_max)
    {
        assert(tau_min.size() == nu_);
        assert(tau_max.size() == nu_);

        tau_min_ = tau_min;
        tau_max_ = tau_max;
    }

    Eigen::VectorXd WbcController::computeTorque(
        const DynamicsData &dyn,
        const Eigen::VectorXd &qdd,
        const Eigen::VectorXd &fc) const
    {
        Eigen::VectorXd tau =
            dyn.M.bottomRows(nu_) * qdd + dyn.h.tail(nu_) - dyn.J_c.transpose().bottomRows(nu_) * fc - dyn.tau_passive.tail(nu_);

        return tau;
    }

    void WbcController::buildProblem(
        const DynamicsData &dyn,
        const Eigen::Vector3d &com_pos_des,
        const Eigen::Matrix3d &torso_R_des,
        const Eigen::VectorXd &q_joint,
        const Eigen::VectorXd &dq_joint,
        const Eigen::VectorXd &q_posture_des,
        const Eigen::VectorXd &contact_ref_world)
    {
        // 构建二次规划问题的矩阵和向量
        // 这里可以根据具体的控制需求来设置 P, q, A, l, u
        // 例如，P 可以是关节位置和速度的权重矩阵
        // q 可以是期望的关节位置和速度
        // A 可以是约束矩阵，例如接触约束
        // l 和 u 可以是约束的下界和上界

        // std::cout
        //     << "nv_=" << nv_
        //     << " nu_=" << nu_
        //     << " nc_=" << nc_
        //     << " nf_=" << nf_
        //     << " nx_=" << nx_
        //     << " ncon_=" << ncon_
        //     << "\nP=" << P_.rows() << "x" << P_.cols()
        //     << "\nA=" << A_.rows() << "x" << A_.cols()
        //     << "\nJcom=" << dyn.J_com.rows() << "x" << dyn.J_com.cols()
        //     << "\nJc=" << dyn.J_c.rows() << "x" << dyn.J_c.cols()
        //     << "\nM=" << dyn.M.rows() << "x" << dyn.M.cols()
        //     << std::endl;

        // 示例：设置 P 为单位矩阵，q 为零向量
        P_.setZero();
        q_.setZero();

        // 示例：设置 A 为接触约束矩阵，l 和 u 为接触力的上下界
        A_.setZero();
        l_.setZero();
        u_.setZero();

        assert(q_joint.size() == nu_);
        assert(dq_joint.size() == nu_);
        assert(q_posture_des.size() == nu_);
        if (contact_ref_world.size() != nf_ || !contact_ref_world.allFinite() ||
            dyn.contact_pos_world.size() != nf_ || dyn.contact_vel_world.size() != nf_)
            throw std::runtime_error("WBC contact reference/state dimensions or values invalid");

        double kp_posture = 36.0;
        double kd_posture = 12.0;

        Eigen::VectorXd qdd_joint_des =
            kp_posture * (q_posture_des - q_joint) - kd_posture * dq_joint;

        const double qdd_posture_max = 20.0;

        qdd_joint_des =
            qdd_joint_des.cwiseMax(-qdd_posture_max)
                .cwiseMin(qdd_posture_max);

        int nbase = nv_ - nu_;

        double w_posture = 10.0;

        P_.block(nbase, nbase, nu_, nu_)
            .diagonal()
            .array() += 2.0 * w_posture;

        q_.segment(nbase, nu_) +=
            -2.0 * w_posture * qdd_joint_des;

        // 这里可以根据 dyn 中的信息来填充 A, l, u

        double kp_com = 100.0; // COM position gain
        double kd_com = 20.0;  // COM velocity gain
        Eigen::Vector3d a_com_des =
            kp_com * (com_pos_des - dyn.com_pos) - kd_com * dyn.com_vel;

        Eigen::Vector3d b_com =
            a_com_des - dyn.Jdot_com_v;

        double w_com = 100.0;

        Eigen::Matrix3d R_err_mat =
            0.5 * (torso_R_des * dyn.torso_R.transpose() - dyn.torso_R * torso_R_des.transpose());

        Eigen::Vector3d e_R;
        e_R << R_err_mat(2, 1),
            R_err_mat(0, 2),
            R_err_mat(1, 0);

        double kp_torso = 16.0;
        double kd_torso = 8.0;

        Eigen::Vector3d alpha_torso_des =
            kp_torso * e_R - kd_torso * dyn.torso_omega;

        const double alpha_torso_max = 5.0;

        alpha_torso_des =
            alpha_torso_des
                .cwiseMax(-alpha_torso_max)
                .cwiseMin(alpha_torso_max);

        Eigen::Vector3d b_torso =
            alpha_torso_des - dyn.Jdot_torso_rot_v;

        double w_torso = 10.0;

        P_.topLeftCorner(nv_, nv_) +=
            2.0 * w_torso *
            dyn.J_torso_rot.transpose() *
            dyn.J_torso_rot;

        q_.head(nv_) +=
            -2.0 * w_torso *
            dyn.J_torso_rot.transpose() *
            b_torso;

        // Pqq​+=2wcom​JcomT​Jcom​

        P_.topLeftCorner(nv_, nv_) += 2 * w_com * dyn.J_com.transpose() * dyn.J_com;
        q_.head(nv_) += -2 * w_com * dyn.J_com.transpose() * b_com;
        double w_qdd = 0.1;
        P_.topLeftCorner(nv_, nv_).diagonal().array() += 2 * w_qdd;
        double w_f = 1e-4;
        P_.bottomRightCorner(nf_, nf_).diagonal().array() += 2 * w_f;

        A_.block(0, 0, nf_, nv_) = dyn.J_c;

        const Eigen::VectorXd contact_acc_des =
            (-4.0 * (dyn.contact_pos_world - contact_ref_world) - 4.0 * dyn.contact_vel_world)
                .cwiseMax(-0.2).cwiseMin(0.2);
        // Multiple points on one rigid foot have dependent rows. Project the
        // requested RHS onto achievable rigid-body accelerations; otherwise
        // finite rotation makes independent point PD equations inconsistent.
        const Eigen::VectorXd contact_rhs = contact_acc_des - dyn.Jdot_c_v;
        l_.segment(0, nf_) = dyn.J_c * dyn.J_c.completeOrthogonalDecomposition().solve(contact_rhs);
        u_.segment(0, nf_) = l_.segment(0, nf_);

        int row_dyn = nf_;

        A_.block(row_dyn, 0, nbase, nv_) =
            dyn.M.topRows(nbase);

        A_.block(row_dyn, nv_, nbase, nf_) =
            -dyn.J_c.transpose().topRows(nbase);

        l_.segment(row_dyn, nbase) =
            -dyn.h.head(nbase) + dyn.tau_passive.head(nbase);

        u_.segment(row_dyn, nbase) =
            -dyn.h.head(nbase) + dyn.tau_passive.head(nbase);

        int row_tau = row_dyn + nbase;

        A_.block(row_tau, 0, nu_, nv_) = dyn.M.bottomRows(nu_);
        A_.block(row_tau, nv_, nu_, nf_) = -dyn.J_c.transpose().bottomRows(nu_);

        l_.segment(row_tau, nu_) =
            tau_min_ - dyn.h.tail(nu_) + dyn.tau_passive.tail(nu_);

        u_.segment(row_tau, nu_) =
            tau_max_ - dyn.h.tail(nu_) + dyn.tau_passive.tail(nu_);

        int row_friction = row_tau + nu_;
        double mu = 0.5;
        double inf = std::numeric_limits<double>::infinity();

        for (int i = 0; i < nc_; ++i)
        {
            int row = row_friction + 5 * i;
            int col = nv_ + 3 * i;

            // 这里填 5 条摩擦约束
            A_(row + 0, col + 0) = 1.0;
            A_(row + 0, col + 2) = -mu;

            l_(row + 0) = -inf;
            u_(row + 0) = 0.0;

            A_(row + 1, col + 0) = -1.0;
            A_(row + 1, col + 2) = -mu;

            l_(row + 1) = -inf;
            u_(row + 1) = 0.0;
            A_(row + 2, col + 1) = 1.0;
            A_(row + 2, col + 2) = -mu;

            l_(row + 2) = -inf;
            u_(row + 2) = 0.0;

            A_(row + 3, col + 1) = -1.0;
            A_(row + 3, col + 2) = -mu;

            l_(row + 3) = -inf;
            u_(row + 3) = 0.0;

            A_(row + 4, col + 2) = 1.0;
            l_(row + 4) = 0.0;
            u_(row + 4) = inf;
        }

        assert(P_.rows() == nx_);
        assert(P_.cols() == nx_);

        assert(q_.size() == nx_);

        assert(A_.rows() == ncon_);
        assert(A_.cols() == nx_);

        assert(l_.size() == ncon_);
        assert(u_.size() == ncon_);
        assert(P_.allFinite());
        assert(q_.allFinite());
        assert(A_.allFinite());
        // double P_sym_err =
        //     (P_ - P_.transpose()).norm();
        // std::cout << "P symmetry error: " << P_sym_err << std::endl;
    }
}