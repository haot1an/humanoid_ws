#include "humanoid_wbc/upper_body_qp.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace humanoid_wbc
{
    namespace
    {
        constexpr int kRowsPerArm = 4; // 手部位置 3 + 前臂俯仰 1
    }

    UpperBodyQp::UpperBodyQp(int nv, std::vector<int> upper_dof,
                             Eigen::VectorXd q_min, Eigen::VectorXd q_max,
                             const UpperBodyQpConfig &cfg)
        : nv_(nv),
          nu_(static_cast<int>(upper_dof.size())),
          upper_dof_(std::move(upper_dof)),
          q_min_(std::move(q_min)),
          q_max_(std::move(q_max)),
          cfg_(cfg),
          v_rest_(Eigen::VectorXd::Zero(nv)),
          A_task_(Eigen::MatrixXd::Zero(kNumArms * kRowsPerArm, nu_)),
          b_task_(Eigen::VectorXd::Zero(kNumArms * kRowsPerArm)),
          w_task_(Eigen::VectorXd::Zero(kNumArms * kRowsPerArm)),
          task_residual_(Eigen::VectorXd::Zero(kNumArms * kRowsPerArm)),
          q_des_(Eigen::VectorXd::Constant(nu_, std::numeric_limits<double>::quiet_NaN())),
          dq_des_(Eigen::VectorXd::Zero(nu_)),
          solver_(nu_, nu_)
    {
        const bool sizes_ok = nu_ > 0 && q_min_.size() == nu_ && q_max_.size() == nu_;
        if (!sizes_ok)
            throw std::runtime_error("UpperBodyQp: upper_dof / q_min / q_max 尺寸不一致");
        for (int k : upper_dof_)
        {
            const bool in_range = k >= 6 && k < nv_; // 上肢不能是浮动基的列
            if (!in_range)
                throw std::runtime_error("UpperBodyQp: upper_dof 越界 " + std::to_string(k));
        }
        const bool limits_ok = (q_min_.array() < q_max_.array()).all();
        if (!limits_ok)
            throw std::runtime_error("UpperBodyQp: q_min 必须小于 q_max");
        const bool cfg_ok = cfg_.dt > 0 && cfg_.dq_max > 0 && cfg_.delta_max > 0 && cfg_.w_reg > 0;
        if (!cfg_ok)
            throw std::runtime_error("UpperBodyQp: dt / dq_max / delta_max / w_reg 必须为正");

        for (int s = 0; s < kNumArms; ++s)
        {
            w_task_.segment<3>(s * kRowsPerArm).setConstant(cfg_.w_hand);
            w_task_(s * kRowsPerArm + 3) = cfg_.w_pitch;
        }
    }

    void UpperBodyQp::reset(const Eigen::VectorXd &q_upper_measured)
    {
        if (q_upper_measured.size() != nu_)
            throw std::runtime_error("UpperBodyQp::reset: 尺寸错误");
        q_des_ = q_upper_measured;
        dq_des_.setZero();
    }

    Eigen::MatrixXd UpperBodyQp::upperColumns(const Eigen::MatrixXd &J) const
    {
        Eigen::MatrixXd out(J.rows(), nu_);
        for (int i = 0; i < nu_; ++i)
            out.col(i) = J.col(upper_dof_[i]);
        return out;
    }

    // 手部位置任务（示例实现）：
    //   J_hand v = J_u dq_u + J_hand v_rest          （v_rest = v 的上肢列置零）
    //   期望   J_u dq_u = v_des − J_hand v_rest       （前馈：扣掉基座运动带来的手速度）
    //   v_des = v_ref + k_hand (p_ref − p)            （反馈）
    void UpperBodyQp::buildHandRows(const UpperBodyKinematics &kin, const UpperBodyReference &ref, int side, int row)
    {
        const Eigen::MatrixXd &J = kin.J_hand_pos[side];
        const Eigen::Vector3d v_des =
            ref.hand_vel_world[side] + cfg_.k_hand * (ref.hand_pos_world[side] - kin.hand_pos_world[side]);
        const Eigen::Vector3d v_from_rest =
            cfg_.feedforward ? Eigen::Vector3d(J * v_rest_) : Eigen::Vector3d::Zero();

        A_task_.middleRows<3>(row) = upperColumns(J);
        b_task_.segment<3>(row) = v_des - v_from_rest;
    }

    // TODO(user): 前臂俯仰任务，目标 a_z → 0（前臂水平）
    //   已推导：ȧ_z = (a × e_z)ᵀ ω = (a × e_z)ᵀ J_rot v
    //   需要填：A_task_.row(row)（1 x nu）与 b_task_(row)，并按 cfg_.feedforward 处理前馈，参照 buildHandRows
    //   可用数据：kin.forearm_axis_world[side]、kin.J_forearm_rot[side]、v_rest_、cfg_.k_pitch、upperColumns()
    //   验收：tools 中的有限差分脚本（待写）比较 J_pitch v 与 a_z 的数值导数

    // ȧ_z = (a × e_z)ᵀ ω = (a × e_z)ᵀ J_rot q̇    →    J_pitch = (a × e_z)ᵀ · J_rot
    //                                               (1×3) · (3×25) = 1×25

    void UpperBodyQp::buildForearmPitchRow(const UpperBodyKinematics &kin, int side, int row)
    {
        const Eigen::Vector3d &a = kin.forearm_axis_world[side];
        const Eigen::RowVectorXd J_pitch = a.cross(Eigen::Vector3d::UnitZ()).transpose() * kin.J_forearm_rot[side];
        const double az_dot_des = cfg_.k_pitch * (0.0 - a.z()); // 反馈：a_z → 0（前臂水平）
        const double az_dot_from_rest =                         // 前馈：基座运动带来的 ȧ_z
            cfg_.feedforward ? J_pitch.dot(v_rest_) : 0.0;

        A_task_.row(row) = upperColumns(J_pitch);
        b_task_(row) = az_dot_des - az_dot_from_rest;
    }

    Eigen::VectorXd UpperBodyQp::tauFeedforward(const UpperBodyKinematics &kin, double payload_mass) const
    {
        const bool mass_ok = std::isfinite(payload_mass) && payload_mass >= 0.0;
        if (!mass_ok)
            throw std::runtime_error("UpperBodyQp::tauFeedforward: payload_mass 必须为有限非负数");
        constexpr double kGravity = 9.81;
        // 负载重力 F = (0, 0, −m/2·g) 产生的广义力为 JᵀF；电机要抵消它，需提供 −JᵀF = Jᵀ·f_hold
        const Eigen::Vector3d f_hold(0.0, 0.0, 0.5 * payload_mass * kGravity);
        Eigen::VectorXd tau(nu_);
        for (int i = 0; i < nu_; ++i)
        {
            const int k = upper_dof_[i];
            tau(i) = kin.bias_force(k);
            for (int s = 0; s < kNumArms; ++s)
                tau(i) += kin.J_load_point[s].col(k).dot(f_hold);
        }
        return tau;
    }

    bool UpperBodyQp::update(const UpperBodyKinematics &kin, const UpperBodyReference &ref,
                             const Eigen::VectorXd &v, const Eigen::VectorXd &q_upper_measured,
                             const Eigen::VectorXd &q_nominal)
    {
        const bool sizes_ok = v.size() == nv_ && q_upper_measured.size() == nu_ && q_nominal.size() == nu_;
        if (!sizes_ok)
            throw std::runtime_error("UpperBodyQp::update: 输入尺寸错误");
        if (!q_des_.allFinite())
            throw std::runtime_error("UpperBodyQp::update: 未调用 reset()，q_des 仍为 NaN");

        v_rest_ = v;
        for (int k : upper_dof_)
            v_rest_(k) = 0.0;

        A_task_.setZero();
        b_task_.setZero();
        for (int s = 0; s < kNumArms; ++s)
        {
            buildHandRows(kin, ref, s, s * kRowsPerArm);
            if (cfg_.w_pitch > 0.0)
                buildForearmPitchRow(kin, s, s * kRowsPerArm + 3);
        }

        // 代价：½ xᵀ P x + gᵀ x
        const Eigen::VectorXd dq_posture = cfg_.k_posture * (q_nominal - q_des_);
        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nu_, nu_);
        const Eigen::MatrixXd P =
            2.0 * (A_task_.transpose() * w_task_.asDiagonal() * A_task_ + (cfg_.w_posture + cfg_.w_reg) * I);
        const Eigen::VectorXd g =
            -2.0 * (A_task_.transpose() * w_task_.asDiagonal() * b_task_ + cfg_.w_posture * dq_posture);

        // 约束：下一拍 q_des 不越过关节限位，且不超过速度上限
        Eigen::VectorXd lo = ((q_min_ - q_des_) / cfg_.dt).cwiseMax(-cfg_.dq_max);
        Eigen::VectorXd hi = ((q_max_ - q_des_) / cfg_.dt).cwiseMin(cfg_.dq_max);
        hi = hi.cwiseMax(lo); // q_des 已越界超过一拍可回退量时，只允许朝限位内侧移动
        P_last_ = P;
        g_last_ = g;
        lo_last_ = lo;
        hi_last_ = hi;

        const bool ok = solver_ready_ ? solver_.updateProblem(P, g, I, lo, hi)
                                      : (solver_ready_ = solver_.initialize(P, g, I, lo, hi));
        Eigen::VectorXd x;
        if (!ok || !solver_.solve(x) || !x.allFinite())
        {
            dq_des_.setZero();
            return false;
        }

        task_residual_ = A_task_ * x - b_task_;
        dq_des_ = x;
        q_des_ += x * cfg_.dt;
        // 抗饱和：手臂被挡住时 q_des 不再远离实测值
        const Eigen::VectorXd delta = Eigen::VectorXd::Constant(nu_, cfg_.delta_max);
        q_des_ = q_des_.cwiseMax(q_upper_measured - delta).cwiseMin(q_upper_measured + delta);
        return true;
    }
}
