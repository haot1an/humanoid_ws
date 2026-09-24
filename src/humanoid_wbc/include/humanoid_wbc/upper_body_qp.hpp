// 上肢速度级加权 QP（torso + 双臂 9 DoF）。设计见 docs/DESIGN.md UBQP-001、docs/DECISIONS.md DEC-003
//
//   决策变量  x = dq_u ∈ R^9（顺序 = 构造时传入的 upper_dof）
//   min  Σ_i w_i ||A_i x − b_i||² + w_posture ||x − dq_posture||² + w_reg ||x||²
//   s.t. max(−dq_max, (q_min − q_des)/dt) ≤ x ≤ min(dq_max, (q_max − q_des)/dt)
//   任务行（每臂）：手部位置 3 行 + 前臂俯仰 1 行
//   输出：q_des += x·dt，再限制 |q_des − q| ≤ delta_max（抗饱和）；dq_des = x
#pragma once
#include <Eigen/Dense>
#include <vector>
#include "humanoid_wbc/osqp_solver.hpp"
#include "humanoid_wbc/upper_body_types.hpp"

namespace humanoid_wbc
{
    struct UpperBodyQpConfig
    {
        double dt = 0.01;         // 100 Hz（DEC-003）
        double k_hand = 20.0;     // 手部位置反馈 [1/s]
        double k_pitch = 10.0;    // 前臂俯仰反馈 [1/s]
        double w_hand = 100.0;    // 残差单位 m/s
        double w_pitch = 0.0;     // 残差单位 1/s；buildForearmPitchRow 实现前保持 0（= 关闭）
        double w_posture = 0.1;   // posture regularization（DEC-002 / DEC-003）
        double k_posture = 2.0;   // dq_posture = k_posture (q_nominal − q_des) [1/s]
        double w_reg = 1e-4;      // dq 阻尼正则，改善接近奇异时的条件数
        double dq_max = 5.0;      // [rad/s] 占位值：H1 MJCF 未给关节速度上限
        double delta_max = 0.2;   // [rad] 抗饱和：|q_des − q| 上限
        bool feedforward = true;  // 是否扣除“非上肢自由度（基座）运动”带来的任务速度
    };

    class UpperBodyQp
    {
    public:
        // upper_dof: 9 个上肢关节在 v (nv) 中的下标；q_min / q_max: 同顺序的关节限位 [rad]
        UpperBodyQp(int nv, std::vector<int> upper_dof,
                    Eigen::VectorXd q_min, Eigen::VectorXd q_max,
                    const UpperBodyQpConfig &cfg);

        // 进入 QP 模式时调用一次：q_des 从实测关节角开始积分（reference 捕获时刻）
        void reset(const Eigen::VectorXd &q_upper_measured);

        // 一个控制周期（dt）。v: 实测广义速度 (nv)。返回 false = QP 失败，q_des 保持、dq_des 置零
        bool update(const UpperBodyKinematics &kin, const UpperBodyReference &ref,
                    const Eigen::VectorXd &v, const Eigen::VectorXd &q_upper_measured,
                    const Eigen::VectorXd &q_nominal);

        // 上肢前馈力矩（nu）：模型重力 g(q) + 已知负载的重力补偿
        //   负载按两臂均分，作用在 kin.load_point_world：τ_load = J_load(:, upper)ᵀ · (0, 0, m/2·g)
        Eigen::VectorXd tauFeedforward(const UpperBodyKinematics &kin, double payload_mass) const;

        const Eigen::VectorXd &qDes() const { return q_des_; }
        const Eigen::VectorXd &dqDes() const { return dq_des_; }
        const Eigen::VectorXd &taskResidual() const { return task_residual_; } // A x − b，按行
        const Eigen::MatrixXd &taskMatrix() const { return A_task_; }         // 最近一次 update 的 A（8 x nu）
        const Eigen::VectorXd &taskVector() const { return b_task_; }         // 最近一次 update 的 b（8）
        // 最近一次 update 交给 OSQP 的问题：½xᵀPx + gᵀx，lo ≤ x ≤ hi（用于与 GPU 批量求解器做一致性核对）
        const Eigen::MatrixXd &lastP() const { return P_last_; }
        const Eigen::VectorXd &lastG() const { return g_last_; }
        const Eigen::VectorXd &lastLo() const { return lo_last_; }
        const Eigen::VectorXd &lastHi() const { return hi_last_; }
        double solveMicroseconds() const { return solver_.solveMicroseconds(); }
        int numUpper() const { return nu_; }

    private:
        // 每臂 4 行：[row, row+3) 手部位置，row+3 前臂俯仰
        void buildHandRows(const UpperBodyKinematics &kin, const UpperBodyReference &ref, int side, int row);
        void buildForearmPitchRow(const UpperBodyKinematics &kin, int side, int row); // 用户实现，有限差分核对：upper_body_jacobian_check

        Eigen::MatrixXd upperColumns(const Eigen::MatrixXd &J) const; // J(:, upper_dof)

        int nv_;
        int nu_;
        std::vector<int> upper_dof_;
        Eigen::VectorXd q_min_, q_max_;
        UpperBodyQpConfig cfg_;

        Eigen::VectorXd v_rest_; // v 中上肢列置零：只保留基座（及腿）运动，用于前馈
        Eigen::MatrixXd A_task_; // 8 x nu
        Eigen::VectorXd b_task_; // 8
        Eigen::VectorXd w_task_; // 8，每行权重
        Eigen::VectorXd task_residual_;

        Eigen::VectorXd q_des_, dq_des_;
        Eigen::MatrixXd P_last_;
        Eigen::VectorXd g_last_, lo_last_, hi_last_;
        OsqpSolver solver_;
        bool solver_ready_ = false;
    };
}
