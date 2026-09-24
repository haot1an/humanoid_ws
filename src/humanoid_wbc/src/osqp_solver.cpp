#include "humanoid_wbc/osqp_solver.hpp"
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>
namespace humanoid_wbc
{

    OsqpSolver::OsqpSolver(int nvar, int ncon)
        : nvar_(nvar),
          ncon_(ncon)
    {
    }
    bool OsqpSolver::initialize(
        const Eigen::MatrixXd &P,
        const Eigen::VectorXd &q,
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &l,
        const Eigen::VectorXd &u)
    {
        solver_.settings()->setVerbosity(false);
        solver_.settings()->setAbsoluteTolerance(1e-6);
        solver_.settings()->setRelativeTolerance(1e-6);
        P_sparse_ = P.sparseView();
        A_sparse_ = A.sparseView();

        solver_.data()->setNumberOfVariables(nvar_);
        solver_.data()->setNumberOfConstraints(ncon_);

        q_ = q;
        l_ = l;
        u_ = u;
        if (!solver_.data()->setHessianMatrix(P_sparse_))
            return false;

        if (!solver_.data()->setGradient(q_))
            return false;

        if (!solver_.data()->setLinearConstraintsMatrix(A_sparse_))
            return false;

        if (!solver_.data()->setLowerBound(l_))
            return false;

        if (!solver_.data()->setUpperBound(u_))
            return false;
        if (!solver_.initSolver())
            return false;

        return true;
    }

    bool OsqpSolver::solve(Eigen::VectorXd &solution)
    {
        
        const auto start = std::chrono::steady_clock::now();
        auto flag = solver_.solveProblem();
        solve_us_ = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();

        if (flag != OsqpEigen::ErrorExitFlag::NoError)
            return false;

        if (solver_.getStatus() != OsqpEigen::Status::Solved)
            return false;

        solution = solver_.getSolution();

        return true;
    }

    bool OsqpSolver::updateProblem(
        const Eigen::MatrixXd &P,
        const Eigen::VectorXd &q,
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &l,
        const Eigen::VectorXd &u)
    {
        if (q.size() != q_.size() || l.size() != l_.size() || u.size() != u_.size())
            return false; // 尺寸不变才能保证 q_/l_/u_ 不重新分配、OsqpEigen 持有的指针仍有效

        P_sparse_ = P.sparseView();
        A_sparse_ = A.sparseView();
        // 先写入成员：若下面因稀疏结构变化触发重新 setup，OsqpEigen 读到的是本拍的 q/l/u
        q_ = q;
        l_ = l;
        u_ = u;

        if (!solver_.updateHessianMatrix(P_sparse_))
            return false;

        if (!solver_.updateGradient(q_))
            return false;

        if (!solver_.updateLinearConstraintsMatrix(A_sparse_))
            return false;
        if (!solver_.updateBounds(l_, u_))
            return false;
        return true;
    }

}