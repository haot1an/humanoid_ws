//OSQP 封装
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>

namespace humanoid_wbc
{

class OsqpSolver
{
public:
    OsqpSolver(int nvar, int ncon);

    bool initialize(
        const Eigen::MatrixXd& P,
        const Eigen::VectorXd& q,
        const Eigen::MatrixXd& A,
        const Eigen::VectorXd& l,
        const Eigen::VectorXd& u);

    bool solve(Eigen::VectorXd& solution);



bool updateProblem(
    const Eigen::MatrixXd& P,
    const Eigen::VectorXd& q,
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& l,
    const Eigen::VectorXd& u);

    const OSQPInfo* info() const { return solver_.workspace()->info; }
    double solveMicroseconds() const { return solve_us_; }

private:
    double solve_us_ = 0.0;
    int nvar_;
    int ncon_;

    OsqpEigen::Solver solver_;

    Eigen::SparseMatrix<double> P_sparse_;
    Eigen::SparseMatrix<double> A_sparse_;

    // OsqpEigen 只保存 q / l / u 的指针、不拷贝（OsqpEigen/Data.hpp setGradient/setLowerBound 注释），
    // Hessian 或 A 稀疏结构变化时会用这些指针重新 setup，所以必须与求解器同生命周期
    Eigen::VectorXd q_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;
};

}