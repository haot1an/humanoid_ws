#pragma once
#include <mujoco/mujoco.h>
#include <string>
#include <vector>
#include "humanoid_wbc/joint_index.hpp"
#include "humanoid_wbc/robot_state.hpp"

class MujocoBridge
{
public:
    MujocoBridge(const mjModel *m, const std::vector<std::string> &names);

    void readState(const mjData *d, RobotState &s) const;
    void writeCommand(const JointCommand &cmd, const RobotState &s, mjData *d);

    int numJoints() const { return idx_.size(); } // 给 RobotState 传尺寸用
    void printSaturationStats() const;
    Eigen::VectorXd lowerControlLimits() const;
    Eigen::VectorXd upperControlLimits() const;

private:
    JointIndex idx_;
    std::vector<long> sat_count_;
};