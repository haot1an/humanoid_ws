#include "humanoid_wbc/mujoco_bridge.hpp"
#include <cassert>
#include <algorithm> // std::clamp
#include <iostream>
#include <stdexcept>
#include <cmath> // std::isfinite

MujocoBridge::MujocoBridge(const mjModel *m, const std::vector<std::string> &names) : idx_(m, names), sat_count_(idx_.size(), 0)
{
    assert(idx_.size() > 0 && idx_[0].qposadr == 7 && idx_[0].dofadr == 6);
}
Eigen::VectorXd MujocoBridge::lowerControlLimits() const
{
    Eigen::VectorXd limits(idx_.size());

    for (int i = 0; i < static_cast<int>(idx_.size()); ++i)
        limits[i] = idx_[i].ctrl_min;

    return limits;
}

Eigen::VectorXd MujocoBridge::upperControlLimits() const
{
    Eigen::VectorXd limits(idx_.size());

    for (int i = 0; i < static_cast<int>(idx_.size()); ++i)
        limits[i] = idx_[i].ctrl_max;

    return limits;
}

void MujocoBridge::readState(const mjData *d, RobotState &s) const
{

    /*
    输入 const mjData* d  →  输出 RobotState& s          (函数是 const，不改 bridge)

1. 断言   s.q_joint.size() == idx_.size()
2. 时间   s.time            = d->time
3. 位置   s.base_pos_world  ← qpos[base+0..2]     世界系
4. 姿态   s.base_quat_wb    ← qpos[base+3..6]     (w,x,y,z)，用 Quaterniond 构造函数
5. 线速度 s.base_vel_world  ← qvel[base+0..2]     世界系，直接抄
6. 角速度 s.base_omega_body ← qvel[base+3..5]     body系，直接抄
7. 关节   循环 i：q_joint[i] ← qpos[idx_[i].qposadr]
                 dq_joint[i] ← qvel[idx_[i].dofadr]*/

    assert(s.q_joint.size() == idx_.size());

    s.time = d->time;

    s.base_quat_wb = Eigen::Quaterniond(d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]);

    s.base_pos_world = Eigen::Map<const Eigen::Vector3d>(d->qpos + 0);
    s.base_vel_world = Eigen::Map<const Eigen::Vector3d>(d->qvel + 0);
    s.base_omega_body = Eigen::Map<const Eigen::Vector3d>(d->qvel + 3);

    for (int i = 0; i < idx_.size(); ++i)
    {
        const auto &e = idx_[i];
        s.q_joint[i] = d->qpos[e.qposadr];
        s.dq_joint[i] = d->qvel[e.dofadr];
    }
}

void MujocoBridge::writeCommand(const JointCommand &cmd, const RobotState &s, mjData *d)
{

    assert(cmd.q_des.size() == idx_.size() && cmd.dq_des.size() == idx_.size() &&
           cmd.kp.size() == idx_.size() && cmd.kd.size() == idx_.size() &&
           cmd.tau_ff.size() == idx_.size() && s.q_joint.size() == idx_.size());

    for (int i = 0; i < idx_.size(); ++i)
    {
        const auto &e = idx_[i];
        double tau = cmd.tau_ff[i] + cmd.kp[i] * (cmd.q_des[i] - s.q_joint[i]) + cmd.kd[i] * (cmd.dq_des[i] - s.dq_joint[i]);

        if (!std::isfinite(tau))
        {
            // NaN 会让 MuJoCo 把全部 19 个力矩清零（engine_forward.c:395），机器人瞬间瘫软。
            // 立刻 throw：宁可停，也不要带着 NaN 静默跑下去。
            throw std::runtime_error("MujocoBridge::writeCommand: tau is not finite for joint " + e.name);
        }

        // 限幅
        if (tau < e.ctrl_min || tau > e.ctrl_max)
        {
            ++sat_count_[i];
            tau = std::clamp(tau, e.ctrl_min, e.ctrl_max); // 自己 clamp，需要 <algorithm>
        }
        d->ctrl[e.actid] = tau;
    }
}

void MujocoBridge::printSaturationStats() const
{
    for (int i = 0; i < idx_.size(); ++i)
        if (sat_count_[i] > 0)
            std::cout << idx_[i].name << " 饱和 " << sat_count_[i] << " 次\n";
}