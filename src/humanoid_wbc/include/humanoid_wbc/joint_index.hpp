#pragma once
#include <mujoco/mujoco.h> 
#include <string>
#include <vector>

struct JointEntry{
    std::string name;
    int jid = -1;  // mjOBJ_JOINT 的 id
    int qposadr = -1;  // qpos 下标        ← jnt_qposadr[jid]
    int dofadr = -1;  // qvel/qacc/力矩 下标 ← jnt_dofadr[jid]
    int actid   = -1;   // mjOBJ_ACTUATOR 的 id
    double ctrl_min = 0.0;
    double ctrl_max = 0.0;
};


class JointIndex {
public:
    // names 定义【你的】关节顺序。查不到 / 类型不对 → 直接 throw
    JointIndex(const mjModel* m, const std::vector<std::string>& names);

    int size() const { return static_cast<int>(entries_.size()); }
    const JointEntry& operator[](int i) const { return entries_.at(i); }

    void print() const;   // 第 1 步验收要用：打全表跟我给你的对照

private:
    std::vector<JointEntry> entries_;
};