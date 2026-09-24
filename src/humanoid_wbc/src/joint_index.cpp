#include "humanoid_wbc/joint_index.hpp"
#include <stdexcept>
#include <iostream>
// names 定义【你的】关节顺序。查不到 / 类型不对 → 直接 throw
JointIndex::JointIndex(const mjModel *m, const std::vector<std::string> &names)
{

    for (const auto &name : names)
    {
        JointEntry e;
        e.name = name;
        e.jid = mj_name2id(m, mjOBJ_JOINT, name.c_str());
        if (e.jid < 0)
        {
            throw std::runtime_error(
                "JointIndex: joint name not found: " + name);
        }

        int jnt_type = m->jnt_type[e.jid];
        if (jnt_type != mjJNT_HINGE)
        {
            throw std::runtime_error(
                "JointIndex: joint type is not HINGE, expected HINGE: " + name);
        }

        e.qposadr = m->jnt_qposadr[e.jid];
        e.dofadr = m->jnt_dofadr[e.jid];
        e.actid = -1;

        for (int aid = 0; aid < m->nu; ++aid)
        {
            if (m->actuator_trntype[aid] == mjTRN_JOINT && m->actuator_trnid[2 * aid] == e.jid)
            {
                e.actid = aid;
                break;
            }
        }
        if (e.actid < 0)
            throw std::runtime_error("JointIndex: no actuator for: " + name);

        // 拿到 actid 之后才能读限幅
        e.ctrl_min = m->actuator_ctrlrange[2 * e.actid + 0];
        e.ctrl_max = m->actuator_ctrlrange[2 * e.actid + 1];

        entries_.push_back(e);
    }
    // ── 收尾检查：整张表的整体性质 ──
    if (static_cast<int>(entries_.size()) != m->nu)
    {
        throw std::runtime_error("JointIndex: 表长 " + std::to_string(entries_.size()) + " 与 m->nu=" + std::to_string(m->nu) + " 不符");
    }

    std::vector<bool> seen(m->nu, false);
    for (const auto &x : entries_)
    {
        if (!(x.ctrl_max > x.ctrl_min))
            throw std::runtime_error("JointIndex: 力矩限幅无效: " + x.name);

        if (seen[x.actid])
            throw std::runtime_error("JointIndex: actuator 被重复占用: " + x.name);
        seen[x.actid] = true;
    }
}

void JointIndex::print() const
{
    for (const auto &entry : entries_)
    {
        std::cout << "Joint: " << entry.name << ", ID: " << entry.jid << ", Actuator ID: " << entry.actid << ", qposadr: " << entry.qposadr << ", dofadr: " << entry.dofadr
                  << ", ctrl_min: " << entry.ctrl_min << ", ctrl_max: " << entry.ctrl_max << std::endl;
    }
}
