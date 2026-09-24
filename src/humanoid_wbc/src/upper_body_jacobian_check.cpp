// 有限差分核对 UpperBodyQp 的任务行（A、b 中的前馈项）是否等于真实运动学导数
//
// 对随机姿态 q 与随机广义速度 v（含基座平移/转动、上肢关节；腿为 0）：
//   解析：task_dot = A·v_u + J·v_rest，其中 J·v_rest 由 b 反推（k 已知、reference 取当前值）
//     手部行：b = v_des − J_hand v_rest，reference = 当前位置 ⇒ v_des = 0 ⇒ J_hand v_rest = −b
//     俯仰行：b = k_pitch(0 − a_z) − J_pitch v_rest ⇒ J_pitch v_rest = −k_pitch a_z − b
//   数值：q± = integratePos(q, ±v, ε)，task_dot ≈ (task(q+) − task(q−)) / 2ε
// 分别对 v_u（仅上肢）与 v_rest（仅基座）检验，区分 A 与前馈项的错误。
#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "humanoid_wbc/h1_config.hpp"
#include "humanoid_wbc/joint_index.hpp"
#include "humanoid_wbc/mujoco_upper_body_kinematics.hpp"
#include "humanoid_wbc/upper_body_qp.hpp"

using namespace humanoid_wbc;

namespace
{
    // 每臂 4 个任务量：手部 x, y, z 与前臂 a_z
    Eigen::Matrix<double, 8, 1> taskValues(const UpperBodyKinematics &kin)
    {
        Eigen::Matrix<double, 8, 1> y;
        for (int s = 0; s < kNumArms; ++s)
        {
            y.segment<3>(4 * s) = kin.hand_pos_world[s];
            y(4 * s + 3) = kin.forearm_axis_world[s].z();
        }
        return y;
    }
}

int main()
{
    const std::string xml = "/home/tt/humanoid_ws/src/humanoid_wbc/model/unitree_h1/scene.xml";
    char err[1000] = "";
    mjModel *m = mj_loadXML(xml.c_str(), nullptr, err, sizeof(err));
    if (m == nullptr)
        throw std::runtime_error(std::string("mj_loadXML: ") + err);
    mjData *d = mj_makeData(m);
    const int nv = m->nv;

    const std::vector<std::string> names(kMjcfJointNames.begin(), kMjcfJointNames.end());
    const JointIndex idx(m, names);
    std::vector<int> upper_joint, upper_dof;
    for (int i = kTorso; i <= kRightElbow; ++i)
    {
        upper_joint.push_back(i);
        upper_dof.push_back(idx[i].dofadr);
    }
    const int nu = static_cast<int>(upper_dof.size());
    Eigen::VectorXd q_min(nu), q_max(nu);
    for (int k = 0; k < nu; ++k)
    {
        const int jid = idx[upper_joint[k]].jid;
        q_min(k) = m->jnt_range[2 * jid];
        q_max(k) = m->jnt_range[2 * jid + 1];
    }

    UpperBodyQpConfig cfg;
    cfg.w_pitch = 1.0; // 只为让俯仰行被组装
    cfg.feedforward = true;
    MujocoUpperBodyKinematics backend(m);
    UpperBodyKinematics kin(nv), kin_p(nv), kin_m(nv);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    const double eps = 1e-6;
    double worst_u = 0.0, worst_rest = 0.0;
    const int n_samples = 20;

    for (int sample = 0; sample < n_samples; ++sample)
    {
        mj_resetData(m, d);
        d->eq_active[mj_name2id(m, mjOBJ_EQUALITY, "hang")] = 0;
        // 随机基座姿态（单位四元数）与随机上肢关节角（限位内 10%~90%）
        Eigen::Quaterniond qb(1.0, 0.3 * uni(rng), 0.3 * uni(rng), 0.3 * uni(rng));
        qb.normalize();
        d->qpos[2] = 1.0;
        d->qpos[3] = qb.w(), d->qpos[4] = qb.x(), d->qpos[5] = qb.y(), d->qpos[6] = qb.z();
        Eigen::VectorXd q_u(nu);
        for (int k = 0; k < nu; ++k)
        {
            const double lo = q_min(k), hi = q_max(k);
            q_u(k) = lo + (0.5 + 0.4 * uni(rng)) * (hi - lo);
            d->qpos[idx[upper_joint[k]].qposadr] = q_u(k);
        }
        mj_forward(m, d);
        backend.update(d, kin);

        Eigen::VectorXd v_u = Eigen::VectorXd::Zero(nv), v_rest = Eigen::VectorXd::Zero(nv);
        for (int k = 0; k < 6; ++k)
            v_rest(k) = uni(rng);
        for (int k = 0; k < nu; ++k)
            v_u(upper_dof[k]) = uni(rng);

        UpperBodyQp qp(nv, upper_dof, q_min, q_max, cfg);
        qp.reset(q_u);
        UpperBodyReference ref;
        for (int s = 0; s < kNumArms; ++s)
        {
            ref.hand_pos_world[s] = kin.hand_pos_world[s];
            ref.hand_vel_world[s].setZero();
        }
        qp.update(kin, ref, v_rest, q_u, q_u); // v = v_rest：b 中的前馈项只含基座速度
        const Eigen::MatrixXd A = qp.taskMatrix();
        const Eigen::VectorXd b = qp.taskVector();

        Eigen::VectorXd vu_small(nu);
        for (int k = 0; k < nu; ++k)
            vu_small(k) = v_u(upper_dof[k]);
        const Eigen::Matrix<double, 8, 1> pred_u = A * vu_small;
        Eigen::Matrix<double, 8, 1> pred_rest;
        for (int s = 0; s < kNumArms; ++s)
        {
            pred_rest.segment<3>(4 * s) = -b.segment<3>(4 * s);
            pred_rest(4 * s + 3) = -cfg.k_pitch * kin.forearm_axis_world[s].z() - b(4 * s + 3);
        }

        auto finiteDiff = [&](const Eigen::VectorXd &v)
        {
            std::vector<mjtNum> q0(d->qpos, d->qpos + m->nq);
            mj_integratePos(m, d->qpos, v.data(), eps);
            mj_forward(m, d);
            backend.update(d, kin_p);
            std::copy(q0.begin(), q0.end(), d->qpos);
            mj_integratePos(m, d->qpos, v.data(), -eps);
            mj_forward(m, d);
            backend.update(d, kin_m);
            std::copy(q0.begin(), q0.end(), d->qpos);
            mj_forward(m, d);
            return Eigen::Matrix<double, 8, 1>((taskValues(kin_p) - taskValues(kin_m)) / (2 * eps));
        };
        const Eigen::Matrix<double, 8, 1> fd_u = finiteDiff(v_u), fd_rest = finiteDiff(v_rest);
        const double err_u = (pred_u - fd_u).cwiseAbs().maxCoeff();
        const double err_rest = (pred_rest - fd_rest).cwiseAbs().maxCoeff();
        worst_u = std::max(worst_u, err_u);
        worst_rest = std::max(worst_rest, err_rest);
        if (sample < 3)
            std::cout << "sample " << sample << "  pitch rows (L,R)  A·v_u=" << pred_u(3) << "," << pred_u(7)
                      << "  FD=" << fd_u(3) << "," << fd_u(7) << " | J·v_rest=" << pred_rest(3) << ","
                      << pred_rest(7) << "  FD=" << fd_rest(3) << "," << fd_rest(7) << "\n";
    }
    std::cout << "samples=" << n_samples << " eps=" << eps << "\n"
              << "max |A·v_u − FD|      over 8 rows = " << worst_u << "\n"
              << "max |J·v_rest − FD|   over 8 rows = " << worst_rest << "\n";
    const bool ok = worst_u < 1e-6 && worst_rest < 1e-6;
    std::cout << (ok ? "PASS" : "FAIL") << " (threshold 1e-6)\n";
    mj_deleteData(d);
    mj_deleteModel(m);
    return ok ? 0 : 1;
}
