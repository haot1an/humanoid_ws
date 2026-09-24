// 上肢 QP 吊挂验证（无 RL、无 GUI）
//
// pelvis 由 scene.xml 的 hang weld 焊在 mocap 锚点上；锚点按 EXP-004 实测骨盆振荡运动：
//   竖直 z：幅值 1.85 cm、2.56 Hz；roll：幅值 0.10 rad、1.28 Hz（默认策略 0.5 m/s 行走，t ≥ 2 s 统计）
// 腿按 Isaac Lab H1 默认角做 PD 保持；上肢（torso + 双臂）按 --mode：
//   fixed：关节角保持在捕获值（基线）      qp：UpperBodyQp 100 Hz
// 两种模式的 PD 增益、tau_ff（qfrc_bias 的上肢分量）完全相同，只差 q_des/dq_des 的来源。
//
// 时间线：[0,1) 默认姿态稳定 → t=1 捕获 reference 与 q_des → t=2 手部目标阶跃 → t=3 锚点开始振荡
// 用法：upper_body_qp_test --mode qp|fixed [--ff 1|0] [--osc 1|0] [--step 0.05] [--duration 12] [--w_pitch 0]
//        [--pose default|carry0|carry15] [--ref captured|nominal] [--payload kg] [--payload_ff 1|0]
//        [--pitch_amp rad] [--csv out.csv]
// --payload：用 mjSpec 在两个 elbow_link 上各挂 m/2 的刚体（前臂轴上距肘 0.15 m），不改 XML；
//            近似为两个独立点质量，不含箱子整体转动惯量与双臂闭链
// --ref nominal：reference 取 t=0 时刻标称姿态的手部位置（负载下垂不会被“捕获”进 reference），q_des 从标称关节角开始
#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "humanoid_wbc/h1_config.hpp"
#include "humanoid_wbc/joint_index.hpp"
#include "humanoid_wbc/mujoco_bridge.hpp"
#include "humanoid_wbc/mujoco_upper_body_kinematics.hpp"
#include "humanoid_wbc/robot_state.hpp"
#include "humanoid_wbc/upper_body_qp.hpp"

using namespace humanoid_wbc;

namespace
{
    constexpr double kCaptureTime = 1.0;
    constexpr double kStepTime = 2.0;
    constexpr double kOscStart = 3.0;
    constexpr double kOscMetricStart = 4.0; // 跳过振荡起始 1 s 瞬态
    constexpr double kLimitMargin = 0.05;   // [rad] 上肢关节限位内缩

    struct Args
    {
        std::string model = "/home/tt/humanoid_ws/src/humanoid_wbc/model/unitree_h1/scene.xml";
        std::string mode = "qp";
        bool ff = true;
        bool osc = true;
        double step = 0.05; // [m] 双手目标沿 world x 的阶跃
        double duration = 12.0;
        double z_amp = 0.0185, z_hz = 2.56, roll_amp = 0.10, roll_hz = 1.28;
        double w_pitch = 0.0; // 前臂俯仰任务权重；0 = 关闭
        std::string pose = "default"; // default = Isaac 默认角；carry0 = 双臂 4 关节全 0；carry15 = 批准的端箱姿态
        std::string ref = "captured";  // captured = t=1 捕获实测手位置；nominal = t=0 标称姿态的手位置
        double payload = 0.0;          // [kg] 箱子质量，两臂均分
        bool payload_ff = false;       // tau_ff 是否加入已知负载的重力补偿
        double pitch_amp = 0.0, pitch_hz = 2.56; // 骨盆 pitch 振荡；EXP-004 fwd_0.5 实测等效幅值 0.0133 rad
        double k_hand = std::nan(""), k_pitch = std::nan(""); // NaN = 使用 UpperBodyQpConfig 默认值
        std::string csv;
        std::string dump_qp; // 每次 QP 调用写一行：t, P(nu*nu 行优先), g, lo, hi, x_osqp
        std::string dump_ctrl; // 每次 QP 调用写一行控制器完整输入输出，供 rl/ torch 版一致性测试
    };

    Args parseArgs(int argc, char **argv)
    {
        Args a;
        for (int i = 1; i + 1 < argc; i += 2)
        {
            const std::string k = argv[i], v = argv[i + 1];
            if (k == "--model") a.model = v;
            else if (k == "--mode") a.mode = v;
            else if (k == "--ff") a.ff = (v != "0");
            else if (k == "--osc") a.osc = (v != "0");
            else if (k == "--step") a.step = std::stod(v);
            else if (k == "--duration") a.duration = std::stod(v);
            else if (k == "--z_amp") a.z_amp = std::stod(v);
            else if (k == "--roll_amp") a.roll_amp = std::stod(v);
            else if (k == "--csv") a.csv = v;
            else if (k == "--dump_qp") a.dump_qp = v;
            else if (k == "--dump_ctrl") a.dump_ctrl = v;
            else if (k == "--w_pitch") a.w_pitch = std::stod(v);
            else if (k == "--pose") a.pose = v;
            else if (k == "--ref") a.ref = v;
            else if (k == "--payload") a.payload = std::stod(v);
            else if (k == "--payload_ff") a.payload_ff = (v != "0");
            else if (k == "--pitch_amp") a.pitch_amp = std::stod(v);
            else if (k == "--k_hand") a.k_hand = std::stod(v);
            else if (k == "--k_pitch") a.k_pitch = std::stod(v);
            else throw std::runtime_error("未知参数 " + k);
        }
        if (a.mode != "qp" && a.mode != "fixed")
            throw std::runtime_error("--mode 只能是 qp 或 fixed");
        if (a.pose != "default" && a.pose != "carry0" && a.pose != "carry15")
            throw std::runtime_error("--pose 只能是 default / carry0 / carry15");
        if (a.ref != "captured" && a.ref != "nominal")
            throw std::runtime_error("--ref 只能是 captured 或 nominal");
        if (!(a.payload >= 0.0))
            throw std::runtime_error("--payload 必须 >= 0");
        return a;
    }

    // Isaac Lab H1 训练契约（docs/validation/2026-09-18/EXP-003/isaac_h1_flat_nomassrand.json）
    struct JointGain
    {
        double q0, kp, kd;
    };
    JointGain isaacContract(const std::string &name)
    {
        static const std::map<std::string, JointGain> table = {
            {"hip_yaw", {0.0, 150, 5}}, {"hip_roll", {0.0, 150, 5}}, {"hip_pitch", {-0.28, 200, 5}},
            {"knee", {0.79, 200, 5}}, {"ankle", {-0.52, 20, 4}}, {"torso", {0.0, 200, 5}},
            {"shoulder_pitch", {0.28, 40, 10}}, {"shoulder_roll", {0.0, 40, 10}},
            {"shoulder_yaw", {0.0, 40, 10}}, {"elbow", {0.52, 40, 10}}};
        std::string key = name;
        for (const std::string prefix : {"left_", "right_"})
            if (key.rfind(prefix, 0) == 0)
                key = key.substr(prefix.size());
        const auto it = table.find(key);
        if (it == table.end())
            throw std::runtime_error("isaacContract: 未知关节 " + name);
        return it->second;
    }

    // 批准的端箱姿态（DEC-003，2026-09-18）：上臂前倾 15°、前臂水平朝前、两手间距 0.36 m
    // 数值由 scripts/view_carry_poses.py --thetas 15 --spacing 0.36 的 IK 求得（残差 1.7e-12）
    // 顺序：肩 pitch, 肩 roll, 肩 yaw, 肘；右臂 roll / yaw 取反
    constexpr double kCarry15Left[4] = {-0.235528, -0.0932, 0.120633, 0.149264};

    double carryJointAngle(const std::string &pose, int joint)
    {
        if (pose == "carry0")
            return 0.0;
        const int k = (joint - kLeftShoulderPitch) % 4;
        const bool right = joint >= kRightShoulderPitch;
        const bool flip = right && (k == 1 || k == 2);
        return flip ? -kCarry15Left[k] : kCarry15Left[k];
    }

    // 用 mjSpec 在两个 elbow_link 上各加一个 m/2 的刚体子节点，位置 = 前臂轴上距肘 load_distance
    mjModel *loadModel(const std::string &xml, double payload, double load_distance)
    {
        char err[1000] = "";
        if (payload <= 0.0)
        {
            mjModel *m = mj_loadXML(xml.c_str(), nullptr, err, sizeof(err));
            if (m == nullptr)
                throw std::runtime_error(std::string("mj_loadXML: ") + err);
            return m;
        }
        mjSpec *spec = mj_parseXML(xml.c_str(), nullptr, err, sizeof(err));
        if (spec == nullptr)
            throw std::runtime_error(std::string("mj_parseXML: ") + err);
        const Eigen::Vector3d axis = Eigen::Vector3d(0.28, 0.0, -0.015).normalized(); // 与 MujocoUpperBodyKinematics 一致
        for (const std::string side : {"left", "right"})
        {
            mjsBody *elbow = mjs_findBody(spec, (side + "_elbow_link").c_str());
            if (elbow == nullptr)
                throw std::runtime_error("mjSpec: 找不到 " + side + "_elbow_link");
            mjsBody *load = mjs_addBody(elbow, nullptr);
            mjs_setName(load->element, (side + "_payload").c_str());
            for (int i = 0; i < 3; ++i)
                load->pos[i] = axis(i) * load_distance;
            load->mass = 0.5 * payload;
            load->explicitinertial = 1;
            const double I = 0.4 * load->mass * 0.03 * 0.03; // 半径 3 cm 实心球，近似点质量
            load->inertia[0] = load->inertia[1] = load->inertia[2] = I;
        }
        mjModel *m = mj_compile(spec, nullptr);
        mj_deleteSpec(spec);
        if (m == nullptr)
            throw std::runtime_error("mj_compile 失败（payload 挂载）");
        return m;
    }

    struct Stats
    {
        double sum_sq = 0.0, max = 0.0;
        long n = 0;
        void add(double x)
        {
            sum_sq += x * x;
            max = std::max(max, std::abs(x));
            ++n;
        }
        double rms() const { return n ? std::sqrt(sum_sq / n) : std::nan(""); }
    };
}

int main(int argc, char **argv)
{
    const Args args = parseArgs(argc, argv);

    constexpr double kLoadDistance = 0.15; // [m] 箱子压在前臂上距肘的距离（DEC-003 d）
    mjModel *m = loadModel(args.model, args.payload, kLoadDistance);
    mjData *d = mj_makeData(m);
    // 控制器侧模型：不含箱子（真机控制器不知道箱子）。仿真用 m（含箱子），运动学/偏置力一律取自 mc
    // 箱子只是无关节的叶子刚体，nq / nv 与关节地址不变，qpos / qvel 可直接拷贝
    mjModel *mc = args.payload > 0.0 ? loadModel(args.model, 0.0, kLoadDistance) : m;
    mjData *dc = args.payload > 0.0 ? mj_makeData(mc) : d;
    if (mc->nq != m->nq || mc->nv != m->nv)
        throw std::runtime_error("控制器模型与仿真模型的 nq / nv 不一致");
    auto syncControllerModel = [&]()
    {
        if (dc == d)
            return;
        mju_copy(dc->qpos, d->qpos, m->nq);
        mju_copy(dc->qvel, d->qvel, m->nv);
        mj_forward(mc, dc);
    };
    double total_mass = 0.0;
    for (int b = 1; b < m->nbody; ++b)
        total_mass += m->body_mass[b];

    const std::vector<std::string> names(kMjcfJointNames.begin(), kMjcfJointNames.end());
    MujocoBridge bridge(m, names);
    const JointIndex idx(m, names);
    const int nj = idx.size();

    // 上肢 = torso + 双臂，顺序沿用 h1_config 的 enum（kTorso .. kRightElbow）
    std::vector<int> upper_joint, upper_dof;
    for (int i = kTorso; i <= kRightElbow; ++i)
    {
        upper_joint.push_back(i);
        upper_dof.push_back(idx[i].dofadr);
    }
    const int nu = static_cast<int>(upper_joint.size());
    Eigen::VectorXd q_min(nu), q_max(nu);
    for (int k = 0; k < nu; ++k)
    {
        const int jid = idx[upper_joint[k]].jid;
        q_min(k) = m->jnt_range[2 * jid] + kLimitMargin;
        q_max(k) = m->jnt_range[2 * jid + 1] - kLimitMargin;
    }

    // 初始状态：pelvis (0,0,1.06) 单位姿态（与 anchor 重合），关节取 Isaac 默认角，weld 生效
    JointCommand cmd(nj);
    for (int i = 0; i < nj; ++i)
    {
        const JointGain g = isaacContract(names[i]);
        // carry0：肩 pitch/roll/yaw 与肘全为 0 → 上臂竖直、前臂水平朝前（2026-09-18 实测 a_z = −0.053）
        const bool arm_joint = i >= kLeftShoulderPitch;
        const double q0 = (args.pose != "default" && arm_joint) ? carryJointAngle(args.pose, i) : g.q0;
        cmd.q_des(i) = q0;
        cmd.kp(i) = g.kp;
        cmd.kd(i) = g.kd;
        d->qpos[idx[i].qposadr] = q0;
    }
    d->qpos[0] = 0.0;
    d->qpos[1] = 0.0;
    d->qpos[2] = 1.06;
    d->qpos[3] = 1.0;
    const int hang = mj_name2id(m, mjOBJ_EQUALITY, "hang");
    const int anchor_body = mj_name2id(m, mjOBJ_BODY, "anchor");
    const int pelvis = mj_name2id(m, mjOBJ_BODY, "pelvis");
    if (hang < 0 || anchor_body < 0 || pelvis < 0 || m->body_mocapid[anchor_body] < 0)
        throw std::runtime_error("scene.xml 中缺少 hang weld / mocap anchor / pelvis");
    const int mocap = m->body_mocapid[anchor_body];
    d->eq_active[hang] = 1;
    mj_forward(m, d);
    mju_copy3(d->mocap_pos + 3 * mocap, d->xpos + 3 * pelvis);
    mju_copy4(d->mocap_quat + 4 * mocap, d->xquat + 4 * pelvis);
    const Eigen::Vector3d anchor_p0(d->mocap_pos[3 * mocap], d->mocap_pos[3 * mocap + 1], d->mocap_pos[3 * mocap + 2]);

    UpperBodyQpConfig cfg;
    cfg.feedforward = args.ff;
    cfg.w_pitch = args.w_pitch;
    if (!std::isnan(args.k_hand))
        cfg.k_hand = args.k_hand;
    if (!std::isnan(args.k_pitch))
        cfg.k_pitch = args.k_pitch;
    UpperBodyQp qp(m->nv, upper_dof, q_min, q_max, cfg);
    MujocoUpperBodyKinematics kin_backend(mc, kLoadDistance);
    UpperBodyKinematics kin(m->nv);
    UpperBodyReference ref;
    RobotState state(nj);
    Eigen::VectorXd v(m->nv), q_upper(nu), q_nominal(nu);

    // 标称姿态（t=0，尚未受重力/负载影响）的手部位置与上肢关节角，供 --ref nominal 使用
    syncControllerModel();
    kin_backend.update(dc, kin);
    UpperBodyReference ref_nominal;
    Eigen::VectorXd q_upper_nominal(nu);
    for (int s = 0; s < kNumArms; ++s)
    {
        ref_nominal.hand_pos_world[s] = kin.hand_pos_world[s];
        ref_nominal.hand_vel_world[s].setZero();
    }
    for (int k = 0; k < nu; ++k)
        q_upper_nominal(k) = d->qpos[idx[upper_joint[k]].qposadr];
    const double ff_mass = args.payload_ff ? args.payload : 0.0;

    const int decim = static_cast<int>(std::lround(cfg.dt / m->opt.timestep));
    const bool decim_ok = decim >= 1 && std::abs(decim * m->opt.timestep - cfg.dt) < 1e-12;
    if (!decim_ok)
        throw std::runtime_error("上肢控制周期不是仿真步长的整数倍");

    std::ofstream csv;
    if (!args.csv.empty())
    {
        csv.open(args.csv);
        csv << "t,anchor_dz,anchor_roll,eLx,eLy,eLz,eRx,eRy,eRz,azL,azR,solve_us,update_us,ok,max_qdes_minus_q,anchor_pitch\n";
    }

    std::ofstream dump;
    if (!args.dump_qp.empty())
        dump.open(args.dump_qp);
    dump.precision(17);
    std::ofstream dump_ctrl;
    if (!args.dump_ctrl.empty())
    {
        dump_ctrl.open(args.dump_ctrl);
        dump_ctrl.precision(17);
        // 列：t, qpos(nq), qvel(nv), q_upper, q_nominal, q_des_before, ref_pos(L,R), ref_vel(L,R), ff_mass, x, q_des_after, tau_ff
        dump_ctrl << "# nq=" << m->nq << " nv=" << m->nv << " nu=" << nu << "\n";
    }

    bool captured = false, stepped = false;
    long qp_calls = 0, qp_fail = 0;
    Stats eL_osc, eR_osc, azL_osc, azR_osc, solve_us, update_us, gap_osc;
    double anchor_roll = 0.0, anchor_pitch = 0.0;
    double settle_time = std::nan("");
    const long n_steps = std::lround(args.duration / m->opt.timestep);

    for (long step = 0; step < n_steps; ++step)
    {
        const double t = d->time;
        if (args.osc && t >= kOscStart)
        {
            const double s = t - kOscStart;
            anchor_roll = args.roll_amp * std::sin(2 * M_PI * args.roll_hz * s);
            anchor_pitch = args.pitch_amp * std::sin(2 * M_PI * args.pitch_hz * s);
            d->mocap_pos[3 * mocap + 2] = anchor_p0.z() + args.z_amp * std::sin(2 * M_PI * args.z_hz * s);
            const Eigen::Quaterniond q_anchor = Eigen::AngleAxisd(anchor_roll, Eigen::Vector3d::UnitX()) *
                                                Eigen::AngleAxisd(anchor_pitch, Eigen::Vector3d::UnitY());
            d->mocap_quat[4 * mocap + 0] = q_anchor.w();
            d->mocap_quat[4 * mocap + 1] = q_anchor.x();
            d->mocap_quat[4 * mocap + 2] = q_anchor.y();
            d->mocap_quat[4 * mocap + 3] = q_anchor.z();
        }

        bridge.readState(d, state);
        if (step % decim == 0)
        {
            mj_forward(m, d); // 保证 xpos / Jacobian / qfrc_bias 对应当前 qpos、qvel
            syncControllerModel();
            kin_backend.update(dc, kin); // 控制器侧模型（不含箱子）
            toGeneralizedVelocity(state, v);
            for (int k = 0; k < nu; ++k)
                q_upper(k) = state.q_joint(upper_joint[k]);

            if (!captured && t >= kCaptureTime)
            {
                // reference 捕获：创建者 = 本测试程序；此后只在 t=2 阶跃一次，不跟随实测
                const bool nominal = args.ref == "nominal";
                if (nominal)
                    ref = ref_nominal;
                else
                    for (int s = 0; s < kNumArms; ++s)
                    {
                        ref.hand_pos_world[s] = kin.hand_pos_world[s];
                        ref.hand_vel_world[s].setZero();
                    }
                q_nominal = nominal ? q_upper_nominal : q_upper;
                qp.reset(q_nominal);
                for (int k = 0; k < nu; ++k)
                    cmd.q_des(upper_joint[k]) = q_nominal(k);
                captured = true;
            }
            if (captured && !stepped && t >= kStepTime)
            {
                for (int s = 0; s < kNumArms; ++s)
                    ref.hand_pos_world[s].x() += args.step;
                stepped = true;
            }

            bool ok = true;
            double upd_us = 0.0;
            if (captured && args.mode == "qp")
            {
                const Eigen::VectorXd q_des_before = qp.qDes();
                const auto t0 = std::chrono::steady_clock::now();
                ok = qp.update(kin, ref, v, q_upper, q_nominal);
                upd_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                ++qp_calls;
                qp_fail += ok ? 0 : 1;
                for (int k = 0; k < nu; ++k)
                {
                    cmd.q_des(upper_joint[k]) = qp.qDes()(k);
                    cmd.dq_des(upper_joint[k]) = qp.dqDes()(k);
                }
                if (dump_ctrl.is_open() && ok)
                {
                    dump_ctrl << t;
                    for (int k = 0; k < m->nq; ++k)
                        dump_ctrl << ',' << dc->qpos[k];
                    for (int k = 0; k < m->nv; ++k)
                        dump_ctrl << ',' << dc->qvel[k];
                    const Eigen::VectorXd tau_ff_now = qp.tauFeedforward(kin, ff_mass);
                    for (const Eigen::VectorXd *vec : std::initializer_list<const Eigen::VectorXd *>{&q_upper, &q_nominal, &q_des_before})
                        for (int k = 0; k < nu; ++k)
                            dump_ctrl << ',' << (*vec)(k);
                    for (const auto *arr : {&ref.hand_pos_world, &ref.hand_vel_world})
                        for (int s = 0; s < kNumArms; ++s)
                            for (int k = 0; k < 3; ++k)
                                dump_ctrl << ',' << (*arr)[s](k);
                    dump_ctrl << ',' << ff_mass;
                    for (const Eigen::VectorXd *vec : {&qp.dqDes(), &qp.qDes(), &tau_ff_now})
                        for (int k = 0; k < nu; ++k)
                            dump_ctrl << ',' << (*vec)(k);
                    dump_ctrl << '\n';
                }
                if (dump.is_open() && ok)
                {
                    dump << t;
                    const Eigen::MatrixXd &P = qp.lastP();
                    for (int r = 0; r < nu; ++r)
                        for (int c = 0; c < nu; ++c)
                            dump << ',' << P(r, c);
                    for (const Eigen::VectorXd *vec : {&qp.lastG(), &qp.lastLo(), &qp.lastHi(), &qp.dqDes()})
                        for (int k = 0; k < nu; ++k)
                            dump << ',' << (*vec)(k);
                    dump << '\n';
                }
                solve_us.add(qp.solveMicroseconds());
                update_us.add(upd_us);
            }
            const Eigen::VectorXd tau_ff_upper = qp.tauFeedforward(kin, ff_mass);
            for (int k = 0; k < nu; ++k)
                cmd.tau_ff(upper_joint[k]) = tau_ff_upper(k);

            if (captured)
            {
                const Eigen::Vector3d eL = ref.hand_pos_world[kLeftArm] - kin.hand_pos_world[kLeftArm];
                const Eigen::Vector3d eR = ref.hand_pos_world[kRightArm] - kin.hand_pos_world[kRightArm];
                const double azL = kin.forearm_axis_world[kLeftArm].z(), azR = kin.forearm_axis_world[kRightArm].z();
                if (stepped && std::isnan(settle_time) && t < kOscStart && std::max(eL.norm(), eR.norm()) < 0.005)
                    settle_time = t - kStepTime;
                double max_gap = 0.0;
                for (int k = 0; k < nu; ++k)
                    max_gap = std::max(max_gap, std::abs(cmd.q_des(upper_joint[k]) - q_upper(k)));
                if (t >= kOscMetricStart)
                {
                    gap_osc.add(max_gap);
                    eL_osc.add(eL.norm());
                    eR_osc.add(eR.norm());
                    azL_osc.add(azL);
                    azR_osc.add(azR);
                }
                if (csv.is_open())
                {
                    const Eigen::Vector3d ap(d->mocap_pos + 3 * mocap);
                    csv << t << ',' << ap.z() - anchor_p0.z() << ',' << anchor_roll << ',' << eL.x() << ',' << eL.y() << ','
                        << eL.z() << ',' << eR.x() << ',' << eR.y() << ',' << eR.z() << ',' << azL << ',' << azR << ','
                        << (args.mode == "qp" ? qp.solveMicroseconds() : 0.0) << ',' << upd_us << ',' << ok << ','
                        << max_gap << ',' << anchor_pitch << '\n';
                }
            }
        }

        bridge.writeCommand(cmd, state, d); // PD 每个仿真步（500 Hz）用最新状态计算
        mj_step(m, d);
    }

    std::cout << "mode=" << args.mode << " ff=" << args.ff << " w_pitch=" << args.w_pitch << " pose=" << args.pose << " ref=" << args.ref
              << " payload=" << args.payload << " payload_ff=" << args.payload_ff << " pitch_amp=" << args.pitch_amp
              << " total_mass=" << total_mass << " k_hand=" << cfg.k_hand << " k_pitch=" << cfg.k_pitch << " osc=" << args.osc << " step=" << args.step
              << " z_amp=" << args.z_amp << " roll_amp=" << args.roll_amp << " duration=" << args.duration << "\n"
              << "step settle (<5 mm, both hands) [s]: " << settle_time << "\n"
              << "osc window t>=" << kOscMetricStart << ": |e_hand| RMS L/R [mm] " << 1e3 * eL_osc.rms() << " / "
              << 1e3 * eR_osc.rms() << "  max L/R [mm] " << 1e3 * eL_osc.max << " / " << 1e3 * eR_osc.max << "\n"
              << "forearm a_z RMS L/R " << azL_osc.rms() << " / " << azR_osc.rms() << "  max|a_z| L/R " << azL_osc.max
              << " / " << azR_osc.max << "  (5 deg -> 0.0872)\n"
              << "max|q_des - q| over upper joints: RMS " << gap_osc.rms() << "  max " << gap_osc.max
              << " rad  (delta_max " << cfg.delta_max << ")\n"
              << "qp calls " << qp_calls << " fail " << qp_fail << "  solve_us rms/max " << solve_us.rms() << " / "
              << solve_us.max << "  update_us rms/max " << update_us.rms() << " / " << update_us.max << "\n";
    bridge.printSaturationStats();

    if (dc != d)
    {
        mj_deleteData(dc);
        mj_deleteModel(mc);
    }
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
