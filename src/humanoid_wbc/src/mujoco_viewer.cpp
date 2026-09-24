#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <limits>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include "humanoid_wbc/mujoco_bridge.hpp"
#include "humanoid_wbc/joint_index.hpp"
#include "humanoid_wbc/robot_state.hpp"
#include "humanoid_wbc/h1_config.hpp"

// ── WBC：写好后取消注释 ───────────────────────────────
#include "humanoid_wbc/dynamics_data.hpp"
#include "humanoid_wbc/mujoco_dynamics.hpp"
#include "humanoid_wbc/contact_set.hpp"
#include "humanoid_wbc/wbc_controller.hpp"
#include "humanoid_wbc/osqp_solver.hpp"

// ────────────────────────────────────────────────────
#include <iomanip>
#include <cassert>

// ============================================================
// MuJoCo 全局对象
// ============================================================

mjModel *model = nullptr;
mjData *data = nullptr;

mjvCamera camera;
mjvOption option;
mjvScene scene;
mjrContext context;
// 单关节激励幅度。0.15 rad 而非 0.3 —— 后者会让手臂力矩预算达到 88%
// （见 scripts/compute_pd_gains.py 的预算校核输出）
constexpr double kStepAmplitude = 0.15;
// ============================================================
// 鼠标状态
// ============================================================

bool button_left = false;
bool button_middle = false;
bool button_right = false;

int sel_joint = 0;   // 当前激励哪个关节
bool excite = false; // 开关
double last_x = 0.0;
double last_y = 0.0;
bool request_print_error = false;
enum class ControlMode
{
    JointPD,
    Wbc
};
ControlMode mode = ControlMode::JointPD;
unsigned long target_revision = 0;

// ============================================================
// 键盘回调
// ============================================================

// 复位到 keyframe "home"（脚踩地面，基座 z=0.98）。
//
// hang=true  → 同时用 weld 把骨盆原地焊住。这是【默认】，因为在 WBC 写好之前
//              关节 PD 撑不住浮动基那 6 个自由度，松手必倒（约 0.4 秒 CoM 就
//              飘出支撑多边形，实测见 docs/learning/2026-09-07.md §6）。
// hang=false → 松手。等阶段 C 的最小 WBC 能站住了再用这个当默认。
void reset(bool hang)
{
    int key = mj_name2id(model, mjOBJ_KEY, "home");
    if (key < 0)
    {
        std::cerr << "模型里找不到 keyframe 'home'" << std::endl;
        return;
    }
    ++target_revision;
    mj_resetDataKeyframe(model, data, key);
    mj_forward(model, data); // 先算出 xpos/xquat，下面 anchor 才能对齐

    int eq = mj_name2id(model, mjOBJ_EQUALITY, "hang");
    if (eq >= 0)
    {
        if (hang)
        {
            // anchor 对齐到骨盆当前位姿 → weld 初始误差为零，不会猛拽
            int pid = mj_name2id(model, mjOBJ_BODY, "pelvis");
            if (pid < 0) throw std::runtime_error("missing pelvis body");
            mju_copy3(data->mocap_pos, data->xpos + 3 * pid);
            mju_copy4(data->mocap_quat, data->xquat + 4 * pid);
        }
        data->eq_active[eq] = hang ? 1 : 0;
    }

    mj_forward(model, data);
    std::cout << "复位：home 姿态，" << (hang ? "已冻结（H 松手）" : "已松手") << std::endl;
}

void keyboard(GLFWwindow *window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (action != GLFW_PRESS)
        return;

    // ESC：退出
    if (key == GLFW_KEY_ESCAPE)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (key == GLFW_KEY_BACKSPACE)
        reset(true);

    // ── 新增：切换激励的关节 ──
    if (key == GLFW_KEY_RIGHT_BRACKET) // ]
    {
        sel_joint = (sel_joint + 1) % kNumJoints;
        std::cout << "[" << sel_joint << "/" << kNumJoints - 1 << "] "
                  << kMjcfJointNames[sel_joint] << std::endl;
    }
    if (key == GLFW_KEY_LEFT_BRACKET) // [   注意 +kNumJoints：C++ 里 -1 % 19 == -1
    {
        sel_joint = (sel_joint - 1 + kNumJoints) % kNumJoints;
        std::cout << "[" << sel_joint << "/" << kNumJoints - 1 << "] "
                  << kMjcfJointNames[sel_joint] << std::endl;
    }
    if (key == GLFW_KEY_V)
        request_print_error = true;
    // ── 新增：开关激励 ──
    if (key == GLFW_KEY_SPACE)
    {
        excite = !excite;
        std::cout << (excite ? "激励 ON   " : "激励 OFF  ")
                  << kMjcfJointNames[sel_joint]
                  << "  " << (excite ? "+0.15 rad" : "回 home") << std::endl;
    }
    if (key == GLFW_KEY_H)
    {
        int eq = mj_name2id(model, mjOBJ_EQUALITY, "hang");
        if (eq < 0) throw std::runtime_error("missing hang equality");
        int pid = mj_name2id(model, mjOBJ_BODY, "pelvis");
        if (pid < 0) throw std::runtime_error("missing pelvis body");

        if (!data->eq_active[eq])
        {
            // 开启前：anchor 对齐到骨盆当前位姿 → 零初始误差，不猛拽
            mju_copy3(data->mocap_pos, data->xpos + 3 * pid);
            mju_copy4(data->mocap_quat, data->xquat + 4 * pid);
        }
        data->eq_active[eq] = !data->eq_active[eq];
        ++target_revision;
        std::cout << (data->eq_active[eq] ? "冻结（原地吊住）" : "松手") << std::endl;
    }

    if (key == GLFW_KEY_M)
    {
        ++target_revision;
        mode = (mode == ControlMode::JointPD) ? ControlMode::Wbc : ControlMode::JointPD;
        std::cout << "模式: " << (mode == ControlMode::JointPD ? "关节 PD" : "WBC")
                  << std::endl;
    }
}

// ============================================================
// 鼠标按键
// ============================================================

void mouse_button(GLFWwindow *window, int /*button*/, int /*action*/, int /*mods*/)
{
    button_left =
        glfwGetMouseButton(
            window,
            GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

    button_middle =
        glfwGetMouseButton(
            window,
            GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

    button_right =
        glfwGetMouseButton(
            window,
            GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    glfwGetCursorPos(
        window,
        &last_x,
        &last_y);
}

// ============================================================
// 鼠标移动
// ============================================================

void mouse_move(GLFWwindow *window, double xpos, double ypos)
{
    if (!button_left &&
        !button_middle &&
        !button_right)
    {
        return;
    }

    double dx = xpos - last_x;
    double dy = ypos - last_y;

    last_x = xpos;
    last_y = ypos;

    int width;
    int height;

    glfwGetWindowSize(
        window,
        &width,
        &height);

    bool shift =
        glfwGetKey(
            window,
            GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(
            window,
            GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

    mjtMouse action;

    if (button_right)
    {
        // 右键：平移
        action = shift
                     ? mjMOUSE_MOVE_H
                     : mjMOUSE_MOVE_V;
    }
    else if (button_left)
    {
        // 左键：旋转
        action = shift
                     ? mjMOUSE_ROTATE_H
                     : mjMOUSE_ROTATE_V;
    }
    else
    {
        // 中键：缩放
        action = mjMOUSE_ZOOM;
    }

    mjv_moveCamera(
        model,
        action,
        dx / static_cast<double>(height),
        dy / static_cast<double>(height),
        &camera);
}

// ============================================================
// 鼠标滚轮
// ============================================================
void scroll(GLFWwindow * /*window*/, double /*xoffset*/, double yoffset)
{
    mjv_moveCamera(
        model,
        mjMOUSE_ZOOM,
        0.0,
        -0.05 * yoffset,
        &camera);
}

// ============================================================
// main
// ============================================================

int main(int argc, char **argv)
{
    const bool headless = argc > 1 && std::string(argv[1]) == "--headless";
    const double duration = headless && argc > 2 ? std::stod(argv[2]) : 60.0;
    // All WBC references belong to the same entry state; home is a reset/PD target.
    const bool entry_posture = !(headless && argc > 3 && std::string(argv[3]) == "home");
    if (argc > 1 && (!headless || argc > 4 ||
        (argc > 3 && std::string(argv[3]) != "entry" && std::string(argv[3]) != "home")))
    {
        std::cerr << "Usage: mujoco_viewer [--headless [seconds [entry|home]]]\n";
        return EXIT_FAILURE;
    }
    if (!std::isfinite(duration) || duration <= 0.0)
        throw std::runtime_error("duration must be finite and positive");

    // ========================================================
    // 1. XML 路径
    // ========================================================

    const char *xml_path =
        "/home/tt/humanoid_ws/src/humanoid_wbc/model/unitree_h1/scene.xml";

    // ========================================================
    // 2. 加载 MuJoCo XML
    // ========================================================

    char error[1000] = "";

    model = mj_loadXML(
        xml_path,
        nullptr,
        error,
        sizeof(error));

    if (!model)
    {
        std::cerr
            << "Failed to load XML:\n"
            << error
            << std::endl;

        return EXIT_FAILURE;
    }

    std::cout
        << "Loaded model: "
        << xml_path
        << std::endl;

    // 打印模型信息
    std::cout
        << "nbody = " << model->nbody << '\n'
        << "ngeom = " << model->ngeom << '\n'
        << "njnt  = " << model->njnt << '\n'
        << "nq    = " << model->nq << '\n'
        << "nv    = " << model->nv << '\n'
        << "nu    = " << model->nu << '\n'
        << std::endl;

    // ========================================================
    // 3. 创建 MuJoCo 状态
    // ========================================================

    data = mj_makeData(model);

    if (!data)
    {
        std::cerr
            << "Failed to create mjData."
            << std::endl;

        mj_deleteModel(model);

        return EXIT_FAILURE;
    }

    // 计算初始状态下的位置、姿态等派生量

    reset(true);

    MujocoBridge bridge(model, {kMjcfJointNames.begin(), kMjcfJointNames.end()});
    RobotState state(bridge.numJoints());
    JointCommand cmd(bridge.numJoints());
    bridge.readState(data, state);
    const Eigen::VectorXd q_home = state.q_joint; // ← 拷一份，之后不再变
    // std::cout << state.q_joint.transpose() << std::endl;
    const double sim_dt = model->opt.timestep;
    const double ctrl_dt = 0.002;
    const int decim = std::lround(ctrl_dt / sim_dt);
    if (decim < 1 || std::abs(decim * sim_dt - ctrl_dt) > 1e-10)
        throw std::runtime_error("control period must be an integer multiple of simulation step");
    JointIndex diagnostic_index(model, {kMjcfJointNames.begin(), kMjcfJointNames.end()});
    for (int i = 0; i < diagnostic_index.size(); ++i)
        if (diagnostic_index[i].dofadr != model->nv - bridge.numJoints() + i)
            throw std::runtime_error("WBC joint DOF ordering mismatch");
    const Eigen::VectorXd tau_min = bridge.lowerControlLimits();
    const Eigen::VectorXd tau_max = bridge.upperControlLimits();
    long step_count = 0;

    // 关节 PD 的默认增益：由 scripts/compute_pd_gains.py 按各关节惯量标定
    // （Kp = I*wn^2，Kd = 2*zeta*I*wn - 模型内建 damping）。
    // 存成具名常量，因为 WBC 模式会把 cmd.kp/kd 清零，切回 PD 时必须恢复。
    const Eigen::VectorXd kp_default =
        Eigen::Map<const Eigen::VectorXd>(kDefaultKp.data(), kNumJoints);
    const Eigen::VectorXd kd_default =
        Eigen::Map<const Eigen::VectorXd>(kDefaultKd.data(), kNumJoints);

    // 浮动基占 qvel[0:6]，19 个关节 dof 紧随其后（dofadr 6..24）。
    // 下面取 qfrc_bias 的关节部分时依赖这个布局，在此把假设写成可执行的断言。
    const int nbase_dof = model->nv - bridge.numJoints();
    if (nbase_dof != 6) throw std::runtime_error("WBC requires six floating-base DOFs");

    // ══ WBC 对象：阶段 A/C 在这里构造（取消注释即可）══════════════
    // 说明：全部在循环外构造一次。矩阵尺寸在构造函数里定死，
    //      控制回路内绝不 resize（见 docs/wbc_design.md §5.3）。
    //
    std::vector<humanoid_wbc::ContactPoint> contacts;
    contacts.push_back({"left_ankle_link",
                        Eigen::Vector3d(-0.035, 0.0, -0.0695)});
    contacts.push_back({"left_ankle_link",
                        Eigen::Vector3d(0.115, 0.000, -0.0695)});
    contacts.push_back({"left_ankle_link",
                        Eigen::Vector3d(0.140, -0.030, -0.0695)});
    contacts.push_back({"left_ankle_link",
                        Eigen::Vector3d(0.140, 0.030, -0.0695)});

    contacts.push_back({"right_ankle_link",
                        Eigen::Vector3d(-0.035, 0.0, -0.0695)});
    contacts.push_back({"right_ankle_link",
                        Eigen::Vector3d(0.115, 0.000, -0.0695)});
    contacts.push_back({"right_ankle_link",
                        Eigen::Vector3d(0.140, -0.030, -0.0695)});
    contacts.push_back({"right_ankle_link",
                        Eigen::Vector3d(0.140, 0.030, -0.0695)});
    humanoid_wbc::DynamicsData dyn(model->nv, 3 * contacts.size());
    humanoid_wbc::MujocoDynamics dynamics(model, contacts);
    humanoid_wbc::WbcController wbc(
        model->nv,
        bridge.numJoints(),
        static_cast<int>(contacts.size()));

    wbc.setTorqueLimits(
        bridge.lowerControlLimits(),
        bridge.upperControlLimits());

    humanoid_wbc::OsqpSolver qp_solver(
        wbc.numVariables(),
        wbc.numConstraints());

    bool qp_initialized = false;

    bool com_target_initialized = false;
    Eigen::VectorXd q_posture_des = q_home;
    Eigen::VectorXd contact_ref_world = Eigen::VectorXd::Zero(dyn.J_c.rows());
    double next_diagnostic_time = 0.0;
    bool run_failed = false;
    long saturation_steps = 0;
    unsigned long seen_target_revision = target_revision;
    long solved_steps = 0;
    double max_mapping_error = 0.0;
    double max_clamp_error = 0.0;
    double max_solve_us = 0.0;
    double sum_solve_us = 0.0;
    double max_primal = 0.0;
    double minimum_margin = std::numeric_limits<double>::infinity();
    int min_ncon = std::numeric_limits<int>::max();
    int max_ncon = 0;
    Eigen::Vector3d com_pos_des = Eigen::Vector3d::Zero();

    Eigen::Matrix3d torso_R_des =
        Eigen::Matrix3d::Identity();
    std::cout
        << "J_c: "
        << dyn.J_c.rows()
        << " x "
        << dyn.J_c.cols()
        << '\n';
    std::cout << "M: "
              << dyn.M.rows() << " x "
              << dyn.M.cols() << '\n';

    std::cout << "h: "
              << dyn.h.size() << '\n';

    std::cout << "tau_passive: "
              << dyn.tau_passive.size() << '\n';
    double symmetry_error = (dyn.M - dyn.M.transpose()).norm();
    std::cout << "M 对称性误差: " << symmetry_error << std::endl;

    // ContactSet     contacts(model);                       // 每脚 4 点，共 nc=24
    // DynamicsData   dyn(model->nv, contacts.dim());
    // MujocoDynamics dynamics(model, contacts);
    // WbcController  wbc(model->nv, bridge.numJoints(), contacts);
    // ═══════════════════════════════════════════════════════════

    cmd.q_des = q_home;
    cmd.dq_des.setZero();
    cmd.kp = kp_default;
    cmd.kd = kd_default;
    cmd.tau_ff.setZero();

    std::cout << "q_home = " << q_home.transpose() << std::endl;
    std::cout << "按键: ] [ 切关节 | SPACE 激励 | V 打印稳态误差 | "
                 "H 冻结/松手 | M 切模式 | BACKSPACE 复位 | ESC 退出"
              << std::endl;

    // 打印 state.q_joint
    // ========================================================
    // 4. 初始化 GLFW
    // ========================================================

    GLFWwindow *window = nullptr;
    if (!headless)
    {
        if (!glfwInit())
        {
            std::cerr
                << "Failed to initialize GLFW."
                << std::endl;

            mj_deleteData(data);
            mj_deleteModel(model);

            return EXIT_FAILURE;
        }

        // ========================================================
        // 5. 创建窗口
        // ========================================================

        window = glfwCreateWindow(
            1280,
            900,
            "MuJoCo H1 C++ Viewer",
            nullptr,
            nullptr);

        if (!window)
        {
            std::cerr
                << "Failed to create GLFW window."
                << std::endl;

            glfwTerminate();

            mj_deleteData(data);
            mj_deleteModel(model);

            return EXIT_FAILURE;
        }

        // 设置当前 OpenGL context
        glfwMakeContextCurrent(window);

        // 开启垂直同步
        glfwSwapInterval(1);

        // ========================================================
        // 6. 初始化 MuJoCo Visualization
        // ========================================================

        mjv_defaultCamera(&camera);
        mjv_defaultOption(&option);
        mjv_defaultScene(&scene);
        mjr_defaultContext(&context);

        // 创建 MuJoCo scene
        mjv_makeScene(
            model,
            &scene,
            3000);

        // 创建 OpenGL rendering context
        mjr_makeContext(
            model,
            &context,
            mjFONTSCALE_150);

        // ========================================================
        // 7. 设置相机
        // ========================================================

        camera.type = mjCAMERA_FREE;

        // 看向模型中心
        camera.lookat[0] = model->stat.center[0];
        camera.lookat[1] = model->stat.center[1];
        camera.lookat[2] = model->stat.center[2];

        // 相机距离
        camera.distance =
            2.5 * model->stat.extent;

        // 水平角
        camera.azimuth = 90.0;

        // 俯视角
        camera.elevation = -20.0;

        std::cout
            << "Camera center: "
            << model->stat.center[0] << " "
            << model->stat.center[1] << " "
            << model->stat.center[2]
            << std::endl;

        std::cout
            << "Model extent: "
            << model->stat.extent
            << std::endl;

        // ========================================================
        // 8. 注册回调
        // ========================================================

        glfwSetKeyCallback(
            window,
            keyboard);

        glfwSetMouseButtonCallback(
            window,
            mouse_button);

        glfwSetCursorPosCallback(
            window,
            mouse_move);

        glfwSetScrollCallback(
            window,
            scroll);

    } // graphical initialization

    // ========================================================
    // 9. 主循环
    // ========================================================

    Eigen::VectorXd qdd_wbc =
        Eigen::VectorXd::Zero(model->nv);

    bool qdd_wbc_valid = false;

    Eigen::VectorXd fc_wbc =
        Eigen::VectorXd::Zero(
            3 * static_cast<int>(contacts.size()));

    bool fc_wbc_valid = false;

    while (headless ? data->time < duration + 2.0 && !run_failed : !glfwWindowShouldClose(window))
    {
        double frame_t0 = data->time;
        while (data->time - frame_t0 < 1.0 / 60.0)
        {
            if (headless && mode == ControlMode::JointPD && data->time >= 2.0 && !run_failed)
            {
                const int eq = mj_name2id(model, mjOBJ_EQUALITY, "hang");
                if (eq < 0) throw std::runtime_error("missing hang equality");
                data->eq_active[eq] = 0;
                mode = ControlMode::Wbc;
            }
            if (seen_target_revision != target_revision)
            {
                seen_target_revision = target_revision;
                com_target_initialized = false;
                qdd_wbc_valid = false;
                fc_wbc_valid = false;
            }
            bridge.readState(data, state);
            mj_forward(model, data);

            mj_rnePostConstraint(model, data);

            dynamics.update(data, dyn);
            if (step_count % decim == 0)
            {

                if (mode == ControlMode::JointPD)
                {
                    com_target_initialized = false;
                    qdd_wbc_valid = false;
                    fc_wbc_valid = false;

                    cmd.q_des = q_home;
                    if (excite)
                        cmd.q_des[sel_joint] += kStepAmplitude;

                    // WBC 分支会把增益清零，切回来必须恢复，否则机器人瘫软
                    cmd.kp = kp_default;
                    cmd.kd = kd_default;
                    cmd.tau_ff.setZero();
                }
                else
                {

                    if (!com_target_initialized)
                    {
                        com_pos_des = dyn.com_pos;
                        torso_R_des = dyn.torso_R;

                        q_posture_des = entry_posture ? state.q_joint : q_home;
                        contact_ref_world = dyn.contact_pos_world;
                        next_diagnostic_time = data->time;
                        std::cout << "ENTRY t=" << data->time
                                  << " home_error=" << (q_home-state.q_joint).cwiseAbs().maxCoeff()
                                  << " posture=" << (entry_posture ? "entry" : "home") << std::endl;
                        com_target_initialized = true;
                    }
                    // 1. 构建当前时刻 QP
                    wbc.buildProblem(
                        dyn,
                        com_pos_des,
                        torso_R_des,
                        state.q_joint,
                        state.dq_joint,
                        q_posture_des,
                        contact_ref_world);

                    // 2. 第一次初始化，之后只更新
                    if (!qp_initialized)
                    {
                        if (!qp_solver.initialize(
                                wbc.P(), wbc.q(),
                                wbc.A(), wbc.l(), wbc.u()))
                        {
                            throw std::runtime_error("Failed to initialize WBC QP");
                        }

                        qp_initialized = true;
                    }
                    else
                    {
                        if (!qp_solver.updateProblem(
                                wbc.P(), wbc.q(),
                                wbc.A(), wbc.l(), wbc.u()))
                        {
                            throw std::runtime_error("Failed to update WBC QP");
                        }
                    }

                    Eigen::VectorXd solution;

                    if (!qp_solver.solve(solution))
                    {
                        std::cerr << "WBC QP 求解失败，回退 JointPD" << std::endl;

                        run_failed = true;
                        fc_wbc_valid = false;
                        std::cerr << "status=" << qp_solver.info()->status
                                  << " primal=" << qp_solver.info()->pri_res
                                  << " dual=" << qp_solver.info()->dua_res
                                  << " solve_us=" << qp_solver.solveMicroseconds() << std::endl;
                        mode = ControlMode::JointPD;
                        com_target_initialized = false;
                        qdd_wbc_valid = false;

                        cmd.q_des = q_home;
                        cmd.dq_des.setZero();

                        cmd.kp = kp_default;
                        cmd.kd = kd_default;
                        cmd.tau_ff.setZero();
                    }
                    else
                    {
                        ++solved_steps;
                        max_solve_us = std::max(max_solve_us, qp_solver.solveMicroseconds());
                        sum_solve_us += qp_solver.solveMicroseconds();
                        max_primal = std::max(max_primal, static_cast<double>(qp_solver.info()->pri_res));
                        Eigen::VectorXd qdd =
                            solution.head(model->nv);

                        Eigen::VectorXd fc =
                            solution.tail(
                                3 * static_cast<int>(contacts.size()));

                        fc_wbc = fc;
                        fc_wbc_valid = true;
                        qdd_wbc = qdd;
                        qdd_wbc_valid = true;
                        Eigen::VectorXd tau =
                            wbc.computeTorque(dyn, qdd, fc);

                        cmd.q_des = state.q_joint;
                        cmd.dq_des = state.dq_joint;

                        cmd.kp.setZero();
                        cmd.kd.setZero();

                        cmd.tau_ff = tau;
                    }


                }
            }

            bridge.writeCommand(cmd, state, data);
            mj_forward(model, data);
            if (mode == ControlMode::Wbc && qdd_wbc_valid && fc_wbc_valid && step_count % decim == 0)
            {
                double clamp_error = 0.0;
                double mapping_error = 0.0;
                Eigen::VectorXd actuator_qp = Eigen::VectorXd::Zero(model->nv);
                for (int i = 0; i < diagnostic_index.size(); ++i)
                {
                    const auto &e = diagnostic_index[i];
                    actuator_qp[e.dofadr] = cmd.tau_ff[i];
                    clamp_error = std::max(clamp_error, std::abs(data->ctrl[e.actid] - cmd.tau_ff[i]));
                    mapping_error = std::max(mapping_error, std::abs(data->qfrc_actuator[e.dofadr] - data->ctrl[e.actid]));
                }
                if (clamp_error > 0.0) ++saturation_steps;
                max_clamp_error = std::max(max_clamp_error, clamp_error);
                max_mapping_error = std::max(max_mapping_error, mapping_error);
                minimum_margin = std::min(minimum_margin,
                    std::min((cmd.tau_ff-tau_min).minCoeff(), (tau_max-cmd.tau_ff).minCoeff()));
                min_ncon = std::min(min_ncon, data->ncon);
                max_ncon = std::max(max_ncon, data->ncon);
                if (data->time + 1e-9 >= next_diagnostic_time)
                {
                    next_diagnostic_time = data->time + 1.0;
                    const auto vec = [&](const mjtNum *v) { return Eigen::Map<const Eigen::VectorXd>(v, model->nv); };
                    int noncontact_rows = 0;
                    for (int row = 0; row < data->nefc; ++row)
                        if (data->efc_type[row] < mjCNSTR_CONTACT_FRICTIONLESS) ++noncontact_rows;
                    const auto contact_decomposition = dyn.J_c.completeOrthogonalDecomposition();
                    const Eigen::VectorXd incompatible_jdot = dyn.Jdot_c_v - dyn.J_c * contact_decomposition.solve(dyn.Jdot_c_v);
                    const Eigen::VectorXd qdd_error = vec(data->qacc) - qdd_wbc;
                    const Eigen::VectorXd contact_qp = dyn.J_c.transpose() * fc_wbc;
                    const Eigen::VectorXd delta_contact = vec(data->qfrc_constraint) - contact_qp;
                    const Eigen::VectorXd delta_actuator = vec(data->qfrc_actuator) - actuator_qp;
                    const Eigen::VectorXd delta_passive = vec(data->qfrc_passive) - dyn.tau_passive;
                    const Eigen::VectorXd qp_residual = dyn.M*qdd_wbc + dyn.h - dyn.tau_passive - actuator_qp - contact_qp;
                    // qfrc_smooth includes qfrc_applied and projected xfrc_applied.
                    const Eigen::VectorXd external = vec(data->qfrc_smooth) - vec(data->qfrc_passive) + vec(data->qfrc_bias) - vec(data->qfrc_actuator);
                    const Eigen::VectorXd lhs = dyn.M * qdd_error;
                    const Eigen::VectorXd rhs = delta_contact + delta_actuator + delta_passive + external - qp_residual;
                    const Eigen::VectorXd posture_error = q_posture_des - state.q_joint;
                    const Eigen::VectorXd raw_des = 36.0*posture_error - 12.0*state.dq_joint;
                    Eigen::Index j, dof, des_joint;
                    raw_des.cwiseAbs().maxCoeff(&des_joint);
                    posture_error.cwiseAbs().maxCoeff(&j);
                    qdd_error.cwiseAbs().maxCoeff(&dof);
                    const auto &e = diagnostic_index[static_cast<int>(j)];
                    const Eigen::Matrix3d R_error = 0.5*(torso_R_des*dyn.torso_R.transpose()-dyn.torso_R*torso_R_des.transpose());
                    const Eigen::Vector3d torso_error(R_error(2,1), R_error(0,2), R_error(1,0));
                    std::cout << std::setprecision(10)
                        << "DIAG t=" << data->time << " joint=" << j << " name=" << e.name
                        << " err=" << posture_error[j] << " q_des=" << q_posture_des[j] << " q=" << state.q_joint[j]
                        << " dq=" << state.dq_joint[j] << " qdd_des=" << std::clamp(raw_des[j],-20.0,20.0)
                        << " des_max=" << raw_des.cwiseMax(-20.0).cwiseMin(20.0).cwiseAbs().maxCoeff()
                        << " des_raw_max=" << raw_des.cwiseAbs().maxCoeff()
                        << " des_joint=" << des_joint << " des_name=" << diagnostic_index[static_cast<int>(des_joint)].name
                        << " qdd_qp=" << qdd_wbc[e.dofadr] << " tau=" << cmd.tau_ff[j]
                        << " ctrl=" << data->ctrl[e.actid] << " actuator=" << data->qfrc_actuator[e.dofadr]
                        << " tau_min=" << tau_min[j] << " tau_max=" << tau_max[j]
                        << " margin=" << std::min(cmd.tau_ff[j]-tau_min[j],tau_max[j]-cmd.tau_ff[j])
                        << " clamp=" << clamp_error << " mapping=" << mapping_error << " saturation_steps=" << saturation_steps
                        << " com=" << (com_pos_des-dyn.com_pos).norm() << " torso=" << torso_error.norm()
                        << " contact_pos=" << (dyn.contact_pos_world-contact_ref_world).cwiseAbs().maxCoeff()
                        << " contact_vel=" << (dyn.J_c*vec(data->qvel)).cwiseAbs().maxCoeff()
                        << " ncon=" << data->ncon << " noncontact_rows=" << noncontact_rows
                        << " contact_qp=" << contact_qp.norm() << " contact_mj=" << vec(data->qfrc_constraint).norm()
                        << " contact_delta=" << delta_contact.norm()
                        << " dof=" << dof << " dof_name=" << (dof < 6 ? "floating_base" : diagnostic_index[static_cast<int>(dof)-6].name)
                        << " qdd_delta=" << qdd_error[dof]
                        << " M_delta=" << lhs.norm() << " actuator_delta=" << delta_actuator.norm()
                        << " passive_delta=" << delta_passive.norm() << " external=" << external.norm()
                        << " qp_dyn=" << qp_residual.norm() << " balance=" << (lhs-rhs).norm()
                        << " contact_rank=" << contact_decomposition.rank()
                        << " incompatible_jdot=" << incompatible_jdot.cwiseAbs().maxCoeff()
                        << " contact_acc=" << (dyn.J_c*qdd_wbc+dyn.Jdot_c_v).cwiseAbs().maxCoeff()
                        << " contact_eq=" << (dyn.J_c*qdd_wbc-wbc.l().head(dyn.J_c.rows())).cwiseAbs().maxCoeff()
                        << " status=" << qp_solver.info()->status_val << " primal=" << qp_solver.info()->pri_res
                        << " dual=" << qp_solver.info()->dua_res << " iter=" << qp_solver.info()->iter
                        << " solve_us=" << qp_solver.solveMicroseconds() << std::endl;
                    std::cout << "DOF_TERMS t=" << data->time << " dof=" << dof
                        << " mass=" << lhs[dof] << " contact=" << delta_contact[dof]
                        << " actuator=" << delta_actuator[dof] << " passive=" << delta_passive[dof]
                        << " external=" << external[dof] << " qp_residual=" << qp_residual[dof] << std::endl;
                }
            }
            mj_step(model, data);

            ++step_count;
            if (headless && (run_failed || data->time >= duration + 2.0)) break;
        }

        if (headless) continue;

        if (request_print_error)
        {
            request_print_error = false;

            const Eigen::VectorXd e =
                (mode == ControlMode::Wbc && com_target_initialized ? q_posture_des : cmd.q_des) - state.q_joint;

            std::cout << "\n── 稳态误差 e = q_des - q ──\n"
                      << "最大关节速度 = " << state.dq_joint.cwiseAbs().maxCoeff()
                      << "  (应 < 1e-4，否则还没稳)\n"
                      << "接触数 = " << data->ncon << "\n";

            for (int i = 0; i < kNumJoints; ++i)
                std::cout << "  " << std::setw(22) << std::left << kMjcfJointNames[i]
                          << std::setw(12) << std::right << e[i] << "\n";
        }

        mjrRect viewport{
            0,
            0,
            0,
            0};

        glfwGetFramebufferSize(
            window,
            &viewport.width,
            &viewport.height);

        // ----------------------------------------------------
        // 更新 MuJoCo 场景
        // ----------------------------------------------------

        mjv_updateScene(
            model,
            data,
            &option,
            nullptr,
            &camera,
            mjCAT_ALL,
            &scene);

        // ----------------------------------------------------
        // 渲染
        // ----------------------------------------------------

        mjr_render(
            viewport,
            &scene,
            &context);

        // OpenGL 双缓冲交换
        glfwSwapBuffers(window);

        // 处理键盘 / 鼠标事件
        glfwPollEvents();
    }

    // ========================================================
    // 10. 释放资源
    // ========================================================

    if (!headless)
    {
        mjv_freeScene(&scene);
        mjr_freeContext(&context);

        glfwDestroyWindow(window);
        glfwTerminate();
    }
    std::cout << "SUMMARY simulated=" << data->time << " solved_steps=" << solved_steps
              << " failed=" << run_failed << " saturation_steps=" << saturation_steps
              << " max_clamp=" << max_clamp_error << " max_mapping=" << max_mapping_error
              << " min_margin=" << minimum_margin << " min_ncon=" << min_ncon << " max_ncon=" << max_ncon
              << " max_primal=" << max_primal << " max_solve_us=" << max_solve_us
              << " mean_solve_us=" << (solved_steps ? sum_solve_us / solved_steps : 0.0) << std::endl;
    mj_deleteData(data);
    mj_deleteModel(model);

    return run_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}