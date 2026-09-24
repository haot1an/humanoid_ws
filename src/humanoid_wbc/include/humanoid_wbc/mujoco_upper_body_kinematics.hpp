// UpperBodyKinematics 的填充者，依赖 MuJoCo   ← 防火墙的 MuJoCo 侧
#pragma once
#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <array>
#include <vector>
#include "humanoid_wbc/upper_body_types.hpp"

namespace humanoid_wbc
{
    class MujocoUpperBodyKinematics
    {
    public:
        // load_distance：负载作用点在前臂轴上距肘关节的距离 [m]（箱子压在前臂上的位置，DEC-003 d）
        explicit MujocoUpperBodyKinematics(const mjModel *model, double load_distance = 0.15);
        ~MujocoUpperBodyKinematics();
        MujocoUpperBodyKinematics(const MujocoUpperBodyKinematics &) = delete;
        MujocoUpperBodyKinematics &operator=(const MujocoUpperBodyKinematics &) = delete;

        // 需要 d 已经过 mj_forward / mj_step（xpos、xmat 为当前状态）；bias_force 在内部零速度副本上计算
        void update(const mjData *data, UpperBodyKinematics &kin);

    private:
        const mjModel *model_;
        std::array<int, kNumArms> elbow_body_id_{-1, -1};
        // elbow_link 系下：手部点 = 前臂末端碰撞球心，h1.xml:178 <geom type="sphere" pos="0.28 0 -0.015">
        Eigen::Vector3d hand_local_{0.28, 0.0, -0.015};
        Eigen::Vector3d axis_local_; // hand_local_ 单位化：前臂轴（肘关节原点 → 手部点）
        Eigen::Vector3d load_local_; // axis_local_ * load_distance
        std::vector<mjtNum> jacp_, jacr_;
        mjData *gravity_data_ = nullptr; // 零速度副本：mj_forward 后 qfrc_bias = g(q)
    };
}
