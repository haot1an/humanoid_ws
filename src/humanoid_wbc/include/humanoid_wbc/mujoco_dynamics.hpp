// 填充者，依赖 MuJoCo   ← 防火墙的 MuJoCo 侧
#pragma once
#include <mujoco/mujoco.h>
#include "humanoid_wbc/dynamics_data.hpp"
#include <vector>
#include <Eigen/Dense>
#include "humanoid_wbc/contact_set.hpp"
namespace humanoid_wbc
{
    class MujocoDynamics
    {
    public:
        MujocoDynamics(const mjModel *model,const std::vector<ContactPoint>& contacts);
        void update(
            mjData *data,
            DynamicsData &dyn);

    private:
        const mjModel *model_;
        std::vector<mjtNum> M_buffer_;
        int torso_body_id_ = -1;
        int com_root_body_id_ = -1;

        struct ResolvedContact
        {
            int body_id;
            Eigen::Vector3d pos_local;
            
        };
        std::vector<ResolvedContact> contacts_;
        std::vector<mjtNum> jac_buffer_;
    };
}