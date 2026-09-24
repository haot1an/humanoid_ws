#include "humanoid_wbc/mujoco_upper_body_kinematics.hpp"
#include <stdexcept>
#include <string>

namespace humanoid_wbc
{
    namespace
    {
        using RowMajorJac = Eigen::Matrix<mjtNum, 3, Eigen::Dynamic, Eigen::RowMajor>;
        using RowMajorMatrix3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
    }

    MujocoUpperBodyKinematics::MujocoUpperBodyKinematics(const mjModel *model, double load_distance)
        : model_(model)
    {
        if (model_ == nullptr)
            throw std::runtime_error("MujocoUpperBodyKinematics: model is null");
        const std::array<const char *, kNumArms> names = {"left_elbow_link", "right_elbow_link"};
        for (int s = 0; s < kNumArms; ++s)
        {
            elbow_body_id_[s] = mj_name2id(model_, mjOBJ_BODY, names[s]);
            if (elbow_body_id_[s] < 0)
                throw std::runtime_error(std::string("MujocoUpperBodyKinematics: 找不到 body ") + names[s]);
        }
        axis_local_ = hand_local_.normalized();
        const bool load_ok = load_distance >= 0.0 && load_distance <= hand_local_.norm();
        if (!load_ok)
            throw std::runtime_error("MujocoUpperBodyKinematics: load_distance 必须在 [0, 前臂长度] 内");
        load_local_ = axis_local_ * load_distance;
        jacp_.resize(3 * model_->nv);
        jacr_.resize(3 * model_->nv);
        gravity_data_ = mj_makeData(model_);
        if (gravity_data_ == nullptr)
            throw std::runtime_error("MujocoUpperBodyKinematics: mj_makeData 失败");
    }

    MujocoUpperBodyKinematics::~MujocoUpperBodyKinematics()
    {
        mj_deleteData(gravity_data_);
    }

    void MujocoUpperBodyKinematics::update(const mjData *data, UpperBodyKinematics &kin)
    {
        if (data == nullptr)
            throw std::runtime_error("MujocoUpperBodyKinematics::update: data is null");
        const int nv = model_->nv;
        const bool sizes_ok = kin.bias_force.size() == nv && kin.J_hand_pos[0].cols() == nv;
        if (!sizes_ok)
            throw std::runtime_error("MujocoUpperBodyKinematics::update: kin 尺寸与模型 nv 不一致");

        for (int s = 0; s < kNumArms; ++s)
        {
            const int b = elbow_body_id_[s];
            Eigen::Map<const Eigen::Vector3d> p_WB(data->xpos + 3 * b);
            Eigen::Map<const RowMajorMatrix3d> R_WB(data->xmat + 9 * b);

            const Eigen::Vector3d p_hand = p_WB + R_WB * hand_local_;
            kin.hand_pos_world[s] = p_hand;
            kin.forearm_axis_world[s] = R_WB * axis_local_;

            // 点 p_hand 的线速度 Jacobian + 所在 body 的角速度 Jacobian（二者都是 world 系）
            mj_jac(model_, data, jacp_.data(), jacr_.data(), p_hand.data(), b);
            kin.J_hand_pos[s] = Eigen::Map<const RowMajorJac>(jacp_.data(), 3, nv);
            kin.J_forearm_rot[s] = Eigen::Map<const RowMajorJac>(jacr_.data(), 3, nv);

            const Eigen::Vector3d p_load = p_WB + R_WB * load_local_;
            kin.load_point_world[s] = p_load;
            mj_jac(model_, data, jacp_.data(), nullptr, p_load.data(), b);
            kin.J_load_point[s] = Eigen::Map<const RowMajorJac>(jacp_.data(), 3, nv);
        }

        // 纯重力：同一 qpos、零速度的副本上 forward，qfrc_bias = g(q)（不含 Coriolis，与 Isaac gravity_compensation_forces 一致）
        mju_copy(gravity_data_->qpos, data->qpos, model_->nq);
        mju_zero(gravity_data_->qvel, nv);
        mj_forward(model_, gravity_data_);
        kin.bias_force = Eigen::Map<const Eigen::VectorXd>(gravity_data_->qfrc_bias, nv);
    }
}
