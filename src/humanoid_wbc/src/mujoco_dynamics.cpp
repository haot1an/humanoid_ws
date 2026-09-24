#include "humanoid_wbc/mujoco_dynamics.hpp"
#include <stdexcept>

namespace humanoid_wbc
{

    MujocoDynamics::MujocoDynamics(const mjModel *model, const std::vector<ContactPoint> &contacts)
        : model_(model)
    {
        if (model_ == nullptr)
            throw std::runtime_error("MujocoDynamics: model is null");
        torso_body_id_ = mj_name2id(model_, mjOBJ_BODY, "torso_link");
        if (torso_body_id_ < 0)
            throw std::runtime_error("Cannot find torso_link");
        com_root_body_id_ = mj_name2id(model_, mjOBJ_BODY, "pelvis");
        if (com_root_body_id_ < 0 || model_->body_subtreemass[com_root_body_id_] <= 0)
            throw std::runtime_error("Cannot find positive-mass pelvis subtree");

        M_buffer_.resize(model_->nv * model_->nv);
        jac_buffer_.resize(3 * model_->nv);
        contacts_.reserve(contacts.size());
        for (const auto &contact : contacts)
        {
            int body_id = mj_name2id(
                model_,
                mjOBJ_BODY,
                contact.body_name.c_str());

            if (body_id < 0)
            {
                throw std::runtime_error("MujocoDynamics: invalid body name");
            }

            // 把 body_id 和 pos_local 存进 contacts_
            contacts_.push_back(ResolvedContact{body_id, contact.pos_local});
        }


    }

    using RowMajorMatrixXd =
        Eigen::Matrix<
            mjtNum,
            Eigen::Dynamic,
            Eigen::Dynamic,
            Eigen::RowMajor>;

    using RowMajorMatrix3d =
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

    void MujocoDynamics::update(
        mjData *data,
        DynamicsData &dyn)
    {

        if (data == nullptr)
        {
            throw std::runtime_error("MujocoDynamics::update: data is null");
        }
        if (dyn.M.rows() != model_->nv || dyn.M.cols() != model_->nv)
        {
            throw std::runtime_error("MujocoDynamics::update: M has incorrect dimensions");
        }

        if (dyn.h.size() != model_->nv)
        {
            throw std::runtime_error("MujocoDynamics::update: h has incorrect dimensions");
        }

        if (dyn.tau_passive.size() != model_->nv)
        {
            throw std::runtime_error("MujocoDynamics::update: tau_passive has incorrect dimensions");
        }

        mj_fullM(model_, data, M_buffer_.data());

        for (std::size_t i = 0; i < contacts_.size(); ++i)
        {
            const auto &contact = contacts_[i];
            // pW​=pWB​+RWB​pB​

            // body 世界位置
            Eigen::Map<const Eigen::Vector3d> p_WB(
                data->xpos + 3 * contact.body_id);

            // body 世界旋转
            Eigen::Map<const RowMajorMatrix3d> R_WB(
                data->xmat + 9 * contact.body_id);

            // local -> world
            Eigen::Vector3d p_W =
                p_WB + R_WB * contact.pos_local;
            dyn.contact_pos_world.segment<3>(3 * i) = p_W;
            mj_jac(
                model_,
                data,
                jac_buffer_.data(),
                nullptr,
                p_W.data(),
                contact.body_id);
            Eigen::Map<const RowMajorMatrixXd> J_c_map(
                jac_buffer_.data(),
                3,
                model_->nv);
            dyn.J_c.block(3 * i, 0, 3, model_->nv) = J_c_map;

            mj_jacDot(
                model_,
                data,
                jac_buffer_.data(),
                nullptr,
                p_W.data(),
                contact.body_id);

            Eigen::Map<const RowMajorMatrixXd> J_c_dot_map(
                jac_buffer_.data(),
                3,
                model_->nv);
            Eigen::Vector3d J_c_dot_map_q = J_c_dot_map * Eigen::Map<const Eigen::VectorXd>(data->qvel, model_->nv);
            dyn.Jdot_c_v.segment<3>(3 * i) = J_c_dot_map_q;

        }

        Eigen::Map<const Eigen::VectorXd> qvel_map(
            data->qvel,
            model_->nv);

        dyn.contact_vel_world =
            dyn.J_c * qvel_map;

        Eigen::Map<const RowMajorMatrixXd> M_map(
            M_buffer_.data(),
            model_->nv,
            model_->nv);

        dyn.M = M_map;
        Eigen::Map<const Eigen::VectorXd> h_map(
            data->qfrc_bias,
            model_->nv);

        dyn.h = h_map;
        Eigen::Map<const Eigen::VectorXd> tau_passive_map(
            data->qfrc_passive,
            model_->nv);

        dyn.tau_passive = tau_passive_map;

        Eigen::Map<const Eigen::Vector3d> com_pos_map(
            data->subtree_com + 3 * com_root_body_id_);

        dyn.com_pos = com_pos_map;

        mj_jacSubtreeCom(
            model_,
            data,
            jac_buffer_.data(),
            com_root_body_id_);

        Eigen::Map<const RowMajorMatrixXd> J_com_map(
            jac_buffer_.data(),
            3,
            model_->nv);

        dyn.J_com = J_com_map;

        Eigen::Map<const Eigen::VectorXd> qvel(
            data->qvel,
            model_->nv);

        dyn.com_vel =
            dyn.J_com * qvel;

        // Same mass-weighted subtree sum as mj_jacSubtreeCom, differentiated.
        dyn.Jdot_com_v.setZero();
        for (int body = com_root_body_id_; body < model_->nbody; ++body)
        {
            if (body > com_root_body_id_ && model_->body_parentid[body] < com_root_body_id_)
                break;
            mj_jacDot(model_, data, jac_buffer_.data(), nullptr, data->xipos + 3 * body, body);
            Eigen::Map<const RowMajorMatrixXd> Jdot_body(jac_buffer_.data(), 3, model_->nv);
            dyn.Jdot_com_v += model_->body_mass[body] * (Jdot_body * qvel);
        }
        dyn.Jdot_com_v /= model_->body_subtreemass[com_root_body_id_];

        Eigen::Map<
            const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>
            R_torso(
                data->xmat + 9 * torso_body_id_);

        dyn.torso_R = R_torso;

        mjtNum torso_vel[6];

        mj_objectVelocity(
            model_,
            data,
            mjOBJ_BODY,
            torso_body_id_,
            torso_vel,
            0);

        dyn.torso_omega << torso_vel[0],
            torso_vel[1],
            torso_vel[2];
        Eigen::Matrix<
            double,
            3,
            Eigen::Dynamic,
            Eigen::RowMajor>
            J_torso_rot(3, model_->nv);

        J_torso_rot.setZero();

        mj_jacBody(
            model_,
            data,
            nullptr,            // 不需要线速度 Jacobian
            J_torso_rot.data(), // rotational Jacobian
            torso_body_id_);

        dyn.J_torso_rot = J_torso_rot;

        // Eigen::Map<const Eigen::VectorXd>
        //     qvel(data->qvel, model_->nv);

        mjtNum torso_acc[6];

        mj_objectAcceleration(
            model_,
            data,
            mjOBJ_BODY,
            torso_body_id_,
            torso_acc,
            0);

        Eigen::Vector3d torso_alpha(
            torso_acc[0],
            torso_acc[1],
            torso_acc[2]);
        Eigen::Map<const Eigen::VectorXd> qacc_torso(
            data->qacc,
            model_->nv);

        dyn.Jdot_torso_rot_v =
            torso_alpha - dyn.J_torso_rot * qacc_torso;


    }

}