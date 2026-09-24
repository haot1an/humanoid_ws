//接触点定义 + 摩擦锥矩阵构造
#pragma once
#include <Eigen/Dense>
#include <string>
namespace humanoid_wbc
{
struct ContactPoint {
    std::string body_name;
    Eigen::Vector3d pos_local;
        
};
}