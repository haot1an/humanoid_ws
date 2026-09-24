cmake_minimum_required(VERSION 3.8)
project(humanoid_wbc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 导出 compile_commands.json，供 clangd / IDE 解析头文件路径
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 让编译器替你把关（-Wshadow 抓过变量遮蔽）
add_compile_options(-Wall -Wextra -Wshadow)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)

find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Eigen3 REQUIRED)
add_library(wbc_core 
    src/joint_index.cpp  
    src/mujoco_bridge.cpp 
    src/mujoco_dynamics.cpp
    src/dynamics_data.cpp
    src/wbc_controller.cpp
    src/osqp_solver.cpp
    src/upper_body_qp.cpp
    src/mujoco_upper_body_kinematics.cpp)
target_include_directories(wbc_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(
    wbc_core
    PUBLIC
    Eigen3::Eigen
    mujoco::mujoco
    osqp::osqp
    OsqpEigen::OsqpEigen
)

add_executable(
    mujoco_viewer
    src/mujoco_viewer.cpp
)

target_link_libraries(
    mujoco_viewer
    mujoco::mujoco
    glfw
    wbc_core
    Eigen3::Eigen
)

ament_target_dependencies(
    mujoco_viewer
    rclcpp
)

# 上肢 QP 吊挂验证（无 GUI），见 docs/DESIGN.md UBQP-001
add_executable(
    upper_body_qp_test
    src/upper_body_qp_test.cpp
)

target_link_libraries(
    upper_body_qp_test
    mujoco::mujoco
    wbc_core
    Eigen3::Eigen
)

# 上肢 QP 任务行的有限差分核对
add_executable(
    upper_body_jacobian_check
    src/upper_body_jacobian_check.cpp
)

target_link_libraries(
    upper_body_jacobian_check
    mujoco::mujoco
    wbc_core
    Eigen3::Eigen
)

install(
    TARGETS mujoco_viewer upper_body_qp_test upper_body_jacobian_check
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()