#pragma once
#include <array>
#include <string>

// ── 唯一的真相来源：你的规范关节顺序 ──
enum H1Joint {
    kLeftHipYaw = 0, kLeftHipRoll, kLeftHipPitch, kLeftKnee, kLeftAnkle,
    kRightHipYaw, kRightHipRoll, kRightHipPitch, kRightKnee, kRightAnkle,
    kTorso,
    kLeftShoulderPitch, kLeftShoulderRoll, kLeftShoulderYaw, kLeftElbow,
    kRightShoulderPitch, kRightShoulderRoll, kRightShoulderYaw, kRightElbow,
    kNumJoints                 // = 19，编译器自己算
};

// ── MuJoCo 后端：按名字寻址 ──
inline constexpr std::array kMjcfJointNames = {   // ← 不写尺寸，从初值列表推导
    "left_hip_yaw", "left_hip_roll","left_hip_pitch","left_knee","left_ankle",
    "right_hip_yaw", "right_hip_roll","right_hip_pitch","right_knee","right_ankle",
    "torso",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"
};
static_assert(kMjcfJointNames.size() == kNumJoints, "关节名字表长度与 enum 不一致！");

// ── 真机后端：按整数下标寻址 ──
// TODO: 数值取自 unitree_sdk2 的 H1 关节索引定义。
//       本机无 SDK，不臆造。形状为：
// inline constexpr std::array<int, kNumJoints> kUnitreeMotorIndex = { ... };
inline constexpr std::array<double, kNumJoints> kDefaultKp = {
      105.77,   // left_hip_yaw
      384.23,   // left_hip_roll
      375.51,   // left_hip_pitch
       90.96,   // left_knee
       42.05,   // left_ankle
      105.77,   // right_hip_yaw
      384.23,   // right_hip_roll
      375.51,   // right_hip_pitch
       90.96,   // right_knee
       42.05,   // right_ankle
      236.90,   // torso
      113.24,   // left_shoulder_pitch
      107.09,   // left_shoulder_roll
       51.17,   // left_shoulder_yaw
       49.24,   // left_elbow
      113.24,   // right_shoulder_pitch
      107.09,   // right_shoulder_roll
       51.17,   // right_shoulder_yaw
       49.24,   // right_elbow
};
static_assert(kDefaultKp.size() == kNumJoints, "kDefaultKp 长度与 enum 不一致！");

inline constexpr std::array<double, kNumJoints> kDefaultKd = {
        9.58,   // left_hip_yaw
       37.42,   // left_hip_roll
       36.55,   // left_hip_pitch
        8.10,   // left_knee
        3.21,   // left_ankle
        9.58,   // right_hip_yaw
       37.42,   // right_hip_roll
       36.55,   // right_hip_pitch
        8.10,   // right_knee
        3.21,   // right_ankle
       22.69,   // torso
       10.32,   // left_shoulder_pitch
        9.71,   // left_shoulder_roll
        4.12,   // left_shoulder_yaw
        3.92,   // left_elbow
       10.32,   // right_shoulder_pitch
        9.71,   // right_shoulder_roll
        4.12,   // right_shoulder_yaw
        3.92,   // right_elbow
};
static_assert(kDefaultKd.size() == kNumJoints, "kDefaultKd 长度与 enum 不一致！");
