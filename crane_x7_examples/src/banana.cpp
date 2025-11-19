// Copyright 2025 ymgchi
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <vector>

#include "angles/angles.h"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <cstdlib>
#include <unistd.h>

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("banana_sorting");

// ヘルパー関数: Poseを作成
geometry_msgs::msg::Pose createPose(
  double x, double y, double z,
  double roll_deg, double pitch_deg, double yaw_deg)
{
  geometry_msgs::msg::Pose pose;
  tf2::Quaternion q;
  
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  
  q.setRPY(
    angles::from_degrees(roll_deg),
    angles::from_degrees(pitch_deg),
    angles::from_degrees(yaw_deg)
  );
  pose.orientation = tf2::toMsg(q);
  
  return pose;
}

// Gazeboで物体をスポーン（ros_gz_simを使用）
static int spawn_count = 0;
bool spawnObjectInGazebo(double x, double y, double z)
{
  // 新しい物体をスポーン
  RCLCPP_INFO(LOGGER, "Spawning new object at (%.2f, %.2f, %.2f)", x, y, z);
  char cmd[2048];
  char name[64];
  snprintf(name, sizeof(name), "wood_cube_%d", spawn_count++);

  snprintf(cmd, sizeof(cmd),
    "ros2 run ros_gz_sim create -world default -name '%s' "
    "-x %f -y %f -z %f "
    "-string '<sdf version=\"1.6\"><model name=\"%s\">"
    "<static>false</static>"
    "<link name=\"link\">"
    "<inertial><mass>0.5</mass>"
    "<inertia><ixx>0.0002</ixx><iyy>0.0002</iyy><izz>0.0002</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>"
    "</inertial>"
    "<collision name=\"collision\"><geometry><box><size>0.05 0.05 0.05</size></box></geometry></collision>"
    "<visual name=\"visual\"><geometry><box><size>0.05 0.05 0.05</size></box></geometry>"
    "<material><ambient>0.8 0.6 0.4 1</ambient><diffuse>0.8 0.6 0.4 1</diffuse></material>"
    "</visual></link></model></sdf>'",
    name, x, y, z, name
  );

  int spawn_result = std::system(cmd);

  if (spawn_result == 0) {
    RCLCPP_INFO(LOGGER, "Object spawned successfully");
    return true;
  } else {
    RCLCPP_WARN(LOGGER, "Failed to spawn object");
    return false;
  }
}

// デカルト空間での直線軌道を計画して実行
bool executeCartesianPath(
  MoveGroupInterface & move_group,
  const geometry_msgs::msg::Pose & target_pose,
  double max_step = 0.01)
{
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target_pose);
  
  moveit_msgs::msg::RobotTrajectory trajectory;
  const double jump_threshold = 0.0;
  const double eef_step = max_step;
  
  double fraction = move_group.computeCartesianPath(
    waypoints, eef_step, jump_threshold, trajectory);
  
  if (fraction < 0.9) {
    RCLCPP_WARN(LOGGER, "Cartesian path planning failed (%.2f%% achieved)", fraction * 100.0);
    return false;
  }
  
  move_group.execute(trajectory);
  return true;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  node_options.allow_undeclared_parameters(true);
  node_options.parameter_overrides({
    {"use_sim_time", true}
  });
  
  auto move_group_arm_node = rclcpp::Node::make_shared("move_group_arm_node", node_options);
  auto move_group_gripper_node = rclcpp::Node::make_shared("move_group_gripper_node", node_options);
  
  // For current state monitor
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(move_group_arm_node);
  executor.add_node(move_group_gripper_node);
  std::thread([&executor]() {executor.spin();}).detach();

  // Wait for robot_description to be available
  RCLCPP_INFO(LOGGER, "Waiting for robot_description...");
  rclcpp::sleep_for(std::chrono::seconds(2));

  // MoveGroupの初期化
  MoveGroupInterface move_group_arm(move_group_arm_node, "arm");
  move_group_arm.setMaxVelocityScalingFactor(0.3);  // 安全のため30%に制限
  move_group_arm.setMaxAccelerationScalingFactor(0.3);

  MoveGroupInterface move_group_gripper(move_group_gripper_node, "gripper");
  move_group_gripper.setMaxVelocityScalingFactor(0.3);
  move_group_gripper.setMaxAccelerationScalingFactor(0.3);
  
  auto gripper_joint_values = move_group_gripper.getCurrentJointValues();
  const double GRIPPER_OPEN = angles::from_degrees(60.0);
  const double GRIPPER_CLOSE = angles::from_degrees(8.0);

  // 位置の定義
  auto pick_pose = createPose(0.2, 0.0, 0.11, -180, 0, -90);
  auto pick_pose_above = createPose(0.2, 0.0, 0.25, -180, 0, -90);

  std::vector<geometry_msgs::msg::Pose> place_poses = {
    createPose(0.35, 0.20, 0.12, -180, 0, -90),   // 右
    createPose(0.35, 0.0, 0.12, -180, 0, -90),    // 中央
    createPose(0.35, -0.20, 0.12, -180, 0, -90)   // 左
  };

  std::vector<geometry_msgs::msg::Pose> place_poses_above = {
    createPose(0.35, 0.20, 0.25, -180, 0, -90),
    createPose(0.35, 0.0, 0.25, -180, 0, -90),
    createPose(0.35, -0.20, 0.25, -180, 0, -90)
  };

  RCLCPP_INFO(LOGGER, "Starting banana sorting demo");

  // 初期姿勢（ホーム）に移動
  RCLCPP_INFO(LOGGER, "Moving to home position");
  move_group_arm.setNamedTarget("home");
  move_group_arm.move();

  // 3回の仕分けタスクを実行
  for (int task = 0; task < 3; task++) {
    RCLCPP_INFO(LOGGER, "=== Task %d/3 ===", task + 1);
    
    // 1. グリッパーを開く
    RCLCPP_INFO(LOGGER, "Opening gripper");
    gripper_joint_values[0] = GRIPPER_OPEN;
    move_group_gripper.setJointValueTarget(gripper_joint_values);
    move_group_gripper.move();
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // 2. ピック位置の上に移動
    RCLCPP_INFO(LOGGER, "Moving above pick position");
    move_group_arm.setPoseTarget(pick_pose_above);
    if (!move_group_arm.move()) {
      RCLCPP_ERROR(LOGGER, "Failed to move above pick position");
      continue;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // 3. 下降してピック位置へ（デカルト軌道）
    RCLCPP_INFO(LOGGER, "Descending to pick position");
    if (!executeCartesianPath(move_group_arm, pick_pose)) {
      RCLCPP_ERROR(LOGGER, "Failed to descend to pick position");
      continue;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // 4. グリッパーを閉じる
    RCLCPP_INFO(LOGGER, "Closing gripper");
    gripper_joint_values[0] = GRIPPER_CLOSE;
    move_group_gripper.setJointValueTarget(gripper_joint_values);
    move_group_gripper.move();
    rclcpp::sleep_for(std::chrono::seconds(1));

    // 5. 上昇
    RCLCPP_INFO(LOGGER, "Lifting object");
    if (!executeCartesianPath(move_group_arm, pick_pose_above)) {
      RCLCPP_ERROR(LOGGER, "Failed to lift object");
      continue;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // 6. 配置位置の上に移動
    RCLCPP_INFO(LOGGER, "Moving above place position %d", task + 1);
    move_group_arm.setPoseTarget(place_poses_above[task]);
    if (!move_group_arm.move()) {
      RCLCPP_ERROR(LOGGER, "Failed to move above place position");
      continue;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // 7. 下降して配置位置へ（デカルト軌道）
    RCLCPP_INFO(LOGGER, "Descending to place position");
    if (!executeCartesianPath(move_group_arm, place_poses[task])) {
      RCLCPP_ERROR(LOGGER, "Failed to descend to place position");
      continue;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // 8. グリッパーを開く
    RCLCPP_INFO(LOGGER, "Opening gripper to release object");
    gripper_joint_values[0] = GRIPPER_OPEN;
    move_group_gripper.setJointValueTarget(gripper_joint_values);
    move_group_gripper.move();
    rclcpp::sleep_for(std::chrono::seconds(1));

    // 9. 上昇
    RCLCPP_INFO(LOGGER, "Lifting from place position");
    if (!executeCartesianPath(move_group_arm, place_poses_above[task])) {
      RCLCPP_WARN(LOGGER, "Failed to lift from place position");
    }
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // 10. ホーム位置へ戻る
    RCLCPP_INFO(LOGGER, "Returning to home position");
    move_group_arm.setNamedTarget("home");
    move_group_arm.move();
    rclcpp::sleep_for(std::chrono::seconds(1));

    // 11. 物体を再スポーン（Gazeboのみ）
    if (task < 2) {  // 最後のタスクではリスポーン不要
      RCLCPP_INFO(LOGGER, "Spawning new object for next task");
      spawnObjectInGazebo(0.2, 0.0, 1.05);
      rclcpp::sleep_for(std::chrono::seconds(2));
    }
    
    RCLCPP_INFO(LOGGER, "Task %d completed", task + 1);
  }

  // 最終的にホーム位置に戻る
  RCLCPP_INFO(LOGGER, "Final: Moving to home position");
  move_group_arm.setNamedTarget("home");
  move_group_arm.move();

  // グリッパーを閉じる
  gripper_joint_values[0] = 0.0;
  move_group_gripper.setJointValueTarget(gripper_joint_values);
  move_group_gripper.move();

  RCLCPP_INFO(LOGGER, "Banana sorting demo completed!");

  rclcpp::shutdown();
  return 0;
}
