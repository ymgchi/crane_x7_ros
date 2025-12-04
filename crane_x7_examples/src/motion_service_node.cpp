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

/**
 * Motion Service Node for CRANE-X7
 *
 * Provides ROS 2 services for robot motion control using MoveIt.
 * Designed to be called from Python nodes for pick-and-place operations.
 *
 * Services:
 *   /motion/move_to_camera_pose (std_srvs/Trigger)
 *   /motion/open_gripper (std_srvs/Trigger)
 *   /motion/close_gripper (std_srvs/Trigger)
 *   /motion/move_to_pose (geometry_msgs/PoseStamped via topic + Trigger)
 */

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "angles/angles.h"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

class MotionServiceNode : public rclcpp::Node
{
public:
  MotionServiceNode(
    const std::shared_ptr<MoveGroupInterface> & arm,
    const std::shared_ptr<MoveGroupInterface> & gripper)
  : Node("motion_service_node"),
    move_group_arm_(arm),
    move_group_gripper_(gripper)
  {
    // Initialize camera pose joints (from color_sorting.cpp)
    camera_pose_joints_ = {
      angles::from_degrees(0.0),
      angles::from_degrees(60.0),
      angles::from_degrees(0.0),
      angles::from_degrees(-120.0),
      angles::from_degrees(0.0),
      angles::from_degrees(-50.0),
      angles::from_degrees(90.0)
    };

    // Gripper angles
    gripper_open_ = angles::from_degrees(60.0);
    gripper_close_ = angles::from_degrees(20.0);

    // Create services
    move_to_camera_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/motion/move_to_camera_pose",
      std::bind(&MotionServiceNode::moveToCameraPoseCallback, this,
                std::placeholders::_1, std::placeholders::_2));

    open_gripper_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/motion/open_gripper",
      std::bind(&MotionServiceNode::openGripperCallback, this,
                std::placeholders::_1, std::placeholders::_2));

    close_gripper_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/motion/close_gripper",
      std::bind(&MotionServiceNode::closeGripperCallback, this,
                std::placeholders::_1, std::placeholders::_2));

    execute_pose_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/motion/execute_pose",
      std::bind(&MotionServiceNode::executePoseCallback, this,
                std::placeholders::_1, std::placeholders::_2));

    execute_cartesian_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/motion/execute_cartesian",
      std::bind(&MotionServiceNode::executeCartesianCallback, this,
                std::placeholders::_1, std::placeholders::_2));

    // Target pose subscriber (set target before calling execute_pose)
    target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/motion/target_pose", 10,
      std::bind(&MotionServiceNode::targetPoseCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Motion Service Node initialized");
    RCLCPP_INFO(this->get_logger(), "Services:");
    RCLCPP_INFO(this->get_logger(), "  /motion/move_to_camera_pose");
    RCLCPP_INFO(this->get_logger(), "  /motion/open_gripper");
    RCLCPP_INFO(this->get_logger(), "  /motion/close_gripper");
    RCLCPP_INFO(this->get_logger(), "  /motion/execute_pose");
    RCLCPP_INFO(this->get_logger(), "  /motion/execute_cartesian");
  }

private:
  void moveToCameraPoseCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    RCLCPP_INFO(this->get_logger(), "Moving to camera observation pose (scan pose)");

    // Use the same scan pose as color_sorting.cpp: createPose(0.25, 0.00, 0.35, -180, 0, 90)
    geometry_msgs::msg::Pose scan_pose;
    scan_pose.position.x = 0.25;
    scan_pose.position.y = 0.00;
    scan_pose.position.z = 0.35;

    // Orientation: roll=-180, pitch=0, yaw=90 (degrees)
    // Convert to quaternion
    double roll = angles::from_degrees(-180.0);
    double pitch = angles::from_degrees(0.0);
    double yaw = angles::from_degrees(90.0);

    double cy = std::cos(yaw * 0.5);
    double sy = std::sin(yaw * 0.5);
    double cp = std::cos(pitch * 0.5);
    double sp = std::sin(pitch * 0.5);
    double cr = std::cos(roll * 0.5);
    double sr = std::sin(roll * 0.5);

    scan_pose.orientation.w = cr * cp * cy + sr * sp * sy;
    scan_pose.orientation.x = sr * cp * cy - cr * sp * sy;
    scan_pose.orientation.y = cr * sp * cy + sr * cp * sy;
    scan_pose.orientation.z = cr * cp * sy - sr * sp * cy;

    move_group_arm_->setStartStateToCurrentState();
    move_group_arm_->setPoseTarget(scan_pose);

    bool success = (move_group_arm_->move() == moveit::core::MoveItErrorCode::SUCCESS);

    response->success = success;
    response->message = success ? "Moved to camera pose" : "Failed to move to camera pose";

    RCLCPP_INFO(this->get_logger(), "Move to camera pose: %s", response->message.c_str());
  }

  void openGripperCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    RCLCPP_INFO(this->get_logger(), "Opening gripper");

    // Get current joint values and set target (same pattern as banana.cpp)
    auto gripper_joint_values = move_group_gripper_->getCurrentJointValues();
    gripper_joint_values[0] = gripper_open_;
    move_group_gripper_->setJointValueTarget(gripper_joint_values);

    bool success = (move_group_gripper_->move() == moveit::core::MoveItErrorCode::SUCCESS);

    response->success = success;
    response->message = success ? "Gripper opened" : "Failed to open gripper";

    RCLCPP_INFO(this->get_logger(), "Open gripper: %s", response->message.c_str());
  }

  void closeGripperCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    RCLCPP_INFO(this->get_logger(), "Closing gripper");

    // Get current joint values and set target (same pattern as banana.cpp)
    auto gripper_joint_values = move_group_gripper_->getCurrentJointValues();
    gripper_joint_values[0] = gripper_close_;
    move_group_gripper_->setJointValueTarget(gripper_joint_values);

    bool success = (move_group_gripper_->move() == moveit::core::MoveItErrorCode::SUCCESS);

    response->success = success;
    response->message = success ? "Gripper closed" : "Failed to close gripper";

    RCLCPP_INFO(this->get_logger(), "Close gripper: %s", response->message.c_str());
  }

  void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    target_pose_ = *msg;
    target_pose_received_ = true;
    RCLCPP_DEBUG(this->get_logger(), "Target pose received: (%.3f, %.3f, %.3f)",
                 msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  }

  void executePoseCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!target_pose_received_) {
      response->success = false;
      response->message = "No target pose received";
      RCLCPP_WARN(this->get_logger(), "Execute pose failed: no target pose");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Executing pose: (%.3f, %.3f, %.3f)",
                target_pose_.pose.position.x,
                target_pose_.pose.position.y,
                target_pose_.pose.position.z);

    move_group_arm_->setStartStateToCurrentState();
    move_group_arm_->setPoseTarget(target_pose_.pose);

    bool success = (move_group_arm_->move() == moveit::core::MoveItErrorCode::SUCCESS);

    response->success = success;
    response->message = success ? "Pose executed" : "Failed to execute pose";
    target_pose_received_ = false;

    RCLCPP_INFO(this->get_logger(), "Execute pose: %s", response->message.c_str());
  }

  void executeCartesianCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!target_pose_received_) {
      response->success = false;
      response->message = "No target pose received";
      RCLCPP_WARN(this->get_logger(), "Execute cartesian failed: no target pose");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Executing cartesian path to: (%.3f, %.3f, %.3f)",
                target_pose_.pose.position.x,
                target_pose_.pose.position.y,
                target_pose_.pose.position.z);

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(target_pose_.pose);

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;
    const double eef_step = 0.01;

    double fraction = move_group_arm_->computeCartesianPath(
      waypoints, eef_step, jump_threshold, trajectory);

    bool success = false;
    if (fraction >= 0.9) {
      auto exec_result = move_group_arm_->execute(trajectory);
      success = (exec_result == moveit::core::MoveItErrorCode::SUCCESS);
    }

    response->success = success;
    response->message = success ?
      "Cartesian path executed" :
      "Failed to execute cartesian path (fraction: " + std::to_string(fraction) + ")";
    target_pose_received_ = false;

    RCLCPP_INFO(this->get_logger(), "Execute cartesian: %s", response->message.c_str());
  }

  std::shared_ptr<MoveGroupInterface> move_group_arm_;
  std::shared_ptr<MoveGroupInterface> move_group_gripper_;

  std::vector<double> camera_pose_joints_;
  double gripper_open_;
  double gripper_close_;

  geometry_msgs::msg::PoseStamped target_pose_;
  bool target_pose_received_ = false;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr move_to_camera_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr open_gripper_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr close_gripper_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr execute_pose_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr execute_cartesian_srv_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);

  // Create nodes for MoveGroupInterface
  auto arm_node = rclcpp::Node::make_shared("motion_service_arm_node", node_options);
  auto gripper_node = rclcpp::Node::make_shared("motion_service_gripper_node", node_options);

  // Multi-threaded executor
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(arm_node);
  executor.add_node(gripper_node);

  // Spin in background for MoveGroupInterface initialization
  std::thread executor_thread([&executor]() {
    executor.spin();
  });

  // Create MoveGroupInterfaces
  RCLCPP_INFO(arm_node->get_logger(), "Creating MoveGroupInterface for arm...");
  auto move_group_arm = std::make_shared<MoveGroupInterface>(arm_node, "arm");
  move_group_arm->setMaxVelocityScalingFactor(0.7);
  move_group_arm->setMaxAccelerationScalingFactor(0.7);
  move_group_arm->setPlanningTime(20.0);
  move_group_arm->setGoalPositionTolerance(0.01);
  move_group_arm->setGoalOrientationTolerance(0.05);
  move_group_arm->allowReplanning(true);

  RCLCPP_INFO(arm_node->get_logger(), "Creating MoveGroupInterface for gripper...");
  auto move_group_gripper = std::make_shared<MoveGroupInterface>(gripper_node, "gripper");
  move_group_gripper->setMaxVelocityScalingFactor(0.7);
  move_group_gripper->setMaxAccelerationScalingFactor(0.7);
  move_group_gripper->setPlanningTime(10.0);
  move_group_gripper->allowReplanning(true);

  // Wait for robot state
  RCLCPP_INFO(arm_node->get_logger(), "Waiting for robot state...");
  int wait_count = 0;
  while (!move_group_arm->getCurrentState() && wait_count < 60) {
    rclcpp::sleep_for(std::chrono::milliseconds(500));
    wait_count++;
  }

  if (!move_group_arm->getCurrentState()) {
    RCLCPP_ERROR(arm_node->get_logger(), "Failed to get robot state after 30 seconds");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(arm_node->get_logger(), "Robot state available, creating service node...");

  // Create service node
  auto service_node = std::make_shared<MotionServiceNode>(move_group_arm, move_group_gripper);
  executor.add_node(service_node);

  RCLCPP_INFO(arm_node->get_logger(), "Motion Service Node ready");

  // Wait for executor thread
  executor_thread.join();

  rclcpp::shutdown();
  return 0;
}
