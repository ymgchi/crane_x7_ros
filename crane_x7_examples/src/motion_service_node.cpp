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
#include "moveit_msgs/msg/constraints.hpp"
#include "moveit_msgs/msg/joint_constraint.hpp"
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
    // Joint names for CRANE-X7 (7-DOF arm)
    joint_names_ = {
      "crane_x7_shoulder_fixed_part_pan_joint",
      "crane_x7_shoulder_revolute_part_tilt_joint",
      "crane_x7_upper_arm_revolute_part_twist_joint",
      "crane_x7_upper_arm_revolute_part_rotate_joint",
      "crane_x7_lower_arm_fixed_part_joint",
      "crane_x7_lower_arm_revolute_part_joint",
      "crane_x7_wrist_joint"
    };
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
      std::bind(
        &MotionServiceNode::moveToCameraPoseCallback, this,
        std::placeholders::_1, std::placeholders::_2));

    open_gripper_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/motion/open_gripper",
      std::bind(
        &MotionServiceNode::openGripperCallback, this,
        std::placeholders::_1, std::placeholders::_2));

    close_gripper_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/motion/close_gripper",
      std::bind(
        &MotionServiceNode::closeGripperCallback, this,
        std::placeholders::_1, std::placeholders::_2));

    execute_pose_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/motion/execute_pose",
      std::bind(
        &MotionServiceNode::executePoseCallback, this,
        std::placeholders::_1, std::placeholders::_2));

    execute_cartesian_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/motion/execute_cartesian",
      std::bind(
        &MotionServiceNode::executeCartesianCallback, this,
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
  void logJointAngles(const std::string & action_name)
  {
    auto joint_values = move_group_arm_->getCurrentJointValues();
    if (joint_values.size() >= 7) {
      RCLCPP_INFO(
        this->get_logger(),
        "[%s] Joint angles (deg): J1=%.1f, J2=%.1f, J3=%.1f, J4=%.1f, J5=%.1f, J6=%.1f, J7=%.1f",
        action_name.c_str(),
        angles::to_degrees(joint_values[0]),
        angles::to_degrees(joint_values[1]),
        angles::to_degrees(joint_values[2]),
        angles::to_degrees(joint_values[3]),
        angles::to_degrees(joint_values[4]),
        angles::to_degrees(joint_values[5]),
        angles::to_degrees(joint_values[6]));
    }
  }

  moveit_msgs::msg::Constraints createArmConstraints()
  {
    // Get current joint angles
    auto joint_values = move_group_arm_->getCurrentJointValues();
    if (joint_values.size() < 7) {
      return moveit_msgs::msg::Constraints();
    }

    moveit_msgs::msg::Constraints constraints;

    // J5 constraint (crane_x7_lower_arm_fixed_part_joint)
    // Keep J5 in range that maintains "elbow down" configuration
    // Safe range: [-90°, +90°] (prevents arm from flipping over)
    {
      const double J5_SAFE_CENTER = angles::from_degrees(0.0);
      const double J5_SAFE_HALF_RANGE = angles::from_degrees(90.0);

      moveit_msgs::msg::JointConstraint j5_constraint;
      j5_constraint.joint_name = "crane_x7_lower_arm_fixed_part_joint";
      j5_constraint.position = J5_SAFE_CENTER;
      j5_constraint.tolerance_above = J5_SAFE_HALF_RANGE;
      j5_constraint.tolerance_below = J5_SAFE_HALF_RANGE;
      j5_constraint.weight = 1.0;
      constraints.joint_constraints.push_back(j5_constraint);

      RCLCPP_DEBUG(
        this->get_logger(),
        "J5 constraint: current=%.1f°, target_range=[%.1f°, %.1f°]",
        angles::to_degrees(joint_values[4]),
        angles::to_degrees(J5_SAFE_CENTER - J5_SAFE_HALF_RANGE),
        angles::to_degrees(J5_SAFE_CENTER + J5_SAFE_HALF_RANGE));
    }

    // J6 constraint (crane_x7_lower_arm_revolute_part_joint)
    // Keep J6 in range for gripper pointing downward
    // Safe range: [-135°, +45°] (centered at -45° for pick operations)
    {
      const double J6_SAFE_CENTER = angles::from_degrees(-45.0);
      const double J6_SAFE_HALF_RANGE = angles::from_degrees(90.0);

      moveit_msgs::msg::JointConstraint j6_constraint;
      j6_constraint.joint_name = "crane_x7_lower_arm_revolute_part_joint";
      j6_constraint.position = J6_SAFE_CENTER;
      j6_constraint.tolerance_above = J6_SAFE_HALF_RANGE;
      j6_constraint.tolerance_below = J6_SAFE_HALF_RANGE;
      j6_constraint.weight = 1.0;
      constraints.joint_constraints.push_back(j6_constraint);

      RCLCPP_DEBUG(
        this->get_logger(),
        "J6 constraint: current=%.1f°, target_range=[%.1f°, %.1f°]",
        angles::to_degrees(joint_values[5]),
        angles::to_degrees(J6_SAFE_CENTER - J6_SAFE_HALF_RANGE),
        angles::to_degrees(J6_SAFE_CENTER + J6_SAFE_HALF_RANGE));
    }

    // J7 constraint (crane_x7_wrist_joint)
    // Safe range: [-45°, +135°]
    {
      const double J7_SAFE_CENTER = angles::from_degrees(45.0);
      const double J7_SAFE_HALF_RANGE = angles::from_degrees(90.0);

      moveit_msgs::msg::JointConstraint j7_constraint;
      j7_constraint.joint_name = "crane_x7_wrist_joint";
      j7_constraint.position = J7_SAFE_CENTER;
      j7_constraint.tolerance_above = J7_SAFE_HALF_RANGE;
      j7_constraint.tolerance_below = J7_SAFE_HALF_RANGE;
      j7_constraint.weight = 1.0;
      constraints.joint_constraints.push_back(j7_constraint);

      RCLCPP_DEBUG(
        this->get_logger(),
        "J7 constraint: current=%.1f°, target_range=[%.1f°, %.1f°]",
        angles::to_degrees(joint_values[6]),
        angles::to_degrees(J7_SAFE_CENTER - J7_SAFE_HALF_RANGE),
        angles::to_degrees(J7_SAFE_CENTER + J7_SAFE_HALF_RANGE));
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Arm constraints applied: J5=[%.1f°,%.1f°], J6=[%.1f°,%.1f°], J7=[%.1f°,%.1f°]",
      -90.0, 90.0, -135.0, 45.0, -45.0, 135.0);

    return constraints;
  }

  bool isArmConfigurationValid(const std::vector<double> & joint_values)
  {
    // Check if arm configuration keeps gripper pointing downward
    // Only validate J5 strictly - it's the main indicator of arm flip
    // J6/J7 are logged as warnings but don't reject the trajectory

    if (joint_values.size() < 7) {
      return false;
    }

    double j5_deg = angles::to_degrees(joint_values[4]);
    double j6_deg = angles::to_degrees(joint_values[5]);
    double j7_deg = angles::to_degrees(joint_values[6]);

    // J5 is the critical check - only reject if way out of range
    // Relaxed range: [-120°, 120°] to allow more flexibility
    if (j5_deg < -120.0 || j5_deg > 120.0) {
      RCLCPP_WARN(
        this->get_logger(),
        "J5 out of safe range: %.1f° (expected [-120°, 120°])", j5_deg);
      return false;
    }

    // J6 and J7 - just log warnings, don't reject
    if (j6_deg < -180.0 || j6_deg > 120.0) {
      RCLCPP_DEBUG(
        this->get_logger(),
        "J6 note: %.1f° (typical range [-180°, 120°])", j6_deg);
    }

    if (j7_deg < -180.0 || j7_deg > 180.0) {
      RCLCPP_DEBUG(
        this->get_logger(),
        "J7 note: %.1f° (typical range [-180°, 180°])", j7_deg);
    }

    return true;
  }

  bool validateTrajectory(const moveit_msgs::msg::RobotTrajectory & trajectory)
  {
    // Check if any point in the trajectory has invalid arm configuration
    for (size_t i = 0; i < trajectory.joint_trajectory.points.size(); ++i) {
      const auto & point = trajectory.joint_trajectory.points[i];
      if (!isArmConfigurationValid(point.positions)) {
        RCLCPP_WARN(
          this->get_logger(),
          "Trajectory point %zu has invalid arm configuration", i);
        return false;
      }
    }
    return true;
  }

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
    if (success) {
      logJointAngles("camera_pose");
    }
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
    RCLCPP_DEBUG(
      this->get_logger(), "Target pose received: (%.3f, %.3f, %.3f)",
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

    RCLCPP_INFO(
      this->get_logger(), "Executing pose: (%.3f, %.3f, %.3f)",
      target_pose_.pose.position.x,
      target_pose_.pose.position.y,
      target_pose_.pose.position.z);

    move_group_arm_->setStartStateToCurrentState();
    move_group_arm_->setPoseTarget(target_pose_.pose);

    // No path constraints - let MoveIt find the best solution
    // Trajectory validation in executeCartesianCallback catches major arm flips

    bool success = (move_group_arm_->move() == moveit::core::MoveItErrorCode::SUCCESS);

    response->success = success;
    response->message = success ? "Pose executed" : "Failed to execute pose";
    target_pose_received_ = false;

    RCLCPP_INFO(this->get_logger(), "Execute pose: %s", response->message.c_str());
    if (success) {
      logJointAngles("execute_pose");
    }
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

    RCLCPP_INFO(
      this->get_logger(), "Executing cartesian path to: (%.3f, %.3f, %.3f)",
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
    std::string failure_reason;

    if (fraction < 0.9) {
      failure_reason = "insufficient fraction: " + std::to_string(fraction);
    } else if (!validateTrajectory(trajectory)) {
      // Trajectory has invalid arm configuration (gripper pointing up)
      failure_reason = "invalid arm configuration (gripper facing upward)";
      RCLCPP_WARN(
        this->get_logger(),
        "Cartesian path rejected: arm would flip to upward-facing configuration");
    } else {
      auto exec_result = move_group_arm_->execute(trajectory);
      success = (exec_result == moveit::core::MoveItErrorCode::SUCCESS);
      if (!success) {
        failure_reason = "execution failed";
      }
    }

    response->success = success;
    response->message = success ?
      "Cartesian path executed" :
      "Failed to execute cartesian path (" + failure_reason + ")";
    target_pose_received_ = false;

    RCLCPP_INFO(this->get_logger(), "Execute cartesian: %s", response->message.c_str());
    if (success) {
      logJointAngles("execute_cartesian");
    }
  }

  std::shared_ptr<MoveGroupInterface> move_group_arm_;
  std::shared_ptr<MoveGroupInterface> move_group_gripper_;

  std::vector<std::string> joint_names_;
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
