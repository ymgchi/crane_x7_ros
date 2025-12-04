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
#include <memory>
#include <string>
#include <cstdlib>

#include "angles/angles.h"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("edge_based_sorting");

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
  RCLCPP_INFO(LOGGER, "Spawning new object at (%.2f, %.2f, %.2f)", x, y, z);
  char cmd[2048];
  char name[64];
  snprintf(name, sizeof(name), "wood_cube_%d", spawn_count++);

  // 青色（テーブルの茶色と区別しやすい）
  std::string rgba = "0.2 0.4 0.8 1";

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
    "<material><ambient>%s</ambient><diffuse>%s</diffuse></material>"
    "</visual></link></model></sdf>'",
    name, x, y, z, name, rgba.c_str(), rgba.c_str()
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
  double max_step = 0.01,
  double min_fraction = 0.25)  // Lower threshold for more flexibility
{
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory trajectory;
  const double jump_threshold = 0.0;
  const double eef_step = max_step;

  double fraction = move_group.computeCartesianPath(
    waypoints, eef_step, jump_threshold, trajectory);

  if (fraction < min_fraction) {
    RCLCPP_WARN(LOGGER, "Cartesian path planning failed (%.2f%% achieved, need %.0f%%)",
                fraction * 100.0, min_fraction * 100.0);
    return false;
  }

  if (fraction < 1.0) {
    RCLCPP_INFO(LOGGER, "Cartesian path partially planned (%.2f%% achieved)", fraction * 100.0);
  }

  move_group.execute(trajectory);
  return true;
}

// 把持ポーズ購読クラス
class GraspPoseListener : public rclcpp::Node
{
public:
  explicit GraspPoseListener(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("grasp_pose_listener", options)
  {
    pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/color_filtered_grasp/target_pose", 10,
      std::bind(&GraspPoseListener::pose_callback, this, std::placeholders::_1));
    table_height_subscription_ = this->create_subscription<std_msgs::msg::Float32>(
      "/color_filtered_grasp/table_height", 10,
      std::bind(&GraspPoseListener::table_height_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Listening to /color_filtered_grasp/target_pose");
  }

  geometry_msgs::msg::PoseStamped getLatestPose()
  {
    return latest_pose_;
  }

  bool hasNewPose()
  {
    bool result = has_new_pose_;
    has_new_pose_ = false;
    return result;
  }

  bool hasTableHeight() const
  {
    return table_height_received_;
  }

  double getTableHeight() const
  {
    return table_height_;
  }

  bool hasMedianTableHeight() const
  {
    return table_height_received_;
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr table_height_subscription_;
  geometry_msgs::msg::PoseStamped latest_pose_;
  bool has_new_pose_ = false;
  double table_height_ = std::numeric_limits<double>::quiet_NaN();
  bool table_height_received_ = false;

  void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    latest_pose_ = *msg;
    has_new_pose_ = true;
    RCLCPP_INFO(this->get_logger(), "Received grasp pose at (%.3f, %.3f, %.3f)",
                msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  }

  void table_height_callback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    table_height_ = static_cast<double>(msg->data);
    table_height_received_ = true;
    RCLCPP_INFO(this->get_logger(), "Received table height: %.3f (base_link)", table_height_);
  }
};

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
  auto grasp_pose_listener = std::make_shared<GraspPoseListener>(node_options);

  // Executor for asynchronous spinning
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(move_group_arm_node);
  executor.add_node(move_group_gripper_node);
  executor.add_node(grasp_pose_listener);
  std::thread([&executor]() {executor.spin();}).detach();

  // TF2 setup for coordinate transformations
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(move_group_arm_node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

  // Wait for robot_description to be available
  RCLCPP_INFO(LOGGER, "Waiting for robot_description and controllers...");
  rclcpp::sleep_for(std::chrono::seconds(5));

  // MoveGroupの初期化
  MoveGroupInterface move_group_arm(move_group_arm_node, "arm");
  move_group_arm.setMaxVelocityScalingFactor(0.7);
  move_group_arm.setMaxAccelerationScalingFactor(0.7);
  move_group_arm.setPlanningTime(20.0);
  move_group_arm.setGoalPositionTolerance(0.01);
  move_group_arm.setGoalOrientationTolerance(0.05);
  move_group_arm.allowReplanning(true);

  MoveGroupInterface move_group_gripper(move_group_gripper_node, "gripper");
  move_group_gripper.setMaxVelocityScalingFactor(0.7);
  move_group_gripper.setMaxAccelerationScalingFactor(0.7);
  move_group_gripper.setPlanningTime(10.0);
  move_group_gripper.allowReplanning(true);

  // camera_exampleと同じ撮影開始姿勢の関節セット
  // 観測姿勢（カメラ水平を重視して左傾きを解消）
  std::vector<double> camera_start_joints = {
    angles::from_degrees(0.0),    // base yaw
    angles::from_degrees(40.0),   // shoulder pitch
    angles::from_degrees(0.0),    // shoulder roll (水平に近づける)
    angles::from_degrees(-100.0), // elbow pitch
    angles::from_degrees(0.0),    // wrist roll
    angles::from_degrees(-80.0),  // wrist pitch
    angles::from_degrees(90.0)    // wrist yaw
  };

  // ロボットの現在状態を取得できるまで待機
  RCLCPP_INFO(LOGGER, "Waiting for current robot state...");
  while (!move_group_arm.getCurrentState()) {
    RCLCPP_WARN(LOGGER, "Waiting for robot state...");
    rclcpp::sleep_for(std::chrono::seconds(1));
  }
  RCLCPP_INFO(LOGGER, "Robot state received!");

  auto gripper_joint_values = move_group_gripper.getCurrentJointValues();
  const double GRIPPER_OPEN = angles::from_degrees(60.0);
  const double GRIPPER_CLOSE = angles::from_degrees(5.0);

  // 配置場所
  geometry_msgs::msg::Pose place_pose = createPose(0.35, 0.25, 0.15, -180, 0, 90);
  const double PLACE_Z_ABOVE = 0.25;  // 配置場所上空

  RCLCPP_INFO(LOGGER, "Starting edge-based sorting demo");

  // 初期姿勢確認
  RCLCPP_INFO(LOGGER, "Moving to home posture");
  move_group_arm.setStartStateToCurrentState();
  move_group_arm.setNamedTarget("home");
  move_group_arm.move();

  RCLCPP_INFO(LOGGER, "Moving to camera_example start posture");
  move_group_arm.setStartStateToCurrentState();
  move_group_arm.setJointValueTarget(camera_start_joints);
  move_group_arm.move();

  // オブジェクトをスポーン
  RCLCPP_INFO(LOGGER, "Spawning wood cubes");
  std::vector<geometry_msgs::msg::Point> spawn_positions = {
    // ロボット正面に落下させ、カメラ視野とワークスペース内に収める
    [](){geometry_msgs::msg::Point p; p.x = 0.30; p.y = 0.00; p.z = 1.05; return p;}(),
    [](){geometry_msgs::msg::Point p; p.x = 0.32; p.y = 0.12; p.z = 1.05; return p;}()
  };

  // ロボットはGazebo世界のZ=1.015m（テーブル上）に設置されている
  const double ROBOT_TABLE_HEIGHT = 1.015;
  RCLCPP_INFO(LOGGER, "=== DEBUG: Expected cube positions ===");
  for (size_t i = 0; i < spawn_positions.size(); ++i) {
    const auto & pos = spawn_positions[i];
    // World frame -> base_link frame (Z offset by robot height)
    double expected_base_z = pos.z - ROBOT_TABLE_HEIGHT;
    RCLCPP_INFO(LOGGER, "Cube %zu: World(%.2f, %.2f, %.2f) -> Expected base_link(%.2f, %.2f, %.3f)",
                i, pos.x, pos.y, pos.z, pos.x, pos.y, expected_base_z);
  }
  RCLCPP_INFO(LOGGER, "======================================");

  for (const auto & pos : spawn_positions) {
    spawnObjectInGazebo(pos.x, pos.y, pos.z);
    rclcpp::sleep_for(std::chrono::milliseconds(200));
  }
  rclcpp::sleep_for(std::chrono::milliseconds(1500));  // 落下と停止・カメラ安定待ち

  // グリッパーを開く
  RCLCPP_INFO(LOGGER, "Opening gripper");
  gripper_joint_values[0] = GRIPPER_OPEN;
  move_group_gripper.setJointValueTarget(gripper_joint_values);
  move_group_gripper.move();
  rclcpp::sleep_for(std::chrono::milliseconds(100));

  // メインループ: 把持ポーズを待って実行
  int pick_count = 0;
  const int MAX_PICKS = 5;
  const double PICK_Z_OFFSET = -0.02;  // 把持位置のオフセット（物体より少し下）
  const double APPROACH_CLEARANCE = 0.03;    // 把持位置からの上方向クリアランス
  const double MIN_APPROACH_Z = 0.20;        // 最小アプローチ高さ（color_sortingのPICK_Z_ABOVEに合わせる）
  const double PICK_Z_LIFT = 0.25;     // 持ち上げ高さ
  // WORKAROUND: In Gazebo simulation, point cloud Z values are in world-like coordinates
  // Robot base_link is at world Z ~1.015m, table surface at ~1.015m
  // Cube top is at world Z ~1.05m (table 1.015 + cube height 0.035)
  const double DEFAULT_MIN_VALID_Z = 0.90;  // Below table surface (world Z)
  const double DEFAULT_MAX_VALID_Z = 1.20;  // Maximum height above table (world Z)
  const double MIN_TABLE_DELTA = -0.05;  // テーブルからの最小オフセット（下に5cmまで許容）
  const double MAX_TABLE_DELTA = 0.20;    // テーブル上から最大20cmまで
  // XY座標のワークスペース範囲（base_link基準）- キューブ検出用に調整
  const double MIN_VALID_X = 0.20;     // ロボット近すぎると衝突の恐れ
  const double MAX_VALID_X = 0.40;     // キューブがある範囲に絞る
  const double MIN_VALID_Y = -0.20;    // キューブ配置範囲に絞る
  const double MAX_VALID_Y = 0.20;

  RCLCPP_INFO(LOGGER, "Waiting for grasp poses from color_filtered_grasp_live...");

  while (rclcpp::ok() && pick_count < MAX_PICKS) {
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    if (!grasp_pose_listener->hasNewPose()) {
      continue;
    }

    auto grasp_pose_stamped = grasp_pose_listener->getLatestPose();

    RCLCPP_INFO(LOGGER, "Received grasp pose in frame '%s' at (%.3f, %.3f, %.3f)",
                grasp_pose_stamped.header.frame_id.c_str(),
                grasp_pose_stamped.pose.position.x,
                grasp_pose_stamped.pose.position.y,
                grasp_pose_stamped.pose.position.z);

    // Check frame_id and handle transformation
    std::string source_frame = grasp_pose_stamped.header.frame_id;
    RCLCPP_INFO(LOGGER, "=== GRASP POSE DEBUG ===");
    RCLCPP_INFO(LOGGER, "Received grasp pose in frame '%s' at (%.3f, %.3f, %.3f)",
                source_frame.c_str(),
                grasp_pose_stamped.pose.position.x,
                grasp_pose_stamped.pose.position.y,
                grasp_pose_stamped.pose.position.z);

    geometry_msgs::msg::PoseStamped grasp_pose_base_link;

    // If pose is already in base_link, use it directly (no TF transform needed)
    if (source_frame == "base_link" || source_frame == "crane_x7/base_link") {
      RCLCPP_INFO(LOGGER, "Pose already in base_link frame - using directly (no TF transform)");
      grasp_pose_base_link = grasp_pose_stamped;
    } else {
      // Map Gazebo frame names to TF frame names
      // Gazebo uses "crane_x7/.../camera_depth" but TF tree exposes camera_link
      if (source_frame.find("camera_depth") != std::string::npos) {
        source_frame = "camera_link";
      }
      grasp_pose_stamped.header.frame_id = source_frame;

      // Transform grasp pose from camera frame to base_link (robot base)
      try {
        tf_buffer->transform(grasp_pose_stamped, grasp_pose_base_link, "base_link",
                            tf2::durationFromSec(1.0));
        RCLCPP_INFO(LOGGER, "Transformed from '%s' to base_link at (%.3f, %.3f, %.3f)",
                    source_frame.c_str(),
                    grasp_pose_base_link.pose.position.x,
                    grasp_pose_base_link.pose.position.y,
                    grasp_pose_base_link.pose.position.z);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(LOGGER, "TF2 transform failed: %s", ex.what());
        continue;
      }
    }

    auto grasp_pose = grasp_pose_base_link.pose;

    // テーブル高さを受信している場合は下限をテーブル+マージンに合わせる
    double min_valid_z = DEFAULT_MIN_VALID_Z;
    double max_valid_z = DEFAULT_MAX_VALID_Z;
    double dz_from_table = std::numeric_limits<double>::quiet_NaN();
    if (grasp_pose_listener->hasMedianTableHeight()) {
      const double table_h = grasp_pose_listener->getTableHeight();
      min_valid_z = std::max(DEFAULT_MIN_VALID_Z, table_h + MIN_TABLE_DELTA);
      max_valid_z = std::min(DEFAULT_MAX_VALID_Z, table_h + MAX_TABLE_DELTA);
      dz_from_table = grasp_pose.position.z - table_h;
      static bool table_logged = false;
      if (!table_logged) {
        table_logged = true;
        RCLCPP_INFO(LOGGER,
          "Using table-relative Z window: [%.3f, %.3f] (table=%.3f, delta [%.3f, %.3f])",
          min_valid_z, max_valid_z, table_h, MIN_TABLE_DELTA, MAX_TABLE_DELTA);
      }
    }

    // Validate XYZ coordinates are within robot workspace
    RCLCPP_INFO(LOGGER, "=== WORKSPACE VALIDATION ===");
    RCLCPP_INFO(LOGGER, "Grasp position: (%.3f, %.3f, %.3f)",
                grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z);
    RCLCPP_INFO(LOGGER, "Valid X range: [%.3f, %.3f]", MIN_VALID_X, MAX_VALID_X);
    RCLCPP_INFO(LOGGER, "Valid Y range: [%.3f, %.3f]", MIN_VALID_Y, MAX_VALID_Y);
    RCLCPP_INFO(LOGGER, "Valid Z range: [%.3f, %.3f]", min_valid_z, max_valid_z);

    if (grasp_pose.position.x < MIN_VALID_X || grasp_pose.position.x > MAX_VALID_X) {
      RCLCPP_WARN(LOGGER, "REJECTED: X=%.3f out of range [%.2f, %.2f]",
                  grasp_pose.position.x, MIN_VALID_X, MAX_VALID_X);
      continue;
    }
    if (grasp_pose.position.y < MIN_VALID_Y || grasp_pose.position.y > MAX_VALID_Y) {
      RCLCPP_WARN(LOGGER, "REJECTED: Y=%.3f out of range [%.2f, %.2f]",
                  grasp_pose.position.y, MIN_VALID_Y, MAX_VALID_Y);
      continue;
    }
    if (grasp_pose.position.z < min_valid_z || grasp_pose.position.z > max_valid_z) {
      if (std::isfinite(dz_from_table)) {
        RCLCPP_WARN(LOGGER,
          "REJECTED: Z=%.3f out of range (table=%.3f, delta=%.3f, allowed=[%.3f, %.3f])",
          grasp_pose.position.z, grasp_pose_listener->getTableHeight(), dz_from_table,
          MIN_TABLE_DELTA, MAX_TABLE_DELTA);
      } else {
        RCLCPP_WARN(LOGGER, "REJECTED: Z=%.3f out of range [%.2f, %.2f]",
                    grasp_pose.position.z, min_valid_z, max_valid_z);
      }
      continue;
    }
    RCLCPP_INFO(LOGGER, "ACCEPTED: Position within workspace");

    // Clamp Z into the validated window for stability
    grasp_pose.position.z = std::max(min_valid_z, std::min(max_valid_z, grasp_pose.position.z));

    // Override orientation to fixed downward grasp (ignoring edge normal direction)
    // This ensures the gripper approaches vertically, which is more reliable than
    // using arbitrary edge orientations that may be unreachable.
    tf2::Quaternion q;
    q.setRPY(0, angles::from_degrees(-90), 0);  // Pitch -90° = straight down
    grasp_pose.orientation.x = q.x();
    grasp_pose.orientation.y = q.y();
    grasp_pose.orientation.z = q.z();
    grasp_pose.orientation.w = q.w();

    RCLCPP_INFO(LOGGER, "✓ Valid grasp pose %d/%d at (%.3f, %.3f, %.3f)",
                pick_count + 1, MAX_PICKS,
                grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z);

    // DEBUG: 期待されるキューブ位置との比較
    // Cube 0: (0.30, 0.00, 0.035), Cube 1: (0.32, 0.12, 0.035)
    double dist_cube0 = std::sqrt(
        std::pow(grasp_pose.position.x - 0.30, 2) +
        std::pow(grasp_pose.position.y - 0.00, 2));
    double dist_cube1 = std::sqrt(
        std::pow(grasp_pose.position.x - 0.32, 2) +
        std::pow(grasp_pose.position.y - 0.12, 2));
    RCLCPP_INFO(LOGGER, "DEBUG: Distance to expected cube0(0.30,0.00): %.3fm, cube1(0.32,0.12): %.3fm",
                dist_cube0, dist_cube1);
    if (dist_cube0 < 0.05) {
      RCLCPP_INFO(LOGGER, "DEBUG: Likely detecting CUBE 0");
    } else if (dist_cube1 < 0.05) {
      RCLCPP_INFO(LOGGER, "DEBUG: Likely detecting CUBE 1");
    } else {
      RCLCPP_WARN(LOGGER, "DEBUG: Detection is FAR from expected cube positions! Possible misdetection.");
    }

    // アプローチポーズ（上空、絶対高さで設定）
    geometry_msgs::msg::Pose approach_pose = grasp_pose;
    // Keep approach close to grasp height but ensure minimum reachable height
    approach_pose.position.z = std::max(MIN_APPROACH_Z, std::min(DEFAULT_MAX_VALID_Z, grasp_pose.position.z + APPROACH_CLEARANCE));

    RCLCPP_INFO(LOGGER, "Approach pose: (%.3f, %.3f, %.3f)",
                approach_pose.position.x, approach_pose.position.y, approach_pose.position.z);

    // 把持ポーズ（検出位置 + オフセット）
    geometry_msgs::msg::Pose pick_pose = grasp_pose;
    pick_pose.position.z += PICK_Z_OFFSET;

    // === STEP 1: APPROACH ===
    RCLCPP_INFO(LOGGER, "=== STEP 1: APPROACH ===");
    RCLCPP_INFO(LOGGER, "Target approach pose: (%.3f, %.3f, %.3f)",
                approach_pose.position.x, approach_pose.position.y, approach_pose.position.z);
    move_group_arm.setStartStateToCurrentState();
    move_group_arm.setPoseTarget(approach_pose);
    move_group_arm.setPlanningTime(20.0);
    if (!move_group_arm.move()) {
      RCLCPP_WARN(LOGGER, "FAILED: Could not reach approach pose. Skipping this target.");
      continue;
    }
    RCLCPP_INFO(LOGGER, "SUCCESS: Reached approach pose");
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // === STEP 2: DESCEND ===
    RCLCPP_INFO(LOGGER, "=== STEP 2: DESCEND ===");
    RCLCPP_INFO(LOGGER, "Target pick pose: (%.3f, %.3f, %.3f)",
                pick_pose.position.x, pick_pose.position.y, pick_pose.position.z);
    move_group_arm.setStartStateToCurrentState();
    move_group_arm.setPoseTarget(pick_pose);
    move_group_arm.setPlanningTime(20.0);
    if (!move_group_arm.move()) {
      RCLCPP_WARN(LOGGER, "FAILED: Could not reach pick pose. Skipping this target.");
      continue;
    }
    RCLCPP_INFO(LOGGER, "SUCCESS: Reached pick pose");
    rclcpp::sleep_for(std::chrono::milliseconds(300));

    // === STEP 3: GRASP ===
    RCLCPP_INFO(LOGGER, "=== STEP 3: GRASP ===");
    RCLCPP_INFO(LOGGER, "Closing gripper to %.1f degrees", GRIPPER_CLOSE);
    gripper_joint_values[0] = GRIPPER_CLOSE;
    move_group_gripper.setJointValueTarget(gripper_joint_values);
    move_group_gripper.move();
    RCLCPP_INFO(LOGGER, "Gripper closed");
    rclcpp::sleep_for(std::chrono::milliseconds(250));

    // === STEP 4: LIFT ===
    RCLCPP_INFO(LOGGER, "=== STEP 4: LIFT ===");
    geometry_msgs::msg::Pose lift_pose = pick_pose;
    lift_pose.position.z = PICK_Z_LIFT;
    RCLCPP_INFO(LOGGER, "Target lift pose: (%.3f, %.3f, %.3f)",
                lift_pose.position.x, lift_pose.position.y, lift_pose.position.z);
    move_group_arm.setStartStateToCurrentState();
    move_group_arm.setPoseTarget(lift_pose);
    move_group_arm.setPlanningTime(20.0);
    if (!move_group_arm.move()) {
      RCLCPP_WARN(LOGGER, "FAILED: Could not lift object. Continuing anyway.");
      continue;
    }
    RCLCPP_INFO(LOGGER, "SUCCESS: Lifted object");
    rclcpp::sleep_for(std::chrono::milliseconds(100));

    // 配置場所へ移動
    geometry_msgs::msg::Pose place_pose_above = place_pose;
    place_pose_above.position.z = PLACE_Z_ABOVE;

    RCLCPP_INFO(LOGGER, "Moving to drop position");
    move_group_arm.setStartStateToCurrentState();
    move_group_arm.setPoseTarget(place_pose_above);
    if (!move_group_arm.move()) {
      RCLCPP_WARN(LOGGER, "Failed to move to drop position");
      continue;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(100));

    // 解放
    RCLCPP_INFO(LOGGER, "Releasing object");
    gripper_joint_values[0] = GRIPPER_OPEN;
    move_group_gripper.setJointValueTarget(gripper_joint_values);
    move_group_gripper.move();
    rclcpp::sleep_for(std::chrono::milliseconds(200));

    // カメラ姿勢に戻る
    RCLCPP_INFO(LOGGER, "Returning to camera posture");
    move_group_arm.setStartStateToCurrentState();
    move_group_arm.setJointValueTarget(camera_start_joints);
    move_group_arm.move();
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    pick_count++;
  }

  // 終了処理
  RCLCPP_INFO(LOGGER, "Final: Moving to camera_example start posture");
  move_group_arm.setStartStateToCurrentState();
  move_group_arm.setJointValueTarget(camera_start_joints);
  move_group_arm.move();

  // グリッパーを閉じる
  gripper_joint_values[0] = 0.0;
  move_group_gripper.setJointValueTarget(gripper_joint_values);
  move_group_gripper.move();

  RCLCPP_INFO(LOGGER, "Edge-based sorting demo completed!");

  rclcpp::shutdown();
  return 0;
}
