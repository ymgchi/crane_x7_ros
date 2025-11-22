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
#include <cstdio>
#include <map>
#include <limits>
#include <algorithm>
#include <optional>

#include "angles/angles.h"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_geometry/pinhole_camera_model.h"

using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("color_sorting");

// 色の定義
enum class Color {
  NONE,
  BLUE,
  YELLOW,
  GREEN
};

// 検出結果
struct DetectionResult {
  Color color;
  geometry_msgs::msg::Pose pose;
  bool detected;
};

// バッチスキャンで溜めるターゲット情報
struct TargetInfo {
  geometry_msgs::msg::Point position;
  geometry_msgs::msg::Quaternion orientation;
  Color color;
};

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
bool spawnObjectInGazebo(double x, double y, double z, Color color)
{
  // 色別の材質設定（ambient/diffuse）
  const std::map<Color, std::string> color_rgba = {
    {Color::BLUE,   "0.0 0.0 1.0 1"},
    {Color::YELLOW, "1.0 1.0 0.0 1"},
    {Color::GREEN,  "0.0 1.0 0.0 1"}
  };
  auto it = color_rgba.find(color);
  std::string rgba = (it != color_rgba.end()) ? it->second : "0.5 0.5 0.5 1";

  RCLCPP_INFO(LOGGER, "Spawning new object at (%.2f, %.2f, %.2f)", x, y, z);
  char cmd[2048];
  char name[64];
  snprintf(name, sizeof(name), "color_cube_%d", spawn_count++);

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


// 色検出クラス
class ColorDetector : public rclcpp::Node
{
public:
  explicit ColorDetector(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("color_detector", options)
  {
    image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/color/image_raw", 10,
      std::bind(&ColorDetector::image_callback, this, std::placeholders::_1));

    depth_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/aligned_depth_to_color/image_raw", 10,
      std::bind(&ColorDetector::depth_callback, this, std::placeholders::_1));

    camera_info_subscription_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera/color/camera_info", 10,
      std::bind(&ColorDetector::camera_info_callback, this, std::placeholders::_1));

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  }

  DetectionResult getLatestDetection()
  {
    return latest_detection_;
  }

  std::vector<DetectionResult> getLatestDetections()
  {
    return latest_detections_;
  }

  void resetDetection()
  {
    latest_detection_.detected = false;
    latest_detection_.color = Color::NONE;
    latest_detections_.clear();
  }

  void resetDetections()
  {
    resetDetection();
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscription_;
  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
  sensor_msgs::msg::Image::SharedPtr depth_image_;
  DetectionResult latest_detection_{Color::NONE, geometry_msgs::msg::Pose(), false};
  std::vector<DetectionResult> latest_detections_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    if (!camera_info_ || !depth_image_) {
      return;
    }

    // Gather candidate frame names to compensate for namespace/prefix differences
    auto build_candidate_frames = [&](const std::string & raw_frame) {
      std::vector<std::string> frames;
      auto add_unique = [&](const std::string & f) {
        if (std::find(frames.begin(), frames.end(), f) == frames.end()) {
          frames.push_back(f);
        }
      };

      add_unique(raw_frame);

      // If frame lacks optical suffix, try adding it
      if (raw_frame.find("_optical_frame") == std::string::npos) {
        add_unique(raw_frame + "_optical_frame");
      }

      // Plain known frames
      add_unique("camera_color_optical_frame");
      add_unique("camera_link");  // as a fallback intermediate link

      // If frame has a prefix (e.g., "crane_x7/..."), reuse it for known names
      auto slash_pos = raw_frame.find('/');
      if (slash_pos != std::string::npos) {
        std::string prefix = raw_frame.substr(0, slash_pos);
        add_unique(prefix + "/camera_color_optical_frame");
        add_unique(prefix + "/camera_link");
      }

      return frames;
    };

    auto cv_img = cv_bridge::toCvShare(msg, msg->encoding);
    cv::Mat hsv_image;
    if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
      cv::cvtColor(cv_img->image, hsv_image, cv::COLOR_BGR2HSV);
    } else {
      cv::cvtColor(cv_img->image, hsv_image, cv::COLOR_RGB2HSV);
    }

    // 青、黄、緑の順で検出を試みる
    std::vector<std::tuple<Color, int, int, int, int, int, int>> color_ranges = {
      {Color::BLUE,   100, 125, 100, 255, 30, 255},  // 青
      {Color::YELLOW,  20,  35, 100, 255, 60, 255},  // 黄（足元色と分離するためS/V高め）
      {Color::GREEN,   40,  80, 100, 255, 30, 255}   // 緑
    };

    for (const auto& [color, low_h, high_h, low_s, high_s, low_v, high_v] : color_ranges) {
      cv::Mat img_thresholded;
      cv::inRange(
        hsv_image,
        cv::Scalar(low_h, low_s, low_v),
        cv::Scalar(high_h, high_s, high_v),
        img_thresholded);

      // ノイズ除去（より強力に）
      cv::morphologyEx(
        img_thresholded, img_thresholded, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));  // 楕円カーネルで強化
      cv::morphologyEx(
        img_thresholded, img_thresholded, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

      // くっついた輪郭を分離（エロージョン）
      cv::morphologyEx(
        img_thresholded, img_thresholded, cv::MORPH_ERODE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

      // 輪郭検出で複数物体を個別に認識
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(img_thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

      // 画像中心に最も近い輪郭を選ぶ（カメラ視野の中心に近い物体を優先）
      double min_distance_to_center = std::numeric_limits<double>::max();
      int best_contour_idx = -1;
      cv::Point2d image_center(camera_info_->width / 2.0, camera_info_->height / 2.0);

      for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > 800) {  // 面積閾値チェック
          // この輪郭の中心を計算
          cv::Moments m = cv::moments(contours[i]);
          if (m.m00 > 0) {
            cv::Point2d center(m.m10 / m.m00, m.m01 / m.m00);
            double dist = cv::norm(center - image_center);

            // 画像中心に最も近いものを選択
            if (dist < min_distance_to_center) {
              min_distance_to_center = dist;
              best_contour_idx = static_cast<int>(i);
            }
          }
        }
      }

      if (best_contour_idx >= 0) {
        // 選択された物体のモーメント計算
        cv::Moments moment = cv::moments(contours[best_contour_idx]);
        double d_area = moment.m00;

        // 物体検出成功
        image_geometry::PinholeCameraModel camera_model;
        camera_model.fromCameraInfo(camera_info_);

        double pixel_x = moment.m10 / d_area;
        double pixel_y = moment.m01 / d_area;
        cv::Point2d point(pixel_x, pixel_y);

        // rectify and clamp to image bounds
      cv::Point2d rect_point = camera_model.rectifyPoint(point);
      cv::Point3d ray = camera_model.projectPixelTo3dRay(rect_point);
      int px = static_cast<int>(std::round(rect_point.x));
      int py = static_cast<int>(std::round(rect_point.y));
      px = std::max(0, std::min(px, static_cast<int>(camera_info_->width) - 1));
      py = std::max(0, std::min(py, static_cast<int>(camera_info_->height) - 1));

        const double DEPTH_OFFSET = 0.015;
        auto cv_depth = cv_bridge::toCvShare(depth_image_, depth_image_->encoding);
        // 3x3 窓の中央値を使用して深度ノイズを低減
        auto depth_median = [&](int cx, int cy) -> std::optional<double> {
          std::vector<double> vals;
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
              int xx = std::clamp(cx + dx, 0, static_cast<int>(camera_info_->width) - 1);
              int yy = std::clamp(cy + dy, 0, static_cast<int>(camera_info_->height) - 1);
              if (depth_image_->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
                double v = cv_depth->image.at<uint16_t>(yy, xx) / 1000.0;
                if (v > 0.0) vals.push_back(v);
              } else if (depth_image_->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
                float v = cv_depth->image.at<float>(yy, xx);
                if (std::isfinite(v) && v > 0.0f) vals.push_back(static_cast<double>(v));
              }
            }
          }
          if (vals.empty()) {
            return std::nullopt;
          }
          std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
          return vals[vals.size() / 2];
        };

        auto median_opt = depth_median(px, py);
        if (!median_opt.has_value()) {
          continue;
        }
        double center_distance = median_opt.value() + DEPTH_OFFSET;

        const double DEPTH_MAX = 1.2;
        const double DEPTH_MIN = 0.15;
        if (center_distance >= DEPTH_MIN && center_distance <= DEPTH_MAX) {
          // カメラ座標系での位置
          cv::Point3d camera_pos(
            ray.x * center_distance,
            ray.y * center_distance,
            ray.z * center_distance);

          // base_link座標系に変換（tfを使用）
          geometry_msgs::msg::PointStamped p_cam, p_base;
          p_cam.header = msg->header;  // カメラフレーム
          p_cam.point.x = camera_pos.x;
          p_cam.point.y = camera_pos.y;
          p_cam.point.z = camera_pos.z;

          bool transformed = false;
          std::string used_source_frame;
          std::string used_target_frame;
          const std::vector<std::string> candidate_sources = build_candidate_frames(p_cam.header.frame_id);
          const std::vector<std::string> candidate_targets = {
            "base_link",
            "crane_x7/base_link",
            "crane_x7/crane_x7_base_link"
          };

          for (const auto & src : candidate_sources) {
            for (const auto & tgt : candidate_targets) {
              try {
                p_cam.header.frame_id = src;
                p_base = tf_buffer_->transform(p_cam, tgt, tf2::durationFromSec(0.1));
                used_source_frame = src;
                used_target_frame = tgt;
                transformed = true;
                break;
              } catch (const tf2::TransformException &) {
                continue;
              }
            }
            if (transformed) {
              break;
            }
          }

          if (transformed) {
            geometry_msgs::msg::Pose object_pose;
            object_pose.position.x = p_base.point.x;
            object_pose.position.y = p_base.point.y;
            object_pose.position.z = p_base.point.z;
            object_pose.orientation.w = 1.0;

            latest_detection_.color = color;
            latest_detection_.pose = object_pose;
            latest_detection_.detected = true;

            // Avoid adding near-identical detections within the same frame
            bool close_to_existing = false;
            for (const auto & d : latest_detections_) {
              double dx = d.pose.position.x - object_pose.position.x;
              double dy = d.pose.position.y - object_pose.position.y;
              double dz = d.pose.position.z - object_pose.position.z;
              if (std::sqrt(dx * dx + dy * dy + dz * dz) < 0.03 && d.color == color) {
                close_to_existing = true;
                break;
              }
            }
            if (!close_to_existing) {
              latest_detections_.push_back(latest_detection_);
            }

            std::string color_name = (color == Color::BLUE) ? "Blue" :
                                     (color == Color::YELLOW) ? "Yellow" : "Green";
            RCLCPP_INFO(LOGGER, "Detected %s object %s (%.3f, %.3f, %.3f) [cam (%.3f, %.3f, %.3f), src=%s -> tgt=%s]",
                        color_name.c_str(),
                        used_target_frame.c_str(),
                        object_pose.position.x, object_pose.position.y, object_pose.position.z,
                        camera_pos.x, camera_pos.y, camera_pos.z,
                        used_source_frame.c_str(), used_target_frame.c_str());
            // Keep searching for other objects in the same frame
          } else {
            RCLCPP_WARN(LOGGER, "Transform failed for frame %s. Tried optical/prefix variants. Skipping detection.",
              msg->header.frame_id.c_str());
          }
        }
      }  // end of if (best_contour_idx >= 0)
    }  // end of for color_ranges
  }

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    camera_info_ = msg;
  }

  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    depth_image_ = msg;
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
  auto color_detector_node = std::make_shared<ColorDetector>(node_options);

  // For current state monitor and color detection
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(move_group_arm_node);
  executor.add_node(move_group_gripper_node);
  executor.add_node(color_detector_node);
  std::thread([&executor]() {executor.spin();}).detach();

  // Wait for robot_description to be available
  RCLCPP_INFO(LOGGER, "Waiting for robot_description and controllers...");
  rclcpp::sleep_for(std::chrono::seconds(5));

  // MoveGroupの初期化
  MoveGroupInterface move_group_arm(move_group_arm_node, "arm");
  move_group_arm.setMaxVelocityScalingFactor(0.7);
  move_group_arm.setMaxAccelerationScalingFactor(0.7);
  move_group_arm.setPlanningTime(20.0);  // プランニングタイムアウトを伸ばしてプラン失敗を減らす
  move_group_arm.setGoalPositionTolerance(0.01);
  move_group_arm.setGoalOrientationTolerance(0.05);
  move_group_arm.allowReplanning(true);

  MoveGroupInterface move_group_gripper(move_group_gripper_node, "gripper");
  move_group_gripper.setMaxVelocityScalingFactor(0.7);
  move_group_gripper.setMaxAccelerationScalingFactor(0.7);
  move_group_gripper.setPlanningTime(10.0);
  move_group_gripper.allowReplanning(true);

  // camera_example（color_detection）と同じ撮影開始姿勢の関節セット
  std::vector<double> camera_start_joints = {
    angles::from_degrees(0.0),
    angles::from_degrees(60.0),
    angles::from_degrees(0.0),
    angles::from_degrees(-120.0),
    angles::from_degrees(0.0),
    angles::from_degrees(-50.0),
    angles::from_degrees(90.0)
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
  const double GRIPPER_CLOSE = angles::from_degrees(5.0);  // 少しだけ強めに閉じる

  // スキャン用観察姿勢（上方から見下ろす・中央のみ）
  std::vector<geometry_msgs::msg::Pose> scan_poses = {
    createPose(0.25, 0.00, 0.35, -180, 0, 90)  // 中央
  };

  // 配置場所（色ごとに異なる位置）: 探索エリアから十分離した扇形配置で衝突と誤検出を回避
  std::map<Color, geometry_msgs::msg::Pose> place_poses = {
    {Color::YELLOW, createPose(0.30,  0.30, 0.15, -180, 0, 90)},   // 左奥
    {Color::BLUE,   createPose(0.30, -0.30, 0.15, -180, 0, 90)},   // 右奥
    {Color::GREEN,  createPose(0.40,  0.00, 0.15, -180, 0, 90)}    // 前方
  };

  std::map<Color, geometry_msgs::msg::Pose> place_poses_above = place_poses;  // ドロップ用に同じ高さで移動のみ

  RCLCPP_INFO(LOGGER, "Starting color sorting demo");
  // 初期姿勢確認（home → camera_exampleスタートに確実に入れる）
  RCLCPP_INFO(LOGGER, "Moving to home posture");
  move_group_arm.setStartStateToCurrentState();
  move_group_arm.setNamedTarget("home");
  move_group_arm.move();
  RCLCPP_INFO(LOGGER, "Moving to camera_example start posture");
  move_group_arm.setStartStateToCurrentState();
  move_group_arm.setJointValueTarget(camera_start_joints);
  move_group_arm.move();

  // シーンを初期化して赤・青・緑をまとめて投入
  RCLCPP_INFO(LOGGER, "Spawning yellow/blue/green cubes");

  std::vector<Color> spawn_sequence = {Color::YELLOW, Color::BLUE, Color::GREEN};
  std::vector<geometry_msgs::msg::Point> spawn_positions = {
    [](){geometry_msgs::msg::Point p; p.x = 0.18; p.y = -0.10; p.z = 1.10; return p;}(),  // YELLOW: 左側
    [](){geometry_msgs::msg::Point p; p.x = 0.25; p.y = 0.00;  p.z = 1.10; return p;}(),  // BLUE: 中央奥
    [](){geometry_msgs::msg::Point p; p.x = 0.18; p.y = 0.10;  p.z = 1.10; return p;}()   // GREEN: 右側
  };
  for (size_t i = 0; i < spawn_sequence.size(); ++i) {
    auto pos = spawn_positions[std::min(i, spawn_positions.size() - 1)];
    spawnObjectInGazebo(pos.x, pos.y, pos.z, spawn_sequence[i]);
    rclcpp::sleep_for(std::chrono::milliseconds(200));
  }
  rclcpp::sleep_for(std::chrono::milliseconds(1200));  // 落下と停止・カメラ安定待ち

  // グリッパーを開く
  RCLCPP_INFO(LOGGER, "Opening gripper");
  gripper_joint_values[0] = GRIPPER_OPEN;
  move_group_gripper.setJointValueTarget(gripper_joint_values);
  move_group_gripper.move();
  rclcpp::sleep_for(std::chrono::milliseconds(100));

  // スキャン〜ピック&プレースを、対象が無くなるまで繰り返す
  while (rclcpp::ok()) {
    // フェーズ1: 一括スキャンしてターゲットリストを作成
    std::vector<TargetInfo> targets;
    auto in_work_area = [](const geometry_msgs::msg::Point & p) {
      return (p.y > -0.20 && p.y < 0.20 && p.x > 0.0);
    };
    auto is_duplicate = [&](const geometry_msgs::msg::Point & p) {
      for (const auto & t : targets) {
        double dx = p.x - t.position.x;
        double dy = p.y - t.position.y;
        double dz = p.z - t.position.z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 0.05) {
          return true;
        }
      }
      return false;
    };

    RCLCPP_INFO(LOGGER, "Scanning all poses to build target list");
    for (size_t i = 0; i < scan_poses.size(); ++i) {
      RCLCPP_INFO(LOGGER, "Scanning pose %zu/%zu", i + 1, scan_poses.size());
      move_group_arm.setStartStateToCurrentState();
      move_group_arm.setPoseTarget(scan_poses[i]);
      if (!move_group_arm.move()) {
        RCLCPP_WARN(LOGGER, "Scan pose %zu failed to reach, continuing", i + 1);
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(500));

      // 視野内の全ての物体を検出するまで繰り返す
      int consecutive_duplicate_or_invalid = 0;
      int total_attempts = 0;
      const int MAX_ATTEMPTS = 20;  // 最大試行回数

      while (consecutive_duplicate_or_invalid < 5 && total_attempts < MAX_ATTEMPTS) {
        color_detector_node->resetDetections();
        rclcpp::sleep_for(std::chrono::milliseconds(300));
        auto detections = color_detector_node->getLatestDetections();
        total_attempts++;

        if (detections.empty()) {
          consecutive_duplicate_or_invalid++;
          continue;
        }

        bool added_new = false;
        for (const auto & detection : detections) {
          if (!detection.detected) {
            continue;
          }
          auto p = detection.pose.position;
          if (!in_work_area(p)) {
            RCLCPP_DEBUG(LOGGER, "Ignoring object outside work area (%.3f, %.3f, %.3f)", p.x, p.y, p.z);
            continue;  // エリア外の物体はスキップ
          }
          if (is_duplicate(p)) {
            RCLCPP_DEBUG(LOGGER, "Skipped duplicate detection near (%.3f, %.3f, %.3f)", p.x, p.y, p.z);
            continue;
          }

          // 新しい物体を発見
          TargetInfo tgt;
          tgt.position = p;
          tgt.orientation = scan_poses[i].orientation;
          tgt.color = detection.color;
          targets.push_back(tgt);

          std::string color_name = (detection.color == Color::BLUE) ? "Blue" :
                                   (detection.color == Color::YELLOW) ? "Yellow" : "Green";
          RCLCPP_INFO(LOGGER, "Added target %s at (%.3f, %.3f, %.3f) from scan pose %zu",
            color_name.c_str(), p.x, p.y, p.z, i + 1);

          added_new = true;
        }

        if (added_new) {
          // 新しい物体を見つけたのでカウンタをリセット
          consecutive_duplicate_or_invalid = 0;
        } else {
          consecutive_duplicate_or_invalid++;
        }
      }
      RCLCPP_INFO(LOGGER, "Scan pose %zu complete. Total targets found: %zu (attempts: %d)",
                  i + 1, targets.size(), total_attempts);
    }

    if (targets.empty()) {
      RCLCPP_INFO(LOGGER, "No targets found; ending loop");
      break;
    }

    // フェーズ2: 順次ピック&プレース
    const double PICK_Z = 0.095;       // 固定高さで下降
    const double PICK_Z_ABOVE = 0.20;  // ホバー高さ
    auto make_pick_pose = [&](const TargetInfo & tgt, double z) {
      geometry_msgs::msg::Pose pose;
      pose.position = tgt.position;
      pose.position.z = z;
      pose.orientation = tgt.orientation;
      return pose;
    };

    for (size_t idx = 0; idx < targets.size(); ++idx) {
      auto & tgt = targets[idx];
      std::string color_name = (tgt.color == Color::BLUE) ? "Blue" :
                               (tgt.color == Color::YELLOW) ? "Yellow" : "Green";
      RCLCPP_INFO(LOGGER, "Processing target %zu/%zu (%s)", idx + 1, targets.size(), color_name.c_str());

      auto pick_pose_above = make_pick_pose(tgt, PICK_Z_ABOVE);
      auto pick_pose = make_pick_pose(tgt, PICK_Z);

      // 接近（ホバー）
      move_group_arm.setStartStateToCurrentState();
      if (!executeCartesianPath(move_group_arm, pick_pose_above, 0.01)) {
        RCLCPP_WARN(LOGGER, "Failed to move to hover for target %zu, skipping", idx + 1);
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(500));

      // 近距離再検出で補正
      color_detector_node->resetDetection();
      rclcpp::sleep_for(std::chrono::milliseconds(500));
      auto refined = color_detector_node->getLatestDetection();
      bool refined_ok = false;
      for (int attempt = 0; attempt < 3; ++attempt) {
        if (attempt > 0) {
          color_detector_node->resetDetection();
          rclcpp::sleep_for(std::chrono::milliseconds(200));
          refined = color_detector_node->getLatestDetection();
        }
        if (!refined.detected) {
          continue;
        }
        if (refined.color != tgt.color) {
          RCLCPP_INFO(LOGGER, "Refine: ignoring other color for target %zu (wanted %s)", idx + 1, color_name.c_str());
          continue;
        }
        if (!in_work_area(refined.pose.position)) {
          RCLCPP_INFO(LOGGER, "Refine: ignoring out-of-work-area detection (%.3f, %.3f, %.3f)", refined.pose.position.x, refined.pose.position.y, refined.pose.position.z);
          continue;
        }
        // Accept refinement
        tgt.position = refined.pose.position;
        pick_pose_above = make_pick_pose(tgt, PICK_Z_ABOVE);
        pick_pose = make_pick_pose(tgt, PICK_Z);
        RCLCPP_INFO(LOGGER, "Refined target %zu to (%.3f, %.3f, %.3f)", idx + 1, tgt.position.x, tgt.position.y, tgt.position.z);
        move_group_arm.setStartStateToCurrentState();
        if (!executeCartesianPath(move_group_arm, pick_pose_above, 0.01)) {
          RCLCPP_WARN(LOGGER, "Failed to adjust hover after refinement for target %zu", idx + 1);
        } else {
          rclcpp::sleep_for(std::chrono::milliseconds(200));
        }
        refined_ok = true;
        break;
      }
      if (!refined_ok) {
        RCLCPP_WARN(LOGGER, "Refinement failed for target %zu, proceeding with initial position", idx + 1);
      }

      // 下降して把持
      RCLCPP_INFO(LOGGER, "Descending to pick target %zu", idx + 1);
      move_group_arm.setStartStateToCurrentState();
      if (!executeCartesianPath(move_group_arm, pick_pose, 0.01)) {
        RCLCPP_WARN(LOGGER, "Failed to descend for target %zu, skipping", idx + 1);
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(300));

      RCLCPP_INFO(LOGGER, "Closing gripper");
      gripper_joint_values[0] = GRIPPER_CLOSE;
      move_group_gripper.setJointValueTarget(gripper_joint_values);
      move_group_gripper.move();
      rclcpp::sleep_for(std::chrono::milliseconds(250));  // 把持後に少し待機して安定させる

      // 上昇
      move_group_arm.setStartStateToCurrentState();
      if (!executeCartesianPath(move_group_arm, pick_pose_above, 0.01)) {
        RCLCPP_WARN(LOGGER, "Failed to lift after pick for target %zu", idx + 1);
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(100));

      // 配置: 安全な下向き姿勢で配置場所へ
      auto drop_pose = place_poses_above[tgt.color];
      // drop_pose には既に安全な姿勢が設定されている
      RCLCPP_INFO(LOGGER, "Moving to drop position for %s", color_name.c_str());
      move_group_arm.setStartStateToCurrentState();
      move_group_arm.setPoseTarget(drop_pose);
      if (!move_group_arm.move()) {
        RCLCPP_WARN(LOGGER, "Failed to move to drop position for target %zu", idx + 1);
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(100));

      RCLCPP_INFO(LOGGER, "Releasing object");
      gripper_joint_values[0] = GRIPPER_OPEN;
      move_group_gripper.setJointValueTarget(gripper_joint_values);
      move_group_gripper.move();
      rclcpp::sleep_for(std::chrono::milliseconds(200));
    }

    // 次のサイクルのためにスキャン姿勢へ戻る
    move_group_arm.setStartStateToCurrentState();
    move_group_arm.setJointValueTarget(camera_start_joints);
    move_group_arm.move();
    rclcpp::sleep_for(std::chrono::milliseconds(200));
  }

  // 最終的に初期観測姿勢に戻る
  RCLCPP_INFO(LOGGER, "Final: Moving to camera_example start posture");
  move_group_arm.setStartStateToCurrentState();
  move_group_arm.setJointValueTarget(camera_start_joints);
  move_group_arm.move();

  // グリッパーを閉じる
  gripper_joint_values[0] = 0.0;
  move_group_gripper.setJointValueTarget(gripper_joint_values);
  move_group_gripper.move();

  RCLCPP_INFO(LOGGER, "Color sorting demo completed!");

  rclcpp::shutdown();
  return 0;
}
