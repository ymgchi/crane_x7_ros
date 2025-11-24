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

// 重複検出閾値（統一定数）
static const double DUPLICATE_DETECTION_THRESHOLD = 0.04;  // 4cm - 同一物体と見なす距離

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
  double detected_z;  // 検出時の実際のZ座標（動的ピック高さ調整用）
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

    // HSV範囲パラメータの初期化（環境に応じて調整可能）
    this->declare_parameter("blue.h_min", 100);
    this->declare_parameter("blue.h_max", 125);
    this->declare_parameter("blue.s_min", 100);
    this->declare_parameter("blue.s_max", 255);
    this->declare_parameter("blue.v_min", 30);
    this->declare_parameter("blue.v_max", 255);

    this->declare_parameter("yellow.h_min", 20);
    this->declare_parameter("yellow.h_max", 35);
    this->declare_parameter("yellow.s_min", 100);
    this->declare_parameter("yellow.s_max", 255);
    this->declare_parameter("yellow.v_min", 60);
    this->declare_parameter("yellow.v_max", 255);

    this->declare_parameter("green.h_min", 40);
    this->declare_parameter("green.h_max", 80);
    this->declare_parameter("green.s_min", 100);
    this->declare_parameter("green.s_max", 255);
    this->declare_parameter("green.v_min", 30);
    this->declare_parameter("green.v_max", 255);

    // パラメータから値を読み込む
    hsv_ranges_[Color::BLUE] = {
      this->get_parameter("blue.h_min").as_int(),
      this->get_parameter("blue.h_max").as_int(),
      this->get_parameter("blue.s_min").as_int(),
      this->get_parameter("blue.s_max").as_int(),
      this->get_parameter("blue.v_min").as_int(),
      this->get_parameter("blue.v_max").as_int()
    };

    hsv_ranges_[Color::YELLOW] = {
      this->get_parameter("yellow.h_min").as_int(),
      this->get_parameter("yellow.h_max").as_int(),
      this->get_parameter("yellow.s_min").as_int(),
      this->get_parameter("yellow.s_max").as_int(),
      this->get_parameter("yellow.v_min").as_int(),
      this->get_parameter("yellow.v_max").as_int()
    };

    hsv_ranges_[Color::GREEN] = {
      this->get_parameter("green.h_min").as_int(),
      this->get_parameter("green.h_max").as_int(),
      this->get_parameter("green.s_min").as_int(),
      this->get_parameter("green.s_max").as_int(),
      this->get_parameter("green.v_min").as_int(),
      this->get_parameter("green.v_max").as_int()
    };

    RCLCPP_INFO(this->get_logger(), "HSV ranges initialized from parameters");
  }

  DetectionResult getLatestDetection()
  {
    return latest_detection_;
  }

  std::vector<DetectionResult> getLatestDetections()
  {
    return latest_detections_;
  }

  // 複数回検出して平均位置を返す（ノイズ低減）
  DetectionResult getAveragedDetection(
    int num_samples = 5,
    int wait_ms = 200,
    Color target_color = Color::NONE,
    std::optional<geometry_msgs::msg::Point> expected_position = std::nullopt,
    double max_distance_from_expected = 0.10)
  {
    std::vector<DetectionResult> samples;

    for (int i = 0; i < num_samples; ++i) {
      resetDetection();
      rclcpp::sleep_for(std::chrono::milliseconds(wait_ms));
      auto detection = getLatestDetection();
      if (!detection.detected) {
        continue;
      }

      // 期待している色のみ採用し、別の色は無視する
      if (target_color != Color::NONE && detection.color != target_color) {
        continue;
      }

      // 期待位置から大きく外れた検出は補正に使わない
      if (expected_position.has_value()) {
        const auto & p = detection.pose.position;
        double dx = p.x - expected_position->x;
        double dy = p.y - expected_position->y;
        double dz = p.z - expected_position->z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist > max_distance_from_expected) {
          continue;
        }
      }

      if (detection.detected) {
        samples.push_back(detection);
      }
    }

    if (samples.empty()) {
      if (target_color != Color::NONE) {
        RCLCPP_WARN(LOGGER, "No valid samples found for target color during refinement");
      }
      DetectionResult result;
      result.detected = false;
      result.color = Color::NONE;
      return result;
    }

    // 平均位置を計算（サンプルはすべて同じ色に絞られている）
    DetectionResult averaged = samples[0];
    averaged.pose.position.x = 0.0;
    averaged.pose.position.y = 0.0;
    averaged.pose.position.z = 0.0;

    for (const auto & sample : samples) {
      averaged.pose.position.x += sample.pose.position.x;
      averaged.pose.position.y += sample.pose.position.y;
      averaged.pose.position.z += sample.pose.position.z;
    }

    averaged.pose.position.x /= samples.size();
    averaged.pose.position.y /= samples.size();
    averaged.pose.position.z /= samples.size();
    averaged.detected = true;

    RCLCPP_INFO(LOGGER, "Averaged %zu/%zu samples for better accuracy (color-filtered)",
                samples.size(), samples.size());

    return averaged;
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

  // HSV範囲パラメータ（実行時調整可能）
  struct HSVRange {
    int h_min, h_max;
    int s_min, s_max;
    int v_min, v_max;
  };
  std::map<Color, HSVRange> hsv_ranges_;

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

    // 青、黄、緑の順で検出を試みる（パラメータ化されたHSV範囲を使用）
    std::vector<Color> colors_to_detect = {Color::BLUE, Color::YELLOW, Color::GREEN};

    for (const auto& color : colors_to_detect) {
      const auto& range = hsv_ranges_[color];
      cv::Mat img_thresholded;
      cv::inRange(
        hsv_image,
        cv::Scalar(range.h_min, range.s_min, range.v_min),
        cv::Scalar(range.h_max, range.s_max, range.v_max),
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
        if (area > 1000) {  // 面積閾値を上げてノイズを除外（800→1000）
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
        // 5x5 窓の中央値を使用して深度ノイズを低減（強化版）
        auto depth_median = [&](int cx, int cy) -> std::optional<double> {
          std::vector<double> vals;
          // 5x5窓に拡大してより多くのサンプルから中央値を計算
          for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
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
              if (std::sqrt(dx * dx + dy * dy + dz * dz) < DUPLICATE_DETECTION_THRESHOLD && d.color == color) {
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
  const double PLACE_EXCLUSION_RADIUS = 0.08;  // 配置済みを再検出しないための半径

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
    auto point_distance = [](const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b) {
      double dx = a.x - b.x;
      double dy = a.y - b.y;
      double dz = a.z - b.z;
      return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    // フェーズ1: 一括スキャンしてターゲットリストを作成
    std::vector<TargetInfo> targets;
    auto in_work_area = [](const geometry_msgs::msg::Point & p) {
      // 作業エリア内の物体のみをターゲットに
      // Z座標が負の場合は除外（TF変換エラー）
      return (p.x > 0.00 && p.x < 0.50 &&
              p.y > -0.20 && p.y < 0.40 &&
              p.z > -0.20);  // Z座標チェック追加
    };
    auto is_duplicate = [&](const geometry_msgs::msg::Point & p) {
      for (const auto & t : targets) {
        double dx = p.x - t.position.x;
        double dy = p.y - t.position.y;
        double dz = p.z - t.position.z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < DUPLICATE_DETECTION_THRESHOLD) {
          return true;
        }
      }
      return false;
    };
    auto is_in_drop_zone = [&](const geometry_msgs::msg::Point & p) {
      // 配置場所は常に検出対象外
      for (const auto & kv : place_poses) {
        const auto & drop_p = kv.second.position;
        if (point_distance(p, drop_p) < PLACE_EXCLUSION_RADIUS) {
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

          // 位置の妥当性チェック
          if (!in_work_area(p)) {
            RCLCPP_INFO(LOGGER, "Ignoring object outside work area (%.3f, %.3f, %.3f) [X: %.2f-%.2f, Y: %.2f-%.2f, Z > %.2f]",
                       p.x, p.y, p.z, 0.00, 0.50, -0.40, 0.40, -0.20);
            continue;  // エリア外の物体はスキップ
          }

          // 極端に遠い位置をフィルタリング（カメラから1m以上遠い場合は除外）
          double distance = std::sqrt(p.x * p.x + p.y * p.y);
          if (distance > 0.50) {
            RCLCPP_INFO(LOGGER, "Ignoring object too far from robot (%.3f, %.3f, distance: %.3f > 0.50)",
                        p.x, p.y, distance);
            continue;
          }

          if (is_in_drop_zone(p)) {
            RCLCPP_INFO(LOGGER, "Ignoring detection near drop zone (%.3f, %.3f, %.3f)", p.x, p.y, p.z);
            continue;
          }

          if (is_duplicate(p)) {
            RCLCPP_INFO(LOGGER, "Skipped duplicate detection near (%.3f, %.3f, %.3f)", p.x, p.y, p.z);
            continue;
          }

          // 新しい物体を発見
          TargetInfo tgt;
          tgt.position = p;
          tgt.orientation = scan_poses[i].orientation;
          tgt.color = detection.color;
          tgt.detected_z = p.z;  // 検出時のZ座標を保存
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
    const double PICK_Z_OFFSET = 0.005;  // 検出Z座標からのオフセット（5mm下げる）
    const double PICK_Z_ABOVE = 0.20;    // ホバー高さ
    const double PICK_Z_LIFT = 0.30;     // 運搬時の持ち上げ高さ（配置済みキューブとの衝突回避）
    auto make_pick_pose = [&](const TargetInfo & tgt, double z) {
      geometry_msgs::msg::Pose pose;
      pose.position = tgt.position;
      pose.position.z = z;
      pose.orientation = tgt.orientation;
      return pose;
    };

    // ピック高さ計算の基準（テーブル中心高さを 0 とし、立方体サイズを考慮）
    const double TABLE_HEIGHT = 0.0;
    const double CUBE_HALF = 0.025;          // 5cm立方体の半分
    const double GRAB_CLEARANCE = 0.010;     // 1cmの余裕をもって掴む
    const double GRIPPER_LENGTH = 0.13;      // 手首から指先までの長さ（要調整）
    auto compute_pick_height = [&](double detected_z) {
      const double MIN_VALID_Z = -0.10;
      const double MAX_VALID_Z = 0.25;
      if (detected_z < MIN_VALID_Z || detected_z > MAX_VALID_Z) {
        return std::optional<double>{};
      }
      // 検出Zのずれ分だけ補正しつつ、物理的に無理のない範囲に収める
      // 指先が物体中心少し上に来る位置を手首座標に変換する
      double nominal = TABLE_HEIGHT + CUBE_HALF + GRAB_CLEARANCE + GRIPPER_LENGTH;
      double delta = detected_z - TABLE_HEIGHT;
      double pick = nominal + delta;
      // クランプ範囲も手首基準で確保
      const double MIN_PICK = TABLE_HEIGHT + 0.015 + GRIPPER_LENGTH;  // 1.5cm + 指長
      const double MAX_PICK = TABLE_HEIGHT + 0.15 + GRIPPER_LENGTH;   // 15cm + 指長
      pick = std::clamp(pick, MIN_PICK, MAX_PICK);
      return std::optional<double>(pick);
    };

    for (size_t idx = 0; idx < targets.size(); ++idx) {
      auto & tgt = targets[idx];
      std::string color_name = (tgt.color == Color::BLUE) ? "Blue" :
                               (tgt.color == Color::YELLOW) ? "Yellow" : "Green";
      RCLCPP_INFO(LOGGER, "Processing target %zu/%zu (%s)", idx + 1, targets.size(), color_name.c_str());

      // 動的ピック高さ計算（信頼できないZはスキップ）
      auto pick_height_opt = compute_pick_height(tgt.detected_z);
      if (!pick_height_opt.has_value()) {
        RCLCPP_WARN(LOGGER, "Detected Z (%.3f) is out of valid range; skipping target %zu",
                    tgt.detected_z, idx + 1);
        continue;
      }
      double dynamic_pick_z = pick_height_opt.value();

      auto pick_pose_above = make_pick_pose(tgt, PICK_Z_ABOVE);
      auto pick_pose = make_pick_pose(tgt, dynamic_pick_z);

      // 接近（ホバー）
      move_group_arm.setStartStateToCurrentState();
      if (!executeCartesianPath(move_group_arm, pick_pose_above, 0.01)) {
        RCLCPP_WARN(LOGGER, "Failed to move to hover for target %zu, skipping", idx + 1);
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(500));

      // 近距離再検出で補正（複数回測定の平均化でノイズ低減、ターゲット色優先）
      RCLCPP_INFO(LOGGER, "Performing averaged close-range detection for target %zu (wanted %s)",
                  idx + 1, color_name.c_str());
      auto refined = color_detector_node->getAveragedDetection(
        5, 200, tgt.color, tgt.position, 0.10);  // 5回測定、200ms間隔、元位置近傍のみ採用
      if (!refined.detected || refined.color != tgt.color) {
        RCLCPP_WARN(LOGGER, "Refinement did not return target color; skipping target %zu", idx + 1);
        continue;
      }

      if (!in_work_area(refined.pose.position)) {
        RCLCPP_WARN(LOGGER, "Refine result out of work area; skipping target %zu", idx + 1);
        continue;
      }

      // Accept refinement strictly
      tgt.position = refined.pose.position;
      tgt.detected_z = refined.pose.position.z;

      // ピック高さを再計算（信頼できないZはスキップ）
      auto refined_pick_height = compute_pick_height(tgt.detected_z);
      if (!refined_pick_height.has_value()) {
        RCLCPP_WARN(LOGGER, "Refined Z (%.3f) out of valid range; skipping target %zu",
                    tgt.detected_z, idx + 1);
        continue;
      }
      dynamic_pick_z = refined_pick_height.value();

      pick_pose_above = make_pick_pose(tgt, PICK_Z_ABOVE);
      pick_pose = make_pick_pose(tgt, dynamic_pick_z);
      move_group_arm.setStartStateToCurrentState();
      if (!executeCartesianPath(move_group_arm, pick_pose_above, 0.01)) {
        RCLCPP_WARN(LOGGER, "Failed to adjust hover after refinement for target %zu", idx + 1);
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(200));

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

      // 上昇（配置済みキューブとの衝突を避けるため高めに持ち上げ）
      auto lift_pose = make_pick_pose(tgt, PICK_Z_LIFT);
      move_group_arm.setStartStateToCurrentState();
      if (!executeCartesianPath(move_group_arm, lift_pose, 0.01)) {
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
