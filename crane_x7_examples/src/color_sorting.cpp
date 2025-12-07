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
#include <random>

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
enum class Color
{
  NONE,
  BLUE,
  YELLOW,
  GREEN
};

// 検出結果
struct DetectionResult
{
  Color color;
  geometry_msgs::msg::Pose pose;
  bool detected;
  // ピクセル座標（カメラ中央合わせ用）
  double pixel_x;
  double pixel_y;
  // カメラ画像サイズ（中央計算用）
  int image_width;
  int image_height;
  // オブジェクトの傾き角度（度、minAreaRectで検出）
  double angle_deg;
};

// バッチスキャンで溜めるターゲット情報
struct TargetInfo
{
  geometry_msgs::msg::Point position;
  geometry_msgs::msg::Quaternion orientation;
  Color color;
  double detected_z;  // 検出時の実際のZ座標（動的ピック高さ調整用）
  double yaw_angle_deg;  // オブジェクトのYaw角度（度、グリッパー回転用）
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
    {Color::BLUE, "0.0 0.0 1.0 1"},
    {Color::YELLOW, "1.0 1.0 0.0 1"},
    {Color::GREEN, "0.0 1.0 0.0 1"}
  };
  auto it = color_rgba.find(color);
  std::string rgba = (it != color_rgba.end()) ? it->second : "0.5 0.5 0.5 1";

  RCLCPP_INFO(LOGGER, "Spawning new object at (%.2f, %.2f, %.2f)", x, y, z);
  char cmd[2048];
  char name[64];
  snprintf(name, sizeof(name), "color_cube_%d", spawn_count++);

  snprintf(
    cmd, sizeof(cmd),
    "ros2 run ros_gz_sim create -world default -name '%s' "
    "-x %f -y %f -z %f "
    "-string '<sdf version=\"1.6\"><model name=\"%s\">"
    "<static>false</static>"
    "<link name=\"link\">"
    "<inertial><mass>0.5</mass>"
    "<inertia><ixx>0.0002</ixx><iyy>0.0002</iyy><izz>0.0002</izz>"
    "<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>"
    "</inertial>"
    "<collision name=\"collision\"><geometry><box><size>0.05 0.05 0.05</size></box></geometry>"
    "</collision>"
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


// 色名を文字列に変換するヘルパー
std::string colorToString(Color color)
{
  switch (color) {
    case Color::BLUE: return "BLUE";
    case Color::YELLOW: return "YELLOW";
    case Color::GREEN: return "GREEN";
    default: return "UNKNOWN";
  }
}

// 色検出クラス
class ColorDetector : public rclcpp::Node
{
public:
  // verbose_loggingフラグ: falseにすると検出ログを抑制
  bool verbose_logging_ = false;  // デバッグ完了後はfalseに

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

    RCLCPP_INFO(
      LOGGER, "Averaged %zu/%zu samples for better accuracy (color-filtered)",
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
  DetectionResult latest_detection_{Color::NONE, geometry_msgs::msg::Pose(), false, 0.0, 0.0, 0, 0};
  std::vector<DetectionResult> latest_detections_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // HSV範囲パラメータ（実行時調整可能）
  struct HSVRange
  {
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

    for (const auto & color : colors_to_detect) {
      const auto & range = hsv_ranges_[color];
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

      // 全ての有効な輪郭を処理（効率的な一括検出）
      for (size_t contour_idx = 0; contour_idx < contours.size(); ++contour_idx) {
        double area = cv::contourArea(contours[contour_idx]);
        if (area <= 1000) {  // 面積閾値以下はノイズとしてスキップ
          continue;
        }

        // この輪郭のモーメント計算
        cv::Moments moment = cv::moments(contours[contour_idx]);
        if (moment.m00 <= 0) {
          continue;
        }
        double d_area = moment.m00;

        // minAreaRectで傾き角度を検出
        cv::RotatedRect rotated_rect = cv::minAreaRect(contours[contour_idx]);
        // OpenCVのminAreaRect角度は-90〜0度の範囲で返される
        // 正方形/立方体の場合、45度を超えていたら90度補正して0度に近づける
        double detected_angle = rotated_rect.angle;
        // カメラ画像の座標系からロボット座標系への変換
        // 画像X軸→ロボットY軸（横方向）、画像Y軸→ロボットX軸（前後方向）
        // minAreaRectの角度は水平方向からの回転角度
        // 角度を-45〜45度の範囲に正規化（キューブは90度周期で対称）
        // OpenCV 4.x: 0〜90度, OpenCV 3.x: -90〜0度
        if (detected_angle > 45.0) {
          detected_angle -= 90.0;
        } else if (detected_angle < -45.0) {
          detected_angle += 90.0;
        }
        // verboseモードで角度をログ出力
        if (verbose_logging_) {
          RCLCPP_INFO(
            LOGGER, "  [ANGLE] minAreaRect angle: %.1f deg (size: %.1f x %.1f)",
            detected_angle, rotated_rect.size.width, rotated_rect.size.height);
        }

        // 物体検出成功
        image_geometry::PinholeCameraModel camera_model;
        camera_model.fromCameraInfo(camera_info_);

        // カメラ内部パラメータの妥当性チェック
        // cx/cyが画像中心付近にあることを確認（不正なcamera_infoをスキップ）
        double expected_cx = camera_info_->width / 2.0;
        double expected_cy = camera_info_->height / 2.0;
        double cx_error = std::abs(camera_model.cx() - expected_cx);
        double cy_error = std::abs(camera_model.cy() - expected_cy);
        if (cx_error > 50.0 || cy_error > 50.0) {
          if (verbose_logging_) {
            RCLCPP_WARN(
              LOGGER,
              "Bad camera_info: cx=%.1f(exp %.1f), cy=%.1f(exp %.1f), skip",
              camera_model.cx(), expected_cx, camera_model.cy(), expected_cy);
          }
          continue;
        }

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
                  if (v > 0.0) {vals.push_back(v);}
                } else if (depth_image_->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
                  float v = cv_depth->image.at<float>(yy, xx);
                  if (std::isfinite(v) && v > 0.0f) {vals.push_back(static_cast<double>(v));}
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

        // デバッグ: 深度値とray情報をログ出力
        if (verbose_logging_) {
          // カメラ内部パラメータを確認
          RCLCPP_INFO(
            LOGGER, "CAM_INTRINSICS: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
            camera_model.fx(), camera_model.fy(), camera_model.cx(), camera_model.cy());
          // ピクセル座標の変化を追跡
          RCLCPP_INFO(
            LOGGER, "PIXEL_COORDS: raw(%.1f,%.1f) rect(%.1f,%.1f) final(%d,%d)",
            pixel_x, pixel_y, rect_point.x, rect_point.y, px, py);
          // 手動計算との比較
          double manual_ray_x = (rect_point.x - camera_model.cx()) / camera_model.fx();
          double manual_ray_y = (rect_point.y - camera_model.cy()) / camera_model.fy();
          RCLCPP_INFO(
            LOGGER, "RAY_COMPARE: lib(%.3f,%.3f,%.3f) manual(%.3f,%.3f,1.000)",
            ray.x, ray.y, ray.z, manual_ray_x, manual_ray_y);
          RCLCPP_INFO(
            LOGGER, "DEPTH_DEBUG: pixel(%d,%d) raw_depth=%.3f center_dist=%.3f",
            px, py, median_opt.value(), center_distance);
        }

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
          const std::vector<std::string> candidate_sources = build_candidate_frames(
            p_cam.header.frame_id);
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
            // ピクセル座標を保存（カメラ中央合わせ用）
            latest_detection_.pixel_x = pixel_x;
            latest_detection_.pixel_y = pixel_y;
            latest_detection_.image_width = static_cast<int>(camera_info_->width);
            latest_detection_.image_height = static_cast<int>(camera_info_->height);
            // 傾き角度を保存（グリッパー回転用）
            latest_detection_.angle_deg = detected_angle;

            // Avoid adding near-identical detections within the same frame
            bool close_to_existing = false;
            for (const auto & d : latest_detections_) {
              double dx = d.pose.position.x - object_pose.position.x;
              double dy = d.pose.position.y - object_pose.position.y;
              double dz = d.pose.position.z - object_pose.position.z;
              if (std::sqrt(dx * dx + dy * dy + dz * dz) < DUPLICATE_DETECTION_THRESHOLD &&
                d.color == color)
              {
                close_to_existing = true;
                break;
              }
            }
            if (!close_to_existing) {
              latest_detections_.push_back(latest_detection_);
            }

            // verboseモードでのみ検出ログを出力（通常は抑制してログノイズを削減）
            if (verbose_logging_) {
              std::string color_name = (color == Color::BLUE) ? "Blue" :
                (color == Color::YELLOW) ? "Yellow" : "Green";
              RCLCPP_INFO(
                LOGGER,
                "%s@%s (%.3f,%.3f,%.3f) [cam(%.3f,%.3f,%.3f)]",
                color_name.c_str(), used_target_frame.c_str(),
                object_pose.position.x, object_pose.position.y,
                object_pose.position.z,
                camera_pos.x, camera_pos.y, camera_pos.z);
            }
            // Keep searching for other objects in the same frame
          } else {
            RCLCPP_WARN(
              LOGGER,
              "Transform failed for frame %s. Tried optical/prefix variants. Skipping detection.",
              msg->header.frame_id.c_str());
          }
        }
      }  // end of for contour_idx
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
  node_options.parameter_overrides(
  {
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
  const double GRIPPER_CLOSE = angles::from_degrees(20.0);  // 5cmキューブ用（しっかり把持）

  // スキャン用観察姿勢（上方から見下ろす・中央のみ）
  std::vector<geometry_msgs::msg::Pose> scan_poses = {
    createPose(0.25, 0.00, 0.35, -180, 0, 90)  // 中央
  };

  // 配置場所（色ごとに異なる位置）
  // 全色を作業エリア外に配置（検証スキャンで再検出されないように）
  // 作業エリア: X: 0.10-0.40, Y: -0.15-0.25
  // 左奥（Y > 0.25、作業エリア外）
  // 右奥（Y < -0.15、作業エリア外）
  // 前方（X > 0.40、作業エリア外）
  std::map<Color, geometry_msgs::msg::Pose> place_poses = {
    {Color::YELLOW, createPose(0.30, 0.30, 0.23, -180, 0, 90)},
    {Color::BLUE, createPose(0.30, -0.30, 0.23, -180, 0, 90)},
    {Color::GREEN, createPose(0.45, 0.00, 0.23, -180, 0, 90)}
  };

  // ドロップ用に同じ高さで移動のみ
  std::map<Color, geometry_msgs::msg::Pose> place_poses_above = place_poses;

  RCLCPP_INFO(LOGGER, "Starting color sorting demo");
  // 初期姿勢（直接カメラ観察姿勢へ移動）
  RCLCPP_INFO(LOGGER, "Moving to camera observation posture");
  move_group_arm.setStartStateToCurrentState();
  move_group_arm.setJointValueTarget(camera_start_joints);
  move_group_arm.move();

  // シーンを初期化して赤・青・緑をまとめて投入
  RCLCPP_INFO(LOGGER, "Spawning cubes with random colors (BLUE/YELLOW/GREEN)");

  // 5個のオブジェクトをスポーン: ランダムに青/黄/緑を割り当て
  std::vector<Color> available_colors = {Color::BLUE, Color::YELLOW, Color::GREEN};
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, available_colors.size() - 1);

  std::vector<Color> spawn_sequence;
  for (int i = 0; i < 5; ++i) {
    spawn_sequence.push_back(available_colors[dist(gen)]);
  }

  // 生成された色をログ出力
  std::string color_str;
  for (size_t i = 0; i < spawn_sequence.size(); ++i) {
    if (i > 0) {color_str += ", ";}
    color_str += colorToString(spawn_sequence[i]);
  }
  RCLCPP_INFO(LOGGER, "Spawn colors: [%s]", color_str.c_str());

  std::vector<geometry_msgs::msg::Point> spawn_positions = {
    []() {geometry_msgs::msg::Point p; p.x = 0.22; p.y = -0.12; p.z = 1.10; return p;}(),
    []() {geometry_msgs::msg::Point p; p.x = 0.30; p.y = -0.06; p.z = 1.10; return p;}(),
    []() {geometry_msgs::msg::Point p; p.x = 0.22; p.y = 0.12;  p.z = 1.10; return p;}(),
    []() {geometry_msgs::msg::Point p; p.x = 0.30; p.y = 0.06;  p.z = 1.10; return p;}(),
    []() {geometry_msgs::msg::Point p; p.x = 0.26; p.y = 0.00;  p.z = 1.10; return p;}()
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
  // 動的オブジェクト数対応: 安全上限のみ設定し、検出される限り処理を続ける
  const int MAX_TOTAL_OBJECTS = 10;  // 安全上限（無限ループ防止）
  int total_processed = 0;
  while (rclcpp::ok() && total_processed < MAX_TOTAL_OBJECTS) {
    // フェーズ1: 一括スキャンしてターゲットリストを作成
    std::vector<TargetInfo> targets;
    auto in_work_area = [](const geometry_msgs::msg::Point & p) {
        // 作業エリア内の物体のみをターゲットに
        // X/Y範囲でフィルタリング、Z範囲は緩め（深度エラーはZ-FIXで補正）
        // カメラは物理的にテーブル下を見えないので、負のZは深度エラー
        return p.x > 0.10 && p.x < 0.40 &&
               p.y > -0.15 && p.y < 0.25 &&
               p.z > -0.05 && p.z < 0.15;  // Z範囲は緩め（深度エラーはZ-FIXで対応）
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

    RCLCPP_INFO(LOGGER, "");
    RCLCPP_INFO(LOGGER, "========================================");
    RCLCPP_INFO(LOGGER, "  PHASE 1: SCANNING WORK AREA");
    RCLCPP_INFO(LOGGER, "========================================");
    for (size_t i = 0; i < scan_poses.size(); ++i) {
      RCLCPP_INFO(LOGGER, "  Scanning position %zu/%zu...", i + 1, scan_poses.size());
      move_group_arm.setStartStateToCurrentState();
      move_group_arm.setPoseTarget(scan_poses[i]);
      if (!move_group_arm.move()) {
        RCLCPP_WARN(LOGGER, "Scan pose %zu failed to reach, continuing", i + 1);
        continue;
      }
      // カメラ安定待ち（ロボット停止後、カメラ画像が安定するまで待機）
      RCLCPP_INFO(LOGGER, "  Waiting for camera to stabilize...");
      rclcpp::sleep_for(std::chrono::milliseconds(1500));
      RCLCPP_INFO(LOGGER, "  Starting detection");

      // 視野内の全ての物体を検出するまで繰り返す
      int consecutive_duplicate_or_invalid = 0;
      int total_attempts = 0;
      const int MAX_ATTEMPTS = 40;  // 最大試行回数（10個のオブジェクト検出に対応）

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
            RCLCPP_INFO(
              LOGGER, "Ignoring object outside work area (%.3f, %.3f, %.3f)",
              p.x, p.y, p.z);
            continue;  // エリア外の物体はスキップ
          }

          // 極端に遠い位置をフィルタリング（カメラから1m以上遠い場合は除外）
          double distance = std::sqrt(p.x * p.x + p.y * p.y);
          if (distance > 0.50) {
            RCLCPP_INFO(
              LOGGER, "Ignoring object too far from robot (%.3f, %.3f, distance: %.3f > 0.50)",
              p.x, p.y, distance);
            continue;
          }

          if (is_duplicate(p)) {
            RCLCPP_INFO(
              LOGGER, "Skipped duplicate detection near (%.3f, %.3f, %.3f)", p.x, p.y,
              p.z);
            continue;
          }

          // 新しい物体を発見 - 5回測定で平均化して精度向上
          std::string color_name = (detection.color == Color::BLUE) ? "Blue" :
            (detection.color == Color::YELLOW) ? "Yellow" : "Green";
          RCLCPP_INFO(
            LOGGER, "Found new %s object, performing averaged detection...",
            color_name.c_str());

          auto averaged = color_detector_node->getAveragedDetection(
            5, 200, detection.color, p, 0.08);  // 5回測定、200ms間隔、同色のみ、8cm以内

          TargetInfo tgt;
          if (averaged.detected && averaged.color == detection.color) {
            tgt.position = averaged.pose.position;
            tgt.detected_z = averaged.pose.position.z;
            RCLCPP_INFO(
              LOGGER, "Using averaged position (%.3f, %.3f, %.3f)",
              tgt.position.x, tgt.position.y, tgt.position.z);
          } else {
            tgt.position = p;
            tgt.detected_z = p.z;
            RCLCPP_WARN(
              LOGGER, "Averaged detection failed, using single detection (%.3f, %.3f, %.3f)",
              p.x, p.y, p.z);
          }
          tgt.orientation = scan_poses[i].orientation;
          tgt.color = detection.color;
          tgt.yaw_angle_deg = detection.angle_deg;  // 傾き角度を保存
          targets.push_back(tgt);

          RCLCPP_INFO(
            LOGGER, "Added target %s at (%.3f, %.3f, %.3f) angle=%.1f° from scan pose %zu",
            color_name.c_str(), tgt.position.x, tgt.position.y, tgt.position.z, tgt.yaw_angle_deg,
            i + 1);

          added_new = true;
        }

        if (added_new) {
          // 新しい物体を見つけたのでカウンタをリセット
          consecutive_duplicate_or_invalid = 0;
        } else {
          consecutive_duplicate_or_invalid++;
        }
      }
      RCLCPP_INFO(
        LOGGER, "Scan pose %zu complete. Total targets found: %zu (attempts: %d)",
        i + 1, targets.size(), total_attempts);
    }

    if (targets.empty()) {
      RCLCPP_INFO(LOGGER, "");
      RCLCPP_INFO(LOGGER, "  SCAN COMPLETE: No targets found");
      RCLCPP_INFO(LOGGER, "========================================");
      break;
    }

    // 処理順序: 青 → 黄 → 緑 の順にソート
    // 緑は作業エリア内に集めるため最後に処理（先に青・黄を作業エリア外に移動）
    std::map<Color, int> color_priority = {
      {Color::BLUE, 1},
      {Color::YELLOW, 2},
      {Color::GREEN, 3}
    };
    std::sort(
      targets.begin(), targets.end(), [&](const TargetInfo & a, const TargetInfo & b) {
        return color_priority[a.color] < color_priority[b.color];
      });

    // スキャン結果のサマリーを表示
    RCLCPP_INFO(LOGGER, "");
    RCLCPP_INFO(LOGGER, "  SCAN COMPLETE: %zu targets found", targets.size());
    RCLCPP_INFO(LOGGER, "  ----------------------------------------");
    for (size_t i = 0; i < targets.size(); ++i) {
      RCLCPP_INFO(
        LOGGER, "    %zu. %s at (%.2f, %.2f, %.2f)",
        i + 1, colorToString(targets[i].color).c_str(),
        targets[i].position.x, targets[i].position.y, targets[i].position.z);
    }
    RCLCPP_INFO(LOGGER, "  ----------------------------------------");
    RCLCPP_INFO(LOGGER, "  Processing order: BLUE -> YELLOW -> GREEN");
    RCLCPP_INFO(LOGGER, "========================================");

    // フェーズ2: 順次ピック&プレース
    RCLCPP_INFO(LOGGER, "");
    RCLCPP_INFO(LOGGER, "========================================");
    RCLCPP_INFO(LOGGER, "  PHASE 2: PICK AND PLACE");
    RCLCPP_INFO(LOGGER, "========================================");

    const double PICK_Z_OFFSET = 0.005;  // 検出Z座標からのオフセット（5mm下げる）
    const double PICK_Z_ABOVE = 0.20;    // ホバー高さ
    const double PICK_Z_LIFT = 0.30;     // 運搬時の持ち上げ高さ（配置済みキューブとの衝突回避）
    // グリッパー-カメラオフセット（カメラ中央とグリッパー中心のズレ補正）
    // TF: gripper_base_link → camera_optical_frame = Y:3.2cm, Z:1.9cm
    // 負の値 = ロボット寄り（手前）に移動
    // テスト結果: +0.025だと上側（奥）にずれた → 負の値が正しい
    const double GRIPPER_CAMERA_OFFSET_X = 0.0;  // X方向オフセット無効化（斜め掴み問題調査用）
    // グリッパー角度の制限（遠心力でオブジェクトが外れるのを防ぐ）
    const double MAX_GRIPPER_YAW = 45.0;  // 最大回転角度（度）
    const double YAW_SCALE = 1.0;          // 角度のスケーリング係数（1.0 = そのまま適用）
    auto make_pick_pose = [&](const TargetInfo & tgt, double z) {
        geometry_msgs::msg::Pose pose;
        pose.position = tgt.position;
        pose.position.x += GRIPPER_CAMERA_OFFSET_X;  // カメラ-グリッパーオフセット補正
        pose.position.z = z;
        // 傾き角度をグリッパーのyaw回転に反映
        // カメラ画像の角度（水平からの回転）→ロボットのZ軸回転（yaw）
        tf2::Quaternion q;
        // 角度をスケーリングして制限
        double scaled_yaw = tgt.yaw_angle_deg * YAW_SCALE;
        if (scaled_yaw > MAX_GRIPPER_YAW) {scaled_yaw = MAX_GRIPPER_YAW;}
        if (scaled_yaw < -MAX_GRIPPER_YAW) {scaled_yaw = -MAX_GRIPPER_YAW;}
        // 基本姿勢: Roll=180°(下向き), Pitch=0°, Yaw=制限付き検出角度
        q.setRPY(
          angles::from_degrees(180.0),  // Roll: 下向き
          angles::from_degrees(0.0),  // Pitch: 前後の傾きなし
          angles::from_degrees(-scaled_yaw)  // Yaw: スケーリング＆制限付き角度
        );
        pose.orientation = tf2::toMsg(q);
        return pose;
      };

    // ピック高さ計算の基準（テーブル中心高さを 0 とし、立方体サイズを考慮）
    const double TABLE_HEIGHT = 0.0;
    const double CUBE_HALF = 0.025;          // 5cm立方体の半分
    const double GRAB_CLEARANCE = -0.025;    // 指先がキューブ中央に来るよう2.5cm下げる
    const double GRIPPER_LENGTH = 0.13;      // 手首から指先までの長さ（要調整）
    auto compute_pick_height = [&](double detected_z) {
        const double MIN_VALID_Z = -0.10;
        const double MAX_VALID_Z = 0.25;
        if (detected_z < MIN_VALID_Z || detected_z > MAX_VALID_Z) {
          return std::optional<double>{};
        }
        // 小さな負のZ値（センサーノイズ）は許容して動的計算に使用
        // 大きな負の値のみクランプ（深度検出の明らかなエラー）
        // Z値を許容範囲にクランプ（センサーノイズ対策）
        const double Z_MIN_CLAMP = -0.03;  // 3cm以内の負の値は許容
        const double Z_MAX_CLAMP = -0.015;  // 青の高さ誤検出対策（Yellow/Greenと同レベルに）
        if (detected_z < Z_MIN_CLAMP) {
          RCLCPP_WARN(
            LOGGER, "    [Z-FIX] Z too low (%.3f), clamping to %.3f", detected_z,
            Z_MIN_CLAMP);
          detected_z = Z_MIN_CLAMP;
        } else if (detected_z > Z_MAX_CLAMP) {
          RCLCPP_WARN(
            LOGGER, "    [Z-FIX] Z too high (%.3f), clamping to %.3f", detected_z,
            Z_MAX_CLAMP);
          detected_z = Z_MAX_CLAMP;
        }
        // 検出Zのずれ分だけ補正しつつ、物理的に無理のない範囲に収める
        // 指先が物体中心少し上に来る位置を手首座標に変換する
        double nominal = TABLE_HEIGHT + CUBE_HALF + GRAB_CLEARANCE + GRIPPER_LENGTH;
        double delta = detected_z - TABLE_HEIGHT;
        double pick = nominal + delta;
        // クランプ範囲も手首基準で確保（シミュレーション用に下限を緩和）
        const double MIN_PICK = 0.0;  // 下限なし
        const double MAX_PICK = TABLE_HEIGHT + 0.15 + GRIPPER_LENGTH;  // 15cm + 指長
        pick = std::clamp(pick, MIN_PICK, MAX_PICK);
        return std::optional<double>(pick);
      };

    for (size_t idx = 0; idx < targets.size(); ++idx) {
      auto & tgt = targets[idx];
      std::string color_name = colorToString(tgt.color);

      RCLCPP_INFO(LOGGER, "");
      RCLCPP_INFO(
        LOGGER, "--- Target %zu/%zu: %s ---", idx + 1, targets.size(),
        color_name.c_str());
      RCLCPP_INFO(
        LOGGER, "    Position: (%.3f, %.3f, %.3f)", tgt.position.x, tgt.position.y,
        tgt.position.z);
      // Yaw角度補正の詳細ログ
      double scaled_yaw = tgt.yaw_angle_deg * YAW_SCALE;
      double limited_yaw = scaled_yaw;
      if (limited_yaw > MAX_GRIPPER_YAW) {limited_yaw = MAX_GRIPPER_YAW;}
      if (limited_yaw < -MAX_GRIPPER_YAW) {limited_yaw = -MAX_GRIPPER_YAW;}
      RCLCPP_INFO(
        LOGGER, "    [YAW] 検出角度: %.1f° → スケール(x%.1f): %.1f° → 制限(±%.0f°): %.1f°",
        tgt.yaw_angle_deg, YAW_SCALE, scaled_yaw, MAX_GRIPPER_YAW, limited_yaw);
      if (std::abs(limited_yaw) > 1.0) {
        RCLCPP_INFO(LOGGER, "    [YAW] グリッパー回転: %.1f° 適用", -limited_yaw);
      } else {
        RCLCPP_INFO(LOGGER, "    [YAW] グリッパー回転: なし（角度が小さいため）");
      }

      // 動的ピック高さ計算（信頼できないZはスキップ）
      auto pick_height_opt = compute_pick_height(tgt.detected_z);
      if (!pick_height_opt.has_value()) {
        RCLCPP_WARN(LOGGER, "    [SKIP] Detected Z (%.3f) is out of valid range", tgt.detected_z);
        continue;
      }
      double dynamic_pick_z = pick_height_opt.value();

      auto pick_pose_above = make_pick_pose(tgt, PICK_Z_ABOVE);
      auto pick_pose = make_pick_pose(tgt, dynamic_pick_z);

      // Step 1: 接近（ホバー）
      RCLCPP_INFO(LOGGER, "    Step 1/6: Moving to hover position");
      move_group_arm.setStartStateToCurrentState();
      if (!executeCartesianPath(move_group_arm, pick_pose_above, 0.01)) {
        RCLCPP_WARN(LOGGER, "    [FAIL] Could not reach hover position, skipping");
        continue;
      }
      // カメラ安定待ち
      rclcpp::sleep_for(std::chrono::milliseconds(1000));

      // Step 2: カメラ中央合わせ（ビジュアルサーボイング）
      RCLCPP_INFO(LOGGER, "    Step 2/6: Camera centering (tolerance: +/-%d px)", 5);
      const int CENTER_MAX_ATTEMPTS = 5;
      const int CENTER_TOLERANCE_PX = 5;  // ±5ピクセル精度
      bool centering_success = false;
      geometry_msgs::msg::Pose current_hover_pose = pick_pose_above;

      for (int centering_attempt = 0; centering_attempt < CENTER_MAX_ATTEMPTS;
        centering_attempt++)
      {
        color_detector_node->resetDetection();
        rclcpp::sleep_for(std::chrono::milliseconds(300));
        auto detection = color_detector_node->getLatestDetection();

        if (!detection.detected || detection.color != tgt.color) {
          RCLCPP_WARN(
            LOGGER, "Centering attempt %d: target color not detected, using current position",
            centering_attempt + 1);
          break;
        }

        // 別のターゲットを誤検出していないかチェック
        // スキャン位置から5cm以上離れたら別ターゲットの可能性
        const double MAX_SCAN_DISTANCE = 0.05;
        double dist_from_scan = std::sqrt(
          std::pow(detection.pose.position.x - tgt.position.x, 2) +
          std::pow(detection.pose.position.y - tgt.position.y, 2));
        if (dist_from_scan > MAX_SCAN_DISTANCE) {
          RCLCPP_WARN(
            LOGGER, "[CENTERING] 誤検出? dist=%.1fmm > %.1fmm",
            dist_from_scan * 1000, MAX_SCAN_DISTANCE * 1000);
          RCLCPP_WARN(
            LOGGER, "  scan:(%.3f,%.3f) -> det:(%.3f,%.3f)",
            tgt.position.x, tgt.position.y,
            detection.pose.position.x, detection.pose.position.y);
          RCLCPP_INFO(LOGGER, "[CENTERING] スキップしてスキャン位置を使用");
          centering_success = true;  // スキャン位置で継続
          break;
        }

        // カメラ中央からのピクセルオフセット計算
        double cx = detection.image_width / 2.0;
        double cy = detection.image_height / 2.0;
        double dx_px = detection.pixel_x - cx;
        double dy_px = detection.pixel_y - cy;

        RCLCPP_INFO(
          LOGGER, "Centering attempt %d: pixel offset (%.1f, %.1f) px",
          centering_attempt + 1, dx_px, dy_px);

        // 許容範囲内なら完了
        if (std::abs(dx_px) <= CENTER_TOLERANCE_PX && std::abs(dy_px) <= CENTER_TOLERANCE_PX) {
          RCLCPP_INFO(
            LOGGER, "Centering complete: object within ±%d px tolerance",
            CENTER_TOLERANCE_PX);
          // 検出位置で最終更新
          tgt.position = detection.pose.position;
          tgt.detected_z = detection.pose.position.z;
          // 角度も再検出値で更新（ホバー直上からの検出がより正確）
          double old_yaw = tgt.yaw_angle_deg;
          tgt.yaw_angle_deg = detection.angle_deg;
          RCLCPP_INFO(
            LOGGER, "    [CENTERING] 角度更新: %.1f° → %.1f°（差: %.1f°）",
            old_yaw, tgt.yaw_angle_deg, tgt.yaw_angle_deg - old_yaw);
          centering_success = true;
          break;
        }

        // ピクセルオフセットを実世界座標に変換
        // 簡易計算: 検出された3D位置と現在のホバー位置の差分を使用
        // （カメラの焦点距離を使った計算の代わりに、直接検出位置を使用）
        double move_x = detection.pose.position.x - current_hover_pose.position.x;
        double move_y = detection.pose.position.y - current_hover_pose.position.y;

        // 移動量を制限（安全のため）
        const double MAX_MOVE = 0.05;  // 5cm以内
        move_x = std::clamp(move_x, -MAX_MOVE, MAX_MOVE);
        move_y = std::clamp(move_y, -MAX_MOVE, MAX_MOVE);

        // センタリング移動量の詳細ログ
        double move_dist = std::sqrt(move_x * move_x + move_y * move_y);
        RCLCPP_INFO(
          LOGGER, "    [CENTERING] %s: 移動量 (X:%.3f, Y:%.3f) m = %.1f mm",
          color_name.c_str(), move_x, move_y, move_dist * 1000);
        RCLCPP_INFO(
          LOGGER, "    [CENTERING]   ホバー位置: (%.3f, %.3f) → 検出位置: (%.3f, %.3f)",
          current_hover_pose.position.x, current_hover_pose.position.y,
          detection.pose.position.x, detection.pose.position.y);

        // 新しいホバー位置へ移動
        current_hover_pose.position.x += move_x;
        current_hover_pose.position.y += move_y;

        move_group_arm.setStartStateToCurrentState();
        if (!executeCartesianPath(move_group_arm, current_hover_pose, 0.01)) {
          RCLCPP_WARN(LOGGER, "Centering move failed, using current position");
          break;
        }
        rclcpp::sleep_for(std::chrono::milliseconds(300));

        // 位置を更新
        tgt.position = detection.pose.position;
        tgt.detected_z = detection.pose.position.z;
      }

      // ピック高さを再計算
      auto refined_pick_height = compute_pick_height(tgt.detected_z);
      if (refined_pick_height.has_value()) {
        dynamic_pick_z = refined_pick_height.value();
      }

      if (!centering_success) {
        RCLCPP_WARN(
          LOGGER, "Centering did not converge for target %zu, proceeding with best estimate",
          idx + 1);
      }

      pick_pose_above = current_hover_pose;
      pick_pose = make_pick_pose(tgt, dynamic_pick_z);
      RCLCPP_INFO(
        LOGGER, "Final pick position for target %zu: (%.3f, %.3f, %.3f)",
        idx + 1, tgt.position.x, tgt.position.y, tgt.position.z);

      // Step 3: 下降して把持位置へ
      RCLCPP_INFO(LOGGER, "    Step 3/6: Descending to pick height (z=%.3f)", dynamic_pick_z);
      move_group_arm.setStartStateToCurrentState();
      if (!executeCartesianPath(move_group_arm, pick_pose, 0.01)) {
        RCLCPP_WARN(LOGGER, "    [FAIL] Could not descend, skipping");
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(300));

      // Step 4: グリッパーを閉じて把持
      RCLCPP_INFO(LOGGER, "    Step 4/6: Closing gripper");
      gripper_joint_values[0] = GRIPPER_CLOSE;
      move_group_gripper.setJointValueTarget(gripper_joint_values);
      move_group_gripper.move();
      rclcpp::sleep_for(std::chrono::milliseconds(250));  // 把持後に少し待機して安定させる

      // Step 5: 上昇（配置済みキューブとの衝突を避けるため高めに持ち上げ）
      RCLCPP_INFO(LOGGER, "    Step 5/6: Lifting object");
      auto lift_pose = make_pick_pose(tgt, PICK_Z_LIFT);
      move_group_arm.setStartStateToCurrentState();
      bool lift_success = executeCartesianPath(move_group_arm, lift_pose, 0.01);
      if (!lift_success) {
        // Cartesian pathが失敗した場合、setPoseTargetでフォールバック
        RCLCPP_WARN(LOGGER, "    Cartesian lift failed, trying setPoseTarget fallback");
        move_group_arm.setStartStateToCurrentState();
        move_group_arm.setPoseTarget(lift_pose);
        lift_success = static_cast<bool>(move_group_arm.move());
      }
      if (!lift_success) {
        RCLCPP_WARN(LOGGER, "    [FAIL] Could not lift, releasing gripper and skipping");
        // グリッパーを開いてオブジェクトを解放（把持したまま次に進まない）
        gripper_joint_values[0] = GRIPPER_OPEN;
        move_group_gripper.setJointValueTarget(gripper_joint_values);
        move_group_gripper.move();
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(100));

      // Step 6: 配置場所へ移動して離す
      auto drop_pose = place_poses_above[tgt.color];
      RCLCPP_INFO(LOGGER, "    Step 6/6: Placing %s at destination", color_name.c_str());
      move_group_arm.setStartStateToCurrentState();
      move_group_arm.setPoseTarget(drop_pose);
      if (!move_group_arm.move()) {
        RCLCPP_WARN(
          LOGGER,
          "    [FAIL] Could not reach drop position, releasing gripper and skipping");
        // グリッパーを開いてオブジェクトを解放（把持したまま次に進まない）
        gripper_joint_values[0] = GRIPPER_OPEN;
        move_group_gripper.setJointValueTarget(gripper_joint_values);
        move_group_gripper.move();
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(100));

      gripper_joint_values[0] = GRIPPER_OPEN;
      move_group_gripper.setJointValueTarget(gripper_joint_values);
      move_group_gripper.move();
      rclcpp::sleep_for(std::chrono::milliseconds(200));

      // 処理数をカウント
      total_processed++;
      RCLCPP_INFO(
        LOGGER, "--- Target %zu/%zu: %s COMPLETE ---", idx + 1,
        targets.size(), color_name.c_str());
    }

    // 安全上限に達した場合のみループ終了（通常はターゲットなしで終了）
    if (total_processed >= MAX_TOTAL_OBJECTS) {
      RCLCPP_WARN(LOGGER, "Reached safety limit (%d objects), ending demo", MAX_TOTAL_OBJECTS);
      break;
    }

    // 次のサイクルのためにスキャン姿勢へ戻る
    move_group_arm.setStartStateToCurrentState();
    move_group_arm.setJointValueTarget(camera_start_joints);
    move_group_arm.move();
    rclcpp::sleep_for(std::chrono::milliseconds(200));
  }

  // ============================================
  // 最終検証フェーズ: 残存オブジェクトの自動再処理
  // ============================================
  RCLCPP_INFO(LOGGER, "");
  RCLCPP_INFO(LOGGER, "========================================");
  RCLCPP_INFO(LOGGER, "  PHASE 3: VERIFICATION");
  RCLCPP_INFO(LOGGER, "========================================");

  const int MAX_VERIFICATION_ATTEMPTS = 2;
  int verification_attempts = 0;

  // 最終検証フェーズ用の定数（メインループ内の定義と同じ）
  const double VERIFY_PICK_Z_ABOVE = 0.20;    // ホバー高さ
  const double VERIFY_PICK_Z_LIFT = 0.30;     // 運搬時の持ち上げ高さ
  const double VERIFY_TABLE_HEIGHT = 0.0;
  const double VERIFY_CUBE_HALF = 0.025;
  const double VERIFY_GRAB_CLEARANCE = -0.07;
  const double VERIFY_GRIPPER_LENGTH = 0.13;
  auto verify_compute_pick_height = [&](double detected_z) -> std::optional<double> {
      const double MIN_VALID_Z = -0.10;
      const double MAX_VALID_Z = 0.25;
      if (detected_z < MIN_VALID_Z || detected_z > MAX_VALID_Z) {
        return std::nullopt;
      }
      double nominal = VERIFY_TABLE_HEIGHT + VERIFY_CUBE_HALF + VERIFY_GRAB_CLEARANCE +
        VERIFY_GRIPPER_LENGTH;
      double delta = detected_z - VERIFY_TABLE_HEIGHT;
      double pick = nominal + delta;
      const double MIN_PICK = 0.0;
      const double MAX_PICK = VERIFY_TABLE_HEIGHT + 0.15 + VERIFY_GRIPPER_LENGTH;
      pick = std::clamp(pick, MIN_PICK, MAX_PICK);
      return pick;
    };

  while (rclcpp::ok() && verification_attempts < MAX_VERIFICATION_ATTEMPTS) {
    // カメラ観察姿勢へ戻る
    RCLCPP_INFO(
      LOGGER, "Verification phase %d: Moving to camera observation posture",
      verification_attempts + 1);
    move_group_arm.setStartStateToCurrentState();
    move_group_arm.setJointValueTarget(camera_start_joints);
    move_group_arm.move();
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // 作業エリア内のオブジェクトをスキャン
    color_detector_node->resetDetections();
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // 複数回検出で確実性を上げる
    std::vector<TargetInfo> remaining_targets;
    auto in_work_area_verify = [](const geometry_msgs::msg::Point & p) {
        return p.x > 0.10 && p.x < 0.40 &&
               p.y > -0.15 && p.y < 0.25 &&
               p.z > -0.05 && p.z < 0.10;
      };

    for (int scan = 0; scan < 3; scan++) {
      color_detector_node->resetDetections();
      rclcpp::sleep_for(std::chrono::milliseconds(300));
      auto detections = color_detector_node->getLatestDetections();

      for (const auto & d : detections) {
        if (!d.detected || !in_work_area_verify(d.pose.position)) {
          continue;
        }
        // 青と黄のみ再処理対象（緑は作業エリア中央に集まっているはず）
        if (d.color != Color::BLUE && d.color != Color::YELLOW) {
          continue;
        }
        // 重複チェック
        bool is_dup = false;
        for (const auto & t : remaining_targets) {
          double dx = d.pose.position.x - t.position.x;
          double dy = d.pose.position.y - t.position.y;
          if (std::sqrt(dx * dx + dy * dy) < DUPLICATE_DETECTION_THRESHOLD) {
            is_dup = true;
            break;
          }
        }
        if (!is_dup) {
          TargetInfo tgt;
          tgt.position = d.pose.position;
          tgt.orientation = scan_poses[0].orientation;
          tgt.color = d.color;
          tgt.detected_z = d.pose.position.z;
          remaining_targets.push_back(tgt);
        }
      }
    }

    if (remaining_targets.empty()) {
      RCLCPP_INFO(LOGGER, "Verification passed: no remaining blue/yellow objects in work area");
      break;
    }

    RCLCPP_WARN(
      LOGGER, "Found %zu remaining objects in work area, re-processing...",
      remaining_targets.size());

    // 再処理（簡易版: ホバー→下降→把持→配置）
    for (const auto & tgt : remaining_targets) {
      std::string color_name = (tgt.color == Color::BLUE) ? "Blue" : "Yellow";
      RCLCPP_INFO(
        LOGGER, "Re-processing %s at (%.3f, %.3f, %.3f)",
        color_name.c_str(), tgt.position.x, tgt.position.y, tgt.position.z);

      auto pick_height_opt = verify_compute_pick_height(tgt.detected_z);
      if (!pick_height_opt.has_value()) {
        continue;
      }
      double pick_z = pick_height_opt.value();

      // ホバー
      auto hover_pose = createPose(
        tgt.position.x, tgt.position.y, VERIFY_PICK_Z_ABOVE,
        -180, 0, 90);
      move_group_arm.setStartStateToCurrentState();
      if (!executeCartesianPath(move_group_arm, hover_pose, 0.01)) {
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(300));

      // 下降
      auto pick_pose = createPose(
        tgt.position.x, tgt.position.y, pick_z,
        -180, 0, 90);
      move_group_arm.setStartStateToCurrentState();
      if (!executeCartesianPath(move_group_arm, pick_pose, 0.01)) {
        continue;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(200));

      // 把持
      gripper_joint_values[0] = GRIPPER_CLOSE;
      move_group_gripper.setJointValueTarget(gripper_joint_values);
      move_group_gripper.move();
      rclcpp::sleep_for(std::chrono::milliseconds(200));

      // 持ち上げ
      auto lift_pose = createPose(
        tgt.position.x, tgt.position.y, VERIFY_PICK_Z_LIFT,
        -180, 0, 90);
      move_group_arm.setStartStateToCurrentState();
      executeCartesianPath(move_group_arm, lift_pose, 0.01);

      // 配置
      auto drop_pose = place_poses[tgt.color];
      move_group_arm.setStartStateToCurrentState();
      move_group_arm.setPoseTarget(drop_pose);
      move_group_arm.move();

      // 離す
      gripper_joint_values[0] = GRIPPER_OPEN;
      move_group_gripper.setJointValueTarget(gripper_joint_values);
      move_group_gripper.move();
      rclcpp::sleep_for(std::chrono::milliseconds(200));

      total_processed++;
    }

    verification_attempts++;
  }

  if (verification_attempts >= MAX_VERIFICATION_ATTEMPTS) {
    RCLCPP_WARN(LOGGER, "Max verification attempts reached, some objects may remain");
  }

  // 最終的にカメラ観察姿勢に戻る
  RCLCPP_INFO(LOGGER, "  Returning to observation posture...");
  move_group_arm.setStartStateToCurrentState();
  move_group_arm.setJointValueTarget(camera_start_joints);
  move_group_arm.move();

  RCLCPP_INFO(LOGGER, "");
  RCLCPP_INFO(LOGGER, "========================================");
  RCLCPP_INFO(LOGGER, "  DEMO COMPLETE");
  RCLCPP_INFO(LOGGER, "========================================");
  RCLCPP_INFO(LOGGER, "  Total objects processed: %d", total_processed);
  RCLCPP_INFO(LOGGER, "========================================");

  rclcpp::shutdown();
  return 0;
}  // NOLINT(readability/fn_size)
