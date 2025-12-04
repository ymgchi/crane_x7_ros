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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/float32.hpp"

#include "pcl_conversions/pcl_conversions.h"
#include "pcl/features/integral_image_normal.h"
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/features/organized_edge_detection.h"
#include "pcl/filters/filter.h"
#include "pcl/filters/passthrough.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/features/normal_3d_omp.h"
#include "pcl/ModelCoefficients.h"
#include "pcl/segmentation/sac_segmentation.h"

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_eigen/tf2_eigen.hpp"
#include "pcl/common/transforms.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <deque>

namespace
{
using PointT = pcl::PointXYZRGB;
constexpr float kRadPerDeg = 0.01745329251994329577f;

tf2::Quaternion normal_to_quaternion(const Eigen::Vector3f & normal)
{
  Eigen::Vector3f z_axis(0.0f, 0.0f, 1.0f);
  Eigen::Vector3f axis = z_axis.cross(normal);
  float axis_norm = axis.norm();
  if (axis_norm < 1e-6f) {
    tf2::Quaternion q;
    q.setRPY(0, 0, 0);
    return q;
  }
  axis /= axis_norm;
  float angle = std::acos(std::max(-1.0f, std::min(1.0f, z_axis.dot(normal))));
  tf2::Quaternion q;
  q.setRotation(tf2::Vector3(axis.x(), axis.y(), axis.z()), angle);
  return q;
}

float color_distance(const PointT & p, const Eigen::Vector3f & target)
{
  // RGB is stored in 0-255; normalize to 0-1 for distance computation
  Eigen::Vector3f c(p.r / 255.0f, p.g / 255.0f, p.b / 255.0f);
  return (c - target).norm();
}

bool is_finite(const PointT & p)
{
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

bool is_finite(const pcl::Normal & n)
{
  return std::isfinite(n.normal_x) && std::isfinite(n.normal_y) && std::isfinite(n.normal_z);
}
}  // namespace

class ColorFilteredGraspPclNode : public rclcpp::Node
{
public:
  ColorFilteredGraspPclNode()
  : Node("color_filtered_grasp_pcl"),
    exclusion_color_(0.6f, 0.4f, 0.2f)
  {
    declare_parameter<std::string>("input_topic", "/camera/depth/color/points");
    declare_parameter<double>("process_period", 1.5);
    declare_parameter<double>("color_filter_radius", 0.15);
    input_topic_ = get_parameter("input_topic").as_string();
    process_period_ = get_parameter("process_period").as_double();
    color_filter_radius_ = get_parameter("color_filter_radius").as_double();

    // Initialize TF2 for coordinate transformation
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    auto sensor_qos = rclcpp::QoS(rclcpp::SensorDataQoS());
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, sensor_qos,
      std::bind(&ColorFilteredGraspPclNode::pointcloud_callback, this, std::placeholders::_1));
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "color_filtered_grasp/target_pose", 10);
    edge_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "color_filtered_grasp/edges", 10);
    table_height_pub_ = create_publisher<std_msgs::msg::Float32>(
      "color_filtered_grasp/table_height", 10);
    transformed_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "color_filtered_grasp/transformed_cloud", 10);

    RCLCPP_INFO(get_logger(), "Listening to %s for point clouds.", input_topic_.c_str());
  }

private:
  std::string input_topic_;
  double process_period_{1.5};
  double color_filter_radius_{-1.0};  // disable color filtering by default for robustness
  double min_grasp_len_{0.02};  // meters
  double max_grasp_len_{0.10};  // meters
  // Workspace bounds (base_link frame after transformation).
  // NOTE: In Gazebo simulation, point cloud data appears to be in world-like coordinates
  // even after TF transformation. Robot base_link is at world Z ~1.015m.
  // These limits define the physical working area.
  // X: forward(+), Y: left(+)/right(-), Z: up(+)
  // Robot workspace for pick-and-place operations
  const float pass_x_min_{0.15f};    // Minimum forward distance from robot base
  const float pass_x_max_{0.45f};    // Maximum forward reach (table area)
  const float pass_y_min_{-0.25f};   // Right side limit
  const float pass_y_max_{0.25f};    // Left side limit
  // WORKAROUND: Z values after TF are world-like coordinates (table at Z~1.015)
  // Cube top is at world Z ~1.05m (table 1.015 + cube height 0.035)
  const float pass_z_min_{0.90f};    // Below table surface (world Z)
  const float pass_z_max_{1.20f};    // Maximum height above table (world Z)
  rclcpp::Time last_process_time_;
  Eigen::Vector3f exclusion_color_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr edge_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr table_height_pub_;
  // Table height tracking (in base_link frame)
  float last_table_height_base_ = std::numeric_limits<float>::quiet_NaN();
  bool has_table_height_base_ = false;

  // Plane coefficients in base_link frame for tilt correction
  // Table plane: Z = plane_a_base_*X + plane_b_base_*Y + plane_c_base_
  float plane_a_base_ = 0.0f;
  float plane_b_base_ = 0.0f;
  float plane_c_base_ = 0.0f;
  bool has_plane_base_ = false;

  // TF2 for coordinate transformation (camera frame → base_link)
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // Cached transform from camera to base_link
  Eigen::Affine3f camera_to_base_transform_;
  std::string resolved_camera_frame_;
  std::string resolved_base_frame_;
  bool transform_cached_{false};

  // Publisher for transformed point cloud (debugging)
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr transformed_cloud_pub_;

  // VoxelGrid downsampling
  float voxel_leaf_size_{0.02f};  // 2cm voxel size (reduced for visualization)

  // Build candidate frame names to compensate for namespace/prefix differences
  // NOTE: In Gazebo simulation, point cloud may be labeled as "camera_link" but
  // the actual coordinate data is in optical frame convention (Z forward).
  // We prioritize optical frames to get correct transformations.
  std::vector<std::string> build_candidate_frames(const std::string & raw_frame)
  {
    std::vector<std::string> frames;
    auto add_unique = [&](const std::string & f) {
      if (std::find(frames.begin(), frames.end(), f) == frames.end()) {
        frames.push_back(f);
      }
    };

    // Prioritize optical frames (point cloud data is typically in optical frame convention)
    add_unique("camera_depth_optical_frame");
    add_unique("camera_color_optical_frame");

    // Then try the raw frame and its optical variant
    if (raw_frame.find("_optical_frame") == std::string::npos) {
      add_unique(raw_frame + "_optical_frame");
    }
    add_unique(raw_frame);

    // camera_link as fallback
    add_unique("camera_link");

    // If frame has a prefix (e.g., "crane_x7/..."), reuse it for known names
    auto slash_pos = raw_frame.find('/');
    if (slash_pos != std::string::npos) {
      std::string prefix = raw_frame.substr(0, slash_pos);
      add_unique(prefix + "/camera_depth_optical_frame");
      add_unique(prefix + "/camera_color_optical_frame");
      add_unique(prefix + "/camera_link");
    }

    return frames;
  }

  // Resolve and cache the transform from camera frame to base_link
  // This is called once when the first point cloud is received
  // Returns true if transform was successfully resolved and cached
  bool resolve_and_cache_transform(const std::string & raw_camera_frame)
  {
    if (transform_cached_) {
      return true;  // Already cached
    }

    const std::vector<std::string> candidate_sources = build_candidate_frames(raw_camera_frame);
    const std::vector<std::string> candidate_targets = {
      "base_link",
      "crane_x7/base_link",
      "crane_x7/crane_x7_base_link"
    };

    for (const auto & src : candidate_sources) {
      for (const auto & tgt : candidate_targets) {
        try {
          geometry_msgs::msg::TransformStamped tf_stamped =
            tf_buffer_->lookupTransform(tgt, src, tf2::TimePointZero, tf2::durationFromSec(0.5));

          // Convert to Eigen transform
          // TF convention: p_target = R * p_source + t
          // Eigen Affine: translate() applies T after R (correct for TF)
          Eigen::Quaternionf q(
            static_cast<float>(tf_stamped.transform.rotation.w),
            static_cast<float>(tf_stamped.transform.rotation.x),
            static_cast<float>(tf_stamped.transform.rotation.y),
            static_cast<float>(tf_stamped.transform.rotation.z));
          Eigen::Vector3f t(
            static_cast<float>(tf_stamped.transform.translation.x),
            static_cast<float>(tf_stamped.transform.translation.y),
            static_cast<float>(tf_stamped.transform.translation.z));

          // Build affine transform: p_target = R * p_source + t
          // Set rotation and translation directly to avoid Eigen operator order confusion
          camera_to_base_transform_ = Eigen::Affine3f::Identity();
          camera_to_base_transform_.linear() = q.toRotationMatrix();  // Set rotation matrix
          camera_to_base_transform_.translation() = t;                // Set translation vector

          resolved_camera_frame_ = src;
          resolved_base_frame_ = tgt;
          transform_cached_ = true;

          RCLCPP_INFO(get_logger(),
            "Resolved TF: '%s' → '%s' (translation: [%.3f, %.3f, %.3f], rotation quat [w=%.3f, x=%.3f, y=%.3f, z=%.3f])",
            src.c_str(), tgt.c_str(), t.x(), t.y(), t.z(),
            q.w(), q.x(), q.y(), q.z());
          return true;

        } catch (const tf2::TransformException &) {
          continue;
        }
      }
    }

    RCLCPP_WARN(get_logger(),
      "Failed to resolve TF from camera to base_link (tried %zu sources × %zu targets)",
      candidate_sources.size(), candidate_targets.size());
    return false;
  }

  // Transform entire point cloud from camera frame to base_link frame
  // Returns nullptr if transformation fails
  pcl::PointCloud<PointT>::Ptr transform_cloud_to_base_link(
    const pcl::PointCloud<PointT>::Ptr & cloud_in,
    const std::string & raw_camera_frame)
  {
    if (!resolve_and_cache_transform(raw_camera_frame)) {
      return nullptr;
    }

    pcl::PointCloud<PointT>::Ptr cloud_out(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*cloud_in, *cloud_out, camera_to_base_transform_);

    // Update frame_id to base_link
    cloud_out->header.frame_id = resolved_base_frame_;

    return cloud_out;
  }

  // Transform pose from camera frame to base_link frame
  // Returns std::nullopt if transformation fails
  std::optional<geometry_msgs::msg::PoseStamped> transform_to_base_link(
    const geometry_msgs::msg::PoseStamped & pose_in_camera)
  {
    // Build candidate source frames from the original frame
    const std::vector<std::string> candidate_sources = build_candidate_frames(pose_in_camera.header.frame_id);

    // List of possible target frames (base_link variations)
    const std::vector<std::string> candidate_targets = {
      "base_link",
      "crane_x7/base_link",
      "crane_x7/crane_x7_base_link"
    };

    for (const auto & src : candidate_sources) {
      for (const auto & tgt : candidate_targets) {
        try {
          // Create a copy with the candidate source frame
          geometry_msgs::msg::PoseStamped pose_src = pose_in_camera;
          pose_src.header.frame_id = src;

          // Transform pose from camera frame to base_link frame
          geometry_msgs::msg::PoseStamped pose_in_base =
            tf_buffer_->transform(pose_src, tgt, tf2::durationFromSec(0.1));

          RCLCPP_INFO(get_logger(),
            "Transformed pose from '%s' to '%s': camera[%.3f, %.3f, %.3f] → base[%.3f, %.3f, %.3f]",
            src.c_str(), tgt.c_str(),
            pose_in_camera.pose.position.x, pose_in_camera.pose.position.y, pose_in_camera.pose.position.z,
            pose_in_base.pose.position.x, pose_in_base.pose.position.y, pose_in_base.pose.position.z);

          return pose_in_base;
        } catch (const tf2::TransformException &) {
          // Try next candidate
          continue;
        }
      }
    }

    RCLCPP_WARN(get_logger(),
      "Failed to transform pose from '%s' to base_link (tried %zu sources × %zu targets).",
      pose_in_camera.header.frame_id.c_str(), candidate_sources.size(), candidate_targets.size());
    return std::nullopt;
  }

  bool validate_grasp_height(const geometry_msgs::msg::PoseStamped & pose_base)
  {
    if (!has_table_height_base_) {
      return true;  // No table estimate yet; keep pose
    }
    const double min_above_table = -0.05;   // allow up to -5cm below plane
    const double max_above_table = 0.16;    // cube高さ+余裕を広げる
    double dz = pose_base.pose.position.z - static_cast<double>(last_table_height_base_);
    if (dz < min_above_table || dz > max_above_table) {
      RCLCPP_WARN(
        get_logger(),
        "Rejected grasp: height offset from table %.3f m (table=%.3f, grasp=%.3f, allowed %.3f..%.3f)",
        dz, last_table_height_base_, pose_base.pose.position.z,
        min_above_table, max_above_table);
      return false;
    }
    return true;
  }

  // Transform plane from camera frame to base_link and compute coefficients
  // The plane in base_link is: Z = a*X + b*Y + c
  // Returns true if successful
  bool compute_plane_coefficients_in_base_link(
    const pcl::ModelCoefficients & plane_cam,
    const Eigen::Vector3f & plane_centroid_cam,
    const std_msgs::msg::Header & header)
  {
    if (plane_cam.values.size() < 4) {
      return false;
    }

    // Get plane normal in camera frame
    Eigen::Vector3f normal_cam(plane_cam.values[0], plane_cam.values[1], plane_cam.values[2]);
    if (normal_cam.norm() < 1e-6f) {
      return false;
    }
    normal_cam.normalize();

    // Create 3 points on the plane in camera frame
    // Point 1: centroid
    // Point 2: centroid + tangent1 * 0.1m
    // Point 3: centroid + tangent2 * 0.1m
    Eigen::Vector3f tangent1, tangent2;
    if (std::abs(normal_cam.z()) > 0.9f) {
      tangent1 = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    } else {
      tangent1 = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    }
    tangent1 = (tangent1 - normal_cam * normal_cam.dot(tangent1)).normalized();
    tangent2 = normal_cam.cross(tangent1).normalized();

    std::array<Eigen::Vector3f, 3> pts_cam = {
      plane_centroid_cam,
      plane_centroid_cam + tangent1 * 0.1f,
      plane_centroid_cam + tangent2 * 0.1f
    };

    // Transform all 3 points to base_link
    std::array<Eigen::Vector3f, 3> pts_base;
    for (size_t i = 0; i < 3; ++i) {
      geometry_msgs::msg::PoseStamped pose_cam;
      pose_cam.header = header;
      pose_cam.pose.position.x = pts_cam[i].x();
      pose_cam.pose.position.y = pts_cam[i].y();
      pose_cam.pose.position.z = pts_cam[i].z();
      pose_cam.pose.orientation.w = 1.0;

      auto pose_base_opt = transform_to_base_link(pose_cam);
      if (!pose_base_opt.has_value()) {
        RCLCPP_WARN(get_logger(), "Failed to transform plane point %zu to base_link", i);
        return false;
      }
      pts_base[i] = Eigen::Vector3f(
        pose_base_opt->pose.position.x,
        pose_base_opt->pose.position.y,
        pose_base_opt->pose.position.z);
    }

    // Fit plane Z = aX + bY + c using least squares
    // [X1 Y1 1] [a]   [Z1]
    // [X2 Y2 1] [b] = [Z2]
    // [X3 Y3 1] [c]   [Z3]
    Eigen::Matrix3f A;
    Eigen::Vector3f b_vec;
    for (int i = 0; i < 3; ++i) {
      A(i, 0) = pts_base[i].x();
      A(i, 1) = pts_base[i].y();
      A(i, 2) = 1.0f;
      b_vec(i) = pts_base[i].z();
    }

    Eigen::Vector3f coeffs = A.colPivHouseholderQr().solve(b_vec);
    plane_a_base_ = coeffs(0);
    plane_b_base_ = coeffs(1);
    plane_c_base_ = coeffs(2);
    has_plane_base_ = true;

    RCLCPP_INFO(get_logger(),
      "Plane in base_link: Z = %.6f*X + %.6f*Y + %.6f (tilt dZ/dX=%.4f, dZ/dY=%.4f)",
      plane_a_base_, plane_b_base_, plane_c_base_,
      plane_a_base_, plane_b_base_);

    return true;
  }

  // Apply tilt correction to Z coordinate
  // Returns Z relative to table surface (table at Z=0)
  double apply_tilt_correction(double x, double y, double z) const
  {
    if (!has_plane_base_) {
      return z;  // No correction available
    }
    double table_z_at_xy = plane_a_base_ * x + plane_b_base_ * y + plane_c_base_;
    double z_corrected = z - table_z_at_xy;
    RCLCPP_INFO(get_logger(),
      "Tilt correction: Z=%.4f → Z_corrected=%.4f (table_z_at_xy=%.4f)",
      z, z_corrected, table_z_at_xy);
    return z_corrected;
  }

  // Broadcast detected object position as TF frame
  void broadcast_target_tf(const geometry_msgs::msg::PoseStamped & pose)
  {
    geometry_msgs::msg::TransformStamped t;
    t.header = pose.header;
    t.child_frame_id = "grasp_target";
    t.transform.translation.x = pose.pose.position.x;
    t.transform.translation.y = pose.pose.position.y;
    t.transform.translation.z = pose.pose.position.z;
    t.transform.rotation = pose.pose.orientation;
    tf_broadcaster_->sendTransform(t);
    RCLCPP_INFO(get_logger(), "Broadcast TF: %s → %s [%.3f, %.3f, %.3f]",
                pose.header.frame_id.c_str(), t.child_frame_id.c_str(),
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z);
  }

  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    auto now = this->get_clock()->now();
    if (last_process_time_.nanoseconds() != 0 &&
      (now - last_process_time_).seconds() < process_period_)
    {
      return;
    }
    last_process_time_ = now;

    const int gmm_components = 3;  // Number of GMM components for clustering

    // Convert ROS message to PCL point cloud
    pcl::PointCloud<PointT>::Ptr cloud_camera(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*msg, *cloud_camera);
    RCLCPP_INFO(get_logger(), "Received point cloud: %zu points in frame '%s'",
                cloud_camera->size(), msg->header.frame_id.c_str());

    // Debug: check point cloud range BEFORE transformation
    {
      size_t finite_before = 0;
      float bx_min = std::numeric_limits<float>::max(), bx_max = std::numeric_limits<float>::lowest();
      float by_min = std::numeric_limits<float>::max(), by_max = std::numeric_limits<float>::lowest();
      float bz_min = std::numeric_limits<float>::max(), bz_max = std::numeric_limits<float>::lowest();
      for (const auto & p : cloud_camera->points) {
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
          finite_before++;
          bx_min = std::min(bx_min, p.x); bx_max = std::max(bx_max, p.x);
          by_min = std::min(by_min, p.y); by_max = std::max(by_max, p.y);
          bz_min = std::min(bz_min, p.z); bz_max = std::max(bz_max, p.z);
        }
      }
      RCLCPP_INFO(get_logger(), "BEFORE transform: %zu finite points, range X[%.2f,%.2f] Y[%.2f,%.2f] Z[%.2f,%.2f]",
                  finite_before, bx_min, bx_max, by_min, by_max, bz_min, bz_max);
    }

    // === EARLY TRANSFORMATION: Convert point cloud from camera frame to base_link ===
    // This corrects for camera tilt and allows processing in robot-centric coordinates
    pcl::PointCloud<PointT>::Ptr cloud = transform_cloud_to_base_link(cloud_camera, msg->header.frame_id);
    if (!cloud) {
      RCLCPP_WARN(get_logger(), "Failed to transform point cloud to base_link. Waiting for TF...");
      return;
    }
    RCLCPP_INFO(get_logger(), "Transformed to base_link: %zu points", cloud->size());

    // Debug: count finite points after transformation
    size_t finite_after_transform = 0;
    float tx_min = std::numeric_limits<float>::max(), tx_max = std::numeric_limits<float>::lowest();
    float ty_min = std::numeric_limits<float>::max(), ty_max = std::numeric_limits<float>::lowest();
    float tz_min = std::numeric_limits<float>::max(), tz_max = std::numeric_limits<float>::lowest();
    for (const auto & p : cloud->points) {
      if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
        finite_after_transform++;
        tx_min = std::min(tx_min, p.x); tx_max = std::max(tx_max, p.x);
        ty_min = std::min(ty_min, p.y); ty_max = std::max(ty_max, p.y);
        tz_min = std::min(tz_min, p.z); tz_max = std::max(tz_max, p.z);
      }
    }
    RCLCPP_INFO(get_logger(), "After transform: %zu finite points, range X[%.2f,%.2f] Y[%.2f,%.2f] Z[%.2f,%.2f]",
                finite_after_transform, tx_min, tx_max, ty_min, ty_max, tz_min, tz_max);

    // Publish transformed cloud for debugging (in base_link frame)
    if (transformed_cloud_pub_->get_subscription_count() > 0) {
      sensor_msgs::msg::PointCloud2 transformed_msg;
      pcl::toROSMsg(*cloud, transformed_msg);
      transformed_msg.header.stamp = msg->header.stamp;
      transformed_msg.header.frame_id = resolved_base_frame_;
      transformed_cloud_pub_->publish(transformed_msg);
    }

    // Apply PassThrough filter on Z-axis (now in base_link frame: Z is height)
    pcl::PointCloud<PointT>::Ptr cloud_filtered_z(new pcl::PointCloud<PointT>());
    pcl::PassThrough<PointT> pass_z;
    pass_z.setInputCloud(cloud);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(pass_z_min_, pass_z_max_);
    pass_z.setKeepOrganized(true);  // Keep organized structure by setting invalid points to NaN
    pass_z.filter(*cloud_filtered_z);
    cloud = cloud_filtered_z;  // Replace with filtered cloud
    RCLCPP_INFO(get_logger(), "After Z filter (%.2f-%.2fm): %zu points (workspace: X[%.2f,%.2f] Y[%.2f,%.2f])",
                pass_z_min_, pass_z_max_, cloud->size(),
                pass_x_min_, pass_x_max_, pass_y_min_, pass_y_max_);

    if (!cloud->isOrganized() || cloud->height <= 1) {
      RCLCPP_WARN(
        get_logger(),
        "Input cloud is not organized (height=%u, width=%u). Please provide an organized PointCloud2 "
        "(e.g., raw depth image points) or structure the cloud before this node.",
        cloud->height, cloud->width);
      return;
    }

    const int width = static_cast<int>(cloud->width);
    const int height = static_cast<int>(cloud->height);

    // Keep organized structure: mask out points outside ROI or near exclusion color by writing NaNs.
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>(*cloud));
    filtered->is_dense = false;
    size_t finite_count = 0;
    size_t initial_finite = 0;
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();

    // Debug counters for filtering
    size_t roi_rejected_x = 0, roi_rejected_y = 0, roi_rejected_z = 0;
    size_t color_rejected = 0;

    for (auto & p : filtered->points) {
      if (!is_finite(p)) {
        // Keep NaN to mark invalid for organized processing.
        continue;
      }
      initial_finite++;
      min_x = std::min(min_x, p.x);
      max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y);
      max_y = std::max(max_y, p.y);
      min_z = std::min(min_z, p.z);
      max_z = std::max(max_z, p.z);

      // Apply workspace bounds filter with detailed rejection tracking
      bool x_ok = (p.x >= pass_x_min_ && p.x <= pass_x_max_);
      bool y_ok = (p.y >= pass_y_min_ && p.y <= pass_y_max_);
      bool z_ok = (p.z >= pass_z_min_ && p.z <= pass_z_max_);
      bool in_workspace = x_ok && y_ok && z_ok;

      // Track rejection reasons
      if (!x_ok) roi_rejected_x++;
      if (!y_ok) roi_rejected_y++;
      if (!z_ok) roi_rejected_z++;

      // Apply color exclusion filter
      bool far_from_exclusion = color_distance(p, exclusion_color_) > static_cast<float>(color_filter_radius_);
      if (in_workspace && !far_from_exclusion) color_rejected++;

      if (!(in_workspace && far_from_exclusion)) {
        p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
      } else {
        finite_count++;
      }
    }

    // Debug: Show rejection breakdown
    RCLCPP_INFO(get_logger(), "=== ROI FILTER DEBUG ===");
    RCLCPP_INFO(get_logger(), "Workspace bounds: X[%.2f,%.2f] Y[%.2f,%.2f] Z[%.2f,%.2f]",
                pass_x_min_, pass_x_max_, pass_y_min_, pass_y_max_, pass_z_min_, pass_z_max_);
    RCLCPP_INFO(get_logger(), "Rejection breakdown: X_out=%zu, Y_out=%zu, Z_out=%zu, color_excl=%zu",
                roi_rejected_x, roi_rejected_y, roi_rejected_z, color_rejected);

    if (initial_finite > 0) {
      RCLCPP_INFO(
        get_logger(),
        "Point cloud range: X[%.3f, %.3f] Y[%.3f, %.3f] Z[%.3f, %.3f] (initial_finite=%zu, after_filter=%zu)",
        min_x, max_x, min_y, max_y, min_z, max_z, initial_finite, finite_count);
    }
    if (finite_count == 0) {
      RCLCPP_WARN(
        get_logger(),
        "No finite points after ROI/color mask. Skipping frame.");
      return;
    }
    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    const double valid_ratio = static_cast<double>(finite_count) / static_cast<double>(total);
    // Require at least 50% valid points AND minimum 50000 points for organized processing
    // This prevents IntegralImageNormalEstimation from encountering sparse clouds that cause Eigen errors
    // Lower density clouds will use the unorganized fallback which is more robust
    // NOTE: Organized path disabled due to IntegralImageNormalEstimation Eigen assertion error
    // Setting impossibly high threshold to always use unorganized fallback
    const size_t min_organized_points = 999999999;  // Always use unorganized path
    const double min_valid_ratio = 1.0;  // 100% - impossible to meet

    if (finite_count < min_organized_points || valid_ratio < min_valid_ratio) {
      RCLCPP_WARN(
        get_logger(),
        "Too few finite points for organized processing (%zu/%zu = %.1f%%, need >%zu and >%.0f%%). Falling back.",
        finite_count, total, valid_ratio * 100.0, min_organized_points, min_valid_ratio * 100.0);

      // Fallback: remove NaNs and run a simple curvature+GMM pipeline (unorganized).
      pcl::PointCloud<PointT>::Ptr unorganized(new pcl::PointCloud<PointT>());
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*filtered, *unorganized, indices);
      if (unorganized->empty()) {
        RCLCPP_WARN(get_logger(), "Fallback: empty cloud after NaN removal.");
        return;
      }

      // VoxelGrid downsampling for faster processing
      pcl::PointCloud<PointT>::Ptr downsampled(new pcl::PointCloud<PointT>());
      pcl::VoxelGrid<PointT> voxel;
      voxel.setInputCloud(unorganized);
      voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
      voxel.filter(*downsampled);
      RCLCPP_INFO(get_logger(), "VoxelGrid downsampling: %zu → %zu points",
                  unorganized->size(), downsampled->size());
      unorganized = downsampled;

      if (unorganized->empty()) {
        RCLCPP_WARN(get_logger(), "Fallback: empty cloud after VoxelGrid.");
        return;
      }

      // RANSAC plane segmentation to remove dominant table surface
      // Point cloud is now in base_link frame, so Z is vertical (up)
      // Table surface should be approximately horizontal (perpendicular to Z axis)
      pcl::SACSegmentation<PointT> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setDistanceThreshold(0.015f);  // 1.5cm tolerance for table surface
      seg.setMaxIterations(1000);
      seg.setProbability(0.99);
      seg.setAxis(Eigen::Vector3f::UnitZ());  // base_link Z is vertical
      seg.setEpsAngle(10.0f * kRadPerDeg);    // Tighter tolerance: table should be near-horizontal
      seg.setInputCloud(unorganized);

      pcl::ModelCoefficients::Ptr plane_coeff(new pcl::ModelCoefficients());
      pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices());
      seg.segment(*plane_inliers, *plane_coeff);

      if (!plane_inliers->indices.empty()) {
        // Calculate plane centroid for height reference (already in base_link frame)
        Eigen::Vector3f plane_sum(0.0f, 0.0f, 0.0f);
        for (int idx : plane_inliers->indices) {
          const auto & p = unorganized->points[static_cast<size_t>(idx)];
          plane_sum += Eigen::Vector3f(p.x, p.y, p.z);
        }
        Eigen::Vector3f plane_centroid = plane_sum / static_cast<float>(plane_inliers->indices.size());

        float plane_ratio = static_cast<float>(plane_inliers->indices.size()) /
                           static_cast<float>(unorganized->size());
        RCLCPP_INFO(get_logger(),
          "RANSAC plane (base_link): %zu inliers (%.1f%%), centroid z=%.3f",
          plane_inliers->indices.size(), plane_ratio * 100.0f, plane_centroid.z());

        // Plane coefficients are already in base_link frame
        // Plane equation: ax + by + cz + d = 0
        // For horizontal table: a≈0, b≈0, c≈1, d=-table_height
        if (plane_coeff->values.size() >= 4) {
          float a = plane_coeff->values[0];
          float b = plane_coeff->values[1];
          float c = plane_coeff->values[2];
          float d = plane_coeff->values[3];

          RCLCPP_INFO(get_logger(),
            "Plane coefficients (base_link): a=%.6f, b=%.6f, c=%.6f, d=%.6f",
            a, b, c, d);

          // Store plane coefficients for tilt correction (Z = -(ax + by + d) / c)
          // Rewrite as: Z = plane_a_base_*X + plane_b_base_*Y + plane_c_base_
          if (std::abs(c) > 0.1f) {
            plane_a_base_ = -a / c;
            plane_b_base_ = -b / c;
            plane_c_base_ = -d / c;
            has_plane_base_ = true;
            RCLCPP_INFO(get_logger(),
              "Table plane in base_link: Z = %.4f*X + %.4f*Y + %.4f (tilt dZ/dX=%.4f, dZ/dY=%.4f)",
              plane_a_base_, plane_b_base_, plane_c_base_, plane_a_base_, plane_b_base_);
          }
        }

        // Store table height directly (already in base_link Z)
        last_table_height_base_ = plane_centroid.z();
        has_table_height_base_ = true;

        // Publish table height
        std_msgs::msg::Float32 height_msg;
        height_msg.data = plane_centroid.z();
        table_height_pub_->publish(height_msg);
        RCLCPP_INFO(get_logger(), "Table height (base_link): %.3f", plane_centroid.z());

        // Extract points NOT on the table (objects above table)
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(unorganized);
        extract.setIndices(plane_inliers);
        extract.setNegative(true);  // Get points NOT in plane
        pcl::PointCloud<PointT>::Ptr objects_cloud(new pcl::PointCloud<PointT>());
        extract.filter(*objects_cloud);

        RCLCPP_INFO(get_logger(),
          "After plane removal: %zu points (removed %zu table points)",
          objects_cloud->size(), plane_inliers->indices.size());

        if (objects_cloud->size() >= 10) {
          unorganized = objects_cloud;
        } else {
          RCLCPP_WARN(get_logger(),
            "Too few points after plane removal (%zu). Proceeding with all points.",
            objects_cloud->size());
        }
      } else {
        RCLCPP_INFO(get_logger(), "No dominant plane found, proceeding with all points.");
      }

      RCLCPP_INFO(get_logger(), "Computing normals for %zu points...", unorganized->size());
      pcl::NormalEstimationOMP<PointT, pcl::Normal> ne_omp;
      auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
      ne_omp.setInputCloud(unorganized);
      ne_omp.setSearchMethod(tree);
      // Use K-search only (radius and K cannot be set simultaneously)
      // Smaller K = more sensitive to local geometry changes (better edge detection)
      ne_omp.setKSearch(15);
      pcl::PointCloud<pcl::Normal>::Ptr normals_omp(new pcl::PointCloud<pcl::Normal>());
      ne_omp.compute(*normals_omp);
      RCLCPP_INFO(get_logger(), "Normal estimation complete: %zu normals computed", normals_omp->size());

      std::vector<float> curvatures;
      curvatures.reserve(normals_omp->size());
      for (const auto & n : normals_omp->points) {
        if (std::isfinite(n.curvature)) {
          curvatures.push_back(n.curvature);
        }
      }
      RCLCPP_INFO(get_logger(), "Valid curvatures: %zu", curvatures.size());
      if (curvatures.size() < static_cast<size_t>(gmm_components)) {
        RCLCPP_WARN(get_logger(), "Fallback: not enough points for curvature (%zu).", curvatures.size());
        return;
      }
      // Use 85th percentile for more inclusive edge detection (was 92nd)
      // Lower percentile = more edges detected = better for small objects
      const float curvature_percentile = 0.85f;
      RCLCPP_INFO(get_logger(), "Extracting edges from curvatures (threshold will be %.0f%% percentile)...",
                  curvature_percentile * 100.0f);
      std::vector<float> sorted = curvatures;
      size_t idx = static_cast<size_t>(curvature_percentile * (sorted.size() - 1));
      std::nth_element(sorted.begin(), sorted.begin() + idx, sorted.end());
      float threshold = sorted[idx];
      RCLCPP_INFO(get_logger(), "Curvature threshold: %.6f", threshold);

      std::vector<size_t> edge_ids;
      edge_ids.reserve(curvatures.size());
      for (size_t i = 0; i < curvatures.size(); ++i) {
        if (curvatures[i] >= threshold) {
          edge_ids.push_back(i);
        }
      }
      RCLCPP_INFO(get_logger(), "Edge points extracted: %zu", edge_ids.size());
      if (edge_ids.size() < static_cast<size_t>(gmm_components)) {
        RCLCPP_WARN(get_logger(), "Fallback: not enough edge points (%zu).", edge_ids.size());
        return;
      }

      std::vector<Eigen::Vector3f> edge_points;
      std::vector<Eigen::Vector3f> edge_normals;
      edge_points.reserve(edge_ids.size());
      edge_normals.reserve(edge_ids.size());

      // Debug: collect Z values for analysis
      std::vector<float> edge_z_values;
      edge_z_values.reserve(edge_ids.size());
      size_t nan_count = 0;
      size_t z_filtered_count = 0;

      for (size_t i : edge_ids) {
        const auto & p = unorganized->points[i];
        const auto & n = normals_omp->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
          !std::isfinite(n.normal_x) || !std::isfinite(n.normal_y) || !std::isfinite(n.normal_z))
        {
          nan_count++;
          continue;
        }
        edge_z_values.push_back(p.z);

        // Z-filtering based on table height (now in base_link frame)
        // In base_link, Z is height (up is positive)
        // Objects ON TOP of the table have larger Z than the table surface
        if (has_table_height_base_) {
          // Keep points that are above the table surface (with some margin below for tolerance)
          const float min_z = last_table_height_base_ - 0.02f;  // 2cm below table (tolerance)
          const float max_z = last_table_height_base_ + 0.20f;  // up to 20cm above table
          if (p.z < min_z || p.z > max_z) {
            z_filtered_count++;
            continue;
          }
        }
        edge_points.emplace_back(p.x, p.y, p.z);
        Eigen::Vector3f nv(n.normal_x, n.normal_y, n.normal_z);
        if (nv.norm() < 1e-6f) {
          continue;
        }
        nv.normalize();
        edge_normals.push_back(nv);
      }

      // Debug logging for Z filter analysis (base_link frame)
      if (!edge_z_values.empty()) {
        float min_edge_z = *std::min_element(edge_z_values.begin(), edge_z_values.end());
        float max_edge_z = *std::max_element(edge_z_values.begin(), edge_z_values.end());
        RCLCPP_INFO(get_logger(),
          "Edge Z range (base_link): [%.3f, %.3f], table_z=%.3f, NaN=%zu, Z-filtered=%zu",
          min_edge_z, max_edge_z, has_table_height_base_ ? last_table_height_base_ : -1.0f,
          nan_count, z_filtered_count);
        if (has_table_height_base_) {
          float filter_min = last_table_height_base_ - 0.02f;
          float filter_max = last_table_height_base_ + 0.20f;
          RCLCPP_INFO(get_logger(),
            "Z filter bounds (base_link): [%.3f, %.3f] (table - 0.02, table + 0.20)",
            filter_min, filter_max);
        }
      }
      RCLCPP_INFO(get_logger(), "Valid edge points after filtering: %zu", edge_points.size());
      if (edge_points.size() < static_cast<size_t>(gmm_components)) {
        RCLCPP_WARN(get_logger(), "Fallback: insufficient valid edge points (%zu).", edge_points.size());
        return;
      }

      RCLCPP_INFO(get_logger(), "Preparing GMM samples (%zu points × 6 features)...", edge_points.size());
      cv::Mat samples(static_cast<int>(edge_points.size()), 6, CV_32F);
      for (size_t i = 0; i < edge_points.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
          samples.at<float>(static_cast<int>(i), k) = edge_points[i][k];
          samples.at<float>(static_cast<int>(i), k + 3) = edge_normals[i][k];
        }
      }
      cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
      int clusters = std::max(1, std::min<int>(gmm_components, static_cast<int>(edge_points.size())));
      RCLCPP_INFO(get_logger(), "Training GMM with %d clusters...", clusters);
      em->setClustersNumber(clusters);
      em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_GENERIC);
      em->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 50, 1e-3));
      if (!em->trainEM(samples, cv::noArray(), cv::noArray(), cv::noArray())) {
        RCLCPP_WARN(get_logger(), "Fallback: GMM training failed.");
        return;
      }
      RCLCPP_INFO(get_logger(), "GMM training complete. Clustering edge points...");
      std::vector<int> labels(samples.rows);
      std::vector<int> counts(clusters, 0);
      cv::Mat probs;
      for (int i = 0; i < samples.rows; ++i) {
        cv::Vec2d res = em->predict2(samples.row(i), probs);
        int lbl = std::clamp(static_cast<int>(res[1]), 0, clusters - 1);
        labels[static_cast<size_t>(i)] = lbl;
        counts[static_cast<size_t>(lbl)]++;
      }
      int best_cluster = static_cast<int>(std::distance(
        counts.begin(), std::max_element(counts.begin(), counts.end())));
      RCLCPP_INFO(get_logger(), "Best cluster: %d with %d points", best_cluster, counts[best_cluster]);
      std::vector<size_t> cluster_indices;
      for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == best_cluster) {
          cluster_indices.push_back(i);
        }
      }
      if (cluster_indices.empty()) {
        RCLCPP_WARN(get_logger(), "Fallback: no points in best cluster.");
        return;
      }

      // Sample cluster points to avoid O(n²) performance issues
      const size_t max_sample_size = 500;
      std::vector<size_t> sampled_indices;
      if (cluster_indices.size() > max_sample_size) {
        // Uniform sampling: take every Nth point
        size_t step = cluster_indices.size() / max_sample_size;
        for (size_t i = 0; i < cluster_indices.size(); i += step) {
          sampled_indices.push_back(cluster_indices[i]);
          if (sampled_indices.size() >= max_sample_size) break;
        }
        RCLCPP_INFO(get_logger(), "Sampled %zu/%zu cluster points for grasp computation",
                    sampled_indices.size(), cluster_indices.size());
      } else {
        sampled_indices = cluster_indices;
        RCLCPP_INFO(get_logger(), "Computing grasp score for %zu cluster points...",
                    sampled_indices.size());
      }

      const float sigma = 0.1f;
      const float inv_sigma_sq = 1.0f / (sigma * sigma);
      float best_score = -std::numeric_limits<float>::infinity();
      size_t best_idx = sampled_indices.front();
      for (size_t idx_i : sampled_indices) {
        const auto & pi = edge_points[idx_i];
        const auto & ni = edge_normals[idx_i];
        float score = 0.0f;
        for (size_t idx_j : sampled_indices) {
          if (idx_i == idx_j) {
            continue;
          }
          const auto & pj = edge_points[idx_j];
          const auto & nj = edge_normals[idx_j];
          float dot = ni.dot(nj);
          float dist = (pi - pj).norm();
          score += dot * std::exp(-0.5f * dist * dist * inv_sigma_sq);
        }
        if (score > best_score) {
          best_score = score;
          best_idx = idx_i;
        }
      }
      Eigen::Vector3f centroid = edge_points[best_idx];
      Eigen::Vector3f normal_sum = edge_normals[best_idx];
      RCLCPP_INFO(get_logger(), "Grasp centroid computed: [%.3f, %.3f, %.3f] with score %.3f",
                  centroid.x(), centroid.y(), centroid.z(), best_score);
      tf2::Quaternion q = normal_to_quaternion(normal_sum);

      // Point cloud is already in base_link frame, so pose is directly in base_link
      std_msgs::msg::Header header;
      header.stamp = msg->header.stamp;
      header.frame_id = resolved_base_frame_;

      // Create pose directly in base_link frame (no transformation needed)
      geometry_msgs::msg::PoseStamped pose_base;
      pose_base.header = header;
      pose_base.pose.position.x = centroid.x();
      pose_base.pose.position.y = centroid.y();
      pose_base.pose.position.z = centroid.z();
      pose_base.pose.orientation.x = q.x();
      pose_base.pose.orientation.y = q.y();
      pose_base.pose.orientation.z = q.z();
      pose_base.pose.orientation.w = q.w();

      if (!validate_grasp_height(pose_base)) {
        return;
      }

      // Tilt correction disabled - output world Z directly for Gazebo simulation
      // geometry_msgs::msg::PoseStamped pose_corrected = pose_base;
      // pose_corrected.pose.position.z = apply_tilt_correction(
      //   pose_base.pose.position.x,
      //   pose_base.pose.position.y,
      //   pose_base.pose.position.z);

      pose_pub_->publish(pose_base);
      broadcast_target_tf(pose_base);

      // Publish edge points for visualization (in base_link frame)
      if (edge_pub_->get_subscription_count() > 0) {
        RCLCPP_INFO(get_logger(), "[UNORGANIZED PATH] Publishing %zu edge points in frame '%s'",
          edge_points.size(), resolved_base_frame_.c_str());

        pcl::PointCloud<PointT>::Ptr edges_base(new pcl::PointCloud<PointT>());
        edges_base->header.frame_id = resolved_base_frame_;
        edges_base->header.stamp = pcl_conversions::toPCL(header.stamp);
        edges_base->points.reserve(edge_points.size());
        for (const auto& pt : edge_points) {
          PointT p;
          p.x = pt.x();
          p.y = pt.y();
          p.z = pt.z();
          p.r = 255;  // Red color for edges
          p.g = 0;
          p.b = 0;
          edges_base->points.push_back(p);
        }
        edges_base->width = edges_base->points.size();
        edges_base->height = 1;
        edges_base->is_dense = true;

        sensor_msgs::msg::PointCloud2 edges_msg;
        pcl::toROSMsg(*edges_base, edges_msg);
        edges_msg.header.stamp = msg->header.stamp;
        edges_msg.header.frame_id = resolved_base_frame_;
        edge_pub_->publish(edges_msg);
      }

      return;
    }

    // Remove dominant horizontal plane (table) with RANSAC before edge detection.
    // Flag to track if RANSAC was applied (requires unorganized fallback)
    bool ransac_applied = false;
    pcl::PointCloud<PointT>::Ptr dense_cloud(new pcl::PointCloud<PointT>());
    std::vector<int> dense_indices;
    pcl::removeNaNFromPointCloud(*filtered, *dense_cloud, dense_indices);
    if (!dense_cloud->empty()) {
      auto run_segmentation = [&](pcl::PointCloud<PointT>::Ptr input_cloud,
                                  const std::string &label,
                                  pcl::SacModel model, float eps_deg,
                                  float distance, int iters,
                                  std::vector<int> &input_indices)
        -> std::optional<pcl::PointIndices::Ptr>
        {
          pcl::SACSegmentation<PointT> seg;
          seg.setOptimizeCoefficients(true);
          seg.setModelType(model);
          seg.setMethodType(pcl::SAC_RANSAC);
          seg.setDistanceThreshold(distance);
          seg.setMaxIterations(iters);
          seg.setProbability(0.99);
          if (model == pcl::SACMODEL_PERPENDICULAR_PLANE) {
            seg.setAxis(Eigen::Vector3f::UnitZ());
            seg.setEpsAngle(eps_deg * kRadPerDeg);
          }
          seg.setInputCloud(input_cloud);
          pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients());
          pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
          seg.segment(*inliers, *coeff);
          if (inliers->indices.empty()) {
            RCLCPP_INFO(get_logger(), "RANSAC %s: no inliers", label.c_str());
            return std::nullopt;
          }

          Eigen::Vector3f plane_sum(0.0f, 0.0f, 0.0f);
          for (int idx : inliers->indices) {
            const auto & p = input_cloud->points[static_cast<size_t>(idx)];
            plane_sum.x() += p.x;
            plane_sum.y() += p.y;
            plane_sum.z() += p.z;
          }
          Eigen::Vector3f plane_centroid = plane_sum / static_cast<float>(inliers->indices.size());

          Eigen::Vector3f normal_cam(0.0f, 0.0f, 1.0f);
          if (coeff->values.size() >= 4) {
            normal_cam = Eigen::Vector3f(coeff->values[0], coeff->values[1], coeff->values[2]);
            if (normal_cam.norm() > 1e-6f) {
              normal_cam.normalize();
            } else {
              normal_cam = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
            }
          }
          const float normal_dot = std::abs(normal_cam.dot(Eigen::Vector3f::UnitZ()));
          const bool normal_ok = (model == pcl::SACMODEL_PERPENDICULAR_PLANE) ?
            (normal_dot > std::cos(20.0f * kRadPerDeg)) : true;
          const bool z_ok = plane_centroid.z() > 0.3f && plane_centroid.z() < 0.8f;

          RCLCPP_INFO(
            get_logger(),
            "RANSAC %s: inliers=%zu (%.1f%%), centroid z=%.3f, normal dot=%.3f, normal_ok=%d, z_ok=%d",
            label.c_str(), inliers->indices.size(),
            100.0f * static_cast<float>(inliers->indices.size()) /
              static_cast<float>(input_cloud->size()),
            plane_centroid.z(), normal_dot, normal_ok, z_ok);

          if (!normal_ok || !z_ok) {
            return std::nullopt;
          }

          // Apply removal to organized cloud via indices map
          for (int idx : inliers->indices) {
            const size_t orig_idx = static_cast<size_t>(input_indices[static_cast<size_t>(idx)]);
            auto & p = filtered->points[orig_idx];
            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
          }
          return inliers;
        };

      auto dense_after_first = dense_cloud;
      auto dense_indices_after_first = dense_indices;

      auto first_inliers = run_segmentation(
        dense_cloud, "pass1 (table)", pcl::SACMODEL_PERPENDICULAR_PLANE,
        25.0f, 0.012f, 1500, dense_indices);

      bool first_applied = false;
      if (first_inliers) {
        size_t finite_after_plane = 0;
        for (const auto & p : filtered->points) {
          if (is_finite(p)) {
            finite_after_plane++;
          }
        }
        float plane_ratio = static_cast<float>(first_inliers.value()->indices.size()) /
          static_cast<float>(dense_cloud->size());
        size_t candidate_remaining =
          finite_count > first_inliers.value()->indices.size() ?
          finite_count - first_inliers.value()->indices.size() : 0;
        RCLCPP_INFO(
          get_logger(),
          "Pass1 removed %zu pts (%.1f%%). Finite after plane: %zu (from %zu).",
          first_inliers.value()->indices.size(), plane_ratio * 100.0f,
          finite_after_plane, finite_count);
        if (first_inliers.value()->indices.size() > 200 && plane_ratio > 0.10f &&
          candidate_remaining >= 5000)
        {
          finite_count = finite_after_plane;
          ransac_applied = true;
          first_applied = true;

          // Recompute dense cloud after removal
          dense_after_first.reset(new pcl::PointCloud<PointT>());
          dense_indices_after_first.clear();
          pcl::removeNaNFromPointCloud(*filtered, *dense_after_first, dense_indices_after_first);
        }
      }

      // 2ndパスは過剰除去となるため停止（2025-11-29時点）

      // Publish detected table height only from first valid pass (table plane)
      if (first_applied) {
        // 再生成して重心と法線を求め直す
        pcl::SACSegmentation<PointT> seg_for_coeff;
        seg_for_coeff.setOptimizeCoefficients(true);
        seg_for_coeff.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg_for_coeff.setAxis(Eigen::Vector3f::UnitZ());
        seg_for_coeff.setEpsAngle(25.0f * kRadPerDeg);
        seg_for_coeff.setMethodType(pcl::SAC_RANSAC);
        seg_for_coeff.setDistanceThreshold(0.012f);
        seg_for_coeff.setMaxIterations(1500);
        seg_for_coeff.setInputCloud(dense_cloud);
        pcl::ModelCoefficients coeff;
        pcl::PointIndices inliers_tmp;
        seg_for_coeff.segment(inliers_tmp, coeff);

        // Point cloud is already in base_link, so plane centroid is directly in base_link
        Eigen::Vector3f plane_sum(0.0f, 0.0f, 0.0f);
        for (int idx : inliers_tmp.indices) {
          const auto & p = dense_cloud->points[static_cast<size_t>(idx)];
          plane_sum.x() += p.x;
          plane_sum.y() += p.y;
          plane_sum.z() += p.z;
        }
        Eigen::Vector3f plane_centroid = plane_sum / static_cast<float>(inliers_tmp.indices.size());

        // Store table height directly (already in base_link Z)
        last_table_height_base_ = plane_centroid.z();
        has_table_height_base_ = true;

        // Publish table height
        std_msgs::msg::Float32 height_msg;
        height_msg.data = plane_centroid.z();
        table_height_pub_->publish(height_msg);
        RCLCPP_INFO(get_logger(), "Organized path: Table height (base_link) = %.3f", plane_centroid.z());
      }
    }

    // Integral image normal計算はNaNが多いと不安定になるため、点の割合もチェック
    const float finite_ratio =
      static_cast<float>(finite_count) /
      static_cast<float>(static_cast<size_t>(width) * static_cast<size_t>(height));
    // RANSAC平面除去後でも十分な点が残るように閾値を緩和 (0.5 -> 0.25)
    if (finite_count < 5000 || finite_ratio < 0.25f) {
      RCLCPP_WARN(
        get_logger(),
        "Too few finite points after ROI/color mask (%zu, %.1f%%). Skipping frame.",
        finite_count, finite_ratio * 100.0f);
      return;
    }
    RCLCPP_INFO(get_logger(), "DEBUG: Finite points after mask: %zu/%zu",
                finite_count, static_cast<size_t>(width) * static_cast<size_t>(height));

    // RANSAC適用後はorganized構造が壊れているため、unorganized fallbackを使用
    if (ransac_applied) {
      RCLCPP_INFO(
        get_logger(),
        "RANSAC plane removal applied, using unorganized fallback (%zu points, %.1f%%).",
        finite_count, finite_ratio * 100.0f);

      // Remove NaNs and process as unorganized cloud
      pcl::PointCloud<PointT>::Ptr unorganized(new pcl::PointCloud<PointT>());
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*filtered, *unorganized, indices);
      if (unorganized->empty()) {
        RCLCPP_WARN(get_logger(), "RANSAC fallback: empty cloud after NaN removal.");
        return;
      }

      // VoxelGrid downsampling for faster processing
      pcl::PointCloud<PointT>::Ptr downsampled(new pcl::PointCloud<PointT>());
      pcl::VoxelGrid<PointT> voxel;
      voxel.setInputCloud(unorganized);
      voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
      voxel.filter(*downsampled);
      RCLCPP_INFO(get_logger(), "VoxelGrid downsampling: %zu → %zu points",
                  unorganized->size(), downsampled->size());
      unorganized = downsampled;

      if (unorganized->empty()) {
        RCLCPP_WARN(get_logger(), "RANSAC fallback: empty cloud after VoxelGrid.");
        return;
      }
      RCLCPP_INFO(get_logger(), "Computing normals for %zu points...", unorganized->size());
      pcl::NormalEstimationOMP<PointT, pcl::Normal> ne_omp;
      auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
      ne_omp.setInputCloud(unorganized);
      ne_omp.setSearchMethod(tree);
      ne_omp.setKSearch(15);
      pcl::PointCloud<pcl::Normal>::Ptr normals_omp(new pcl::PointCloud<pcl::Normal>());
      ne_omp.compute(*normals_omp);
      RCLCPP_INFO(get_logger(), "Normal estimation complete: %zu normals computed", normals_omp->size());

      std::vector<float> curvatures;
      curvatures.reserve(normals_omp->size());
      for (const auto & n : normals_omp->points) {
        if (std::isfinite(n.curvature)) {
          curvatures.push_back(n.curvature);
        }
      }
      RCLCPP_INFO(get_logger(), "Valid curvatures: %zu", curvatures.size());
      if (curvatures.size() < 3) {
        RCLCPP_WARN(get_logger(), "RANSAC fallback: not enough points for curvature (%zu).", curvatures.size());
        return;
      }
      const float curvature_percentile = 0.85f;
      RCLCPP_INFO(get_logger(), "Extracting edges from curvatures (threshold will be %.0f%% percentile)...",
                  curvature_percentile * 100.0f);
      std::vector<float> sorted = curvatures;
      size_t idx = static_cast<size_t>(curvature_percentile * (sorted.size() - 1));
      std::nth_element(sorted.begin(), sorted.begin() + idx, sorted.end());
      float threshold = sorted[idx];
      RCLCPP_INFO(get_logger(), "Curvature threshold: %.6f", threshold);

      std::vector<size_t> edge_ids;
      edge_ids.reserve(curvatures.size());
      for (size_t i = 0; i < curvatures.size(); ++i) {
        if (curvatures[i] >= threshold) {
          edge_ids.push_back(i);
        }
      }
      RCLCPP_INFO(get_logger(), "Edge points extracted: %zu", edge_ids.size());
      if (edge_ids.size() < 3) {
        RCLCPP_WARN(get_logger(), "RANSAC fallback: not enough edge points (%zu).", edge_ids.size());
        return;
      }

      std::vector<Eigen::Vector3f> edge_points;
      std::vector<Eigen::Vector3f> edge_normals;
      edge_points.reserve(edge_ids.size());
      edge_normals.reserve(edge_ids.size());
      for (size_t i : edge_ids) {
        const auto & p = unorganized->points[i];
        const auto & n = normals_omp->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
          !std::isfinite(n.normal_x) || !std::isfinite(n.normal_y) || !std::isfinite(n.normal_z))
        {
          continue;
        }
        if (has_table_height_base_) {
          // In base_link, Z is height (up is positive)
          // Keep points above table surface
          const float min_z = last_table_height_base_ - 0.02f;  // 2cm below table (tolerance)
          const float max_z = last_table_height_base_ + 0.20f;  // up to 20cm above table
          if (p.z < min_z || p.z > max_z) {
            continue;
          }
        }
        edge_points.emplace_back(p.x, p.y, p.z);
        Eigen::Vector3f nv(n.normal_x, n.normal_y, n.normal_z);
        if (nv.norm() < 1e-6f) {
          continue;
        }
        nv.normalize();
        edge_normals.push_back(nv);
      }
      RCLCPP_INFO(get_logger(), "Valid edge points after filtering (base_link): %zu", edge_points.size());
      if (edge_points.size() < 3) {
        RCLCPP_WARN(get_logger(), "RANSAC fallback: insufficient valid edge points (%zu).", edge_points.size());
        return;
      }

      const int gmm_components = 3;
      RCLCPP_INFO(get_logger(), "Preparing GMM samples (%zu points × 6 features)...", edge_points.size());
      cv::Mat samples(static_cast<int>(edge_points.size()), 6, CV_32F);
      for (size_t i = 0; i < edge_points.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
          samples.at<float>(static_cast<int>(i), k) = edge_points[i][k];
          samples.at<float>(static_cast<int>(i), k + 3) = edge_normals[i][k];
        }
      }
      cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
      int clusters = std::max(1, std::min<int>(gmm_components, static_cast<int>(edge_points.size())));
      RCLCPP_INFO(get_logger(), "Training GMM with %d clusters...", clusters);
      em->setClustersNumber(clusters);
      em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_GENERIC);
      em->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 50, 1e-3));
      if (!em->trainEM(samples, cv::noArray(), cv::noArray(), cv::noArray())) {
        RCLCPP_WARN(get_logger(), "RANSAC fallback: GMM training failed.");
        return;
      }
      RCLCPP_INFO(get_logger(), "GMM training complete. Clustering edge points...");
      std::vector<int> labels(samples.rows);
      std::vector<int> counts(clusters, 0);
      cv::Mat probs;
      for (int i = 0; i < samples.rows; ++i) {
        cv::Vec2d res = em->predict2(samples.row(i), probs);
        int lbl = std::clamp(static_cast<int>(res[1]), 0, clusters - 1);
        labels[static_cast<size_t>(i)] = lbl;
        counts[static_cast<size_t>(lbl)]++;
      }
      int best_cluster = static_cast<int>(std::distance(
        counts.begin(), std::max_element(counts.begin(), counts.end())));
      RCLCPP_INFO(get_logger(), "Best cluster: %d with %d points", best_cluster, counts[best_cluster]);
      std::vector<size_t> cluster_indices;
      for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == best_cluster) {
          cluster_indices.push_back(i);
        }
      }
      if (cluster_indices.empty()) {
        RCLCPP_WARN(get_logger(), "RANSAC fallback: no points in best cluster.");
        return;
      }

      const size_t max_sample_size = 500;
      std::vector<size_t> sampled_indices;
      if (cluster_indices.size() > max_sample_size) {
        size_t step = cluster_indices.size() / max_sample_size;
        for (size_t i = 0; i < cluster_indices.size(); i += step) {
          sampled_indices.push_back(cluster_indices[i]);
          if (sampled_indices.size() >= max_sample_size) break;
        }
        RCLCPP_INFO(get_logger(), "Sampled %zu/%zu cluster points for grasp computation",
                    sampled_indices.size(), cluster_indices.size());
      } else {
        sampled_indices = cluster_indices;
        RCLCPP_INFO(get_logger(), "Computing grasp score for %zu cluster points...",
                    sampled_indices.size());
      }

      const float sigma = 0.1f;
      const float inv_sigma_sq = 1.0f / (sigma * sigma);
      float best_score = -std::numeric_limits<float>::infinity();
      size_t best_idx = sampled_indices.front();
      for (size_t idx_i : sampled_indices) {
        const auto & pi = edge_points[idx_i];
        const auto & ni = edge_normals[idx_i];
        float score = 0.0f;
        for (size_t idx_j : sampled_indices) {
          if (idx_i == idx_j) {
            continue;
          }
          const auto & pj = edge_points[idx_j];
          const auto & nj = edge_normals[idx_j];
          float dot = ni.dot(nj);
          float dist = (pi - pj).norm();
          score += dot * std::exp(-0.5f * dist * dist * inv_sigma_sq);
        }
        if (score > best_score) {
          best_score = score;
          best_idx = idx_i;
        }
      }
      Eigen::Vector3f centroid = edge_points[best_idx];
      Eigen::Vector3f normal_sum = edge_normals[best_idx];
      RCLCPP_INFO(get_logger(), "Grasp centroid computed: [%.3f, %.3f, %.3f] with score %.3f",
                  centroid.x(), centroid.y(), centroid.z(), best_score);
      tf2::Quaternion q = normal_to_quaternion(normal_sum);

      // Point cloud is already in base_link frame
      std_msgs::msg::Header header;
      header.stamp = msg->header.stamp;
      header.frame_id = resolved_base_frame_;

      // Create pose directly in base_link frame (no transformation needed)
      geometry_msgs::msg::PoseStamped pose_base;
      pose_base.header = header;
      pose_base.pose.position.x = centroid.x();
      pose_base.pose.position.y = centroid.y();
      pose_base.pose.position.z = centroid.z();
      pose_base.pose.orientation.x = q.x();
      pose_base.pose.orientation.y = q.y();
      pose_base.pose.orientation.z = q.z();
      pose_base.pose.orientation.w = q.w();

      if (!validate_grasp_height(pose_base)) {
        return;
      }

      // Tilt correction disabled - output world Z directly for Gazebo simulation
      // geometry_msgs::msg::PoseStamped pose_corrected = pose_base;
      // pose_corrected.pose.position.z = apply_tilt_correction(
      //   pose_base.pose.position.x,
      //   pose_base.pose.position.y,
      //   pose_base.pose.position.z);

      pose_pub_->publish(pose_base);
      broadcast_target_tf(pose_base);

      // Publish edge points for visualization (in base_link frame)
      if (edge_pub_->get_subscription_count() > 0) {
        RCLCPP_INFO(get_logger(), "[RANSAC PATH] Publishing %zu edge points in frame '%s'",
          edge_points.size(), resolved_base_frame_.c_str());

        pcl::PointCloud<PointT>::Ptr edges_base(new pcl::PointCloud<PointT>());
        edges_base->header.frame_id = resolved_base_frame_;
        edges_base->header.stamp = pcl_conversions::toPCL(header.stamp);
        edges_base->points.reserve(edge_points.size());
        for (const auto& pt : edge_points) {
          PointT p;
          p.x = pt.x();
          p.y = pt.y();
          p.z = pt.z();
          p.r = 255;  // Red color for edges
          p.g = 0;
          p.b = 0;
          edges_base->points.push_back(p);
        }
        edges_base->width = edges_base->points.size();
        edges_base->height = 1;
        edges_base->is_dense = true;

        sensor_msgs::msg::PointCloud2 edges_msg;
        pcl::toROSMsg(*edges_base, edges_msg);
        edges_msg.header.stamp = msg->header.stamp;
        edges_msg.header.frame_id = resolved_base_frame_;
        edge_pub_->publish(edges_msg);
      }

      return;
    }

    // Verify organized structure is still valid
    // IntegralImageNormalEstimation requires width >= 3 and height >= 3
    if (!filtered->isOrganized() || filtered->width < 3 || filtered->height < 3) {
      RCLCPP_WARN(get_logger(),
                  "Filtered cloud has invalid organized structure (width=%u, height=%u). Skipping frame.",
                  filtered->width, filtered->height);
      return;
    }

    RCLCPP_INFO(get_logger(), "DEBUG: Organized cloud: width=%u, height=%u",
                filtered->width, filtered->height);

    // Compute organized normals (integral image) to feed OrganizedEdgeFromNormals.
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
    ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setDepthDependentSmoothing(true);
    ne.setInputCloud(filtered);

    try {
      ne.compute(*normals);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Normal estimation failed: %s. Skipping frame.", e.what());
      return;
    } catch (...) {
      RCLCPP_ERROR(get_logger(), "Normal estimation failed with unknown error. Skipping frame.");
      return;
    }

    if (normals->empty() || normals->size() != filtered->size()) {
      RCLCPP_WARN(get_logger(), "Normal estimation produced invalid results. Skipping frame.");
      return;
    }
    RCLCPP_INFO(get_logger(), "DEBUG: Normals computed for %zu points", normals->size());

    // Edge detection (organized) -> edge map
    pcl::PointCloud<pcl::Label> edge_labels;
    std::vector<pcl::PointIndices> label_indices;
    pcl::OrganizedEdgeFromNormals<PointT, pcl::Normal, pcl::Label> oed;
    int edge_types = pcl::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_OCCLUDING;
    oed.setEdgeType(edge_types);
    oed.setInputCloud(filtered);
    oed.setInputNormals(normals);

    try {
      oed.compute(edge_labels, label_indices);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Edge detection failed: %s. Skipping frame.", e.what());
      return;
    } catch (...) {
      RCLCPP_ERROR(get_logger(), "Edge detection failed with unknown error. Skipping frame.");
      return;
    }
    (void)edge_labels;  // labels per point not used downstream; indices suffice

    std::vector<uint8_t> edge_map(static_cast<size_t>(width * height), 0);
    auto mark_edges = [&edge_map](const pcl::PointIndices & indices) {
        for (int idx : indices.indices) {
          if (idx >= 0 && static_cast<size_t>(idx) < edge_map.size()) {
            edge_map[static_cast<size_t>(idx)] = 1;
          }
        }
      };
    // Use occluding + high curvature as per paper focus
    if (label_indices.size() >
      static_cast<size_t>(pcl::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_HIGH_CURVATURE))
    {
      mark_edges(label_indices[
        pcl::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_OCCLUDING]);
      mark_edges(label_indices[
        pcl::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_HIGH_CURVATURE]);
    }
    size_t edge_pixel_count = std::accumulate(edge_map.begin(), edge_map.end(), static_cast<size_t>(0));
    if (edge_pixel_count < 5) {
      RCLCPP_WARN(get_logger(), "Too few edge pixels detected (%zu).", edge_pixel_count);
    } else {
      RCLCPP_INFO(get_logger(), "DEBUG: Edge pixels detected: %zu", edge_pixel_count);
    }

    pcl::PointCloud<PointT>::Ptr edges(new pcl::PointCloud<PointT>());

    struct Candidate
    {
      Eigen::Vector3f p;
      Eigen::Vector3f n;
    };
    std::vector<Candidate> candidates;
    auto idx_from_xy = [width](int x, int y) {return y * width + x;};

    // 4-direction scanning over 2D edge map
    const std::array<std::pair<int, int>, 4> directions = {
      std::make_pair(1, 0),   // horizontal
      std::make_pair(0, 1),   // vertical
      std::make_pair(1, 1),   // diagonal /
      std::make_pair(1, -1)   // diagonal down-right to up-left
    };

    auto point_from_idx = [&](int idx) -> std::optional<PointT> {
        const auto & p = filtered->points[static_cast<size_t>(idx)];
        if (!is_finite(p)) {
          return std::nullopt;
        }
        return p;
      };

    auto normal_from_idx = [&](int idx) -> std::optional<Eigen::Vector3f> {
        const auto & n = normals->points[static_cast<size_t>(idx)];
        if (!is_finite(n)) {
          return std::nullopt;
        }
        Eigen::Vector3f v(n.normal_x, n.normal_y, n.normal_z);
        if (v.norm() < 1e-6f) {
          return std::nullopt;
        }
        v.normalize();
        return v;
      };

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int start_idx = idx_from_xy(x, y);
        if (edge_map[static_cast<size_t>(start_idx)] == 0) {
          continue;
        }
        auto p_start_opt = point_from_idx(start_idx);
        auto n_start_opt = normal_from_idx(start_idx);
        if (!p_start_opt || !n_start_opt) {
          continue;
        }

        // Keep a copy for visualization
        PointT vis_pt = *p_start_opt;
        vis_pt.r = 255; vis_pt.g = 0; vis_pt.b = 0;
        edges->points.push_back(vis_pt);

        for (const auto & dir : directions) {
          int dx = dir.first;
          int dy = dir.second;
          int px = x - dx;
          int py = y - dy;
          // Avoid duplicating the same segment in opposite traversal
          if (px >= 0 && px < width && py >= 0 && py < height) {
            int prev_idx = idx_from_xy(px, py);
            if (edge_map[static_cast<size_t>(prev_idx)] == 1) {
              continue;
            }
          }

          int cx = x + dx;
          int cy = y + dy;
          int end_idx = -1;
          while (cx >= 0 && cx < width && cy >= 0 && cy < height) {
            int idx = idx_from_xy(cx, cy);
            if (edge_map[static_cast<size_t>(idx)] == 1) {
              end_idx = idx;
              break;
            }
            cx += dx;
            cy += dy;
          }

          if (end_idx < 0) {
            continue;
          }

          auto p_end_opt = point_from_idx(end_idx);
          auto n_end_opt = normal_from_idx(end_idx);
          if (!p_end_opt || !n_end_opt) {
            continue;
          }

          Eigen::Vector3f ps(p_start_opt->x, p_start_opt->y, p_start_opt->z);
          Eigen::Vector3f pe(p_end_opt->x, p_end_opt->y, p_end_opt->z);
          float seg_len = (pe - ps).norm();
          if (seg_len < static_cast<float>(min_grasp_len_) ||
            seg_len > static_cast<float>(max_grasp_len_))
          {
            continue;
          }

          Eigen::Vector3f mid = 0.5f * (ps + pe);
          if (has_table_height_base_) {
            // In base_link, Z is height (up is positive)
            // Keep points above table surface
            const float min_z = last_table_height_base_ - 0.02f;  // 2cm below table (tolerance)
            const float max_z = last_table_height_base_ + 0.20f;  // up to 20cm above table
            if (mid.z() < min_z || mid.z() > max_z) {
              continue;
            }
          }
          Eigen::Vector3f nmid = (*n_start_opt + *n_end_opt) * 0.5f;
          if (nmid.norm() < 1e-6f) {
            continue;
          }
          nmid.normalize();
          candidates.push_back({mid, nmid});
        }
      }
    }

    edges->width = edges->points.size();
    edges->height = 1;
    edges->is_dense = false;

    if (candidates.size() < 2) {
      RCLCPP_WARN(
        get_logger(),
        "Not enough grasp candidates after edge scan (got %zu). Check organized input and filters.",
        candidates.size());
      return;
    }

    // GMM clustering (OpenCV EM) over (p, n)
    const int feature_dim = 6;
    cv::Mat samples(static_cast<int>(candidates.size()), feature_dim, CV_32F);
    for (size_t i = 0; i < candidates.size(); ++i) {
      for (int k = 0; k < 3; ++k) {
        samples.at<float>(static_cast<int>(i), k) = candidates[i].p[k];
        samples.at<float>(static_cast<int>(i), k + 3) = candidates[i].n[k];
      }
    }

    cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
    int clusters = std::max(1, std::min<int>(4, static_cast<int>(candidates.size())));
    em->setClustersNumber(clusters);
    em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_GENERIC);
    em->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 50, 1e-3));

    if (!em->trainEM(samples, cv::noArray(), cv::noArray(), cv::noArray())) {
      RCLCPP_WARN(get_logger(), "GMM training failed.");
      return;
    }

    std::vector<int> candidate_labels(samples.rows);
    std::vector<int> counts(clusters, 0);
    cv::Mat probs;
    for (int i = 0; i < samples.rows; ++i) {
      cv::Mat sample = samples.row(i);
      cv::Vec2d res = em->predict2(sample, probs);
      int lbl = static_cast<int>(res[1]);
      lbl = std::clamp(lbl, 0, clusters - 1);
      candidate_labels[static_cast<size_t>(i)] = lbl;
      counts[static_cast<size_t>(lbl)]++;
    }

    int best_cluster = static_cast<int>(std::distance(
      counts.begin(),
      std::max_element(counts.begin(), counts.end())));

    std::vector<size_t> cluster_indices;
    cluster_indices.reserve(candidates.size());
    for (size_t i = 0; i < candidate_labels.size(); ++i) {
      if (candidate_labels[i] == best_cluster) {
        cluster_indices.push_back(i);
      }
    }

    if (cluster_indices.empty()) {
      RCLCPP_WARN(get_logger(), "No candidates in the largest cluster.");
      return;
    }

    // Score within the best cluster using J(k_i)
    const float sigma = 0.1f;
    const float inv_sigma_sq = 1.0f / (sigma * sigma);
    float best_score = -std::numeric_limits<float>::infinity();
    size_t best_idx = cluster_indices.front();

    for (size_t idx_i : cluster_indices) {
      const auto & ci = candidates[idx_i];
      float score = 0.0f;
      for (size_t idx_j : cluster_indices) {
        if (idx_i == idx_j) {
          continue;
        }
        const auto & cj = candidates[idx_j];
        float dot = ci.n.dot(cj.n);
        float dist = (ci.p - cj.p).norm();
        float w = std::exp(-0.5f * dist * dist * inv_sigma_sq);
        score += dot * w;
      }
      if (score > best_score) {
        best_score = score;
        best_idx = idx_i;
      }
    }

    const auto & chosen = candidates[best_idx];
    Eigen::Vector3f centroid = chosen.p;
    Eigen::Vector3f normal_sum = chosen.n;

    tf2::Quaternion q = normal_to_quaternion(normal_sum);

    // Point cloud is already in base_link frame
    std_msgs::msg::Header header;
    header.stamp = msg->header.stamp;
    header.frame_id = resolved_base_frame_;

    // Create pose directly in base_link frame (no transformation needed)
    geometry_msgs::msg::PoseStamped pose_base;
    pose_base.header = header;
    pose_base.pose.position.x = centroid.x();
    pose_base.pose.position.y = centroid.y();
    pose_base.pose.position.z = centroid.z();
    pose_base.pose.orientation.x = q.x();
    pose_base.pose.orientation.y = q.y();
    pose_base.pose.orientation.z = q.z();
    pose_base.pose.orientation.w = q.w();

    if (!validate_grasp_height(pose_base)) {
      return;
    }

    // Tilt correction disabled - output world Z directly for Gazebo simulation
    // geometry_msgs::msg::PoseStamped pose_corrected = pose_base;
    // pose_corrected.pose.position.z = apply_tilt_correction(
    //   pose_base.pose.position.x,
    //   pose_base.pose.position.y,
    //   pose_base.pose.position.z);

    pose_pub_->publish(pose_base);
    broadcast_target_tf(pose_base);

    // Publish edge points for visualization (in base_link frame)
    if (edge_pub_->get_subscription_count() > 0) {
      RCLCPP_INFO(get_logger(), "[ORGANIZED PATH] Publishing %zu edge points in frame '%s'",
        edges->size(), resolved_base_frame_.c_str());

      sensor_msgs::msg::PointCloud2 edges_msg;
      edges->header.frame_id = resolved_base_frame_;
      pcl::toROSMsg(*edges, edges_msg);
      edges_msg.header.stamp = msg->header.stamp;
      edges_msg.header.frame_id = resolved_base_frame_;
      edge_pub_->publish(edges_msg);
    }

    RCLCPP_INFO(
      get_logger(),
      "Published grasp target at (%.3f, %.3f, %.3f) in frame '%s' with normal (%.3f, %.3f, %.3f) "
      "(points: %zu edges: %zu cluster_size: %zu)",
      pose_base.pose.position.x,
      pose_base.pose.position.y,
      pose_base.pose.position.z,
      pose_base.header.frame_id.c_str(),
      normal_sum.x(), normal_sum.y(), normal_sum.z(),
      filtered->size(), edges->size(), cluster_indices.size());
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ColorFilteredGraspPclNode>());
  rclcpp::shutdown();
  return 0;
}
