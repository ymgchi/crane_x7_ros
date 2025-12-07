#!/usr/bin/env python3
# Copyright 2025 ymgchi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Live color-filtered grasp candidate finder for Gazebo/RealSense point clouds.

Subscribes to PointCloud2 (e.g., /camera/depth/color/points), optionally transforms
to robot base frame, filters out points near a specified box color, extracts edges
via curvature, clusters with GMM, and publishes the densest cluster center as a
grasp target. Optional edge points are published for RViz visualization.
"""

import math
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from sklearn.mixture import GaussianMixture
from std_msgs.msg import Header

# TF2 imports
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs  # noqa: F401 (required for PoseStamped transform)


@dataclass
class LiveConfig:
    input_topic: str = "/camera/depth/color/points"
    output_frame: str = "base_link"  # 出力座標系
    transform_cloud: bool = True  # 点群をbase_linkに変換するか
    process_period: float = 1.5  # seconds between processing frames
    voxel_size: float = 0.01
    # base_link座標系でのワークスペース範囲
    pass_x: tuple = (0.0, 0.6)    # ロボット前方
    pass_y: tuple = (-0.4, 0.4)   # 左右
    pass_z: tuple = (0.0, 0.5)    # 高さ（テーブル上）
    neighbor_radius: float = 0.03
    max_neighbors: int = 40
    curvature_percentile: float = 92.0
    color_filter_radius: float = -1.0  # 負値で色フィルタ無効（デバッグ優先）
    box_color: tuple = (0.6, 0.4, 0.2)  # 茶色（テーブル色）
    gmm_components: int = 3


class PointCloudPreprocessor:
    def __init__(self, exclusion_color: tuple, radius: float):
        self.exclusion_color = np.array(exclusion_color)
        self.radius = radius

    def apply(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        distances = np.linalg.norm(colors - self.exclusion_color, axis=1)
        keep_mask = distances > self.radius

        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(pts[keep_mask])
        filtered.colors = o3d.utility.Vector3dVector(colors[keep_mask])
        return filtered


def estimate_curvature(
    pcd: o3d.geometry.PointCloud, radius: float, max_nn: int
) -> tuple[np.ndarray, np.ndarray]:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = np.zeros(len(pts))
    for idx, pt in enumerate(pts):
        _, neighbor_ids, _ = kdtree.search_radius_vector_3d(pt, radius)
        if len(neighbor_ids) < 5:
            continue
        neighbors = pts[neighbor_ids]
        cov = np.cov(neighbors.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        curvatures[idx] = eigvals.min() / eigvals.sum()
    return curvatures, normals


def extract_edges(pcd: o3d.geometry.PointCloud, curvatures: np.ndarray, percentile: float):
    threshold = np.percentile(curvatures, percentile)
    edge_mask = curvatures >= threshold
    edge_indices = np.where(edge_mask)[0]
    edges = pcd.select_by_index(edge_indices)
    edges.paint_uniform_color([1.0, 0.0, 0.0])
    return edges, edge_mask


def cluster_edges(
    edge_points: np.ndarray, edge_normals: np.ndarray, config: LiveConfig
) -> tuple[np.ndarray, np.ndarray]:
    gmm = GaussianMixture(
        n_components=config.gmm_components,
        covariance_type="full",
        random_state=0,
    )
    labels = gmm.fit_predict(edge_points)
    counts = np.bincount(labels, minlength=config.gmm_components)
    target_label = counts.argmax()
    center = gmm.means_[target_label]
    grasp_normals = edge_normals[labels == target_label]
    grasp_normal = grasp_normals.mean(axis=0)
    norm = np.linalg.norm(grasp_normal)
    if norm > 1e-6:
        grasp_normal /= norm
    else:
        grasp_normal = np.array([0.0, 0.0, 1.0])
    return center, grasp_normal


def pointcloud2_to_o3d(msg: PointCloud2) -> Optional[o3d.geometry.PointCloud]:
    # Use read_points to pull XYZ and RGB (or RGBA) data.
    field_names = [f.name for f in msg.fields]
    has_rgb = "rgb" in field_names
    has_rgba = "rgba" in field_names
    if not (has_rgb or has_rgba):
        return None

    data = []
    color_idx = field_names.index("rgb") if has_rgb else field_names.index("rgba")
    for p in pc2.read_points(msg, skip_nans=True):
        # p is tuple including x,y,z and rgb/rgba float packed
        if len(p) <= color_idx:
            continue
        x, y, z = p[0], p[1], p[2]
        rgb_float = p[color_idx]
        # Pack float into int then split channels
        rgb_uint = struct.unpack("I", struct.pack("f", rgb_float))[0]
        r = (rgb_uint & 0x00FF0000) >> 16
        g = (rgb_uint & 0x0000FF00) >> 8
        b = (rgb_uint & 0x000000FF)
        data.append((x, y, z, r, g, b))

    if not data:
        return None

    arr = np.array(data, dtype=np.float32)
    pts = arr[:, :3]
    colors = arr[:, 3:] / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def o3d_to_pointcloud2(pcd: o3d.geometry.PointCloud, frame_id: str, stamp) -> PointCloud2:
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    cloud_data = []
    for p, c in zip(pts, cols):
        r, g, b = (np.clip(c * 255, 0, 255)).astype(np.uint8)
        rgb_uint = (int(r) << 16) | (int(g) << 8) | int(b)
        rgb_float = struct.unpack("f", struct.pack("I", rgb_uint))[0]
        cloud_data.append((p[0], p[1], p[2], rgb_float))
    return pc2.create_cloud(header, fields, cloud_data)


def normal_to_quaternion(normal: np.ndarray) -> tuple[float, float, float, float]:
    """Compute quaternion that rotates +Z to the given normal."""
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, normal)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    axis /= axis_norm
    angle = math.acos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
    half = angle / 2.0
    sin_half = math.sin(half)
    return (axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, math.cos(half))


class ColorFilteredGraspLive(Node):
    def __init__(self):
        super().__init__("color_filtered_grasp_live")
        self.cfg = LiveConfig()
        self.preproc = PointCloudPreprocessor(self.cfg.box_color, self.cfg.color_filter_radius)
        self.last_process = 0.0

        # パラメータ宣言
        self.declare_parameter("input_topic", self.cfg.input_topic)
        self.declare_parameter("output_frame", self.cfg.output_frame)
        self.declare_parameter("transform_cloud", self.cfg.transform_cloud)
        self.declare_parameter("process_period", self.cfg.process_period)
        self.declare_parameter("box_color", list(self.cfg.box_color))
        self.declare_parameter("color_filter_radius", self.cfg.color_filter_radius)
        # ワークスペース範囲（base_link座標系）
        self.declare_parameter("pass_x_min", self.cfg.pass_x[0])
        self.declare_parameter("pass_x_max", self.cfg.pass_x[1])
        self.declare_parameter("pass_y_min", self.cfg.pass_y[0])
        self.declare_parameter("pass_y_max", self.cfg.pass_y[1])
        self.declare_parameter("pass_z_min", self.cfg.pass_z[0])
        self.declare_parameter("pass_z_max", self.cfg.pass_z[1])

        # パラメータ取得
        topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.cfg.output_frame = (
            self.get_parameter("output_frame").get_parameter_value().string_value
        )
        self.cfg.transform_cloud = (
            self.get_parameter("transform_cloud").get_parameter_value().bool_value
        )
        self.cfg.process_period = self.get_parameter("process_period").value
        box_color = self.get_parameter("box_color").value
        self.preproc.exclusion_color = np.array(box_color)
        self.preproc.radius = self.get_parameter("color_filter_radius").value

        # ワークスペース範囲
        self.cfg.pass_x = (
            self.get_parameter("pass_x_min").value,
            self.get_parameter("pass_x_max").value
        )
        self.cfg.pass_y = (
            self.get_parameter("pass_y_min").value,
            self.get_parameter("pass_y_max").value
        )
        self.cfg.pass_z = (
            self.get_parameter("pass_z_min").value,
            self.get_parameter("pass_z_max").value
        )

        # TF2セットアップ
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_ready = False  # TFが利用可能になるまで待つフラグ

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.subscription = self.create_subscription(
            PointCloud2, topic, self.pointcloud_callback, qos_profile=sensor_qos
        )
        self.pose_pub = self.create_publisher(PoseStamped, "color_filtered_grasp/target_pose", 10)
        self.edge_pub = self.create_publisher(PointCloud2, "color_filtered_grasp/edges", 10)

        self.get_logger().info("=== Color Filtered Grasp Live ===")
        self.get_logger().info(f"Input topic: {topic}")
        self.get_logger().info(f"Output frame: {self.cfg.output_frame}")
        self.get_logger().info(f"Transform cloud: {self.cfg.transform_cloud}")
        self.get_logger().info(f"Workspace X: {self.cfg.pass_x}")
        self.get_logger().info(f"Workspace Y: {self.cfg.pass_y}")
        self.get_logger().info(f"Workspace Z: {self.cfg.pass_z}")
        self.get_logger().info("=================================")

    def _get_source_frame(self, frame_id: str) -> str:
        """Normalize Gazebo-prefixed camera frames."""
        # Gazeboは crane_x7/.../camera_depth のようなprefixed frame_idを使用
        # TFツリーに存在するフレーム名に変換

        # optical frame はそのまま使用（camera_depth_optical_frame等）
        if "optical_frame" in frame_id:
            # Gazebo prefix を除去
            # 例: crane_x7/camera/camera_depth_optical_frame -> camera_depth_optical_frame
            if "/" in frame_id:
                return frame_id.split("/")[-1]
            return frame_id

        # camera_linkへの正規化（非optical frame）
        if "camera_depth" in frame_id or "camera_color" in frame_id:
            return "camera_link"

        return frame_id

    def _transform_to_matrix(self, transform) -> np.ndarray:
        """Create 4x4 transformation matrix from TransformStamped."""
        t = transform.transform.translation
        r = transform.transform.rotation

        # クォータニオンから回転行列
        x, y, z, w = r.x, r.y, r.z, r.w
        rot = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

        # 4x4同次変換行列
        matrix = np.eye(4)
        matrix[:3, :3] = rot
        matrix[:3, 3] = [t.x, t.y, t.z]
        return matrix

    def _transform_pointcloud(self, msg: PointCloud2) -> Optional[PointCloud2]:
        """Transform point cloud to output_frame (base_link)."""
        source_frame = self._get_source_frame(msg.header.frame_id)
        target_frame = self.cfg.output_frame

        if source_frame == target_frame:
            return msg

        try:
            # TF変換を取得（タイムアウト0.5秒）
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                msg.header.stamp,
                timeout=Duration(seconds=0.5)
            )

            # 変換行列を取得
            tf_matrix = self._transform_to_matrix(transform)

            # 点群データを読み取り
            field_names = [f.name for f in msg.fields]
            has_rgb = "rgb" in field_names
            has_rgba = "rgba" in field_names

            if not (has_rgb or has_rgba):
                self.get_logger().warn("Point cloud has no RGB field")
                return None

            # 点群を変換して新しいメッセージを作成
            color_field = "rgb" if has_rgb else "rgba"
            transformed_data = []

            for p in pc2.read_points(msg, skip_nans=True):
                # 元の座標
                pt = np.array([p[0], p[1], p[2], 1.0])
                # 変換適用
                pt_transformed = tf_matrix @ pt

                # RGBフィールドのインデックスを取得
                color_idx = field_names.index(color_field)
                rgb_float = p[color_idx]

                transformed_data.append((
                    pt_transformed[0],
                    pt_transformed[1],
                    pt_transformed[2],
                    rgb_float
                ))

            if not transformed_data:
                return None

            # 新しいPointCloud2メッセージを作成
            header = Header()
            header.stamp = msg.header.stamp
            header.frame_id = target_frame

            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            transformed_msg = pc2.create_cloud(header, fields, transformed_data)

            if not self.tf_ready:
                self.tf_ready = True
                self.get_logger().info(
                    f"TF ready: {source_frame} -> {target_frame}"
                )

            return transformed_msg

        except TransformException as e:
            if self.tf_ready:
                self.get_logger().warn(f"TF変換失敗: {e}")
            else:
                self.get_logger().debug(f"TF待機中: {e}")
            return None

    def pointcloud_callback(self, msg: PointCloud2):
        now = time.time()
        if now - self.last_process < self.cfg.process_period:
            return
        self.last_process = now

        # 点群をbase_linkに変換（設定で有効な場合）
        if self.cfg.transform_cloud:
            msg_transformed = self._transform_pointcloud(msg)
            if msg_transformed is None:
                return
            working_frame = self.cfg.output_frame
        else:
            msg_transformed = msg
            working_frame = self._get_source_frame(msg.header.frame_id)

        pcd = pointcloud2_to_o3d(msg_transformed)
        if pcd is None or len(pcd.points) == 0:
            self.get_logger().warn("Empty or unsupported point cloud.")
            return

        self.get_logger().info(f"[{working_frame}] Initial points: {len(pcd.points)}")

        # ワークスペースフィルタ（base_link座標系）
        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        mask = (
            (pts[:, 0] > self.cfg.pass_x[0])
            & (pts[:, 0] < self.cfg.pass_x[1])
            & (pts[:, 1] > self.cfg.pass_y[0])
            & (pts[:, 1] < self.cfg.pass_y[1])
            & (pts[:, 2] > self.cfg.pass_z[0])
            & (pts[:, 2] < self.cfg.pass_z[1])
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        self.get_logger().info(f"[{working_frame}] After workspace filter: {len(pcd.points)}")

        if self.cfg.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.cfg.voxel_size)
            self.get_logger().info(
                f"[{working_frame}] After voxel downsampling: {len(pcd.points)}"
            )

        filtered = self.preproc.apply(pcd)
        self.get_logger().info(f"[{working_frame}] After color filtering: {len(filtered.points)}")
        if len(filtered.points) < 30:
            self.get_logger().warn("Too few points after color filtering.")
            return

        curvatures, normals = estimate_curvature(
            filtered, self.cfg.neighbor_radius, self.cfg.max_neighbors
        )
        edges, edge_mask = extract_edges(filtered, curvatures, self.cfg.curvature_percentile)
        edge_points = np.asarray(edges.points)
        edge_normals = normals[edge_mask]
        if len(edge_points) < self.cfg.gmm_components:
            self.get_logger().warn("Not enough edge points for clustering.")
            return

        center, grasp_normal = cluster_edges(edge_points, edge_normals, self.cfg)
        qx, qy, qz, qw = normal_to_quaternion(grasp_normal)

        # ポーズを出力（すでにbase_link座標系）
        pose = PoseStamped()
        pose.header.frame_id = working_frame
        pose.header.stamp = msg.header.stamp
        pose.pose.position.x = float(center[0])
        pose.pose.position.y = float(center[1])
        pose.pose.position.z = float(center[2])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.pose_pub.publish(pose)

        # エッジ点群も同じ座標系で出力
        if self.edge_pub.get_subscription_count() > 0:
            self.edge_pub.publish(o3d_to_pointcloud2(edges, working_frame, msg.header.stamp))

        self.get_logger().info(
            f"[{working_frame}] Grasp target: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) "
            f"normal: ({grasp_normal[0]:.2f}, {grasp_normal[1]:.2f}, {grasp_normal[2]:.2f}) "
            f"(points: {len(filtered.points)}, edges: {len(edge_points)})"
        )


def main():
    rclpy.init()
    node = ColorFilteredGraspLive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
