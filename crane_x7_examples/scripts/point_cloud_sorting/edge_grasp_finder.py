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
Edge-based grasp point detection from point clouds.
Based on color_filtered_grasp_live.py, adapted for region-specific detection.
"""

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import open3d as o3d

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from sklearn.mixture import GaussianMixture
from tf2_ros import Buffer, TransformListener, TransformException

from .color_detector import Color


@dataclass
class GraspTarget:
    """Grasp target with position and approach direction."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    color: Color = Color.NONE
    confidence: float = 0.0
    edge_count: int = 0


class EdgeGraspFinder:
    """
    Find optimal grasp points using edge detection on point clouds.

    This class:
    1. Receives point cloud data
    2. Transforms to base_link frame
    3. Filters to a specific region (from color detection)
    4. Computes curvature to find edges
    5. Clusters edges to find the best grasp point
    """

    # Processing parameters
    VOXEL_SIZE = 0.005  # 5mm voxel for downsampling
    NEIGHBOR_RADIUS = 0.02  # 2cm for curvature estimation
    MAX_NEIGHBORS = 30
    CURVATURE_PERCENTILE = 90.0  # Top 10% as edges
    GMM_COMPONENTS = 3

    def __init__(
        self,
        node: Node,
        output_frame: str = "base_link",
    ):
        """
        Initialize EdgeGraspFinder.

        Args:
            node: ROS 2 node for subscriptions and TF
            output_frame: Target coordinate frame for points
        """
        self._node = node
        self._logger = node.get_logger()
        self._output_frame = output_frame

        # TF setup
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._tf_ready = False

        # Point cloud storage
        self._latest_cloud: Optional[PointCloud2] = None

        # Subscription with sensor QoS
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._cloud_sub = node.create_subscription(
            PointCloud2,
            "/camera/depth/color/points",
            self._cloud_callback,
            qos_profile=sensor_qos,
        )

        # Publishers for RViz visualization
        self._edge_cloud_pub = node.create_publisher(
            PointCloud2, "/edge_detection/edge_points", 10
        )
        self._region_cloud_pub = node.create_publisher(
            PointCloud2, "/edge_detection/region_points", 10
        )
        self._grasp_marker_pub = node.create_publisher(
            Marker, "/edge_detection/grasp_marker", 10
        )

        self._logger.info("EdgeGraspFinder initialized")
        self._logger.info("  Visualization topics:")
        self._logger.info("    /edge_detection/edge_points (red)")
        self._logger.info("    /edge_detection/region_points (green)")
        self._logger.info("    /edge_detection/grasp_marker (sphere)")

    def _cloud_callback(self, msg: PointCloud2):
        """Store latest point cloud."""
        self._latest_cloud = msg

    def find_grasp_in_region(
        self,
        center: Tuple[float, float, float],
        radius: float = 0.05,
        color: Color = Color.NONE,
    ) -> Optional[GraspTarget]:
        """
        Find optimal grasp point within a spherical region.

        Args:
            center: Center point (x, y, z) in base_link frame
            radius: Search radius in meters
            color: Associated color for the target

        Returns:
            GraspTarget if found, None otherwise
        """
        if self._latest_cloud is None:
            self._logger.warn("No point cloud available")
            return None

        # Transform point cloud to output frame
        pcd = self._transform_and_convert(self._latest_cloud)
        if pcd is None or len(pcd.points) == 0:
            return None

        # Filter to region around center
        pcd = self._filter_to_region(pcd, center, radius)
        if len(pcd.points) < 30:
            self._logger.warn(f"Too few points in region: {len(pcd.points)}")
            return None

        # Publish region point cloud (green)
        self._publish_region_cloud(pcd)

        # Downsample
        pcd = pcd.voxel_down_sample(self.VOXEL_SIZE)
        if len(pcd.points) < 10:
            return None

        # Compute curvature and extract edges
        curvatures, normals = self._estimate_curvature(pcd)
        edges, edge_mask = self._extract_edges(pcd, curvatures)

        if len(edges.points) < self.GMM_COMPONENTS:
            self._logger.warn(f"Too few edge points: {len(edges.points)}")
            return None

        # Publish edge point cloud (red)
        self._publish_edge_cloud(edges)

        # Cluster edges to find grasp point
        edge_points = np.asarray(edges.points)
        edge_normals = normals[edge_mask]

        grasp_center, grasp_normal = self._cluster_edges(edge_points, edge_normals)

        # Publish grasp marker (sphere)
        self._publish_grasp_marker(grasp_center, color)

        return GraspTarget(
            position=grasp_center,
            normal=grasp_normal,
            color=color,
            confidence=1.0,
            edge_count=len(edge_points),
        )

    def _transform_and_convert(self, msg: PointCloud2) -> Optional[o3d.geometry.PointCloud]:
        """Transform point cloud to output frame and convert to Open3D."""
        source_frame = self._normalize_frame(msg.header.frame_id)

        try:
            transform = self._tf_buffer.lookup_transform(
                self._output_frame,
                source_frame,
                msg.header.stamp,
                timeout=Duration(seconds=0.5),
            )
        except TransformException as e:
            if self._tf_ready:
                self._logger.warn(f"TF transform failed: {e}")
            return None

        if not self._tf_ready:
            self._tf_ready = True
            self._logger.info(f"TF ready: {source_frame} -> {self._output_frame}")

        # Build transformation matrix
        tf_matrix = self._transform_to_matrix(transform)

        # Convert and transform points
        return self._convert_and_transform(msg, tf_matrix)

    def _normalize_frame(self, frame_id: str) -> str:
        """Normalize Gazebo-prefixed frame names."""
        if "optical_frame" in frame_id:
            if "/" in frame_id:
                return frame_id.split("/")[-1]
            return frame_id
        if "camera_depth" in frame_id or "camera_color" in frame_id:
            return "camera_link"
        return frame_id

    def _transform_to_matrix(self, transform) -> np.ndarray:
        """Convert TransformStamped to 4x4 transformation matrix."""
        t = transform.transform.translation
        r = transform.transform.rotation

        # Quaternion to rotation matrix
        x, y, z, w = r.x, r.y, r.z, r.w
        rot = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])

        matrix = np.eye(4)
        matrix[:3, :3] = rot
        matrix[:3, 3] = [t.x, t.y, t.z]
        return matrix

    def _convert_and_transform(
        self, msg: PointCloud2, tf_matrix: np.ndarray
    ) -> Optional[o3d.geometry.PointCloud]:
        """Convert PointCloud2 to Open3D and apply transformation."""
        field_names = [f.name for f in msg.fields]
        has_rgb = "rgb" in field_names or "rgba" in field_names
        color_field = "rgb" if "rgb" in field_names else "rgba" if "rgba" in field_names else None

        points = []
        colors = []

        for p in pc2.read_points(msg, skip_nans=True):
            # Transform point
            pt = np.array([p[0], p[1], p[2], 1.0])
            pt_transformed = tf_matrix @ pt

            points.append(pt_transformed[:3])

            # Extract color if available
            if color_field:
                color_idx = field_names.index(color_field)
                rgb_float = p[color_idx]
                rgb_uint = struct.unpack("I", struct.pack("f", rgb_float))[0]
                r = ((rgb_uint >> 16) & 0xFF) / 255.0
                g = ((rgb_uint >> 8) & 0xFF) / 255.0
                b = (rgb_uint & 0xFF) / 255.0
                colors.append([r, g, b])

        if not points:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        if colors:
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        return pcd

    def _filter_to_region(
        self,
        pcd: o3d.geometry.PointCloud,
        center: Tuple[float, float, float],
        radius: float,
    ) -> o3d.geometry.PointCloud:
        """Filter point cloud to spherical region around center."""
        points = np.asarray(pcd.points)
        center_np = np.array(center)

        distances = np.linalg.norm(points - center_np, axis=1)
        mask = distances < radius

        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(points[mask])

        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            filtered.colors = o3d.utility.Vector3dVector(colors[mask])

        return filtered

    def _estimate_curvature(
        self, pcd: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate curvature at each point."""
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.NEIGHBOR_RADIUS, max_nn=self.MAX_NEIGHBORS
            )
        )

        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        curvatures = np.zeros(len(points))
        for idx, pt in enumerate(points):
            _, neighbor_ids, _ = kdtree.search_radius_vector_3d(pt, self.NEIGHBOR_RADIUS)
            if len(neighbor_ids) < 5:
                continue

            neighbors = points[neighbor_ids]
            cov = np.cov(neighbors.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 1e-12)
            curvatures[idx] = eigvals.min() / eigvals.sum()

        return curvatures, normals

    def _extract_edges(
        self, pcd: o3d.geometry.PointCloud, curvatures: np.ndarray
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Extract edge points based on curvature threshold."""
        threshold = np.percentile(curvatures, self.CURVATURE_PERCENTILE)
        edge_mask = curvatures >= threshold
        edge_indices = np.where(edge_mask)[0]

        edges = pcd.select_by_index(edge_indices.tolist())
        edges.paint_uniform_color([1.0, 0.0, 0.0])  # Red for visualization

        return edges, edge_mask

    def _cluster_edges(
        self, edge_points: np.ndarray, edge_normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster edge points and find densest cluster center."""
        n_components = min(self.GMM_COMPONENTS, len(edge_points))

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=0,
        )
        labels = gmm.fit_predict(edge_points)

        # Find densest cluster
        counts = np.bincount(labels, minlength=n_components)
        target_label = counts.argmax()

        center = gmm.means_[target_label]

        # Average normal of cluster
        cluster_normals = edge_normals[labels == target_label]
        grasp_normal = cluster_normals.mean(axis=0)
        norm = np.linalg.norm(grasp_normal)
        if norm > 1e-6:
            grasp_normal /= norm
        else:
            grasp_normal = np.array([0.0, 0.0, 1.0])

        return center, grasp_normal

    def has_cloud(self) -> bool:
        """Check if point cloud data is available."""
        return self._latest_cloud is not None

    def _publish_region_cloud(self, pcd: o3d.geometry.PointCloud):
        """Publish region point cloud for RViz visualization (green)."""
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return

        header = Header()
        header.stamp = self._node.get_clock().now().to_msg()
        header.frame_id = self._output_frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Green color
        r, g, b = 0, 255, 0
        rgb = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]

        cloud_data = []
        for pt in points:
            cloud_data.append([pt[0], pt[1], pt[2], rgb])

        cloud_data = np.array(cloud_data, dtype=np.float32)

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(cloud_data)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = cloud_data.tobytes()

        self._region_cloud_pub.publish(msg)

    def _publish_edge_cloud(self, pcd: o3d.geometry.PointCloud):
        """Publish edge point cloud for RViz visualization (red)."""
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return

        header = Header()
        header.stamp = self._node.get_clock().now().to_msg()
        header.frame_id = self._output_frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Red color
        r, g, b = 255, 0, 0
        rgb = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]

        cloud_data = []
        for pt in points:
            cloud_data.append([pt[0], pt[1], pt[2], rgb])

        cloud_data = np.array(cloud_data, dtype=np.float32)

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(cloud_data)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = cloud_data.tobytes()

        self._edge_cloud_pub.publish(msg)

    def _publish_grasp_marker(self, position: np.ndarray, color: Color):
        """Publish grasp point marker for RViz visualization."""
        marker = Marker()
        marker.header.stamp = self._node.get_clock().now().to_msg()
        marker.header.frame_id = self._output_frame
        marker.ns = "grasp_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.03  # 3cm sphere
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        # Color based on detected color
        if color == Color.BLUE:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        elif color == Color.YELLOW:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif color == Color.GREEN:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0  # Magenta for unknown

        marker.color.a = 0.8
        marker.lifetime.sec = 5  # 5 seconds

        self._grasp_marker_pub.publish(marker)
