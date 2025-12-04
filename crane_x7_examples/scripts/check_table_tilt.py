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

"""Check table tilt using point cloud data."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
import struct
import numpy as np
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs

class TableTiltChecker(Node):
    def __init__(self):
        super().__init__("table_tilt_checker")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Use sensor data QoS (best effort reliability)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub = self.create_subscription(
            PointCloud2, "/camera/depth/color/points", self.callback, sensor_qos)
        self.checked = False

    def callback(self, msg):
        if self.checked:
            return
        self.checked = True

        # Sample points from different image locations
        # Format: (row_pct, col_pct) - percentage of image height/width
        sample_locations = [
            (0.7, 0.3),  # Front-left (image bottom)
            (0.7, 0.5),  # Front-center
            (0.7, 0.7),  # Front-right
            (0.5, 0.3),  # Mid-left
            (0.5, 0.5),  # Center
            (0.5, 0.7),  # Mid-right
            (0.3, 0.3),  # Back-left (image top)
            (0.3, 0.5),  # Back-center
            (0.3, 0.7),  # Back-right
        ]

        results = []
        for row_pct, col_pct in sample_locations:
            row = int(msg.height * row_pct)
            col = int(msg.width * col_pct)
            idx = row * msg.width + col
            offset = idx * msg.point_step

            if offset + 12 > len(msg.data):
                continue

            x, y, z = struct.unpack_from("fff", bytes(msg.data), offset)

            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            if z < 0.3 or z > 0.8:  # Table depth range
                continue

            # Transform to base_link
            try:
                pt_cam = PointStamped()
                pt_cam.header = msg.header
                pt_cam.point.x = x
                pt_cam.point.y = y
                pt_cam.point.z = z

                pt_base = self.tf_buffer.transform(pt_cam, "base_link", timeout=rclpy.duration.Duration(seconds=0.5))

                results.append({
                    "loc": "({:.1f},{:.1f})".format(row_pct, col_pct),
                    "cam": (x, y, z),
                    "base": (pt_base.point.x, pt_base.point.y, pt_base.point.z)
                })
            except Exception as e:
                self.get_logger().warn("TF error: {}".format(e))

        print("\n=== Table Tilt Analysis ===")
        print("Sample points transformed from camera_depth_optical_frame to base_link:\n")
        header = "{:<12} {:<14} {:<10} {:<10} {:<10}".format(
            "Location", "Cam Z (depth)", "Base X", "Base Y", "Base Z")
        print(header)
        print("-" * 60)

        z_values = []
        for r in results:
            cam_z = r["cam"][2]
            base_x, base_y, base_z = r["base"]
            line = "{:<12} {:<14.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
                r["loc"], cam_z, base_x, base_y, base_z)
            print(line)
            z_values.append((base_x, base_z))

        if len(z_values) >= 2:
            # Check Z range
            zs = [v[1] for v in z_values]
            z_range = max(zs) - min(zs)
            print("\nBase Z range: {:.4f}m".format(z_range))
            print("If >0.02m, table appears tilted in base_link frame")

        rclpy.shutdown()

def main():
    rclpy.init()
    node = TableTiltChecker()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
