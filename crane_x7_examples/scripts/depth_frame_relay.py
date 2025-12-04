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
Republish depth image with a unified frame_id to match RGB frame.
This avoids frame_id mismatch warnings in depth_image_proc point_cloud_xyzrgb_node.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image


class DepthFrameRelay(Node):
    def __init__(self):
        super().__init__("depth_frame_relay")
        self.declare_parameter("input_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("output_topic", "/camera/aligned_depth_to_color/image_raw_frame_fixed")
        self.declare_parameter("target_frame", "camera_depth_optical_frame")

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sub = self.create_subscription(Image, input_topic, self.callback, qos_profile=qos)
        self.pub = self.create_publisher(Image, output_topic, qos_profile=qos)
        self.get_logger().info(
            f"Relaying depth from '{input_topic}' to '{output_topic}' with frame_id='{self.target_frame}'"
        )

    def callback(self, msg: Image):
        msg.header.frame_id = self.target_frame
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = DepthFrameRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
