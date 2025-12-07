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

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_moveit = get_package_share_directory("crane_x7_moveit_config")

    # Launch arguments
    start_rviz_arg = DeclareLaunchArgument(
        "start_rviz",
        default_value="false",
        description="If true, launch RViz; defaults to false to avoid duplicate RViz",
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=os.path.join(pkg_moveit, "launch", "run_move_group.rviz"),
        description="RViz config to use when start_rviz is true",
    )

    output_frame_arg = DeclareLaunchArgument(
        "output_frame",
        default_value="base_link",
        description="Output coordinate frame for grasp poses",
    )

    transform_cloud_arg = DeclareLaunchArgument(
        "transform_cloud",
        default_value="true",
        description="If true, transform point cloud to output_frame before processing",
    )

    return LaunchDescription(
        [
            start_rviz_arg,
            rviz_config_arg,
            output_frame_arg,
            transform_cloud_arg,
            Node(
                package="crane_x7_examples",
                executable="color_filtered_grasp_live.py",
                name="color_filtered_grasp_live",
                output="screen",
                parameters=[
                    {
                        "input_topic": "/camera/depth/color/points",
                        "output_frame": LaunchConfiguration("output_frame"),
                        "transform_cloud": LaunchConfiguration("transform_cloud"),
                        "process_period": 1.5,
                        "box_color": [0.6, 0.4, 0.2],
                        "color_filter_radius": 0.18,
                        # base_link座標系でのワークスペース範囲
                        "pass_x_min": 0.0,
                        "pass_x_max": 0.6,
                        "pass_y_min": -0.4,
                        "pass_y_max": 0.4,
                        "pass_z_min": 0.0,
                        "pass_z_max": 0.5,
                    }
                ],
            ),
            Node(
                condition=IfCondition(LaunchConfiguration("start_rviz")),
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", LaunchConfiguration("rviz_config")],
                output="screen",
            ),
        ]
    )
