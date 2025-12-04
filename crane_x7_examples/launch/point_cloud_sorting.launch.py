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
Launch file for point cloud-based color sorting demo.

This demo combines HSV color detection with edge-based grasp point detection
for improved pick accuracy.

Launches:
- motion_service_node: C++ node providing MoveIt services
- point_cloud_sorting_node: Python node for detection and control logic
"""

import os

from ament_index_python.packages import get_package_share_directory
from crane_x7_description.robot_description_loader import RobotDescriptionLoader
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
import yaml


def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:
        return None


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None


def generate_launch_description():
    # Load robot description for MoveIt
    description_loader = RobotDescriptionLoader()

    robot_description_semantic_config = load_file(
        'crane_x7_moveit_config', 'config/crane_x7.srdf')
    robot_description_semantic = {
        'robot_description_semantic': robot_description_semantic_config}

    kinematics_yaml = load_yaml('crane_x7_moveit_config', 'config/kinematics.yaml')

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time",
    )

    num_objects_arg = DeclareLaunchArgument(
        "num_objects",
        default_value="5",
        description="Number of objects to spawn",
    )

    use_edge_detection_arg = DeclareLaunchArgument(
        "use_edge_detection",
        default_value="true",
        description="Use edge detection for grasp refinement",
    )

    # Motion service node (C++ MoveIt interface)
    # Requires robot_description parameters for MoveGroupInterface
    motion_service_node = Node(
        package="crane_x7_examples",
        executable="motion_service_node",
        name="motion_service",
        output="screen",
        parameters=[
            {"robot_description": description_loader.load()},
            robot_description_semantic,
            kinematics_yaml,
        ],
    )

    # Point cloud sorting node (Python detection and control logic)
    point_cloud_sorting_node = Node(
        package="crane_x7_examples",
        executable="point_cloud_sorting_node.py",
        name="point_cloud_sorting",
        output="screen",
        parameters=[
            {
                "num_objects": LaunchConfiguration("num_objects"),
                "use_edge_detection": LaunchConfiguration("use_edge_detection"),
            }
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        num_objects_arg,
        use_edge_detection_arg,
        SetParameter(name='use_sim_time', value=LaunchConfiguration('use_sim_time')),
        motion_service_node,
        point_cloud_sorting_node,
    ])
