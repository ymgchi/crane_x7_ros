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
from crane_x7_description.robot_description_loader import RobotDescriptionLoader
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import SetParameter, Node
from launch.substitutions import LaunchConfiguration
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
    description_loader = RobotDescriptionLoader()

    robot_description_semantic_config = load_file(
        'crane_x7_moveit_config', 'config/crane_x7.srdf')
    robot_description_semantic = {
        'robot_description_semantic': robot_description_semantic_config}

    kinematics_yaml = load_yaml('crane_x7_moveit_config', 'config/kinematics.yaml')

    declare_start_realsense = DeclareLaunchArgument(
        'start_realsense', default_value='false',
        description='Start realsense2_camera driver (set true for real hardware)'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description=('Set true when using the gazebo simulator.')
    )

    # RealSense driver (publishes /camera/depth/color/points)
    realsense_node = Node(
        condition=IfCondition(LaunchConfiguration('start_realsense')),
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        namespace='camera',
        output='screen',
        parameters=[{
            'pointcloud.enable': True,
            'pointcloud.ordered_pc': True,
            'pointcloud.allow_no_texture_points': True,
            'align_depth.enable': True,
            'rgb_camera.enable_auto_exposure': True,
            'depth_module.enable_auto_exposure': True,
            'publish_tf': True,
            'tf_publish_rate': 0.0,
            'unite_imu_method': 0,
            'enable_sync': False,
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        arguments=['--ros-args', '--log-level', 'info']
    )

    # color_filtered_grasp_live ノード
    grasp_detector_node = Node(
        package='crane_x7_examples',
        executable='color_filtered_grasp_pcl',
        name='color_filtered_grasp_pcl',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'input_topic': '/camera/depth/color/points'},
            {'process_period': 1.5}
        ]
    )

    # edge_based_sorting ノード
    sorting_node = Node(
        name='edge_based_sorting_node',
        package='crane_x7_examples',
        executable='edge_based_sorting',
        output='screen',
        parameters=[
            {'robot_description': description_loader.load()},
            robot_description_semantic,
            kinematics_yaml
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_start_realsense,
        SetParameter(name='use_sim_time', value=LaunchConfiguration('use_sim_time')),
        realsense_node,
        grasp_detector_node,
        sorting_node
    ])
