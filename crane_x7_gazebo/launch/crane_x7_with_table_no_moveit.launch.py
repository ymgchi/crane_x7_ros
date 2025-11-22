# Copyright 2022 RT Corporation
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
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch_ros.actions import SetParameter


def generate_launch_description():
    # PATHを追加で通さないとSTLファイルが読み込まれない
    env = {'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': os.environ['LD_LIBRARY_PATH'],
           'IGN_GAZEBO_RESOURCE_PATH': os.path.dirname(
               get_package_share_directory('crane_x7_description'))}
    world_file = os.path.join(
        get_package_share_directory('crane_x7_gazebo'), 'worlds', 'table.sdf')
    gui_config = os.path.join(
        get_package_share_directory('crane_x7_gazebo'), 'gui', 'gui.config')
    # -r オプションで起動時にシミュレーションをスタートしないと、コントローラが起動しない
    ign_gazebo = ExecuteProcess(
            cmd=['ign gazebo -r', world_file, '--gui-config', gui_config],
            output='screen',
            additional_env=env,
            shell=True
        )

    ignition_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', '/robot_description',
                   '-name', 'crane_x7',
                   '-z', '1.015',
                   '-allow_renaming', 'true'],
    )

    description_loader = RobotDescriptionLoader()
    description_loader.use_gazebo = 'true'
    description_loader.gz_control_config_package = 'crane_x7_control'
    description_loader.gz_control_config_file_path = 'config/crane_x7_controllers.yaml'
    description = description_loader.load()

    # MoveItは起動しない（プログラム側で起動する）

    spawn_joint_state_controller = ExecuteProcess(
                cmd=['ros2 run controller_manager spawner joint_state_controller'],
                shell=True,
                output='screen',
            )

    spawn_arm_controller = ExecuteProcess(
                cmd=['ros2 run controller_manager spawner crane_x7_arm_controller'],
                shell=True,
                output='screen',
            )

    spawn_gripper_controller = ExecuteProcess(
                cmd=['ros2 run controller_manager spawner crane_x7_gripper_controller'],
                shell=True,
                output='screen',
            )

    bridge = Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
                output='screen'
            )

    # Camera bridges
    camera_image_bridge = Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                arguments=[
                    '/camera/color@sensor_msgs/msg/Image[ignition.msgs.Image',
                    '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
                    '/camera/depth@sensor_msgs/msg/Image[ignition.msgs.Image'
                ],
                remappings=[
                    ('/camera/color', '/camera/color/image_raw'),
                    ('/camera/camera_info', '/camera/color/camera_info'),
                    ('/camera/depth', '/camera/aligned_depth_to_color/image_raw')
                ],
                output='screen'
            )

    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        ign_gazebo,
        ignition_spawn_entity,
        # move_groupは起動しない
        spawn_joint_state_controller,
        spawn_arm_controller,
        spawn_gripper_controller,
        bridge,
        camera_image_bridge
    ])
