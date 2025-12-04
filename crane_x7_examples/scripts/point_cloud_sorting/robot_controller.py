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
Service-based robot controller for CRANE-X7.

Communicates with the C++ motion_service_node via ROS 2 services.

References:
- motion_service_node.cpp (C++ MoveIt interface)
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from std_srvs.srv import Trigger

from .color_detector import Color


@dataclass
class RobotConfig:
    """Robot configuration parameters."""
    # Pick heights (meters)
    pick_z_above: float = 0.20  # Hover height
    pick_z_lift: float = 0.30   # Lift height after grasp

    # Place positions for each color (x, y, z)
    place_positions: dict = None

    def __post_init__(self):
        if self.place_positions is None:
            self.place_positions = {
                Color.YELLOW: (0.30, 0.30, 0.23),   # Left rear
                Color.BLUE: (0.30, -0.30, 0.23),    # Right rear
                Color.GREEN: (0.45, 0.00, 0.23),    # Front
            }


class RobotController:
    """
    Service-based robot controller for pick and place operations.

    Communicates with motion_service_node via ROS 2 services.
    """

    SERVICE_TIMEOUT = 30.0  # seconds

    def __init__(
        self,
        node: Node,
        config: Optional[RobotConfig] = None,
        use_sim_time: bool = True,
    ):
        """
        Initialize RobotController.

        Args:
            node: ROS 2 node
            config: Robot configuration parameters
            use_sim_time: Whether to use simulation time
        """
        self._node = node
        self._logger = node.get_logger()
        self._config = config or RobotConfig()

        # Create service clients
        self._move_to_camera_client = node.create_client(
            Trigger, "/motion/move_to_camera_pose"
        )
        self._open_gripper_client = node.create_client(
            Trigger, "/motion/open_gripper"
        )
        self._close_gripper_client = node.create_client(
            Trigger, "/motion/close_gripper"
        )
        self._execute_pose_client = node.create_client(
            Trigger, "/motion/execute_pose"
        )
        self._execute_cartesian_client = node.create_client(
            Trigger, "/motion/execute_cartesian"
        )

        # Create target pose publisher
        self._target_pose_pub = node.create_publisher(
            PoseStamped, "/motion/target_pose", 10
        )

        self._logger.info("RobotController initialized (service-based)")

    def _wait_for_future(self, future, timeout_sec: float = None) -> bool:
        """
        Wait for a future to complete using polling.

        This is used instead of spin_until_future_complete because
        the executor is already spinning in a background thread.

        Args:
            future: The future to wait for
            timeout_sec: Timeout in seconds (default: SERVICE_TIMEOUT)

        Returns:
            True if future completed, False if timed out
        """
        if timeout_sec is None:
            timeout_sec = self.SERVICE_TIMEOUT

        start_time = time.time()
        while not future.done():
            if time.time() - start_time > timeout_sec:
                return False
            time.sleep(0.05)  # 50ms polling interval
        return True

    def wait_for_services(self, timeout_sec: float = 30.0) -> bool:
        """Wait for motion services to be available."""
        self._logger.info("Waiting for motion services...")

        services = [
            ("/motion/move_to_camera_pose", self._move_to_camera_client),
            ("/motion/open_gripper", self._open_gripper_client),
            ("/motion/close_gripper", self._close_gripper_client),
            ("/motion/execute_pose", self._execute_pose_client),
            ("/motion/execute_cartesian", self._execute_cartesian_client),
        ]

        for name, client in services:
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self._logger.error(f"Service {name} not available")
                return False
            self._logger.info(f"  {name}: ready")

        self._logger.info("All motion services available")
        return True

    def move_to_camera_pose(self) -> bool:
        """Move to camera observation pose."""
        self._logger.info("Moving to camera observation pose")

        request = Trigger.Request()
        future = self._move_to_camera_client.call_async(request)

        # Wait for result using polling
        if not self._wait_for_future(future):
            self._logger.error("Service call timed out")
            return False

        if future.result() is None:
            self._logger.error("Service call failed")
            return False

        result = future.result()
        if not result.success:
            self._logger.warn(f"Move to camera pose failed: {result.message}")
        return result.success

    def open_gripper(self) -> bool:
        """Open the gripper."""
        self._logger.info("Opening gripper")

        request = Trigger.Request()
        future = self._open_gripper_client.call_async(request)

        if not self._wait_for_future(future):
            self._logger.error("Service call timed out")
            return False

        if future.result() is None:
            self._logger.error("Service call failed")
            return False

        result = future.result()
        return result.success

    def close_gripper(self) -> bool:
        """Close the gripper."""
        self._logger.info("Closing gripper")

        request = Trigger.Request()
        future = self._close_gripper_client.call_async(request)

        if not self._wait_for_future(future):
            self._logger.error("Service call timed out")
            return False

        if future.result() is None:
            self._logger.error("Service call failed")
            return False

        result = future.result()
        return result.success

    def _move_to_pose(self, pose: Pose, cartesian: bool = False) -> bool:
        """Move end effector to pose."""
        # Publish target pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.header.stamp = self._node.get_clock().now().to_msg()
        pose_stamped.pose = pose
        self._target_pose_pub.publish(pose_stamped)

        # Small delay to ensure pose is received
        time.sleep(0.1)

        # Call appropriate service
        if cartesian:
            client = self._execute_cartesian_client
        else:
            client = self._execute_pose_client

        request = Trigger.Request()
        future = client.call_async(request)

        if not self._wait_for_future(future):
            self._logger.error("Service call timed out")
            return False

        if future.result() is None:
            self._logger.error("Service call failed")
            return False

        result = future.result()
        if not result.success:
            self._logger.warn(f"Move to pose failed: {result.message}")
        return result.success

    def pick_at(
        self,
        position: Tuple[float, float, float],
        yaw_deg: float = 0.0,
    ) -> bool:
        """
        Execute pick operation at given position.

        Args:
            position: (x, y, z) in base_link frame
            yaw_deg: Gripper rotation angle in degrees

        Returns:
            True if successful
        """
        x, y, z = position

        # Step 1: Move to hover position
        self._logger.info(f"Step 1: Moving to hover position above ({x:.3f}, {y:.3f})")
        hover_pose = self._create_pick_pose(x, y, self._config.pick_z_above, yaw_deg)
        if not self._move_to_pose(hover_pose):
            self._logger.warn("Failed to reach hover position")
            return False

        # Step 2: Open gripper
        self._logger.info("Step 2: Opening gripper")
        self.open_gripper()

        # Step 3: Descend to pick height
        pick_z = self._compute_pick_height(z)
        self._logger.info(f"Step 3: Descending to pick height (z={pick_z:.3f})")
        pick_pose = self._create_pick_pose(x, y, pick_z, yaw_deg)
        if not self._move_to_pose(pick_pose, cartesian=True):
            self._logger.warn("Failed to descend")
            return False

        # Step 3.5: Wait for arm to stabilize before gripping
        self._logger.info("Step 3.5: Waiting for arm to stabilize...")
        time.sleep(0.85)

        # Step 4: Close gripper
        self._logger.info("Step 4: Closing gripper")
        self.close_gripper()

        # Step 5: Lift object
        self._logger.info("Step 5: Lifting object")
        lift_pose = self._create_pick_pose(x, y, self._config.pick_z_lift, yaw_deg)
        if not self._move_to_pose(lift_pose, cartesian=True):
            self._logger.warn("Failed to lift, trying non-cartesian")
            if not self._move_to_pose(lift_pose, cartesian=False):
                self.open_gripper()  # Release if lift fails
                return False

        return True

    def place_at_color(self, color: Color) -> bool:
        """
        Place object at color-specific location.

        Args:
            color: Target color for placement location

        Returns:
            True if successful
        """
        if color not in self._config.place_positions:
            self._logger.warn(f"No place position for color {color}")
            return False

        x, y, z = self._config.place_positions[color]
        self._logger.info(f"Placing {color.name} at ({x:.2f}, {y:.2f}, {z:.2f})")

        # Move to place position
        place_pose = self._create_pick_pose(x, y, z, 0.0)
        if not self._move_to_pose(place_pose):
            self._logger.warn("Failed to reach place position")
            self.open_gripper()
            return False

        # Release object
        self.open_gripper()

        return True

    def _create_pick_pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw_deg: float,
    ) -> Pose:
        """Create pick/place pose with downward orientation."""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        # Orientation: pointing down (roll=180, pitch=0, yaw=variable)
        roll = math.radians(180.0)
        pitch = 0.0
        yaw = math.radians(-yaw_deg)

        # Convert RPY to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        pose.orientation.w = cr * cp * cy + sr * sp * sy
        pose.orientation.x = sr * cp * cy - cr * sp * sy
        pose.orientation.y = cr * sp * cy + sr * cp * sy
        pose.orientation.z = cr * cp * sy - sr * sp * cy

        return pose

    def _compute_pick_height(self, detected_z: float) -> float:
        """Compute optimal pick height based on detected Z."""
        # Constants from color_sorting.cpp
        TABLE_HEIGHT = 0.0
        CUBE_HALF = 0.025
        GRAB_CLEARANCE = -0.025
        GRIPPER_LENGTH = 0.13

        # Clamp detected_z to valid range
        Z_MIN_CLAMP = -0.03
        Z_MAX_CLAMP = -0.015
        if detected_z < Z_MIN_CLAMP:
            detected_z = Z_MIN_CLAMP
        elif detected_z > Z_MAX_CLAMP:
            detected_z = Z_MAX_CLAMP

        nominal = TABLE_HEIGHT + CUBE_HALF + GRAB_CLEARANCE + GRIPPER_LENGTH
        delta = detected_z - TABLE_HEIGHT
        pick = nominal + delta

        # Clamp to valid pick range
        MIN_PICK = 0.0
        MAX_PICK = TABLE_HEIGHT + 0.15 + GRIPPER_LENGTH
        return max(MIN_PICK, min(MAX_PICK, pick))

    def wait_for_ready(self, timeout_sec: float = 30.0) -> bool:
        """Wait for robot to be ready (services available)."""
        return self.wait_for_services(timeout_sec)
