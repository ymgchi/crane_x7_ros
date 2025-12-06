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
Point cloud-based sorting demo.

Combines HSV color detection with edge-based grasp point detection
for improved pick accuracy compared to color_sorting.cpp.

Architecture:
    RGB Image -> ColorDetector -> Color + Bounding Box
                      |
                      v
    Point Cloud -> EdgeGraspFinder -> Precise Grasp Point
                      |
                      v
                RobotController -> Pick & Place
"""

import random
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from point_cloud_sorting import (
    ColorDetector,
    Color,
    DetectionResult,
    EdgeGraspFinder,
    GraspTarget,
    RobotController,
    RobotConfig,
)


@dataclass
class SortingTarget:
    """Combined target from color detection and edge grasp finding."""
    color: Color
    color_position: tuple  # From color detector (x, y, z)
    grasp_position: Optional[tuple] = None  # From edge grasp finder
    grasp_normal: Optional[tuple] = None
    yaw_angle: float = 0.0


class PointCloudSortingNode(Node):
    """
    Main node for point cloud-based color sorting.

    Processing flow:
    1. Spawn objects (Gazebo)
    2. Move to camera pose
    3. Scan and detect all objects (color + position)
    4. For each target:
       a. Hover above target
       b. Refine grasp point using edge detection
       c. Pick object
       d. Place at color-specific location
    """

    # Work area limits (base_link frame)
    WORK_AREA_X = (0.10, 0.40)
    WORK_AREA_Y = (-0.15, 0.25)
    WORK_AREA_Z = (-0.05, 0.15)

    # Edge detection search radius around color center
    EDGE_SEARCH_RADIUS = 0.05  # 5cm

    # Spawn positions (Gazebo world frame, z=1.10 to drop onto table)
    SPAWN_POSITIONS = [
        (0.22, -0.12, 1.10),
        (0.30, -0.06, 1.10),
        (0.22, 0.12, 1.10),
        (0.30, 0.06, 1.10),
        (0.26, 0.00, 1.10),
    ]

    def __init__(self):
        super().__init__("point_cloud_sorting")

        # Declare parameters (use_sim_time is auto-declared by ROS 2)
        self.declare_parameter("num_objects", 5)
        self.declare_parameter("use_edge_detection", True)

        use_sim_time = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        self._num_objects = self.get_parameter("num_objects").value
        self._use_edge_detection = self.get_parameter("use_edge_detection").value

        self.get_logger().info("=== Point Cloud Sorting Demo ===")
        self.get_logger().info(f"  use_sim_time: {use_sim_time}")
        self.get_logger().info(f"  num_objects: {self._num_objects}")
        self.get_logger().info(f"  use_edge_detection: {self._use_edge_detection}")
        self.get_logger().info("================================")

        # Initialize components
        self._color_detector = ColorDetector(self)
        self._edge_finder = EdgeGraspFinder(self)
        self._robot = RobotController(self, use_sim_time=use_sim_time)

        self._spawn_count = 0
        self.executor = None  # Will be set by main()

    def run(self):
        """Execute the sorting demo."""
        self.get_logger().info("Starting point cloud sorting demo")

        # Wait for camera data to be available
        self.get_logger().info("Waiting for camera data...")
        wait_count = 0
        while not self._color_detector.is_ready() and wait_count < 60:
            time.sleep(0.5)
            wait_count += 1
            if wait_count % 10 == 0:
                self.get_logger().info(f"Still waiting for camera data... ({wait_count}/60)")

        if not self._color_detector.is_ready():
            self.get_logger().error("Camera data not available after 30 seconds, exiting")
            return

        self.get_logger().info("Camera data available!")

        # Wait for robot to be ready
        self.get_logger().info("Waiting for robot state...")
        if not self._robot.wait_for_ready(timeout_sec=30.0):
            self.get_logger().error("Robot not ready, continuing anyway...")

        # Additional wait for MoveIt planning scene
        time.sleep(3)

        # Move to camera pose
        self.get_logger().info("Moving to camera observation pose")
        self._robot.move_to_camera_pose()

        # Spawn objects
        self.get_logger().info("Spawning objects")
        self._spawn_objects()
        time.sleep(3)  # Wait for objects to settle (increased)

        # Open gripper
        self._robot.open_gripper()

        # Main sorting loop
        max_iterations = 10
        total_processed = 0

        for iteration in range(max_iterations):
            self.get_logger().info("")
            self.get_logger().info("=" * 40)
            self.get_logger().info(f"  ITERATION {iteration + 1}")
            self.get_logger().info("=" * 40)

            # Phase 1: Scan and detect targets
            targets = self._scan_work_area()

            if not targets:
                self.get_logger().info("No targets found, demo complete")
                break

            self.get_logger().info(f"Found {len(targets)} targets")

            # Sort by color priority (BLUE -> YELLOW -> GREEN)
            color_priority = {Color.BLUE: 1, Color.YELLOW: 2, Color.GREEN: 3}
            targets.sort(key=lambda t: color_priority.get(t.color, 99))

            # Phase 2: Pick and place each target
            for idx, target in enumerate(targets):
                self.get_logger().info("")
                self.get_logger().info(f"--- Target {idx + 1}/{len(targets)}: {target.color.name} ---")

                success = self._process_target(target)
                if success:
                    total_processed += 1

                # Return to camera pose
                self._robot.move_to_camera_pose()
                time.sleep(0.5)

        # Final summary
        self.get_logger().info("")
        self.get_logger().info("=" * 40)
        self.get_logger().info("  DEMO COMPLETE")
        self.get_logger().info(f"  Total processed: {total_processed}")
        self.get_logger().info("=" * 40)

    def _spawn_objects(self):
        """Spawn colored cubes in Gazebo."""
        colors = [Color.BLUE, Color.YELLOW, Color.GREEN]

        # Random color selection
        spawn_colors = [random.choice(colors) for _ in range(self._num_objects)]

        color_str = ", ".join(c.name for c in spawn_colors)
        self.get_logger().info(f"Spawn colors: [{color_str}]")

        for i, color in enumerate(spawn_colors):
            pos = self.SPAWN_POSITIONS[i % len(self.SPAWN_POSITIONS)]
            self._spawn_cube(pos[0], pos[1], pos[2], color)
            time.sleep(0.2)

    def _spawn_cube(self, x: float, y: float, z: float, color: Color) -> bool:
        """Spawn a single cube in Gazebo."""
        color_rgba = {
            Color.BLUE: "0.0 0.0 1.0 1",
            Color.YELLOW: "1.0 1.0 0.0 1",
            Color.GREEN: "0.0 1.0 0.0 1",
        }
        rgba = color_rgba.get(color, "0.5 0.5 0.5 1")

        name = f"color_cube_{self._spawn_count}"
        self._spawn_count += 1

        self.get_logger().info(f"Spawning {color.name} cube at ({x:.2f}, {y:.2f}, {z:.2f})")

        sdf = f'''<sdf version="1.6"><model name="{name}">
<static>false</static>
<link name="link">
<inertial><mass>0.5</mass>
<inertia><ixx>0.0002</ixx><iyy>0.0002</iyy><izz>0.0002</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
</inertial>
<collision name="collision"><geometry><box><size>0.05 0.05 0.05</size></box></geometry></collision>
<visual name="visual"><geometry><box><size>0.05 0.05 0.05</size></box></geometry>
<material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
</visual></link></model></sdf>'''

        cmd = [
            "ros2", "run", "ros_gz_sim", "create",
            "-world", "default",
            "-name", name,
            "-x", str(x), "-y", str(y), "-z", str(z),
            "-string", sdf,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            self.get_logger().warn(f"Spawn failed: {e}")
            return False

    def _scan_work_area(self) -> List[SortingTarget]:
        """Scan work area and detect all targets."""
        self.get_logger().info("Scanning work area...")

        # Reset detector
        self._color_detector.reset()
        time.sleep(2.0)  # Camera stabilization (increased)

        # Perform detection
        targets = []
        seen_positions = []

        for attempt in range(5):  # Multiple detection attempts
            detections = self._color_detector.detect()
            self.get_logger().info(f"  Scan attempt {attempt + 1}: {len(detections)} raw detections")

            for det in detections:
                if not det.detected:
                    continue

                pos = (det.pose.position.x, det.pose.position.y, det.pose.position.z)
                self.get_logger().debug(
                    f"    Raw: {det.color.name} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )

                # Check work area bounds
                if not self._in_work_area(pos):
                    self.get_logger().debug(f"    -> Outside work area")
                    continue

                # Check for duplicates
                if self._is_duplicate(pos, seen_positions):
                    self.get_logger().debug(f"    -> Duplicate")
                    continue

                seen_positions.append(pos)
                targets.append(SortingTarget(
                    color=det.color,
                    color_position=pos,
                    yaw_angle=det.angle_deg,
                ))
                self.get_logger().info(
                    f"  Detected {det.color.name} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )

            time.sleep(0.5)

        self.get_logger().info(f"  Total unique targets: {len(targets)}")
        return targets

    def _in_work_area(self, pos: tuple) -> bool:
        """Check if position is within work area."""
        x, y, z = pos
        return (
            self.WORK_AREA_X[0] < x < self.WORK_AREA_X[1] and
            self.WORK_AREA_Y[0] < y < self.WORK_AREA_Y[1] and
            self.WORK_AREA_Z[0] < z < self.WORK_AREA_Z[1]
        )

    def _is_duplicate(self, pos: tuple, seen: List[tuple], threshold: float = 0.04) -> bool:
        """Check if position is duplicate of seen positions."""
        for seen_pos in seen:
            dist = sum((a - b) ** 2 for a, b in zip(pos, seen_pos)) ** 0.5
            if dist < threshold:
                return True
        return False

    def _process_target(self, target: SortingTarget) -> bool:
        """Process a single target: refine grasp, pick, and place."""
        x, y, z = target.color_position

        # Step 1: Refine grasp point using edge detection (if enabled)
        if self._use_edge_detection and self._edge_finder.has_cloud():
            self.get_logger().info("  Refining grasp point with edge detection...")

            grasp = self._edge_finder.find_grasp_in_region(
                center=(x, y, z),
                radius=self.EDGE_SEARCH_RADIUS,
                color=target.color,
            )

            if grasp is not None:
                target.grasp_position = tuple(grasp.position)
                target.grasp_normal = tuple(grasp.normal)

                # Log improvement
                dx = grasp.position[0] - x
                dy = grasp.position[1] - y
                dz = grasp.position[2] - z
                offset = (dx**2 + dy**2 + dz**2) ** 0.5
                self.get_logger().info(
                    f"  Edge refinement: offset={offset*1000:.1f}mm, edges={grasp.edge_count}"
                )
            else:
                self.get_logger().info("  Edge detection failed, using color center")

        # Determine final pick position
        if target.grasp_position is not None:
            pick_pos = target.grasp_position
            self.get_logger().info(
                f"  Using EDGE position: ({pick_pos[0]:.3f}, {pick_pos[1]:.3f}, {pick_pos[2]:.3f})"
            )
        else:
            pick_pos = target.color_position
            self.get_logger().info(
                f"  Using COLOR position: ({pick_pos[0]:.3f}, {pick_pos[1]:.3f}, {pick_pos[2]:.3f})"
            )

        # Step 2: Execute pick
        if not self._robot.pick_at(pick_pos, target.yaw_angle):
            self.get_logger().warn("  Pick failed")
            return False

        # Step 3: Execute place
        if not self._robot.place_at_color(target.color):
            self.get_logger().warn("  Place failed")
            return False

        self.get_logger().info(f"  {target.color.name} complete!")
        return True


def main():
    rclpy.init()

    node = PointCloudSortingNode()

    # Use multi-threaded executor for service calls
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Store executor reference in node for service calls
    node.executor = executor

    # Run in separate thread
    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down...")
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print("[point_cloud_sorting] Shutdown complete")


if __name__ == "__main__":
    main()
