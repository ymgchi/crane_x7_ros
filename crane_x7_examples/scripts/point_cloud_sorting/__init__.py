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
Point cloud-based sorting module.

This module provides:
- ColorDetector: HSV-based color detection from RGB images
- EdgeGraspFinder: Edge-based grasp point detection from point clouds
- RobotController: MoveIt-based robot control via pymoveit2
"""

from .color_detector import ColorDetector, Color, DetectionResult
from .edge_grasp_finder import EdgeGraspFinder, GraspTarget
from .robot_controller import RobotController, RobotConfig

__all__ = [
    "ColorDetector",
    "Color",
    "DetectionResult",
    "EdgeGraspFinder",
    "GraspTarget",
    "RobotController",
    "RobotConfig",
]
