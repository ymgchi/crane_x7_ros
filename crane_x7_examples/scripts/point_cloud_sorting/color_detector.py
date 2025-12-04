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
HSV-based color detection from RGB images.
Ported from color_sorting.cpp ColorDetector class.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
import numpy as np
import cv2

from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, PointStamped
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs  # noqa: F401 - registers transform types


class Color(Enum):
    """Detected object colors."""
    NONE = auto()
    BLUE = auto()
    YELLOW = auto()
    GREEN = auto()


@dataclass
class HSVRange:
    """HSV color range for detection."""
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int


@dataclass
class DetectionResult:
    """Single object detection result."""
    color: Color = Color.NONE
    pose: Pose = field(default_factory=Pose)
    detected: bool = False
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    image_width: int = 0
    image_height: int = 0
    angle_deg: float = 0.0  # Object rotation angle from minAreaRect
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


# Default HSV ranges (tuned for Gazebo simulation)
DEFAULT_HSV_RANGES = {
    Color.BLUE: HSVRange(100, 125, 100, 255, 30, 255),
    Color.YELLOW: HSVRange(20, 35, 100, 255, 60, 255),
    Color.GREEN: HSVRange(40, 80, 100, 255, 30, 255),
}


class ColorDetector:
    """
    HSV-based color detector for RGB images.

    Detects blue, yellow, and green objects and computes their 3D positions
    using depth image and camera intrinsics.
    """

    # Detection parameters
    MIN_CONTOUR_AREA = 1000  # Minimum contour area in pixels
    DEPTH_OFFSET = 0.015  # Depth calibration offset (meters)
    DEPTH_MIN = 0.15  # Minimum valid depth (meters)
    DEPTH_MAX = 1.2  # Maximum valid depth (meters)
    DUPLICATE_THRESHOLD = 0.04  # 4cm - same object threshold

    def __init__(
        self,
        node: Node,
        hsv_ranges: Optional[dict] = None,
        target_frame: str = "base_link",
    ):
        """
        Initialize ColorDetector.

        Args:
            node: ROS 2 node for subscriptions and TF
            hsv_ranges: Custom HSV ranges dict {Color: HSVRange}
            target_frame: Target coordinate frame for 3D positions
        """
        self._node = node
        self._logger = node.get_logger()
        self._hsv_ranges = hsv_ranges or DEFAULT_HSV_RANGES.copy()
        self._target_frame = target_frame

        self._bridge = CvBridge()
        self._camera_model = PinholeCameraModel()

        # TF setup
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

        # Data storage
        self._latest_image: Optional[Image] = None
        self._latest_depth: Optional[Image] = None
        self._camera_info: Optional[CameraInfo] = None
        self._latest_detections: List[DetectionResult] = []

        # Subscriptions
        self._image_sub = node.create_subscription(
            Image, "/camera/color/image_raw", self._image_callback, 10
        )
        self._depth_sub = node.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw", self._depth_callback, 10
        )
        self._info_sub = node.create_subscription(
            CameraInfo, "/camera/color/camera_info", self._info_callback, 10
        )

        self._logger.info("ColorDetector initialized")

    def _image_callback(self, msg: Image):
        """Store latest RGB image."""
        if self._latest_image is None:
            self._logger.info(f"First RGB image received: {msg.width}x{msg.height}, encoding={msg.encoding}")
        self._latest_image = msg

    def _depth_callback(self, msg: Image):
        """Store latest depth image."""
        if self._latest_depth is None:
            self._logger.info(f"First depth image received: {msg.width}x{msg.height}")
        self._latest_depth = msg

    def _info_callback(self, msg: CameraInfo):
        """Store camera info and update model."""
        if self._camera_info is None:
            self._logger.info(f"First camera info received: {msg.width}x{msg.height}")
        self._camera_info = msg
        self._camera_model.fromCameraInfo(msg)

    def is_ready(self) -> bool:
        """Check if detector has received all required data."""
        return (
            self._latest_image is not None and
            self._latest_depth is not None and
            self._camera_info is not None
        )

    def detect(self) -> List[DetectionResult]:
        """
        Perform detection on current images.

        Returns:
            List of DetectionResult for all detected objects.
        """
        if self._latest_image is None:
            self._logger.debug("No RGB image received yet")
            return []
        if self._latest_depth is None:
            self._logger.debug("No depth image received yet")
            return []
        if self._camera_info is None:
            self._logger.debug("No camera info received yet")
            return []

        self._logger.debug(f"Detecting: image={self._latest_image.width}x{self._latest_image.height}")

        # Convert to OpenCV format
        try:
            if self._latest_image.encoding == "bgr8":
                cv_image = self._bridge.imgmsg_to_cv2(self._latest_image, "bgr8")
            else:
                cv_image = self._bridge.imgmsg_to_cv2(self._latest_image, "rgb8")
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            cv_depth = self._bridge.imgmsg_to_cv2(
                self._latest_depth, desired_encoding="passthrough"
            )
        except Exception as e:
            self._logger.error(f"Image conversion failed: {e}")
            return []

        # Convert to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        detections = []
        for color in [Color.BLUE, Color.YELLOW, Color.GREEN]:
            color_detections = self._detect_color(
                hsv_image, cv_depth, color, self._latest_image.header
            )
            detections.extend(color_detections)

        # Remove duplicates
        self._latest_detections = self._remove_duplicates(detections)
        return self._latest_detections

    def _detect_color(
        self,
        hsv_image: np.ndarray,
        depth_image: np.ndarray,
        color: Color,
        header,
    ) -> List[DetectionResult]:
        """Detect objects of a specific color."""
        hsv_range = self._hsv_ranges[color]

        # Threshold
        mask = cv2.inRange(
            hsv_image,
            (hsv_range.h_min, hsv_range.s_min, hsv_range.v_min),
            (hsv_range.h_max, hsv_range.s_max, hsv_range.v_max),
        )

        # Log mask pixel count
        mask_pixels = np.sum(mask > 0)
        self._logger.info(f"  {color.name}: {mask_pixels} mask pixels")

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Erode to separate adjacent objects
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel_small)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            self._logger.info(f"  {color.name}: {len(contours)} contours found")

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_CONTOUR_AREA:
                self._logger.info(f"    Contour area {area:.0f} < {self.MIN_CONTOUR_AREA} (skipped)")
                continue

            self._logger.info(f"    Processing contour with area {area:.0f}")
            result = self._process_contour(contour, depth_image, color, header)
            if result is not None:
                results.append(result)
            else:
                self._logger.info(f"    -> process_contour returned None")

        return results

    def _process_contour(
        self,
        contour: np.ndarray,
        depth_image: np.ndarray,
        color: Color,
        header,
    ) -> Optional[DetectionResult]:
        """Process a single contour and compute 3D position."""
        # Compute centroid
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            self._logger.debug("    -> moments m00 <= 0")
            return None

        pixel_x = moments["m10"] / moments["m00"]
        pixel_y = moments["m01"] / moments["m00"]

        # Get rotation angle from minAreaRect
        rotated_rect = cv2.minAreaRect(contour)
        angle = rotated_rect[2]
        # Normalize angle to -45 to 45 degrees
        if angle > 45.0:
            angle -= 90.0
        elif angle < -45.0:
            angle += 90.0

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Get depth at centroid (5x5 median filter)
        depth = self._get_median_depth(depth_image, int(pixel_x), int(pixel_y))
        if depth is None:
            self._logger.info(f"    -> No valid depth at pixel ({pixel_x:.0f}, {pixel_y:.0f})")
            return None

        depth += self.DEPTH_OFFSET
        if depth < self.DEPTH_MIN or depth > self.DEPTH_MAX:
            self._logger.info(f"    -> Depth {depth:.3f}m out of range [{self.DEPTH_MIN}, {self.DEPTH_MAX}]")
            return None

        self._logger.info(f"    -> Depth at ({pixel_x:.0f}, {pixel_y:.0f}): {depth:.3f}m")

        # Project to 3D
        point_2d = (pixel_x, pixel_y)
        rect_point = self._camera_model.rectifyPoint(point_2d)
        ray = self._camera_model.projectPixelTo3dRay(rect_point)

        camera_point = (
            ray[0] * depth,
            ray[1] * depth,
            ray[2] * depth,
        )

        self._logger.info(f"    -> Camera point: ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f})")

        # Transform to target frame
        pose = self._transform_to_base(camera_point, header)
        if pose is None:
            self._logger.info(f"    -> TF transform failed (frame: {header.frame_id})")
            return None

        self._logger.info(f"    -> Base point: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})")

        return DetectionResult(
            color=color,
            pose=pose,
            detected=True,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            image_width=self._camera_info.width,
            image_height=self._camera_info.height,
            angle_deg=angle,
            bounding_box=(x, y, w, h),
        )

    def _get_median_depth(
        self, depth_image: np.ndarray, cx: int, cy: int, window: int = 5
    ) -> Optional[float]:
        """Get median depth in a window around the center point."""
        h, w = depth_image.shape[:2]
        half = window // 2

        values = []
        raw_values = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                x = max(0, min(w - 1, cx + dx))
                y = max(0, min(h - 1, cy + dy))

                if depth_image.dtype == np.uint16:
                    v = depth_image[y, x] / 1000.0  # mm to m
                else:
                    v = float(depth_image[y, x])

                raw_values.append(v)
                if v > 0 and np.isfinite(v):
                    values.append(v)

        if not values:
            # Debug: show raw depth values
            self._logger.debug(f"      Depth at ({cx}, {cy}): all invalid. Raw values (sample): {raw_values[:9]}")
            return None

        return float(np.median(values))

    def _transform_to_base(
        self, camera_point: Tuple[float, float, float], header
    ) -> Optional[Pose]:
        """Transform point from camera frame to target frame."""
        # Try different source frame names
        source_frames = [
            header.frame_id,
            "camera_color_optical_frame",
            "camera_depth_optical_frame",
        ]

        # Handle prefixed frames
        if "/" in header.frame_id:
            prefix = header.frame_id.split("/")[0]
            source_frames.extend([
                f"{prefix}/camera_color_optical_frame",
                f"{prefix}/camera_depth_optical_frame",
            ])

        point_stamped = PointStamped()
        point_stamped.header = header
        point_stamped.point.x = camera_point[0]
        point_stamped.point.y = camera_point[1]
        point_stamped.point.z = camera_point[2]

        for source_frame in source_frames:
            try:
                point_stamped.header.frame_id = source_frame
                transformed = self._tf_buffer.transform(
                    point_stamped, self._target_frame, timeout=Duration(seconds=0.1)
                )

                pose = Pose()
                pose.position.x = transformed.point.x
                pose.position.y = transformed.point.y
                pose.position.z = transformed.point.z
                pose.orientation.w = 1.0
                return pose

            except TransformException:
                continue

        return None

    def _remove_duplicates(
        self, detections: List[DetectionResult]
    ) -> List[DetectionResult]:
        """Remove duplicate detections within threshold distance."""
        unique = []
        for det in detections:
            is_duplicate = False
            for existing in unique:
                dx = det.pose.position.x - existing.pose.position.x
                dy = det.pose.position.y - existing.pose.position.y
                dz = det.pose.position.z - existing.pose.position.z
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                if dist < self.DUPLICATE_THRESHOLD and det.color == existing.color:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(det)
        return unique

    def get_latest_detections(self) -> List[DetectionResult]:
        """Get the most recent detection results."""
        return self._latest_detections

    def reset(self):
        """Clear stored detections."""
        self._latest_detections = []
