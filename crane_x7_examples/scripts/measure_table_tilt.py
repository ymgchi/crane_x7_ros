#!/usr/bin/env python3
"""
Measure table surface tilt in base_link coordinates.
This script samples points from the point cloud, transforms them to base_link,
and analyzes if the table appears tilted after TF transformation.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
import struct
import numpy as np
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs


class TableTiltMeasurer(Node):
    def __init__(self):
        super().__init__("table_tilt_measurer")

        # Use sim_time
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

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
        self.sample_count = 0
        self.max_samples = 3
        self.all_results = []

    def callback(self, msg):
        if self.sample_count >= self.max_samples:
            return

        self.get_logger().info(f"Processing sample {self.sample_count + 1}/{self.max_samples}")
        self.get_logger().info(f"Point cloud frame_id: {msg.header.frame_id}")

        # Sample points in a grid pattern across the image
        # We want to sample the table surface at different X positions (front to back)
        sample_grid = []
        for row_pct in [0.4, 0.5, 0.6, 0.7]:  # Different depths (front to back)
            for col_pct in [0.3, 0.5, 0.7]:   # Left to right
                sample_grid.append((row_pct, col_pct))

        results = []
        for row_pct, col_pct in sample_grid:
            row = int(msg.height * row_pct)
            col = int(msg.width * col_pct)
            idx = row * msg.width + col
            offset = idx * msg.point_step

            if offset + 12 > len(msg.data):
                continue

            x, y, z = struct.unpack_from("fff", bytes(msg.data), offset)

            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            if z < 0.2 or z > 0.9:  # Table depth range
                continue

            # Transform to base_link
            try:
                pt_cam = PointStamped()
                pt_cam.header = msg.header
                pt_cam.point.x = float(x)
                pt_cam.point.y = float(y)
                pt_cam.point.z = float(z)

                pt_base = self.tf_buffer.transform(
                    pt_cam, "base_link",
                    timeout=rclpy.duration.Duration(seconds=1.0))

                results.append({
                    "img_loc": f"({row_pct:.1f},{col_pct:.1f})",
                    "cam": (x, y, z),
                    "base": (pt_base.point.x, pt_base.point.y, pt_base.point.z)
                })
            except Exception as e:
                self.get_logger().warn(f"TF error: {e}")

        if len(results) < 4:
            self.get_logger().warn(f"Not enough valid points ({len(results)}), retrying...")
            return

        self.sample_count += 1
        self.all_results.append(results)

        # Print results for this sample
        self.print_results(results, self.sample_count)

        if self.sample_count >= self.max_samples:
            self.analyze_all_samples()
            rclpy.shutdown()

    def print_results(self, results, sample_num):
        print(f"\n=== Sample {sample_num}: Table Surface Analysis ===")
        print(f"Points transformed from camera frame to base_link:\n")
        header = f"{'Img Loc':<12} {'Cam Z':<10} {'Base X':<10} {'Base Y':<10} {'Base Z':<10}"
        print(header)
        print("-" * 55)

        for r in results:
            cam_z = r["cam"][2]
            base_x, base_y, base_z = r["base"]
            line = f"{r['img_loc']:<12} {cam_z:<10.3f} {base_x:<10.3f} {base_y:<10.3f} {base_z:<10.3f}"
            print(line)

        # Analyze Z variation by X position (front to back)
        print("\n--- Z variation by X position (table tilt check) ---")
        by_x = {}
        for r in results:
            base_x = round(r["base"][0], 2)
            base_z = r["base"][2]
            if base_x not in by_x:
                by_x[base_x] = []
            by_x[base_x].append(base_z)

        for x in sorted(by_x.keys()):
            z_vals = by_x[x]
            z_mean = np.mean(z_vals)
            z_std = np.std(z_vals) if len(z_vals) > 1 else 0
            print(f"X={x:+.2f}m: Z_mean={z_mean:.4f}m (std={z_std:.4f}m, n={len(z_vals)})")

    def analyze_all_samples(self):
        print("\n" + "=" * 60)
        print("=== FINAL ANALYSIS: Table Tilt in base_link ===")
        print("=" * 60)

        # Aggregate all points
        all_points = []
        for results in self.all_results:
            for r in results:
                all_points.append(r["base"])

        all_points = np.array(all_points)

        # Fit plane to points: z = ax + by + c
        # Using least squares
        X = all_points[:, 0]  # base X
        Y = all_points[:, 1]  # base Y
        Z = all_points[:, 2]  # base Z

        # Design matrix for z = ax + by + c
        A = np.column_stack([X, Y, np.ones_like(X)])
        coeffs, residuals, rank, s = np.linalg.lstsq(A, Z, rcond=None)
        a, b, c = coeffs

        print(f"\nFitted plane: Z = {a:.6f}*X + {b:.6f}*Y + {c:.6f}")
        print(f"\nInterpretation:")
        print(f"  - dZ/dX = {a:.6f} m/m  (tilt in X direction)")
        print(f"    If positive: table rises toward robot (+X)")
        print(f"    If negative: table drops toward robot (+X)")
        print(f"  - dZ/dY = {b:.6f} m/m  (tilt in Y direction)")
        print(f"    If positive: table rises to the left (+Y)")
        print(f"    If negative: table rises to the right (-Y)")

        # Calculate tilt angles
        tilt_x_deg = np.degrees(np.arctan(a))
        tilt_y_deg = np.degrees(np.arctan(b))
        print(f"\n  - X-axis tilt angle: {tilt_x_deg:.2f}°")
        print(f"  - Y-axis tilt angle: {tilt_y_deg:.2f}°")

        # Z range
        z_range = Z.max() - Z.min()
        z_mean = Z.mean()
        z_std = Z.std()
        print(f"\n  - Z range: {z_range*1000:.1f}mm")
        print(f"  - Z mean: {z_mean:.4f}m")
        print(f"  - Z std: {z_std*1000:.1f}mm")

        # Verdict
        print("\n" + "-" * 40)
        if abs(a) > 0.02 or abs(b) > 0.02:
            print("VERDICT: Table appears TILTED in base_link frame!")
            print("         TF transformation may have issues.")
        else:
            print("VERDICT: Table appears FLAT in base_link frame.")
            print("         TF transformation is working correctly.")
        print("-" * 40)


def main():
    rclpy.init()
    node = TableTiltMeasurer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
