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
Color-filtered grasp candidate demo.

Based on "Robotic grasp detection toward unknown objects using 3D edge detection
and Gaussian mixture model for clustering candidates", with an added color
filtering step to suppress box edges.

Requirements: Open3D, NumPy, scikit-learn, Matplotlib.
"""

import copy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.mixture import GaussianMixture


@dataclass
class PipelineConfig:
    neighbor_radius: float = 0.03  # radius for normal/curvature estimation (m)
    max_neighbors: int = 40  # maximum neighbors for KDTree search
    curvature_percentile: float = 92.0  # percentile used to mark edge points
    color_filter_radius: float = 0.18  # distance in RGB space for exclusion
    box_color: tuple = (0.6, 0.4, 0.2)  # color to remove in mode B
    gmm_components: int = 3  # number of Gaussian components
    noise_std: float = 0.002  # positional noise added to synthetic data


def generate_scene(
    seed: int = 0,
    floor_size: float = 0.5,
    box_dims=(0.25, 0.25, 0.18),
    target_color=(0.0, 1.0, 0.0),
    config: PipelineConfig = PipelineConfig(),
) -> o3d.geometry.PointCloud:
    """Create a synthetic point cloud with a floor, a box, and a target object."""

    rng = np.random.default_rng(seed)

    def sample_floor(num_points: int = 2500):
        xs = rng.uniform(-floor_size / 2, floor_size / 2, num_points)
        ys = rng.uniform(-floor_size / 2, floor_size / 2, num_points)
        zs = rng.normal(0.0, 0.0005, num_points)
        pts = np.stack([xs, ys, zs], axis=1)
        colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (num_points, 1))
        return pts, colors

    def sample_box(num_points_per_face: int = 1200):
        w, d, h = box_dims
        color = np.array(config.box_color)
        pts, colors = [], []

        def wall(x=None, y=None):
            if x is not None:
                xs = np.full(num_points_per_face, x)
                ys = rng.uniform(-d / 2, d / 2, num_points_per_face)
            else:
                xs = rng.uniform(-w / 2, w / 2, num_points_per_face)
                ys = np.full(num_points_per_face, y)
            zs = rng.uniform(0.0, h, num_points_per_face)
            return np.stack([xs, ys, zs], axis=1)

        pts.append(wall(x=-w / 2))
        pts.append(wall(x=w / 2))
        pts.append(wall(y=-d / 2))
        pts.append(wall(y=d / 2))

        # Thin bottom plate for the box.
        xs = rng.uniform(-w / 2, w / 2, num_points_per_face // 2)
        ys = rng.uniform(-d / 2, d / 2, num_points_per_face // 2)
        zs = np.zeros_like(xs)
        pts.append(np.stack([xs, ys, zs], axis=1))

        pts = np.concatenate(pts, axis=0)
        colors = np.tile(color, (pts.shape[0], 1))
        return pts, colors

    def sample_target(num_points: int = 2000):
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.12)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(target_color)
        mesh.translate([0.0, 0.0, 0.06])  # center inside the box
        pcd = mesh.sample_points_poisson_disk(num_points)
        return np.asarray(pcd.points), np.asarray(pcd.colors)

    floor_pts, floor_colors = sample_floor()
    box_pts, box_colors = sample_box()
    target_pts, target_colors = sample_target()

    pts = np.vstack([floor_pts, box_pts, target_pts])
    colors = np.vstack([floor_colors, box_colors, target_colors])

    pts += rng.normal(scale=config.noise_std, size=pts.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


class PointCloudPreprocessor:
    def __init__(self, exclusion_color: tuple, radius: float):
        self.exclusion_color = np.array(exclusion_color)
        self.radius = radius

    def apply(self, pcd: o3d.geometry.PointCloud, enable_color_filter: bool) -> o3d.geometry.PointCloud:
        """Either pass the cloud through (mode A) or filter out the exclusion color (mode B)."""
        if not enable_color_filter:
            return copy.deepcopy(pcd)

        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        distances = np.linalg.norm(colors - self.exclusion_color, axis=1)
        keep_mask = distances > self.radius

        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(pts[keep_mask])
        filtered.colors = o3d.utility.Vector3dVector(colors[keep_mask])
        return filtered


def estimate_curvature(
    pcd: o3d.geometry.PointCloud, radius: float, max_nn: int
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normals and curvature using PCA on local neighborhoods."""
    # Normals for grasp orientation; Open3D handles the KDTree internally.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = np.zeros(len(pts))

    for idx, pt in enumerate(pts):
        _, neighbor_ids, _ = kdtree.search_radius_vector_3d(pt, radius)
        if len(neighbor_ids) < 5:
            continue
        neighbors = pts[neighbor_ids]
        cov = np.cov(neighbors.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        curvatures[idx] = eigvals.min() / eigvals.sum()

    return curvatures, normals


def extract_edges(pcd: o3d.geometry.PointCloud, curvatures: np.ndarray, percentile: float):
    """Pick high-curvature points as edges and color them red for visualization."""
    threshold = np.percentile(curvatures, percentile)
    edge_mask = curvatures >= threshold
    edge_indices = np.where(edge_mask)[0]
    edges = pcd.select_by_index(edge_indices)
    edges.paint_uniform_color([1.0, 0.0, 0.0])
    return edges, edge_mask


def cluster_edges(
    edge_points: np.ndarray, edge_normals: np.ndarray, config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run GMM on edge points and return the densest cluster center and normals."""
    if len(edge_points) < config.gmm_components:
        raise ValueError("Not enough edge points to cluster.")

    gmm = GaussianMixture(
        n_components=config.gmm_components,
        covariance_type="full",
        random_state=0,
    )
    labels = gmm.fit_predict(edge_points)
    counts = np.bincount(labels, minlength=config.gmm_components)
    target_label = counts.argmax()

    center = gmm.means_[target_label]
    grasp_normals = edge_normals[labels == target_label]
    grasp_normal = grasp_normals.mean(axis=0)
    norm = np.linalg.norm(grasp_normal)
    if norm > 1e-6:
        grasp_normal /= norm
    else:
        grasp_normal = np.array([0.0, 0.0, 1.0])

    return center, grasp_normal, labels


def visualize_results(before: dict, after: dict):
    """Visualize before/after color filtering results side-by-side."""
    fig = plt.figure(figsize=(12, 6))
    scenarios = [("Before: no color filter", before), ("After: filter box color", after)]

    for idx, (title, data) in enumerate(scenarios, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        pts = np.asarray(data["pcd"].points)
        cols = np.asarray(data["pcd"].colors)
        edges = np.asarray(data["edges"].points)

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=4, alpha=0.9)
        ax.scatter(edges[:, 0], edges[:, 1], edges[:, 2], c="red", s=10, alpha=0.9)

        center = data["grasp_center"]
        normal = data["grasp_normal"]
        ax.scatter(center[0], center[1], center[2], c="k", s=80, marker="x")
        ax.quiver(
            center[0],
            center[1],
            center[2],
            normal[0] * 0.06,
            normal[1] * 0.06,
            normal[2] * 0.06,
            color="k",
            linewidth=2,
        )

        ax.set_title(title)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.view_init(elev=20, azim=-60)
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)
        ax.set_zlim(-0.02, 0.25)

    plt.tight_layout()
    plt.show()


class ColorFilteredGraspDemo:
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.preproc = PointCloudPreprocessor(config.box_color, config.color_filter_radius)

    def _process(self, pcd: o3d.geometry.PointCloud, enable_color_filter: bool):
        filtered = self.preproc.apply(pcd, enable_color_filter)
        curvatures, normals = estimate_curvature(
            filtered, radius=self.config.neighbor_radius, max_nn=self.config.max_neighbors
        )
        edges, edge_mask = extract_edges(filtered, curvatures, percentile=self.config.curvature_percentile)
        edge_points = np.asarray(edges.points)
        edge_normals = normals[edge_mask]

        center, grasp_normal, labels = cluster_edges(edge_points, edge_normals, self.config)

        return {
            "pcd": filtered,
            "edges": edges,
            "grasp_center": center,
            "grasp_normal": grasp_normal,
            "cluster_labels": labels,
        }

    def run(self):
        base_scene = generate_scene(config=self.config)
        before = self._process(base_scene, enable_color_filter=False)
        after = self._process(base_scene, enable_color_filter=True)
        visualize_results(before, after)


if __name__ == "__main__":
    demo = ColorFilteredGraspDemo()
    demo.run()
