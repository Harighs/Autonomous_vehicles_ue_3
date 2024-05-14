import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import numpy as np
import open3d as o3d

class NDTRegistration:
    def __init__(self, resolution):
        self.resolution = resolution  # Voxel grid size

    def voxel_index(self, point):
        """Compute voxel index for a point."""
        return np.floor(point / self.resolution).astype(np.int32)

    def create_voxel_grid(self, point_cloud):
        """Create a voxel grid from a point cloud."""
        indices = np.apply_along_axis(self.voxel_index, 1, np.asarray(point_cloud.points))
        voxel_dict = {}
        for index, p in zip(indices, np.asarray(point_cloud.points)):
            index_tuple = tuple(index)
            if index_tuple not in voxel_dict:
                voxel_dict[index_tuple] = []
            voxel_dict[index_tuple].append(p)
        return voxel_dict

    def compute_voxel_stats(self, voxel_grid):
        """Compute mean and covariance for each voxel."""
        stats = {}
        for voxel, points in voxel_grid.items():
            if len(points) > 0:
                points = np.array(points)
                mean = np.mean(points, axis=0)
                covariance = np.cov(points, rowvar=False)
                stats[voxel] = (mean, covariance)
        return stats

    def register(self, source, target):
        """Perform NDT registration."""
        source_voxel = self.create_voxel_grid(source)
        target_voxel = self.create_voxel_grid(target)

        source_stats = self.compute_voxel_stats(source_voxel)
        target_stats = self.compute_voxel_stats(target_voxel)

        # Initialize transformation to identity
        transformation = np.eye(4)

        # Simplified registration logic for demonstration
        for _ in range(10):  # Fixed number of iterations
            transformed_points = np.dot(np.asarray(source.points), transformation[:3, :3].T) + transformation[:3, 3]
            transformed_source = o3d.geometry.PointCloud()
            transformed_source.points = o3d.utility.Vector3dVector(transformed_points)
            # Update transformation based on some criteria - not implemented here

        return transformation

def load_point_cloud(filename):
    """ Load a PCD file into an Open3D point cloud """
    return o3d.io.read_point_cloud(filename)

def apply_voxel_downsampling(pcd, voxel_size=0.05):
    """ Downsample the point cloud using voxel downsampling """
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def load_ground_truth(filename):
    """ Load ground truth positions from a CSV file """
    return pd.read_csv(filename)

def apply_ndt_registration(source, target, init_pose=np.eye(4), voxel_size=0.05, max_iter=50):
    """ Apply NDT registration between source and target point clouds """
    # Downsample the point cloud
    source_down = apply_voxel_downsampling(source, voxel_size)
    target_down = apply_voxel_downsampling(target, voxel_size)

    # Estimate normals for better registration results
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Perform NDT registration

    return result

# Project directory setup
# current file as project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(project_dir, 'dataset')
frames_dir = sorted(os.listdir(os.path.join(dataset_dir, 'frames')))
# Load map and ground truth
map_pcd = load_point_cloud(os.path.join(dataset_dir, 'map.pcd'))
frames_cloud = o3d.io.read_point_cloud(os.path.join(dataset_dir, 'frames', frames_dir[0]))
ground_truth = load_ground_truth(os.path.join(dataset_dir, 'ground_truth.csv'))

# Processing parameters
voxel_size = 0.1
max_iter = 50

# Perform NDT registration
result = apply_ndt_registration(frames_cloud, map_pcd, voxel_size=voxel_size, max_iter=max_iter)

# Usage
# source_pcd = o3d.io.read_point_cloud("source.pcd")
# target_pcd = o3d.io.read_point_cloud("target.pcd")
ndt = NDTRegistration(resolution=1.0)
transformation = ndt.register(source_pcd, target_pcd)
print("Estimated Transformation Matrix:")
print(transformation)