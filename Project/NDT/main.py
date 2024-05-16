import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import numpy as np
import open3d as o3d
import pandas as pd


#### Gaussian Distributions ###
def compute_gaussian_distributions(points, voxel_indices): # tick
    voxel_dict = {}
    
    for point, index in zip(points, voxel_indices):
        index_tuple = tuple(index)
        if index_tuple not in voxel_dict:
            voxel_dict[index_tuple] = []
        voxel_dict[index_tuple].append(point)
    
    voxel_mean = {}
    voxel_covariance = {}
    
    for index in voxel_dict:
        voxel_points = np.array(voxel_dict[index])
        
        # Ensure there are enough points to compute a covariance matrix
        if len(voxel_points) < 2:
            continue
        
        q = np.mean(voxel_points, axis=0)
        C = np.cov(voxel_points, rowvar=False)
        
        # Add a small identity matrix to the covariance to avoid singular matrix issues
        C += np.eye(C.shape[0]) * 1e-6
        
        voxel_mean[index] = q
        voxel_covariance[index] = C
    
    return voxel_mean, voxel_covariance

### Rotation Matrix from RPY ###
def rpy_to_rotation_matrix(roll, pitch, yaw): # tick
    # Compute individual rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combine the rotation matrices
    R = R_z @ R_y @ R_x
    return R

### Voxelization ###
def voxelize(points, voxel_size): # tick
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(int)
    return voxel_indices, min_bounds

def compare_transformations(computed, ground_truth): # tick
    # Assuming ground_truth contains x, y, z, roll, pitch, yaw
    translation_gt = ground_truth[[' x', ' y', ' z']].values
    rpy_gt = ground_truth[[' roll', ' pitch', ' yaw']].values
    rotation_gt = rpy_to_rotation_matrix(*rpy_gt)
    
    gt_transform = np.eye(4)
    gt_transform[:3, :3] = rotation_gt
    gt_transform[:3, 3] = translation_gt
    
    print("Ground Truth Transformation Matrix:\n", gt_transform)
    print("Difference in Transformation:\n", computed - gt_transform)
    
### NDT Registration ###
def ndt_registration(source_points, voxel_mean, voxel_covariance, voxel_size, min_bounds, max_iterations=30, tolerance=1e-6): # tick
    # Initialize transformation
    transformation = np.eye(4)
    
    def get_voxel_index(point):
        return tuple(np.floor((point - min_bounds) / voxel_size).astype(int))

    for iteration in range(max_iterations):
        transformed_points = (transformation[:3, :3] @ source_points.T + transformation[:3, 3].reshape(-1, 1)).T
        
        likelihood = 0
        gradient = np.zeros((6,))
        
        for point in transformed_points:
            voxel_index = get_voxel_index(point)
            if voxel_index in voxel_mean:
                q = voxel_mean[voxel_index]
                C = voxel_covariance[voxel_index]
                
                diff = point - q
                likelihood += np.exp(-0.5 * diff.T @ np.linalg.inv(C) @ diff)
                # Compute the gradient based on the likelihood (not fully implemented here)

        # Update the transformation using the gradient (simplified, gradient step needs implementation)
        # transformation_update = compute_transformation_update(gradient)
        # transformation = transformation @ transformation_update
        
        if np.linalg.norm(gradient) < tolerance:
            break
    
    return transformation

### Load Point Cloud ###
def load_point_cloud(filename): # tick
    """ Load a PCD file into an Open3D point cloud """
    return o3d.io.read_point_cloud(filename)

def apply_voxel_downsampling(pcd, voxel_size=0.05):
    """ Downsample the point cloud using voxel downsampling """
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def load_ground_truth(filename): # tick
    """ Load ground truth positions from a CSV file """
    return pd.read_csv(filename)
    
#### Main ####
# Project directory setup
project_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = '/'.join(project_dir.split('/')[:-2])
dataset_dir = os.path.join(project_dir, 'dataset')
frames_dir = sorted(os.listdir(os.path.join(dataset_dir, 'frames')))

# Load map and ground truth
map_pcd = load_point_cloud(os.path.join(dataset_dir, 'map.pcd'))
map_points = np.asarray(map_pcd.points)

# Load first frame
frames_cloud = o3d.io.read_point_cloud(os.path.join(dataset_dir, 'frames', frames_dir[0]))
frame_points = np.asarray(frames_cloud.points)

# Load ground truth
ground_truth = load_ground_truth(os.path.join(dataset_dir, 'ground_truth.csv'))

voxel_size = 1.0  # Define the voxel size
voxel_indices, min_bounds = voxelize(map_points, voxel_size)
voxel_mean, voxel_covariance = compute_gaussian_distributions(map_points, voxel_indices)

# Voxelization
transformation = ndt_registration(frame_points, voxel_mean, voxel_covariance, voxel_size, min_bounds)

# Compare the transformation with ground truth
frame_index = 0  # Example frame index
ground_truth_transform = ground_truth.iloc[frame_index]

print("Computed Transformation:\n", transformation)
print("Ground Truth Transformation for Frame 0:\n", ground_truth_transform)

compare_transformations(transformation, ground_truth_transform)
