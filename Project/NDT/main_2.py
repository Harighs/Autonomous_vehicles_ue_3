import open3d as o3d
import numpy as np
import pandas as pd
import os
import time
#
# #### Gaussian Distributions ###
# def compute_gaussian_distributions(points, voxel_indices):
#     voxel_dict = {}
#
#     for point, index in zip(points, voxel_indices):
#         index_tuple = tuple(index)
#         if index_tuple not in voxel_dict:
#             voxel_dict[index_tuple] = []
#         voxel_dict[index_tuple].append(point)
#
#     voxel_mean = {}
#     voxel_covariance = {}
#
#     for index in voxel_dict:
#         voxel_points = np.array(voxel_dict[index])
#
#         if len(voxel_points) < 2:
#             continue
#
#         q = np.mean(voxel_points, axis=0)
#         C = np.cov(voxel_points, rowvar=False)
#         C += np.eye(C.shape[0]) * 1e-6
#
#         voxel_mean[index] = q
#         voxel_covariance[index] = C
#
#     return voxel_mean, voxel_covariance
#
# ### Rotation Matrix from RPY ###
# def rpy_to_rotation_matrix(roll, pitch, yaw):
#     # Calculate cosines and sines
#     cy = np.cos(yaw)
#     sy = np.sin(yaw)
#     cp = np.cos(pitch)
#     sp = np.sin(pitch)
#     cr = np.cos(roll)
#     sr = np.sin(roll)
#
#     # Create rotation matrices
#     R_z = np.array([
#         [cy, -sy, 0],
#         [sy, cy, 0],
#         [0, 0, 1]
#     ])
#
#     R_y = np.array([
#         [cp, 0, sp],
#         [0, 1, 0],
#         [-sp, 0, cp]
#     ])
#
#     R_x = np.array([
#         [1, 0, 0],
#         [0, cr, -sr],
#         [0, sr, cr]
#     ])
#
#     # Combine rotations
#     R = np.dot(R_z, np.dot(R_y, R_x))
#     return R
#
# ### Voxelization ###
# def voxelize(points, voxel_size):
#     min_bounds = np.min(points, axis=0)
#     voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(int)
#     return voxel_indices, min_bounds
#
# def compare_transformations(computed, ground_truth):
#     translation_gt = ground_truth[[' x', ' y', ' z']].values
#     rpy_gt = ground_truth[[' roll', ' pitch', ' yaw']].values
#     rotation_gt = rpy_to_rotation_matrix(*rpy_gt)
#
#     gt_transform = np.eye(4)
#     gt_transform[:3, :3] = rotation_gt
#     gt_transform[:3, 3] = translation_gt
#
#     # print("Ground Truth Transformation Matrix:\n", gt_transform)
#     # print("Difference in Transformation:\n", computed - gt_transform)
#
# ### NDT Registration ###
# def ndt_registration(source_points, voxel_mean, voxel_covariance, voxel_size, min_bounds, max_iterations=30, tolerance=1e-6):
#     transformation = np.eye(4)
#
#     def get_voxel_index(point):
#         return tuple(np.floor((point - min_bounds) / voxel_size).astype(int))
#
#     for iteration in range(max_iterations):
#         transformed_points = (transformation[:3, :3] @ source_points.T + transformation[:3, 3].reshape(-1, 1)).T
#
#         likelihood = 0
#         gradient = np.zeros((6,))
#
#         for point in transformed_points:
#             voxel_index = get_voxel_index(point)
#             if voxel_index in voxel_mean:
#                 q = voxel_mean[voxel_index]
#                 C = voxel_covariance[voxel_index]
#
#                 diff = point - q
#                 likelihood += np.exp(-0.5 * diff.T @ np.linalg.inv(C) @ diff)
#                 # Compute the gradient based on the likelihood (not fully implemented here)
#
#         # Update the transformation using the gradient (simplified, gradient step needs implementation)
#         # transformation_update = compute_transformation_update(gradient)
#         # transformation = transformation @ transformation_update
#
#         if np.linalg.norm(gradient) < tolerance:
#             break
#
#     return transformation
#
# ### Load Point Cloud ###
# def load_point_cloud(filename):
#     return o3d.io.read_point_cloud(filename)
#
# def apply_voxel_downsampling(pcd, voxel_size=0.05):
#     return pcd.voxel_down_sample(voxel_size=voxel_size)
#
# def load_ground_truth(filename):
#     return pd.read_csv(filename)
#
# #### Main ####
# # Project directory setup
# project_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = '/'.join(project_dir.split('/')[:-2])
# dataset_dir = os.path.join(project_dir, 'dataset')
# frames_dir = sorted(os.listdir(os.path.join(dataset_dir, 'frames')))
#
# # Load map and ground truth
# map_pcd = load_point_cloud(os.path.join(dataset_dir, 'map.pcd'))
# map_points = np.asarray(map_pcd.points)
#
# # Load ground truth
# ground_truth = load_ground_truth(os.path.join(dataset_dir, 'ground_truth.csv'))
#
# voxel_size = 1.0
# voxel_indices, min_bounds = voxelize(map_points, voxel_size)
# voxel_mean, voxel_covariance = compute_gaussian_distributions(map_points, voxel_indices)
#
# # Placeholder for initial transformation (identity)
# initial_transformation = np.eye(4)
#
# merged_pcd = o3d.geometry.PointCloud()
# # Loop through all frames
# for i, frame_file in enumerate(frames_dir):
#     frame_pcd = load_point_cloud(os.path.join(dataset_dir, 'frames', frame_file))
#     frame_points = np.asarray(frame_pcd.points)
#
#     start_time = time.time()
#     transformation = ndt_registration(frame_points, voxel_mean, voxel_covariance, voxel_size, min_bounds)
#     end_time = time.time()
#
#     aligned_filename = os.path.join(dataset_dir, 'aligned_frames', f'aligned_{frame_file}.pcd')
#     frame_pcd.transform(transformation)
#     merged_pcd += frame_pcd
#
#
#     # Calculate lateral error (example calculation)
#     gt_transform = ground_truth.iloc[i]
#     translation_gt = gt_transform[[' x', ' y', ' z']].values
#     rpy_gt = gt_transform[[' roll', ' pitch', ' yaw']].values
#     rotation_gt = rpy_to_rotation_matrix(*rpy_gt)
#
#     gt_transform_matrix = np.eye(4)
#     gt_transform_matrix[:3, :3] = rotation_gt
#     gt_transform_matrix[:3, 3] = translation_gt
#
#     # Compute the lateral error (difference in transformation)
#     difference = np.linalg.norm(transformation[:3, 3] - gt_transform_matrix[:3, 3])
#     print(f"Frame {i}: Lateral error = {difference:.2f} meters, Computation time = {end_time - start_time:.2f} seconds")
#
#     # Update initial transformation for next iteration (could be based on odometry or other)
#     initial_transformation = transformation
#
#     # Optional: Compare transformations
#     compare_transformations(transformation, gt_transform)
#
# # Optional: Voxel downsample the merged point cloud to reduce noise and size
# merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.05)
#
# # Save the merged point cloud
# merged_filename = os.path.join(dataset_dir, 'merged_map.pcd')
# o3d.io.write_point_cloud(merged_filename, merged_pcd)


pcd = o3d.io.read_point_cloud("/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project3/Autonomous_vehicles_ue_3/localization/Project/ICP/aligned_car.pcd")
o3d.visualization.draw_geometries([pcd])
