import open3d as o3d
import pandas as pd
import numpy as np
import os

class NDTRegistration:
    def __init__(self, resolution):
        self.resolution = resolution  # Voxel grid size

    def voxel_index(self, point):
        return np.floor(point / self.resolution).astype(np.int32)

    def create_voxel_grid(self, point_cloud):
        indices = np.apply_along_axis(self.voxel_index, 1, np.asarray(point_cloud.points))
        voxel_dict = {}
        for index, p in zip(indices, np.asarray(point_cloud.points)):
            index_tuple = tuple(index)
            if index_tuple not in voxel_dict:
                voxel_dict[index_tuple] = []
            voxel_dict[index_tuple].append(p)
        return voxel_dict

    def compute_voxel_stats(self, voxel_grid):
        stats = {}
        for voxel, points in voxel_grid.items():
            if len(points) > 0:
                points = np.array(points)
                mean = np.mean(points, axis=0)
                covariance = np.cov(points, rowvar=False)
                stats[voxel] = (mean, covariance)
        return stats

    def register(self, source, target):
        source_voxel = self.create_voxel_grid(source)
        target_voxel = self.create_voxel_grid(target)

        source_stats = self.compute_voxel_stats(source_voxel)
        target_stats = self.compute_voxel_stats(target_voxel)
        transformation = np.eye(4)
        return transformation

def load_ground_truth(file_path):
    return pd.read_csv(file_path, delimiter=',')

def load_pcd(frame_number, dataset_folder):
    file_path = os.path.join(dataset_folder, f"frames/frame_{frame_number}.pcd")
    if os.path.exists(file_path):
        return o3d.io.read_point_cloud(file_path)
    else:
        return None

def transform_point_cloud(pcd, x, y, z, roll, pitch, yaw):
    R = np.array([
        [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
        [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    pcd.transform(T)
    return pcd

def main():
    dataset_folder = 'localization/dataset'
    ground_truth_file = 'localization/dataset/ground_truth.csv'
    reference_pcd_path = 'localization/dataset/reference_map.pcd'  # Path to the reference point cloud

    ground_truth = load_ground_truth(ground_truth_file)
    ground_truth.columns = [col.strip().lower() for col in ground_truth.columns]
    reference_pcd = o3d.io.read_point_cloud(reference_pcd_path)  # Load reference point cloud once

    ndt = NDTRegistration(resolution=1.0)  # Initialize NDT registration

    for index, row in ground_truth.iterrows():
        frame_number = int(row['frame'])
        source_pcd = load_pcd(frame_number, dataset_folder)
        if source_pcd is not None:
            transformed_pcd = transform_point_cloud(source_pcd, row['x'], row['y'], row['z'], row['roll'], row['pitch'], row['yaw'])
            transformation_ndt = ndt.register(transformed_pcd, reference_pcd)
            transformed_pcd.transform(transformation_ndt)  # Apply NDT transformation
            o3d.visualization.draw_geometries([transformed_pcd, reference_pcd])

if __name__ == "__main__":
    main()
