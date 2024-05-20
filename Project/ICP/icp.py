import open3d as o3d
import numpy as np
import time
import pandas as pd
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# User-defined parameters
VOXEL_SIZE = 0.5
THRESHOLD = 0.1
MAXIMUM_ITERATION = 100
LATERAL_ERROR_MAX = 1.2

# Paths
root_dir = Path(__file__).parents[2]
MAP_PATH = root_dir / "dataset/map.pcd"
FRAMES_PATH = root_dir / "dataset/frames"
GROUND_TRUTH_PATH = root_dir / "dataset/ground_truth.csv"
SAVING_PATH = root_dir / "Project/ICP"

# Ensure the saving directory exists
SAVING_PATH.mkdir(parents=True, exist_ok=True)

map_cloud = o3d.io.read_point_cloud(str(MAP_PATH))

ground_truth = pd.read_csv(str(GROUND_TRUTH_PATH))

extracted_cars = []

for i in range(len(ground_truth)):
    frame_file = os.path.join(FRAMES_PATH, f"frame_{i}.pcd")

    if not os.path.exists(frame_file):
        print(f"Frame file {frame_file} not found.")
        continue

    frame_cloud = o3d.io.read_point_cloud(frame_file)
    frame_cloud_down = frame_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)

    x, y, z = ground_truth.iloc[i][1:4]
    roll, pitch, yaw = np.deg2rad(ground_truth.iloc[i][4:7])

    r = R.from_euler('xyz', [roll, pitch, yaw])
    rotation_matrix = r.as_matrix()
    translation_vector = np.array([x, y, z])
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    frame_cloud_down.transform(transformation_matrix)
    extracted_cars.append(frame_cloud_down)

time_rec = []
lateral_err = []
aligned_car_clouds = []

for i, car_cloud in enumerate(extracted_cars):
    start_time = time.time()

    reg_p2p = o3d.pipelines.registration.registration_icp(
        car_cloud, map_cloud, THRESHOLD, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAXIMUM_ITERATION)
    )

    end_time = time.time()
    time_rec.append(end_time - start_time)

    lateral_error = o3d.pipelines.registration.evaluate_registration(
        car_cloud, map_cloud, THRESHOLD, reg_p2p.transformation
    ).inlier_rmse
    lateral_err.append(lateral_error)

    if lateral_error > LATERAL_ERROR_MAX:
        print(f"Lateral error ({lateral_error:.2f} m) is greater than the maximum allowed (1.2 m).")
        break

    car_cloud.transform(reg_p2p.transformation)
    aligned_car_clouds.append(car_cloud)

    print(f"Processed {i + 1}/{len(extracted_cars)} frames.")

merged_aligned_cars = o3d.geometry.PointCloud()
for car_cloud in aligned_car_clouds:
    merged_aligned_cars += car_cloud

# Save the aligned car point clouds
o3d.io.write_point_cloud(str(SAVING_PATH / "aligned_car.pcd"), merged_aligned_cars)

# Results
print(f"-->>  Mean time per frame: {np.mean(time_rec):.2f} s")
print(f"-->>  Mean lateral error: {np.mean(lateral_err):.2f} m")

# Visualization (Optional)
pcd = o3d.io.read_point_cloud(str(SAVING_PATH / "aligned_car.pcd"))
o3d.visualization.draw_geometries([pcd])
