import open3d as o3d
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd
import os

#
# Paths
map_path = "/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project3/Autonomous_vehicles_ue_3/localization/dataset/map.pcd"
frames_path = "/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project3/Autonomous_vehicles_ue_3/localization/dataset/frames"
ground_truth_path = '/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project3/Autonomous_vehicles_ue_3/localization/dataset/ground_truth.csv'

# point cloud
map_cloud = o3d.io.read_point_cloud(map_path)

# ground truth
ground_truth = pd.read_csv(ground_truth_path)

extracted_cars = []

# Iterate through each frame
for i in range(len(ground_truth)):
    frame_file = os.path.join(frames_path, f"frame_{i}.pcd")

    if not os.path.exists(frame_file):
        print(f"Frame file {frame_file} not found.")
        continue

    frame_cloud = o3d.io.read_point_cloud(frame_file)
    frame_cloud_down = frame_cloud.voxel_down_sample(voxel_size=0.5)

    # Extract ground truth information
    x, y, z = ground_truth.iloc[i][1:4]
    roll, pitch, yaw = np.deg2rad(ground_truth.iloc[i][4:7])

    # Create transformation matrix
    r = R.from_euler('xyz', [roll, pitch, yaw])
    rotation_matrix = r.as_matrix()
    translation_vector = np.array([x, y, z])
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    # Apply initial transformation
    frame_cloud_down.transform(transformation_matrix)
    extracted_cars.append(frame_cloud_down)

# ICP configuration
threshold = 0.1
initial_transformation = np.eye(4)
maximum_iteration = 100

time_rec = []
lateral_err = []
aligned_car_clouds = []

# Localize each frame using ICP
for i, car_cloud in enumerate(extracted_cars):
    start_time = time.time()

    reg_p2p = o3d.pipelines.registration.registration_icp(
        car_cloud, map_cloud, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=maximum_iteration)
    )

    end_time = time.time()
    time_rec.append(end_time - start_time)

    # Evaluate registration error
    lateral_error = o3d.pipelines.registration.evaluate_registration(
        car_cloud, map_cloud, threshold, reg_p2p.transformation
    ).inlier_rmse
    lateral_err.append(lateral_error)

    if lateral_error > 1.2:
        print(f"Lateral error ({lateral_error:.2f} m) is greater than the maximum allowed (1.2 m).")
        break

    # Apply final transformation to car cloud
    car_cloud.transform(reg_p2p.transformation)
    aligned_car_clouds.append(car_cloud)

    # Update initial transformation
    initial_transformation = reg_p2p.transformation

    print(f"Processed {i + 1}/{len(extracted_cars)} frames.")

# Merge aligned car point clouds
merged_aligned_cars = o3d.geometry.PointCloud()
for car_cloud in aligned_car_clouds:
    merged_aligned_cars += car_cloud

# Save the aligned car point clouds
o3d.io.write_point_cloud("aligned_car.pcd", merged_aligned_cars)

# Results
print(f"----->>>>>>>>>>  Mean time per frame: {np.mean(time_rec):.2f} s")
print(f"----->>>>>>>>>>  Mean lateral error: {np.mean(lateral_err):.2f} m")

# # ######################################################### End of solution #########################################################

# After saving the aligned point cloud, you can visualize it using the following code:
pcd = o3d.io.read_point_cloud("/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project3/Autonomous_vehicles_ue_3/localization/Project/ICP/aligned_car.pcd")
o3d.visualization.draw_geometries([pcd])

