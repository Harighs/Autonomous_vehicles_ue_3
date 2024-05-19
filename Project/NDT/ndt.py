import open3d as o3d
import numpy as np
import pandas as pd
import os
import time
from scipy.spatial.transform import Rotation as R
from helper import NDT, Pose
from scipy.stats import multivariate_normal
from typing import Tuple, List


class Pose:
    """
    Pose class for 3D transformations
    """

    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        """
        Constructor of the class

        Args:
            x (float): X offset of the pose
            y (float): Y offset of the pose
            z (float): Z offset of the pose
            roll (float): Roll angle of the pose
            pitch (float): Pitch angle of the pose
            yaw (float): Yaw angle of the pose
        """
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def get_transformation(self) -> np.array:
        """
        Method to obtain the transformation matrix of a given pose.

        Returns:
            np.array: 4x4 transformation matrix
        """
        R_matrix = R.from_euler('xyz', [self.roll, self.pitch, self.yaw]).as_matrix()
        t_vector = np.array([self.x, self.y, self.z]).reshape((3, 1))
        transformation = np.eye(4)
        transformation[:3, :3] = R_matrix
        transformation[:3, 3] = t_vector.flatten()
        return transformation

    def __add__(self, other):
        """
        Addition method
        Args:
            other (Pose): Pose to add

        Returns:
            Pose: new pose that represents the addition of two poses.
        """
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        roll = self.roll + other.roll
        pitch = self.pitch + other.pitch
        yaw = self.yaw + other.yaw
        return Pose(x, y, z, roll, pitch, yaw)


class Cell:
    """
    Cell implementation for a NDT grid in 3D
    """

    def __init__(self):
        """
        Constructor by default, the cell is empty.
        with 0 mean and zero covariance.
        """
        self.mean = np.zeros((3, 1))
        self.cov = np.zeros((3, 3))
        self.rv = None
        self.points = []

    def set_points(self, points: np.array) -> None:
        """
        Method to populate the cell. This method fills the mean and covariance
        members of the cell

        Args:
            points (np.array): points that fall in a given cell.

        """
        self.points = points
        if len(points) > 0:
            self.mean = np.mean(points[:, :3], axis=0)
            self.cov = np.cov(points[:, :3].T)

            epsilon = 1e-5
            if np.any(np.diag(self.cov) == 0):
                self.cov += np.eye(self.cov.shape[0]) * epsilon
            self.rv = multivariate_normal(self.mean, self.cov)
        else:
            self.mean = None
            self.cov = None

    def pdf(self, point: np.array) -> float:
        """
        Probability that a given point lies on the given cell.

        Args:
            point (np.array): (x,y,z) point to calculate the probability.

        Returns:
            float: probability.
        """
        if self.mean is None:
            return 0.0
        else:
            pdf = np.exp(-0.5 * (point - self.mean) @ np.linalg.inv(self.cov) @ (point - self.mean).T)
            return pdf


class NDT:
    """
    Normal Distributions Transform class for 3D point cloud alignment
    """

    def __init__(self, x_step: float, y_step: float, z_step: float, xlim: Tuple[int, int] = None, ylim: Tuple[int, int] = None, zlim: Tuple[int, int] = None):
        """
        Constructor
        Args:
            x_step (float): Resolution of the grid in x direction
            y_step (float): Resolution of the grid in y direction
            z_step (float): Resolution of the grid in z direction
            xlim (Tuple[int, int], optional): limits of our grid in x direction. Defaults to None.
            ylim (Tuple[int, int], optional): limits of our grid in y direction. Defaults to None.
            zlim (Tuple[int, int], optional): limits of our grid in z direction. Defaults to None.
        """
        self.x_step = x_step
        self.y_step = y_step
        self.z_step = z_step
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.grid = None
        self.bbox = None

    def set_input_cloud(self, pcd: np.array) -> None:
        """
        Method to populate the NDT grid given an input point cloud. It is in charge to calculate the
        cell that each point belongs to and populate each cell.

        Args:
            pcd (np.array): pointcloud with shape (n_points, 3)
        """
        x_min_pcd, y_min_pcd, z_min_pcd = np.min(pcd[:, :3], axis=0) - 1
        x_max_pcd, y_max_pcd, z_max_pcd = np.max(pcd[:, :3], axis=0) + 1

        if self.xlim is None:
            self.xlim = [x_min_pcd, x_max_pcd]
        if self.ylim is None:
            self.ylim = [y_min_pcd, y_max_pcd]
        if self.zlim is None:
            self.zlim = [z_min_pcd, z_max_pcd]

        x_min, x_max = self.xlim
        y_min, y_max = self.ylim
        z_min, z_max = self.zlim

        num_voxels_x = int(np.ceil((x_max - x_min) / self.x_step))
        num_voxels_y = int(np.ceil((y_max - y_min) / self.y_step))
        num_voxels_z = int(np.ceil((z_max - z_min) / self.z_step))

        xs = np.linspace(x_min, x_max, num_voxels_x)
        ys = np.linspace(y_min, y_max, num_voxels_y)
        zs = np.linspace(z_min, z_max, num_voxels_z)

        self.grid = [[[Cell() for _ in range(num_voxels_x - 1)] for _ in range(num_voxels_y - 1)] for _ in range(num_voxels_z - 1)]
        self.bbox = [(x_min, y_min, z_min), (x_max, y_max, z_max)]

        for i in range(len(zs) - 1):
            for j in range(len(ys) - 1):
                for k in range(len(xs) - 1):
                    mask = np.where((pcd[:, 0] >= xs[k]) &
                                    (pcd[:, 0] <= xs[k + 1]) &
                                    (pcd[:, 1] >= ys[j]) &
                                    (pcd[:, 1] <= ys[j + 1]) &
                                    (pcd[:, 2] >= zs[i]) &
                                    (pcd[:, 2] <= zs[i + 1]))

                    q = pcd[mask]
                    self.grid[i][j][k].set_points(q)

    def get_cell(self, point: np.array) -> Cell:
        """
        Returns the cell that point belongs to.

        Args:
            point (np.array): query point which we want to know the cell

        Returns:
            Cell: Cell where the point is located.
        """
        x_min, x_max = self.xlim
        y_min, y_max = self.ylim
        z_min, z_max = self.zlim
        width = int(np.ceil((x_max - x_min) / self.x_step)) - 1
        height = int(np.ceil((y_max - y_min) / self.y_step)) - 1
        depth = int(np.ceil((z_max - z_min) / self.z_step)) - 1

        c = int((point[0] - x_min) / self.x_step)
        r = int((point[1] - y_min) / self.y_step)
        d = int((point[2] - z_min) / self.z_step)

        if (c >= 0 and c < width) and (r >= 0 and r < height) and (d >= 0 and d < depth):
            return self.grid[d][r][c]
        else:
            return None

    def align(self, pcd: np.array, init_pose: Pose, max_iterations: int = 100, eps: float = 1e-3) -> Tuple[Pose, List[Tuple[np.array, np.array, float]]]:
        """
        Principal method that aligns a given point cloud with the point cloud that was used to populate the NDT grid.

        Args:
            pcd (np.array): Point cloud to be aligned.
            init_pose (Pose): Estimated initial pose.
            max_iterations (int, optional): Maximum number of iterations to calculate the alignment. Defaults to 100.
            eps (float, optional): Threshold criteria to check if the algorithm has converged to a solution. Defaults to 1e-3.

        Returns:
            Tuple[Pose, List[Tuple[np.array, np.array, float]]]: - Pose between the point cloud and the map.
                                                                  - List of [Rotation, translation, score] in each iteration for animation purposes.
        """
        pose = init_pose
        cache_list = []
        for iteration in range(max_iterations):
            transformation_matrix = pose.get_transformation()
            transformed_pcd = (transformation_matrix[:3, :3] @ pcd[:, :3].T + transformation_matrix[:3, 3].reshape(3, 1)).T

            score = self.calculate_score(transformed_pcd)
            cache_list.append((transformation_matrix[:3, :3], transformation_matrix[:3, 3], score))

            delta_T = self.newtons_method(transformed_pcd, pose)

            alpha = self.compute_step_length(delta_T, pcd, pose, score)

            pose.x += alpha * delta_T[0, 0]
            pose.y += alpha * delta_T[1, 0]
            pose.z += alpha * delta_T[2, 0]
            pose.roll += alpha * delta_T[3, 0]
            pose.pitch += alpha * delta_T[4, 0]
            pose.yaw += alpha * delta_T[5, 0]

        return pose, cache_list

    def newtons_method(self, pcd: np.array, pose: Pose) -> np.array:
        """
        Implementation of one step of Newton's method, with the equations given in class

        Args:
            pcd (np.array): Point cloud to calculate Newton's method.

        Returns:
            np.array: vector with the change of the parameters (delta_tx, delta_ty, delta_tz, delta_roll, delta_pitch, delta_yaw)
        """
        gradient = np.zeros((1, 6))
        H = np.zeros((6, 6))
        for point in pcd:
            cell = self.get_cell(point)

            if cell is None or len(cell.points) <= 2:
                continue
            point = np.reshape(point[:3], (1, 3))
            delta_g, delta_H = self.gradient_jacobian_point(point, pose, cell)
            gradient = gradient + delta_g
            H = H + delta_H

        H = self.pos_definite(H, 0, 5)
        delta_T = -np.linalg.inv(H) @ gradient.T
        return delta_T

    def gradient_jacobian_point(self, point: np.array, pose: Pose, cell: Cell) -> Tuple[np.array, np.array]:
        """
        Helper function to calculate the Jacobian and Hessian for a given point.

        Args:
            point (np.array): Point used to calculate one summand of the score
            pose (Pose): current pose.
            cell (Cell): cell where the point belongs to.

        Returns:
            Tuple[np.array, np.array]: - delta_gradient: The gradient calculated with the input point
                                       - delta_H: The Hessian calculated with the given point.
        """
        mean = cell.mean
        cov = cell.cov
        cov_inv = np.linalg.inv(cov)
        q = point - mean
        expo = np.exp(-0.5 * (q @ cov_inv @ q.T))
        J = self.calculate_jacobian(point, pose)
        delta_gradient = (q @ cov_inv @ J) * expo
        delta_H = self.calculate_hessian(point, pose, cell, J)
        return delta_gradient, delta_H

    def calculate_jacobian(self, point: np.array, pose: Pose) -> np.array:
        """
        Calculate the Jacobian of the score given a point and the angle of its pose

        Args:
            point (np.array): Point used to calculate the Jacobian
            pose (Pose): current pose.

        Returns:
            np.array: Calculated Jacobian.
        """
        x = point[:, 0].item()
        y = point[:, 1].item()
        z = point[:, 2].item()
        roll = pose.roll
        pitch = pose.pitch
        yaw = pose.yaw
        J = np.zeros((3, 6))
        J[:3, :3] = np.eye(3)
        J[0, 3] = -x * np.sin(roll) - y * np.cos(roll) * np.sin(pitch) + z * np.cos(roll) * np.cos(pitch)
        J[1, 3] = x * np.cos(roll) - y * np.sin(roll) * np.sin(pitch) + z * np.sin(roll) * np.cos(pitch)
        J[2, 3] = -y * np.cos(pitch) - z * np.sin(pitch)
        J[0, 4] = y * np.sin(roll) - z * np.cos(roll)
        J[1, 4] = -x * np.sin(roll) + z * np.sin(roll)
        J[2, 4] = -x * np.cos(roll) - y * np.sin(roll)
        J[0, 5] = -y * np.sin(pitch) - z * np.cos(pitch)
        J[1, 5] = x * np.sin(pitch) - z * np.cos(pitch)
        J[2, 5] = x * np.cos(pitch) + y * np.sin(pitch)
        return J

    def calculate_hessian(self, point: np.array, pose: Pose, cell: Cell, J: np.array) -> np.array:
        """
        Helper function to calculate the Hessian matrix of a given point.

        Args:
            point (np.array): Point used to calculate part of the Hessian
            pose (Pose): current pose.
            cell (Cell): Cell that the point belongs to.
            J (np.array): Jacobian of the score using the point and pose

        Returns:
            np.array: Calculated Hessian.
        """
        x = point[:, 0].item()
        y = point[:, 1].item()
        z = point[:, 2].item()
        mean = cell.mean
        cov = cell.cov
        cov_inv = np.linalg.inv(cov)
        q = point - mean
        expo = np.exp(-0.5 * (q @ cov_inv @ q.T))

        dq2 = np.zeros((3, 6))
        dq2[0, 3] = -x * np.cos(pose.roll) + y * np.sin(pose.roll) * np.sin(pose.pitch) - z * np.sin(pose.roll) * np.cos(pose.pitch)
        dq2[1, 3] = -x * np.sin(pose.roll) - y * np.cos(pose.roll) * np.sin(pose.pitch) + z * np.cos(pose.roll) * np.cos(pose.pitch)
        dq2[2, 3] = y * np.cos(pose.pitch) + z * np.sin(pose.pitch)

        H1 = (-q @ cov_inv @ J).T @ (-q @ cov_inv @ J)
        H2 = (-q @ cov_inv @ dq2).T @ np.array([[0, 0, 0, 1, 1, 1]])
        H3 = -J.T @ cov_inv @ J
        H = -expo * (H1 + H2 + H3)

        return H

    def pos_definite(self, H: np.array, start: float, increment: float, max_iterations=100) -> np.array:
        """
        Function to ensure that the Matrix H is positive definite.

        Args:
            H (np.array): Hessian matrix that is going to be checked
            start (float): Start lambda that has to be added in case H is not positive definite.
            increment (float): Increment in lambda for each iteration.
            max_iterations (int, optional): Maximum amount of iterations to check if H is positive definite. Defaults to 100.

        Returns:
            np.array: Positive definite Hessian
        """
        I = np.eye(H.shape[0])
        pos_H = H + start * I

        for _ in range(max_iterations):
            eigenvalues = np.linalg.eigvals(pos_H)

            if np.all(eigenvalues > 0):
                break

            pos_H = pos_H + increment * I

        return pos_H

    def calculate_score(self, points: np.array) -> float:
        """
        Calculate the score of a given point cloud

        Args:
            points (np.array): point cloud used to calculate the score.

        Returns:
            float: obtained score.
        """
        score = 0
        for point in points:
            cell = self.get_cell(point[:3])
            if not cell is None and len(cell.points) > 2:
                score += cell.pdf(point[:3])
        return score

    def compute_step_length(self, T: np.array, source: np.array, pose: Pose, curr_score: float) -> float:
        """
        Heuristic way to calculate alpha.

        Args:
            T (np.array): delta_T obtained with Newton's method.
            source (np.array): source point cloud
            pose (Pose): current pose
            curr_score (float): current score

        Returns:
            float: obtained alpha
        """
        source = source[:, :3]
        T = T.copy()
        max_param = max(abs(T[0, 0]), max(abs(T[1, 0]), max(abs(T[2, 0]), max(abs(T[3, 0]), max(abs(T[4, 0]), abs(T[5, 0]))))))
        mlength = 1.0
        if max_param > 0.2:
            mlength = 0.1 / max_param
            T *= mlength

        best_alpha = 0

        # Try smaller steps
        alpha = 1.0
        for _ in range(40):
            adj_score = self.adjustment_score(alpha, T, source, pose)
            if adj_score > curr_score:
                best_alpha = alpha
                curr_score = adj_score
            alpha *= 0.7

        if best_alpha == 0:
            # Try larger steps
            alpha = 2.0
            for _ in range(10):
                adj_score = self.adjustment_score(alpha, T, source, pose)
                if adj_score > curr_score:
                    best_alpha = alpha
                    curr_score = adj_score
                alpha *= 2

        return best_alpha * mlength

    def adjustment_score(self, alpha: float, T: np.array, source: np.array, pose: Pose) -> float:
        """
        Obtained score if we applied a given alpha to update our pose.
        Args:
            alpha (float): Tentative alpha
            T (np.array): Current delta in the parameters
            source (np.array): Source point cloud.
            pose (Pose): current pose.

        Returns:
            float: Obtained score.
        """
        T = T.copy()
        T *= alpha
        p_cpy = Pose(0, 0, 0, 0, 0, 0)
        p_cpy = pose + p_cpy
        p_cpy.x += T[0, 0]
        p_cpy.y += T[1, 0]
        p_cpy.z += T[2, 0]
        p_cpy.roll += T[3, 0]
        p_cpy.pitch += T[4, 0]
        p_cpy.yaw += T[5, 0]

        transformation_matrix = p_cpy.get_transformation()
        transformed_scan = (transformation_matrix[:3, :3] @ source.T + transformation_matrix[:3, 3].reshape(3, 1)).T

        return self.calculate_score(transformed_scan)



# # Paths 
# map_path = "/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project3/Autonomous_vehicles_ue_3/localization/dataset/map.pcd"
map_path = "/home/hari/Projects/Autonomous_vehicles_ue_3/dataset/map.pcd"
# frames_path = "/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project3/Autonomous_vehicles_ue_3/localization/dataset/frames"
frames_path = "/home/hari/Projects/Autonomous_vehicles_ue_3/dataset/frames"
# ground_truth_path = '/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project3/Autonomous_vehicles_ue_3/localization/dataset/ground_truth.csv'
ground_truth_path = '/home/hari/Projects/Autonomous_vehicles_ue_3/dataset/ground_truth.csv'

# Load the point cloud
map_cloud = o3d.io.read_point_cloud(map_path)

# Load ground truth data
ground_truth = pd.read_csv(ground_truth_path)

# Initialize lists
extracted_cars = []

# Prepare the frames
for i in range(len(ground_truth)):
    frame_file = os.path.join(frames_path, f"frame_{i}.pcd")

    if not os.path.exists(frame_file):
        print(f"Frame file {frame_file} not found.")
        continue

    frame_cloud = o3d.io.read_point_cloud(frame_file)
    frame_cloud_down = frame_cloud.voxel_down_sample(voxel_size=0.5)

    # Ground truth transformation
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

# Initialize Pose
init_pose = Pose(0, 0, 0, 0, 0, 0)  # Make a good initial guess

# Create NDT object
ndt = NDT(15, 15, 15)  # Adjust grid resolution if necessary

# Set input cloud to populate grids
target_pcd = np.asarray(map_cloud.points)
ndt.set_input_cloud(target_pcd)

# Initialize lists for timing and errors
time_rec = []
lateral_err = []

# Use pose from the previous iteration as the starting point for the next iteration
current_pose = init_pose

map_cloud_kdtree = o3d.geometry.KDTreeFlann(map_cloud)
for i, car_cloud in enumerate(extracted_cars):
    start_time = time.time()

    source_pcd = np.asarray(car_cloud.points)
    current_pose, cache_list = ndt.align(source_pcd, current_pose, max_iterations=15, eps=1e-3)

    end_time = time.time()
    time_rec.append(end_time - start_time)

    # Transform the source point cloud using the calculated pose
    transformation_matrix = current_pose.get_transformation()
    transformed_points = (transformation_matrix[:3, :3] @ source_pcd[:, :3].T + transformation_matrix[:3, 3].reshape(3, 1)).T

    # Compute the registration error using nearest neighbor distances
    dists = []
    for point in transformed_points:
        [_, idx, dist] = map_cloud_kdtree.search_knn_vector_3d(point, 1)
        dists.append(dist[0])
    reg_error = np.mean(dists)
    lateral_err.append(reg_error)

    print('Lateral error is :', reg_error)
    if reg_error > 1.2:
        print(f"Lateral error ({reg_error:.2f} m) is greater than the maximum allowed (1.2 m).")
        break

    print(f"Processed {i + 1}/{len(extracted_cars)} frames.")

# Results
print(f"----->>>>>>>>>>  Mean time per frame: {np.mean(time_rec):.2f} s")
print(f"----->>>>>>>>>>  Mean lateral error: {np.mean(lateral_err):.2f} m")

# Save the aligned car point clouds
merged_aligned_cars = o3d.geometry.PointCloud()
for car_cloud in extracted_cars:
    merged_aligned_cars += car_cloud
o3d.io.write_point_cloud("aligned_car_ndt.pcd", merged_aligned_cars)



# Visualize the results
import open3d as o3d
pcd = o3d.io.read_point_cloud("/home/ari/Workplace/JKU/SEM_2/Autonomous_sys/Project3/Autonomous_vehicles_ue_3/localization/Project/NDT/aligned_car_ndt.pcd")
o3d.visualization.draw_geometries([pcd])