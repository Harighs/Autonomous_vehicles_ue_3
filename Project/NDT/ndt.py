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
    Pose class
    """

    def __init__(self, x: float, y: float, yaw: float):
        """
        Constructor of the class

        Args:
            x (float): X offset of the pose
            y (float): Y offset of the pose
            yaw (float): Yaw angle of the pose
        """
        self.x = x
        self.y = y
        self.yaw = yaw

    def get_transformation(self) -> Tuple[np.array, np.array]:
        """
        Method to obtain the Rotation matrix and translation
        vector of a given pose.

        Returns:
            Tuple[np.array, np.array]: Rotation as a 2x2 matrix
                                       translation vector as 2x1 matrix
        """
        x = self.x
        y = self.y
        yaw = self.yaw
        R = np.asarray([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        t = np.asarray([[x], [y]])
        return R, t

    def __add__(self, other):
        """
        Addition method
        Args:
            other (Pose): Pose to add

        Returns:
            Pose: new pose that represents the addition
                  of two poses.
        """
        x = self.x + other.x
        y = self.y + other.y
        yaw = self.yaw + other.yaw
        return Pose(x, y, yaw)


class Cell:
    """
    Cell implementation for and NDT grid
    """

    def __init__(self):
        """
        Constructor by default, the cell is empty.
        with 0 mean and zero covariance.
        """
        self.mean = np.zeros((2, 1))
        self.cov = np.zeros((2, 2))
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
            self.mean = np.mean(points[:, :2], axis=0)
            self.cov = np.cov(points[:, :2].T)

            # print('cov is :', self.cov)
            # print('mean is :', self.mean)

            self.rv = multivariate_normal(self.mean, self.cov)

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
            point (np.array): (x,y) point to calculate the probability.

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
    Normal distribution class.

    This class performs all the required functions needed to perform the aligment of
    an scan to a map.
    """

    def __init__(self, x_step: float, y_step: float, ylim: Tuple[int, int] = None, xlim: Tuple[int, int] = None):
        """
        Constructor
        Args:
            x_step (float): Resolution of the grid in x direction
            y_step (float): Resolution of the grid in y direction
            ylim (Tuple[int, int], optional): limits of our grid in y direction. Defaults to None.
            xlim (Tuple[int, int], optional): limits of our grid in x direction. Defaults to None.
        """
        self.x_step = x_step
        self.y_step = y_step
        self.xlim = xlim
        self.ylim = ylim
        self.grid = None
        self.bbox = None

    def set_input_cloud(self, pcd: np.array) -> None:
        """
        Method to populate the NDT grid given a input point cloud. It is in charge to calculate the
        cell that each point belongs to and populate each cell.

        Args:
            pcd (np.array): pointcloud with shape (n_points,3)
        """

        x_min_pcd, y_min_pcd = np.min(pcd[:, :2], axis=0) - 1
        x_max_pcd, y_max_pcd = np.max(pcd[:, :2], axis=0) + 1

        if self.xlim is None:
            self.xlim = [x_min_pcd, x_max_pcd]
        if self.ylim is None:
            self.ylim = [y_min_pcd, y_max_pcd]

        x_min, x_max = self.xlim
        y_min, y_max = self.ylim

        num_voxels_x = int(np.ceil((x_max - x_min) / self.x_step))
        num_voxels_y = int(np.ceil((y_max - y_min) / self.y_step))
        xs = np.linspace(x_min, x_max, num_voxels_x)
        ys = np.linspace(y_min, y_max, num_voxels_y)

        self.grid = [[Cell() for _ in range(num_voxels_x - 1)] for _ in range(num_voxels_y - 1)]
        self.bbox = [(x_min, y_min), (x_max, y_max)]
        for i in range(len(ys) - 1):
            for j in range(len(xs) - 1):
                mask = np.where((pcd[:, 0] >= xs[j]) &
                                (pcd[:, 0] <= xs[j + 1]) &
                                (pcd[:, 1] >= ys[i]) &
                                (pcd[:, 1] <= ys[i + 1]))

                q = pcd[mask]
                self.grid[i][j].set_points(q)

    def get_cell(self, point: np.array) -> Cell:
        x_min, x_max = self.xlim
        y_min, y_max = self.ylim
        width = int(np.ceil((x_max - x_min) / self.x_step)) - 1
        height = int(np.ceil((y_max - y_min) / self.y_step)) - 1
        if not (self.x_step > 0 and self.y_step > 0):
            raise ValueError("Grid step sizes must be positive")
        c = int((point[0] - x_min) / self.x_step) if not np.isnan(point[0]) else None
        r = int((point[1] - y_min) / self.y_step) if not np.isnan(point[1]) else None
        if c is None or r is None or c < 0 or c >= width or r < 0 or r >= height:
            return None
        return self.grid[r][c]


    def align(self, pcd: np.array, init_pose: Pose, max_iterations: int = 100, eps: float = 1e-3) -> Tuple[
        Pose, List[Tuple[np.array, np.array, float]]]:
        """
        Principal method that aligns a given pointcloud with the pointcloud that
        was used to populate the NDT grid.

        Args:
            pcd (np.array): Pointcloud to be aligned.
            init_pose (_type_): Estimated initial pose.
            max_iterations (int, optional): Maximum number of iterations to calculate the aligment. Defaults to 100.
            eps (_type_, optional): Threshold criteria to check if the algorithm has converged to a solution. Defaults to 1e-3.

        Returns:
            Tuple[Pose, List[np.array, np.array, float]]: - Pose between the pointcloud and the map.
                                                          - List of [Rotation, translation, score] in each iteration for animation
                                                            porpuses.
        """
        pose = init_pose
        cache_list = []
        for iteration in range(max_iterations):

            R, t = pose.get_transformation()
            transformed_pcd = R @ pcd[:, :2].T + t
            transformed_pcd = transformed_pcd.T

            score = self.calculate_score(transformed_pcd)
            cache_list.append((R, t, score))

            delta_T = self.newtons_method(transformed_pcd, pose)

            alpha = self.compute_step_length(delta_T, pcd, pose, score)

            pose.x += alpha * delta_T[0, 0]
            pose.y += alpha * delta_T[1, 0]
            pose.yaw += alpha * delta_T[2, 0]

            if pose.yaw > 2 * np.pi:
                n = np.floor(pose.yaw / 2 * np.pi)
                pose.yaw -= n * (2 * np.pi)

        return pose, cache_list

    def newtons_method(self, pcd: np.array, pose: Pose) -> np.array:
        """
        Implementation of one step of the newtons method, with the equations given in class

        Args:
            pcd (np.array): Pointcloud to calculate the newtons method.

        Returns:
            np.array: vector with the change of the parameters (delta_tx,delta_ty,delta_yaw)
        """
        gradient = np.zeros((1, 3))
        H = np.zeros((3, 3))
        for point in pcd:
            cell = self.get_cell(point)

            if cell is None or len(cell.points) <= 2:
                continue
            point = np.reshape(point[:2], (1, 2))
            delta_g, delta_H = self.gradient_jacobian_point(point, pose.yaw, cell)
            gradient = gradient + delta_g
            H = H + delta_H

        H = self.pos_definite(H, 0, 5)
        delta_T = -np.linalg.inv(H) @ gradient.T
        return delta_T

    def gradient_jacobian_point(self, point: np.array, theta: float, cell: Cell) -> Tuple[np.array, np.array]:
        """
        Helper function to calculate the jacobian and hessian for a given point.

        Args:
            point (np.array): Point used to calculate one summand of the score
            theta (float): yaw angle of the current pose.
            cell (Cell): cell where the point belongs to.

        Returns:
            Tuple[np.array, np.array]: - delta_gradient: The gradient calculated with the input point
                                       - delta_H: The hessian calculated with the given point.
        """
        mean = cell.mean
        cov = cell.cov
        cov_inv = np.linalg.inv(cov)
        q = point - mean
        expo = np.exp(-0.5 * (q @ cov_inv @ q.T))
        J = self.calculate_jacobian(point, theta)
        delta_gradient = (q @ cov_inv @ J) * expo
        delta_H = self.calculate_hessian(point, theta, cell, J)
        return delta_gradient, delta_H

    def calculate_jacobian(self, point: np.array, theta: float) -> np.array:
        """
        Calculate the jacobian of the score given a point and the angle of its pose

        Args:
            point (np.array): Point used to calculate the jacobian
            theta (float): Angle of the pose.

        Returns:
            np.array: Calculated Jacobian. Please see the equations given in the lesson.
        """
        x = point[:, 0].item()
        y = point[:, 1].item()
        J = np.zeros((2, 3))
        J[0, 0] = 1.0
        J[1, 1] = 1.0
        J[0, 2] = -x * np.sin(theta) - y * np.cos(theta)
        J[1, 2] = x * np.cos(theta) - y * np.sin(theta)
        return J

    def calculate_hessian(self, point: np.array, theta: float, cell: Cell, J: np.array) -> np.array:
        """
        Helper function to calculate the Hessian matrix of a given point.

        Args:
            point (np.array): Point used to calculate part of the hessian
            theta (float): Angle of the pose.
            cell (Cell): Cell that the point belongs to.
            J (np.array): Jacobian of the score using the point and theta

        Returns:
            np.array: Calculated Hessian. Please see the equations given in the lesson.
        """
        x = point[:, 0].item()
        y = point[:, 1].item()
        mean = cell.mean
        cov = cell.cov
        cov_inv = np.linalg.inv(cov)
        q = point - mean
        expo = np.exp(-0.5 * (q @ cov_inv @ q.T))

        dq2 = np.zeros((2, 3))
        dq2[0, 2] = -x * np.cos(theta) + y * np.sin(theta)
        dq2[1, 2] = -x * np.sin(theta) - y * np.cos(theta)

        H1 = (-q @ cov_inv @ J).T @ (-q @ cov_inv @ J)
        H2 = (-q @ cov_inv @ dq2).T @ np.asarray([[0, 0, 1]])
        H3 = -J.T @ cov_inv @ J
        H = -expo * (H1 + H2 + H3)

        # other implementation easier to understand
        # q1 = np.reshape(J[:,0],(2,1))
        # q2 = np.reshape(J[:,1],(2,1))
        # q3 = np.reshape(J[:,2],(2,1))

        # H1[0,0] = (-q@cov_inv@q1)@(-q@cov_inv@q1)
        # H1[0,1] = (-q@cov_inv@q1)@(-q@cov_inv@q2)
        # H1[0,2] = (-q@cov_inv@q1)@(-q@cov_inv@q3)
        # H1[1,0] = (-q@cov_inv@q2)@(-q@cov_inv@q1)
        # H1[1,1] = (-q@cov_inv@q2)@(-q@cov_inv@q2)
        # H1[1,2] = (-q@cov_inv@q2)@(-q@cov_inv@q3)
        # H1[2,0] = (-q@cov_inv@q3)@(-q@cov_inv@q1)
        # H1[2,1] = (-q@cov_inv@q3)@(-q@cov_inv@q2)
        # H1[2,2] = (-q@cov_inv@q3)@(-q@cov_inv@q3)

        # H2 = np.zeros((3,3))

        # H2[0,0] = (-q1.T@cov_inv@q1)
        # H2[0,1] = (-q2.T@cov_inv@q1)
        # H2[0,2] = (-q3.T@cov_inv@q1)
        # H2[1,0] = (-q1.T@cov_inv@q2)
        # H2[1,1] = (-q2.T@cov_inv@q2)
        # H2[1,2] = (-q3.T@cov_inv@q2)
        # H2[2,0] = (-q1.T@cov_inv@q3)
        # H2[2,1] = (-q2.T@cov_inv@q3)
        # H2[2,2] = (-q3.T@cov_inv@q3)

        # dq2_33 = np.zeros((2,1))
        # dq2_33[0,0] = -x*np.cos(theta) + y*np.sin(theta)
        # dq2_33[1,0] = -x*np.sin(theta) - y*np.cos(theta)
        # H = H1 + H2
        # H[2,2] = H[2,2] + -q@cov_inv@dq2_33
        # H = -expo*H

        return H

    def pos_definite(self, H: np.array, start: float, increment: float, max_iterations=100) -> np.array:
        """
        Function to secure that the Matrix H is definite positive.

        Args:
            H (np.array): Hessian matrix that is going to be checked
            start (float): Start lambda that has to be added in case H is not positive definite.
            increment (_type_): Increment in lamba for each iteration.
            max_iterations (int, optional): Maximum amount of iterations to check if H is positive definite. Defaults to 100.

        Returns:
            np.array: Positive definitie Hessian
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
        Calculate the score of a given pointcloud

        Args:
            points (float): pointcloud used to calculate the score.

        Returns:
            float: obtained score.
        """
        score = 0
        for point in points:
            point = point[:2]
            cell = self.get_cell(point[:2])
            if not cell is None and len(cell.points) > 2:
                score += cell.pdf(point)
        return score

    def compute_step_length(self, T: np.array, source: np.array, pose: Pose, curr_score: float) -> float:
        """
        Euristic way to calculate alpha.

        T -> T + alpha*delta_T

        Args:
            T (np.array): delta_T obtained with the newtons method.
            source (np.array): source pointcloud
            pose (Pose): current pose
            curr_score (float): current score

        Returns:
            float: obtained alpha
        """
        source = source[:, :2]
        T = T.copy()
        max_param = max(T[0, 0], max(T[1, 0], T[2, 0]))
        mlength = 1.0
        if max_param > 0.2:
            mlength = 0.1 / max_param
            T *= mlength

        best_alpha = 0

        # Try smaller steps
        alpha = 1.0
        for i in range(40):
            # print("Adjusting alpha smaller")
            adj_score = self.adjustment_score(alpha, T, source, pose)
            if adj_score > curr_score:
                best_alpha = alpha
                curr_score = adj_score
            alpha *= 0.7

        if best_alpha == 0:
            # Try larger steps
            alpha = 2.0
            for i in range(10):
                # print("Adjusting alpha bigger")
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
        score = 0
        T *= alpha
        p_cpy = Pose(0, 0, 0)
        p_cpy = pose + p_cpy
        p_cpy.x += T[0, 0]
        p_cpy.y += T[1, 0]
        p_cpy.yaw += T[2, 0]

        if p_cpy.yaw > 2 * np.pi:
            n = np.floor(p_cpy.yaw / 2 * np.pi)
            p_cpy.yaw -= n * (2 * np.pi)

        R, t = p_cpy.get_transformation()

        transformed_scan = R @ source[:, :2].T + t
        transformed_scan = transformed_scan.T

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
init_pose = Pose(0, 0, 0)  # Make a good initial guess

# Create NDT object
ndt = NDT(12, 12)  # Adjust grid resolution if necessary

# Set input cloud to populate grids
target_pcd = np.asarray(map_cloud.points)
ndt.set_input_cloud(target_pcd)

# Initialize lists for timing and errors
time_rec = []
lateral_err = []

# Create a KDTree for the target point cloud
target_kdtree = o3d.geometry.KDTreeFlann(map_cloud)

# Use pose from the previous iteration as the starting point for the next iteration
current_pose = init_pose

for i, car_cloud in enumerate(extracted_cars):
    start_time = time.time()

    source_pcd = np.asarray(car_cloud.points)
    current_pose, cache_list = ndt.align(source_pcd, current_pose, max_iterations=15, eps=1e-3)

    end_time = time.time()
    time_rec.append(end_time - start_time)

    # Transform the source point cloud using the calculated pose
    R, t = current_pose.get_transformation()
    transformed_points = R @ source_pcd[:, :2].T + t
    transformed_points = transformed_points.T

    # Convert transformed points to 3D by adding a z-coordinate of 0
    transformed_points_3d = np.hstack((transformed_points, np.zeros((transformed_points.shape[0], 1))))
    print('transformed_points_3d:', transformed_points_3d.shape)

    # Compute the registration error using nearest neighbor distances
    dists = []
    for point in transformed_points_3d:
        [_, idx, dist] = target_kdtree.search_knn_vector_3d(point, 1)
        dists.append(dist[0])
    reg_error = np.mean(dists)
    lateral_err.append(reg_error)

    if reg_error > 1.2:
        print(f"Lateral error ({reg_error:.2f} m) is greater than the maximum allowed (1.2 m).")
        # break

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