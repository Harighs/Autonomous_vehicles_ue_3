import numpy as np
from scipy.spatial import KDTree
from typing import Tuple

class ICP:
    """
    Implementation of an ICP class
    """

    def __init__(
        self, max_iterations: int = 50, max_dist: float = 10, tol: float = 1e-5
    ) -> None:
        """
        Args:
            max_iterations (int, optional): Maximum iterations for the ICP algorithm. Defaults to 50.
            max_dist (float, optional): Max distance to cosider coincidences. Defaults to 10.
            tol (float, optional): Maximum tolerance to check if the algorithm converged. Defaults to 1e-5.
        """
        self.max_iterations = max_iterations
        self.max_dist = max_dist
        self.tol = tol

    def align(
        self, source: np.array, target: np.array, method: str = "svd"
    ) -> Tuple[np.array, np.array]:
        """
        Align function to compute the transformation from source to target. This function can be extended use different
        methods rather than only ICP.

        Args:
            source (np.array): Pointcloud to be aligned
            target (np.array): Base Pointcloud, in localization, this is the map
            method (str, optional): Method to perform ICP. Defaults to "svd".

        Raises:
            ValueError: If method is different to the ones programmed.

        Returns:
            Tuple[np.array, np.array]: Returns the rotation matrix and translation from source to target.
            Also returns a list with all Rotations and translations for each iteration in ICP.
        """
        if method == "svd":
            return self._align_svd(source, target)
        else:
            raise ValueError(f"Method {method} is not supported.")

    def _align_svd(
        self, source: np.array, target: np.array
    ) -> Tuple[np.array, np.array]:
        """
        Align function to compute the transformation from source to target using Single Value Decomposition.

        Args:
            source (np.array): Pointcloud to be aligned
            target (np.array): Base Pointcloud, in localization, this is the map

        Returns:
            Tuple[np.array, np.array]: Returns the rotation matrix and translation from source to target.
            Also returns a list with all Rotations and translations for each iteration in ICP.

        """

        mean_target = np.mean(target, axis=0)
        centered_target = target - mean_target
        tree = KDTree(centered_target)

        source_copy = source.copy()

        R = np.eye(target.shape[1])
        t = np.zeros((target.shape[1], 1))

        R_list = [R]
        t_list = [t]
        corres_values = []

        for iteration in range(self.max_iterations):
            # Center source
            mean_source = np.mean(source_copy, axis=0)
            centered_source = source_copy - mean_source
            # Find nearest neighbors
            distances, indices = tree.query(centered_source)
            # Compute correspondences
            correspondences = np.asarray([(i, j) for i, j in enumerate(indices)])
            mask = distances < self.max_dist
            # Filter correspondences
            correspondences = correspondences[mask, :]
            distances = distances[mask]
            #  Compute covariance matrix
            E = np.dot(
                (centered_source[correspondences[:, 0]]).T,
                centered_target[correspondences[:, 1]],
            )

            # Singular value decomposition
            U, S, Vt = np.linalg.svd(E)
            # Compute rotation and translation
            Rn = np.dot(Vt.T, U.T)
            tn = mean_target.reshape((target.shape[1], 1)) - np.dot(
                Rn, mean_source.reshape((target.shape[1], 1))
            )

            # Apply transform to point cloud
            source_copy = Rn.dot(source_copy.T) + tn
            source_copy = source_copy.T

            # Update transformation
            t = Rn @ t + tn
            R = np.dot(R, Rn)
            t_list.append(t.copy())
            R_list.append(R.copy())
            corres_values.append(correspondences.copy())

            if np.allclose(tn, 0, atol=self.tol) and np.allclose(Rn, np.eye(Rn.shape[0]), atol=self.tol):
                break

        return R, t, R_list, t_list, corres_values