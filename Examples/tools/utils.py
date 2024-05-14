import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, axes, colors
from pypcd4 import PointCloud
from scipy.stats import multivariate_normal
from typing import List, Tuple
from IPython.display import HTML


def pcd_from_path(file_path: str) -> np.ndarray:
    """
    Loads point clouds from PCD files using the PointCloud library (assumed to be open3d or similar).

    Parameters:
        file_path (str): Path to a .pcd file.

    Returns:
        np.ndarray: Numpy array representing the point cloud, shape [n_points, m_channels].
    
    Raises:
        ValueError: If the file format is not 'pcd'.
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.endswith(".pcd"):
        raise ValueError('Only ".pcd" format is accepted.')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    pc = PointCloud.from_path(file_path)
    return pc.numpy()


def plot_pcd(ax: axes.Axes, points: np.ndarray, **kwargs):
    """
    Plots point cloud data on a given matplotlib axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot the point cloud.
        points (np.ndarray): The point cloud data, expected shape [n_points, at least 2].
        **kwargs: Additional keyword arguments passed to matplotlib scatter plot.

    Raises:
        ValueError: If 'points' does not have the correct dimensions.
    """
    if points.ndim < 2 or points.shape[1] < 2:
        raise ValueError("The 'points' array must have at least two dimensions [n_points, at least 2].")

    ax.scatter(points[:, 0], points[:, 1], **kwargs)


def plot_grid(ax: axes.Axes, ndt, color: str = "blue", linestyle: str = '--'):
    """
    Plots a grid on the given axes corresponding to the NDT cell structure.

    Parameters:
    ax (matplotlib.axes.Axes): The matplotlib axes to plot on.
    ndt (NDT object): An object containing the bounding box and step sizes of the NDT grid.
    color (str, optional): Color of the grid lines. Default is "blue".
    color (str, optional): Linestyle of the grid lines. Default is "--".
    """
    x_min, y_min = ndt.bbox[0]
    x_max, y_max = ndt.bbox[1]
    num_voxels_x = int(np.ceil((x_max - x_min) / ndt.x_step))
    num_voxels_y = int(np.ceil((y_max - y_min) / ndt.y_step))
    xs = np.linspace(x_min, x_max, num_voxels_x)
    ys = np.linspace(y_min, y_max, num_voxels_y)

    # Plotting x lines
    for x in xs:
        ax.plot([x, x], [y_min, y_max], color=color, linestyle=linestyle)

    # Plotting y lines
    for y in ys:
        ax.plot([x_min, x_max], [y, y], color=color, linestyle=linestyle)


def plot_ndt(ax: axes.Axes, ndt, plot_points: bool = False):
    """
    Plots the results of a Normal Distribution Transform (NDT) on an axes.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        ndt (NDT object): The NDT object containing grid and bbox.
        plot_points (bool): If True, plots individual points within each cell.

    Returns:
        matplotlib.axes.Axes: The modified axes with the NDT plot.
    """
    # Unpack the bounding box and generate a grid
    x_min, y_min = ndt.bbox[0]
    x_max, y_max = ndt.bbox[1]
    xs = np.linspace(x_min, x_max, 1000)
    ys = np.linspace(y_min, y_max, 1000)
    X, Y = np.meshgrid(xs, ys)
    pos = np.stack([X.ravel(), Y.ravel()], axis=-1)
    Z = np.zeros(X.shape)

    # Evaluate the distribution across the grid
    for i, row in enumerate(ndt.grid):
        for j, cell in enumerate(row):
            if not cell.points.size:
                continue

            # Plot points if requested
            if plot_points:
                label = "Target or map" if i == len(ndt.grid)-1 and j == len(row)-1 else None
                plot_pcd(ax, cell.points, color="red", label=label)

            # Safely calculate the probability density function
            try:
                mean = cell.mean
                cov = cell.cov
                rv = multivariate_normal(mean, cov)
                Z += rv.pdf(pos).reshape(X.shape)
            except Exception as e:
                print(f"Error processing cell at ({i}, {j}): {e}")

    # Display the probability density
    im = ax.imshow(Z, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax)

    # Plot additional grid elements
    plot_grid(ax, ndt, color="blue")

    return ax


def animate_icp_results(P: np.ndarray,
                        Q: np.ndarray,
                        rotation_matrices: List[np.ndarray],
                        translations: List[np.ndarray],
                        correspondences: List[Tuple],
                        x_limits: List[float] | None = None,
                        y_limits: List[float] | None = None):
    """
    Animate the iterative closest point (ICP) results showing the alignment of two point sets.

    Args:
        P (np.ndarray): Source points to be aligned as a Nx3 numpy array.
        Q (np.ndarray): Target points as a Nx3 numpy array.
        rotation_matrices (list of np.ndarray): List of 3x3 rotation matrices applied to the source points.
        translations (list of np.ndarray): List of 3x1 translation vectors applied to the source points.
        correspondences (list of tuples): List of correspondence pairs for each iteration.
        x_limits (tuple): X-axis limits for the plot as (min, max).
        y_limits (tuple): Y-axis limits for the plot as (min, max).

    Returns:
        HTML: Animation of the ICP process in HTML format for IPython display.
    """
    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set figure limits
    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.set_aspect('equal')
    ax.set_title('ICP Animation - Iteration 0')
    ax.grid(True)

    P = P.T
    Q = Q.T
    # Unpack initial positions
    x_source, y_source, _ = P
    x_target, y_target, _ = Q

    # Prepare initial scatter plots
    scatter_source = ax.scatter(x_source, y_source, color='blue', label='Source Points')
    scatter_target = ax.scatter(x_target, y_target, color='orangered', label='Target Points')
    ax.legend()

    # Lines for correspondence visualization
    lines = [ax.plot([], [], 'grey', lw=0.5)[0] for _ in range(len(correspondences[0]))]

    def update(frame):
        """Update the plot for the animation."""
        ax.set_title(f'ICP Animation - Iteration {frame + 1}')

        # Apply transformations
        moving_transformed = rotation_matrices[frame] @ P + translations[frame]
        scatter_source.set_offsets(np.c_[moving_transformed[0, :], moving_transformed[1, :]])

        # Update correspondence lines
        for line, (i, j) in zip(lines, correspondences[frame]):
            line.set_data([Q[0, j], moving_transformed[0, i]], 
                          [Q[1, j], moving_transformed[1, i]])

        return scatter_target, *lines

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(rotation_matrices)-1, interval=500, blit=True)

    # Close the plot to avoid double display when used in notebooks
    plt.close(fig)
    
    # Return animation as HTML
    return HTML(anim.to_jshtml())

def animate_ndt_results(P, ndt, init_pose, delta_T_list, xlim, ylim):
    """A function used to animate the iterative processes we use."""
    pose = init_pose
    source_pcd = P.copy()
    fig = plt.figure(figsize=(10, 6))
    anim_ax = fig.add_subplot(111)
    
    anim_ax.set_aspect("equal")
    text = anim_ax.set_title("Iteration 0")
    score_text = anim_ax.text(0.05, 1.0, 'Title', transform=anim_ax.transAxes,
        fontsize=12, fontweight='bold', va='top')
    anim_ax = plot_ndt(anim_ax, ndt, plot_points=True)
    plot_pcd(anim_ax, P, color="green", label = "Source or scan")
    (T_line,) = anim_ax.plot(P[:, 0], P[:, 1],"o", color="yellow", label="Transformed source")
    anim_ax.set(xlim=xlim, ylim=ylim)
    anim_ax.legend()
    plt.close()

    def animate(frame):
        R, t, score = delta_T_list[frame]
        transformed_pcd = R@source_pcd[:,:2].T+t
        P = transformed_pcd.T 
        T_line.set_data(P[:,0], P[:,1])
        text.set_text(f"Iteration {frame}")
        score_text.set_text(f"Score: {score:.4f}")
        return (T_line,)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(delta_T_list) - 1, interval=500, blit=True
    )
    return HTML(anim.to_jshtml())

def animate_markov_loc_results_1D(posteriors_list: List[np.ndarray], xs: np.ndarray, xlim: List[float], ylim: List[float]):
    """
    Animates the results of a Markov localization process.

    Args:
        posteriors_list (list of np.array): List of posterior distributions over time.
        xs (list): List of x positions corresponding to the posteriors.
        xlim (list): Limits for the x-axis.
        ylim (list): Limits for the y-axis.

    Returns:
        HTML: HTML representation of the animation for IPython display.
    """
    if not posteriors_list:
        raise ValueError("posteriors_list must not be empty")
    
    for frame, posterior in enumerate(posteriors_list):
        if len(posterior) != len(xs):
            raise ValueError(f"Size mistmatch at frame {frame} posterior size []")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # Initial drawing
    bars = ax.bar(xs, posteriors_list[0].flatten(), animated=True)
    ax.set_title("Iteration 0")

    def animate(frame):
        # Update the bar heights instead of redrawing them
        for bar, height in zip(bars, posteriors_list[frame].flatten()):
            bar.set_height(height)
        ax.set_title(f"Iteration {frame}")
        return bars

    anim = animation.FuncAnimation(fig, animate, frames=len(posteriors_list), interval=500, blit=True)
    
    plt.close(fig)  # Prevent the static plot from being displayed in the notebook
    return HTML(anim.to_jshtml())


def animate_markov_loc_results_2D(posteriors_list: List[np.ndarray], xlim: List[float], ylim: List[float]):
    """
    Animates the results of a Markov localization process in 2D with a dynamically updating color bar.

    Args:
        posteriors_list (list of np.array): List of posterior distributions over time.
        xlim (list): Limits for the x-axis.
        ylim (list): Limits for the y-axis.

    Returns:
        HTML: HTML representation of the animation for IPython display.
    """
    if not posteriors_list:
        raise ValueError("posteriors_list must not be empty")

    # Determine color scale limits for consistent coloring across frames
    vmin = min([np.min(frame) for frame in posteriors_list])
    vmax = max([np.max(frame) for frame in posteriors_list])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Initial drawing
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    heatmap = ax.imshow(posteriors_list[0], norm=norm, origin="lower", 
                        extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='viridis')
    ax.set_title("Iteration 0")

    # Create a color bar
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Probability Density')

    def animate(frame):
        # Update the heatmap data
        heatmap.set_data(posteriors_list[frame])
        ax.set_title(f"Iteration {frame}")
        return (heatmap,)

    anim = animation.FuncAnimation(fig, animate, frames=len(posteriors_list), interval=500, blit=True)
    
    plt.close(fig)  # Prevent the static plot from being displayed in the notebook
    return HTML(anim.to_jshtml())