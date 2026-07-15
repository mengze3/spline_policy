import numpy as np
from matplotlib.collections import LineCollection
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d


def gradient_colors(num_points):
    """Red -> purple -> blue gradient used throughout this repo's trajectory
    plots (uncertainty panels, flow-field renders, etc.)."""
    blue = np.array([0.4, 0.6, 1.0])
    purple = np.array([0.85, 0.75, 1.0])
    red = np.array([1.0, 0.2, 0.2])
    half = num_points // 2
    first_half = np.linspace(0, 1, half, endpoint=False)[:, None]
    colors1 = red + (purple - red) * first_half
    second_half = np.linspace(0, 1, num_points - half)[:, None]
    colors2 = purple + (blue - purple) * second_half
    return np.vstack((colors1, colors2))


def glow_line(ax, points, lw, alpha=0.15, glow_scale=2.3, zorder=1):
    """Draw a `gradient_colors`-shaded trajectory as a soft glow (thick,
    faint) plus a crisp centerline (thin, solid)."""
    segments = np.stack([points[:-1], points[1:]], axis=1)
    colors = gradient_colors(len(segments))
    ax.add_collection(
        LineCollection(segments, colors=colors, linewidths=lw * glow_scale, alpha=alpha, zorder=zorder)
    )
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=lw, zorder=zorder + 1))
    return colors


def integrate_trajectories(field, start_positions, dt=0.01, n_steps=200):
    """
    Perform RK4 trajectory integration from multiple start points in a 2D vector field.

    Parameters:
    - field: (N, 4) array, each row is [x, y, u, v]
    - start_positions: (M, 2) array of initial positions
    - dt: float, time step
    - n_steps: int, number of steps

    Returns:
    - trajectories: list of (L_i, 2) arrays, one for each trajectory
    """
    X, Y, U, V = np.hsplit(field, 4)
    X = X.squeeze()
    Y = Y.squeeze()
    U = U.squeeze()
    V = V.squeeze()

    # Extract unique grid coordinates
    x_unique = np.unique(X)
    y_unique = np.unique(Y)
    nx = len(x_unique)
    ny = len(y_unique)

    # Reshape to 2D velocity grids
    U_grid = U.reshape(ny, nx)
    V_grid = V.reshape(ny, nx)

    # Grid bounds
    x_min, x_max = x_unique[0], x_unique[-1]
    y_min, y_max = y_unique[0], y_unique[-1]

    def velocity_fn(pos):
        # pos: (2,) array
        x, y = pos
        ix = (x - x_min) / (x_max - x_min) * (nx - 1)
        iy = (y - y_min) / (y_max - y_min) * (ny - 1)
        coords = np.array([[iy], [ix]])
        vx = map_coordinates(U_grid, coords, order=1, mode="nearest")[0]
        vy = map_coordinates(V_grid, coords, order=1, mode="nearest")[0]
        return np.array([vx, vy])

    # Integrate each trajectory separately
    trajectories = []
    for start_pos in start_positions:
        current = np.array(start_pos, dtype=np.float64)
        traj = [current.copy()]
        for _ in range(n_steps):
            k1 = velocity_fn(current)
            if np.linalg.norm(k1) < 1e-6:
                break
            k2 = velocity_fn(current + 0.5 * dt * k1)
            k3 = velocity_fn(current + 0.5 * dt * k2)
            k4 = velocity_fn(current + dt * k3)
            v = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            current = current + dt * v
            if not (x_min <= current[0] <= x_max and y_min <= current[1] <= y_max):
                break
            traj.append(current.copy())
        trajectories.append(np.array(traj))

    return trajectories


def plot_trajectory_with_arrows(
    ax,
    traj,
    color="hotpink",
    linewidth=1.5,
    arrow_scale=0.5,
    zorder=2,
    alpha=1.0,
):
    """
    Plot a 2D trajectory with a single directional arrow at 1/4 of the path.

    Parameters:
    - ax: matplotlib axis
    - traj: (N, 2) array of points
    - color: line and arrow color
    - linewidth: line thickness
    - arrow_scale: scaling factor for arrow size
    """
    # Plot the trajectory line
    ax.plot(
        traj[:, 0],
        traj[:, 1],
        color=color,
        linewidth=linewidth,
        zorder=zorder,
        alpha=alpha,
    )

    if len(traj) < 2:
        return

    idx = int(len(traj) * 0.1)
    if idx >= len(traj) - 1:
        idx = len(traj) - 2  # avoid out of bounds

    # Get the direction vector at that point
    x = traj[idx, 0]
    y = traj[idx, 1]
    dx = traj[idx + 1, 0] - traj[idx, 0]
    dy = traj[idx + 1, 1] - traj[idx, 1]

    ax.quiver(
        x,
        y,
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        color=color,
        width=0.02,
        zorder=zorder,
        alpha=alpha,
    )


def interpolate_trajectory(trajectory, num_interpolated_points):
    num_inference_steps, nPoints, Da = trajectory.shape

    interpolated_trajectory = []

    for step in range(num_inference_steps):
        points = trajectory[step]
        interpolator = interp1d(np.arange(nPoints), points, axis=0, kind="linear")
        new_points = interpolator(np.linspace(0, nPoints - 1, num_interpolated_points))
        interpolated_trajectory.append(new_points)

    return np.array(interpolated_trajectory)


def plot_trajectory_with_shadow(
    ax, traj, zorder, markeredgewidth=2, markersize=10, linewidth=3
):
    ax.plot(
        traj[:, 0],
        traj[:, 1],
        marker="o",
        color="white",
        markerfacecolor="white",
        markeredgecolor="white",
        markeredgewidth=markeredgewidth,
        markersize=markersize,
        linewidth=linewidth,
        zorder=zorder,
        alpha=1.0,
    )

    ax.plot(
        traj[:, 0],
        traj[:, 1],
        marker="o",
        color="#1f77b4",
        markerfacecolor="#1f77b4",
        markeredgecolor="#1f77b4",
        markeredgewidth=markeredgewidth,
        markersize=markersize * 1.5,
        linewidth=linewidth * 2,
        zorder=zorder - 1,
        alpha=0.6,
    )

    return ax


def merge_trajs(traj_list, rtol=1e-5, atol=1e-8):
    unique_trajs = list()

    for traj in traj_list:
        is_duplicate = False
        for ut in unique_trajs:
            if traj.shape == ut.shape and np.allclose(traj, ut, rtol=rtol, atol=atol):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_trajs.append(traj)
    unique_trajs = np.array(unique_trajs)
    merged_traj = np.concatenate(
        [t[:, :2] for t in unique_trajs], axis=0
    )  # shape (T_total, 2)

    return merged_traj


def trajectory_with_start2end_label(ax, x, y, c):
    assert len(x) == len(y) == len(c)
    ax.scatter(x[0], y[0], c=c[0:1], marker="*", s=120, zorder=9)
    ax.scatter(x[0], y[0], c="white", marker="*", s=40, zorder=10)
    if len(x) > 2:
        ax.scatter(x[1:-1], y[1:-1], c=c[1:-1], marker="o", s=80, zorder=8)
    if len(x) > 1:
        ax.scatter(x[-1], y[-1], c=c[-1:], marker="^", s=120, zorder=9)
        ax.scatter(x[-1], y[-1], c="white", marker="^", s=40, zorder=10)
    return ax


def trajectory_with_gradient_color(ax, x, y, c, s=40):
    assert len(x) == len(y) == len(c)
    ax.scatter(x[:], y[:], c=c[:], marker="o", s=s, zorder=8)
    return ax
