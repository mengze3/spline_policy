import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

# Create two rectangles that form a T shape, centered at origin
def create_tee_vertices(scale=30):
    length = 4
    # Horizontal bar
    vertices1 = np.array(
        [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
    )
    # Vertical bar
    vertices2 = np.array(
        [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
    )
    return vertices1, vertices2


# Apply rotation and translation to a set of vertices
def rotate_and_translate(vertices, angle_rad, translation):
    R = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    rotated = vertices @ R.T
    return rotated + np.array(translation)


# Return two matplotlib Polygon patches that represent a T-shape
def create_tee_patch(
    position=(0, 0), yaw=0, scale=30 / 512 * 0.4, color=(0.0, 1.0, 1.0, 0.7)
):
    vertices1, vertices2 = create_tee_vertices(scale)
    v1 = rotate_and_translate(vertices1, yaw, position)
    v2 = rotate_and_translate(vertices2, yaw, position)
    patch1 = Polygon(
        v1,
        closed=True,
        facecolor=color,
        edgecolor="lightgray",
        linewidth=1.5,
        zorder=10,
    )
    patch2 = Polygon(
        v2,
        closed=True,
        facecolor=color,
        edgecolor="lightgray",
        linewidth=1.5,
        zorder=10,
    )
    return patch1, patch2


if __name__ == "__main__":
    fig, ax = plt.subplots()

    position = (0, 0)
    yaw = np.deg2rad(0)
    scale = 30 / 512 * 0.4

    tee_patch1, tee_patch2 = create_tee_patch(position=position, yaw=yaw, scale=scale)

    ax.add_patch(tee_patch1)
    ax.add_patch(tee_patch2)
    ax.set_aspect("equal")

    x, y = position
    ax.set_xlim(x - 3 * scale, x + 3 * scale)
    ax.set_ylim(y - 3 * scale, y + 5 * scale)

    plt.grid(True)
    plt.show()
