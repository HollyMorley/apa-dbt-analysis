"""Plots of camera positions and calibration quality."""
import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams['svg.fonttype'] = 'none'

cameras_extrinsics = {
    "side": {
        "tvec": np.array([[-298.85353394], [65.67187339], [1071.78906513]]),
        "rotm": np.array(
            [
                [0.9999789, -0.00207372, 0.00615665],
                [0.00621094, 0.02727888, -0.99960857],
                [0.00190496, 0.99962571, 0.02729118],
            ]
        ),
    },
    "front": {
        "tvec": np.array([[-76.42235183], [18.56898049], [1243.26951668]]),
        "rotm": np.array(
            [
                [0.03650804, 0.99931535, -0.00600009],
                [0.00385228, -0.00614478, -0.9999737],
                [-0.99932593, 0.03648397, -0.00407397],
            ]
        ),
    },
    "overhead": {
        "tvec": np.array([[-201.40483901], [272.68542377], [2188.82953675]]),
        "rotm": np.array(
            [
                [0.9987034, 0.00421961, -0.05073173],
                [-0.00395296, -0.98712181, -0.15992155],
                [-0.0507532, 0.15991474, -0.98582523],
            ]
        ),
    },
}


def add_floor(ax, x0, x1, y_max, **kwargs):
    """Add floor surface to mock travelator"""
    verts = np.array([[x0, 0, 0], [x1, 0, 0], [x1, y_max, 0], [x0, y_max, 0]])
    ax.add_collection3d(Poly3DCollection([verts], **kwargs))


def draw_camera_frustum(
    ax, R, t, scale=200, aspect=4 / 3, color="gray", alpha=0.15
):
    """Draw a camera as a pyramid (frustum).

    Parameters
    ----------
    ax: matplotlib Axes3D
        Axes to draw the camera on.
    R : np.ndarray, shape (3, 3)
        Rotation matrix (world to camera).
    t : np.ndarray, shape (3, 1)
        Translation vector (in camera coordinates).
    scale : float
        Depth of the frustum in mm.
    aspect : float
        Width/height ratio of the image plane.
    color : str
        Face colour of the frustum. By default, 'gray'.
    alpha : float
        Opacity of the frustum. By default, 0.15.

    """
    # Compute four far-plane corners in camera space
    hw, hh = scale * aspect / 2, scale / 2  # half-width / half-height
    cam_corners = np.array(
        [
            [-hw, -hh, scale],
            [hw, -hh, scale],
            [hw, hh, scale],
            [-hw, hh, scale],
        ]
    )

    # Compute four far-plane corners in world space
    world_corners = (R.T @ (cam_corners.T - t)).T  # shape (4, 3)

    # Build faces: 4 triangular sides + rectangular far face
    pos = (-R.T @ t).flatten()  # camera centre in world coords
    faces = [
        [pos, world_corners[i], world_corners[(i + 1) % 4]] for i in range(4)
    ]
    faces.append(world_corners)  # far face

    # Add the frustum as a translucent Poly3DCollection
    poly = Poly3DCollection(
        faces,
        alpha=alpha,
        facecolor=color,
        edgecolor="k",
        linewidth=1.0,
    )
    ax.add_collection3d(poly)


# --- Figure ---

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.set_proj_type("ortho")

# --- Draw captured volume ---

# Travelator dimensions (mm)
w, h, d = 940.0, 100.0, 50.0  # belt width, box height, belt depth
box_dx = 100.0  # end‐platforms X extension

# Plot wireframe box
xs = [-box_dx, w + box_dx]
ys = [0, d]
zs = [0, h]
corners = np.array([[x, y, z] for z in zs for y in ys for x in xs])
edges = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),  # bottom
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),  # top
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # verticals
]
for i, j in edges:
    ax.plot(*corners[[i, j]].T, color="black", lw=0.5, zorder=110)


# Plot belt and end platform surfaces
# # Plot black end-platforms
# for x0 in (-box_dx, w):
#     add_floor(
#         ax, x0, x0 + box_dx, d, facecolors="white", edgecolors="none", zorder=1
#     )

# --- Draw belts ---
# Plot belts
add_floor(
    ax,
    0,
    w,
    d,  # white belt
    facecolors="white",
    edgecolors="red",
    linewidths=1.5,
    zorder=2,
)

# Plot belts division line
ax.plot([w / 2] * 2, [0, d], [0, 0], color="red", lw=1.5, zorder=100)

# --- Plot cameras in world coordinate system ---
# For each camera, invert the extrinsic parameters to recover the
# camera's position in world coordinates, then draw its local axes.

shaft_length = 100.0  # length of camera orientation arrows (mm)
axis_colours = ["r", "g", "b"]
for cam, ext in cameras_extrinsics.items():
    R = ext["rotm"]  # rotation matrix (world to camera)
    t = ext["tvec"]  # translation vector (in camera coordinates)
    pos = (-R.T @ t).flatten()  # camera position in world coordinates

    # draw local camera axes as coloured arrows
    for vec, col in zip(R, axis_colours):
        direction = vec / np.linalg.norm(vec) * shaft_length
        ax.quiver(
            *pos, *direction, color=col, arrow_length_ratio=0.5, length=1.7
        )

    # draw camera as a translucent pyramid
    draw_camera_frustum(ax, R, t, scale=150, color="gray")

    # add camera label
    ax.text(pos[0] - 50, pos[1] - 50, pos[2] + 150, s=cam, c="b", fontsize=12)


#  --- Format axes ---
ax.set(
    xlim=(-200, 1400),
    ylim=(-1000, 200),
    zlim=(-500, 2000),
    xlabel="X (mm)",
    ylabel="Y (mm)",
    zlabel="Z (mm)",
)

# ax.set_aspect("equal")
x_range = 1400 - (-200)  # 1600
y_range = 200 - (-1000)  # 1200
z_range = 2000 - (-500)  # 2500
ax.set_box_aspect((x_range, y_range, z_range))

ax.tick_params(labelsize=9)
ax.locator_params(nbins=4)
ax.grid(False)
ax.view_init(elev=20, azim=-30)
fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.99)

savedir = r"H:\Characterisation_v2\CamsPoses"
os.makedirs(savedir, exist_ok=True)

fig.savefig(os.path.join(savedir, "cams_pose.png"), dpi=400)
fig.savefig(os.path.join(savedir, "cams_pose.svg"), dpi=400)