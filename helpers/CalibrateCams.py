"""Estimates camera pose from calibration landmarks using solvePnP."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from helpers.utils_3d_reconstruction import CameraData, BeltPoints

class BasicCalibration:
    def __init__(self, calibration_coords):
        self.cameras = CameraData()
        # self.cameras_specs = self.cameras.specs
        self.cameras_intrinsics = self.cameras.intrinsic_matrices

        self.belt_pts = BeltPoints(calibration_coords)
        self.belt_coords_CCS = self.belt_pts.coords_CCS
        self.belt_coords_WCS = self.belt_pts.coords_WCS

    def estimate_cams_pose(self):
        cameras_extrinsics = self.cameras.compute_cameras_extrinsics(self.belt_coords_WCS, self.belt_coords_CCS)
        #self.print_reprojection_errors(cameras_extrinsics)
        return cameras_extrinsics

    def print_reprojection_errors(self, cameras_extrinsics, with_guess=False):
        if with_guess:
            print('Reprojection errors (w/ initial guess):')
        else:
            print('Reprojection errors:')
        for cam, data in cameras_extrinsics.items():
            print(f'{cam}: {data["repr_err"]}')

    def plot_cam_pose(self,cameras_extrinsics):
        #fig, ax = self.belt_pts.plot_WCS()

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(projection="3d")
        ax.set_proj_type('ortho')

        # helper to draw a 3D line
        def draw_line(p, q, **kw):
            ax.plot([p[0], q[0]],
                    [p[1], q[1]],
                    [p[2], q[2]],
                    **kw)

        # dimensions
        w, h, d = 940.0, 50.0, 100.0  # belt width, belt depth, box height
        box_dx = 100.0  # end‐box X extension

        # 1) big outer wireframe box
        X0, Y0, Z0 = -box_dx, 0.0, 0.0
        X1, Y1, Z1 = w + box_dx, h, d
        corners = np.array([
            [X0, Y0, Z0], [X1, Y0, Z0], [X1, Y1, Z0], [X0, Y1, Z0],
            [X0, Y0, Z1], [X1, Y0, Z1], [X1, Y1, Z1], [X0, Y1, Z1],
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            draw_line(corners[i], corners[j], color="black", lw=0.5)

        # 2) left and right end floors (black)
        for x0 in (-box_dx, w):
            floor_c = np.array([
                [x0, 0, Z0],
                [x0 + box_dx, 0, Z0],
                [x0 + box_dx, h, Z0],
                [x0, h, Z0],
            ])
            floor = Poly3DCollection([floor_c],
                                     facecolors="black",
                                     edgecolors="none",
                                     zorder=1)
            ax.add_collection3d(floor)

        # 3) belt floor (white)
        belt_c = np.array([
            [0, 0, Z0],
            [w, 0, Z0],
            [w, h, Z0],
            [0, h, Z0],
        ])
        belt = Poly3DCollection([belt_c],
                                facecolors="white",
                                edgecolors="black",
                                linewidths=0.5,
                                zorder=2)
        ax.add_collection3d(belt)

        # 4) belt mid‑line
        ax.plot([w / 2, w / 2], [0, h], [Z0, Z0],
                color="black", lw=0.5, zorder=100)

        shaft_length = 100.0
        cone_height = 35.0  # how tall the arrowhead is
        cone_radius = 14.0  # how fat the arrowhead is

        for cam in self.cameras.specs:
            R = cameras_extrinsics[cam]["rotm"]
            COB = R.T
            pos = (-COB @ cameras_extrinsics[cam]["tvec"]).flatten()

            ax.scatter(*pos, s=10, c="b", marker=".", lw=0.5, zorder=4)
            ax.text(pos[0] - 50, pos[1] - 50, pos[2] + 100, s=cam, c="b", zorder=4, fontsize=7)

            for vec, col in zip(R, ["r", "g", "b"]):
                # 1) normalize and compute the tip of the shaft
                vec_unit = vec / np.linalg.norm(vec)
                tip = pos + vec_unit * shaft_length

                # 2) draw the shaft for all three colors
                ax.plot([pos[0], tip[0]],
                        [pos[1], tip[1]],
                        [pos[2], tip[2]],
                        color=col, lw=0.5, zorder=4)

                # 3) only for the blue axis, add a 3D cone head
                if col == "b":
                    base = tip - vec_unit * cone_height
                    self.draw_cone(ax, base, tip,
                              radius=cone_radius,
                              resolution=16,
                              facecolors="blue",
                              edgecolors="none",
                              zorder=5)

        ax.set_xlabel("X (mm)", fontsize=7)
        ax.set_ylabel("Y (mm)", fontsize=7)
        ax.set_zlabel("Z (mm)", fontsize=7)
        # set tick fontsize
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='z', labelsize=7)
        ax.grid(False)

        # get azimuth and elevation
        azim = ax.azim
        # round to 1 decimal place
        azim = round(azim, 1)
        elev = ax.elev
        elev = round(elev, 1)

        fig.savefig(rf"H:\Dual-belt_APAs\Plots\Jan25\Characterisation\Tracking\cam_pose_az{azim}_el{elev}.svg",
                    format='svg',
                    bbox_inches='tight',  # trim whitespace
                    pad_inches=0)

    def draw_cone(ax, base, tip, radius=5, resolution=16, **kw):
        """
        Draws a cone pointing from base → tip with the given radius at the base.
        """
        # direction vector
        v = np.array(tip) - np.array(base)
        length = np.linalg.norm(v)
        if length == 0:
            return
        v = v / length

        # find two orthonormal vectors u, w perpendicular to v
        # pick an arbitrary vector not parallel to v
        not_v = np.array([1,0,0])
        if abs(np.dot(v, not_v)) > 0.9:
            not_v = np.array([0,1,0])
        u = np.cross(v, not_v)
        u /= np.linalg.norm(u)
        w = np.cross(v, u)

        # base circle
        angles = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        circle = [base + radius*(np.cos(a)*u + np.sin(a)*w) for a in angles]

        # build triangular faces (tip, circle[i], circle[i+1])
        faces = []
        for i in range(resolution):
            j = (i+1) % resolution
            faces.append([tuple(tip), tuple(circle[i]), tuple(circle[j])])

        cone = Poly3DCollection(faces, **kw)
        ax.add_collection3d(cone)


