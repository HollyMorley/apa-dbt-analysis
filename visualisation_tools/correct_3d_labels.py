"""Interactive tool for correcting 3D DLC tracking errors.

Projects 3D coordinates into all three camera views (side, front, overhead),
lets the user click to place bodyparts at correct positions, re-triangulates,
and saves corrections back to HDF5.

Controls:
    Left/Right    -- navigate frames (+/-1, hold Shift for +/-10)
    D             -- delete (NaN) the selected bodypart for this frame
    Z             -- undo last correction
    S             -- save corrections to file
    R             -- reset zoom/pan on all views
    Scroll        -- zoom in/out on hovered subplot (persists across frames)
    Middle drag   -- pan (persists across frames)
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from PIL import Image, ImageEnhance

matplotlib.rcParams.update({'font.size': 8, 'font.family': 'Arial'})

from helpers.config import micestuff
from helpers.utils_3d_reconstruction import CameraData

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# -------------------- Camera extrinsics (calibrated) --------------------

EXTRINSICS = {
    'side': {
        'tvec': np.array([[-298.85353394], [65.67187339], [1071.78906513]]),
        'rotm': np.array([
            [0.9999789, -0.00207372, 0.00615665],
            [0.00621094, 0.02727888, -0.99960857],
            [0.00190496, 0.99962571, 0.02729118],
        ]),
    },
    'front': {
        'tvec': np.array([[-76.42], [18.57], [1243.27]]),
        'rotm': np.array([
            [0.03650804, 0.99931535, -0.00600009],
            [0.00385228, -0.00614478, -0.9999737],
            [-0.99932593, 0.03648397, -0.00407397],
        ]),
    },
    'overhead': {
        'tvec': np.array([[-201.40], [272.69], [2188.83]]),
        'rotm': np.array([
            [0.9987034, 0.00421961, -0.05073173],
            [-0.00395296, -0.98712181, -0.15992155],
            [-0.0507532, 0.15991474, -0.98582523],
        ]),
    },
}

CAMERAS = ['side', 'front', 'overhead']


# -------------------- Projection helpers --------------------

def project_points(pts_3d, K, R, t):
    """Project Nx3 world points to Nx2 pixel coordinates using cv2.projectPoints."""
    pts = np.asarray(pts_3d, dtype=np.float64).reshape(-1, 1, 3)
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(pts, rvec, t, K, distCoeffs=np.array([]))
    return projected.reshape(-1, 2)


def get_projection_matrix(K, R, t):
    """Return 3x4 projection matrix P = K @ [R|t]."""
    t_col = t.reshape(3, 1)
    return K @ np.hstack([R, t_col])


def triangulate_from_views(pts_2d_dict, intrinsics, extrinsics):
    """Triangulate a 3D point from 2D observations in multiple views.

    Parameters
    ----------
    pts_2d_dict : dict
        {cam_name: (x, y)} for each camera with an observation.
    intrinsics : dict
        {cam_name: K} intrinsic matrices.
    extrinsics : dict
        {cam_name: {'rotm': R, 'tvec': t}} extrinsic parameters.

    Returns
    -------
    np.ndarray, shape (3,) -- triangulated 3D point.
    float -- mean reprojection error in pixels.
    """
    cams = list(pts_2d_dict.keys())
    if len(cams) < 2:
        return None, np.inf

    P_mats = []
    pts_2d_list = []
    for cam in cams:
        P = get_projection_matrix(intrinsics[cam], extrinsics[cam]['rotm'], extrinsics[cam]['tvec'])
        P_mats.append(P)
        pts_2d_list.append(pts_2d_dict[cam])

    # Use first two cameras for cv2.triangulatePoints
    pts1 = np.array(pts_2d_list[0], dtype=np.float64).reshape(2, 1)
    pts2 = np.array(pts_2d_list[1], dtype=np.float64).reshape(2, 1)
    pt_4d = cv2.triangulatePoints(P_mats[0], P_mats[1], pts1, pts2)
    pt_3d = (pt_4d[:3] / pt_4d[3]).flatten()

    # If more than 2 cameras, refine with DLT using all views
    if len(cams) > 2:
        # Build the DLT system
        A = []
        for i, cam in enumerate(cams):
            P = P_mats[i]
            x, y = pts_2d_list[i]
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        pt_3d = (X[:3] / X[3])

    # Compute reprojection error
    errors = []
    for i, cam in enumerate(cams):
        proj = project_points(pt_3d.reshape(1, 3), intrinsics[cam],
                              extrinsics[cam]['rotm'], extrinsics[cam]['tvec'])
        err = np.linalg.norm(proj.flatten() - np.array(pts_2d_list[i]))
        errors.append(err)
    mean_err = np.mean(errors)

    return pt_3d, mean_err


# -------------------- Main application --------------------

class Correct3DLabels:
    """Tkinter application for correcting 3D DLC labels across camera views."""

    def __init__(self, root, video_paths, coords_file, frame_start=0, frame_end=None):
        self.root = root
        self.root.title("3D Label Correction Tool")

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Camera data
        cam = CameraData(basic=True)
        self.intrinsics = {c: cam.intrinsic_matrices[c] for c in CAMERAS}
        self.extrinsics = EXTRINSICS

        # Bodyparts and skeleton
        self.bodyparts = micestuff['bodyparts']
        self.skeleton_pairs = micestuff['skeleton']

        # Load 3D coordinates — use corrected version if it exists
        base, ext = os.path.splitext(coords_file)
        corrected_file = base + '_corrected' + ext
        if os.path.exists(corrected_file):
            self.coords_file = corrected_file
            print(f"Loading corrected file: {corrected_file}")
        else:
            self.coords_file = coords_file
        self.coords_df = pd.read_hdf(self.coords_file)
        if (self.coords_df.columns.nlevels == 3
                and self.coords_df.columns.names == ['scorer', 'bodyparts', 'coords']):
            self.coords_df.columns = self.coords_df.columns.droplevel('scorer')

        # Frame range
        self.frame_start = frame_start
        self.frame_end = frame_end if frame_end is not None else len(self.coords_df)
        self.frame_end = min(self.frame_end, len(self.coords_df))
        self.current_frame = self.frame_start

        # Load video captures
        self.caps = {}
        self.video_available = {}
        for c in CAMERAS:
            path = video_paths.get(c, '')
            if path and os.path.exists(path):
                self.caps[c] = cv2.VideoCapture(path)
                self.video_available[c] = True
            else:
                self.caps[c] = None
                self.video_available[c] = False

        # UI state
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.brightness_var = tk.DoubleVar(value=0.0)
        self.marker_size_var = tk.DoubleVar(value=6.0)

        self.selected_bodypart = None
        self.panning = False
        self.pan_start = None

        # Camera view checkboxes — which views to use for triangulation
        self.cam_enabled = {c: tk.BooleanVar(value=True) for c in CAMERAS}

        # Store zoom/pan state per camera so it persists across redraws
        self.view_limits = {cam: None for cam in CAMERAS}  # None = auto/reset

        # Undo stack: list of (frame_idx, bodypart_name, old_xyz_or_None)
        self.undo_stack = []

        # Colors (viridis)
        cmap = plt.get_cmap('viridis')
        n = len(self.bodyparts)
        self.bp_colors = {}
        self.bp_colors_hex = {}
        for i, bp in enumerate(self.bodyparts):
            rgba = cmap(i / max(n - 1, 1))
            self.bp_colors[bp] = rgba[:3]
            self.bp_colors_hex[bp] = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

        self.setup_ui()

        # Bind keys
        self.root.bind('<Left>', self.on_key_left)
        self.root.bind('<Right>', self.on_key_right)
        self.root.bind('<Shift-Left>', self.on_key_shift_left)
        self.root.bind('<Shift-Right>', self.on_key_shift_right)
        self.root.bind('<d>', self.on_key_delete)
        self.root.bind('<D>', self.on_key_delete)
        self.root.bind('<z>', self.on_key_undo)
        self.root.bind('<Z>', self.on_key_undo)
        self.root.bind('<s>', self.on_key_save)
        self.root.bind('<S>', self.on_key_save)
        self.root.bind('<r>', self.on_key_reset_zoom)
        self.root.bind('<R>', self.on_key_reset_zoom)

        self.display_frame()

    # -------------------- UI setup --------------------

    def setup_ui(self):
        """Build the full UI layout."""
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top control bar
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=3)

        # Marker size
        tk.Label(control_frame, text="Marker Size", font=("Helvetica", 8)).pack(side=tk.LEFT, padx=3)
        tk.Scale(control_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                 resolution=1, variable=self.marker_size_var,
                 command=lambda _: self.display_frame(), length=120,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=3)

        # Contrast
        tk.Label(control_frame, text="Contrast", font=("Helvetica", 8)).pack(side=tk.LEFT, padx=3)
        tk.Scale(control_frame, from_=0.2, to=3.0, orient=tk.HORIZONTAL,
                 resolution=0.1, variable=self.contrast_var,
                 command=lambda _: self.display_frame(), length=120,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=3)

        # Brightness
        tk.Label(control_frame, text="Brightness", font=("Helvetica", 8)).pack(side=tk.LEFT, padx=3)
        tk.Scale(control_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                 resolution=5, variable=self.brightness_var,
                 command=lambda _: self.display_frame(), length=120,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=3)

        # Camera view checkboxes — which views to use for triangulation
        sep = tk.Frame(control_frame, width=2, bg='grey70')
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        tk.Label(control_frame, text="Use views:", font=("Helvetica", 8, "bold")).pack(side=tk.LEFT, padx=3)
        for cam in CAMERAS:
            tk.Checkbutton(control_frame, text=cam.capitalize(),
                           variable=self.cam_enabled[cam],
                           font=("Helvetica", 8)).pack(side=tk.LEFT, padx=2)

        # Frame counter
        frame_nav = tk.Frame(control_frame)
        frame_nav.pack(side=tk.LEFT, padx=15)

        self.frame_label = tk.Label(frame_nav,
                                    text=f"Frame: {self.current_frame}/{self.frame_end - 1}",
                                    font=("Helvetica", 9, "bold"))
        self.frame_label.pack()

        btn_frame = tk.Frame(frame_nav)
        btn_frame.pack()
        tk.Button(btn_frame, text="<<10", command=lambda: self.go_to_frame(-10)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="<", command=lambda: self.go_to_frame(-1)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text=">", command=lambda: self.go_to_frame(1)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text=">>10", command=lambda: self.go_to_frame(10)).pack(side=tk.LEFT, padx=2)

        # Save button
        tk.Button(control_frame, text="Save (S)", command=self.save_corrections,
                  font=("Helvetica", 9, "bold"), bg="#4CAF50", fg="white").pack(side=tk.RIGHT, padx=10)

        # Undo button
        tk.Button(control_frame, text="Undo (Z)", command=lambda: self.on_key_undo(None),
                  font=("Helvetica", 9, "bold"), bg="#f44336", fg="white").pack(side=tk.RIGHT, padx=5)

        # Reset zoom button
        tk.Button(control_frame, text="Reset Zoom (R)", command=self.reset_zoom,
                  font=("Helvetica", 8)).pack(side=tk.RIGHT, padx=5)

        # Right panel: bodypart list for selection
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=3, pady=1)

        label_canvas = tk.Canvas(right_panel, width=220)
        label_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        label_scrollbar = tk.Scrollbar(right_panel, orient=tk.VERTICAL, command=label_canvas.yview)
        label_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        label_canvas.configure(yscrollcommand=label_scrollbar.set)

        label_frame = tk.Frame(label_canvas, width=220)
        label_canvas.create_window((0, 0), window=label_frame, anchor="nw")
        label_frame.bind("<Configure>",
                         lambda e: label_canvas.configure(scrollregion=label_canvas.bbox("all")))

        # Order bodyparts by group: head/ears, back, tail, then each limb together
        self.bp_display_order = [
            'Nose', 'EarL', 'EarR',
            'Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6',
            'Back7', 'Back8', 'Back9', 'Back10', 'Back11', 'Back12',
            'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6',
            'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12',
            'ForepawToeL', 'ForepawKnuckleL', 'ForepawAnkleL', 'ForepawKneeL',
            'ForepawToeR', 'ForepawKnuckleR', 'ForepawAnkleR', 'ForepawKneeR',
            'HindpawToeL', 'HindpawKnuckleL', 'HindpawAnkleL', 'HindpawKneeL',
            'HindpawToeR', 'HindpawKnuckleR', 'HindpawAnkleR', 'HindpawKneeR',
        ]
        # Add any bodyparts from config that aren't in the display order
        for bp in self.bodyparts:
            if bp not in self.bp_display_order:
                self.bp_display_order.append(bp)

        self.bp_var = tk.StringVar(value='')
        self.label_buttons = {}

        # Group headers for visual separation
        group_starts = {
            'Nose': '--- Head ---',
            'Back1': '--- Back ---',
            'Tail1': '--- Tail ---',
            'ForepawToeL': '--- Forepaw L ---',
            'ForepawToeR': '--- Forepaw R ---',
            'HindpawToeL': '--- Hindpaw L ---',
            'HindpawToeR': '--- Hindpaw R ---',
        }

        for bp in self.bp_display_order:
            if bp not in self.bodyparts:
                continue
            # Add group header if needed
            if bp in group_starts:
                header = tk.Label(label_frame, text=group_starts[bp],
                                  font=("Helvetica", 8, "bold"), anchor='w', padx=4)
                header.pack(fill=tk.X, pady=(4, 0))
            color = self.bp_colors_hex[bp]
            btn = tk.Radiobutton(label_frame, text=bp, variable=self.bp_var, value=bp,
                                 indicatoron=0, width=22, bg=color, font=("Helvetica", 9),
                                 anchor='w', padx=6,
                                 command=lambda b=bp: self._select_bodypart(b))
            btn.pack(fill=tk.X, pady=1)
            self.label_buttons[bp] = btn

        # Matplotlib figure with 3 subplots
        self.fig, self.axs = plt.subplots(3, 1, figsize=(12, 9))
        self.fig.subplots_adjust(left=0.02, right=0.998, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)

        for ax in self.axs:
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.set_xticks([])
            ax.set_yticks([])

        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Connect matplotlib events — click to place, scroll to zoom, middle to pan
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        # Status bar
        self.status_var = tk.StringVar(value="Select a bodypart from the list, then click to place it")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN,
                              anchor=tk.W, font=("Helvetica", 8))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # -------------------- Frame reading --------------------

    def read_video_frame(self, cam, frame_idx):
        """Read a single frame from the given camera's video capture."""
        cap = self.caps.get(cam)
        if cap is None or not self.video_available[cam]:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    def apply_contrast_brightness(self, frame):
        """Apply contrast and brightness adjustments to a BGR frame."""
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast)
        img_arr = np.array(pil_img, dtype=np.float32)
        img_arr = np.clip(img_arr + brightness, 0, 255).astype(np.uint8)
        return img_arr

    # -------------------- 3D coordinate access --------------------

    def get_3d_coords(self, frame_idx, bodypart):
        """Get (x, y, z) for a bodypart at a frame, or None if NaN."""
        try:
            x = self.coords_df.loc[frame_idx, (bodypart, 'x')]
            y = self.coords_df.loc[frame_idx, (bodypart, 'y')]
            z = self.coords_df.loc[frame_idx, (bodypart, 'z')]
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                return None
            return np.array([x, y, z])
        except (KeyError, IndexError):
            return None

    def set_3d_coords(self, frame_idx, bodypart, xyz):
        """Set (x, y, z) for a bodypart at a frame. xyz=None sets NaN."""
        if xyz is None:
            self.coords_df.loc[frame_idx, (bodypart, 'x')] = np.nan
            self.coords_df.loc[frame_idx, (bodypart, 'y')] = np.nan
            self.coords_df.loc[frame_idx, (bodypart, 'z')] = np.nan
        else:
            self.coords_df.loc[frame_idx, (bodypart, 'x')] = xyz[0]
            self.coords_df.loc[frame_idx, (bodypart, 'y')] = xyz[1]
            self.coords_df.loc[frame_idx, (bodypart, 'z')] = xyz[2]

    # -------------------- Display --------------------

    def display_frame(self):
        """Render all three camera views with projected skeleton overlay."""
        fi = self.current_frame
        marker_size = self.marker_size_var.get()

        # Save current zoom/pan limits before clearing
        for ax_idx, cam in enumerate(CAMERAS):
            ax = self.axs[ax_idx]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if xlim != (0.0, 1.0) and ylim != (0.0, 1.0):
                self.view_limits[cam] = (xlim, ylim)

        for ax_idx, cam in enumerate(CAMERAS):
            ax = self.axs[ax_idx]
            ax.cla()
            ax.set_xticks([])
            ax.set_yticks([])

            # Read and display video frame
            frame_img = self.read_video_frame(cam, fi)
            if frame_img is not None:
                frame_img = self.apply_contrast_brightness(frame_img)
                ax.imshow(frame_img)
            else:
                ax.set_facecolor('#333333')

            ax.set_title(f"{cam.capitalize()} View", fontsize=8, pad=2)

            K = self.intrinsics[cam]
            R = self.extrinsics[cam]['rotm']
            t = self.extrinsics[cam]['tvec']

            # Draw skeleton connections first (so dots are on top)
            for bp1_name, bp2_name in self.skeleton_pairs:
                pt1 = self.get_3d_coords(fi, bp1_name)
                pt2 = self.get_3d_coords(fi, bp2_name)
                if pt1 is not None and pt2 is not None:
                    proj1 = project_points(pt1.reshape(1, 3), K, R, t).flatten()
                    proj2 = project_points(pt2.reshape(1, 3), K, R, t).flatten()
                    ax.plot([proj1[0], proj2[0]], [proj1[1], proj2[1]],
                            color='black', linewidth=0.8, alpha=0.6, zorder=1)

            # Draw bodypart dots
            for bp in self.bodyparts:
                pt_3d = self.get_3d_coords(fi, bp)
                if pt_3d is None:
                    continue
                proj = project_points(pt_3d.reshape(1, 3), K, R, t).flatten()
                color = self.bp_colors[bp]

                ax.scatter(proj[0], proj[1], s=marker_size ** 2, c=[color],
                           edgecolors='none', zorder=2)

                # Highlight selected bodypart with white ring
                if bp == self.selected_bodypart:
                    ax.scatter(proj[0], proj[1], s=(marker_size + 4) ** 2,
                               facecolors='none', edgecolors='white', linewidths=1.5, zorder=3)

            # Restore zoom/pan limits if previously set
            if self.view_limits[cam] is not None:
                xlim, ylim = self.view_limits[cam]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        # Update frame label
        self.frame_label.config(text=f"Frame: {fi}/{self.frame_end - 1}")

        # Update status bar
        status_parts = [f"Frame: {fi}"]
        if self.selected_bodypart:
            status_parts.append(f"Selected: {self.selected_bodypart}")
            pt = self.get_3d_coords(fi, self.selected_bodypart)
            if pt is not None:
                status_parts.append(f"3D: ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})")
            else:
                status_parts.append("(NaN)")
            # Show which views are enabled
            enabled = [c for c in CAMERAS if self.cam_enabled[c].get()]
            status_parts.append(f"Views: {'+'.join(enabled)}")
        else:
            status_parts.append("Select a bodypart from the list to start")
        self.status_var.set("  |  ".join(status_parts))

        self.canvas.draw_idle()

    # -------------------- Navigation --------------------

    def go_to_frame(self, delta):
        """Move to a different frame by delta."""
        new_frame = self.current_frame + delta
        new_frame = max(self.frame_start, min(new_frame, self.frame_end - 1))
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self.display_frame()

    def on_key_left(self, event):
        self.go_to_frame(-1)

    def on_key_right(self, event):
        self.go_to_frame(1)

    def on_key_shift_left(self, event):
        self.go_to_frame(-10)

    def on_key_shift_right(self, event):
        self.go_to_frame(10)

    # -------------------- Selection --------------------

    def _select_bodypart(self, bp_name):
        """Select a bodypart from the list."""
        self.selected_bodypart = bp_name
        self.display_frame()

    def _advance_to_next_bodypart(self):
        """Auto-advance to the next bodypart in the display order."""
        if self.selected_bodypart is None:
            return
        try:
            idx = self.bp_display_order.index(self.selected_bodypart)
        except ValueError:
            return
        next_idx = idx + 1
        if next_idx < len(self.bp_display_order):
            next_bp = self.bp_display_order[next_idx]
            self.selected_bodypart = next_bp
            self.bp_var.set(next_bp)

    def _get_cam_for_axes(self, ax):
        """Return the camera name for a given matplotlib axes."""
        for i, cam in enumerate(CAMERAS):
            if self.axs[i] == ax:
                return cam
        return None

    # -------------------- Click to place --------------------

    def on_mouse_press(self, event):
        """Handle mouse press: click to place label, middle-click to pan."""
        if event.inaxes is None:
            return

        cam = self._get_cam_for_axes(event.inaxes)
        if cam is None:
            return

        # Middle button: start pan
        if event.button == 2:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
            self.pan_ax = event.inaxes
            self.pan_xlim = event.inaxes.get_xlim()
            self.pan_ylim = event.inaxes.get_ylim()
            return

        # Left click: place label
        if event.button == 1:
            if self.selected_bodypart is None:
                self.status_var.set("Select a bodypart from the list first")
                return
            if not self.cam_enabled[cam].get():
                self.status_var.set(f"{cam.capitalize()} view is unchecked — click ignored")
                return
            self._apply_correction(cam, event.xdata, event.ydata)
            self._advance_to_next_bodypart()
            self.display_frame()

    def on_mouse_release(self, event):
        """Handle mouse release: finish pan."""
        if self.panning:
            self.panning = False

    def on_mouse_move(self, event):
        """Handle mouse move: pan only."""
        if self.panning and event.inaxes == self.pan_ax and event.xdata is not None:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            self.pan_ax.set_xlim(self.pan_xlim[0] - dx, self.pan_xlim[1] - dx)
            self.pan_ax.set_ylim(self.pan_ylim[0] - dy, self.pan_ylim[1] - dy)
            self.canvas.draw_idle()

    def _apply_correction(self, click_cam, new_x, new_y):
        """Place label using click position + reprojections from enabled views only."""
        fi = self.current_frame
        bp = self.selected_bodypart

        old_xyz = self.get_3d_coords(fi, bp)
        old_xyz_copy = old_xyz.copy() if old_xyz is not None else None
        self.undo_stack.append((fi, bp, old_xyz_copy))

        # The clicked view uses the new position
        pts_2d = {click_cam: (new_x, new_y)}

        # Only include other views that are checked AND have an existing 3D point
        if old_xyz is not None:
            for cam in CAMERAS:
                if cam == click_cam:
                    continue
                if not self.cam_enabled[cam].get():
                    continue
                proj = project_points(old_xyz.reshape(1, 3),
                                      self.intrinsics[cam],
                                      self.extrinsics[cam]['rotm'],
                                      self.extrinsics[cam]['tvec']).flatten()
                pts_2d[cam] = (proj[0], proj[1])

        if len(pts_2d) >= 2:
            new_xyz, reproj_err = triangulate_from_views(
                pts_2d, self.intrinsics, self.extrinsics)
            if new_xyz is not None:
                self.set_3d_coords(fi, bp, new_xyz)
                views_used = '+'.join(pts_2d.keys())
                self.status_var.set(
                    f"Placed {bp} using {views_used}  |  Reproj err: {reproj_err:.2f}px")
            else:
                self.status_var.set(f"Triangulation failed for {bp}")
        else:
            self.status_var.set(f"Need at least 2 views — check more camera boxes")

    # -------------------- Zoom / scroll --------------------

    def on_scroll(self, event):
        """Zoom in/out on the hovered subplot."""
        if event.inaxes is None:
            return
        ax = event.inaxes
        scale = 0.8 if event.button == 'up' else 1.25

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xc = event.xdata
        yc = event.ydata

        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale

        ax.set_xlim(xc - new_w / 2, xc + new_w / 2)
        ax.set_ylim(yc - new_h / 2, yc + new_h / 2)

        self.canvas.draw_idle()

    def reset_zoom(self):
        """Reset all camera views to auto zoom/pan."""
        self.view_limits = {cam: None for cam in CAMERAS}
        self.display_frame()
        self.status_var.set("Zoom/pan reset to default")

    def on_key_reset_zoom(self, event):
        self.reset_zoom()

    # -------------------- Key actions --------------------

    def on_key_delete(self, event):
        """Delete (NaN) the selected bodypart for this frame."""
        if self.selected_bodypart is None:
            return
        fi = self.current_frame
        bp = self.selected_bodypart
        old_xyz = self.get_3d_coords(fi, bp)
        old_xyz_copy = old_xyz.copy() if old_xyz is not None else None
        self.undo_stack.append((fi, bp, old_xyz_copy))
        self.set_3d_coords(fi, bp, None)
        self.status_var.set(f"Deleted {bp} at frame {fi}")
        self.display_frame()

    def on_key_undo(self, event):
        """Undo the last correction."""
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return
        fi, bp, old_xyz = self.undo_stack.pop()
        self.set_3d_coords(fi, bp, old_xyz)
        self.selected_bodypart = bp
        self.bp_var.set(bp)
        self.status_var.set(f"Undone correction to {bp} at frame {fi}")
        self.current_frame = fi
        self.display_frame()

    def on_key_save(self, event):
        """Save via keyboard shortcut."""
        self.save_corrections()

    # -------------------- Save --------------------

    def save_corrections(self):
        """Save corrected coordinates to the _corrected HDF5 file."""
        # Always save to _corrected version of the original file (avoid double-suffix)
        base, ext = os.path.splitext(self.coords_file)
        if base.endswith('_corrected'):
            out_path = self.coords_file
        else:
            out_path = base + '_corrected' + ext

        self.coords_df.to_hdf(out_path, key='df_with_missing', mode='w')
        self.status_var.set(f"Saved corrections to {out_path}")
        messagebox.showinfo("Saved", f"Corrections saved to:\n{out_path}")

    # -------------------- Cleanup --------------------

    def cleanup(self):
        """Release video captures."""
        for cap in self.caps.values():
            if cap is not None:
                cap.release()


# -------------------- Entry point --------------------

if __name__ == '__main__':
    VIDEO_PATHS = {
        'side': r'H:\movement\sample_data\short_form\DLC_single-mouse_DBTravelator_sideview_run41.avi',
        'front': r'H:\movement\sample_data\short_form\DLC_single-mouse_DBTravelator_frontview_run41.avi',
        'overhead': r'H:\movement\sample_data\short_form\DLC_single-mouse_DBTravelator_overheadview_run41.avi',
    }
    COORDS_FILE = r'H:\movement\sample_data\short_form\DLC_single-mouse_DBTravelator_3D_run41.h5'
    FRAME_START = 0
    FRAME_END = 417

    root = tk.Tk()
    app = Correct3DLabels(root, VIDEO_PATHS, COORDS_FILE,
                          frame_start=FRAME_START, frame_end=FRAME_END)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
