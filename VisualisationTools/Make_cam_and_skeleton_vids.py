# -*- coding: utf-8 -*-
"""
Create two MP4s for the same run and frame range:
1) camera snippet cut from a recorded video
2) 3D skeleton snippet rendered from tracked coordinates

Fill in the placeholders in the __main__ block.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# your config
from Helpers.Config_23 import micestuff  # provides 'bodyparts' and 'skeleton'

# -------------------- camera snippet --------------------

def make_camera_snippet(
    video_path,
    out_path,
    start_frame,
    end_frame,
    fps=247
):
    """
    Cut frames [start_frame, end_frame] inclusive from video_path and save to out_path (mp4).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Invalid start or end frame")
    if end_frame >= total_frames:
        end_frame = total_frames - 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not open writer for: {out_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame
    while current <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        current += 1

    cap.release()
    writer.release()
    print(f"Saved camera snippet: {out_path}  [{start_frame}..{end_frame}] at {fps} fps")


# -------------------- 3D skeleton snippet --------------------

def make_3d_snippet(
    coords_df,              # DataFrame indexed by frame with MultiIndex columns (part, coord) where coord in ['x','y','z']
    out_path,               # mp4 output
    start_frame,
    end_frame,
    fps=247,
    xlim=(0, 600),
    ylim=(0, 53.5),
    zlim=(0, 53.5),
    add_belts=True,
    base_view=(10, 0),      # elev, azim
    rotate=False,           # set True to spin view over time
    dpi=220,
    marker_size=2
):
    """
    Render a 3D animation of the skeleton across frames [start_frame, end_frame] and save to mp4.
    Assumes coords_df has columns (part, 'x'|'y'|'z') and an index that contains those frame numbers.
    Uses micestuff['bodyparts'] and micestuff['skeleton'].
    """
    bodyparts = micestuff['bodyparts']
    skeleton_pairs = micestuff['skeleton']

    # try to slice by absolute frame numbers, else fallback to positional
    frames_list = list(range(start_frame, end_frame + 1))
    try:
        temp = coords_df.loc[frames_list].copy()
    except Exception:
        temp = coords_df.iloc[start_frame:end_frame + 1].copy()

    # limit to bodyparts you care about if extra columns exist
    # expects columns as MultiIndex (part, coord)
    cols = []
    for p in bodyparts:
        for c in ('x', 'y', 'z'):
            if (p, c) in temp.columns:
                cols.append((p, c))
    if not cols:
        raise ValueError("Could not find any (part, coord) columns matching micestuff['bodyparts'].")

    temp = temp.loc[:, cols]
    temp = temp.reindex(columns=pd.MultiIndex.from_tuples(cols, names=['part', 'coords']))
    temp = temp.reset_index(drop=True)

    n_frames = len(temp)
    if n_frames <= 0:
        raise ValueError("No frames found for the chosen range")

    # figure and axes
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111, projection='3d')
    elev0, azim0 = base_view
    ax.view_init(elev=elev0, azim=azim0)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([
        (xlim[1]-xlim[0]) / max(1e-6, (ylim[1]-ylim[0])),
        1.0,
        (zlim[1]-zlim[0]) / max(1e-6, (ylim[1]-ylim[0]))
    ])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # optional belts
    belts = []
    if add_belts:
        belt1_verts = [[(0, 0, 0), (470, 0, 0), (470, 53.5, 0), (0, 53.5, 0)]]
        belt2_verts = [[(471, 0, 0), (600, 0, 0), (600, 53.5, 0), (471, 53.5, 0)]]
        belt1 = Poly3DCollection(belt1_verts, facecolors='blue', edgecolors='none', alpha=0.2)
        belt2 = Poly3DCollection(belt2_verts, facecolors='blue', edgecolors='none', alpha=0.2)
        ax.add_collection3d(belt1)
        ax.add_collection3d(belt2)
        belts = [belt1, belt2]

    # colormap and artists
    viridis = plt.get_cmap('viridis')
    colors = viridis(np.linspace(0, 1, len(bodyparts)))

    # one artist per part
    part_lines = {}
    for i, part in enumerate(bodyparts):
        ln, = ax.plot([], [], [], 'o-', ms=marker_size, color=colors[i], lw=0.8)
        part_lines[part] = ln

    # one artist per skeleton pair
    sk_lines = {}
    for pair in skeleton_pairs:
        ln, = ax.plot([], [], [], '-', color='black', lw=0.6)
        sk_lines[pair] = ln

    # helpers
    def get_xyz(fi, part):
        return (
            temp.loc[fi, (part, 'x')],
            temp.loc[fi, (part, 'y')],
            temp.loc[fi, (part, 'z')],
        )

    # init with NaN arrays so shapes are valid for 3D artists
    def init():
        nan1 = np.array([np.nan])        # shape (1,)
        nan2 = np.array([np.nan, np.nan])# shape (2,)
        for ln in part_lines.values():
            ln.set_data(nan1, nan1)
            ln.set_3d_properties(nan1)
        for ln in sk_lines.values():
            ln.set_data(nan2, nan2)
            ln.set_3d_properties(nan2)
        return belts + list(part_lines.values()) + list(sk_lines.values())

    def update(fi):
        # points
        for part, ln in part_lines.items():
            x, y, z = get_xyz(fi, part)
            ln.set_data(np.array([x]), np.array([y]))
            ln.set_3d_properties(np.array([z]))

        # segments
        for (p1, p2), ln in sk_lines.items():
            x1, y1, z1 = get_xyz(fi, p1)
            x2, y2, z2 = get_xyz(fi, p2)
            ln.set_data(np.array([x1, x2]), np.array([y1, y2]))
            ln.set_3d_properties(np.array([z1, z2]))

        if rotate:
            ax.view_init(elev=base_view[0], azim=base_view[1] + (fi * 360.0 / max(1, n_frames)))
        return belts + list(part_lines.values()) + list(sk_lines.values())

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)
    ani.save(out_path, writer='ffmpeg', fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Saved 3D snippet: {out_path}  [{start_frame}..{end_frame}] at {fps} fps")


# -------------------- example usage --------------------

if __name__ == "__main__":
    # Fill these in
    VIDEO_PATH   = r"C:\Users\hmorl\Documents\HM_20230316_APACharExt_FAA-1035246_LR_side_1.avi"
    CAMERA_OUT   = r"H:\Dual-belt_APAs\side_cam_run.mp4"
    COORDS_FILE  = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round7_Jan25\APAChar_LowHigh\Extended\allmice.pickle"
    MOUSE_KEY    = "1035246"
    FRAME_START  = 249195
    FRAME_END    = 249411
    FPS          = 247

    # 1) camera snippet
    make_camera_snippet(
        video_path=VIDEO_PATH,
        out_path=CAMERA_OUT,
        start_frame=FRAME_START,
        end_frame=FRAME_END,
        fps=FPS
    )

    # 2) 3D snippet
    with open(COORDS_FILE, "rb") as f:
        loaded = pd.read_pickle(f)

    if isinstance(loaded, dict) and MOUSE_KEY in loaded:
        coords_df = loaded[MOUSE_KEY]
    else:
        coords_df = loaded  # assume already a DataFrame

    THREE_D_OUT = r"H:\Dual-belt_APAs\3d_coord_run.mp4"

    make_3d_snippet(
        coords_df=coords_df,
        out_path=THREE_D_OUT,
        start_frame=FRAME_START,
        end_frame=FRAME_END,
        fps=FPS,
        xlim=(0, 600),
        ylim=(0, 53.5),
        zlim=(0, 53.5),
        add_belts=True,
        base_view=(10, 0),
        rotate=False,
        dpi=220,
        marker_size=2
    )
