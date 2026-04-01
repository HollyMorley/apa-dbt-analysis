# -*- coding: utf-8 -*-
"""
Create two MP4s (and optionally GIFs) for the same run and frame range:
1) camera snippet cut from a recorded video
2) 3D skeleton snippet rendered from tracked coordinates

Fill in the placeholders in the __main__ block.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

# your config
from helpers.config import micestuff  # provides 'bodyparts' and 'skeleton'

# -------------------- camera snippet --------------------

def make_camera_snippet(
    video_path,
    out_path,
    start_frame,
    end_frame,
    fps=247,
    slowdown_factor=1.0,
    save_gif=False,
    return_gif_frames=False
):
    """
    Cut frames [start_frame, end_frame] inclusive from video_path and save to out_path (mp4).
    slowdown_factor: >1 slows the video down (e.g. 5 = 5x slower).
    save_gif: if True, also saves a .gif alongside the mp4.
    return_gif_frames: if True, return list of PIL Image frames (for combining with other GIFs).
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

    out_fps = fps / slowdown_factor

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not open writer for: {out_path}")

    gif_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame
    while current <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        if save_gif or return_gif_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gif_frames.append(Image.fromarray(rgb))
        current += 1

    cap.release()
    writer.release()
    print(f"Saved camera snippet: {out_path}  [{start_frame}..{end_frame}] at {out_fps:.1f} fps (slowdown {slowdown_factor}x)")

    if save_gif and gif_frames:
        gif_path = os.path.splitext(out_path)[0] + '.gif'
        gif_duration_ms = round(1000 / out_fps)  # ms per frame, rounded to match 3D GIF
        gif_frames[0].save(
            gif_path, save_all=True, append_images=gif_frames[1:],
            duration=gif_duration_ms, loop=0
        )
        print(f"Saved camera GIF: {gif_path}  ({gif_duration_ms}ms/frame)")

    if return_gif_frames:
        return gif_frames
    return None


# -------------------- pcutoff filtering --------------------

def filter_by_pcutoff(coords_df, pcutoff=0.9):
    """
    For each bodypart, if a 'likelihood' column exists, set x/y/z to NaN
    where likelihood < pcutoff, then drop likelihood columns.
    Works with MultiIndex columns at level 0 = bodypart, level 1 = coord.
    """
    df = coords_df.copy()
    bodyparts_in_df = df.columns.get_level_values(0).unique()

    for bp in bodyparts_in_df:
        if (bp, 'likelihood') in df.columns:
            mask = df[(bp, 'likelihood')] < pcutoff
            for c in ('x', 'y', 'z'):
                if (bp, c) in df.columns:
                    df.loc[mask, (bp, c)] = np.nan
            df = df.drop(columns=[(bp, 'likelihood')])

    print(f"Filtered coordinates by pcutoff={pcutoff}")
    return df


def interpolate_missing(coords_df, method='linear', max_gap=None):
    """
    Interpolate NaN gaps (e.g. from pcutoff filtering) per bodypart per coordinate.
    method: interpolation method passed to pandas (e.g. 'linear', 'cubic', 'spline').
    max_gap: max consecutive NaN frames to fill. None = fill all gaps.
    """
    df = coords_df.copy()
    bodyparts_in_df = df.columns.get_level_values(0).unique()

    n_before = df.isna().sum().sum()
    for bp in bodyparts_in_df:
        for c in ('x', 'y', 'z'):
            if (bp, c) in df.columns:
                df[(bp, c)] = df[(bp, c)].interpolate(method=method, limit=max_gap, limit_direction='both')
    n_after = df.isna().sum().sum()

    print(f"Interpolated {n_before - n_after} NaN values ({method}, max_gap={max_gap})")
    return df


# -------------------- 3D skeleton snippet --------------------

def make_3d_snippet(
    coords_df,              # DataFrame indexed by frame with MultiIndex columns (part, coord) where coord in ['x','y','z']
    out_path,               # mp4 output
    start_frame,
    end_frame,
    fps=247,
    slowdown_factor=1.0,
    xlim=(-50, 600),
    ylim=(0, 50),
    zlim=(0, 50),
    add_belts=True,
    add_platform=True,
    base_view=(10, 0),      # elev, azim
    rotate=False,           # set True to spin view over time
    rotate_degrees=360,     # total degrees of rotation over the clip
    dpi=220,
    marker_size=2,
    pcutoff=None,           # if set, filter out low-confidence points (requires 'likelihood' column)
    interpolate=True,       # interpolate NaN gaps after pcutoff filtering
    interp_method='linear', # interpolation method ('linear', 'cubic', etc.)
    interp_max_gap=None,    # max consecutive NaN frames to interpolate (None = no limit)
    save_gif=False,
    return_gif_frames=False
):
    """
    Render a 3D animation of the skeleton across frames [start_frame, end_frame] and save to mp4.
    Assumes coords_df has columns (part, 'x'|'y'|'z') and an index that contains those frame numbers.
    Uses micestuff['bodyparts'] and micestuff['skeleton'].

    slowdown_factor: >1 slows the video down (e.g. 5 = 5x slower).
    rotate_degrees: total azimuth rotation over the whole clip (default 360).
    pcutoff: if provided, NaN out any bodypart coords where likelihood < pcutoff.
    interpolate: if True (and pcutoff is set), interpolate the NaN gaps.
    interp_method: pandas interpolation method (default 'linear').
    interp_max_gap: max consecutive NaN frames to fill (None = fill all).
    save_gif: if True, also saves a .gif alongside the mp4.
    return_gif_frames: if True, return list of PIL Image frames (for combining with other GIFs).
    """
    bodyparts = micestuff['bodyparts']
    skeleton_pairs = micestuff['skeleton']

    # apply pcutoff filtering if requested
    if pcutoff is not None:
        coords_df = filter_by_pcutoff(coords_df, pcutoff=pcutoff)
        if interpolate:
            coords_df = interpolate_missing(coords_df, method=interp_method, max_gap=interp_max_gap)

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

    out_fps = fps / slowdown_factor

    # figure and axes — wide and short to minimise vertical whitespace
    fig = plt.figure(figsize=(9, 2.5))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=-0.25, right=1.25, top=1.25, bottom=-0.25)
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

    # optional belts (two separate belts with visible transition at x=470)
    belt_w = 50  # belt width in mm
    scenery = []
    if add_belts:
        belt1_verts = [[(0, 0, 0), (470, 0, 0), (470, belt_w, 0), (0, belt_w, 0)]]
        belt2_verts = [[(470, 0, 0), (600, 0, 0), (600, belt_w, 0), (470, belt_w, 0)]]
        belt1 = Poly3DCollection(belt1_verts, facecolors='steelblue', edgecolors='grey', linewidths=0.5, alpha=0.25)
        belt2 = Poly3DCollection(belt2_verts, facecolors='cornflowerblue', edgecolors='grey', linewidths=0.5, alpha=0.25)
        ax.add_collection3d(belt1)
        ax.add_collection3d(belt2)
        # transition line at x=470
        ax.plot([470, 470], [0, belt_w], [0, 0], color='k', lw=1.5, ls='-', label='belt transition')
        scenery.extend([belt1, belt2])

    # optional start platform (box from x=-50 to x=0, width=belt_w, z=0 to z=10)
    if add_platform:
        pz = 4  # platform height
        # top face
        top = [[ (-50, 0, pz), (0, 0, pz), (0, belt_w, pz), (-50, belt_w, pz) ]]
        # front face (y=0)
        front = [[ (-50, 0, 0), (0, 0, 0), (0, 0, pz), (-50, 0, pz) ]]
        # back face (y=belt_w)
        back = [[ (-50, belt_w, 0), (0, belt_w, 0), (0, belt_w, pz), (-50, belt_w, pz) ]]
        # left face (x=-50)
        left = [[ (-50, 0, 0), (-50, belt_w, 0), (-50, belt_w, pz), (-50, 0, pz) ]]
        # right face (x=0) — where platform meets belt
        right = [[ (0, 0, 0), (0, belt_w, 0), (0, belt_w, pz), (0, 0, pz) ]]

        for verts, alpha in [(top, 0.95), (front, 0.85), (back, 0.85), (left, 0.85), (right, 0.85)]:
            face = Poly3DCollection(verts, facecolors='k', edgecolors='dimgrey', linewidths=0.5, alpha=alpha)
            ax.add_collection3d(face)
            scenery.append(face)

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
        return scenery + list(part_lines.values()) + list(sk_lines.values())

    # list to capture rendered frames during animation (for GIF / combining)
    captured_frames = []
    capture = save_gif or return_gif_frames
    gif_dpi = dpi  # use full DPI for GIF/frame capture quality

    def update(fi):
        # points
        for part, ln in part_lines.items():
            x, y, z = get_xyz(fi, part)
            # hide point if NaN (filtered by pcutoff)
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                ln.set_data(np.array([np.nan]), np.array([np.nan]))
                ln.set_3d_properties(np.array([np.nan]))
            else:
                ln.set_data(np.array([x]), np.array([y]))
                ln.set_3d_properties(np.array([z]))

        # segments
        for (p1, p2), ln in sk_lines.items():
            x1, y1, z1 = get_xyz(fi, p1)
            x2, y2, z2 = get_xyz(fi, p2)
            # hide segment if either endpoint is NaN
            if any(np.isnan(v) for v in (x1, y1, z1, x2, y2, z2)):
                ln.set_data(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
                ln.set_3d_properties(np.array([np.nan, np.nan]))
            else:
                ln.set_data(np.array([x1, x2]), np.array([y1, y2]))
                ln.set_3d_properties(np.array([z1, z2]))

        if rotate:
            ax.view_init(elev=elev0, azim=azim0 + (fi * rotate_degrees / max(1, n_frames - 1)))

        # capture frame while it's freshly rendered
        if capture:
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            captured_frames.append(Image.fromarray(np.asarray(buf).copy()))

        return scenery + list(part_lines.values()) + list(sk_lines.values())

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)
    ani.save(out_path, writer='ffmpeg', fps=out_fps, dpi=dpi)
    print(f"Saved 3D snippet: {out_path}  [{start_frame}..{end_frame}] at {out_fps:.1f} fps (slowdown {slowdown_factor}x)")

    # auto-crop whitespace from captured frames using the union of all bounding boxes
    if capture and captured_frames:
        from PIL import ImageChops
        # find the tightest crop that fits all frames (so the crop is consistent)
        union_bbox = None
        for frame in captured_frames:
            bg = Image.new(frame.mode, frame.size, (255, 255, 255, 255))
            diff = ImageChops.difference(frame, bg)
            bbox = diff.getbbox()
            if bbox:
                if union_bbox is None:
                    union_bbox = bbox
                else:
                    union_bbox = (
                        min(union_bbox[0], bbox[0]),
                        min(union_bbox[1], bbox[1]),
                        max(union_bbox[2], bbox[2]),
                        max(union_bbox[3], bbox[3]),
                    )
        if union_bbox:
            # add a small margin
            margin = 4
            union_bbox = (
                max(0, union_bbox[0] - margin),
                max(0, union_bbox[1] - margin),
                min(captured_frames[0].width, union_bbox[2] + margin),
                min(captured_frames[0].height, union_bbox[3] + margin),
            )
            captured_frames = [f.crop(union_bbox) for f in captured_frames]

    if save_gif and captured_frames:
        gif_path = os.path.splitext(out_path)[0] + '.gif'
        gif_duration_ms = round(1000 / out_fps)
        captured_frames[0].save(
            gif_path, save_all=True, append_images=captured_frames[1:],
            duration=gif_duration_ms, loop=0
        )
        print(f"Saved 3D GIF: {gif_path}  ({gif_duration_ms}ms/frame)")

    plt.close(fig)

    if return_gif_frames:
        return captured_frames
    return None


# -------------------- combined GIF --------------------

def _combine_frame_lists(cam_frames, skeleton_frames, layout='vertical', max_width=None):
    """
    Combine two frame lists into one, stacking vertically or horizontally.
    max_width: if set, resize the wider stream (usually camera) down to this width first.
    Returns a list of RGB PIL Images.
    """
    n = min(len(cam_frames), len(skeleton_frames))
    if n == 0:
        raise ValueError("No frames to combine")

    combined = []
    for i in range(n):
        cam = cam_frames[i].convert('RGBA')
        skel = skeleton_frames[i].convert('RGBA')

        # downsize camera frames to target width before combining
        if max_width and cam.width > max_width:
            scale = max_width / cam.width
            cam = cam.resize((max_width, int(cam.height * scale)), Image.LANCZOS)

        if layout == 'vertical':
            target_w = cam.width
            scale = target_w / skel.width
            skel = skel.resize((target_w, int(skel.height * scale)), Image.LANCZOS)
            total_h = cam.height + skel.height
            combo = Image.new('RGBA', (target_w, total_h), (255, 255, 255, 255))
            combo.paste(cam, (0, 0))
            combo.paste(skel, (0, cam.height))
        else:
            target_h = cam.height
            scale = target_h / skel.height
            skel = skel.resize((int(skel.width * scale), target_h), Image.LANCZOS)
            total_w = cam.width + skel.width
            combo = Image.new('RGBA', (total_w, target_h), (255, 255, 255, 255))
            combo.paste(cam, (0, 0))
            combo.paste(skel, (cam.width, 0))

        combined.append(combo.convert('RGB'))
    return combined


def make_combined_gif(
    cam_frames,
    skeleton_frames,
    out_path,
    fps=49.4,
    layout='vertical',
    max_width=600            # resize camera frames to this width before combining
):
    """
    Combine camera and 3D skeleton GIF frames into a single synchronized GIF.
    Camera frames are resized to max_width first (they are typically much larger than
    the 3D render), then the skeleton is scaled to match. This keeps file size manageable.
    """
    combined = _combine_frame_lists(cam_frames, skeleton_frames, layout, max_width=max_width)

    # quantize to 256 colours with dithering for better GIF quality
    combined = [f.quantize(colors=256, method=0, dither=1).convert('RGB') for f in combined]

    gif_duration_ms = round(1000 / fps)
    combined[0].save(
        out_path, save_all=True, append_images=combined[1:],
        duration=gif_duration_ms, loop=0, optimize=True
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved combined GIF: {out_path}  ({len(combined)} frames, {gif_duration_ms}ms/frame, {size_mb:.1f}MB)")


def make_combined_mp4(
    cam_frames,
    skeleton_frames,
    out_path,
    fps=49.4,
    layout='vertical',
    max_width=None           # None = full resolution
):
    """
    Combine camera and 3D skeleton frames into a single synchronized MP4.
    Full quality, small file size.
    """
    combined = _combine_frame_lists(cam_frames, skeleton_frames, layout, max_width=max_width)

    # convert first frame to get dimensions
    first = np.array(combined[0])
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame in combined:
        arr = np.array(frame)
        writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    writer.release()
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved combined MP4: {out_path}  ({len(combined)} frames, {size_mb:.1f}MB)")


# -------------------- example usage --------------------

if __name__ == "__main__":
    # ---- paths ----
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MEDIA_DIR    = os.path.join(PROJECT_ROOT, 'docs', 'media')
    os.makedirs(MEDIA_DIR, exist_ok=True)

    # Fill these in
    VIDEO_PATH   = r"H:\movement\sample_data\short_form\DLC_single-mouse_DBTravelator_sideview_run41.avi"
    COORDS_FILE  = r"H:\movement\sample_data\short_form\DLC_single-mouse_DBTravelator_3D_run41.h5"
    MOUSE_KEY    = "1035246"
    FRAME_START  = 0
    FRAME_END    = 417
    FPS          = 247
    SLOWDOWN     = 5.0       # 5x slower playback
    PCUTOFF      = 0.9       # filter out low-confidence DLC points

    CAMERA_OUT   = os.path.join(MEDIA_DIR, 'side_cam_run.mp4')
    THREE_D_OUT  = os.path.join(MEDIA_DIR, '3d_coord_run.mp4')
    COMBINED_GIF = os.path.join(MEDIA_DIR, 'combined_run.gif')
    COMBINED_MP4 = os.path.join(MEDIA_DIR, 'combined_run.mp4')

    # 1) camera snippet — save mp4 and return frames for combined GIF
    cam_frames = make_camera_snippet(
        video_path=VIDEO_PATH,
        out_path=CAMERA_OUT,
        start_frame=FRAME_START,
        end_frame=FRAME_END,
        fps=FPS,
        slowdown_factor=SLOWDOWN,
        save_gif=False,
        return_gif_frames=True
    )

    # 2) load coordinate data
    if "pickle" in COORDS_FILE:
        with open(COORDS_FILE, "rb") as f:
            loaded = pd.read_pickle(f)

        if isinstance(loaded, dict) and MOUSE_KEY in loaded:
            coords_df = loaded[MOUSE_KEY]
        else:
            coords_df = loaded  # assume already a DataFrame
    elif ".h5" in COORDS_FILE:
        coords_df = pd.read_hdf(COORDS_FILE)
        # check if scorer is in columns and if so, remove it
        if coords_df.columns.nlevels == 3 and coords_df.columns.names == ['scorer', 'bodyparts', 'coords']:
            coords_df.columns = coords_df.columns.droplevel('scorer')

    # 3) 3D skeleton snippet — save mp4 and return frames for combined GIF
    skel_frames = make_3d_snippet(
        coords_df=coords_df,
        out_path=THREE_D_OUT,
        start_frame=FRAME_START,
        end_frame=FRAME_END,
        fps=FPS,
        slowdown_factor=SLOWDOWN,
        xlim=(-50, 600),
        ylim=(0, 50),
        zlim=(0, 50),
        add_belts=True,
        add_platform=True,
        base_view=(10, 0),
        rotate=True,
        rotate_degrees=360,
        dpi=220,
        marker_size=1,
        pcutoff=PCUTOFF,
        interpolate=True,
        interp_method='linear',
        interp_max_gap=10,
        save_gif=False,
        return_gif_frames=True
    )

    # 4) combined outputs — camera on top, skeleton below, perfectly synced
    out_fps = FPS / SLOWDOWN
    make_combined_mp4(cam_frames, skel_frames, COMBINED_MP4, fps=out_fps, layout='vertical')
    make_combined_gif(cam_frames, skel_frames, COMBINED_GIF, fps=out_fps, layout='vertical', max_width=600)
