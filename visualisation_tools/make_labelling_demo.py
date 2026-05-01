"""
Labelling demo animation showing the pose-estimation pipeline.

All rendering uses cv2.projectPoints with a virtual pinhole camera —
no matplotlib. This ensures seamless transitions between phases.

Phase 1 - Real side-cam video plays up to a configurable pause point
Phase 2 - Video freezes; labelled scatter points appear sequentially,
          then skeleton connections fade in
Phase 3 - Video fades away, skeleton overlay stays in place
Phase 4 - Skeleton resumes running, virtual camera orbits around scene

Outputs GIF and/or MP4.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import CubicSpline

from helpers.config import micestuff
from helpers.utils_3d_reconstruction import CameraData
from visualisation_tools.make_cam_and_skeleton_vids import filter_by_pcutoff

# -------------------- camera geometry --------------------

# Calibrated side-camera extrinsics (from cams_pose.py)
SIDE_CAM_EXTRINSICS = {
    'tvec': np.array([[-298.85353394], [65.67187339], [1071.78906513]]),
    'rotm': np.array([
        [0.9999789, -0.00207372, 0.00615665],
        [0.00621094, 0.02727888, -0.99960857],
        [0.00190496, 0.99962571, 0.02729118],
    ]),
}


def _get_side_camera():
    """Return intrinsic matrix K, rotation matrix R, translation t for the side camera."""
    cam = CameraData(basic=True)
    K = cam.intrinsic_matrices['side']
    R = SIDE_CAM_EXTRINSICS['rotm']
    t = SIDE_CAM_EXTRINSICS['tvec']
    return K, R, t


def _look_at(cam_pos, target, up=np.array([0., 0., 1.])):
    """
    Compute OpenCV-convention R, t for a camera at cam_pos looking at target.

    Returns R (3x3), t (3x1) such that p_cam = R @ p_world + t.
    Camera axes: X=right, Y=down, Z=forward (optical axis).
    """
    fwd = target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    norm_right = np.linalg.norm(right)
    if norm_right < 1e-6:
        # fwd is parallel to up — use a fallback up direction
        up = np.array([1., 0., 0.])
        right = np.cross(fwd, up)
        norm_right = np.linalg.norm(right)
    right = right / norm_right
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    t = (-R @ cam_pos).reshape(3, 1)
    return R, t


def _project(pts_3d, K, R, t, scale=1.0):
    """
    Project Nx3 world points to Nx2 pixel coordinates.

    Parameters
    ----------
    pts_3d : array-like, shape (N, 3)
    K : (3, 3) intrinsic matrix
    R : (3, 3) rotation matrix (world -> camera)
    t : (3, 1) translation vector
    scale : float
        Pixel coordinate scale (e.g. target_width / original_width).

    Returns
    -------
    np.ndarray, shape (N, 2)
    """
    pts = np.asarray(pts_3d, dtype=np.float64).reshape(-1, 1, 3)
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(pts, rvec, t, K, distCoeffs=np.array([]))
    return projected.reshape(-1, 2) * scale


# -------------------- scenery geometry --------------------

BELT_W = 50.0  # belt width in mm

# Belt surface polygons (z=0 plane)
BELT1_VERTS = np.array([[0, 0, 0], [470, 0, 0], [470, BELT_W, 0], [0, BELT_W, 0]])
BELT2_VERTS = np.array([[470, 0, 0], [600, 0, 0], [600, BELT_W, 0], [470, BELT_W, 0]])
TRANSITION_LINE = np.array([[470, 0, 0], [470, BELT_W, 0]])

# Platform faces
PZ = 4.0  # platform height
PLATFORM_FACES = [
    np.array([[-50, 0, PZ], [0, 0, PZ], [0, BELT_W, PZ], [-50, BELT_W, PZ]]),          # top
    np.array([[-50, 0, 0], [0, 0, 0], [0, 0, PZ], [-50, 0, PZ]]),                       # front
    np.array([[-50, BELT_W, 0], [0, BELT_W, 0], [0, BELT_W, PZ], [-50, BELT_W, PZ]]),   # back
    np.array([[-50, 0, 0], [-50, BELT_W, 0], [-50, BELT_W, PZ], [-50, 0, PZ]]),          # left
    np.array([[0, 0, 0], [0, BELT_W, 0], [0, BELT_W, PZ], [0, 0, PZ]]),                  # right
]


# -------------------- label reveal order --------------------

def _build_label_sequence():
    """
    Return a list of steps, where each step is a list of bodypart names
    to reveal simultaneously.

    Order: tail tip -> tail base -> back (rear->front) -> head,
    then all four limbs in parallel (toe -> knuckle -> ankle -> knee).
    """
    tail = [[f'Tail{i}'] for i in range(12, 0, -1)]
    back = [[f'Back{i}'] for i in range(12, 0, -1)]
    head = [['Nose'], ['EarL', 'EarR']]
    limbs = [
        ['ForepawToeL', 'ForepawToeR', 'HindpawToeL', 'HindpawToeR'],
        ['ForepawKnuckleL', 'ForepawKnuckleR', 'HindpawKnuckleL', 'HindpawKnuckleR'],
        ['ForepawAnkleL', 'ForepawAnkleR', 'HindpawAnkleL', 'HindpawAnkleR'],
        ['ForepawKneeL', 'ForepawKneeR', 'HindpawKneeL', 'HindpawKneeR'],
    ]
    return tail + back + head + limbs


# -------------------- spline chains --------------------

# Sequential chains that should be drawn as smooth curves, not segments
BACK_CHAIN = [f'Back{i}' for i in range(1, 13)]
TAIL_CHAIN = [f'Tail{i}' for i in range(1, 13)]
SPINE_CHAIN = BACK_CHAIN + TAIL_CHAIN  # full spine for body fill

# Skeleton pairs that are part of spline chains (skip individual segment drawing)
_SPLINE_PAIRS = set()
for chain in [BACK_CHAIN, TAIL_CHAIN]:
    for i in range(len(chain) - 1):
        _SPLINE_PAIRS.add((chain[i], chain[i + 1]))
# Also Back12→Tail1 connection
_SPLINE_PAIRS.add(('Back12', 'Tail1'))


def _fit_spline_2d(points, n_interp=50):
    """Fit a smooth cubic spline through 2D points, return interpolated coords."""
    pts = np.array(points)
    t = np.linspace(0, 1, len(pts))
    t_fine = np.linspace(0, 1, n_interp)
    cs_x = CubicSpline(t, pts[:, 0])
    cs_y = CubicSpline(t, pts[:, 1])
    return list(zip(cs_x(t_fine).astype(int), cs_y(t_fine).astype(int)))


# -------------------- 2D drawing helpers --------------------

def _draw_labels(img, visible_parts, positions_2d, bp_colors, marker_radius=2,
                 trail_history=None, trail_radius=None, trail_max_alpha=0.6,
                 depths=None):
    """Draw coloured circles for visible bodyparts on a copy of img.

    Parameters
    ----------
    trail_history : list of dict or None
        If provided, a list of past positions_2d dicts (oldest first).
        Each past frame's dots are drawn with decreasing opacity.
    trail_radius : int or None
        Radius of trail dots. None = same as marker_radius.
    trail_max_alpha : float
        Peak opacity for the most recent trail dot (0-1). Oldest trail
        dot fades towards 0. Default 0.6.
    depths : dict or None
        {bodypart: camera_z_depth}. If provided, dots are drawn far-to-near
        so closer bodyparts render on top (painter's algorithm).
    """
    base = img.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Sort bodyparts by depth (farthest first) for correct occlusion
    if depths:
        sorted_parts = sorted(visible_parts,
                              key=lambda bp: depths.get(bp, 0), reverse=True)
    else:
        sorted_parts = list(visible_parts)

    # Draw trail (older frames first, fading out)
    if trail_history:
        n_trail = len(trail_history)
        r = trail_radius if trail_radius is not None else marker_radius
        for ti, past_positions in enumerate(trail_history):
            alpha = int(255 * (ti + 1) / (n_trail + 1) * trail_max_alpha)
            for bp in sorted_parts:
                pos = past_positions.get(bp)
                if pos is None or np.isnan(pos[0]) or np.isnan(pos[1]):
                    continue
                px, py = pos
                col = bp_colors[bp]
                draw.ellipse([px - r, py - r, px + r, py + r],
                             fill=(*col, alpha), outline=(*col, alpha))

    # Draw current frame dots (full opacity, on top) — far to near
    for bp in sorted_parts:
        pos = positions_2d.get(bp)
        if pos is None or np.isnan(pos[0]) or np.isnan(pos[1]):
            continue
        px, py = pos
        col = bp_colors[bp]
        draw.ellipse([px - marker_radius, py - marker_radius,
                      px + marker_radius, py + marker_radius],
                     fill=(*col, 255), outline=(*col, 255))

    return Image.alpha_composite(base, overlay).convert('RGB')


def _draw_skeleton(img, visible_parts, positions_2d, skeleton_pairs,
                   bp_colors, alpha=1.0, line_width=2, depths=None):
    """Draw skeleton connections: spline curves for back/tail, straight lines for limbs.

    Parameters
    ----------
    depths : dict or None
        {bodypart: camera_z_depth}. If provided, lines are drawn far-to-near
        for correct occlusion during camera orbit.
    """
    base = img.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    a = int(255 * alpha)

    # --- Draw smooth spline curves for back and tail ---
    full_spine = BACK_CHAIN + TAIL_CHAIN
    spine_pts = []
    spine_colors = []
    for bp in full_spine:
        if bp not in visible_parts:
            continue
        pos = positions_2d.get(bp)
        if pos is None or np.isnan(pos[0]) or np.isnan(pos[1]):
            # Break the chain at gaps
            if len(spine_pts) >= 3:
                spline = _fit_spline_2d(spine_pts, n_interp=max(20, len(spine_pts) * 5))
                for i in range(len(spline) - 1):
                    t = i / max(len(spline) - 1, 1)
                    ci = int(t * (len(spine_colors) - 1))
                    col = spine_colors[min(ci, len(spine_colors) - 1)]
                    draw.line([spline[i], spline[i + 1]], fill=(*col, a),
                              width=line_width)
            spine_pts = []
            spine_colors = []
            continue
        spine_pts.append([pos[0], pos[1]])
        spine_colors.append(bp_colors.get(bp, (80, 80, 80)))

    # Draw remaining spine segment
    if len(spine_pts) >= 3:
        spline = _fit_spline_2d(spine_pts, n_interp=max(20, len(spine_pts) * 5))
        for i in range(len(spline) - 1):
            t = i / max(len(spline) - 1, 1)
            ci = int(t * (len(spine_colors) - 1))
            col = spine_colors[min(ci, len(spine_colors) - 1)]
            draw.line([spline[i], spline[i + 1]], fill=(*col, a), width=line_width)
    elif len(spine_pts) == 2:
        col = spine_colors[0]
        draw.line([tuple(int(v) for v in spine_pts[0]),
                   tuple(int(v) for v in spine_pts[1])], fill=(*col, a), width=line_width)

    # --- Draw straight lines for non-spine connections (limbs, head) ---
    # Sort by average depth (farthest first) for correct occlusion
    limb_pairs = [(p1, p2) for p1, p2 in skeleton_pairs if (p1, p2) not in _SPLINE_PAIRS]
    if depths:
        limb_pairs.sort(
            key=lambda pair: (depths.get(pair[0], 0) + depths.get(pair[1], 0)) / 2,
            reverse=True)

    for p1, p2 in limb_pairs:
        if p1 not in visible_parts or p2 not in visible_parts:
            continue
        pos1, pos2 = positions_2d.get(p1), positions_2d.get(p2)
        if pos1 is None or pos2 is None:
            continue
        if any(np.isnan(v) for v in (*pos1, *pos2)):
            continue
        # Use the colour of the second (more distal) bodypart in the pair
        col = bp_colors.get(p2, bp_colors.get(p1, (80, 80, 80)))
        draw.line([tuple(int(v) for v in pos1), tuple(int(v) for v in pos2)],
                  fill=(*col, a), width=line_width)

    return Image.alpha_composite(base, overlay).convert('RGB')


def _draw_full_overlay(bg, visible_parts, positions_2d, bp_colors,
                       skeleton_pairs, marker_radius=2, line_width=2,
                       trail_history=None, trail_radius=None,
                       trail_max_alpha=0.6, depths=None):
    """Draw scatter points + skeleton lines on a background."""
    img = _draw_skeleton(bg, visible_parts, positions_2d, skeleton_pairs,
                         bp_colors, alpha=1.0, line_width=line_width, depths=depths)
    img = _draw_labels(img, visible_parts, positions_2d, bp_colors, marker_radius,
                       trail_history=trail_history, trail_radius=trail_radius,
                       trail_max_alpha=trail_max_alpha, depths=depths)
    return img


def _draw_scenery_on(img, K, R, t, scale, draw_belts=True, draw_platform=True):
    """Draw belt surfaces and platform as projected filled polygons."""
    base = img.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if draw_belts:
        for verts, color in [(BELT1_VERTS, (70, 130, 180, 64)),
                             (BELT2_VERTS, (100, 149, 237, 64))]:
            pts = _project(verts, K, R, t, scale)
            poly = [tuple(p.astype(int)) for p in pts]
            draw.polygon(poly, fill=color, outline=(128, 128, 128, 100))
        # transition line
        pts = _project(TRANSITION_LINE, K, R, t, scale)
        draw.line([tuple(pts[0].astype(int)), tuple(pts[1].astype(int))],
                  fill=(0, 0, 0, 200), width=2)

    if draw_platform:
        for face in PLATFORM_FACES:
            pts = _project(face, K, R, t, scale)
            poly = [tuple(p.astype(int)) for p in pts]
            draw.polygon(poly, fill=(40, 40, 40, 200), outline=(80, 80, 80, 255))

    return Image.alpha_composite(base, overlay).convert('RGB')


def _render_full_scene(cam_w, cam_h, fi, get_xyz, bodyparts, skeleton_pairs,
                       bp_colors, K, R, t, scale=1.0,
                       marker_radius=2, line_width=2,
                       bg_color=(255, 255, 255),
                       draw_belts=True, draw_platform=True,
                       trail_history=None, trail_radius=None,
                       trail_max_alpha=0.6):
    """
    Render a complete frame: background, scenery, skeleton, bodypart dots.

    Uses cv2.projectPoints for all geometry.

    Parameters
    ----------
    trail_history : list of dict or None
        Past positions_2d dicts (oldest first) for drawing fading trails.
    """
    bg = Image.new('RGB', (cam_w, cam_h), bg_color)

    # scenery
    bg = _draw_scenery_on(bg, K, R, t, scale, draw_belts, draw_platform)

    # project bodyparts and compute camera-space depth for z-ordering
    positions = {}
    depths = {}
    pts_3d, bp_order = [], []
    for bp in bodyparts:
        x, y, z = get_xyz(fi, bp)
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            positions[bp] = (np.nan, np.nan)
        else:
            pts_3d.append([x, y, z])
            bp_order.append(bp)
    if pts_3d:
        pts_3d_arr = np.array(pts_3d)
        pts_2d = _project(pts_3d_arr, K, R, t, scale)
        # Camera-space z: depth = (R @ pt + t)[2] for each point
        t_col = t.reshape(3, 1)
        cam_coords = R @ pts_3d_arr.T + t_col  # 3 x N
        for i, bp in enumerate(bp_order):
            positions[bp] = (pts_2d[i][0], pts_2d[i][1])
            depths[bp] = cam_coords[2, i]  # z in camera space

    all_parts = set(bodyparts)
    bg = _draw_skeleton(bg, all_parts, positions, skeleton_pairs,
                        bp_colors, alpha=1.0, line_width=line_width, depths=depths)
    bg = _draw_labels(bg, all_parts, positions, bp_colors, marker_radius,
                      trail_history=trail_history, trail_radius=trail_radius,
                      trail_max_alpha=trail_max_alpha, depths=depths)
    return bg, positions


# -------------------- main function --------------------

def make_labelling_demo(
    video_path,
    coords_df,
    out_path,
    start_frame,
    end_frame,
    fps=247,
    video_speed=0.1,            # playback speed for video phases (0.1 = 10% real speed)
    animation_speed=0.1,        # playback speed for skeleton animation (0.1 = 10% real speed)
    # --- phase timing ---
    pause_frame=200,            # video frame index at which to pause
    frames_per_label=3,
    skeleton_appear_frames=12,
    cam_fade_frames=20,
    hold_frozen_frames=8,
    # --- panning ---
    pan_total_degrees=200,      # total azimuth rotation during pan
    pan_elev_peak=0,            # peak elevation offset in degrees (0 = level orbit, 90 = bird's eye at midpoint)
    pan_zoom=1.0,               # peak zoom factor at midpoint of orbit (< 1 = zoom out, > 1 = zoom in, 1.0 = no change)
    pan_end_frac=1.0,           # fraction of phase 4 where orbit completes (e.g. 0.8 = finishes at 80%, holds for remaining 20%)
    track_mouse=True,           # shift orbit to follow mean mouse X (keeps subject centred)
    # --- scene ---
    add_belts=True,
    add_platform=True,
    trail_length=5,             # number of past frames to show as fading trail (phase 4 only)
    trail_radius=None,          # trail dot radius (None = same as marker_radius)
    trail_max_alpha=0.6,        # peak opacity of most recent trail dot (0-1)
    cut_end_frames=0,           # number of frames to trim from the end
    label_fadeout_frames=30,    # fade labels/skeleton/trails to invisible over last N frames (0 = off)
    fade_back_frames=30,        # frames to crossfade back to first video frame (0 = off)
    # --- filtering ---
    pcutoff=0.9,
    interpolate=True,
    interp_method='linear',
    interp_max_gap=10,
    smooth=True,
    smooth_method='savgol',     # 'savgol', 'median', or 'butterworth'
    smooth_window=7,            # window length (must be odd) — savgol/median only
    smooth_polyorder=2,         # polynomial order (savgol only)
    smooth_cutoff_hz=10,        # cutoff frequency in Hz (butterworth only)
    smooth_butter_order=2,      # filter order (butterworth only)
    upsample_factor=1,          # interpolate N sub-frames per original frame (1 = no upsampling)
    # --- output ---
    target_width=600,
    gif_width=None,             # GIF resolution (None = same as target_width). Set lower for smaller file size.
    gif_max_mb=10.0,            # target max GIF file size in MB — auto-skips frames to stay under
    marker_radius=2,
    skeleton_line_width=2,
    bg_color=(255, 255, 255),
    save_gif=True,
    save_mp4=True,
):
    """
    Render the labelling demo animation. All 3D rendering uses
    cv2.projectPoints with a virtual pinhole camera — no matplotlib.

    Phase 1: Side-cam video plays up to the pause point.
    Phase 2: Video freezes; scatter points appear on the camera frame
             (projected via real side-camera extrinsics), then skeleton
             lines fade in.
    Phase 3: Video fades away; the skeleton overlay stays in place.
    Phase 4: Skeleton resumes running. A virtual camera starts at the
             real side-camera position and orbits around the scene.

    Parameters
    ----------
    pause_frame : int
        Absolute video frame index at which to pause.
    pan_total_degrees : float
        Total azimuth the virtual camera orbits during phase 4.
    pan_elev_peak : float
        Peak elevation offset in degrees reached at the midpoint of the orbit.
        0 = level orbit. 90 = bird's eye view at midpoint, returning to
        level at the end (sinusoidal arc). Default 0.
    pan_zoom : float
        Peak focal-length scale at the midpoint of the orbit (sinusoidal
        arc, returns to 1.0 at the end). < 1 zooms out, > 1 zooms in,
        1.0 = no zoom change.
    """
    bodyparts = micestuff['bodyparts']
    skeleton_pairs = micestuff['skeleton']
    # Output FPS based on animation speed (phase 4 drives the frame rate)
    out_fps = fps * max(1, upsample_factor) * animation_speed
    # Video frames repeated to match desired video speed at the output FPS
    video_repeat = max(1, round(upsample_factor * animation_speed / video_speed))
    print(f'Speeds: video={video_speed}x, animation={animation_speed}x, '
          f'out_fps={out_fps:.1f}, video_repeat={video_repeat}')
    label_sequence = _build_label_sequence()

    # ---- real side camera ----
    K_real, R_real, t_real = _get_side_camera()
    cam_pos_real = (-R_real.T @ t_real).flatten()
    print(f'Side camera position (world): {cam_pos_real}')

    # ---- pcutoff → slice → interpolate → smooth ----
    if pcutoff is not None:
        coords_df = filter_by_pcutoff(coords_df, pcutoff=pcutoff)

    # ---- slice coordinate data ----
    frames_range = list(range(start_frame, end_frame + 1))
    try:
        cdf = coords_df.loc[frames_range].copy()
    except Exception:
        cdf = coords_df.iloc[start_frame:end_frame + 1].copy()

    cols = []
    for p in bodyparts:
        for c in ('x', 'y', 'z'):
            if (p, c) in cdf.columns:
                cols.append((p, c))
    cdf = cdf.loc[:, cols]
    cdf = cdf.reindex(columns=pd.MultiIndex.from_tuples(cols, names=['part', 'coords']))
    cdf = cdf.reset_index(drop=True)

    # ---- interpolate on slice only (avoids cross-gap filling from full trace) ----
    if interpolate:
        n_before = cdf.isna().sum().sum()
        for col in cdf.columns:
            cdf[col] = cdf[col].interpolate(
                method=interp_method, limit=interp_max_gap,
                limit_area='inside')
        n_after = cdf.isna().sum().sum()
        print(f'Interpolated {n_before - n_after} NaN values on slice '
              f'({interp_method}, max_gap={interp_max_gap}, inside only)')

    # ---- smooth ----
    if smooth:
        if smooth_method == 'butterworth':
            nyq = fps / 2.0
            cutoff_norm = smooth_cutoff_hz / nyq
            if cutoff_norm >= 1.0:
                print(f'Warning: cutoff {smooth_cutoff_hz}Hz >= Nyquist {nyq}Hz, skipping filter')
            else:
                b, a = butter(smooth_butter_order, cutoff_norm, btype='low')
                print(f'Smoothing coordinates (butterworth, cutoff={smooth_cutoff_hz}Hz, '
                      f'order={smooth_butter_order}, Nyquist={nyq:.0f}Hz)')
                for col in cdf.columns:
                    series = cdf[col]
                    valid = series.notna()
                    # filtfilt needs at least 3*max(len(a),len(b)) contiguous samples
                    min_len = 3 * (smooth_butter_order + 1)
                    if valid.sum() < min_len:
                        continue
                    cdf.loc[valid, col] = filtfilt(b, a, series[valid].values)
        else:
            win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            print(f'Smoothing coordinates ({smooth_method}, window={win})')
            for col in cdf.columns:
                series = cdf[col]
                valid = series.notna()
                if valid.sum() < win:
                    continue
                if smooth_method == 'savgol':
                    cdf.loc[valid, col] = savgol_filter(
                        series[valid].values, win, smooth_polyorder)
                elif smooth_method == 'median':
                    cdf.loc[valid, col] = (
                        series[valid].rolling(win, center=True, min_periods=1).median()
                    ).values
                else:
                    raise ValueError(f'Unknown smooth_method: {smooth_method!r}')

    # ---- upsample (temporal interpolation via cubic spline) ----
    if upsample_factor > 1:
        n_orig = len(cdf)
        new_idx = np.arange(0, n_orig - 1 + 1e-9, 1.0 / upsample_factor)
        cdf_up = cdf.reindex(new_idx).interpolate(method='cubicspline')
        cdf_up = cdf_up.reset_index(drop=True)
        cdf = cdf_up
        print(f'Upsampled {n_orig} -> {len(cdf)} frames ({upsample_factor}x, cubic spline)')

    # ---- clip bodyparts past belt end (x > 600mm) ----
    # DLC can confidently predict positions for bodyparts leaving the frame,
    # so pcutoff alone doesn't catch them. Hard-clip at the belt boundary.
    belt_end_x = 575.0
    for bp in bodyparts:
        if (bp, 'x') in cdf.columns:
            past_belt = cdf[(bp, 'x')] > belt_end_x
            if past_belt.any():
                for c in ('x', 'y', 'z'):
                    if (bp, c) in cdf.columns:
                        cdf.loc[past_belt, (bp, c)] = np.nan
    n_clipped = cdf.isna().sum().sum()
    print(f'Clipped bodyparts past x>{belt_end_x}mm')

    n_total = len(cdf)

    # Pause indices: separate for video frames vs coordinate frames
    pause_idx_raw = pause_frame - start_frame
    pause_idx_coord = max(1, min(pause_idx_raw * max(1, upsample_factor), n_total - 2))
    pause_idx_video = pause_idx_raw * video_repeat

    def get_xyz(fi, part):
        return (cdf.loc[fi, (part, 'x')],
                cdf.loc[fi, (part, 'y')],
                cdf.loc[fi, (part, 'z')])

    # ---- extract camera frames ----
    n_video_frames = end_frame - start_frame + 1
    print('Extracting camera frames...')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Could not open video: {video_path}')

    cam_frames_raw = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(n_video_frames):
        ok, frame = cap.read()
        if not ok:
            break
        cam_frames_raw.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    orig_w = cam_frames_raw[0].width
    if target_width and orig_w > target_width:
        px_scale = target_width / orig_w
        new_h = int(cam_frames_raw[0].height * px_scale)
        cam_frames_scaled = [f.resize((target_width, new_h), Image.LANCZOS)
                             for f in cam_frames_raw]
    else:
        px_scale = 1.0
        cam_frames_scaled = cam_frames_raw

    # Duplicate video frames for desired video playback speed
    cam_frames = []
    for f in cam_frames_scaled:
        cam_frames.extend([f] * video_repeat)

    cam_w, cam_h = cam_frames[0].size
    print(f'Output frame size: {cam_w} x {cam_h}')

    # ---- project bodyparts at the pause frame (real camera) ----
    def get_positions_real(fi):
        positions = {}
        pts_3d, bp_order = [], []
        for bp in bodyparts:
            x, y, z = get_xyz(fi, bp)
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                positions[bp] = (np.nan, np.nan)
            else:
                pts_3d.append([x, y, z])
                bp_order.append(bp)
        if pts_3d:
            pts_2d = _project(np.array(pts_3d), K_real, R_real, t_real, px_scale)
            for bp, pt in zip(bp_order, pts_2d):
                positions[bp] = (pt[0], pt[1])
        return positions

    pause_positions = get_positions_real(pause_idx_coord)

    # ---- bodypart colours (viridis, 0-255 for PIL) ----
    viridis = plt.get_cmap('viridis')
    bp_colors = {}
    for i, bp in enumerate(bodyparts):
        rgba = viridis(i / max(1, len(bodyparts) - 1))
        bp_colors[bp] = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

    # ---- virtual camera for panning (phase 4) ----
    # Compute orbit parameters from the real camera position.
    # Orbit centre z is at approximate mouse body height (~10mm above belt).
    # When track_mouse is on, orbit centre X is set to the mouse's mean X
    # over phase 4, so the orbit revolves around the subject. This keeps
    # the mouse centred from all angles without any tracking offset.
    orbit_centre_x = 235.0
    if track_mouse:
        phase4_xs = []
        for fi in range(min(pause_idx_coord, len(cdf)), len(cdf)):
            for bp in bodyparts:
                if (bp, 'x') in cdf.columns:
                    x = cdf.loc[fi, (bp, 'x')]
                    if not np.isnan(x):
                        phase4_xs.append(x)
        if phase4_xs:
            orbit_centre_x = np.mean(phase4_xs)
    orbit_centre = np.array([orbit_centre_x, BELT_W / 2, 10.0])
    rel = cam_pos_real - orbit_centre
    orbit_radius = np.linalg.norm(rel)
    start_azim = np.degrees(np.arctan2(rel[1], rel[0]))
    start_elev = np.degrees(np.arcsin(rel[2] / orbit_radius))

    print(f'Orbit: radius={orbit_radius:.0f}mm, '
          f'start_azim={start_azim:.1f}deg, start_elev={start_elev:.1f}deg')

    def orbit_camera(azim_deg, elev_deg):
        """
        Camera on the orbit sphere. Orientation is computed by composing
        the real camera R with separate azimuth and elevation rotations —
        this guarantees the first frame is exactly R_real (no jump / tilt)
        and elevation changes produce clean tilt without roll.
        """
        az = np.radians(azim_deg)
        el = np.radians(elev_deg)
        pos = orbit_centre + orbit_radius * np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])

        delta_az = np.radians(azim_deg - start_azim)
        delta_el = np.radians(elev_deg - start_elev)

        # Azimuth: rotation around world Z axis
        if abs(delta_az) > 1e-10:
            R_az, _ = cv2.Rodrigues(np.array([0., 0., delta_az]))
        else:
            R_az = np.eye(3)

        # Elevation: rotation around the horizontal axis perpendicular to
        # the radial direction at the *current* azimuth. This axis is
        # (sin(az), -cos(az), 0), which tilts the camera cleanly up/down.
        if abs(delta_el) > 1e-10:
            el_axis = np.array([np.sin(az), -np.cos(az), 0.])
            R_el, _ = cv2.Rodrigues(el_axis * delta_el)
        else:
            R_el = np.eye(3)

        R_delta = R_el @ R_az

        # Compose: camera orientation = R_real rotated by the orbit motion
        R = R_real @ R_delta.T
        t = (-R @ pos).reshape(3, 1)
        return R, t

    def smoothstep(x):
        """Smooth ease-in/ease-out: 0→1 with zero derivative at endpoints."""
        x = max(0.0, min(1.0, x))
        return x * x * (3 - 2 * x)

    def smootherstep(x):
        """Extra-smooth ease: 0→1 with zero 1st AND 2nd derivative at endpoints."""
        x = max(0.0, min(1.0, x))
        return x * x * x * (x * (6 * x - 15) + 10)

    def blend(img_a, img_b, alpha):
        """alpha=1 -> all img_a, alpha=0 -> all img_b."""
        a = np.asarray(img_a, dtype=np.float32)
        b = np.asarray(img_b, dtype=np.float32)
        return Image.fromarray(
            (alpha * a + (1 - alpha) * b).clip(0, 255).astype(np.uint8))

    # ---- build output frames ----
    output_frames = []
    all_parts = set(bodyparts)
    blank_bg = Image.new('RGB', (cam_w, cam_h), bg_color)

    # Phase 1: camera video plays up to pause point
    print(f'Phase 1: video plays ({pause_idx_video} frames at {video_speed}x)')
    for i in range(pause_idx_video):
        output_frames.append(cam_frames[i])

    frozen_cam = cam_frames[pause_idx_video]

    # Phase 2a: hold frozen camera frame
    print(f'Phase 2a: hold frozen frame ({hold_frozen_frames} frames)')
    for _ in range(hold_frozen_frames):
        output_frames.append(frozen_cam)

    # Phase 2b: labels appear on frozen camera frame
    n_label_steps = len(label_sequence)
    total_label_frames = n_label_steps * frames_per_label
    print(f'Phase 2b: label reveal ({n_label_steps} steps x {frames_per_label} '
          f'= {total_label_frames} frames)')
    visible = set()
    for parts in label_sequence:
        visible.update(parts)
        labelled = _draw_labels(frozen_cam, visible, pause_positions, bp_colors,
                                marker_radius=marker_radius)
        for _ in range(frames_per_label):
            output_frames.append(labelled)

    # Phase 2c: skeleton connections fade in
    print(f'Phase 2c: skeleton connections ({skeleton_appear_frames} frames)')
    labelled_all = _draw_labels(frozen_cam, all_parts, pause_positions, bp_colors,
                                marker_radius=marker_radius)
    for i in range(skeleton_appear_frames):
        t_frac = (i + 1) / skeleton_appear_frames
        frame = _draw_skeleton(labelled_all, all_parts, pause_positions,
                               skeleton_pairs, bp_colors, alpha=t_frac,
                               line_width=skeleton_line_width)
        output_frames.append(frame)

    # Phase 3: video fades away, skeleton overlay stays in place.
    # No position interpolation needed — K_real is used throughout.
    print(f'Phase 3: camera fades away ({cam_fade_frames} frames)')
    skeleton_on_cam = _draw_full_overlay(
        frozen_cam, all_parts, pause_positions, bp_colors, skeleton_pairs,
        marker_radius=marker_radius, line_width=skeleton_line_width)
    skeleton_on_blank = _draw_full_overlay(
        blank_bg, all_parts, pause_positions, bp_colors, skeleton_pairs,
        marker_radius=marker_radius, line_width=skeleton_line_width)
    for i in range(cam_fade_frames):
        t_frac = (i + 1) / cam_fade_frames
        frame = blend(skeleton_on_cam, skeleton_on_blank, 1.0 - t_frac)
        output_frames.append(frame)

    # Phase 4: skeleton runs, virtual camera orbits.
    # orbit_camera composes R_real with the incremental orbit rotation,
    # so at frame 0 it returns exactly R_real/t_real — no jump at all.
    # When track_mouse is on, orbit_centre is already shifted to the
    # mouse's mean X, so no per-frame offset is needed.
    remaining = n_total - pause_idx_coord
    end_azim = start_azim + pan_total_degrees

    print(f'Phase 4: running + pan ({remaining} frames, '
          f'azim {start_azim:.0f} -> {end_azim:.0f}deg, '
          f'elev_peak={pan_elev_peak:.0f}deg, '
          f'zoom_peak={pan_zoom:.2f}, '
          f'orbit_centre_x={orbit_centre[0]:.0f}mm)')

    trail_buffer_3d = []  # rolling buffer of 3D positions for trail re-projection

    for i in range(remaining):
        fi = pause_idx_coord + i
        t_frac = i / max(1, remaining - 1)
        # Scale t_frac so the orbit completes at pan_end_frac of the phase,
        # then holds at the home position for the rest.
        t_scaled = min(t_frac / pan_end_frac, 1.0) if pan_end_frac > 0 else 1.0
        # Ease the orbit progression: gentle start/stop, faster in the middle.
        # smootherstep has zero 1st and 2nd derivative at endpoints, so the
        # camera eases out of and back into the side-view position gracefully.
        t_orbit = smootherstep(t_scaled)
        azim = start_azim + (end_azim - start_azim) * t_orbit
        # Elevation arc: sharp peak at midpoint of orbit progress (not time).
        elev = start_elev + pan_elev_peak * np.sin(np.pi * t_orbit) ** 4

        R_orb, t_orb = orbit_camera(azim, elev)

        # Zoom arc tied to orbit progress (not time)
        zoom = 1.0 + (pan_zoom - 1.0) * np.sin(np.pi * t_orbit)
        K_use = K_real.copy()
        K_use[0, 0] *= zoom
        K_use[1, 1] *= zoom

        # Re-project trail 3D positions with current camera
        trail_hist = None
        if trail_length > 0 and trail_buffer_3d:
            trail_hist = []
            for past_3d in trail_buffer_3d:
                past_2d = {}
                pts_3d_list, bp_order = [], []
                for bp in bodyparts:
                    xyz = past_3d.get(bp)
                    if xyz is None or np.isnan(xyz[0]):
                        past_2d[bp] = (np.nan, np.nan)
                    else:
                        pts_3d_list.append(list(xyz))
                        bp_order.append(bp)
                if pts_3d_list:
                    pts_2d = _project(np.array(pts_3d_list), K_use, R_orb, t_orb, px_scale)
                    for bp, pt in zip(bp_order, pts_2d):
                        past_2d[bp] = (pt[0], pt[1])
                trail_hist.append(past_2d)

        frame, positions = _render_full_scene(
            cam_w, cam_h, fi, get_xyz, bodyparts, skeleton_pairs,
            bp_colors, K_use, R_orb, t_orb, scale=px_scale,
            marker_radius=marker_radius, line_width=skeleton_line_width,
            bg_color=bg_color,
            draw_belts=add_belts, draw_platform=add_platform,
            trail_history=trail_hist, trail_radius=trail_radius,
            trail_max_alpha=trail_max_alpha)

        # Fade labels/skeleton/trails out over the last N frames before trim
        if label_fadeout_frames > 0:
            effective_end = remaining - cut_end_frames * max(1, upsample_factor)
            frames_to_end = effective_end - i
            if frames_to_end <= label_fadeout_frames:
                label_alpha = max(0.0, frames_to_end / label_fadeout_frames)
                # Render scenery-only frame and blend
                scenery_only = Image.new('RGB', (cam_w, cam_h), bg_color)
                scenery_only = _draw_scenery_on(
                    scenery_only, K_use, R_orb, t_orb, px_scale,
                    add_belts, add_platform)
                frame = blend(frame, scenery_only, label_alpha)

        output_frames.append(frame)

        # Store 3D positions in trail buffer
        if trail_length > 0:
            positions_3d = {}
            for bp in bodyparts:
                x, y, z = get_xyz(fi, bp)
                positions_3d[bp] = (x, y, z)
            trail_buffer_3d.append(positions_3d)
            if len(trail_buffer_3d) > trail_length:
                trail_buffer_3d.pop(0)

    # ---- trim end frames ----
    if cut_end_frames > 0:
        output_frames = output_frames[:-cut_end_frames]
        print(f'Trimmed last {cut_end_frames} frames')

    # Phase 5: crossfade from last frame back to first video frame (for GIF loop)
    if fade_back_frames > 0:
        print(f'Phase 5: fade back to first video frame ({fade_back_frames} frames)')
        last_frame = output_frames[-1]
        first_frame = cam_frames[0]
        for i in range(fade_back_frames):
            t_frac = (i + 1) / fade_back_frames
            output_frames.append(blend(last_frame, first_frame, 1.0 - t_frac))

    # ---- save outputs ----
    base_path = os.path.splitext(out_path)[0]

    if save_mp4:
        mp4_path = base_path + '.mp4'
        h, w = np.array(output_frames[0]).shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(mp4_path, fourcc, out_fps, (w, h))
        for frame in output_frames:
            writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        writer.release()
        size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
        print(f'Saved MP4: {mp4_path}  ({len(output_frames)} frames, {size_mb:.1f} MB)')

    if save_gif:
        gif_path = base_path + '.gif'
        # GIF format: delay is in centiseconds, and most renderers clamp
        # anything below 20ms to ~100ms. So we must skip frames to keep
        # the effective delay >= 20ms while matching the intended speed.
        min_gif_delay_ms = 20
        raw_delay_ms = 1000 / out_fps
        if raw_delay_ms < min_gif_delay_ms:
            frame_skip = int(np.ceil(min_gif_delay_ms / raw_delay_ms))
            gif_duration_ms = round(raw_delay_ms * frame_skip)
            gif_frames = output_frames[::frame_skip]
            print(f'GIF: skipping every {frame_skip} frames to avoid '
                  f'GIF min-delay issue ({raw_delay_ms:.1f}ms -> {gif_duration_ms}ms)')
        else:
            gif_duration_ms = round(raw_delay_ms)
            gif_frames = output_frames

        # Downscale GIF frames if gif_width is set
        gw = gif_width if gif_width is not None else target_width
        if gw != target_width:
            orig_w, orig_h = gif_frames[0].size
            scale = gw / orig_w
            gh = int(orig_h * scale)
            gif_frames = [f.resize((gw, gh), Image.LANCZOS) for f in gif_frames]
            print(f'GIF: resized {orig_w}x{orig_h} -> {gw}x{gh}')

        # Build a custom 256-colour palette that guarantees all bodypart colours
        # are included, then fills remaining slots from sampled frames.
        n_reserved = len(bp_colors)
        n_image_colors = 256 - n_reserved

        # Sample frames to extract the most common image colours
        n_samples = min(20, len(gif_frames))
        sample_indices = np.linspace(0, len(gif_frames) - 1, n_samples, dtype=int)
        sample_w = gif_frames[0].width
        sample_h = gif_frames[0].height
        composite = Image.new('RGB', (sample_w, sample_h * n_samples))
        for i, idx in enumerate(sample_indices):
            composite.paste(gif_frames[idx], (0, i * sample_h))
        image_palette_img = composite.quantize(colors=n_image_colors, method=0, dither=0)
        image_palette = image_palette_img.getpalette()  # flat list [r,g,b,r,g,b,...]

        # Build full palette: bodypart colours first, then image colours
        full_palette = []
        for bp in bodyparts:
            col = bp_colors[bp]
            full_palette.extend([col[0], col[1], col[2]])
        # Add image-derived colours
        full_palette.extend(image_palette[:n_image_colors * 3])
        # Pad to 256 entries if needed
        while len(full_palette) < 768:
            full_palette.extend([0, 0, 0])

        # Create a palette image for quantization
        palette_img = Image.new('P', (1, 1))
        palette_img.putpalette(full_palette[:768])

        quantized = []
        for f in gif_frames:
            q = f.quantize(palette=palette_img, dither=1)
            quantized.append(q)

        quantized[0].save(
            gif_path, save_all=True, append_images=quantized[1:],
            duration=gif_duration_ms, loop=0, optimize=False)
        size_mb = os.path.getsize(gif_path) / (1024 * 1024)
        print(f'Saved GIF: {gif_path}  ({len(gif_frames)} frames, '
              f'{gif_duration_ms} ms/frame, {size_mb:.1f} MB)')

    print(f'Total output frames: {len(output_frames)}')
    return output_frames


# -------------------- example usage --------------------

if __name__ == '__main__':
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MEDIA_DIR = os.path.join(PROJECT_ROOT, 'docs', 'media')
    os.makedirs(MEDIA_DIR, exist_ok=True)

    VIDEO_PATH  = r'H:\movement\sample_data\short_form\DLC_single-mouse_DBTravelator_sideview_run41.avi'
    COORDS_FILE = r'H:\movement\sample_data\short_form\DLC_single-mouse_DBTravelator_3D_run41_corrected.h5'
    FRAME_START = 0
    FRAME_END   = 387
    FPS         = 247
    VIDEO_SPEED = 1.00
    ANIM_SPEED  = 0.06
    PCUTOFF     = 0.95

    OUT_PATH = os.path.join(MEDIA_DIR, 'labelling_demo')

    coords_df = pd.read_hdf(COORDS_FILE)
    if (coords_df.columns.nlevels == 3
            and coords_df.columns.names == ['scorer', 'bodyparts', 'coords']):
        coords_df.columns = coords_df.columns.droplevel('scorer')

    make_labelling_demo(
        video_path=VIDEO_PATH,
        coords_df=coords_df,
        out_path=OUT_PATH,
        start_frame=FRAME_START,
        end_frame=FRAME_END,
        fps=FPS,
        video_speed=VIDEO_SPEED,
        animation_speed=ANIM_SPEED,
        # --- timing ---
        pause_frame=190,
        frames_per_label=3,
        skeleton_appear_frames=12,
        cam_fade_frames=20,
        hold_frozen_frames=8,
        # --- panning ---
        pan_total_degrees=360,
        pan_elev_peak=80,           # bird's eye view at midpoint (sinusoidal arc)
        pan_zoom=0.5,               # peak zoom-out at midpoint (< 1 = wider FOV, 1.0 = no change)
        pan_end_frac=0.70,          # orbit finishes at 85% of phase 4, holds home view for remaining 15%
        track_mouse=True,           # shift orbit to follow mean mouse X position
        # --- scene ---
        add_belts=True,
        add_platform=True,
        trail_length=50,             # fading scatter trail (phase 4 only), 0 = off
        trail_radius=1,             # trail dot radius (None = same as marker_radius)
        trail_max_alpha=1,        # peak opacity of most recent trail dot (0-1)
        cut_end_frames=20,          # trim last N frames (e.g. mouse disappearing)
        label_fadeout_frames=30,    # fade labels/skeleton/trails out over last N frames
        fade_back_frames=20,        # crossfade back to first video frame for GIF loop
        # --- filtering ---
        pcutoff=PCUTOFF,
        interpolate=True,
        interp_method='linear',
        interp_max_gap=40,
        smooth=True,
        smooth_method='butterworth', # 'savgol', 'median', or 'butterworth'
        smooth_window=9,            # must be odd (savgol/median only)
        smooth_polyorder=3,         # savgol only
        smooth_cutoff_hz=10,        # low-pass cutoff in Hz (butterworth only)
        smooth_butter_order=2,      # filter order (butterworth only)
        upsample_factor=2,          # interpolate sub-frames (4x = cubic spline)
        # --- output ---
        target_width=1920,          # MP4 resolution
        gif_width=550,              # GIF resolution (smaller = smaller file)
        marker_radius=4,
        skeleton_line_width=2,
        bg_color=(255, 255, 255),
        save_gif=True,
        save_mp4=True,
    )
