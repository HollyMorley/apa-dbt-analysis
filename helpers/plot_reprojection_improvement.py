import os
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pycalib.calib import triangulate

from helpers.CalibrateCams import BasicCalibration
from helpers.utils_3d_reconstruction import CameraData

# ── paths ──────────────────────────────────────────────────────────────
CALIB_BASE = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\CameraCalibration"
LABEL_BASE = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling"
SAVE_DIR = r"H:\Characterisation_v2\calibration_optimisation"

VIEWS = ["side", "front", "overhead"]
CALIB_LANDMARKS = {"StartPlatL", "StepL", "StartPlatR", "StepR",
                   "Door", "TransitionL", "TransitionR"}


# ── helpers ────────────────────────────────────────────────────────────
def load_calibration_csv(path):
    """Load a calibration_labels CSV into the DataFrame format BasicCalibration expects."""
    return pd.read_csv(path)


def extrinsics_from_calib(calib_csv_path):
    """Compute camera extrinsics + intrinsics from a calibration labels CSV."""
    df = load_calibration_csv(calib_csv_path)
    calib = BasicCalibration(df)
    extrinsics = calib.estimate_cams_pose()
    intrinsics = calib.cameras_intrinsics
    return extrinsics, intrinsics


def load_bodypart_labels(folder_name):
    """
    Load manually-labelled body part coordinates from all 3 camera views.

    Returns
    -------
    dict  :  {view: {frame_num: {bodypart: (x, y) or None}}}
    """
    parts = folder_name.rsplit("_", 1)
    base, num = parts[0], parts[1]

    data = {}
    for view in VIEWS:
        view_cap = view.capitalize()
        h5_path = os.path.join(
            LABEL_BASE, view_cap, f"{base}_{view}_{num}", "CollectedData_Holly.h5"
        )
        if not os.path.exists(h5_path):
            print(f"  Warning: {h5_path} not found, skipping {view}")
            data[view] = {}
            continue

        df = pd.read_hdf(h5_path)
        view_data = {}
        for idx in df.index:
            img_name = idx[2]  # e.g. 'img165731.png'
            frame_num = int(re.search(r"img(\d+)", img_name).group(1))
            row = df.loc[idx]

            bodyparts = {}
            for bp in row.index.get_level_values("bodyparts").unique():
                x = row[("Holly", bp, "x")]
                y = row[("Holly", bp, "y")]
                if pd.notna(x) and pd.notna(y):
                    bodyparts[bp] = (float(x), float(y))
                else:
                    bodyparts[bp] = None
            view_data[frame_num] = bodyparts
        data[view] = view_data

    return data


def get_projection_matrix(extrinsics, intrinsics, view):
    """Build 3x4 projection matrix P = K @ [R|t]."""
    K = intrinsics[view]
    R = extrinsics[view]["rotm"]
    t = extrinsics[view]["tvec"]
    if t.ndim == 1:
        t = t[:, np.newaxis]
    return K @ np.hstack((R, t))


def _reproject_error(point_3d, orig_2d, extrinsics, intrinsics, view):
    """Compute reprojection error for a single 3D point in one view."""
    projected, _ = cv2.projectPoints(
        point_3d.reshape(1, 3),
        extrinsics[view]["rvec"],
        extrinsics[view]["tvec"],
        intrinsics[view],
        np.array([]),
    )
    proj_2d = projected[0].flatten()
    return np.sqrt((proj_2d[0] - orig_2d[0]) ** 2 + (proj_2d[1] - orig_2d[1]) ** 2)


def compute_paired_errors(pre_ext, post_ext, intrinsics, labels_data):
    """
    Compute paired pre/post reprojection errors for each (frame, bodypart, view).

    Returns
    -------
    pre_errors : dict  {view: [error_px, ...]}
    post_errors : dict  {view: [error_px, ...]}

    The i-th element of pre_errors[view] and post_errors[view] correspond
    to the same (frame, bodypart) observation — they are paired.
    """
    all_frames = set()
    for v in VIEWS:
        all_frames |= set(labels_data[v].keys())

    all_bodyparts = set()
    for view in VIEWS:
        for frame_data in labels_data[view].values():
            all_bodyparts |= set(frame_data.keys())
    bodyparts = sorted(all_bodyparts - CALIB_LANDMARKS)

    pre_errors = {v: [] for v in VIEWS}
    post_errors = {v: [] for v in VIEWS}

    for frame in sorted(all_frames):
        for bp in bodyparts:
            # Gather 2D observations
            pts_2d = []
            available_views = []
            for view in VIEWS:
                if frame not in labels_data[view]:
                    continue
                coord = labels_data[view][frame].get(bp)
                if coord is None:
                    continue
                pts_2d.append(coord)
                available_views.append(view)

            if len(pts_2d) < 2:
                continue

            pts_2d_arr = np.array(pts_2d, dtype=np.float64)

            # Triangulate with pre extrinsics
            P_pre = np.array([get_projection_matrix(pre_ext, intrinsics, v) for v in available_views])
            pt3d_pre = triangulate(pts_2d_arr, P_pre)

            # Triangulate with post extrinsics
            P_post = np.array([get_projection_matrix(post_ext, intrinsics, v) for v in available_views])
            pt3d_post = triangulate(pts_2d_arr, P_post)

            # Only keep if both triangulations succeeded
            if pt3d_pre is None or len(pt3d_pre) < 3:
                continue
            if pt3d_post is None or len(pt3d_post) < 3:
                continue

            pt3d_pre = pt3d_pre[:3].astype(np.float64)
            pt3d_post = pt3d_post[:3].astype(np.float64)

            for view, orig_2d in zip(available_views, pts_2d):
                err_pre = _reproject_error(pt3d_pre, orig_2d, pre_ext, intrinsics, view)
                err_post = _reproject_error(pt3d_post, orig_2d, post_ext, intrinsics, view)
                pre_errors[view].append(err_pre)
                post_errors[view].append(err_post)

    return pre_errors, post_errors


# ── plotting ───────────────────────────────────────────────────────────
def plot_pooled(all_pre, all_post, save_dir):
    """
    Single figure, 3 subplots (one per camera). All bodyparts and recordings
    pooled. Violin plot with statistical tests.
    """
    from scipy.stats import wilcoxon, mannwhitneyu

    n_recordings = len(all_pre["_n_recordings"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    stats_lines = []
    stats_lines.append(f"Statistical tests (n_recordings={n_recordings})")
    stats_lines.append("=" * 60)

    for ax, view in zip(axes, VIEWS):
        pre_vals_raw = np.array(all_pre[view])
        post_vals_raw = np.array(all_post[view])

        # Remove extreme outliers (>99.5th percentile of combined data)
        combined = np.concatenate([pre_vals_raw, post_vals_raw])
        clip_upper = np.percentile(combined, 99.5)
        pre_vals = pre_vals_raw[pre_vals_raw <= clip_upper]
        post_vals = post_vals_raw[post_vals_raw <= clip_upper]
        n_clipped = (len(pre_vals_raw) - len(pre_vals)) + (len(post_vals_raw) - len(post_vals))

        # Violin plot
        parts = ax.violinplot(
            [pre_vals, post_vals], positions=[0, 1],
            showmeans=True, showmedians=True, showextrema=False,
        )
        # Colour the violins
        colors = ["#d62728", "#1f77b4"]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("grey")
        parts["cmedians"].set_linestyle("--")

        # Stats on full (unclipped) data
        pre_mean, post_mean = np.mean(pre_vals_raw), np.mean(post_vals_raw)
        pre_median, post_median = np.median(pre_vals_raw), np.median(post_vals_raw)

        # Paired Wilcoxon (samples are matched: same frames/bodyparts)
        n = min(len(pre_vals_raw), len(post_vals_raw))
        wilcox_stat, wilcox_p = wilcoxon(pre_vals_raw[:n], post_vals_raw[:n])
        # Also Mann-Whitney U (unpaired, for robustness)
        mwu_stat, mwu_p = mannwhitneyu(pre_vals_raw, post_vals_raw, alternative="two-sided")

        # Significance stars
        def p_to_stars(p):
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            return "n.s."

        stars = p_to_stars(wilcox_p)

        ax.set_ylim(bottom=-0.1)
        # Draw significance bracket inside the plot
        ymax = ax.get_ylim()[1]
        bracket_y = ymax * 0.85
        tick_y = ymax * 0.82
        ax.plot([0, 0, 1, 1], [tick_y, bracket_y, bracket_y, tick_y],
                color="black", lw=1)
        ax.text(0.5, bracket_y + ymax * 0.01, stars,
                ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{view.capitalize()} camera")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"])
        ax.grid(axis="y", alpha=0.3)

        # Legend-style text box with stats
        textstr = (
            f"Pre:  mean={pre_mean:.2f}, med={pre_median:.2f}\n"
            f"Post: mean={post_mean:.2f}, med={post_median:.2f}\n"
            f"n={n}"
        )
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Collect stats for printing
        pct_change_mean = (post_mean - pre_mean) / pre_mean * 100
        pct_change_med = (post_median - pre_median) / pre_median * 100
        stats_lines.append(f"\n{view.upper()} camera (n={n}):")
        stats_lines.append(f"  Mean:   {pre_mean:.3f} -> {post_mean:.3f} px ({pct_change_mean:+.1f}%)")
        stats_lines.append(f"  Median: {pre_median:.3f} -> {post_median:.3f} px ({pct_change_med:+.1f}%)")
        stats_lines.append(f"  Wilcoxon signed-rank: W={wilcox_stat:.0f}, p={wilcox_p:.2e} {stars}")
        stats_lines.append(f"  Mann-Whitney U:       U={mwu_stat:.0f}, p={mwu_p:.2e}")

    axes[0].set_ylabel("Reprojection error (px)")
    fig.suptitle(
        "Reprojection error: pre vs post calibration optimisation\n"
        f"(all body parts and recordings pooled, n_recordings={n_recordings})",
        fontsize=11, y=0.98,
    )
    fig.subplots_adjust(top=0.85, bottom=0.1, wspace=0.3)

    save_path = os.path.join(save_dir, "reprojection_improvement_pooled")
    fig.savefig(save_path + ".png", dpi=200)
    fig.savefig(save_path + ".svg")
    plt.close(fig)
    print(f"Plot saved to {save_path}.png")

    # Print and save stats
    stats_text = "\n".join(stats_lines)
    print("\n" + stats_text)
    with open(os.path.join(save_dir, "reprojection_stats.txt"), "w") as f:
        f.write(stats_text)


def plot_deltas(all_pre, all_post, save_dir):
    """
    Violin plot of paired deltas (post - pre) for each camera view.
    Negative values = improvement (post error < pre error).
    """
    from scipy.stats import wilcoxon

    n_recordings = len(all_pre["_n_recordings"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    for ax, view in zip(axes, VIEWS):
        pre = np.array(all_pre[view])
        post = np.array(all_post[view])
        deltas = post - pre  # negative = improvement

        # Clip for violin display
        clip_lo, clip_hi = np.percentile(deltas, [0.5, 99.5])
        deltas_clipped = deltas[(deltas >= clip_lo) & (deltas <= clip_hi)]

        parts = ax.violinplot(
            [deltas_clipped], positions=[0],
            showmeans=True, showmedians=True, showextrema=False,
        )
        parts["bodies"][0].set_facecolor("#7f7f7f")
        parts["bodies"][0].set_alpha(0.6)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("grey")
        parts["cmedians"].set_linestyle("--")

        # Zero line
        ax.axhline(0, color="black", lw=0.8, ls=":")

        # Stats (on full unclipped data)
        mean_delta = np.mean(deltas)
        median_delta = np.median(deltas)
        pct_improved = np.sum(deltas < 0) / len(deltas) * 100
        wilcox_stat, wilcox_p = wilcoxon(deltas)

        if wilcox_p < 0.001:
            stars = "***"
        elif wilcox_p < 0.01:
            stars = "**"
        elif wilcox_p < 0.05:
            stars = "*"
        else:
            stars = "n.s."

        textstr = (
            f"mean={mean_delta:+.3f} px\n"
            f"median={median_delta:+.3f} px\n"
            f"{pct_improved:.1f}% improved\n"
            f"n={len(deltas)}\n"
            f"Wilcoxon p={wilcox_p:.1e} {stars}"
        )
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(f"{view.capitalize()} camera")
        ax.set_xticks([0])
        ax.set_xticklabels([f"\u0394 error"])
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Post \u2212 Pre reprojection error (px)")
    fig.suptitle(
        "Change in reprojection error per labelled point after optimisation\n"
        f"(negative = improvement, n_recordings={n_recordings})",
        fontsize=11, y=0.98,
    )
    fig.subplots_adjust(top=0.82, bottom=0.1, wspace=0.3)

    save_path = os.path.join(save_dir, "reprojection_improvement_deltas")
    fig.savefig(save_path + ".png", dpi=200)
    fig.savefig(save_path + ".svg")
    plt.close(fig)
    print(f"Delta plot saved to {save_path}.png")


# ── main ───────────────────────────────────────────────────────────────
def process_recording(folder_name):
    """Process a single recording: compute pre/post reprojection errors."""
    calib_dir = os.path.join(CALIB_BASE, folder_name)
    pre_path = os.path.join(calib_dir, "calibration_labels.csv")
    post_path = os.path.join(calib_dir, "calibration_labels_enhanced.csv")

    if not os.path.exists(pre_path) or not os.path.exists(post_path):
        return None

    print(f"Processing {folder_name}...")

    pre_ext, intrinsics = extrinsics_from_calib(pre_path)
    post_ext, _ = extrinsics_from_calib(post_path)

    labels_data = load_bodypart_labels(folder_name)
    if not any(labels_data[v] for v in VIEWS):
        print(f"  No body part labels found, skipping.")
        return None

    pre_errors, post_errors = compute_paired_errors(pre_ext, post_ext, intrinsics, labels_data)

    return pre_errors, post_errors


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    folders = [
        f for f in os.listdir(CALIB_BASE)
        if os.path.isdir(os.path.join(CALIB_BASE, f))
        and os.path.exists(os.path.join(CALIB_BASE, f, "calibration_labels.csv"))
        and os.path.exists(os.path.join(CALIB_BASE, f, "calibration_labels_enhanced.csv"))
    ]
    print(f"Found {len(folders)} recordings with pre+post calibration labels.\n")

    # Pool errors across all recordings
    all_pre = {v: [] for v in VIEWS}
    all_post = {v: [] for v in VIEWS}
    n_recordings = 0

    for folder in sorted(folders):
        result = process_recording(folder)
        if result is not None:
            pre_errors, post_errors = result
            n_recordings += 1
            for v in VIEWS:
                all_pre[v].extend(pre_errors[v])
                all_post[v].extend(post_errors[v])
            n_pre = sum(len(pre_errors[v]) for v in VIEWS)
            n_post = sum(len(post_errors[v]) for v in VIEWS)
            print(f"  {n_pre} pre / {n_post} post error samples\n")

    all_pre["_n_recordings"] = list(range(n_recordings))  # store count for title
    all_post["_n_recordings"] = list(range(n_recordings))

    if n_recordings > 0:
        plot_pooled(all_pre, all_post, SAVE_DIR)
        plot_deltas(all_pre, all_post, SAVE_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
