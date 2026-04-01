"""Composite figure of mouse pose across experimental conditions."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

def make_mouse_composite(video_path: str, final_frame: int,
                         interval: int = 50, n_prev: int = 6, n_after: int = 1,
                         output_path: str = r"H:\Dual-belt_APAs\Plots\Jan25\Characterisation\Tracking\composite_max.svg"):
    """
    Extracts `n_prev` frames before `final_frame` at `interval` spacing,
    the `final_frame` itself, and `n_after` frames after at the same interval,
    then averages them into a single composite image.
    """
    # build list of frame indices
    prev_idxs = [final_frame - interval * i for i in range(n_prev, 0, -1)]
    after_idxs = [final_frame + interval * i for i in range(1, n_after + 1)]
    frame_idxs = prev_idxs + [final_frame] + after_idxs

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frames = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: could not read frame {idx}")
            continue
        # convert BGR→RGB and to float
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32))

    cap.release()

    if not frames:
        raise RuntimeError("No frames were read; check your indices and file path.")

    # compute average composite
    comp = np.max(frames, axis=0).astype(np.uint8)

    mean_comp = np.mean(frames, axis=0).astype(np.uint8)
    strong = cv2.addWeighted(comp, 0.5, mean_comp, 0.5, 0)

    final = comp

    fig, ax = plt.subplots(figsize=(8, 4), dpi=700)
    ax.imshow(final)
    ax.axis('off')

    output_svg = r"H:\Dual-belt_APAs\Plots\Jan25\Characterisation\Tracking\composite_max.svg"
    fig.savefig(output_svg,
                format='svg',
                bbox_inches='tight',  # trim whitespace
                pad_inches=0)  # no padding
    plt.close(fig)
    print(f"SVG saved to {output_svg}")


if __name__ == "__main__":
    # --- EDIT THESE ---
    video_path = r"H:\Dual-belt_APAs\videos\Round_3\HM_20230316_APACharExt_FAA-1035246_LR_side_1.avi"
    final_frame = 358161
    # ------------------

    make_mouse_composite(video_path, final_frame)
