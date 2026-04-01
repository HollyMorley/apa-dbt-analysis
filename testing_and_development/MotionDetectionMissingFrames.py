import cv2
import numpy as np
from multiprocessing import Pool, cpu_count


def load_video_chunk(path, start_frame, num_frames):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames


def detect_motion(frames):
    motion_metrics = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion_metric = np.sum(thresh)
        motion_metrics.append(motion_metric)
    return motion_metrics


def process_chunk(args):
    path, start_frame, num_frames = args
    frames = load_video_chunk(path, start_frame, num_frames)
    return detect_motion(frames)


def get_total_frames(path):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def parallel_motion_detection(video_path, chunk_size=1000):
    total_frames = get_total_frames(video_path)
    num_chunks = total_frames // chunk_size + (1 if total_frames % chunk_size else 0)
    chunks = [(video_path, i * chunk_size, chunk_size) for i in range(num_chunks)]

    with Pool(cpu_count()) as pool:
        motion_metrics = pool.map(process_chunk, chunks)

    return [metric for sublist in motion_metrics for metric in sublist]


def compare_motion(motion1, motion2):
    best_offset = 0
    min_diff = float('inf')
    for offset in range(-len(motion1) // 2, len(motion1) // 2):
        diff = sum((motion1[i] - motion2[i + offset]) ** 2 for i in
                   range(max(0, -offset), min(len(motion1), len(motion2) - offset)))
        if diff < min_diff:
            min_diff = diff
            best_offset = offset
    return best_offset


def insert_blank_frames(frames, count):
    blank_frame = np.zeros_like(frames[0])
    for _ in range(count):
        frames.insert(0, blank_frame)
    return frames


def save_corrected_video(frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]), False)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    out.release()


# Paths to videos
video1_path = r"H:\Dual-belt_APAs\videos\Round_3\20230308\HM_20230308_APACharRepeat_FAA-1035244_L_side_1.avi"
video2_path = r"H:\Dual-belt_APAs\videos\Round_3\20230308\HM_20230308_APACharRepeat_FAA-1035244_L_front_1.avi"
video3_path = r"H:\Dual-belt_APAs\videos\Round_3\20230308\HM_20230308_APACharRepeat_FAA-1035244_L_overhead_1.avi"

# Motion detection
motion1 = parallel_motion_detection(video1_path)
motion2 = parallel_motion_detection(video2_path)
motion3 = parallel_motion_detection(video3_path)

# Further processing...
