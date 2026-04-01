"""Interactive viewer for evaluating DeepLabCut tracking quality frame by frame."""
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.widgets import Button as MplButton
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import cm


# Function to select video paths via file dialog
def select_video_paths():
    root = Tk()
    root.withdraw()
    chosen_video_path = askopenfilename(title="Select a Video (Side, Front, or Overhead)")
    if not chosen_video_path:
        raise ValueError("No video selected.")

    if '_side_' in chosen_video_path:
        chosen_cam = 'side'
    elif '_front_' in chosen_video_path:
        chosen_cam = 'front'
    elif '_overhead_' in chosen_video_path:
        chosen_cam = 'overhead'
    else:
        raise ValueError("Unknown camera view selected. Please select a side, front, or overhead video.")

    video_base = os.path.splitext(chosen_video_path)[0].replace(f'_{chosen_cam}_1',
                                                                '')  # Base path without view-specific suffix
    video_paths = {
        'side': f"{video_base}_side_1.avi",
        'front': f"{video_base}_front_1.avi",
        'overhead': f"{video_base}_overhead_1.avi"
    }

    # Find the corresponding coordinate files in the same directories as the videos
    coord_paths = {
        'side': find_matching_coord_file(video_paths['side'], 'side'),
        'front': find_matching_coord_file(video_paths['front'], 'front'),
        'overhead': find_matching_coord_file(video_paths['overhead'], 'overhead')
    }

    return chosen_cam, video_paths, coord_paths


def find_matching_coord_file(video_path, view):
    video_dir = os.path.dirname(video_path)
    video_core_name = os.path.basename(video_path).replace(f'_{view}_1.avi', '')  # Extract the core part of the name

    # List all .h5 files in the directory
    coord_files = [f for f in os.listdir(video_dir) if f.endswith('.h5') and view in f]

    # Search for any .h5 file that contains the core video name and the specific view
    matching_files = [f for f in coord_files if video_core_name in f and f'_{view}_' in f]

    if not matching_files:
        raise FileNotFoundError(f"No matching .h5 file found for {video_path} with view {view}")

    # If multiple files match, choose the one with the longest common prefix
    matching_files.sort(key=lambda f: len(os.path.commonprefix([video_core_name, f])), reverse=True)

    return os.path.join(video_dir, matching_files[0])


def load_timestamps(video_path, view):
    # Adjust the timestamp file to match the provided structure
    timestamp_file = video_path.replace('.avi', '_Timestamps.csv')
    if not os.path.exists(timestamp_file):
        raise FileNotFoundError(f"Timestamp file not found at: {timestamp_file}")
    timestamps = pd.read_csv(timestamp_file)
    return timestamps


def zero_timestamps(timestamps):
    timestamps['Timestamp'] = timestamps['Timestamp'] - timestamps['Timestamp'][0]
    return timestamps


# Use the function to get video and coordinate paths
chosen_cam, video_paths, coord_paths = select_video_paths()

# Confirm the paths being used
print(f"Using {chosen_cam} video path: {video_paths[chosen_cam]}")
print(f"Using side coordinate path: {coord_paths['side']}")
print(f"Using front coordinate path: {coord_paths['front']}")
print(f"Using overhead coordinate path: {coord_paths['overhead']}")

# Load timestamps
timestamps_side = zero_timestamps(load_timestamps(video_paths['side'], 'side'))
timestamps_front = zero_timestamps(load_timestamps(video_paths['front'], 'front'))
timestamps_overhead = zero_timestamps(load_timestamps(video_paths['overhead'], 'overhead'))


def adjust_timestamps(side_timestamps, other_timestamps):
    mask = other_timestamps['Timestamp'].diff() < 4.045e+6
    other_timestamps_single_frame = other_timestamps[mask]
    side_timestamps_single_frame = side_timestamps[mask]
    diff = other_timestamps_single_frame['Timestamp'] - side_timestamps_single_frame['Timestamp']

    # Find the best fit line for the lower half of the data by straightening the line
    model = LinearRegression().fit(side_timestamps_single_frame['Timestamp'].values.reshape(-1, 1), diff.values)
    slope = model.coef_[0]
    intercept = model.intercept_
    straightened_diff = diff - (slope * side_timestamps_single_frame['Timestamp'] + intercept)
    correct_diff_idx = np.where(straightened_diff < straightened_diff.mean())

    model_true = LinearRegression().fit(
        side_timestamps_single_frame['Timestamp'].values[correct_diff_idx].reshape(-1, 1),
        diff.values[correct_diff_idx])
    slope_true = model_true.coef_[0]
    intercept_true = model_true.intercept_
    adjusted_timestamps = other_timestamps['Timestamp'] - (slope_true * other_timestamps['Timestamp'] + intercept_true)
    return adjusted_timestamps


timestamps_side_adj = timestamps_side['Timestamp']
timestamps_front_adj = adjust_timestamps(timestamps_side, timestamps_front)
timestamps_overhead_adj = adjust_timestamps(timestamps_side, timestamps_overhead)


def match_frames(timestamps_side, timestamps_front, timestamps_overhead):
    buffer_ns = int(4.04e+6)  # Frame duration in nanoseconds

    # Ensure the timestamps are sorted
    timestamps_side = timestamps_side.sort_values().reset_index(drop=True).astype('float64')
    timestamps_front = timestamps_front.sort_values().reset_index(drop=True).astype('float64')
    timestamps_overhead = timestamps_overhead.sort_values().reset_index(drop=True).astype('float64')

    # Prepare the DataFrames with Frame Numbers
    side_df = pd.DataFrame({'Timestamp': timestamps_side, 'Frame_number_side': range(len(timestamps_side))})
    front_df = pd.DataFrame({'Timestamp': timestamps_front, 'Frame_number_front': range(len(timestamps_front))})
    overhead_df = pd.DataFrame(
        {'Timestamp': timestamps_overhead, 'Frame_number_overhead': range(len(timestamps_overhead))})

    # Perform asof merge to find the closest matching frames within the buffer
    matched_front = pd.merge_asof(
        side_df,
        front_df,
        on='Timestamp',
        direction='nearest',
        tolerance=buffer_ns,
        suffixes=('_side', '_front')
    )

    matched_all = pd.merge_asof(
        matched_front,
        overhead_df,
        on='Timestamp',
        direction='nearest',
        tolerance=buffer_ns,
        suffixes=('_side', '_overhead')
    )

    # Handle NaNs explicitly by setting unmatched frames to -1
    matched_frames = matched_all[['Frame_number_side', 'Frame_number_front', 'Frame_number_overhead']].applymap(
        lambda x: int(x) if pd.notnull(x) else -1
    ).values.tolist()

    return matched_frames


matched_frames = match_frames(timestamps_side_adj, timestamps_front_adj, timestamps_overhead_adj)

# The rest of the original code follows here, but now the frames will be synchronized using the matched_frames list.
# Further processing code...

# Initialize Video Capture for the chosen camera
cap = cv2.VideoCapture(video_paths[chosen_cam])
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
pcutoff = 0.9  # initial cutoff value

# Load the coordinates
coords = pd.read_hdf(coord_paths[chosen_cam])
coords = coords.droplevel(0, axis=1)
coords.columns = ['_'.join(col).strip() for col in coords.columns.values]
coords['frame'] = coords.index

# Body parts extraction
bodyparts = []
for col in coords.columns:
    if '_x' in col:
        bodypart = col.split('_')[0]
        if bodypart not in bodyparts:
            bodyparts.append(bodypart)

cmap = cm.get_cmap('viridis', len(bodyparts))
color_map = {bodypart: cmap(i) for i, bodypart in enumerate(bodyparts)}

scatter_size = 50


# Function to remove duplicates based on image names
def remove_duplicates(h5_filename):
    if os.path.exists(h5_filename):
        data = pd.read_hdf(h5_filename, key='df')
        data = data[~data.index.duplicated(keep='last')]
        data.to_hdf(h5_filename, key='df', mode='w')
        print(f"Removed duplicates from {h5_filename}")


remove_duplicates("CollectedData_Holly_init.h5")  # Initial cleanup of duplicates


def construct_extracted_dir(chosen_cam, video_base_name):
    return f"H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/{chosen_cam.capitalize()}/{video_base_name}"


def plot_frame(frame_idx):
    global current_frame, scatter_points
    current_frame = frame_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        return

    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_coords = coords[coords['frame'] == frame_idx]

    scatter_points = []  # Clear scatter points list for each frame

    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                        textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind, scatter, label):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        annot.set_text(label)
        annot.get_bbox_patch().set_facecolor(color_map[label])
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for label, scatter in scatter_points:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind, scatter, label)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    # Plot each body part
    for col in frame_coords.columns:
        if 'likelihood' in col:
            bodypart = col.split('_likelihood')[0]
            likelihood = frame_coords[f'{bodypart}_likelihood'].values[0]
            x = frame_coords[f'{bodypart}_x'].values[0]
            y = frame_coords[f'{bodypart}_y'].values[0]
            color = color_map[bodypart]

            if likelihood > pcutoff:
                scatter = ax.scatter(x, y, s=scatter_size, color=color, edgecolors='k', marker='o')
            else:
                scatter = ax.scatter(x, y, s=scatter_size, color=color, edgecolors='k', marker='x', alpha=0.2)

            scatter_points.append((bodypart, scatter))

    # Full list of skeleton pairs
    skeleton_pairs = [
        ("Nose", "EarR"), ("Nose", "EarL"), ("Nose", "Back1"), ("EarR", "EarL"), ("Back1", "Back2"), ("Back2", "Back3"),
        ("Back3", "Back4"), ("Back4", "Back5"), ("Back5", "Back6"), ("Back6", "Back7"), ("Back7", "Back8"), ("Back8", "Back9"),
        ("Back9", "Back10"), ("Back10", "Back11"), ("Back11", "Back12"), ("Back12", "Tail1"), ("Tail1", "Tail2"), ("Tail2", "Tail3"),
        ("Tail3", "Tail4"), ("Tail4", "Tail5"), ("Tail5", "Tail6"), ("Tail6", "Tail7"), ("Tail7", "Tail8"), ("Tail8", "Tail9"),
        ("Tail9", "Tail10"), ("Tail10", "Tail11"), ("Tail11", "Tail12"), ("ForepawToeR", "ForepawKnuckleR"), ("ForepawKnuckleR", "ForepawAnkleR"),
        ("ForepawAnkleR", "ForepawKneeR"), ("ForepawToeL", "ForepawKnuckleL"), ("ForepawKnuckleL", "ForepawAnkleL"), ("ForepawAnkleL", "ForepawKneeL"),
        ("HindpawToeR", "HindpawKnuckleR"), ("HindpawKnuckleR", "HindpawAnkleR"), ("HindpawAnkleR", "HindpawKneeR"), ("HindpawToeL", "HindpawKnuckleL"),
        ("HindpawKnuckleL", "HindpawAnkleL"), ("HindpawAnkleL", "HindpawKneeL"), ("Back3", "ForepawKneeR"), ("Back3", "ForepawKneeL"),
        ("Back10", "HindpawKneeR"), ("Back10", "HindpawKneeL")
    ]

    # Draw the skeleton if the toggle is on
    if skeleton_visible:
        for part1, part2 in skeleton_pairs:
            if f'{part1}_x' in frame_coords.columns and f'{part2}_x' in frame_coords.columns:
                x1, y1 = frame_coords[f'{part1}_x'].values[0], frame_coords[f'{part1}_y'].values[0]
                x2, y2 = frame_coords[f'{part2}_x'].values[0], frame_coords[f'{part2}_y'].values[0]
                likelihood1 = frame_coords[f'{part1}_likelihood'].values[0]
                likelihood2 = frame_coords[f'{part2}_likelihood'].values[0]

                # Draw the line only if both points have high enough likelihoods
                if likelihood1 > pcutoff and likelihood2 > pcutoff:
                    line_color = color_map[part1]  # Color based on the first label in the pair
                    ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=1)

    ax.set_title(f'Frame {frame_idx}')
    fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", hover)



def update_scatter_size(val):
    global scatter_size
    scatter_size = val
    plot_frame(current_frame)


def update(val):
    frame_idx = int(slider.val)
    plot_frame(frame_idx)


def skip_frames(skip):
    new_frame = current_frame + skip
    if new_frame < 0:
        new_frame = 0
    elif new_frame >= frame_count:
        new_frame = frame_count - 1
    slider.set_val(new_frame)


def zoom(event):
    fig.canvas.manager.toolbar.zoom()


def pan(event):
    fig.canvas.manager.toolbar.pan()


def home(event):
    fig.canvas.manager.toolbar.home()

def toggle_skeleton():
    global skeleton_visible
    skeleton_visible = not skeleton_visible
    plot_frame(current_frame)

def restore_scorer_level(df):
    df.columns = pd.MultiIndex.from_tuples([('Holly', *col.split('_')) for col in df.columns])
    return df


def extract_frame():
    def save_frame_and_coords(camera):
        video_cap = cv2.VideoCapture(video_paths[camera])
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = video_cap.read()
        if ret:
            vid_name = os.path.basename(video_paths[camera]).split('.')[0]
            camera_dir = construct_extracted_dir(camera, vid_name)
            if not os.path.exists(camera_dir):
                os.makedirs(camera_dir)
            img_filename = os.path.join(camera_dir, f"img{current_frame}.png")
            cv2.imwrite(img_filename, frame)
            print(f"Saved: {img_filename}")

            # Load the coordinate data
            frame_coords = pd.read_hdf(coord_paths[camera])
            frame_coords = frame_coords.droplevel(0, axis=1)
            frame_coords.columns = ['_'.join(col).strip() for col in frame_coords.columns.values]
            frame_coords['frame'] = frame_coords.index
            frame_coords = frame_coords[frame_coords['frame'] == current_frame]

            # Apply the pcutoff filter and remove likelihood columns
            for col in frame_coords.columns:
                if '_likelihood' in col:
                    bodypart = col.split('_likelihood')[0]
                    mask = frame_coords[col] < pcutoff
                    x_col = f'{bodypart}_x'
                    y_col = f'{bodypart}_y'
                    # Set x and y to NaN where likelihood is below pcutoff
                    frame_coords.loc[mask, [x_col, y_col]] = np.nan

            # Drop the likelihood columns
            filtered_coords = frame_coords.drop(columns=[col for col in frame_coords.columns if 'likelihood' in col])

            frame_coords_with_scorer = restore_scorer_level(filtered_coords)

            new_index = pd.MultiIndex.from_tuples([(f'labeled_data', vid_name, f"img{current_frame}.png")])
            frame_coords_with_scorer.set_index(new_index, append=False, inplace=True)

            csv_filename = os.path.join(camera_dir, "CollectedData_Holly_init.csv")
            h5_filename = os.path.join(camera_dir, "CollectedData_Holly_init.h5")

            # Check if the HDF5 file exists and load existing data
            if os.path.exists(h5_filename):
                existing_data = pd.read_hdf(h5_filename, key='df')
                # Remove any previous entry for the same image
                existing_data = existing_data[
                    ~existing_data.index.get_level_values(2).isin([f"img{current_frame}.png"])]
                combined_data = pd.concat([existing_data, frame_coords_with_scorer], axis=0)
            else:
                combined_data = frame_coords_with_scorer

            # Ensure that the combined data is deduplicated by index
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]

            # Save the combined data to both CSV and HDF5 formats
            combined_data.to_csv(csv_filename, index=True)
            combined_data.to_hdf(h5_filename, key='df', mode='w')

            print(f"Appended data and saved to: {csv_filename} and {h5_filename}")

    save_frame_and_coords(chosen_cam)
    other_views = {'side', 'front', 'overhead'} - {chosen_cam}
    for view in other_views:
        save_frame_and_coords(view)


# Create Matplotlib figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

# Create scrollbar
ax_slider = plt.axes([0.1, 0.25, 0.8, 0.03])
slider = Slider(ax_slider, 'Frame', 0, frame_count - 1, valinit=current_frame, valfmt='%0.0f')
slider.on_changed(update)

ax_scatter_slider = plt.axes([0.8, 0.1, 0.15, 0.03])
scatter_slider = Slider(ax_scatter_slider, 'Size', 5, 100, valinit=scatter_size, valfmt='%0.0f')
scatter_slider.on_changed(update_scatter_size)

button_positions = [0.1, 0.18, 0.26, 0.34, 0.42, 0.5, 0.58, 0.66]
skip_values = [-1000, -100, -10, -1, 1, 10, 100, 1000]

buttons = []
for pos, skip in zip(button_positions, skip_values):
    ax_btn = plt.axes([pos, 0.1, 0.08, 0.03])
    btn = MplButton(ax_btn, f'{skip:+}')
    btn.on_clicked(lambda event, s=skip: skip_frames(s))
    buttons.append(btn)

ax_zoom = plt.axes([0.7, 0.05, 0.08, 0.03])
btn_zoom = MplButton(ax_zoom, 'Zoom')
btn_zoom.on_clicked(zoom)

ax_pan = plt.axes([0.8, 0.05, 0.08, 0.03])
btn_pan = MplButton(ax_pan, 'Pan')
btn_pan.on_clicked(pan)

ax_home = plt.axes([0.9, 0.05, 0.08, 0.03])
btn_home = MplButton(ax_home, 'Home')
btn_home.on_clicked(home)

ax_skeleton = plt.axes([0.55, 0.05, 0.12, 0.04])
btn_skeleton = MplButton(ax_skeleton, 'Toggle Skeleton')
skeleton_visible = False  # Initial state

btn_skeleton.on_clicked(lambda event: toggle_skeleton())


ax_extract = plt.axes([0.44, 0.05, 0.12, 0.04])
btn_extract = MplButton(ax_extract, 'Extract Frame')
btn_extract.on_clicked(lambda event: extract_frame())

# Show the initial frame
plot_frame(current_frame)

plt.show()
