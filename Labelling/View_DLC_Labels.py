import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.widgets import Button as MplButton
import os
import mplcursors
from matplotlib import cm
from matplotlib.backend_tools import ToolZoom, ToolPan
from matplotlib.transforms import Bbox


# Load video and deeplabcut coordinates
video_path = r"C:\Users\hmorl\Documents\HM_20230316_APACharExt_FAA-1035246_LR_side_1.avi"
coord_path = (r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\Round3\20230316\HM_20230316_APACharExt_FAA-1035246_LR_side_1DLC_resnet50_DLC_DualBeltAug2shuffle1_1200000.h5")
    #(r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\Round2\20230316\HM_20230316_APACharExt_FAA-1035246_LR_front_1DLC_resnet50_DLC_DualBeltAug3shuffle1_1000000.h5")
    #(r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\Round3\20230316\HM_20230316_APACharExt_FAA-1035246_LR_side_1DLC_resnet50_DLC_DualBeltAug2shuffle1_1200000.h5")


# video_path = r"H:\Dual-belt_APAs\videos\Round_3\20230306\HM_20230306_APACharRepeat_FAA-1035244_L_side_1.avi"
# coord_path = (r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_AnalysedFiles\Round2\20230306\HM_20230306_APACharRepeat_FAA-1035244_L_side_1DLC_resnet50_DLC_DualBeltAug2shuffle1_1200000.h5")
print(f"Video path: {video_path}")

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file does not exist at path: {video_path}")
    exit(1)

# Check if coordinate file exists
if not os.path.exists(coord_path):
    print(f"Error: Coordinate file does not exist at path: {coord_path}")
    exit(1)

saved_frames_dir = r"H:\Dual-belt_APAs\Plots\Jan25\Characterisation\Tracking"
cam_name = video_path.split("\\")[-1].split(".")[0].split("_")[-2]

extracted_frames_dir = "extracted_frames"
if not os.path.exists(extracted_frames_dir):
    os.makedirs(extracted_frames_dir)

extracted_coords = pd.DataFrame()

print("Reading coordinates...")
coords = pd.read_hdf(coord_path)
print("Finished")

# Flatten the multi-index columns and rename them
coords = coords.droplevel(0, axis=1)
coords.columns = ['_'.join(col).strip() for col in coords.columns.values]
coords['frame'] = coords.index

print("Coords DataFrame:")
print(coords.head())
print(coords.columns)

print("Reading video...")
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Finished")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit(1)
else:
    print(f"Total number of frames: {frame_count}")

# Set initial values
current_frame = 0
pcutoff = 0.9  # initial cutoff value

# Extract body parts in the order they appear in the DataFrame
bodyparts = []
for col in coords.columns:
    if '_x' in col:
        bodypart = col.split('_')[0]
        if bodypart not in bodyparts:
            bodyparts.append(bodypart)

# Create the color map based on the original order
cmap = cm.get_cmap('viridis', len(bodyparts))
color_map = {bodypart: cmap(i) for i, bodypart in enumerate(bodyparts)}


# Initial scatter point size
scatter_size = 12

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

    # Create a new annotation object to handle hovering labels
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

    for col in frame_coords.columns:
        if 'likelihood' in col:
            bodypart = col.split('_likelihood')[0]
            likelihood = frame_coords[f'{bodypart}_likelihood'].values[0]
            x = frame_coords[f'{bodypart}_x'].values[0]
            y = frame_coords[f'{bodypart}_y'].values[0]
            color = color_map[bodypart]  # Get the color for this bodypart

            exclusion_labels = [
                'StartPlatL', 'StartPlatR', 'TransitionL', 'TransitionR', 'StepL', 'StepR', 'Door'
            ]
            # check if bodypart is in exclusion_labels
            if any(exclusion in bodypart for exclusion in exclusion_labels):
                continue

            if likelihood > pcutoff:
                scatter = ax.scatter(
                    x, y,
                    s=scatter_size,
                    color='red',#color,
                    edgecolors='red', #'k',
                    marker='o',
                    linewidth=0
                )
            else:
                scatter = ax.scatter(
                    x, y,
                    s=scatter_size,
                    color='red',#color,
                    edgecolors='k',
                    marker='x',
                    alpha=0.2,
                    linewidth=0
                )

            scatter_points.append((bodypart, scatter))

    ax.set_title(f'Frame {frame_idx}')
    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

# Update function for scatter size slider
def update_scatter_size(val):
    global scatter_size
    scatter_size = val
    plot_frame(current_frame)

# Scrollbar update function
def update(val):
    frame_idx = int(slider.val)
    plot_frame(frame_idx)

# Function to skip frames
def skip_frames(skip):
    new_frame = current_frame + skip
    if new_frame < 0:
        new_frame = 0
    elif new_frame >= frame_count:
        new_frame = frame_count - 1
    slider.set_val(new_frame)

# Zoom, pan, and home button functions
def zoom(event):
    fig.canvas.manager.toolbar.zoom()

def pan(event):
    fig.canvas.manager.toolbar.pan()

def home(event):
    fig.canvas.manager.toolbar.home()

def restore_scorer_level(df):
    df.columns = pd.MultiIndex.from_tuples([('scorer', *col.split('_')) for col in df.columns])
    return df

def extract_frame():
    global extracted_coords

    # Extract and save the current frame as an image
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        img_filename = os.path.join(extracted_frames_dir, f"frame_{current_frame}.png")
        cv2.imwrite(img_filename, frame)
        print(f"Saved: {img_filename}")

    # Extract and save the corresponding row from coords
    frame_coords = coords[coords['frame'] == current_frame]

    # Restore scorer level before saving
    frame_coords_with_scorer = restore_scorer_level(frame_coords)

    # Append to the extracted coordinates DataFrame
    extracted_coords = pd.concat([extracted_coords, frame_coords_with_scorer])

    # Save to CSV and HDF5
    csv_filename = os.path.join(extracted_frames_dir, "extracted_coords.csv")
    h5_filename = os.path.join(extracted_frames_dir, "extracted_coords.h5")

    extracted_coords.to_csv(csv_filename, index=False)
    extracted_coords.to_hdf(h5_filename, key='df', mode='w')

    print(f"Saved: {csv_filename} and {h5_filename}")

def save_frame_with_coords():
    # Redraw the canvas to update positions
    fig.canvas.draw()

    # Get the figure size in inches
    fig_width, fig_height = fig.get_size_inches()

    # Get the axes position as a fraction of the figure and convert to inches.
    pos = ax.get_position()
    bbox_inches = Bbox.from_bounds(pos.x0 * fig_width,
                                   pos.y0 * fig_height,
                                   pos.width * fig_width,
                                   pos.height * fig_height)

    # Build the file path – using the same directory as extracted_frames_dir and naming with the current frame index
    save_filename = os.path.join(saved_frames_dir, f"frame_{current_frame}_{cam_name}.svg")
    # Save the entire figure but crop to the bounding box (so only the main axis is saved)
    fig.savefig(save_filename, bbox_inches=bbox_inches)
    print(f"Saved frame with coords to {save_filename}")


# Create Matplotlib figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

# Create scrollbar
ax_slider = plt.axes([0.1, 0.25, 0.8, 0.03])
slider = Slider(ax_slider, 'Frame', 0, frame_count - 1, valinit=current_frame, valfmt='%0.0f')
slider.on_changed(update)

# Scatter size slider next to skip buttons
ax_scatter_slider = plt.axes([0.8, 0.1, 0.15, 0.03])
scatter_slider = Slider(ax_scatter_slider, 'Size', 5, 100, valinit=scatter_size, valfmt='%0.0f')
scatter_slider.on_changed(update_scatter_size)

# Create skip buttons, zoom, pan, and home buttons
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

ax_extract = plt.axes([0.44, 0.05, 0.12, 0.04])  # Position the button appropriately
btn_extract = MplButton(ax_extract, 'Extract Frame')
btn_extract.on_clicked(lambda event: extract_frame())

ax_save = plt.axes([0.56, 0.05, 0.12, 0.04])  # adjust position as needed
btn_save = MplButton(ax_save, 'Save frame with coords')
btn_save.on_clicked(lambda event: save_frame_with_coords())


# Show the initial frame
plot_frame(current_frame)

plt.show()