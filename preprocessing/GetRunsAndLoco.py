"""Segments continuous recordings into individual trials and classifies run phases and limb states."""
import os, glob, re, csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
import joblib
import cv2
import time

import helpers.utils as utils
from helpers.config import *
from gait import GaitFeatureExtraction as gfe
from helpers import ManualRunAdjustment as mra
from helpers import ManualFileAdjustment as mfa
from helpers.ConditionsFinder import BaseConditionFiles

SENTINEL_VALUE = -1

class GetRuns:
    def __init__(self, file, mouseID, date,
                 exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None, vmt_level=None, prep=None,
                 save_frames=True):
        self.file, self.mouseID, self.date = file, mouseID, date

        # store the experiment details for error logging
        self.exp = exp
        self.speed = speed
        self.repeat_extend = repeat_extend
        self.exp_wash = exp_wash
        #self.day = day
        self.vmt_type = vmt_type
        self.vmt_level = vmt_level
        self.prep = prep

        # whether or not to load/save frames from video
        self.save_frames = save_frames

        # if self.day is empty, look for 'Day' in the file name and extract number after it
        if day is None:
            match = re.search(r'Day(\d+)', self.file)
            if match:
                # Example: store the raw number or "DayX"
                self.day = match.group(1)  # e.g. '1'
            # self.day = "Day" + match.group(1)  # e.g. 'Day1'
            else:
                self.day = None
        else:
            self.day = None

        self.model, self.label_encoders, self.feature_columns = self.load_model()
        self.trial_starts, self.trial_ends = [], []
        self.run_starts, self.run_ends_steps, self.run_ends, self.transitions, self.taps = [], [], [], [], []
        self.buffer = 125

        # Always load the raw data
        self.data = self.get_data()

        # Error logging
        self.error_log_file = os.path.join(paths['filtereddata_folder'], 'error_log.csv')


    def load_model(self):
        model_filename = os.path.join(paths['filtereddata_folder'], 'LimbStuff', 'limb_classification_model.pkl')
        model = joblib.load(model_filename)

        # Load label encoders
        label_encoders_path = os.path.join(paths['filtereddata_folder'], 'LimbStuff', 'label_encoders.pkl')
        label_encoders = joblib.load(label_encoders_path)

        # Load feature columns
        feature_columns_path = os.path.join(paths['filtereddata_folder'], 'LimbStuff', 'feature_columns.pkl')
        feature_columns = joblib.load(feature_columns_path)

        # Return model, label_encoders, and feature_columns
        return model, label_encoders, feature_columns

    def get_saved_data_filename(self):
        # Generate a consistent filename for saving/loading self.data
        base_dir = os.path.dirname(self.file)
        filename = os.path.basename(self.file).replace('.h5', '_Runs.h5')
        return os.path.join(base_dir, filename)

    def save_data(self):
        filename = self.get_saved_data_filename()
        self.data.to_hdf(filename, key='real_world_coords_runs')
        print(f"Saved self.data to {filename}")

    def _write_run_summary(self):
        """
        Writes summary of runs to a CSV log, rewriting any existing entry for the same file.
          - File
          - MouseID
          - # Registered (including sentinel placeholders)
          - # Recorded (actual data runs in self.data)
          - # Missing
          - # Dropped
        """
        summary_log_file = os.path.join(paths['filtereddata_folder'], 'run_summary_log.csv')

        # Calculate the run statistics
        n_registered_runs = len(self.trial_starts)
        n_recorded_runs = sum(start != SENTINEL_VALUE for start in self.trial_starts)

        n_missing_runs = 0
        if self.date in mra.missing_runs and self.mouseID in mra.missing_runs[self.date]:
            n_missing_runs = len(mra.missing_runs[self.date][self.mouseID])

        n_dropped_runs_placeholder = 0
        if self.date in mra.runs_to_drop_placeholder and self.mouseID in mra.runs_to_drop_placeholder[self.date]:
            n_dropped_runs_placeholder = len(mra.runs_to_drop_placeholder[self.date][self.mouseID])

        n_dropped_runs_completely = 0
        if self.date in mra.runs_to_drop_completely and self.mouseID in mra.runs_to_drop_completely[self.date]:
            n_dropped_runs_completely = len(mra.runs_to_drop_completely[self.date][self.mouseID])

        # Columns for the CSV
        columns = logging['run_summary']

        # 1) Read existing CSV (if it exists) into a DataFrame
        if os.path.exists(summary_log_file):
            summary_df = pd.read_csv(summary_log_file)
        else:
            summary_df = pd.DataFrame(columns=columns)

        # 2) Remove any existing entry for this file (so we can overwrite)
        summary_df = summary_df[summary_df['File'] != self.file]

        # 3) Add the new entry
        new_row = {
            'File': self.file,
            'exp': self.exp,
            'speed': self.speed,
            'repeat_extend': self.repeat_extend,
            'exp_wash': self.exp_wash,
            'day': self.day,
            'vmt_type': self.vmt_type,
            'vmt_level': self.vmt_level,
            'prep': self.prep,
            'MouseID': self.mouseID,
            'RegisteredRuns': n_registered_runs,
            'RecordedRuns': n_recorded_runs,
            'MissingRuns': n_missing_runs,
            'DroppedRunsPlaceholder': n_dropped_runs_placeholder,
            'DroppedRunsCompletely': n_dropped_runs_completely
        }
        summary_df = summary_df.append(new_row, ignore_index=True)
        summary_df = summary_df.sort_values(by=['File'])

        # 4) Write the updated DataFrame back to CSV
        summary_df.to_csv(summary_log_file, index=False)

    def visualise_video_frames(self, view, start_frame, end_frame):
        if not self.save_frames:
            print("Skipping visualise_video_frames because save_frames=False")
            return
        day = os.path.basename(self.file).split('_')[1]
        filename_pattern = '_'.join(os.path.basename(self.file).split('_')[:-1])
        video_files = glob.glob(os.path.join(paths['video_folder'], f"{day}/{filename_pattern}_{view}*.avi"))
        if len(video_files) > 1:
            raise ValueError("Multiple video files found for the same view. Please check the video files.")
        else:
            video_file = video_files[0]

        # Open the video file
        cap = cv2.VideoCapture(video_file)

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure end_frame does not exceed total_frames
        if end_frame > total_frames:
            end_frame = total_frames

        # Get the width and height of the video frame
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Buffer to hold the frames from start_frame to end_frame
        frames = []

        # Read through the video to buffer the frames between start_frame and end_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx}")
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib

        # Release the video capture after buffering the frames
        cap.release()

        # Initialize Matplotlib figure and axis
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.25)

        # Display the first frame
        frame_image = ax.imshow(frames[0])
        ax.set_xticks([])
        ax.set_yticks([])

        # Create the slider for frame selection
        ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Frame', start_frame, end_frame, valinit=start_frame, valfmt='%d')

        # Update the displayed frame when the slider is changed
        def update(val):
            frame_num = int(slider.val) - start_frame
            frame_image.set_data(frames[frame_num])  # Update the image data with the selected frame
            plt.draw()

        slider.on_changed(update)

        # Show the Matplotlib window
        plt.show()

    def get_data(self):
        # Get the data from the h5 file
        data = pd.read_hdf(self.file, key='real_world_coords')
        # label multiindex columns as 'bodyparts' and 'coords'
        data.columns = pd.MultiIndex.from_tuples([(col[0], col[1]) for col in data.columns], names=['bodyparts', 'coords'])

        return data

    def _remove_existing_error_entries_for_file(self):
        """
        Removes any existing error log entries for the current file (self.file).
        """
        if not os.path.exists(self.error_log_file):
            # No existing log file to clean
            return

        # Read existing entries
        error_df = pd.read_csv(self.error_log_file)

        # Keep only rows that do NOT match the current file
        error_df = error_df[error_df['File'] != self.file]

        # Write back to the log file
        error_df.to_csv(self.error_log_file, index=False)
        print(f"Cleared old error log entries for {self.file}")

    def get_runs(self):
        # Remove old entries for this file from the error log
        self._remove_existing_error_entries_for_file()
        self.find_trials()
        self.find_steps()
        self.index_by_run()
        print("Runs extracted successfully")
        print("Extracted runs (real): ", len(self.data.index.get_level_values('Run').unique()))
        print("Extracted runs: (symbolic)", len(self.trial_starts))
        self._write_run_summary()
        self.find_run_stages()
        self.save_data()

    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------- Finding trials -------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------

    def find_trials(self):
        door_open, door_close = self.door_opn_cls()
        mouse_present = self.find_mouse_on_belt()

        door_open_checked = []
        door_close_checked = []

        self.trial_starts = door_open
        self.trial_ends = door_close

        # Filter only valid trials
        valid_trials = [(start, end) for start, end in zip(door_open, door_close)
                        if start != SENTINEL_VALUE and end != SENTINEL_VALUE]

        # Check for mouse presence within valid trials
        for open_frame, close_frame in valid_trials:
            present_frames = mouse_present[(mouse_present >= open_frame) & (mouse_present <= close_frame)]
            if len(present_frames) > 0:
                door_open_checked.append(open_frame)
                door_close_checked.append(close_frame)

    def door_opn_cls(self):
        # Extract the rolling window
        rolling_window = self.data.loc(axis=1)['Door', 'z'].rolling(window=10, center=True, min_periods=0)
        # Apply NumPy for faster computation
        door_movement = rolling_window.apply(lambda x: np.subtract(x[-1], x[0]) if len(x) >= 2 else 0, raw=True)
        door_closed = door_movement.index[door_movement.abs() < 1]
        door_closed_chunks = utils.Utils().find_blocks(door_closed, gap_threshold=50, block_min_size=fps*2)

        # check if after the end of each closed chunk there is a door opening, where the door marker is not visible

        door_open = []
        door_close = []
        for cidx, chunk in enumerate(door_closed_chunks):
            chunk_end = chunk[-1]

            # After a closed chunk ends, check for the absence of the door marker (NaNs or missing values in 'z')
            window_after_chunk = self.data.loc[chunk_end:chunk_end + 2000, ('Door', 'z')]

            # Check if there is a long period where the door marker is missing (NaNs)
            door_out_of_frame = window_after_chunk.isna()
            if door_out_of_frame.sum() > 1500:  # if there is any NaN in the window after the chunk
                # Retrieve the chunk value to denote the beginnning of the trial
                door_open.append(chunk[-1])
                if cidx + 1 < len(door_closed_chunks):
                    door_close.append(door_closed_chunks[cidx + 1][0])
        if door_close[0] < door_open[0]:
            door_open.append(0)
        if door_close[-1] < door_open[-1]:
            door_close.append(self.data.index[-1])

        door_open, door_close = np.sort(np.array(door_open)), np.sort(np.array(door_close))

        # 1) Insert placeholders for missing runs
        if self.date in mra.missing_runs and self.mouseID in mra.missing_runs[self.date]:
            for idx in sorted(mra.missing_runs[self.date][self.mouseID]):
                # Safely insert SENTINEL_VALUE at the position where a run was missing
                if idx <= len(door_open):
                    door_open = np.insert(door_open, idx, SENTINEL_VALUE)
                    door_close = np.insert(door_close, idx, SENTINEL_VALUE)

        #print(f"Runs before dropping: {door_open}")

        # 2) Then blank out the runs that need to be dropped (but keep as placeholders)
        if self.date in mra.runs_to_drop_placeholder and self.mouseID in mra.runs_to_drop_placeholder[self.date]:
            for idx in mra.runs_to_drop_placeholder[self.date][self.mouseID]:
                if 0 <= idx < len(door_open):
                    door_open[idx] = SENTINEL_VALUE
                    door_close[idx] = SENTINEL_VALUE

        #print(f"Runs before dropping, after blocking out: {door_open}")

        # 3) Finally, drop the runs that need to be dropped completely
        if self.date in mra.runs_to_drop_completely and self.mouseID in mra.runs_to_drop_completely[self.date]:
            for idx in sorted(mra.runs_to_drop_completely[self.date][self.mouseID], reverse=True):
                if 0 <= idx < len(door_open):
                    door_open = np.delete(door_open, idx)
                    door_close = np.delete(door_close, idx)

        #print(f"Runs after dropping: {door_open}")

        return door_open, door_close

    def find_forward_facing_bool(self, data, xthreshold, zthreshold, nosezthreshold=45, nose=False):
        # filter by when mouse facing forward
        back_median = data.loc(axis=1)[
            ['Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11',
             'Back12'], 'x'].median(axis=1)
        tail_median = data.loc(axis=1)[
            ['Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11',
             'Tail12'], 'x'].median(axis=1)
        #smooth both medians and put back into series
        back_median = pd.Series(gaussian_filter1d(back_median, sigma=1), index=back_median.index)
        tail_median = pd.Series(gaussian_filter1d(tail_median, sigma=1), index=tail_median.index)
        back_tail_mask = back_median - tail_median > xthreshold

        # interpolate and smooth nose and tail to avoid short gaps
        interp = data.loc(axis=1)[['Nose','Tail1'], ['x','z']].interpolate(method='linear', axis=0)

        nose_tail_x_mask = interp['Nose','x'] > interp['Tail1','x']
        nose_tail_z_mask = interp['Nose','z'] - interp['Tail1','z'] < zthreshold

        nose_mask = interp['Nose','z'] < nosezthreshold

        # nose_tail_x_mask = data.loc[:, ('Nose', 'x')] > data.loc[:, ('Tail1', 'x')]
        # nose_tail_z_mask = data.loc[:, ('Nose', 'z')] - data.loc[:, ('Tail1', 'z')] < zthreshold
        # #belt1_mask = data.loc[:, ('Nose', 'x')] < 470
        if nose == True:
            facing_forward_mask = back_tail_mask & nose_tail_x_mask & nose_tail_z_mask & nose_mask #& belt1_mask
        else:
            facing_forward_mask = back_tail_mask & nose_tail_x_mask & nose_tail_z_mask
        return facing_forward_mask


    def find_mouse_on_belt(self):
        # find the frame where the mouse crosses the finish line (x=600?) for the first time after the trial start
        facing_forward_mask = self.find_forward_facing_bool(self.data, xthreshold=0, zthreshold=40)
        mouse_on_belt = facing_forward_mask & (self.data.loc[:, ('Nose', 'x')] > 200) & (self.data.loc[:, ('Nose', 'x')] < 500)
        mouse_on_belt_index = mouse_on_belt.index[mouse_on_belt]
        return mouse_on_belt_index

    def index_by_run(self):
        # Initialize empty lists for run indices and frame indices
        run_idx = []
        frame_idx = []

        # Iterate through each run's trial starts and ends
        for r, (start, end) in enumerate(zip(self.trial_starts, self.trial_ends)):
            if start == SENTINEL_VALUE or end == SENTINEL_VALUE:
                # Skip invalid runs represented by SENTINEL_VALUE
                continue

            # Append run index and corresponding frame indices
            run_idx.extend([r] * (end - start + 1))
            frame_idx.extend(range(start, end + 1))

        # Create a MultiIndex for valid runs
        new_data_idx = pd.MultiIndex.from_arrays([run_idx, frame_idx], names=['Run', 'FrameIdx'])

        # Trim self.data to include only the valid frames (as in the original behavior)
        data_snippet = self.data.loc[frame_idx]
        data_snippet.index = new_data_idx

        # Update self.data to include only trimmed and indexed data
        self.data = data_snippet

    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------- Finding steps --------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    def find_steps(self):
        # Process each run in parallel
        Steps = []
        RunBounds = []
        Runbacks = []
        for r in range(len(self.trial_starts)):
            mins = self.trial_starts[r]/fps/60
            secs = self.trial_starts[r]/fps%60
            print(f"Trial: {r} at {mins} mins {secs} secs")
            if self.trial_starts[r] == SENTINEL_VALUE or self.trial_ends[r] == SENTINEL_VALUE:
                print(f"Skipping run {r} due to manual exclusion.")
                Steps.append(pd.DataFrame())  # or whatever placeholder you like
                RunBounds.append(None)
                Runbacks.append(None)
                continue
            steps, run_bounds, runbacks = self.process_run(r)
            Steps.append(steps)
            RunBounds.append(run_bounds)
            Runbacks.append(runbacks)

        # Combine the steps from all runs
        StepsALL = pd.concat(Steps)
        StepsALL = StepsALL.reindex(self.data.index)
        for paw in StepsALL.columns:
            self.data[(paw, 'SwSt')] = StepsALL[paw].values

        # fill in 'running' column in self.data between run start and end values with 1's
        valid_RunBounds = [rb for rb in RunBounds if rb is not None] # todo check this (and below) if correct for conserving run positions
        self.data['running'] = False
        for start, end in valid_RunBounds:
            self.data.loc[start:end, 'running'] = True

        # fill in 'rb' column in self.data between runback start and end values with 1's
        valid_RunBacks = [rb for rb in Runbacks if rb is not None]
        self.data['rb'] = False
        for rb in valid_RunBacks:
            for start, end in rb:
                self.data.loc[start:end, 'rb'] = True


    def process_run(self, r):
        try:
            #print(f'Analysing run {r}')
            # Create a copy of the data relevant to this trial
            trial_start = self.trial_starts[r]
            trial_end = self.trial_ends[r]
            if trial_start == SENTINEL_VALUE or trial_end == SENTINEL_VALUE:
                print(f"Skipping run {r} due to sentinel value.")
                return pd.DataFrame(), None, None  # Return placeholders for skipped runs

            # Calculate the run data
            run_data = self.data.loc[trial_start:trial_end].copy()

            # Pass run_data to methods that should operate within trial bounds
            run_bounds, runbacks = self.find_real_run_vs_rbs(r, run_data)

            if len(run_bounds) == 1:
                run_bounds = run_bounds[0]
                steps = self.classify_steps_in_run(r, run_bounds)
                #print(f"Run {r} completed")
                return steps, run_bounds, runbacks
            elif len(run_bounds) > 1:
                raise ValueError("More than one run detected in a trial (in find_steps)")
            elif len(run_bounds) == 0:
                print(f"No run detected in trial {r}")
                return pd.DataFrame(), None, None
        except Exception as e:
            print(f"Error processing trial at run {r}: {e}")
            return pd.DataFrame()

    def show_steps_in_videoframes(self, view='Side'):
        if not self.save_frames:
            print("Skipping show_steps_in_videoframes because save_frames=False")
            return

        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # Extract valid runs and their start/end frames from self.data
        valid_runs = []
        for run in self.data.index.get_level_values('Run').unique():
            run_frames = self.data.loc[run].index.get_level_values('FrameIdx')
            if len(run_frames) > 0:
                start_frame = run_frames.min()
                end_frame = run_frames.max()
                valid_runs.append((run, start_frame, end_frame))

        if not valid_runs:
            raise ValueError("No valid runs available to display.")

        # Prepare run numbers for dropdown menu
        run_numbers = [f"Run {run}" for run, _, _ in valid_runs]

        # Load the video file
        day = os.path.basename(self.file).split('_')[1]
        filename_pattern = '_'.join(os.path.basename(self.file).split('_')[:-1])
        video_files = glob.glob(os.path.join(paths['video_folder'], f"{day}/{filename_pattern}_{view}*.avi"))
        if len(video_files) > 1:
            raise ValueError("Multiple video files found for the same view. Please check the video files.")
        elif not video_files:
            raise ValueError("No video file found for the specified view.")
        else:
            video_file = video_files[0]

        # Initialize Tkinter window
        root = tk.Tk()
        root.title("Run Steps Visualization")

        # Create dropdown menu for valid runs
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        selected_run = tk.StringVar(value=run_numbers[0])
        run_dropdown = ttk.OptionMenu(top_frame, selected_run, run_numbers[0], *run_numbers,
                                      command=lambda _: update_run())
        run_dropdown.pack(side=tk.LEFT, padx=5, pady=5)

        # Create figure and canvas
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=bottom_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Define paws and limb boxes
        paws = ['HindpawR', 'HindpawL', 'ForepawR', 'ForepawL']
        limb_boxes = {}
        box_positions = {
            'HindpawR': [0.1, 0.05, 0.35, 0.08],
            'HindpawL': [0.1, 0.15, 0.35, 0.08],
            'ForepawR': [0.55, 0.05, 0.35, 0.08],
            'ForepawL': [0.55, 0.15, 0.35, 0.08],
        }
        for paw in paws:
            left, bottom, width, height = box_positions[paw]
            ax_box = fig.add_axes([left, bottom, width, height])
            rect = patches.Rectangle((0, 0), 1, 1, facecolor='grey')
            ax_box.add_patch(rect)
            ax_box.axis('off')
            ax_box.set_xlim(0, 1)
            ax_box.set_ylim(0, 1)
            ax_box.set_title(paw, fontsize=10)
            limb_boxes[paw] = rect

        # Initialize variables
        run_index = 0
        start_frame, end_frame = valid_runs[run_index][1], valid_runs[run_index][2]
        frames = self.load_frames(video_file, start_frame, end_frame)
        if not frames:
            raise ValueError(f"No frames loaded for Run {run_index}.")

        frame_index = 0
        actual_frame_number = start_frame + frame_index

        # Display initial frame
        frame_image = ax.imshow(frames[frame_index])
        ax.axis('off')
        frame_text = ax.text(0.5, 1.02, f'Run: {run_index}, Frame: {actual_frame_number}',
                             transform=ax.transAxes, ha='center', fontsize=12)

        # Slider setup
        slider_ax = fig.add_axes([0.15, 0.85, 0.7, 0.03])
        frame_slider = Slider(slider_ax, 'Frame', 0, len(frames) - 1, valinit=0, valfmt='%d')

        # Function to display frames
        def display_frame():
            nonlocal frame_index, actual_frame_number
            frame_image.set_data(frames[frame_index])
            actual_frame_number = start_frame + frame_index
            frame_text.set_text(f'Run: {run_index}, Frame: {actual_frame_number}')
            self.update_limb_boxes(run_index, actual_frame_number, limb_boxes, paws)
            canvas.draw_idle()

        # Update slider and frames
        def update_frame(val):
            nonlocal frame_index
            frame_index = int(frame_slider.val)
            display_frame()

        def update_run():
            nonlocal run_index, start_frame, end_frame, frames, frame_index
            run_index = int(selected_run.get().split(" ")[1])
            _, start_frame, end_frame = next((r for r in valid_runs if r[0] == run_index), None)

            # Load frames for the new run
            frames.clear()
            new_frames = self.load_frames(video_file, start_frame, end_frame)
            if not new_frames:
                print(f"No frames loaded for Run {run_index}. Skipping...")
                return
            frames.extend(new_frames)

            frame_index = 0
            frame_slider.valmin = 0
            frame_slider.valmax = len(frames) - 1
            frame_slider.set_val(0)
            display_frame()

        # Connect slider and keyboard events
        frame_slider.on_changed(update_frame)

        def on_key_press(event):
            nonlocal frame_index
            if event.key == 'right' and frame_index < len(frames) - 1:
                frame_index += 1
            elif event.key == 'left' and frame_index > 0:
                frame_index -= 1
            frame_slider.set_val(frame_index)

        fig.canvas.mpl_connect('key_press_event', on_key_press)

        # Start display
        display_frame()
        root.mainloop()

    def load_frames(self, video_file, start_frame, end_frame):
        # Load frames from video file between start_frame and end_frame
        cap = cv2.VideoCapture(video_file)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {i}")
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def update_limb_boxes(self, run_index, actual_frame_number, limb_boxes, paws):
        index_levels = self.data.index.names
        if 'RunStage' in index_levels:
            # Index levels are ['Run', 'RunStage', 'FrameIdx']
            indexer = (run_index, slice(None), actual_frame_number)
        else:
            # Index levels are ['Run', 'FrameIdx']
            indexer = (run_index, actual_frame_number)

        for paw in paws:
            try:
                SwSt = self.data.loc[indexer, (paw, 'SwSt')]
                if isinstance(SwSt, pd.Series):
                    SwSt = SwSt.iloc[0]
                # Handle NaN or missing values
                if pd.isnull(SwSt) or SwSt == 'unknown':
                    SwSt = 'unknown'
                else:
                    SwSt = int(SwSt)
                # Map labels to colors
                color_mapping = {1: 'green', 0: 'red', 'unknown': 'grey'}
                limb_boxes[paw].set_facecolor(color_mapping.get(SwSt, 'grey'))
            except KeyError:
                limb_boxes[paw].set_facecolor('grey')

    def classify_steps_in_run(self, r, run_bounds):
        # Compute start_frame and end_frame with buffer
        start_frame = run_bounds[0] - self.buffer
        end_frame = run_bounds[1] + self.buffer

        # Ensure frames are within self.data index range
        start_frame = max(start_frame, self.data.index.min())
        end_frame = min(end_frame, self.data.index.max())

        # Verify that frames exist in self.data
        if not set(range(start_frame, end_frame + 1)).issubset(self.data.index):
            print(f"Warning: Some frames from {start_frame} to {end_frame} are not in self.data index.")

        # Extract features from the paws directly from self.data
        paw_columns = [
            'ForepawToeR', 'ForepawKnuckleR', 'ForepawAnkleR', 'ForepawKneeR',
            'ForepawToeL', 'ForepawKnuckleL', 'ForepawAnkleL', 'ForepawKneeL',
            'HindpawToeR', 'HindpawKnuckleR', 'HindpawAnkleR', 'HindpawKneeR',
            'HindpawToeL', 'HindpawKnuckleL', 'HindpawAnkleL', 'HindpawKneeL',
            'Nose', 'Tail1', 'Tail12'
        ]
        coords = ['x', 'z']

        # Fetch data from self.data to include buffer frames
        paw_data = self.data.loc[start_frame:end_frame, (paw_columns, coords)]
        paw_data.columns = ['{}_{}'.format(bp, coord) for bp, coord in paw_data.columns]

        # 1) Identify columns that have enough valid data for a cubic spline
        enough_points_for_spline = paw_data.notna().sum(axis=0) >= 4
        cols_for_spline = paw_data.columns[enough_points_for_spline]

        # 2) Spline interpolate only those columns
        interpolated_subset = paw_data[cols_for_spline].interpolate(
            method='spline',
            order=3,
            limit=20,
            limit_direction='both',
            axis=0
        )

        # 3) Merge back into original DataFrame shape
        interpolated = paw_data.copy()
        interpolated[cols_for_spline] = interpolated_subset

        # 4) Do the same smoothing as in your code snippet
        data_array = interpolated.values
        all_nan_cols = np.isnan(data_array).all(axis=0)
        smoothed_array = np.empty_like(data_array)
        smoothed_array[:, all_nan_cols] = np.nan
        valid_cols = ~all_nan_cols
        smoothed_array[:, valid_cols] = gaussian_filter1d(
            data_array[:, valid_cols],
            sigma=2,
            axis=0,
            mode='nearest'
        )

        # 5) Put it back into a DataFrame with the same shape and index
        smoothed_paw_data = pd.DataFrame(
            smoothed_array, index=paw_data.index, columns=paw_data.columns
        )

        # 6) Restore MultiIndex columns (if applicable)
        smoothed_paw_data.columns = pd.MultiIndex.from_tuples(
            [tuple(col.rsplit('_', 1)) for col in smoothed_paw_data.columns],
            names=['bodyparts', 'coords']
        )

        # Proceed with feature extraction using smoothed_paw_data
        feature_extractor = gfe.FeatureExtractor(data=smoothed_paw_data, fps=fps)
        frames_to_process = smoothed_paw_data.index
        feature_extractor.extract_features(frames_to_process)
        features_df = feature_extractor.features_df

        # Ensure that features_df contains the same features used during training
        features_df = features_df[self.feature_columns]

        # Predict stance/swing/unknown
        stance_pred = self.model.predict(features_df)

        # Decode predictions using label_encoders
        decoded_predictions = {}
        paw_labels = ['ForepawR', 'ForepawL', 'HindpawR', 'HindpawL']
        for idx, paw in enumerate(paw_labels):
            y_pred_col = stance_pred[:, idx]
            label_enc = self.label_encoders[paw]
            decoded_labels = label_enc.inverse_transform(y_pred_col)
            decoded_predictions[paw] = decoded_labels

        # Convert decoded predictions to DataFrame
        stance_pred_df = pd.DataFrame(decoded_predictions, index=features_df.index)

        # Plot stances using decoded labels
        self.plot_stances(r, stance_pred_df)

        # Return the DataFrame with decoded predictions
        return stance_pred_df

    def plot_stances(self, r, stance_pred_df):
        # Map labels to numerical values for plotting
        label_to_num = {'0': 0, '1': 1, 'unknown': np.nan}
        frames = np.arange(stance_pred_df.shape[0])  # X-axis: Frames or time steps

        # Create a figure with two subplots: one for forepaws and one for hindpaws
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot for Forepaws (ForepawR and ForepawL)
        for i, paw in enumerate(['ForepawR', 'ForepawL']):  # Loop through Forepaws
            y_values = stance_pred_df[paw].map(label_to_num)
            ax1.plot(frames, y_values, label=paw, marker='o')

        # Add labels, title, and legend for Forepaws
        ax1.set_ylabel('Stance/Swing')
        ax1.set_title('Stance/Swing Periods for Forepaws')
        ax1.legend()
        ax1.grid(True)

        # Plot for Hindpaws (HindpawR and HindpawL)
        for i, paw in enumerate(['HindpawR', 'HindpawL']):  # Loop through Hindpaws
            y_values = stance_pred_df[paw].map(label_to_num)
            ax2.plot(frames, y_values, label=paw, marker='o')

        # Add labels, title, and legend for Hindpaws
        ax2.set_xlabel('Frames')
        ax2.set_ylabel('Stance/Swing')
        ax2.set_title('Stance/Swing Periods for Hindpaws')
        ax2.legend()
        ax2.grid(True)

        # Show the plot
        plt.tight_layout()
        # Save plot to file
        plot_path = os.path.join(paths['filtereddata_folder'], 'LimbStuff', 'RunStances')
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(os.path.join(plot_path, f'stance_periods_run{r}.png'))
        # Close the plot
        plt.close()

    def visualize_run_steps(self, run_number, view='Front'):
        if not self.save_frames:
            print("Skipping visualize_run_steps because save_frames=False")
            return

        import matplotlib.pyplot as plt

        day = os.path.basename(self.file).split('_')[1]
        filename_pattern = '_'.join(os.path.basename(self.file).split('_')[:-1])
        video_files = glob.glob(os.path.join(paths['video_folder'], f"{day}/{filename_pattern}_{view}*.avi"))
        if len(video_files) > 1:
            raise ValueError("Multiple video files found for the same view. Please check the video files.")
        elif len(video_files) == 0:
            raise ValueError("No video file found for the specified view.")
        else:
            video_file = video_files[0]

        # Get start and end frames for the specified run
        start_frame = int(self.trial_starts[run_number])
        end_frame = int(self.trial_ends[run_number])

        # Open the video file and preload frames
        cap = cv2.VideoCapture(video_file)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {i}")
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise ValueError("No frames were loaded. Cannot proceed with visualization.")

        # Set initial frame and index setup
        frame_index = 0
        actual_frame_number = start_frame + frame_index

        # Initialize figure and axes
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.25)

        # Display the first frame
        frame_image = ax.imshow(frames[frame_index])
        ax.axis('off')

        # Add frame number text at the top
        frame_text = ax.text(0.5, 1.02, f'Frame: {actual_frame_number}', transform=ax.transAxes, ha='center', fontsize=12)

        # Define paws and limb boxes
        paws = ['ForepawL', 'HindpawL', 'ForepawR', 'HindpawR']
        limb_boxes = {}

        # Set limb box positions
        box_positions = {
            'ForepawL': [0.1, 0.05, 0.35, 0.08],
            'HindpawL': [0.1, 0.15, 0.35, 0.08],
            'ForepawR': [0.55, 0.05, 0.35, 0.08],
            'HindpawR': [0.55, 0.15, 0.35, 0.08],
        }

        for paw in paws:
            left, bottom, width, height = box_positions[paw]
            ax_box = fig.add_axes([left, bottom, width, height])
            rect = patches.Rectangle((0, 0), 1, 1, facecolor='grey')
            ax_box.add_patch(rect)
            ax_box.axis('off')
            ax_box.set_xlim(0, 1)
            ax_box.set_ylim(0, 1)
            ax_box.set_title(paw, fontsize=10)
            limb_boxes[paw] = rect

        # Create skip buttons with fixed delta values, using partial for callbacks
        button_positions = [(0.05 + i * 0.09, 0.9, 0.08, 0.04) for i in range(8)]
        button_labels = ['-1000', '-100', '-10', '-1', '+1', '+10', '+100', '+1000']
        button_deltas = [-1000, -100, -10, -1, 1, 10, 100, 1000]

        for pos, label, delta in zip(button_positions, button_labels, button_deltas):
            ax_button = fig.add_axes(pos)
            button = Button(ax_button, label)
            button.on_clicked(partial(self._update_frame_via_button, delta=delta))

        # Create slider above limb boxes
        ax_slider = plt.axes([0.15, 0.86, 0.7, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, len(frames) - 1, valinit=frame_index, valfmt='%d')

        def update_frame(delta=None, frame_number=None):
            nonlocal frame_index, actual_frame_number
            if delta is not None:
                frame_index = max(0, min(frame_index + delta, len(frames) - 1))
            elif frame_number is not None:
                frame_index = frame_number
            actual_frame_number = start_frame + frame_index

            # Update the frame image
            frame_image.set_data(frames[frame_index])

            # Update limb boxes based on stance or swing
            if actual_frame_number in self.data.index:
                for paw in paws:
                    SwSt = self.data.loc[actual_frame_number, (paw, 'SwSt')]
                    if SwSt == 1:
                        limb_boxes[paw].set_facecolor('green')
                    elif SwSt == 0:
                        limb_boxes[paw].set_facecolor('red')
                    else:
                        limb_boxes[paw].set_facecolor('grey')
            else:
                for paw in paws:
                    limb_boxes[paw].set_facecolor('grey')

            # Update frame number display
            frame_text.set_text(f'Frame: {actual_frame_number}')

            # Synchronize slider
            slider.eventson = False
            slider.set_val(frame_index)
            slider.eventson = True

            # Refresh canvas
            fig.canvas.draw_idle()
            plt.pause(0.001)

        # Slider event to move frames
        def slider_update(val):
            frame_number = int(slider.val)
            update_frame(frame_number=frame_number)
        slider.on_changed(slider_update)

        # Set update_frame function to be accessible within _update_frame_via_button
        self.update_frame = update_frame

        plt.show()

    def _update_frame_via_button(self, delta, event=None):
        # Call update_frame with the delta value
        self.update_frame(delta=delta)

    def check_run_starts(self, forward_data, forward_chunks):
        valid_forward_chunks = []
        for chunk in forward_chunks:
            # Extract data for the current chunk
            chunk_data = forward_data.loc[chunk[0]:chunk[-1]]
            # Get the x-coordinate of Tail1
            tail1_x = chunk_data.loc[:, ('Tail1', 'x')]
            # Check if any Tail1 x-coordinate is between 0 and 30
            if ((tail1_x >= 0) & (tail1_x <= 30)).any():
                # Include this chunk as it meets the condition
                valid_forward_chunks.append(chunk)
            #else:
                # Exclude the chunk and optionally log this action
               # print(f"Excluding chunk {chunk} as Tail1 x-coordinate is not between 0 and 30.")
        return valid_forward_chunks

    def check_post_runs(self, forward_data, forward_chunks):
        post_transition_mask = forward_data.loc[:, ('Tail6', 'x')] > 470
        forward_post_mask = forward_data.loc[:, ('Nose', 'x')] > forward_data.loc[:, ('Tail1', 'x')] # added this in 21/11/2024
        mask = post_transition_mask & forward_post_mask
        post_transition = forward_data[mask]
        first_transition = post_transition.index[0]
        # drop runs that occur after the first transition
        correct_forward_chunks = [chunk for chunk in forward_chunks if chunk[0] < first_transition]
        return correct_forward_chunks

    def check_real_forward_gap(self, run_data, forward_chunks):
        checked_forward_chunks = forward_chunks.copy()
        i = 0
        while i < len(checked_forward_chunks) - 1:
            upper_bound = checked_forward_chunks[i+1][0]
            lower_bound = checked_forward_chunks[i][-1]
            data = run_data.loc[lower_bound:upper_bound]

            # check that nose x is never < tail1 x and that nose x is never < 0
            nose_tail_mask = data.loc(axis=1)['Nose', 'x'] - data.loc(axis=1)['Tail1', 'x'] < -20
            nose_tail_data = data[nose_tail_mask]

            nose_mask = data.loc(axis=1)['Nose', 'x'] < 0
            nose_data = data[nose_mask]

            if len(nose_tail_data) > 0 and len(nose_data) > 0:
                # keep this chunk as it is
                pass
            else:
                # combine this chunk and the next chunk into one chunk
                checked_forward_chunks[i] = [checked_forward_chunks[i][0], checked_forward_chunks[i+1][-1]]
                checked_forward_chunks = np.delete(checked_forward_chunks, i+1, axis=0)
            i += 1
        return checked_forward_chunks

    def check_run_backs(self, data, forward_chunks):
        runbacks = []
        true_run = []
        i = 0

        # only check for runbacks if there are more than 1 forward runs/up to the second to last forward run
        while i < len(forward_chunks) - 1:
            run = forward_chunks[i]
            this_run_start = run[0]
            next_run = forward_chunks[i + 1]
            next_run_start = next_run[0]
            run_data = data.loc[this_run_start:next_run_start]

            # check if mice run backwards between this run and the next
            runback_mask = run_data.loc(axis=1)['Nose', 'x'] < run_data.loc(axis=1)['Tail1', 'x']
            runback_data = run_data[runback_mask]

            # check if mice step off the platform between this run and the next # todo they dont have to step off??
            if len(runback_data) > 0:
                step_off_mask = run_data.loc(axis=1)['Tail1', 'x'] < 0 # i think this is checking that after running back the mice exit the belt
                step_off_data = run_data[step_off_mask]

                # if mice meet these conditions, add this run to the runbacks list
                if len(step_off_data) > 0:
                    runbacks.append(run)
                else:
                    true_run.append(run)
            else:
                # # No backwards running detected in this snippet
                # raise ValueError("No backwards running detected in this snippet")

                # check if reappears on the start platform between entering belt on this run and the next
                on_belt_mask = np.all(run_data.loc(axis=1)[['Tail1','Nose'], 'x'] > 0,axis=1)
                on_belt_data = run_data[on_belt_mask]
                on_belt_chunks = utils.Utils().find_blocks(on_belt_data.index, gap_threshold=20, block_min_size=20)
                last_on_belt = on_belt_chunks[-1][-1]

                on_plat_mask = np.any(run_data.loc(axis=1)[['Nose','EarR','EarL','Tail1','ForepawToeL','HindpawToeL'], 'x'] < 0, axis=1)
                on_plat_data = run_data[on_plat_mask]

                reappear_mask = on_plat_data.index > last_on_belt
                reappear_data = on_plat_data[reappear_mask]

                if len(reappear_data) > 0:
                    runbacks.append(run)
                else:
                    true_run.append(run) # even if this is not a true run, will be picked up back in find_real_run_vs_rbs as multiple true runs
            i += 1

        true_run.append(forward_chunks[-1])

        return runbacks, true_run

    def find_real_run_vs_rbs(self, r, run_data):
        #run_data = self.data.loc(axis=0)[self.trial_starts[r]:self.trial_ends[r]]

        # Use run_data instead of self.data
        trial_start = self.trial_starts[r]
        trial_end = self.trial_ends[r]
        if trial_start == SENTINEL_VALUE or trial_end == SENTINEL_VALUE:
            raise ValueError(f"Cannot process run {r} due to sentinel value.")

        run_data = run_data.loc[trial_start:trial_end]

        # filter by when mouse facing forward
        facing_forward_mask = self.find_forward_facing_bool(run_data, xthreshold=20, zthreshold=45, nosezthreshold=45, nose=False) #todo: changed zthreshold from 40 to 30 on 14/1/25 as picking up when mouse standing up and added nose z abs thresh # on 15/1/25 changed z thresh to 45 as dont think there is need to break up runs like that
        facing_forward = run_data[facing_forward_mask]
        forward_chunks = utils.Utils().find_blocks(facing_forward.index, gap_threshold=50, block_min_size=25) #todo: changed gap_threshold from 10 to 20 on 14/1/25 # on 15/1/25 changed gap threshold to 50 from 20
        if len(forward_chunks) > 1:
            # check all forward runs contain data from start of run
            forward_chunks = self.check_run_starts(facing_forward, forward_chunks)
            if len(forward_chunks) > 1:
                forward_chunks = self.check_post_runs(facing_forward, forward_chunks)
                if len(forward_chunks) > 1:
                    # check if any backwards running or appearance on stationary platform between forward chunks to combine chunks from the same run back together
                    forward_chunks = self.check_real_forward_gap(run_data, forward_chunks) #todo: this is new and experimental!! (16/1/25)
                    if len(forward_chunks) > 1:
                        runbacks, forward_chunks = self.check_run_backs(run_data, forward_chunks)
                        if len(forward_chunks) > 1:
                            raise ValueError("More than one run detected in a trial")
                    else:
                        runbacks = []
                else:
                    runbacks = []
            else:
                runbacks = []
        elif len(forward_chunks) == 0:
            raise ValueError("No runs detected in a trial")
        else:
            runbacks = []
        return forward_chunks, runbacks

    #-------------------------------------------------------------------------------------------------------------------
    #------------------------------------------ Finding runstages ------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    def find_run_stages(self):
        # Ensure the error log file has headers
        if not os.path.exists(self.error_log_file):
            with open(self.error_log_file, 'w', newline='') as log_file:
                writer = csv.writer(log_file)
                writer.writerow(logging['error'])

        runs = self.data.index.get_level_values('Run').unique()
        self.data['initiating_limb'] = np.nan

        # Initialize lists to store run stages
        self.run_starts = [None] * len(self.trial_starts)
        self.run_ends = [None] * len(self.trial_starts)
        self.run_ends_steps = [None] * len(self.trial_starts)
        self.transitions = [None] * len(self.trial_starts)
        self.taps = [None] * len(self.trial_starts)

        for r in runs:
            if self.trial_starts[r] == SENTINEL_VALUE or self.trial_ends[r] == SENTINEL_VALUE:
                print(f"Skipping run {r} due to sentinel value.")
                continue

            try:
                step_data, limb_data, paw_touchdown = self.get_run_data(r)

                # 1. RunStart
                try:
                    self.find_run_start(r, paw_touchdown, limb_data)
                except Exception as e:
                    self.run_starts[r] = [] # Fill with blank value
                    self._log_error('RunStart', str(e), r, overwrite=False)

                # 2. RunEnd
                self.find_run_end(r, paw_touchdown, limb_data)

                # 3. Transition
                try:
                    self.find_transition(r, paw_touchdown, limb_data)
                except Exception as e:
                    self.transitions[r] = []  # Fill with blank value
                    self._log_error('Transition', str(e), r, overwrite=False)

                # 4. Taps
                self.find_taps(r, limb_data)
            except Exception as e:
                print(f"Error processing run stages at run {r}: {e}")

        self.create_runstage_index()
        self.plot_run_stage_frames('Transition', 'Side')
        self.plot_run_stage_frames('RunStart', 'Side')



    def _log_error(self, error_type, error_message, run_number, overwrite=False):
        """
        Logs an error message to the error log file.

        Args:
            error_type (str): The type of error (e.g., 'RunStart', 'Transition').
            error_message (str): A detailed error message.
            run_number (int): The run number associated with the error.
            overwrite (bool): Whether to overwrite the log file (default: False).
        """
        mode = 'w' if overwrite else 'a'  # Set mode to 'w' for overwrite or 'a' for append

        # Write to the log file
        try:
            with open(self.error_log_file, mode, newline='') as log_file:
                writer = csv.writer(log_file)

                if overwrite:
                    writer.writerow(logging['error'])

                # Write all columns *in the exact same order*
                writer.writerow([
                    self.file,
                    self.exp,
                    self.speed,
                    self.repeat_extend,
                    self.exp_wash,
                    self.day,
                    self.vmt_type,
                    self.vmt_level,
                    self.prep,
                    self.mouseID,
                    run_number,
                    error_type,
                    error_message,
                ])
        except Exception as e:
            print(f"Failed to write to the error log file: {e}")

        # After writing, we also read back, sort, and re-save so it's tidier.
        df = pd.read_csv(self.error_log_file)
        df = df.sort_values(by=['File','RunNumber'])
        df.to_csv(self.error_log_file, index=False)

    def get_run_data(self, r):
        step_data = self.data.loc(axis=0)[r].loc(axis=1)[['ForepawR', 'ForepawL', 'HindpawR', 'HindpawL']]
        limb_data = self.data.loc(axis=0)[r].loc(axis=1)[['ForepawToeR', 'ForepawKnuckleR', 'ForepawAnkleR', 'ForepawKneeR',
                                                            'ForepawToeL', 'ForepawKnuckleL', 'ForepawAnkleL', 'ForepawKneeL',
                                                            'HindpawToeR', 'HindpawKnuckleR', 'HindpawAnkleR', 'HindpawKneeR',
                                                            'HindpawToeL', 'HindpawKnuckleL', 'HindpawAnkleL', 'HindpawKneeL']]
        step_data = step_data.replace(['unknown','1', '0'], [np.nan, 1, 0])

        paw_labels=['ForepawR', 'ForepawL', 'HindpawR', 'HindpawL']
        # check when all 4 paws are first on the belt - need x postion of all paws to be > 0 and paws to be in stance
        paw_touchdown = self.detect_paw_touchdown(step_data, limb_data, paw_labels, x_threshold=0)

        return step_data, limb_data, paw_touchdown

    def find_first_touchdown_paw_block(self, touchdown_series, first_touchdown_x4):
        # Extract frames where the paw is touching down before first_touchdown_x4
        paw_touchdown_frames = touchdown_series[touchdown_series.index < first_touchdown_x4 + 1]
        paw_touchdown_frames = paw_touchdown_frames[
            paw_touchdown_frames]  # Only frames where the paw is touching down

        # If there are no touchdown frames before first_touchdown_x4, skip
        if paw_touchdown_frames.empty:
            # print(f"No touchdown frames found for {paw} before frame {first_touchdown_x4}")
            return

        # Use find_blocks to find continuous blocks of touchdown frames
        blocks = utils.Utils().find_blocks(paw_touchdown_frames.index, gap_threshold=5, block_min_size=0)
        return blocks

    def find_run_start(self, r, paw_touchdown, limb_data):
        paw_labels = paw_touchdown.columns
        first_touchdown_x4, paw_x4, valid_touchdown, valid_touchdown_lenient = self.find_touchdown_all4s(paw_touchdown, paw_labels, r, time='first', limb_data=limb_data)

        # Find the first touchdown frame for each paw leading up to first_touchdown_x4
        paw_first_touchdowns = {}
        for paw in paw_labels:
            # Get the touchdown Series for the paw
            touchdown_series = valid_touchdown[paw]

            blocks = self.find_first_touchdown_paw_block(touchdown_series, first_touchdown_x4)

            if len(blocks) == 0 and paw_x4 == paw:
                continue
            elif len(blocks) == 0 and paw_x4 != paw:
                # use lenient touchdown detection
                touchdown_series = valid_touchdown_lenient[paw]
                blocks = self.find_first_touchdown_paw_block(touchdown_series, first_touchdown_x4)

            # Find the block that ends at or just before first_touchdown_x4 for this paw
            block_found = False
            for block in reversed(blocks):
                # check if the block ends at or before first_touchdown_x4
                if block[0] <= first_touchdown_x4:
                    # Also check if hindpaw has touched down before this block
                    hindpaw_touchdowns = valid_touchdown[['HindpawR', 'HindpawL']]
                    hindpaw_preceding_touchdown = hindpaw_touchdowns[hindpaw_touchdowns.index < block[0]]
                    hind_paw_preceding_present = hindpaw_preceding_touchdown.any(axis=0).any()
                    if hind_paw_preceding_present and len(blocks) > 1:
                        # this is not the block we are looking for
                        continue
                    else:
                        first_touchdown_frame = block[0]
                        paw_first_touchdowns[paw] = first_touchdown_frame
                        block_found = True
                        break
            if not block_found:
                raise ValueError(f"No stepping sequence found leading up to {first_touchdown_x4} for {paw}, run {r}")

        if paw_first_touchdowns:
            # Find the earliest first touchdown frame among all paws
            earliest_first_touchdown_frame = min(paw_first_touchdowns.values())
            # Identify which paw(s) have this earliest frame
            initiating_paws = [paw for paw, frame in paw_first_touchdowns.items() if
                               frame == earliest_first_touchdown_frame]
            if np.logical_and(len(initiating_paws) == 1, 'Hind' not in initiating_paws[0]):
                #self.run_starts.append(earliest_first_touchdown_frame)
                self.run_starts[r] = earliest_first_touchdown_frame
                # adjust 'running' column to start at the first touchdown frame
                current_bound0 = self.data.loc(axis=0)[r].loc(axis=1)['running'].index[self.data.loc(axis=0)[r].loc(axis=1)['running'] == True][0]
                for f in range(earliest_first_touchdown_frame,current_bound0):
                    self.data.loc[(r,f),'running'] = True
                # set the initiating limb for this run
                self.data.loc[(r,earliest_first_touchdown_frame),'initiating_limb'] = initiating_paws[0]
            else:
                print(f"Run {r}: Multiple or no initiating paws found.")
        else:
            raise ValueError(f"No valid stepping sequences found leading up to the all-paws touchdown in run {r}")

    def find_run_end(self, r, paw_touchdown, limb_data):
        paw_labels = paw_touchdown.columns

        # find frame where first paw touches down for the last time (end of stance)
        last_touchdown_x4, valid_touchdown = self.find_touchdown_all4s(paw_touchdown, paw_labels, r, time='last', limb_data=limb_data)

        # find frame where mouse exits frame
        tail_data = self.data.loc(axis=0)[r, last_touchdown_x4:].loc(axis=1)[['Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7',
                                                          'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12']]
        tail_present = tail_data[tail_data.any(axis=1)]
        tail_blocks = utils.Utils().find_blocks(tail_present.index.get_level_values('FrameIdx'), gap_threshold=100, block_min_size=20)
        run_end = tail_blocks[0][-1]

        # adjust 'running' column to end at the run end frame
        current_bound1 = self.data.loc(axis=0)[r].loc(axis=1)['running'].index[self.data.loc(axis=0)[r].loc(axis=1)['running'] == True][-1]
        for f in range(current_bound1,run_end):
            self.data.loc[(r,f),'running'] = True

        self.run_ends[r] = run_end
        self.run_ends_steps[r] = last_touchdown_x4

        # self.run_ends.append(run_end)
        # self.run_ends_steps.append(last_touchdown_x4)

    def find_transition(self, r, paw_touchdown, limb_data):
        """
        Identifies the transition phase in a run by focusing on forepaws and their movements.

        Parameters:
            r (int): Run index.
            paw_touchdown (DataFrame): Frame-by-frame touchdown status (swing, stance) for each limb.
            limb_data (DataFrame): Raw x/y/z coordinate data for each limb marker.

        Updates:
            - `self.transitions`: List of transition frames.
            - `self.data['initiating_limb']`: Marks the initiating limb for transitions in `self.data`.
        """

        # Step 1: Focus on forepaws
        forepaw_labels = pd.unique([paw for paw in limb_data.columns.get_level_values(level=0) if 'Forepaw' in paw])

        # Step 2: Trim paw_touchdown and limb_data to the run's start and end (touchdown)
        paw_touchdown_snippet = paw_touchdown.loc[self.run_starts[r]:self.run_ends[r], ['ForepawL', 'ForepawR']]
        limb_data_snippet = limb_data.loc[self.run_starts[r]:self.run_ends[r], forepaw_labels]

        # Step 3: Find frames where forepaws enter touchdown (swing to stance)
        last_frame_swing_mask = paw_touchdown_snippet.shift(1) == False
        current_frame_stance_mask = paw_touchdown_snippet == True
        swing_to_stance_mask = np.logical_and(last_frame_swing_mask, current_frame_stance_mask)

        # Step 4: Get mean of interpolated x in toe and knuckle for each frame
        limb_x = limb_data_snippet.loc(axis=1)[:, 'x'].droplevel('coords', axis=1)
        limb_x = limb_x.drop(columns=['ForepawKneeR', 'ForepawKneeL', 'ForepawAnkleR', 'ForepawAnkleL'])
        limb_x.columns = pd.MultiIndex.from_tuples([(col, col[-1]) for col in limb_x.columns],
                                                   names=['bodyparts', 'side'])

        limb_x_interp = limb_x.interpolate(method='spline', order=3, limit=20, limit_direction='both', axis=0)
        limb_x_mean = limb_x_interp.groupby(axis=1, level='side').mean()

        # Step 5: Find frames where the limb is past the transition point in f0 and f+1 (post-transition position)
        post_transition_mask = np.logical_and(limb_x_mean > 470, limb_x_mean.shift(-1) > 470)

        # Step 6: Verify column order (ensure alignment of masks)
        if not np.logical_and(post_transition_mask.columns[0] == 'L', paw_touchdown_snippet.columns[0] == 'ForepawL'):
            raise ValueError("Columns are not in the expected order!!")

        # Step 7: Create transition masks
        transition_mask = np.logical_and(post_transition_mask.values, paw_touchdown_snippet.values)  # Basic transition
        transition_step_mask = np.logical_and(post_transition_mask.values, swing_to_stance_mask.values)  # Backup

        # Step 8: Find first frame for each mask
        transition_frame = paw_touchdown_snippet[transition_mask].index[0]
        transition_frame_step = paw_touchdown_snippet[transition_step_mask].index[0]

        # Step 9: Determine initiating paw for the transition (basic or sliding)
        transition_paw_mask = np.all(paw_touchdown_snippet[transition_mask].iloc[1:5] == True, axis=0)
        transition_paws = paw_touchdown_snippet[transition_mask].iloc[0][transition_paw_mask].index

        if len(transition_paws) > 1:
            # Case 1: Both paws are detected in the transition frame
            if transition_frame == transition_frame_step:
                position = np.where(transition_step_mask)[0][0]
                paw_mask = transition_step_mask[position]
                transition_paw = paw_touchdown_snippet.loc[transition_frame_step].loc[paw_mask].index[0]
                other_paw = paw_touchdown_snippet.loc[transition_frame_step].loc[~paw_mask].index[0]
                self.data.loc[(r, transition_frame), 'initiating_limb'] = f"{other_paw}_slid"
            else:
                # Case 1a: One paw slides, and the other steps
                position = np.where(transition_step_mask)[0][0]
                paw_mask = transition_step_mask[position]
                transition_paw = paw_touchdown_snippet.loc[transition_frame_step].loc[paw_mask].index[0]
                other_paw = paw_touchdown_snippet.loc[transition_frame].loc[~paw_mask].index[0]
                self.data.loc[(r, transition_frame), 'initiating_limb'] = f"{other_paw}_slid"
        elif len(transition_paws) == 0:
            # Case 2: No paws meet the criteria for the transition frame
            if transition_frame != transition_frame_step:
                slide_mask = np.all(paw_touchdown_snippet.loc[:transition_frame].iloc[-10:] == True, axis=0)
                slide_paws = paw_touchdown_snippet.columns[slide_mask]
                if len(slide_paws) == 1:
                    transition_paw = slide_paws[0]
                    self.data.loc[(r, transition_frame), 'initiating_limb'] = f"{transition_paw}_slid"
                else:
                    raise ValueError("No or multiple paws detected during slide resolution")
            else:
                transition_paw = []
                print(f"No paw detected in transition frame for run {r}")
        else:
            # Case 3: Single paw meets the criteria
            transition_paw = transition_paws[0]

        # Step 10: Handle stepping transitions
        if transition_frame == transition_frame_step:
            self.data.loc[(r, transition_frame), 'initiating_limb'] = transition_paw
            self.transitions[r] = transition_frame
        else:
            transition_paw_mask_step = np.all(paw_touchdown_snippet.loc[transition_frame_step:].iloc[1:10] == True,
                                              axis=0)
            transition_paw_step = paw_touchdown_snippet[transition_step_mask].iloc[0][transition_paw_mask_step].index
            if len(transition_paw_step) > 1:
                raise ValueError("More than one paw detected in transition frame")
            elif len(transition_paw_step) == 0:
                raise ValueError("No paw detected in transition frame")
            else:
                transition_paw_step = transition_paw_step[0]
                self.data.loc[(r, transition_frame_step), 'initiating_limb'] = transition_paw_step
                self.data.loc[(r, transition_frame), 'initiating_limb'] = f"{transition_paw}_slid"
            self.transitions[r] = transition_frame_step

    def find_taps(self, r, limb_data):
        if self.run_starts[r]:
            # find taps and can measure this as a duration of time where mouse has a paw either hovering or touching the belt without stepping
            pre_run_data = self.data.loc(axis=0)[r, :self.run_starts[r]]
            forepaw_labels = pd.unique([paw for paw in limb_data.columns.get_level_values(level=0) if 'Forepaw' in paw])

            limb_data_snippet = limb_data.loc[pre_run_data.index.get_level_values('FrameIdx'), forepaw_labels]

            # Get y of forepaw toes
            toe_z = limb_data_snippet.loc(axis=1)[['ForepawToeL', 'ForepawToeR'], 'z'].droplevel('coords', axis=1)
            toe_x = limb_data_snippet.loc(axis=1)[['ForepawToeL', 'ForepawToeR'], 'x'].droplevel('coords', axis=1)

            tap_mask = np.logical_and(toe_z < 1, toe_x > 0, toe_x < 50)
            taps_L = toe_x['ForepawToeL'][tap_mask['ForepawToeL']]
            taps_R = toe_x['ForepawToeR'][tap_mask['ForepawToeR']]

            # find blocks of taps
            tap_blocks_L = utils.Utils().find_blocks(taps_L.index, gap_threshold=5, block_min_size=5)
            tap_blocks_R = utils.Utils().find_blocks(taps_R.index, gap_threshold=5, block_min_size=5)

            # ensure taps are not too close to the start of the run or a runback
            runback_idxs = self.data.loc(axis=1)['rb'].index[self.data.loc(axis=1)['rb'] == True]
            tap_blocks_L_final = []
            tap_blocks_R_final = []
            buffer = 50
            if len(tap_blocks_L) > 0:
                for block in tap_blocks_L:
                    if np.logical_or(
                            any([self.run_starts[r] - block[0] <= buffer, self.run_starts[r] - block[-1] <= buffer]),
                            np.any([np.in1d(list(range(block[0],block[1])), runback_idxs.get_level_values('FrameIdx')), np.in1d(list(range(block[0],block[1])), runback_idxs.get_level_values('FrameIdx') - buffer)])):
                        #print(f"Tap block removed from run {r} at {block}")
                        pass
                    else:
                        tap_blocks_L_final.append(block)
            if len(tap_blocks_R) > 0:
                for block in tap_blocks_R:
                    if np.logical_or(
                            any([self.run_starts[r] - block[0] <= buffer, self.run_starts[r] - block[-1] <= buffer]),
                            np.any([np.in1d(list(range(block[0], block[1])), runback_idxs.get_level_values('FrameIdx')),
                                    np.in1d(list(range(block[0], block[1])),
                                            runback_idxs.get_level_values('FrameIdx') - buffer)])):
                        #print(f"Tap block removed from run {r} at {block}")
                        pass
                    else:
                        tap_blocks_R_final.append(block)

            # return list of frames contained within tap blocks (ie the range between block ends) as list for each side
            tap_frames_L = [list(range(block[0], block[-1])) for block in tap_blocks_L_final]
            tap_frames_R = [list(range(block[0], block[-1])) for block in tap_blocks_R_final]

            self.taps[r] = (tap_frames_L, tap_frames_R)
        else:
            self.taps[r] = ([], [])

    def create_runstage_index(self):
        # First add in a 'TapsL' and 'TapsR' column
        self.data['TapsL'] = False
        self.data['TapsR'] = False
        for r, taps in enumerate(self.taps):
            if taps is None:
                # means we never stored any tap info for run r
                continue
            # Now taps is guaranteed to be something like ([], [])
            left_taps, right_taps = taps  # or taps[0], taps[1]
            for tap_idx in left_taps:
                self.data.loc[(r, tap_idx), 'TapsL'] = True
            for tap_idx in right_taps:
                self.data.loc[(r, tap_idx), 'TapsR'] = True

        # Instantiate RunStage column
        self.data['RunStage'] = 'None'

        # Prepare to collect indices to drop after 'run_end' for each run
        indices_to_drop = []

        # Get MultiIndex levels for 'Run' and 'FrameIdx'
        run_levels = self.data.index.get_level_values('Run')
        frame_levels = self.data.index.get_level_values('FrameIdx')

        for r in run_levels.unique():
            if r == SENTINEL_VALUE:
                continue  # Skip sentinel-valued runs

            # Retrieve run stage bounds; skip if any are missing
            run_start = self.run_starts[r]
            transition = self.transitions[r]
            run_end = self.run_ends[r]
            run_end_steps = self.run_ends_steps[r]

            # Skip runs with missing data
            if not run_start or not transition:
                print(f"Skipping run {r} due to missing stage bounds.")
                continue

            # Boolean mask for the current run
            is_current_run = (run_levels == r)

            # TrialStart stage
            trialstart_mask = is_current_run & (frame_levels < run_start)
            self.data.loc[trialstart_mask, 'RunStage'] = 'TrialStart'

            # RunStart stage
            runstart_mask = is_current_run & (frame_levels >= run_start) & (frame_levels < transition)
            self.data.loc[runstart_mask, 'RunStage'] = 'RunStart'

            # Transition stage
            transition_mask = is_current_run & (frame_levels >= transition) & (frame_levels <= run_end_steps)
            self.data.loc[transition_mask, 'RunStage'] = 'Transition'

            # RunEnd stage
            runend_mask = is_current_run & (frame_levels > run_end_steps) & (frame_levels <= run_end)
            self.data.loc[runend_mask, 'RunStage'] = 'RunEnd'

            # Collect indices to drop after run_end for this run
            drop_mask = is_current_run & (frame_levels > run_end)
            indices_to_drop.extend(self.data.index[drop_mask])

        # Drop all collected indices at once
        if indices_to_drop:
            self.data.drop(index=indices_to_drop, inplace=True)

        # Add in runbacks from 'rb' column
        runbacks = self.data.loc(axis=1)['rb'].index[self.data.loc(axis=1)['rb'] == True]
        self.data.loc[runbacks, 'RunStage'] = 'RunBack'

        # Set 'RunStage' as part of the index and reorder
        self.data.set_index('RunStage', append=True, inplace=True)
        self.data = self.data.reorder_levels(['Run', 'RunStage', 'FrameIdx'])

    ################################################# Helper functions #################################################
    def find_touchdown_all4s(self, paw_touchdown, paw_labels, r, time, limb_data):
        # Check all touchdown during running phase
        touchdownx4 = {}
        valid_touchdown = pd.DataFrame(columns=paw_touchdown.columns, index=paw_touchdown.index)
        valid_touchdown_lenient = pd.DataFrame(columns=paw_touchdown.columns, index=paw_touchdown.index)
        for paw in paw_labels:
            touchdown_series = paw_touchdown[paw]
            if touchdown_series.any():
                # Frames where the paw is in touchdown
                timed_touchdown_frame_mask = touchdown_series == True

                # Get paw prefix and side
                if 'Forepaw' in paw:
                    paw_prefix = 'Forepaw'
                elif 'Hindpaw' in paw:
                    paw_prefix = 'Hindpaw'
                else:
                    raise ValueError(f"Unknown paw label: {paw}")

                side = paw[-1]  # 'R' or 'L'

                # Get markers for this paw (only toe and knuckle markers)
                markers = [col for col in limb_data.columns
                           if col[0].startswith(paw_prefix) and col[0].endswith(side)
                           and ('Toe' in col[0] or 'Knuckle' in col[0])]

                # Get x positions for the paw's markers
                limb_positions = limb_data.loc[:, markers]

                # Calculate the mean x and z position for the paw
                x = limb_positions.xs('x', level='coords', axis=1)
                z = limb_positions.xs('z', level='coords', axis=1)

                x_interp = x.interpolate(method='spline', order=3, limit=20, limit_direction='both', axis=0)
                z_interp = z.interpolate(method='spline', order=3, limit=20, limit_direction='both', axis=0)

                x_mean = x_interp.mean(axis=1)
                z_mean = z_interp.mean(axis=1)

                # Further filter touchdown_series to only include frames where x_mean > 0
                x_condition = x_mean > 1 # changed 5/12/24 to fit 243 06 run 22 better
                z_condition = z_mean < 2

                # Combine with touchdown_series
                valid_touchdown_series = touchdown_series[timed_touchdown_frame_mask & x_condition & z_condition]
                valid_touchdown[paw].loc(axis=0)[valid_touchdown_series.index] = valid_touchdown_series

                valid_touchdown_series_lenient = touchdown_series[timed_touchdown_frame_mask & x_condition]
                valid_touchdown_lenient[paw].loc(axis=0)[valid_touchdown_series_lenient.index] = valid_touchdown_series_lenient

                # Now find the first or last valid touchdown frame
                if valid_touchdown_series.empty:
                    raise ValueError(f"No valid touchdown frames found for {paw} in run {r}")
                else:
                    if time == 'first':
                        # check
                        timed_touchdown_frame = valid_touchdown_series.index[0]

                    elif time == 'last':
                        timed_touchdown_frame = valid_touchdown_series.index[-1]
                    touchdownx4[paw] = timed_touchdown_frame
            else:
                raise ValueError(f"No touchdown detected for {paw} in run {r}")

        # fill nans in valid_touchdown with False
        valid_touchdown.fillna(False, inplace=True)
        valid_touchdown_lenient.fillna(False, inplace=True)

        if time == 'first':
            # Find last paw to touch down for first time
            timed_touchdown = max(touchdownx4.values())
            stepping_paw = [paw for paw, frame in touchdownx4.items() if frame == timed_touchdown][0]
            return timed_touchdown, stepping_paw, valid_touchdown, valid_touchdown_lenient

        elif time == 'last':
            # Find first paw to touch down for last time
            timed_touchdown = min(touchdownx4.values())
            return timed_touchdown, valid_touchdown

    def detect_paw_touchdown(self, step_data, limb_data, paw_labels, x_threshold):
        """
        Detects paw touchdown frames for each paw individually based on:
        - Paw is in stance phase (step_data).
        - Paw's x-position is greater than x_threshold (limb_data).

        Parameters:
        - step_data: DataFrame with paw stance/swing data for each paw (values are 1.0 for stance, 0.0 for swing).
        - limb_data: DataFrame with limb position data, must have x positions for paws.
        - paw_labels: List of paw labels to check (e.g., ['ForepawR', 'ForepawL', 'HindpawR', 'HindpawL']).
        - x_threshold: The x-position threshold (e.g., 0) to check if paw has crossed the start line.

        Returns:
        - touchdown_frames: DataFrame with index as frames and columns as paw_labels, values are True/False indicating if conditions are met.
        """
        # Ensure indices are aligned
        common_index = step_data.index.intersection(limb_data.index)
        step_data = step_data.loc[common_index]
        limb_data = limb_data.loc[common_index]

        # Initialize a DataFrame to hold the touchdown status for each paw
        touchdown_frames = pd.DataFrame(index=common_index, columns=paw_labels)

        for paw in paw_labels:
            # Extract paw prefix and side
            if 'Forepaw' in paw:
                paw_prefix = 'Forepaw'
            elif 'Hindpaw' in paw:
                paw_prefix = 'Hindpaw'
            else:
                raise ValueError(f"Unknown paw label: {paw}")

            side = paw[-1]  # 'R' or 'L'

            # Check if paw is in stance (assuming stance is labeled as 1.0)
            in_stance = step_data[paw] == 1

            # Get markers for this paw
            markers = [col for col in limb_data.columns if col[0].startswith(paw_prefix) and col[0].endswith(side)]

            # Get x positions for the paw's markers
            x_positions = limb_data.loc[:, markers]

            # Calculate the mean x position for the paw
            x_mean = x_positions.xs('x', level='coords', axis=1).mean(axis=1)

            # Check if x position is greater than the threshold
            x_condition = x_mean > x_threshold

            # Combine conditions
            touchdown_frames[paw] = np.logical_and(in_stance.values.flatten(), x_condition.values.flatten())

        return touchdown_frames

    ################################################ Plotting functions ################################################

    def plot_run_stage_frames(self, run_stage, view='Side'):
        """
        Save the first video frame for a specified run stage across all runs.
        Parameters:
        - run_stage: str, the run stage to visualize ('RunStart', 'Transition', 'RunEnd')
        - view: str, the camera view to use ('Side', 'Front', etc.')
        """
        if not self.save_frames:
            print("Skipping plot_run_stage_frames because save_frames=False")
            return

        runs = self.data.index.get_level_values('Run').unique()

        # Video file selection
        day = os.path.basename(self.file).split('_')[1]
        filename_pattern = '_'.join(os.path.basename(self.file).split('_')[:-1])
        video_files = glob.glob(os.path.join(paths['video_folder'], f"{day}/{filename_pattern}_{view}*.avi"))
        if not video_files:
            raise ValueError(f"No video files found for view '{view}'.")
        video_file = video_files[0]

        # Mouse ID and experiment condition extraction
        file_path_parts = self.file.split(os.sep)
        mouse_id, experiment_condition = None, []
        for i, part in enumerate(file_path_parts):
            if 'Round' in file_path_parts[i - 1]:
                experiment_condition.append(part)
            elif 'FAA-' in part:
                mouse_id = part
            elif experiment_condition:
                experiment_condition.append(part)
        experiment_condition = '_'.join(experiment_condition)
        if not mouse_id or not experiment_condition:
            raise ValueError("Mouse ID or experiment condition not found in the file path.")

        save_dir = os.path.join(paths['filtereddata_folder'], 'LimbStuff', 'check_runstages',
                                f"{experiment_condition}\\{run_stage}\\_{mouse_id}")
        os.makedirs(save_dir, exist_ok=True)

        # Frame saving function
        def save_run_frame(r):
            cap = cv2.VideoCapture(video_file)
            try:
                # Ensure the run and run stage exist
                if run_stage not in self.data.loc[r].index.get_level_values('RunStage'):
                    cap.release()
                    return

                run_stage_idxs = self.data.loc[r].loc[run_stage].index.get_level_values('FrameIdx')
                if run_stage_idxs.empty:
                    cap.release()
                    return

                # Extract run stage data
                run_stage_data = self.data.loc(axis=0)[r, :, run_stage_idxs].droplevel(['Run', 'RunStage'])
            except KeyError:
                print(f"Run {r} missing data for stage '{run_stage}'. Skipping.")
                cap.release()
                return

            # Get the first frame index
            frame_idx = run_stage_idxs[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Could not read frame {frame_idx} from video for run {r}. Skipping.")
                cap.release()
                return

            # Add text to the frame
            text = f"Run {r}, RunStage: {run_stage}, Frame: {frame_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, (10, 30), font, 1, (255, 255, 255), 2)
            save_path = os.path.join(save_dir, f"Run_{r}_Stage_{run_stage}_Frame_{frame_idx}.png")
            cv2.imwrite(save_path, frame)
            cap.release()

        for r in runs:
            save_run_frame(r)

        print(f"Images saved in: {save_dir}")


class GetAllFiles:
    def __init__(self, directory=None, overwrite=False,
                 exp=None, speed=None, repeat_extend=None, exp_wash=None,
                 day=None, vmt_type=None, vmt_level=None, prep=None,
                 save_frames=True):
        self.directory = directory
        self.overwrite = overwrite

        # store experiment conditions for error logging
        self.exp = exp
        self.speed = speed
        self.repeat_extend = repeat_extend
        self.exp_wash = exp_wash
        self.day = day
        self.vmt_type = vmt_type
        self.vmt_level = vmt_level
        self.prep = prep

        self.save_frames = save_frames

        # path to a dedicated file-level error log
        self.file_error_log_file = os.path.join(paths['filtereddata_folder'], 'file_error_log.csv')

        # Ensure file_error_log.csv has a header if not already
        if not os.path.exists(self.file_error_log_file):
            with open(self.file_error_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'File',
                    'exp',
                    'speed',
                    'repeat_extend',
                    'exp_wash',
                    'day',
                    'vmt_type',
                    'vmt_level',
                    'prep',
                    'MouseID',
                    'ErrorMessage',
                    'Timestamp'
                ])

    def _log_file_error(self, file, error_message, mouseID=None):
        # If overwrite == True, remove existing rows for this file so they don't accumulate duplicates.
        if self.overwrite and os.path.exists(self.file_error_log_file):
            existing_df = pd.read_csv(self.file_error_log_file)
            existing_df = existing_df[existing_df['File'] != file]
            existing_df.to_csv(self.file_error_log_file, index=False)

        # Now read again (or create a new DataFrame if empty) and append the new row.
        if os.path.exists(self.file_error_log_file):
            df = pd.read_csv(self.file_error_log_file)
        else:
            df = pd.DataFrame(columns=[
                'File', 'exp', 'speed', 'repeat_extend', 'exp_wash', 'day',
                'vmt_type', 'vmt_level', 'prep', 'MouseID', 'ErrorMessage', 'Timestamp'
            ])

        new_row = {
            'File': file,
            'exp': self.exp,
            'speed': self.speed,
            'repeat_extend': self.repeat_extend,
            'exp_wash': self.exp_wash,
            'day': self.day,
            'vmt_type': self.vmt_type,
            'vmt_level': self.vmt_level,
            'prep': self.prep,
            'MouseID': mouseID if mouseID else '',
            'ErrorMessage': error_message,
            'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        df = df.append(new_row, ignore_index=True)

        # Sort rows for easier reading
        df = df.sort_values(by=['File'])
        df.to_csv(self.file_error_log_file, index=False)

    def remove_log_entries(self, file):
        # Define paths to the logs
        run_log_file = os.path.join(paths['filtereddata_folder'], 'run_summary_log.csv')
        error_log_file = os.path.join(paths['filtereddata_folder'], 'error_log.csv')

        # Remove entries from run log
        if os.path.exists(run_log_file):
            run_log_df = pd.read_csv(run_log_file)
            run_log_df = run_log_df[run_log_df['File'] != file]
            run_log_df.to_csv(run_log_file, index=False)

        # Remove entries from error log
        if os.path.exists(error_log_file):
            error_log_df = pd.read_csv(error_log_file)
            error_log_df = error_log_df[error_log_df['File'] != file]
            error_log_df.to_csv(error_log_file, index=False)

    def GetFiles(self):
        files = utils.Utils().GetListofMappedFiles(self.directory)  # gets dictionary of side, front and overhead 3D files

        for j in range(0, len(files)):
            match = re.search(r'FAA-(\d+)', files[j])
            mouseID = match.group(1)
            pattern = "*%s*_Runs.h5" % mouseID
            dir = os.path.dirname(files[j])
            date = files[j].split(os.sep)[-1].split('_')[1]

            if date in mfa.Files_to_be_dropped and mouseID in mfa.Files_to_be_dropped[date]:
                # check if file exists, if it does delete the file, and from the run log, if not continue
                self.remove_log_entries(files[j])
                if glob.glob(os.path.join(dir, pattern)):
                    os.remove(glob.glob(os.path.join(dir, pattern))[0])
                    print(f"File {mouseID} from {date} was deleted due to known issues.")
                else:
                    print(f"Skipping {mouseID} from {date} due to known issues.")
                continue

            if not glob.glob(os.path.join(dir, pattern)) or self.overwrite:
                print(f"###############################################################"
                      f"\nFinding runs and extracting gait for {mouseID}...\n###############################################################")
                get_runs = GetRuns(files[j], mouseID, date,
                               exp=self.exp,
                               speed=self.speed,
                               repeat_extend=self.repeat_extend,
                               exp_wash=self.exp_wash,
                               day=self.day,
                               vmt_type=self.vmt_type,
                               vmt_level=self.vmt_level,
                               prep=self.prep,
                               save_frames=self.save_frames)
                try:
                    get_runs.get_runs()
                except Exception as e:
                    print(f"Error processing file {files[j]}: {str(e)}")
                    # Log to the new file_error_log.csv
                    self._log_file_error(files[j], str(e), mouseID)
            else:
                print(f"Data for {mouseID} already exists. Skipping...")

        print('All experiments have been mapped to real-world coordinates and saved.')

class GetConditionFiles(BaseConditionFiles):
    def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None, day=None,
                 vmt_type=None, vmt_level=None, prep=None, overwrite=False, save_frames=True):
        super().__init__(
            exp=exp, speed=speed, repeat_extend=repeat_extend, exp_wash=exp_wash,
            day=day, vmt_type=vmt_type, vmt_level=vmt_level, prep=prep, overwrite=overwrite
        )
        self.save_frames = save_frames

    def process_final_directory(self, directory):
        # Instead of rewriting the same logic, just call GetAllFiles
        GetAllFiles(
            directory=directory,
            overwrite=self.overwrite,
            exp=self.exp,
            speed=self.speed,
            repeat_extend=self.repeat_extend,
            exp_wash=self.exp_wash,
            day=self.day,
            vmt_type=self.vmt_type,
            vmt_level=self.vmt_level,
            prep=self.prep,
            save_frames=self.save_frames
        ).GetFiles()


# class GetConditionFiles:
#     def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None, day=None, vmt_type=None,
#                  vmt_level=None, prep=None, overwrite=False, save_frames=True):
#         self.exp, self.speed, self.repeat_extend, self.exp_wash, self.day, self.vmt_type, self.vmt_level, self.prep, self.overwrite, self.save_frames = (
#             exp, speed, repeat_extend, exp_wash, day, vmt_type, vmt_level, prep, overwrite, save_frames)
#
#     def get_dirs(self):
#         if self.speed:
#             exp_speed_name = f"{self.exp}_{self.speed}"
#         else:
#             exp_speed_name = self.exp
#         base_path = os.path.join(paths['filtereddata_folder'], exp_speed_name)
#
#         # join any of the conditions that are not None in the order they appear in the function as individual directories
#         conditions = [self.repeat_extend, self.exp_wash, self.day, self.vmt_type, self.vmt_level, self.prep]
#         conditions = [c for c in conditions if c is not None]
#
#         # if Repeats in conditions, add 'Wash' directory in the next position in the list
#         if 'Repeats' in conditions:
#             idx = conditions.index('Repeats')
#             conditions.insert(idx + 1, 'Wash')
#         condition_path = os.path.join(base_path, *conditions)
#
#         if os.path.exists(condition_path):
#             print(f"Directory found: {condition_path}")
#         else:
#             raise FileNotFoundError(f"No path found {condition_path}")
#
#         # Recursively find and process the final data directories
#         self._process_subdirectories(condition_path)
#
#     def _process_subdirectories(self, current_path):
#         """
#         Recursively process directories and get to the final data directories.
#         """
#         subdirs = [d for d in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, d))]
#         subdirs = [sd for sd in subdirs if sd.lower() != 'bin']
#
#         # If subdirectories exist, traverse deeper
#         if len(subdirs) > 0:
#             print(f"Subdirectories found in {current_path}: {subdirs}")
#             for subdir in subdirs:
#                 full_subdir_path = os.path.join(current_path, subdir)
#                 # Recursively process subdirectory
#                 self._process_subdirectories(full_subdir_path)
#         else:
#             # No more subdirectories, assume this is the final directory with data
#             print(f"Final directory: {current_path}")
#             try:
#                 GetAllFiles(
#                     directory=current_path,
#                     overwrite=self.overwrite,
#                     exp=self.exp,
#                     speed=self.speed,
#                     repeat_extend=self.repeat_extend,
#                     exp_wash=self.exp_wash,
#                     day=self.day,
#                     vmt_type=self.vmt_type,
#                     vmt_level=self.vmt_level,
#                     prep=self.prep,
#                     save_frames=self.save_frames
#                 ).GetFiles()
#             except Exception as e:
#                 print(f"Error processing directory {current_path}: {e}")


def main():
    # Get all data
    # GetALLRuns(directory=directory).GetFiles()
    ### maybe instantiate first to protect entry point of my script
    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp',overwrite=False, save_frames=True).get_dirs() # should do all 3 days

    GetConditionFiles(exp='APAChar', speed='LowHigh', repeat_extend='Extended', overwrite=False, save_frames=True).get_dirs()
    GetConditionFiles(exp='APAChar', speed='LowMid', repeat_extend='Extended', overwrite=False, save_frames=True).get_dirs()
    GetConditionFiles(exp='APAChar', speed='HighLow', repeat_extend='Extended', overwrite=False, save_frames=True).get_dirs()

if __name__ == "__main__":
    # directory = input("Enter the directory path: ")
    # main(directory)
    main()
