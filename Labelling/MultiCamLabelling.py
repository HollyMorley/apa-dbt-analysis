import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
import matplotlib
matplotlib.rcParams.update({'font.size': 8, 'font.family': 'Arial'})
from PIL import Image, ImageTk, ImageEnhance
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import time
from functools import lru_cache
from pycalib.calib import triangulate
import pickle

import Helpers.MultiCamLabelling_config as config
from Helpers.CalibrateCams import BasicCalibration

import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception as e:
    pass  # In case you're not on Windows or it fails for another reason.

def get_video_name_with_view(video_name, view):
    split_video_name = video_name.split('_')
    split_video_name.insert(-1, view)
    return '_'.join(split_video_name)

def debounce(wait):
    def decorator(fn):
        last_call = [0]

        def debounced(*args, **kwargs):
            current_time = time.time()
            if current_time - last_call[0] >= wait:
                last_call[0] = current_time
                return fn(*args, **kwargs)

        return debounced

    return decorator

class MainTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Parts Labeling Tool")

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self.main_menu()

    def main_menu(self):
        self.clear_root()
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20)

        extract_button = tk.Button(main_frame, text="Extract Frames from Videos", command=self.extract_frames_menu)
        extract_button.pack(pady=5)

        calibrate_button = tk.Button(main_frame, text="Calibrate Camera Positions", command=self.calibrate_cameras_menu)
        calibrate_button.pack(pady=5)

        label_button = tk.Button(main_frame, text="Label Frames", command=self.label_frames_menu)
        label_button.pack(pady=5)

        replace_calib_button = tk.Button(main_frame, text="Replace Calibration", command=self.replace_calibration)
        replace_calib_button.pack(pady=5)

    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def extract_frames_menu(self):
        self.clear_root()
        ExtractFramesTool(self.root, self)

    def calibrate_cameras_menu(self):
        self.clear_root()
        CalibrateCamerasTool(self.root, self)

    def label_frames_menu(self):
        self.clear_root()
        LabelFramesTool(self.root, self)

    def replace_calibration(self):
        self.clear_root()
        ReplaceCalibrationLabels(self.root, self)


class ExtractFramesTool:
    def __init__(self, root, main_tool):
        self.root = root
        self.main_tool = main_tool
        self.video_path = ""
        self.video_name = ""
        self.video_date = ""
        self.camera_view = ""
        self.cap_side = None
        self.cap_front = None
        self.cap_overhead = None
        self.total_frames = 0
        self.current_frame_index = 0
        self.contrast_var = tk.DoubleVar(value=config.DEFAULT_CONTRAST)
        self.brightness_var = tk.DoubleVar(value=config.DEFAULT_BRIGHTNESS)

        self.extract_frames()

    def extract_frames(self):
        self.main_tool.clear_root()

        self.video_path = filedialog.askopenfilename(title="Select Video File")
        if not self.video_path:
            self.main_tool.main_menu()
            return

        self.video_name, self.video_date, self.camera_view = self.parse_video_path(self.video_path)
        self.video_name_stripped = '_'.join(self.video_name.split('_')[:-1])  # Remove the camera view part
        self.cap_side = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0

        self.cap_front = cv2.VideoCapture(self.get_corresponding_video_path('front'))
        self.cap_overhead = cv2.VideoCapture(self.get_corresponding_video_path('overhead'))

        # Print the total number of frames for each video
        total_frames_side = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_front = int(self.cap_front.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_overhead = int(self.cap_overhead.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Total frames - Side: {total_frames_side}, Front: {total_frames_front}, Overhead: {total_frames_overhead}")

        # Load timestamps
        timestamps_side = self.zero_timestamps(self.load_timestamps('side'))
        timestamps_front = self.zero_timestamps(self.load_timestamps('front'))
        timestamps_overhead = self.zero_timestamps(self.load_timestamps('overhead'))

        # Adjust timestamps to offset the drift in front and overhead cameras (where frame rates are different)
        timestamps_front_adj = self.adjust_timestamps(timestamps_side, timestamps_front)
        timestamps_overhead_adj = self.adjust_timestamps(timestamps_side, timestamps_overhead)
        timestamps_side_adj = timestamps_side['Timestamp'].astype(float) # adjust so compatible with scaled front and overhead

        # Extract matching frames
        self.match_frames(timestamps_side_adj, timestamps_front_adj, timestamps_overhead_adj)

        self.show_frames_extraction()

    def load_timestamps(self, view):
        video_name = '_'.join(self.video_name.split('_')[:-1])  # Remove the camera view part
        video_number = self.video_name.split('_')[-1]
        timestamp_file = video_name + f"_{view}_{video_number}_Timestamps.csv"
        timestamp_path = os.path.join(os.path.dirname(self.video_path), timestamp_file)
        timestamps = pd.read_csv(timestamp_path)
        return timestamps

    def zero_timestamps(self, timestamps):
        timestamps['Timestamp'] = timestamps['Timestamp'] - timestamps['Timestamp'][0]
        return timestamps

    def adjust_timestamps(self, side_timestamps, other_timestamps):
        mask = other_timestamps['Timestamp'].diff() < 4.045e+6
        other_timestamps_single_frame = other_timestamps[mask]
        side_timestamps_single_frame = side_timestamps[mask]
        diff = other_timestamps_single_frame['Timestamp'] - side_timestamps_single_frame['Timestamp']

        # find the best fit line for the lower half of the data by straightning the line
        model = LinearRegression().fit(side_timestamps_single_frame['Timestamp'].values.reshape(-1, 1), diff.values)
        slope = model.coef_[0]
        intercept = model.intercept_
        straightened_diff = diff - (slope * side_timestamps_single_frame['Timestamp'] + intercept)
        correct_diff_idx = np.where(straightened_diff < straightened_diff.mean())

        model_true = LinearRegression().fit(side_timestamps_single_frame['Timestamp'].values[correct_diff_idx].reshape(-1, 1), diff.values[correct_diff_idx])
        slope_true = model_true.coef_[0]
        intercept_true = model_true.intercept_
        adjusted_timestamps = other_timestamps['Timestamp'] - (slope_true * other_timestamps['Timestamp'] + intercept_true)
        return adjusted_timestamps

    def match_frames(self, timestamps_side, timestamps_front, timestamps_overhead):
        buffer_ns = int(4.04e+6)  # Frame duration in nanoseconds

        # Ensure the timestamps are sorted
        timestamps_side = timestamps_side.sort_values().reset_index(drop=True)
        timestamps_front = timestamps_front.sort_values().reset_index(drop=True)
        timestamps_overhead = timestamps_overhead.sort_values().reset_index(drop=True)

        # Convert timestamps to DataFrame for merging
        side_df = pd.DataFrame({'Timestamp': timestamps_side, 'Frame_number_side': range(len(timestamps_side))})
        front_df = pd.DataFrame({'Timestamp': timestamps_front, 'Frame_number_front': range(len(timestamps_front))})
        overhead_df = pd.DataFrame(
            {'Timestamp': timestamps_overhead, 'Frame_number_overhead': range(len(timestamps_overhead))})

        # Perform asof merge to find the closest matching frames within the buffer
        matched_front = pd.merge_asof(side_df, front_df, on='Timestamp', direction='nearest', tolerance=buffer_ns,
                                      suffixes=('_side', '_front'))
        matched_all = pd.merge_asof(matched_front, overhead_df, on='Timestamp', direction='nearest',
                                    tolerance=buffer_ns, suffixes=('_side', '_overhead'))

        # Check column names
        print(matched_all.columns)

        # Handle NaNs explicitly by setting unmatched frames to -1
        matched_frames = matched_all[['Frame_number_side', 'Frame_number_front', 'Frame_number_overhead']].applymap(
            lambda x: int(x) if pd.notnull(x) else -1).values.tolist()

        self.matched_frames = matched_frames

    def show_frames_extraction(self):
        self.main_tool.clear_root()

        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, pady=10)

        self.slider = tk.Scale(control_frame, from_=0, to=len(self.matched_frames) - 1, orient=tk.HORIZONTAL,
                               length=600,
                               command=self.update_frame_label)
        self.slider.pack(side=tk.LEFT, padx=5)

        self.frame_label = tk.Label(control_frame, text=f"Frame: {self.matched_frames[0][0]}")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        skip_frame = tk.Frame(self.root)
        skip_frame.pack(side=tk.TOP, pady=10)
        self.add_skip_buttons(skip_frame)

        control_frame_right = tk.Frame(self.root)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

        extract_button = tk.Button(control_frame_right, text="Extract Frames", command=self.save_extracted_frames)
        extract_button.pack(pady=5)

        back_button = tk.Button(control_frame_right, text="Back to Main Menu", command=self.main_tool.main_menu)
        back_button.pack(pady=5)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.display_frame(0)

    def add_skip_buttons(self, parent):
        buttons = [
            ("<< 1000", -1000), ("<< 100", -100), ("<< 10", -10), ("<< 1", -1),
            (">> 1", 1), (">> 10", 10), (">> 100", 100), (">> 1000", 1000)
        ]
        for i, (text, step) in enumerate(buttons):
            button = tk.Button(parent, text=text, command=lambda s=step: self.skip_frames(s))
            button.grid(row=0, column=i, padx=5)

    def skip_frames(self, step):
        new_frame_number = self.current_frame_index + step
        new_frame_number = max(0, min(new_frame_number, len(self.matched_frames) - 1))
        self.slider.set(new_frame_number)
        self.display_frame(new_frame_number)

    def display_frame(self, index):
        self.current_frame_index = index
        frame_side, frame_front, frame_overhead = self.matched_frames[index]

        # Read frames from respective positions
        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_side)
        ret_side, frame_side_img = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_front)
        ret_front, frame_front_img = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_overhead)
        ret_overhead, frame_overhead_img = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            frame_side_img = self.apply_contrast_brightness(frame_side_img)
            frame_front_img = self.apply_contrast_brightness(frame_front_img)
            frame_overhead_img = self.apply_contrast_brightness(frame_overhead_img)

            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[2].cla()

            self.axs[0].imshow(cv2.cvtColor(frame_side_img, cv2.COLOR_BGR2RGB))
            self.axs[1].imshow(cv2.cvtColor(frame_front_img, cv2.COLOR_BGR2RGB))
            self.axs[2].imshow(cv2.cvtColor(frame_overhead_img, cv2.COLOR_BGR2RGB))

            self.axs[0].set_title('Side View')
            self.axs[1].set_title('Front View')
            self.axs[2].set_title('Overhead View')

            self.canvas.draw()

    def update_frame_label(self, val):
        index = int(val)
        self.frame_label.config(text=f"Frame: {self.matched_frames[index][0]}")
        self.display_frame(index)

    def apply_contrast_brightness(self, frame):
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        img_contrast = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(img_contrast)
        img_brightness = enhancer.enhance(brightness)
        return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)

    def parse_video_path(self, video_path):
        video_file = os.path.basename(video_path)
        parts = video_file.split('_')
        date = parts[1]
        camera_view = [part for part in parts if part in ['side', 'front', 'overhead']][0]
        name_parts = [part for part in parts if part not in ['side', 'front', 'overhead']]
        name = '_'.join(name_parts).replace('.avi', '')  # Remove .avi extension if present
        return name, date, camera_view

    def get_corresponding_video_path(self, view):
        base_path = os.path.dirname(self.video_path)
        video_file = os.path.basename(self.video_path)
        corresponding_file = video_file.replace(self.camera_view, view).replace('.avi', '')
        return os.path.join(base_path, f"{corresponding_file}.avi")

    def save_extracted_frames(self):
        frame_side, frame_front, frame_overhead = self.matched_frames[self.current_frame_index]

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_side)
        ret_side, frame_side_img = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_front)
        ret_front, frame_front_img = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_overhead)
        ret_overhead, frame_overhead_img = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            video_names = {view: get_video_name_with_view(self.video_name, view) for view in
                           ['side', 'front', 'overhead']}

            side_path = os.path.join(config.FRAME_SAVE_PATH_TEMPLATE["side"].format(video_name=video_names['side']),
                                     f"img{frame_side}.png")
            front_path = os.path.join(config.FRAME_SAVE_PATH_TEMPLATE["front"].format(video_name=video_names['front']),
                                      f"img{frame_front}.png")
            overhead_path = os.path.join(
                config.FRAME_SAVE_PATH_TEMPLATE["overhead"].format(video_name=video_names['overhead']),
                f"img{frame_overhead}.png")

            os.makedirs(os.path.dirname(side_path), exist_ok=True)
            os.makedirs(os.path.dirname(front_path), exist_ok=True)
            os.makedirs(os.path.dirname(overhead_path), exist_ok=True)

            cv2.imwrite(side_path, frame_side_img)
            cv2.imwrite(front_path, frame_front_img)
            cv2.imwrite(overhead_path, frame_overhead_img)
            messagebox.showinfo("Info", "Frames saved successfully")


class CalibrateCamerasTool:
    def __init__(self, root, main_tool):
        self.root = root
        self.main_tool = main_tool
        self.video_path = ""
        self.video_name = ""
        self.video_date = ""
        self.camera_view = ""
        self.cap_side = None
        self.cap_front = None
        self.cap_overhead = None
        self.total_frames = 0
        self.current_frame_index = 0
        self.contrast_var = tk.DoubleVar(value=config.DEFAULT_CONTRAST)
        self.brightness_var = tk.DoubleVar(value=config.DEFAULT_BRIGHTNESS)
        self.marker_size_var = tk.DoubleVar(value=config.DEFAULT_MARKER_SIZE)
        self.mode = 'calibration'
        self.matched_frames = []

        self.calibration_points_static = {}
        self.dragging_point = None
        self.crosshair_lines = []
        self.panning = False
        self.pan_start = None

        self.labels = config.CALIBRATION_LABELS
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])
        self.current_view = tk.StringVar(value="side")

        self.calibrate_cameras_menu()

    def calibrate_cameras_menu(self):
        self.main_tool.clear_root()

        self.video_path = filedialog.askopenfilename(title="Select Video File")
        if not self.video_path:
            self.main_tool.main_menu()
            return

        self.video_name, self.video_date, self.camera_view = self.parse_video_path(self.video_path)
        self.cap_side = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0

        self.cap_front = cv2.VideoCapture(self.get_corresponding_video_path('front'))
        self.cap_overhead = cv2.VideoCapture(self.get_corresponding_video_path('overhead'))

        # Load timestamps
        timestamps_side = self.zero_timestamps(self.load_timestamps('side'))
        timestamps_front = self.zero_timestamps(self.load_timestamps('front'))
        timestamps_overhead = self.zero_timestamps(self.load_timestamps('overhead'))

        # Adjust timestamps to offset the drift in front and overhead cameras (where frame rates are different)
        timestamps_front_adj = self.adjust_timestamps(timestamps_side, timestamps_front)
        timestamps_overhead_adj = self.adjust_timestamps(timestamps_side, timestamps_overhead)
        timestamps_side_adj = timestamps_side['Timestamp'].astype(
            float)  # adjust so compatible with scaled front and overhead

        # Extract matching frames
        self.match_frames(timestamps_side_adj, timestamps_front_adj, timestamps_overhead_adj)

        self.calibration_file_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=self.video_name)
        enhanced_calibration_file = self.calibration_file_path.replace('.csv', '_enhanced.csv')
        default_calibration_file = config.DEFAULT_CALIBRATION_FILE_PATH

        self.mode = 'calibration'

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        settings_frame = tk.Frame(control_frame)
        settings_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(settings_frame, text="Marker Size", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_MARKER_SIZE, to=config.MAX_MARKER_SIZE, orient=tk.HORIZONTAL,
                 resolution=config.MARKER_SIZE_STEP, variable=self.marker_size_var,
                 command=self.update_marker_size, length=200, font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Contrast", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_CONTRAST, to=config.MAX_CONTRAST, orient=tk.HORIZONTAL,
                 resolution=config.CONTRAST_STEP, variable=self.contrast_var,
                 command=self.update_contrast_brightness, length=200, font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Brightness", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_BRIGHTNESS, to=config.MAX_BRIGHTNESS, orient=tk.HORIZONTAL,
                 resolution=config.BRIGHTNESS_STEP, variable=self.brightness_var,
                 command=self.update_contrast_brightness, length=200, font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)

        frame_control = tk.Frame(control_frame)
        frame_control.pack(side=tk.LEFT, padx=20)

        self.frame_label = tk.Label(frame_control, text="Frame: 0")
        self.frame_label.pack()

        self.slider = tk.Scale(frame_control, from_=0, to=len(self.matched_frames) - 1, orient=tk.HORIZONTAL,
                               length=400,
                               command=self.update_frame_label)
        self.slider.pack()

        skip_frame = tk.Frame(frame_control)
        skip_frame.pack()
        self.add_skip_buttons(skip_frame)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=20)

        home_button = tk.Button(button_frame, text="Home", command=self.reset_view)
        home_button.pack(pady=5)

        save_button = tk.Button(button_frame, text="Save Calibration Points", command=self.save_calibration_points)
        save_button.pack(pady=5)

        back_button = tk.Button(button_frame, text="Back to Main Menu", command=self.main_tool.main_menu)
        back_button.pack(pady=5)

        control_frame_right = tk.Frame(main_frame)
        control_frame_right.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        for label in self.labels:
            color = self.label_colors[label]
            label_frame = tk.Frame(control_frame_right)
            label_frame.pack(fill=tk.X, pady=2)
            color_box = tk.Label(label_frame, bg=color, width=2)
            color_box.pack(side=tk.LEFT, padx=5)
            label_button = tk.Radiobutton(label_frame, text=label, variable=self.current_label, value=label,
                                          indicatoron=0, width=20)
            label_button.pack(side=tk.LEFT)

        for view in ["side", "front", "overhead"]:
            tk.Radiobutton(control_frame_right, text=view.capitalize(), variable=self.current_view, value=view).pack(
                pady=2)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("motion_notify_event", self.update_crosshair)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.calibration_points_static = {label: {"side": None, "front": None, "overhead": None} for label in
                                          self.labels}

        if os.path.exists(enhanced_calibration_file):
            self.load_calibration_points(enhanced_calibration_file)
        elif os.path.exists(self.calibration_file_path):
            response = messagebox.askyesnocancel("Calibration Found",
                                                 "Calibration labels found. Do you want to load them? (Yes to load current, No to load default, Cancel to skip)")
            if response is None:
                pass
            elif response:
                self.load_calibration_points(self.calibration_file_path)
            else:
                if os.path.exists(default_calibration_file):
                    self.load_calibration_points(default_calibration_file)
                else:
                    messagebox.showinfo("Default Calibration Not Found", "Default calibration file not found.")
        else:
            if os.path.exists(default_calibration_file):
                if messagebox.askyesno("Default Calibration",
                                       "No specific calibration file found. Do you want to load the default calibration labels?"):
                    self.load_calibration_points(default_calibration_file)
            else:
                messagebox.showinfo("Default Calibration Not Found", "Default calibration file not found.")

        self.show_frames()

    def match_frames(self, timestamps_side, timestamps_front, timestamps_overhead):
        buffer_ns = int(4.04e+6)  # Frame duration in nanoseconds

        # Ensure the timestamps are sorted
        timestamps_side = timestamps_side.sort_values().reset_index(drop=True)
        timestamps_front = timestamps_front.sort_values().reset_index(drop=True)
        timestamps_overhead = timestamps_overhead.sort_values().reset_index(drop=True)

        # Convert timestamps to DataFrame for merging
        side_df = pd.DataFrame({'Timestamp': timestamps_side, 'Frame_number_side': range(len(timestamps_side))})
        front_df = pd.DataFrame({'Timestamp': timestamps_front, 'Frame_number_front': range(len(timestamps_front))})
        overhead_df = pd.DataFrame(
            {'Timestamp': timestamps_overhead, 'Frame_number_overhead': range(len(timestamps_overhead))})

        # Perform asof merge to find the closest matching frames within the buffer
        matched_front = pd.merge_asof(side_df, front_df, on='Timestamp', direction='nearest', tolerance=buffer_ns,
                                      suffixes=('_side', '_front'))
        matched_all = pd.merge_asof(matched_front, overhead_df, on='Timestamp', direction='nearest',
                                    tolerance=buffer_ns, suffixes=('_side', '_overhead'))

        # Check column names
        print(matched_all.columns)

        # Handle NaNs explicitly by setting unmatched frames to -1
        matched_frames = matched_all[['Frame_number_side', 'Frame_number_front', 'Frame_number_overhead']].applymap(
            lambda x: int(x) if pd.notnull(x) else -1).values.tolist()

        self.matched_frames = matched_frames
        print(f"Matched frames: {len(matched_frames)}")

    def load_timestamps(self, view):
        video_name = '_'.join(self.video_name.split('_')[:-1])  # Remove the camera view part
        video_number = self.video_name.split('_')[-1]
        timestamp_file = video_name + f"_{view}_{video_number}_Timestamps.csv"
        timestamp_path = os.path.join(os.path.dirname(self.video_path), timestamp_file)
        timestamps = pd.read_csv(timestamp_path)
        return timestamps

    def zero_timestamps(self, timestamps):
        timestamps['Timestamp'] = timestamps['Timestamp'] - timestamps['Timestamp'][0]
        return timestamps

    def adjust_timestamps(self, side_timestamps, other_timestamps):
        mask = other_timestamps['Timestamp'].diff() < 4.045e+6
        other_timestamps_single_frame = other_timestamps[mask]
        side_timestamps_single_frame = side_timestamps[mask]
        diff = other_timestamps_single_frame['Timestamp'] - side_timestamps_single_frame['Timestamp']

        # find the best fit line for the lower half of the data by straightening the line
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
        adjusted_timestamps = other_timestamps['Timestamp'] - (
                    slope_true * other_timestamps['Timestamp'] + intercept_true)
        return adjusted_timestamps

    def update_marker_size(self, val):
        current_xlim = [ax.get_xlim() for ax in self.axs]
        current_ylim = [ax.get_ylim() for ax in self.axs]
        self.marker_size = self.marker_size_var.get()
        self.show_frames()
        for ax, xlim, ylim in zip(self.axs, current_xlim, current_ylim):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def load_calibration_points(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.set_index(["bodyparts", "coords"], inplace=True)
            for label in df.index.levels[0]:
                for view in ["side", "front", "overhead"]:
                    if not pd.isna(df.loc[(label, 'x'), view]):
                        x, y = df.loc[(label, 'x'), view], df.loc[(label, 'y'), view]
                        self.calibration_points_static[label][view] = self.axs[
                            ["side", "front", "overhead"].index(view)].scatter(
                            x, y, c=self.label_colors[label], s=self.marker_size_var.get() * 10, label=label
                        )
            self.canvas.draw()
            messagebox.showinfo("Info", "Calibration points loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration points: {e}")

    def update_frame_label(self, val):
        self.current_frame_index = int(val)
        self.show_frames()

    def add_skip_buttons(self, parent):
        buttons = [
            ("<< 1000", -1000), ("<< 100", -100), ("<< 10", -10), ("<< 1", -1),
            (">> 1", 1), (">> 10", 10), (">> 100", 100), (">> 1000", 1000)
        ]
        for i, (text, step) in enumerate(buttons):
            button = tk.Button(parent, text=text, command=lambda s=step: self.skip_frames(s))
            button.grid(row=0, column=i, padx=5)

    def skip_frames(self, step):
        new_frame_number = self.current_frame_index + step
        new_frame_number = max(0, min(new_frame_number, self.total_frames - 1))
        self.current_frame_index = new_frame_number
        self.slider.set(new_frame_number)
        self.frame_label.config(text=f"Frame: {new_frame_number}/{self.total_frames - 1}")
        self.show_frames()

    def parse_video_path(self, video_path):
        video_file = os.path.basename(video_path)
        parts = video_file.split('_')
        date = parts[1]
        camera_view = [part for part in parts if part in ['side', 'front', 'overhead']][0]
        name_parts = [part for part in parts if part not in ['side', 'front', 'overhead']]
        name = '_'.join(name_parts).replace('.avi', '')  # Remove .avi extension if present
        return name, date, camera_view

    def get_corresponding_video_path(self, view):
        base_path = os.path.dirname(self.video_path)
        video_file = os.path.basename(self.video_path)
        corresponding_file = video_file.replace(self.camera_view, view).replace('.avi', '')
        return os.path.join(base_path, f"{corresponding_file}.avi")

    def save_calibration_points(self):
        calibration_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=self.video_name)
        os.makedirs(os.path.dirname(calibration_path), exist_ok=True)

        data = {"bodyparts": [], "coords": [], "side": [], "front": [], "overhead": []}
        for label, coords in self.calibration_points_static.items():
            for coord in ['x', 'y']:
                data["bodyparts"].append(label)
                data["coords"].append(coord)
                for view in ["side", "front", "overhead"]:
                    if coords[view] is not None:
                        x, y = coords[view].get_offsets()[0]
                        if coord == 'x':
                            data[view].append(x)
                        else:
                            data[view].append(y)
                    else:
                        data[view].append(None)

        df = pd.DataFrame(data)
        df.to_csv(calibration_path, index=False)

        messagebox.showinfo("Info", "Calibration points saved successfully")

    def generate_label_colors(self, labels):
        colormap = plt.get_cmap('hsv')
        colors = [colormap(i / len(labels)) for i in range(len(labels))]
        return {label: self.rgb_to_hex(color) for label, color in zip(labels, colors)}

    def rgb_to_hex(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    def update_contrast_brightness(self, val):
        self.show_frames()

    def apply_contrast_brightness(self, frame):
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        img_contrast = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(img_contrast)
        img_brightness = enhancer.enhance(brightness)
        return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)

    def on_scroll(self, event):
        if event.inaxes:
            ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            xdata = event.xdata
            ydata = event.ydata

            if xdata is not None and ydata is not None:
                zoom_factor = 0.9 if event.button == 'up' else 1.1

                new_xlim = [xdata + (x - xdata) * zoom_factor for x in xlim]
                new_ylim = [ydata + (y - ydata) * zoom_factor for y in ylim]

                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)

                self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.button == 2:
            self.panning = True
            self.pan_start = (event.x, event.y)

    def on_mouse_release(self, event):
        if event.button == 2:
            self.panning = False
            self.pan_start = None

    def on_mouse_move(self, event):
        if self.panning and self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_start = (event.x, event.y)

            ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            scale_x = (xlim[1] - xlim[0]) / self.canvas.get_width_height()[0]
            scale_y = (ylim[1] - ylim[0]) / self.canvas.get_width_height()[1]

            ax.set_xlim(xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
            ax.set_ylim(ylim[0] - dy * scale_y, ylim[1] - dy * scale_y)
            self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[["side", "front", "overhead"].index(view)]
        color = self.label_colors[self.current_label.get()]
        marker_size = self.marker_size_var.get()

        if event.button == MouseButton.RIGHT:
            if event.key == 'shift':
                self.delete_closest_point(ax, event)
            else:
                label = self.current_label.get()
                if self.calibration_points_static[label][view] is not None:
                    self.calibration_points_static[label][view].remove()
                self.calibration_points_static[label][view] = ax.scatter(event.xdata, event.ydata, c=color, s=marker_size * 10, label=label)
                self.canvas.draw()
        elif event.button == MouseButton.LEFT:
            self.dragging_point = self.find_closest_point(ax, event)

    def find_closest_point(self, ax, event):
        min_dist = float('inf')
        closest_point = None
        for label, points in self.calibration_points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = points[self.current_view.get()]
        return closest_point if min_dist < 10 else None

    def delete_closest_point(self, ax, event):
        min_dist = float('inf')
        closest_point_label = None
        for label, points in self.calibration_points_static.items():
            if points[self.current_view.get()] is not None:
                x, y = points[self.current_view.get()].get_offsets()[0]
                dist = np.hypot(x - event.xdata, y - event.ydata)
                if dist < min_dist:
                    min_dist = dist
                    closest_point_label = label

        if closest_point_label:
            self.calibration_points_static[closest_point_label][self.current_view.get()].remove()
            self.calibration_points_static[closest_point_label][self.current_view.get()] = None
            self.canvas.draw()

    def on_drag(self, event):
        if self.dragging_point is None or event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[["side", "front", "overhead"].index(view)]

        if event.button == MouseButton.LEFT:
            self.dragging_point.set_offsets((event.xdata, event.ydata))
            self.canvas.draw()

    def update_crosshair(self, event):
        for line in self.crosshair_lines:
            line.remove()
        self.crosshair_lines = []

        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.crosshair_lines.append(event.inaxes.axhline(y, color='cyan', linestyle='--', linewidth=0.5))
            self.crosshair_lines.append(event.inaxes.axvline(x, color='cyan', linestyle='--', linewidth=0.5))
            self.canvas.draw_idle()

    def reset_view(self):
        for ax in self.axs:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        self.contrast_var.set(config.DEFAULT_CONTRAST)
        self.brightness_var.set(config.DEFAULT_BRIGHTNESS)
        self.show_frames()

    def show_frames(self, val=None):
        frame_number = self.current_frame_index
        self.frame_label.config(text=f"Frame: {frame_number}/{len(self.matched_frames) - 1}")

        frame_side, frame_front, frame_overhead = self.matched_frames[frame_number]

        self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, frame_side)
        ret_side, frame_side_img = self.cap_side.read()

        self.cap_front.set(cv2.CAP_PROP_POS_FRAMES, frame_front)
        ret_front, frame_front_img = self.cap_front.read()

        self.cap_overhead.set(cv2.CAP_PROP_POS_FRAMES, frame_overhead)
        ret_overhead, frame_overhead_img = self.cap_overhead.read()

        if ret_side and ret_front and ret_overhead:
            frame_side_img = self.apply_contrast_brightness(frame_side_img)
            frame_front_img = self.apply_contrast_brightness(frame_front_img)
            frame_overhead_img = self.apply_contrast_brightness(frame_overhead_img)

            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[2].cla()

            self.axs[0].imshow(cv2.cvtColor(frame_side_img, cv2.COLOR_BGR2RGB))
            self.axs[1].imshow(cv2.cvtColor(frame_front_img, cv2.COLOR_BGR2RGB))
            self.axs[2].imshow(cv2.cvtColor(frame_overhead_img, cv2.COLOR_BGR2RGB))

            self.axs[0].set_title('Side View')
            self.axs[1].set_title('Front View')
            self.axs[2].set_title('Overhead View')

            self.show_static_points()
            self.canvas.draw_idle()

    def show_static_points(self):
        for label, points in self.calibration_points_static.items():
            for view, point in points.items():
                if point is not None:
                    ax = self.axs[["side", "front", "overhead"].index(view)]
                    ax.add_collection(point)
                    point.set_sizes([self.marker_size_var.get() * 10])
        self.canvas.draw()


class LabelFramesTool:
    def __init__(self, root, main_tool):
        self.root = root
        self.main_tool = main_tool
        self.video_name = ""
        self.video_date = ""
        self.extracted_frames_path = {}
        self.current_frame_index = 0
        self.contrast_var = tk.DoubleVar(value=config.DEFAULT_CONTRAST)
        self.brightness_var = tk.DoubleVar(value=config.DEFAULT_BRIGHTNESS)
        self.marker_size_var = tk.DoubleVar(value=config.DEFAULT_MARKER_SIZE)
        self.mode = 'labeling'
        self.calibration_data = None
        self.fig = None
        self.axs = None
        self.canvas = None
        self.labels = config.BODY_PART_LABELS
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value='Nose')
        self.current_view = tk.StringVar(value="side")
        self.projection_view = tk.StringVar(value="side")  # view to project points from
        self.body_part_points = {}
        self.calibration_points_static = {label: {"side": None, "front": None, "overhead": None} for label in config.CALIBRATION_LABELS}
        self.cam_reprojected_points = {'near': {}, 'far': {}}
        self.frames = {'side': [], 'front': [], 'overhead': []}
        self.projection_lines = {'side': None, 'front': None, 'overhead': None}
        self.P = None
        self.tooltip = None
        self.label_buttons = []
        self.tooltip_window = None
        self.matched_frames = []  # Add this to ensure matched_frames is initialized
        self.frame_names = {'side': [], 'front': [], 'overhead': []}
        self.frame_numbers = {'side': [], 'front': [], 'overhead': []}

        self.crosshair_lines = []
        self.dragging_point = None
        self.panning = False
        self.pan_start = None

        self.spacer_lines_active = False
        self.spacer_lines_points = []
        self.spacer_lines = []

        # Add bindings for key press and release
        self.root.bind_all('<KeyPress-Tab>', self.on_tab_press)
        self.root.bind_all('<KeyRelease-Tab>', self.on_tab_release)

        # Initialize flag for Tab key state
        self.tab_pressed = False

        self.last_update_time = 0

        self.label_frames_menu()

    def label_frames_menu(self):
        self.main_tool.clear_root()

        calibration_folder_path = filedialog.askdirectory(title="Select Calibration Folder")

        if not calibration_folder_path:
            self.main_tool.main_menu()
            return

        self.video_name = os.path.basename(calibration_folder_path)
        self.video_date = self.extract_date_from_folder_path(calibration_folder_path)
        self.calibration_file_path = os.path.join(calibration_folder_path, "calibration_labels.csv")
        enhanced_calibration_file = self.calibration_file_path.replace('.csv', '_enhanced.csv')

        if not os.path.exists(self.calibration_file_path) and not os.path.exists(enhanced_calibration_file):
            messagebox.showerror("Error", "No corresponding camera calibration data found.")
            return

        base_path = os.path.dirname(os.path.dirname(calibration_folder_path))

        video_names = {view: get_video_name_with_view(self.video_name, view) for view in ['side', 'front', 'overhead']}

        self.extracted_frames_path = {
            'side': os.path.normpath(os.path.join(base_path, "Side", video_names['side'])),
            'front': os.path.normpath(os.path.join(base_path, "Front", video_names['front'])),
            'overhead': os.path.normpath(os.path.join(base_path, "Overhead", video_names['overhead']))
        }

        if not all(os.path.exists(path) for path in self.extracted_frames_path.values()):
            missing_paths = [path for path in self.extracted_frames_path.values() if not os.path.exists(path)]
            messagebox.showerror("Error", "One or more corresponding extracted frames folders not found.")
            return

        self.current_frame_index = 0

        self.show_loading_popup()

        self.frames = {'side': [], 'front': [], 'overhead': []}
        self.root.after(100, self.load_frames, enhanced_calibration_file)

    def show_loading_popup(self):
        self.loading_popup = tk.Toplevel(self.root)
        self.loading_popup.geometry("300x100")
        self.loading_popup.title("Loading")
        label = tk.Label(self.loading_popup, text="Loading frames, please wait...")
        label.pack(pady=20, padx=20)
        self.root.update_idletasks()

    def extract_date_from_folder_path(self, folder_path):
        parts = folder_path.split(os.sep)
        for part in parts:
            if part.isdigit() and len(part) == 8:
                return part
        return None

    def load_frames(self, enhanced_calibration_file):
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        for view in self.frames.keys():
            frame_files = sorted(
                (f for f in os.listdir(self.extracted_frames_path[view]) if
                 os.path.splitext(f)[1].lower() in valid_extensions),
                key=lambda x: os.path.getctime(os.path.join(self.extracted_frames_path[view], x))
            )
            for file in frame_files:
                frame = cv2.imread(os.path.join(self.extracted_frames_path[view], file))
                self.frames[view].append(frame)
                self.frame_names[view].append(file)
                image_number = int(os.path.splitext(file)[0].replace('img', ''))
                self.frame_numbers[view].append(image_number)

        min_frame_count = min(len(self.frames[view]) for view in self.frames if self.frames[view])
        for view in self.frames:
            self.frames[view] = self.frames[view][:min_frame_count]
            self.frame_numbers[view] = self.frame_numbers[view][:min_frame_count]

        print(f"Number of frames loaded for side: {len(self.frames['side'])}")
        print(f"Number of frames loaded for front: {len(self.frames['front'])}")
        print(f"Number of frames loaded for overhead: {len(self.frames['overhead'])}")

        if min_frame_count == 0:
            messagebox.showerror("Error", "No frames found in the directories.")
            self.loading_popup.destroy()
            return

        self.loading_popup.destroy()

        self.body_part_points = {
            frame_idx: {label: {"side": None, "front": None, "overhead": None} for label in self.labels}
            for frame_idx in range(min_frame_count)
        }

        # Initialize optimization checkboxes
        self.optimization_checkboxes = [tk.BooleanVar(value=False) for _ in range(min_frame_count)]

        self.match_frames()  # Call match_frames here

        self.setup_labeling_ui()

        video_names = {view: os.path.basename(self.extracted_frames_path[view]) for view in
                       ['side', 'front', 'overhead']}

        for view in ['side', 'front', 'overhead']:
            label_file_path = os.path.join(config.LABEL_SAVE_PATH_TEMPLATE[view].format(video_name=video_names[view]),
                                           "CollectedData_Holly_init.csv")
            if os.path.exists(label_file_path):
                self.load_existing_labels(label_file_path, view)

        if os.path.exists(enhanced_calibration_file):
            self.load_calibration_data(enhanced_calibration_file)
        else:
            self.load_calibration_data(self.calibration_file_path)

        self.display_frame()

    def match_frames(self):
        min_frame_count = min(len(self.frames['side']), len(self.frames['front']), len(self.frames['overhead']))
        self.matched_frames = [(i, i, i) for i in range(min_frame_count)]
        print(f"Matched frames: {len(self.matched_frames)}")

    def setup_labeling_ui(self):
        self.main_tool.clear_root()

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        settings_frame = tk.Frame(control_frame)
        settings_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(settings_frame, text="Marker Size", font=("Helvetica", 8)).pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_MARKER_SIZE, to=config.MAX_MARKER_SIZE, orient=tk.HORIZONTAL,
                 resolution=config.MARKER_SIZE_STEP, variable=self.marker_size_var,
                 command=self.update_marker_size, length=250, font=("Helvetica", 8)).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Contrast", font=("Helvetica", 8)).pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_CONTRAST, to=config.MAX_CONTRAST, orient=tk.HORIZONTAL,
                 resolution=config.CONTRAST_STEP, variable=self.contrast_var,
                 command=self.update_contrast_brightness, length=250, font=("Helvetica", 8)).pack(side=tk.LEFT, padx=5)

        tk.Label(settings_frame, text="Brightness", font=("Helvetica", 8)).pack(side=tk.LEFT, padx=5)
        tk.Scale(settings_frame, from_=config.MIN_BRIGHTNESS, to=config.MAX_BRIGHTNESS, orient=tk.HORIZONTAL,
                 resolution=config.BRIGHTNESS_STEP, variable=self.brightness_var,
                 command=self.update_contrast_brightness, length=250, font=("Helvetica", 8)).pack(side=tk.LEFT, padx=5)

        frame_control = tk.Frame(control_frame)
        frame_control.pack(side=tk.LEFT, padx=20)

        self.frame_label = tk.Label(frame_control,
                                    text=f"Frame: {self.current_frame_index + 1}/{len(self.frames['side'])}")
        self.frame_label.pack()

        self.prev_button = tk.Button(frame_control, text="<<", command=lambda: self.skip_labeling_frames(-1))
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(frame_control, text=">>", command=lambda: self.skip_labeling_frames(1))
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Add optimization checkbox
        self.optimization_checkbox = tk.Checkbutton(
            frame_control,
            text="Optimization",
            variable=self.optimization_checkboxes[self.current_frame_index],
            onvalue=True,
            offvalue=False
        )
        self.optimization_checkbox.pack(side=tk.LEFT, padx=5)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=20)

        home_button = tk.Button(button_frame, text="Home", command=self.reset_view)
        home_button.pack(pady=5)

        save_button = tk.Button(button_frame, text="Save Labels", command=self.save_labels)
        save_button.pack(pady=5)

        spacer_lines_button = tk.Button(button_frame, text="Spacer Lines", command=self.toggle_spacer_lines)
        spacer_lines_button.pack(pady=5)

        optimize_button = tk.Button(button_frame, text="Optimize Calibration", command=self.optimize_calibration)
        optimize_button.pack(pady=5)

        view_frame = tk.Frame(control_frame)
        view_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        tk.Label(view_frame, text="Label View").pack()
        self.current_view = tk.StringVar(value="side")
        for view in ["side", "front", "overhead"]:
            tk.Radiobutton(view_frame, text=view.capitalize(), variable=self.current_view, value=view).pack(side=tk.TOP,
                                                                                                            pady=2)

        projection_frame = tk.Frame(control_frame)
        projection_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        tk.Label(projection_frame, text="Projection View").pack()
        self.projection_view = tk.StringVar(value="side")
        for view in ["side", "front", "overhead"]:
            tk.Radiobutton(projection_frame, text=view.capitalize(), variable=self.projection_view, value=view).pack(
                side=tk.TOP, pady=2)

        control_frame_right = tk.Frame(control_frame)
        control_frame_right.pack(side=tk.RIGHT, padx=20)

        exit_button = tk.Button(control_frame_right, text="Exit", command=self.confirm_exit)
        exit_button.pack(pady=5)

        back_button = tk.Button(control_frame_right, text="Back to Main Menu", command=self.main_tool.main_menu)
        back_button.pack(pady=5)

        control_frame_labels = tk.Frame(main_frame)
        control_frame_labels.pack(side=tk.RIGHT, fill=tk.Y, padx=3, pady=1)  # Reduce padding to minimize space

        self.labels = config.BODY_PART_LABELS  # + ['Door']  # Add 'Door' label here
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])

        # Modify the label frame and scrollbar creation
        # Modify the label frame and scrollbar creation
        self.label_canvas = tk.Canvas(control_frame_labels, width=220)  # Set a fixed width for the label canvas
        self.label_canvas.pack(side=tk.LEFT, fill=tk.Y,
                               expand=False)  # Set to expand in Y direction only for vertical scrolling
        self.label_scrollbar = tk.Scrollbar(control_frame_labels, orient=tk.VERTICAL, command=self.label_canvas.yview)
        self.label_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.label_canvas.configure(yscrollcommand=self.label_scrollbar.set)

        self.label_frame = tk.Frame(self.label_canvas, width=220)
        self.label_canvas.create_window((0, 0), window=self.label_frame, anchor="nw")
        self.label_frame.bind("<Configure>",
                              lambda e: self.label_canvas.configure(scrollregion=self.label_canvas.bbox("all")))

        # Adjust label button settings
        for label in self.labels:
            if label != 'Door' and label in config.CALIBRATION_LABELS:
                continue  # Skip adding button for static calibration labels except 'Door'
            color = self.label_colors[label]
            label_button = tk.Radiobutton(self.label_frame, text=label, variable=self.current_label, value=label,
                                          indicatoron=0, width=15, bg=color, font=("Helvetica", 8),
                                          command=lambda l=label: self.on_label_select(l))
            label_button.pack(fill=tk.X, pady=1)
            self.label_buttons.append(label_button)

            # Bind the left mouse click event to show the popup with debug print
            label_button.bind("<Button-1>", lambda event, l=label: self.show_label_popup(l, event))

        # Ensure "Nose" is selected by default
        self.current_label.set("Nose")
        self.update_label_button_selection()

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 10))  # Adjust figure size for better fit
        self.fig.subplots_adjust(left=0.02, right=0.999, top=0.99, bottom=0.01, wspace=0.01, hspace=0.005)

        for ax in self.axs:
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            ax.yaxis.set_major_locator(plt.MultipleLocator(50))
            # ax.grid(visible=True, linestyle='--', linewidth=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tooltip = self.fig.text(0, 0, "", va="bottom", ha="left", fontsize=8,
                                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1), zorder=10)
        self.tooltip.set_visible(False)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("motion_notify_event", self.update_crosshair)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        #self.canvas.mpl_connect("motion_notify_event", self.show_tooltip)

        self.display_frame()

    def update_label_button_selection(self):
        for button in self.label_buttons:
            if button.cget('text') == self.current_label.get():
                button.select()
            else:
                button.deselect()

    def load_existing_labels(self, label_file_path, view):
        # Replace filepath with h5 file
        label_file_path = label_file_path.replace('.csv', '.h5')
        df = pd.read_hdf(label_file_path, key='df')

        for frame_idx_pos, frame_idx in enumerate(df.index):
            for label in self.labels:
                # Check if the required key exists in the DataFrame before accessing it
                if ('Holly', label, 'x') in df.columns and ('Holly', label, 'y') in df.columns:
                    x, y = df.loc[frame_idx, ('Holly', label, 'x')], df.loc[frame_idx, ('Holly', label, 'y')]
                    if not np.isnan(x) and not np.isnan(y):
                        self.body_part_points[frame_idx_pos][label][view] = (x, y)
                else:
                    print(f"Label '{label}' not found in the DataFrame for frame {frame_idx}.")

    # def show_tooltip(self, event):
    #     if event.inaxes in self.axs:
    #         marker_size = self.marker_size_var.get() * 10  # Assuming marker size is scaled
    #         for label, views in self.body_part_points[self.current_frame_index].items():
    #             for view, coords in views.items():
    #                 if view == self.current_view.get() and coords is not None:
    #                     x, y = coords
    #                     if np.hypot(x - event.xdata, y - event.ydata) < marker_size:
    #                         widget = self.canvas.get_tk_widget()
    #                         self.show_custom_tooltip(widget, label)
    #                         return
    #     self.hide_custom_tooltip()
    #
    # def show_custom_tooltip(self, wdgt, text):
    #     if self.tooltip_window:
    #         self.tooltip_window.destroy()
    #     self.tooltip_window = tk.Toplevel(wdgt)
    #     self.tooltip_window.overrideredirect(True)
    #
    #     tk.Label(self.tooltip_window, text=text, background='yellow').pack()
    #     self.tooltip_window.update_idletasks()
    #
    #     x_center = wdgt.winfo_pointerx() + 20
    #     y_center = wdgt.winfo_pointery() + 20
    #     self.tooltip_window.geometry(f"+{x_center}+{y_center}")
    #
    #     wdgt.bind('<Leave>', self.hide_custom_tooltip)
    #
    # def hide_custom_tooltip(self, event=None):
    #     if self.tooltip_window:
    #         self.tooltip_window.destroy()
    #         self.tooltip_window = None

    def show_label_popup(self, label, event):
        # Get the Matplotlib canvas widget
        widget = self.canvas.get_tk_widget()

        # Calculate the position of the Matplotlib canvas on the screen
        widget_x = widget.winfo_rootx()
        widget_y = widget.winfo_rooty()

        # Determine the subplot (Axes) where the click occurred
        ax = event.inaxes
        if ax is None:
            return

        # Convert the data coordinates (event.xdata, event.ydata) to display coordinates (pixels)
        display_coords = ax.transData.transform((event.xdata, event.ydata))

        # Invert the y-coordinate for the Tkinter coordinate system
        # Calculate the height of the canvas
        canvas_height = widget.winfo_height()
        display_x, display_y = display_coords

        # Adjust the y position by subtracting it from the canvas height
        display_y = canvas_height - display_y

        # Calculate popup coordinates relative to the root window
        popup_x = int(widget_x + display_x)
        popup_y = int(widget_y + display_y)

        # Create a small popup window
        popup = tk.Toplevel(self.root)
        popup.wm_overrideredirect(True)  # Remove window decorations
        popup.attributes('-topmost', True)  # Keep popup on top
        popup.lift()  # Ensure it appears above all other windows

        # Position the popup near the scatter point
        popup.geometry(f"+{popup_x + 10}+{popup_y + 10}")

        # Add label text to the popup
        tk.Label(popup, text=label, background='yellow', font=("Helvetica", 12)).pack()

        # Close the popup after 2 seconds
        self.root.after(2000, popup.destroy)

    def get_p(self, view, extrinsics=None, return_value=False):
        if extrinsics is None:
            extrinsics = self.calibration_data['extrinsics']
        if self.calibration_data:
            # Camera intrinsics
            K = self.calibration_data['intrinsics'][view]

            # Camera extrinsics
            R = extrinsics[view]['rotm']
            t = extrinsics[view]['tvec']

            # Ensure t is a column vector
            if t.ndim == 1:
                t = t[:, np.newaxis]

            # Form the projection matrix
            P = np.dot(K, np.hstack((R, t)))

            if return_value:
                return P
            else:
                self.P = P

    def get_camera_center(self, view):
        if self.calibration_data:
            # Extract the rotation matrix and translation vector
            R = self.calibration_data['extrinsics'][view]['rotm']
            t = self.calibration_data['extrinsics'][view]['tvec']

            # Compute the camera center in world coordinates
            camera_center = -np.dot(np.linalg.inv(R), t)

            return camera_center.flatten()  # Flatten to make it a 1D array

    def find_projection(self, view, bp):
        self.get_p(view)  # Ensure projection matrix is updated
        # Find 3D point with self.P and the current 2D point
        if self.body_part_points[self.current_frame_index][bp][view] is not None:
            x, y = self.body_part_points[self.current_frame_index][bp][view]

            if x is not None and y is not None:
                # Create the homogeneous coordinates for the 2D point
                uv = np.array([x, y, 1.0])

                # Compute the pseudo-inverse of the projection matrix
                P_inv = np.linalg.pinv(self.P)

                # Find the 3D point in homogeneous coordinates
                X = np.dot(P_inv, uv)

                # Normalize to get the 3D point
                X /= X[-1]

                return X[:3]  # Return only the x, y, z components
        return None

    def get_line_equation(self, point_3d, camera_center):
        # Extract the coordinates of the points
        x1, y1, z1 = point_3d
        x2, y2, z2 = camera_center

        def line_at_t(t):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            z = z1 + t * (z2 - z1)
            return x, y, z

        return line_at_t

    def find_t_for_coordinate(self, val, coord_index, point_3d, camera_center):
        x1, y1, z1 = point_3d
        x2, y2, z2 = camera_center

        if coord_index == 0:  # x-coordinate
            t = (val - x1) / (x2 - x1)
        elif coord_index == 1:  # y-coordinate
            t = (val - y1) / (y2 - y1)
        elif coord_index == 2:  # z-coordinate
            t = (val - z1) / (z2 - z1)
        else:
            raise ValueError("coord_index must be 0 (x), 1 (y), or 2 (z)")

        return t

    def find_3d_edges(self, view, bp):
        # Get the 3D coordinates of the body part
        point_3d = self.find_projection(view, bp)

        # Get the camera center
        camera_center = self.get_camera_center(view)

        if point_3d is not None and camera_center is not None:
            # Get the line equation
            line_at_t = self.get_line_equation(point_3d, camera_center)

            # Determine the appropriate dimension based on the view
            if view == "side":
                coord_index = 1  # y-coordinate for side view
            elif view == "front":
                coord_index = 0  # x-coordinate for front view
            elif view == "overhead":
                coord_index = 2  # z-coordinate for overhead view

            # Get the 3D coordinates of the near edge
            near_edge = line_at_t(self.find_t_for_coordinate(0, coord_index, point_3d, camera_center))

            # Get the 3D coordinates of the far edge
            far_edge_value = self.calibration_data['belt points WCS'].T[coord_index].max()
            if view == 'front':
                far_edge_value += 140  # Add the length of the belt to the far edge
            if view == 'overhead':
                far_edge_value = 40  # Set the far edge to fixed height for overhead view
            far_edge = line_at_t(self.find_t_for_coordinate(far_edge_value, coord_index, point_3d, camera_center))

            return near_edge, far_edge

        return None, None

    def reproject_3d_to_2d(self):
        view = self.projection_view.get()
        bp = self.current_label.get()

        # reset the reprojected points
        self.cam_reprojected_points['near'] = {}
        self.cam_reprojected_points['far'] = {}

        near_edge, far_edge = self.find_3d_edges(view, bp)
        if near_edge is not None and far_edge is not None:
            # Define the views and exclude the projection view
            views = ['side', 'front', 'overhead']
            views.remove(view)

            for wcs in [near_edge, far_edge]:
                # Loop through the other views
                for other_view in views:
                    CCS_repr, _ = cv2.projectPoints(
                        wcs,
                        cv2.Rodrigues(self.calibration_data['extrinsics'][other_view]['rotm'])[0],
                        self.calibration_data['extrinsics'][other_view]['tvec'],
                        self.calibration_data['intrinsics'][other_view],
                        np.array([]),
                    )
                    self.cam_reprojected_points['near' if wcs is near_edge else 'far'][other_view] = CCS_repr[
                        0].flatten()

    def draw_reprojected_points(self):
        self.reproject_3d_to_2d()
        for view in ['side', 'front', 'overhead']:
            if view != self.projection_view.get():
                ax = self.axs[["side", "front", "overhead"].index(view)]
                # Clear previous lines if they exist
                if self.projection_lines[view] is not None:
                    self.projection_lines[view].remove()
                    self.projection_lines[view] = None

                frame = cv2.cvtColor(self.frames[view][self.current_frame_index], cv2.COLOR_BGR2RGB)
                frame = self.apply_contrast_brightness(frame)
                ax.imshow(frame)
                ax.set_title(f'{view.capitalize()} View', fontsize=8)
                ax.axis('on')

                if view in self.cam_reprojected_points['near'] and view in self.cam_reprojected_points['far']:
                    near_point = self.cam_reprojected_points['near'][view]
                    far_point = self.cam_reprojected_points['far'][view]

                    # Draw the line between near and far points and store it
                    self.projection_lines[view], = ax.plot(
                        [near_point[0], far_point[0]], [near_point[1], far_point[1]],
                        color='red', linestyle='--', linewidth=0.5
                    )

                # Apply tick settings explicitly
                ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=8)

        self.show_body_part_points()  # Redraw body part points to ensure they are displayed correctly
        self.canvas.draw_idle()

    def on_label_select(self, label):
        self.current_label.set(label)
        self.draw_reprojected_points()

    def on_label_click(self, label):
        print(f"Label clicked: {label}")
        self.current_label.set(label)
        self.update_label_button_selection()
        self.draw_reprojected_points()

    def skip_labeling_frames(self, step):
        self.current_frame_index += step
        self.current_frame_index = max(0, min(self.current_frame_index, len(self.frames['side']) - 1))
        self.frame_label.config(text=f"Frame: {self.current_frame_index + 1}/{len(self.frames['side'])}")
        self.display_frame()
        self.current_label.set("Nose")

    def display_frame(self):
        self.frame_label.config(text=f"Frame: {self.current_frame_index + 1}/{len(self.frames['side'])}")

        self.optimization_checkbox.config(variable=self.optimization_checkboxes[self.current_frame_index])

        # Other existing code to display frames...
        frame_side, frame_front, frame_overhead = self.matched_frames[self.current_frame_index]

        frame_side_img = self.frames['side'][frame_side]
        frame_front_img = self.frames['front'][frame_front]
        frame_overhead_img = self.frames['overhead'][frame_overhead]

        frame_side_img = self.apply_contrast_brightness(frame_side_img)
        frame_front_img = self.apply_contrast_brightness(frame_front_img)
        frame_overhead_img = self.apply_contrast_brightness(frame_overhead_img)

        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()

        self.axs[0].imshow(cv2.cvtColor(frame_side_img, cv2.COLOR_BGR2RGB))
        self.axs[1].imshow(cv2.cvtColor(frame_front_img, cv2.COLOR_BGR2RGB))
        self.axs[2].imshow(cv2.cvtColor(frame_overhead_img, cv2.COLOR_BGR2RGB))

        self.axs[0].set_title('Side View', fontsize=8)
        self.axs[1].set_title('Front View', fontsize=8)
        self.axs[2].set_title('Overhead View', fontsize=8)

        for ax in self.axs:
            # Apply tick settings explicitly
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=8)

        self.show_body_part_points()
        self.canvas.draw()

        # Reset to 'Nose' label
        self.current_label.set("Nose")
        self.update_label_button_selection()

    def show_body_part_points(self, draw=True):
        # Store references to previous scatter collections to avoid removing all collections
        if not hasattr(self, 'scatter_refs'):
            self.scatter_refs = {view: {} for view in ['side', 'front', 'overhead']}

        current_points = self.body_part_points[self.current_frame_index]

        for ax, view in zip(self.axs, ['side', 'front', 'overhead']):
            # Only remove existing scatters if they are going to be replaced
            if view in self.scatter_refs:
                for label, scatter in self.scatter_refs[view].items():
                    if scatter is not None:
                        scatter.remove()
                self.scatter_refs[view].clear()

            for label, coords in current_points.items():
                if coords[view] is not None:
                    x, y = coords[view]
                    if label in config.CALIBRATION_LABELS:
                        # Calibration labels: white with red outlines
                        scatter = ax.scatter(
                            x, y, c='white', edgecolors='red', linewidths=1.5,
                            s=self.marker_size_var.get() * 10, label=label
                        )
                    else:
                        # Normal labels
                        color = self.label_colors[label]
                        scatter = ax.scatter(
                            x, y, c=color, s=self.marker_size_var.get() * 10, label=label
                        )
                    self.scatter_refs[view][label] = scatter  # Store reference for later removal
                else:
                    self.scatter_refs[view][label] = None

        if draw:
            self.canvas.draw_idle()

    def toggle_spacer_lines(self):
        self.spacer_lines_active = not self.spacer_lines_active
        if not self.spacer_lines_active:
            self.remove_spacer_lines()
            self.spacer_lines_points = []
        else:
            self.spacer_lines_points = []

    def remove_spacer_lines(self):
        for line in self.spacer_lines:
            line.remove()
        self.spacer_lines = []
        self.canvas.draw_idle()

    def draw_spacer_lines(self, ax, start_point, end_point):
        if len(self.spacer_lines) > 0:
            self.remove_spacer_lines()

        x_values = np.linspace(start_point[0], end_point[0], num=12)
        for x in x_values:
            line = ax.axvline(x=x, color='pink', linestyle=':', linewidth=1)
            self.spacer_lines.append(line)

        self.canvas.draw_idle()

    def on_tab_press(self, event):
        """Set flag when Tab key is pressed."""
        self.tab_pressed = True

    def on_tab_release(self, event):
        """Reset flag when Tab key is released."""
        self.tab_pressed = False

    def on_click(self, event):
        if event.inaxes not in self.axs:
            return

        view = self.current_view.get()
        ax = self.axs[["side", "front", "overhead"].index(view)]
        label = self.current_label.get()
        color = self.label_colors[label]
        marker_size = self.marker_size_var.get()

        frame_points = self.body_part_points[self.current_frame_index]

        if event.button == MouseButton.RIGHT:
            if self.spacer_lines_active:
                if len(self.spacer_lines_points) < 2:
                    self.spacer_lines_points.append((event.xdata, event.ydata))
                    if len(self.spacer_lines_points) == 2:
                        self.draw_spacer_lines(ax, self.spacer_lines_points[0], self.spacer_lines_points[1])
                return
            if event.key == 'shift':
                self.delete_closest_point(ax, event, frame_points)
            else:
                # Remove existing label point if present
                if frame_points[label][view] is not None:
                    frame_points[label][view] = None  # Remove the old placement
                    self.show_body_part_points(draw=False)  # Clear old points without redrawing yet

                # Add the new point
                frame_points[label][view] = (event.xdata, event.ydata)
                ax.scatter(event.xdata, event.ydata, c=color, s=marker_size * 10, label=label)

                self.canvas.draw_idle()  # Redraw the canvas to update the display
                self.advance_label()
                self.draw_reprojected_points()
        elif event.button == MouseButton.LEFT:
            if label == 'Door' or label not in config.CALIBRATION_LABELS:
                self.dragging_point = self.find_closest_point(ax, event, frame_points)
            if self.tab_pressed:
                # Display label name popup
                label_near_click = self.find_label_near_click(event)
                if label_near_click:
                    self.show_label_popup(label_near_click, event)

    def find_label_near_click(self, event):
        """Check if a click is near any scatter point."""
        click_threshold = 10  # This value defines the clickable area around the scatter points
        for label, views in self.body_part_points[self.current_frame_index].items():
            for view, coords in views.items():
                if view == self.current_view.get() and coords is not None:
                    x, y = coords
                    # Check if the click is within the area of a scatter point
                    if np.hypot(x - event.xdata, y - event.ydata) <= click_threshold:
                        print(f"Click near label: {label}")  # Debug statement
                        return label
        return None

    @debounce(0.1)
    def on_drag(self, event):
        if self.dragging_point is None or event.inaxes not in self.axs:
            return

        label, view, _ = self.dragging_point
        ax = self.axs[["side", "front", "overhead"].index(view)]

        if event.button == MouseButton.LEFT:
            # Check for "Door" or non-calibration labels
            if label == 'Door' or label not in config.CALIBRATION_LABELS:
                self.body_part_points[self.current_frame_index][label][view] = (event.xdata, event.ydata)
                self.show_body_part_points(draw=False)
                self.draw_reprojected_points()
                self.canvas.draw_idle()

    def find_closest_point(self, ax, event, frame_points):
        min_dist = float('inf')
        closest_point = None
        for label, views in frame_points.items():
            for view, coords in views.items():
                if coords is not None:
                    x, y = coords
                    dist = np.hypot(x - event.xdata, y - event.ydata)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = (label, view, coords)
        return closest_point if min_dist < 10 else None

    def delete_closest_point(self, ax, event, frame_points):
        min_dist = float('inf')
        closest_point_label = None
        closest_view = None
        for label, views in frame_points.items():
            for view, coords in views.items():
                if coords is not None:
                    x, y = coords
                    dist = np.hypot(x - event.xdata, y - event.ydata)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point_label = label
                        closest_view = view

        if closest_point_label and closest_view:
            if closest_point_label == 'Door' or closest_point_label not in config.CALIBRATION_LABELS:
                frame_points[closest_point_label][closest_view] = None
                self.display_frame()

    def load_calibration_data(self, calibration_data_path):
        try:
            calibration_coordinates = pd.read_csv(calibration_data_path)
            calib = BasicCalibration(calibration_coordinates)
            cameras_extrinsics = calib.estimate_cams_pose()
            cameras_intrinsics = calib.cameras_intrinsics
            belt_points_WCS = calib.belt_coords_WCS
            belt_points_CCS = calib.belt_coords_CCS

            self.calibration_data = {
                'extrinsics': cameras_extrinsics,
                'intrinsics': cameras_intrinsics,
                'belt points WCS': belt_points_WCS,
                'belt points CCS': belt_points_CCS
            }

            for label in config.CALIBRATION_LABELS:
                for view in ['side', 'front', 'overhead']:
                    x_vals = calibration_coordinates[
                        (calibration_coordinates['bodyparts'] == label) & (calibration_coordinates['coords'] == 'x')][
                        view].values
                    y_vals = calibration_coordinates[
                        (calibration_coordinates['bodyparts'] == label) & (calibration_coordinates['coords'] == 'y')][
                        view].values

                    if len(x_vals) > 0 and len(y_vals) > 0:
                        x = x_vals[0]
                        y = y_vals[0]
                        self.calibration_points_static[label][view] = (x, y)
                        if label != 'Door':
                            for frame in self.body_part_points.keys():
                                self.body_part_points[frame][label][view] = (x, y)
                    else:
                        self.calibration_points_static[label][view] = None
                        print(f"Missing data for {label} in {view} view")

            self.update_projection_matrices()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration data: {e}")
            print(f"Error loading calibration data: {e}")

    def update_projection_matrices(self):
        for view in ["side", "front", "overhead"]:
            self.get_p(view)  # Update P matrix for each view

    def update_marker_size(self, val):
        current_xlim = [ax.get_xlim() for ax in self.axs]
        current_ylim = [ax.get_ylim() for ax in self.axs]
        self.marker_size = self.marker_size_var.get()
        self.display_frame()
        for ax, xlim, ylim in zip(self.axs, current_xlim, current_ylim):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

    # def update_contrast_brightness(self, val):
    #     current_time = time.time()
    #     if current_time - self.last_update_time > 0.1:  # 100ms debounce time
    #         current_xlim = [ax.get_xlim() for ax in self.axs]
    #         current_ylim = [ax.get_ylim() for ax in self.axs]
    #         self.display_frame()
    #         for ax, xlim, ylim in zip(self.axs, current_xlim, current_ylim):
    #             ax.set_xlim(xlim)
    #             ax.set_ylim(ylim)
    #         self.canvas.draw_idle()
    #         self.last_update_time = current_time

    @debounce(0.1)  # Debounce with a 100ms wait time
    def update_contrast_brightness(self, val):
        self.redraw_frame()

    def redraw_frame(self):
        current_xlim = [ax.get_xlim() for ax in self.axs]
        current_ylim = [ax.get_ylim() for ax in self.axs]
        self.display_frame()
        for ax, xlim, ylim in zip(self.axs, current_xlim, current_ylim):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def apply_contrast_brightness(self, frame):
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        frame_bytes = frame.tobytes()
        return self.cached_apply_contrast_brightness(frame_bytes, frame.shape, contrast, brightness)

    @lru_cache(maxsize=128)
    def cached_apply_contrast_brightness(self, frame_bytes, frame_shape, contrast, brightness):
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)  # Convert bytes back to numpy array
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        img_contrast = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(img_contrast)
        img_brightness = enhancer.enhance(brightness)
        return cv2.cvtColor(np.array(img_brightness), cv2.COLOR_RGB2BGR)

    @debounce(0.1)  # Debounce with a 100ms wait time
    def on_scroll(self, event):
        if event.inaxes:
            ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            xdata = event.xdata
            ydata = event.ydata

            if xdata is not None and ydata is not None:
                zoom_factor = 0.9 if event.button == 'up' else 1.1

                new_xlim = [xdata + (x - xdata) * zoom_factor for x in xlim]
                new_ylim = [ydata + (y - ydata) * zoom_factor for y in ylim]

                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)

                self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.button == 2:
            self.panning = True
            self.pan_start = (event.x, event.y)

    def on_mouse_release(self, event):
        if event.button == 2:
            self.panning = False
            self.pan_start = None
        elif event.button == MouseButton.LEFT:
            self.dragging_point = None
        elif event.button == MouseButton.RIGHT and self.spacer_lines_active:
            if len(self.spacer_lines_points) == 2:
                self.spacer_lines_points = []
                self.spacer_lines_active = False

    def on_mouse_move(self, event):
        if self.panning and self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_start = (event.x, event.y)

            ax = self.axs[["side", "front", "overhead"].index(self.current_view.get())]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            scale_x = (xlim[1] - xlim[0]) / self.canvas.get_width_height()[0]
            scale_y = (ylim[1] - ylim[0]) / self.canvas.get_width_height()[1]

            ax.set_xlim(xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
            ax.set_ylim(ylim[0] - dy * scale_y, ylim[1] - dy * scale_y)
            self.canvas.draw_idle()

    def update_crosshair(self, event):
        for line in self.crosshair_lines:
            line.remove()
        self.crosshair_lines = []

        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.crosshair_lines.append(event.inaxes.axhline(y, color='cyan', linestyle='--', linewidth=0.5))
            self.crosshair_lines.append(event.inaxes.axvline(x, color='cyan', linestyle='--', linewidth=0.5))
            self.canvas.draw_idle()

    def advance_label(self):
        current_index = self.labels.index(self.current_label.get())
        next_index = (current_index + 1) % len(self.labels)
        if next_index != 0 or len(self.labels) == 1:
            self.current_label.set(self.labels[next_index])
        else:
            self.current_label.set('')  # No more labels to advance to
        self.draw_reprojected_points()  # Update the reprojected points for the new label

    def reset_view(self):
        for ax in self.axs:
            ax.set_xlim(0, ax.get_images()[0].get_array().shape[1])
            ax.set_ylim(ax.get_images()[0].get_array().shape[0], 0)
        self.contrast_var.set(config.DEFAULT_CONTRAST)
        self.brightness_var.set(config.DEFAULT_BRIGHTNESS)
        self.marker_size_var.set(config.DEFAULT_MARKER_SIZE)
        self.display_frame()

    def save_labels(self):
        video_names = {view: os.path.basename(self.extracted_frames_path[view]) for view in
                       ['side', 'front', 'overhead']}
        save_paths = {
            'side': os.path.join(config.LABEL_SAVE_PATH_TEMPLATE['side'].format(video_name=video_names['side']),
                                "CollectedData_Holly_init.csv"),
            'front': os.path.join(config.LABEL_SAVE_PATH_TEMPLATE['front'].format(video_name=video_names['front']),
                                "CollectedData_Holly_init.csv"),
            'overhead': os.path.join(config.LABEL_SAVE_PATH_TEMPLATE['overhead'].format(video_name=video_names['overhead']),
                                "CollectedData_Holly_init.csv")
        }
        for path in save_paths.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {view: [] for view in ['side', 'front', 'overhead']}
        for frame_idx, labels in self.body_part_points.items():
            for label, views in labels.items():
                for view, coords in views.items():
                    if coords is not None:
                        x, y = coords
                        frame_number = self.matched_frames[frame_idx][['side', 'front', 'overhead'].index(view)]
                        filename = self.frame_names[view][frame_number]
                        video_filename = os.path.basename(self.extracted_frames_path[view])
                        data[view].append((frame_idx, label, x, y, "Holly", video_filename, filename))

        for view, view_data in data.items():
            df_view = pd.DataFrame(view_data, columns=["frame_index", "label", "x", "y", "scorer", "video_filename",
                                                  "frame_filename"])

            # Initialize an empty DataFrame with the correct columns
            multi_cols = pd.MultiIndex.from_product([['Holly'], self.labels, ['x', 'y']],
                                                    names=['scorer', 'bodyparts', 'coords'])
            multi_idx = pd.MultiIndex.from_tuples(
                [('labeled_data', video_names[view], filename) for filename in df_view['frame_filename'].unique()])
            df_ordered = pd.DataFrame(index=multi_idx, columns=multi_cols)

            for _, row in df_view.iterrows():
                df_ordered.loc[('labeled_data', row.video_filename, row.frame_filename), ('Holly', row.label, 'x')] = row.x
                df_ordered.loc[('labeled_data', row.video_filename, row.frame_filename), ('Holly', row.label, 'y')] = row.y

            # Convert the DataFrame to numeric values to ensure saving works
            df_ordered = df_ordered.apply(pd.to_numeric)

            # Save the DataFrame
            save_path = save_paths[view]
            print(f"Saving to {save_path}")
            try:
                df_ordered.to_csv(save_path)
                df_ordered.to_hdf(save_path.replace(".csv", ".h5"), key='df', mode='w', format='fixed')
            except PermissionError as e:
                print(f"PermissionError: {e}")
                messagebox.showerror("Error",
                                     f"Unable to save the file at {save_path}. Please check the file permissions.")

        print("Labels saved successfully")
        messagebox.showinfo("Info", "Labels saved successfully")


    def parse_video_path(self, video_path):
        video_file = os.path.basename(video_path)
        parts = video_file.split('_')
        date = parts[1]
        camera_view = [part for part in parts if part in ['side', 'front', 'overhead']][0]
        name_parts = [part for part in parts if part not in ['side', 'front', 'overhead']]
        name = '_'.join(name_parts).replace('.avi', '')  # Remove the .avi extension if present
        name_with_camera = f"{name}_{camera_view}"
        return name_with_camera, date, camera_view

    def generate_label_colors(self, labels):
        colormap = plt.get_cmap('hsv')
        body_part_labels = [label for label in labels if label not in config.CALIBRATION_LABELS]
        colors = [colormap(i / len(body_part_labels)) for i in range(len(body_part_labels))]
        label_colors = {}
        for label in labels:
            if label in config.CALIBRATION_LABELS:
                label_colors[label] = '#ffffff'  # White color for calibration labels
            else:
                label_colors[label] = self.rgb_to_hex(
                    colors.pop(0))  # Assign colors from the colormap to body part labels
        return label_colors

    def rgb_to_hex(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    ##################################### Calibration enhancements ############################################
    def optimize_calibration(self):
        self.error_history = []
        reference_points = config.OPTIMIZATION_REFERENCE_LABELS

        # Collect labels from frames with the optimization checkbox checked
        checked_frames_indices = [i for i, var in enumerate(self.optimization_checkboxes) if var.get()]

        print(f"Checked frames indices for optimization: {checked_frames_indices}")

        if not checked_frames_indices:
            messagebox.showwarning("Warning", "No frames selected for optimization.")
            return

        initial_total_error, initial_errors = self.compute_reprojection_error(reference_points, checked_frames_indices)
        print(f"Initial total reprojection error for {reference_points}: \n{initial_total_error}")
        for label, views in initial_errors.items():
            print(f"Initial reprojection error for {label}: {views}")

        initial_flat_points = self.flatten_calibration_points()
        args = (reference_points, checked_frames_indices)

        bounds = [(initial_flat_points[i] - 3.0, initial_flat_points[i] + 3.0) for i in range(len(initial_flat_points))]

        print("Optimizing calibration points...")
        # Start the timer
        start_time = time.time()

        # Capture original calibration points before optimisation
        original_points = {
            label: {view: list(self.calibration_points_static[label][view])
                    for view in ['side', 'front', 'overhead']}
            for label in config.CALIBRATION_LABELS}

        result = minimize(self.objective_function, initial_flat_points, args=args, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 100000, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False})

        # End the timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds.")

        optimized_points = self.reshape_calibration_points(result.x)

        for label, views in optimized_points.items():
            for view, point in views.items():
                self.calibration_points_static[label][view] = point

        self.recalculate_camera_parameters()

        new_total_error, new_errors = self.compute_reprojection_error(reference_points, checked_frames_indices)
        print(f"New total reprojection error for {reference_points}: \n{new_total_error}")
        for label, views in new_errors.items():
            print(f"New reprojection error for {label}: {views}")

        self.save_optimized_calibration_points()

        self.update_calibration_labels_and_projection()

        self.display_frame()

        # Save error history and generate plot
        error_history_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(
            video_name=self.video_name).replace('.csv', '_error_history.pkl')
        with open(error_history_path, 'wb') as f:
            pickle.dump({
                'error_history': self.error_history,
                'n_frames': len(checked_frames_indices),
                'labels': reference_points
            }, f)

        self.plot_optimisation_results(
            error_history_path,
            n_frames=len(checked_frames_indices),
            labels=reference_points,
            original_points=original_points,
            frame_index=checked_frames_indices[0]
        )

    def plot_optimisation_results(self, save_path_base, n_frames, labels,
                                  original_points, frame_index):

        fig, axs = plt.subplots(4, 1, figsize=(20, 24))

        for i, view in enumerate(['side', 'front', 'overhead']):
            frame = self.frames[view][frame_index]
            axs[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            for j, label in enumerate(config.CALIBRATION_LABELS):
                orig = original_points[label][view]
                opt = self.calibration_points_static[label][view]
                if orig is not None:
                    axs[i].scatter(orig[0], orig[1], c='red', marker='+', s=60,
                                   label='Original' if j == 0 else '')
                if opt is not None:
                    axs[i].scatter(opt[0], opt[1], c='blue', marker='+', s=60,
                                   label='Optimised' if j == 0 else '')

            axs[i].set_title(f'{view.capitalize()} view — frame {frame_index}',
                             fontsize=12)
            axs[i].axis('off')
            if i == 0:
                axs[i].legend(loc='upper right', frameon=False, fontsize=10)

        # Convergence curve
        axs[3].plot(self.error_history, color='#2c7bb6', linewidth=1.2)
        axs[3].set_xlabel('Iteration', fontsize=12)
        axs[3].set_ylabel('Total weighted reprojection error (px)',
                          fontsize=12)
        axs[3].set_title('Calibration optimisation convergence', fontsize=13)
        axs[3].spines['top'].set_visible(False)
        axs[3].spines['right'].set_visible(False)

        axs[3].text(0.5, -0.18,
                    f'Frames used: {n_frames}     Labels used: {len(labels)}     ({", ".join(labels)})',
                    transform=axs[3].transAxes, fontsize=9, ha='center',
                    va='top',
                    color='dimgrey', style='italic')

        plot_path = save_path_base.replace('_error_history.pkl',
                                           '_optimisation_results.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path.replace('.png', '.svg'), bbox_inches='tight')
        plt.close(fig)
        print(f"Optimisation plot saved to {plot_path}")

    def compute_reprojection_error(self, labels, frame_indices, extrinsics=None, weighted=False):
        errors = {label: {"side": 0, "front": 0, "overhead": 0} for label in labels}
        cams = ["side", "front", "overhead"]
        total_error = 0

        for frame_index in frame_indices:
            frame_index = int(frame_index)  # Ensure frame_index is an integer
            side_frame, front_frame, overhead_frame = self.matched_frames[frame_index]
            for label in labels:
                point_3d = self.triangulate(label, extrinsics, side_frame, front_frame, overhead_frame)
                if point_3d is not None:
                    point_3d = point_3d[:3]
                    projections = self.project_to_view(point_3d, extrinsics)

                    for view, frame in zip(cams, [side_frame, front_frame, overhead_frame]):
                        if self.body_part_points[frame][label][view] is not None:
                            original_x, original_y = self.body_part_points[frame][label][view]
                            if view in projections:
                                projected_x, projected_y = projections[view]
                                error = np.sqrt(
                                    (projected_x - original_x) ** 2 + (projected_y - original_y) ** 2)
                                if weighted:
                                    weight = config.REFERENCE_LABEL_WEIGHTS.get(view, {}).get(label, 1.0)
                                    error *= weight
                                errors[label][view] = error
                                total_error += error
        return total_error, errors

    def triangulate(self, label, extrinsics=None, side_frame=None, front_frame=None, overhead_frame=None):
        P = []
        coords = []

        frame_mapping = {'side': side_frame, 'front': front_frame, 'overhead': overhead_frame}
        for view, frame in frame_mapping.items():
            if self.body_part_points[frame][label][view] is not None:
                P.append(self.get_p(view, extrinsics=extrinsics, return_value=True))
                coords.append(self.body_part_points[frame][label][view])

        if len(P) < 2 or len(coords) < 2:
            return None

        P = np.array(P)
        coords = np.array(coords)

        point_3d = triangulate(coords, P)
        return point_3d

    def project_to_view(self, point_3d, extrinsics=None):
        projections = {}
        for view in ["side", "front", "overhead"]:
            if extrinsics is None:
                extrinsics = self.calibration_data['extrinsics']
            if extrinsics[view] is not None:
                CCS_repr, _ = cv2.projectPoints(
                    point_3d,
                    cv2.Rodrigues(extrinsics[view]['rotm'])[0],
                    extrinsics[view]['tvec'],
                    self.calibration_data['intrinsics'][view],
                    np.array([]),
                )
                projections[view] = CCS_repr[0].flatten()
        return projections

    def flatten_calibration_points(self):
        flat_points = []
        for label in config.CALIBRATION_LABELS:
            for view in ['side', 'front', 'overhead']:
                if self.calibration_points_static[label][view] is not None:
                    flat_points.extend(self.calibration_points_static[label][view])
        return np.array(flat_points, dtype=float)

    def objective_function(self, flat_points, *args):
        reference_points = args[0]  # Extract reference points from args
        frame_indices = args[1]  # Extract frame indices from args
        calibration_points = self.reshape_calibration_points(flat_points)
        temp_extrinsics = self.estimate_extrinsics(calibration_points)

        total_error, _ = self.compute_reprojection_error(reference_points, frame_indices, temp_extrinsics,
                                                         weighted=True)
        self.error_history.append(total_error)
        print("Total error: %s" %total_error)
        return total_error

    def estimate_extrinsics(self, calibration_points):
        calibration_coordinates = pd.DataFrame([
            {'bodyparts': label, 'coords': coord, 'side': calibration_points[label]['side'][i],
             'front': calibration_points[label]['front'][i],
             'overhead': calibration_points[label]['overhead'][i]}
            for label in calibration_points
            for i, coord in enumerate(['x', 'y'])
        ])

        calib = BasicCalibration(calibration_coordinates)
        cameras_extrinsics = calib.estimate_cams_pose()
        return cameras_extrinsics

    def reshape_calibration_points(self, flat_points):
        calibration_points = {label: {"side": None, "front": None, "overhead": None} for label in
                              config.CALIBRATION_LABELS}
        i = 0
        for label in config.CALIBRATION_LABELS:
            for view in ['side', 'front', 'overhead']:
                if self.calibration_points_static[label][view] is not None:
                    calibration_points[label][view] = [flat_points[i], flat_points[i + 1]]
                    i += 2
        return calibration_points

    def recalculate_camera_parameters(self):
        calibration_coordinates = pd.DataFrame([
            {'bodyparts': label, 'coords': coord, 'side': self.calibration_points_static[label]['side'][i],
             'front': self.calibration_points_static[label]['front'][i],
             'overhead': self.calibration_points_static[label]['overhead'][i]}
            for label in self.calibration_points_static
            for i, coord in enumerate(['x', 'y'])
        ])

        calib = BasicCalibration(calibration_coordinates)
        cameras_extrinsics = calib.estimate_cams_pose()
        cameras_intrinsics = calib.cameras_intrinsics
        belt_points_WCS = calib.belt_coords_WCS
        belt_points_CCS = calib.belt_coords_CCS

        self.calibration_data = {
            'extrinsics': cameras_extrinsics,
            'intrinsics': cameras_intrinsics,
            'belt points WCS': belt_points_WCS,
            'belt points CCS': belt_points_CCS
        }

    def save_optimized_calibration_points(self):
        calibration_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=self.video_name).replace('.csv',
                                                                                                            '_enhanced.csv')
        os.makedirs(os.path.dirname(calibration_path), exist_ok=True)

        data = {"bodyparts": [], "coords": [], "side": [], "front": [], "overhead": []}
        for label, coords in self.calibration_points_static.items():
            if label in config.CALIBRATION_LABELS:
                for coord in ['x', 'y']:
                    data["bodyparts"].append(label)
                    data["coords"].append(coord)
                    for view in ["side", "front", "overhead"]:
                        if coords[view] is not None:
                            x, y = coords[view]
                            if coord == 'x':
                                data[view].append(x)
                            else:
                                data[view].append(y)
                        else:
                            data[view].append(None)

        df = pd.DataFrame(data)
        try:
            df.to_csv(calibration_path, index=False)
            messagebox.showinfo("Info", "Optimized calibration points saved successfully")
        except PermissionError:
            print(
                f"Permission denied: Unable to save the file at {calibration_path}. Please check the file permissions.")
            messagebox.showerror("Error",
                                 f"Unable to save the file at {calibration_path}. Please check the file permissions.")

    def update_calibration_labels_and_projection(self):
        enhanced_calibration_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=self.video_name).replace(
            '.csv', '_enhanced.csv')
        if os.path.exists(enhanced_calibration_path):
            self.load_calibration_data(enhanced_calibration_path)
            self.update_projection_matrices()
            self.display_frame()
        else:
            print(f"Enhanced calibration file not found: {enhanced_calibration_path}")

        ############################################################################################################

    def confirm_exit(self):
        answer = messagebox.askyesnocancel("Exit", "Do you want to save the labels before exiting?")
        if answer is not None:
            if answer:  # Yes, save labels and exit
                self.save_labels()
            self.root.quit()  # Exit without saving if No or after saving if Yes




class ReplaceCalibrationLabels():
    # button in MainTool to adjust all CollectedData_Holly_init.csv files under config.dir to replace the current calibration labels (excluding 'Door') with the original calibration labels
    def __init__(self, root, main_tool):
        self.root = root
        self.root.title("Replace Calibration Labels")
        self.root.geometry("300x100")
        self.root.resizable(False, False)
        self.main_tool = main_tool

        self.dir = config.dir

        self.replace_button = tk.Button(self.root, text="Replace Calibration Labels", command=self.replace_labels)
        self.replace_button.pack(pady=20)

        # add a Main Menu button to return to the main tool
        self.main_menu_button = tk.Button(self.root, text="Main Menu", command=self.return_to_main_menu)
        self.main_menu_button.pack(pady=20)


    def replace_labels(self):
        for view in ["Side", "Front", "Overhead"]:
            view_dir = '/'.join([self.dir, view])
            video_dirs = [f for f in os.listdir(view_dir) if os.path.isdir(os.path.join(view_dir, f))]
            for video_name in video_dirs:
                video_dir = '/'.join([view_dir, video_name])
                for file in os.listdir(video_dir):
                    if file.endswith("CollectedData_Holly_init.h5"):
                        try:
                            file_path = '/'.join([video_dir, file])
                            #file_path = os.path.join(video_dir, file)
                            df = pd.read_hdf(file_path)
                            df = self.replace_calibration_labels(df, video_name)
                            df = self.replace_labeled_data_hyphen(df)
                            df.to_csv(file_path.replace("CollectedData_Holly_init.h5","CollectedData_Holly.csv"))
                            df.to_hdf(file_path.replace("CollectedData_Holly_init.h5", "CollectedData_Holly.h5"), key='df_with_missing', mode='w', format='fixed')
                        except:
                            print(f"Error replacing calibration labels for {file_path}")
        messagebox.showinfo("Info", "Calibration labels replaced successfully")

    def replace_calibration_labels(self, df, video_name):
        # remove view name from video_name
        video_name, date, view = self.parse_video_path(video_name)
        # change uppercase view to lowercase
        view = view.lower()

        calibration_labels_path = config.CALIBRATION_SAVE_PATH_TEMPLATE.format(video_name=video_name)
        calibration_labels = pd.read_csv(calibration_labels_path)
        calibration_labels = calibration_labels[calibration_labels['bodyparts'] != 'Door']

        for label in calibration_labels['bodyparts'].unique():
            for coord in ['x', 'y']:
                calibration = calibration_labels[(calibration_labels['bodyparts'] == label) & (calibration_labels['coords'] == coord)].loc(axis=1)[view]
                # fill all rows with the same calibration value
                df.loc(axis=1)['Holly', label, coord] = calibration.values[0]

        return df

    def replace_labeled_data_hyphen(self, df):
        # replace 'labeled_data' in index level 0 with 'labeled-data'
        df = df.rename(index={'labeled_data': 'labeled-data'}, level=0)
        return df


    def parse_video_path(self, video_path):
        video_file = os.path.basename(video_path)
        parts = video_file.split('_')
        date = parts[1]
        camera_view = [part for part in parts if part in ['side', 'front', 'overhead']][0]
        name_parts = [part for part in parts if part not in ['side', 'front', 'overhead']]
        name = '_'.join(name_parts).replace('.avi', '')  # Remove .avi extension if present
        return name, date, camera_view

    def return_to_main_menu(self):
        self.root.destroy()
        self.main_tool.root.deiconify()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainTool(root)
    root.mainloop()