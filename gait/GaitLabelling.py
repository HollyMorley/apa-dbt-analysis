"""
Gait Labelling Tool for Manual Annotation of Limb Stance/Swing States

This matplotlib-based GUI tool enables manual labelling of limb states (stance vs swing)
in dual-camera mouse gait videos. It displays synchronised side and front view frames
with interactive toggle buttons for labelling four limbs.

Pipeline:
    1. Load matched image pairs from side/front camera directories
    2. Extract frame numbers from filenames and match corresponding views
    3. Display images with interactive buttons for four limbs
    4. Label each limb as stance (green), swing (red), or unknown (black)
    5. Save labels to CSV with frame number and subdirectory identifiers

Label States:
    - True (green): Stance phase (paw on ground)
    - False (red): Swing phase (paw in air)
    - None (black): Unknown/unlabelled

User Interactions:
    - Left click on limb button: cycles True → False → True
    - Right click on limb button: resets to None (unknown)
    - Next/Previous buttons: navigate frames (auto-saves current labels)
    - Zoom button: activate rectangle selector on side view
    - Reset View: restore original zoom
    - Save button: writes all labels to CSV

Output CSV Format:
    Columns: Frame, Subdirectory, HindpawL, ForepawL, HindpawR, ForepawR
    Values: 1 (stance), 0 (swing), 'unknown'

Note: This is a prototype tool. Future napari-based implementation will include
keyboard shortcuts, timeline scrubber, and pose tracking overlay.
"""

import pandas as pd
import os, re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.widgets import Button, RectangleSelector
import tkinter as tk
from tkinter import messagebox

# Custom Button class that doesn't change color on hover or click
class NoFlashButton(Button):
    def __init__(self, *args, **kwargs):
        super(NoFlashButton, self).__init__(*args, **kwargs)

    def on_enter(self, event):
        pass  # Do nothing to prevent hover color change

    def on_leave(self, event):
        pass  # Do nothing to prevent hover color change

    def on_press(self, event):
        pass  # Do nothing to prevent color change on press

    def on_release(self, event):
        if not self.eventson:
            return
        if self.ignore(event):
            return
        contains, attrd = self.ax.contains(event)
        if not contains:
            return
        for cid, func in self._clickobservers.items():
            func(event)

class ImageLabeler:
    def __init__(self, base_dir_side, base_dir_front, subdirs_to_include, output_file):
        self.base_dir_side = base_dir_side
        self.base_dir_front = base_dir_front
        self.subdirs_to_include = subdirs_to_include
        self.output_file = output_file
        self.image_files = self.load_image_files()
        self.current_index = 0
        self.labels = {}  # Store labels with (frame_num, subdir) as keys
        self.num_images = len(self.image_files)
        self.zoom_rect_selector = None  # For zoom functionality
        self.ax_side = None  # Will be set in label_images
        self.ax_front = None  # Will be set in label_images
        self.image_display_side = None  # Will be set in label_images
        self.image_display_front = None  # Will be set in label_images

        # Initialize Tkinter root (needed for message boxes)
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window

        # Load existing labels if the output file exists
        if os.path.exists(self.output_file):
            self.load_existing_labels()

    def load_image_files(self):
        image_files = []
        for subdir in self.subdirs_to_include:
            dir_path_side = os.path.join(self.base_dir_side, subdir)
            # Replace 'side' with 'front' in subdirectory name for front images
            subdir_front = subdir.replace('side', 'front')
            dir_path_front = os.path.join(self.base_dir_front, subdir_front)
            if not os.path.exists(dir_path_side):
                print(f"Warning: Side subdirectory {dir_path_side} does not exist.")
                continue
            if not os.path.exists(dir_path_front):
                print(f"Warning: Front subdirectory {dir_path_front} does not exist.")
                continue
            files_side = [f for f in os.listdir(dir_path_side) if f.endswith('.png') or f.endswith('.jpg')]
            files_front = [f for f in os.listdir(dir_path_front) if f.endswith('.png') or f.endswith('.jpg')]

            # Sort the files by filename (alphabetically)
            files_side.sort()
            files_front.sort()

            # Match images by their sorted order
            min_length = min(len(files_side), len(files_front))

            for i in range(min_length):
                f_side = files_side[i]
                f_front = files_front[i]
                # Extract frame number from side image filename
                match_side = re.search(r'img(\d+)', f_side)
                if match_side:
                    frame_num = int(match_side.group(1))
                else:
                    frame_num = i  # Use index if frame number not found
                filepath_side = os.path.join(dir_path_side, f_side)
                filepath_front = os.path.join(dir_path_front, f_front)
                image_files.append({
                    'frame_num': frame_num,
                    'filepath_side': filepath_side,
                    'filepath_front': filepath_front,
                    'subdir': subdir  # Include original subdirectory for labeling
                })
        # Sort the images by subdirectory and frame number
        sorted_files = sorted(image_files, key=lambda x: (x['subdir'], x['frame_num']))
        return sorted_files

    def load_existing_labels(self):
        try:
            labels_df = pd.read_csv(self.output_file, na_values=['unknown'])
            for _, row in labels_df.iterrows():
                frame_num = row['Frame']
                subdir = row['Subdirectory']
                key = (frame_num, subdir)
                self.labels[key] = {
                    'HindpawL': self.convert_label(row['HindpawL']),
                    'ForepawL': self.convert_label(row['ForepawL']),
                    'HindpawR': self.convert_label(row['HindpawR']),
                    'ForepawR': self.convert_label(row['ForepawR']),
                }
            print(f"Loaded existing labels from {self.output_file}")
        except Exception as e:
            print(f"Failed to load existing labels: {e}")

    def convert_label(self, value):
        if pd.isna(value):
            return None  # Represent 'unknown' labels as None
        else:
            try:
                return bool(int(value))
            except ValueError:
                return None

    def load_image(self, index):
        image_info = self.image_files[index]
        image_path_side = image_info['filepath_side']
        image_path_front = image_info['filepath_front']
        frame_num = image_info['frame_num']
        image_side = plt.imread(image_path_side)
        image_front = plt.imread(image_path_front)
        return image_side, image_front, frame_num

    def load_current_labels(self):
        frame_info = self.image_files[self.current_index]
        frame_num = int(frame_info['frame_num'])  # Ensure frame_num is int
        subdir = frame_info['subdir'].strip()  # Remove any leading/trailing whitespace
        key = (frame_num, subdir)
        print(f"Current key: {key}")
        if key in self.labels:
            self.limb_states = self.labels[key].copy()
            print(f"Found labels for key {key}: {self.limb_states}")
        else:
            self.limb_states = {
                'HindpawL': None,
                'ForepawL': None,
                'HindpawR': None,
                'ForepawR': None
            }
            print(f"No labels found for key {key}. Using default states.")
        self.update_button_colors()

    def update_button_colors(self):
        for limb, button in [('HindpawL', self.hindpawL_button),
                             ('ForepawL', self.forepawL_button),
                             ('HindpawR', self.hindpawR_button),
                             ('ForepawR', self.forepawR_button)]:
            state = self.limb_states[limb]
            if state is True:
                color = 'green'
            elif state is False:
                color = 'red'
            elif state is None:
                color = 'black'
            else:
                color = 'black'  # Default to black if state is unexpected
            button.label.set_color(color)
            button.ax.figure.canvas.draw_idle()

    def label_images(self):
        # Before initializing Matplotlib figure, ask the user
        if self.labels:
            answer = messagebox.askyesno("Start Position", "Do you want to start from the first frame?")
            if answer:
                self.current_index = 0
            else:
                # Find the index of the last labeled frame in self.image_files
                labeled_indices = []
                for idx, image_info in enumerate(self.image_files):
                    key = (image_info['frame_num'], image_info['subdir'])
                    if key in self.labels:
                        labeled_indices.append(idx)
                if labeled_indices:
                    self.current_index = max(labeled_indices) + 1  # Start from the next frame after the last labeled
                    if self.current_index >= self.num_images:
                        self.current_index = self.num_images - 1  # Ensure index is within bounds
                else:
                    self.current_index = 0  # No labeled frames found
        else:
            self.current_index = 0

        # Initialize Matplotlib figure and axes
        fig = plt.figure(figsize=(16, 9))

        # Positions for axes
        button_area_height = 0.22  # Height allocated for the buttons at the bottom
        front_area_height = 0.4  # Increase front area height to move it higher
        side_area_height = 1 - button_area_height - front_area_height  # Remaining height for the side image

        # Side image occupies the top portion
        self.ax_side = fig.add_axes([0, front_area_height + button_area_height, 1, side_area_height - 0.05])

        # Front image occupies below the side image
        self.ax_front = fig.add_axes([0, button_area_height + 0.05, 0.6, front_area_height - 0.05])

        # Initialize limb states for the current image
        self.limb_states = {
            'HindpawL': None,
            'ForepawL': None,
            'HindpawR': None,
            'ForepawR': None
        }

        # Create navigation buttons at the bottom
        nav_button_y = 0.1
        nav_button_height = 0.05

        bprev = Button(plt.axes([0.3, nav_button_y, 0.1, nav_button_height]), 'Prev')
        bnext = Button(plt.axes([0.6, nav_button_y, 0.1, nav_button_height]), 'Next')

        # Add zoom and reset view buttons
        zoom_button = Button(plt.axes([0.05, nav_button_y, 0.1, nav_button_height]), 'Zoom')
        reset_button = Button(plt.axes([0.15, nav_button_y, 0.1, nav_button_height]), 'Reset View')

        # Add save button
        save_button = Button(plt.axes([0.85, nav_button_y, 0.1, nav_button_height]), 'Save')

        # Create paw buttons to the left/middle, aligned with the midline
        button_width = 0.15
        button_height = 0.05
        button_spacing_x = 0.05
        button_spacing_y = 0.05

        # Calculate total width of paw buttons including spacing
        total_button_width = button_width * 2 + button_spacing_x

        # Position paw buttons so the leftmost side aligns with the midline (x=0.5)
        start_x = 0.5  # Left edge of the paw button grid

        # Y positions
        paw_button_y_top = button_area_height + front_area_height - button_height - 0.1  # Adjusted Y-position for the top row
        paw_button_y_bottom = paw_button_y_top - button_height - button_spacing_y

        # Define button positions
        hindpawL_button_ax = plt.axes([start_x, paw_button_y_top, button_width, button_height])
        forepawL_button_ax = plt.axes([start_x + button_width + button_spacing_x, paw_button_y_top, button_width, button_height])
        hindpawR_button_ax = plt.axes([start_x, paw_button_y_bottom, button_width, button_height])
        forepawR_button_ax = plt.axes([start_x + button_width + button_spacing_x, paw_button_y_bottom, button_width, button_height])

        # Create custom buttons that do not change color on click or hover and assign to self
        self.hindpawL_button = NoFlashButton(hindpawL_button_ax, 'HindpawL', color='lightgrey', hovercolor='grey')
        self.forepawL_button = NoFlashButton(forepawL_button_ax, 'ForepawL', color='lightgrey', hovercolor='grey')
        self.hindpawR_button = NoFlashButton(hindpawR_button_ax, 'HindpawR', color='lightgrey', hovercolor='grey')
        self.forepawR_button = NoFlashButton(forepawR_button_ax, 'ForepawR', color='lightgrey', hovercolor='grey')

        # Now, load labels for the current image if available
        self.load_current_labels()

        # Load the current image
        image_side, image_front, frame_num = self.load_image(self.current_index)
        self.image_display_side = self.ax_side.imshow(image_side)
        self.image_display_front = self.ax_front.imshow(image_front)
        self.ax_side.axis('off')  # Hide axes ticks
        self.ax_front.axis('off')  # Hide axes ticks

        # Function to update the image display
        def update_image():
            image_side, image_front, frame_num = self.load_image(self.current_index)
            if self.image_display_side is None:
                self.image_display_side = self.ax_side.imshow(image_side)
            else:
                self.image_display_side.set_data(image_side)
            if self.image_display_front is None:
                self.image_display_front = self.ax_front.imshow(image_front)
            else:
                self.image_display_front.set_data(image_front)
            self.ax_side.set_xlim(0, image_side.shape[1])
            self.ax_side.set_ylim(image_side.shape[0], 0)
            self.ax_front.set_xlim(0, image_front.shape[1])
            self.ax_front.set_ylim(image_front.shape[0], 0)
            frame_info = self.image_files[self.current_index]
            subdir = frame_info['subdir']
            self.ax_side.set_title(f"Frame {frame_num} ({self.current_index + 1}/{self.num_images})\nSubdirectory: {subdir}", fontsize=12)
            # Update button colors based on limb states
            self.update_button_colors()
            plt.draw()

        # Handlers for paw buttons
        def toggle_hindpawL(event):
            if event.button == 1:
                # Left-click: toggle between False and True
                current_state = self.limb_states['HindpawL']
                if current_state is None or current_state is False:
                    self.limb_states['HindpawL'] = True
                elif current_state is True:
                    self.limb_states['HindpawL'] = False
            elif event.button == 3:
                # Right-click: set to unknown (None)
                self.limb_states['HindpawL'] = None
            self.update_button_colors()

        def toggle_forepawL(event):
            if event.button == 1:
                current_state = self.limb_states['ForepawL']
                if current_state is None or current_state is False:
                    self.limb_states['ForepawL'] = True
                elif current_state is True:
                    self.limb_states['ForepawL'] = False
            elif event.button == 3:
                self.limb_states['ForepawL'] = None
            self.update_button_colors()

        def toggle_hindpawR(event):
            if event.button == 1:
                current_state = self.limb_states['HindpawR']
                if current_state is None or current_state is False:
                    self.limb_states['HindpawR'] = True
                elif current_state is True:
                    self.limb_states['HindpawR'] = False
            elif event.button == 3:
                self.limb_states['HindpawR'] = None
            self.update_button_colors()

        def toggle_forepawR(event):
            if event.button == 1:
                current_state = self.limb_states['ForepawR']
                if current_state is None or current_state is False:
                    self.limb_states['ForepawR'] = True
                elif current_state is True:
                    self.limb_states['ForepawR'] = False
            elif event.button == 3:
                self.limb_states['ForepawR'] = None
            self.update_button_colors()

        self.hindpawL_button.on_clicked(toggle_hindpawL)
        self.forepawL_button.on_clicked(toggle_forepawL)
        self.hindpawR_button.on_clicked(toggle_hindpawR)
        self.forepawR_button.on_clicked(toggle_forepawR)

        # Handlers for next and previous image buttons
        def next_image(event):
            self.save_current_labels()
            if self.current_index < self.num_images - 1:
                self.current_index += 1
                self.load_current_labels()
                update_image()

        def prev_image(event):
            self.save_current_labels()
            if self.current_index > 0:
                self.current_index -= 1
                self.load_current_labels()
                update_image()

        bnext.on_clicked(next_image)
        bprev.on_clicked(prev_image)

        # Save labels based on the limb states
        def save_labels_callback(event):
            self.save_current_labels()
            try:
                self.save_labels()
                # Show success message
                messagebox.showinfo("Save Successful", f"Labels saved to {self.output_file}")
            except Exception as e:
                # Show error message
                messagebox.showerror("Save Failed", f"Failed to save labels:\n{str(e)}")

        save_button.on_clicked(save_labels_callback)

        # Zoom functionality
        def zoom_callback(event):
            if self.zoom_rect_selector is None:
                self.zoom_rect_selector = RectangleSelector(self.ax_side, onselect, drawtype='box',
                                                            useblit=True, button=[1],
                                                            minspanx=5, minspany=5, spancoords='pixels',
                                                            interactive=True)
            else:
                self.zoom_rect_selector.set_active(True)

        zoom_button.on_clicked(zoom_callback)

        # Reset view functionality

        def reset_callback(event):
            if self.zoom_rect_selector is not None:
                self.zoom_rect_selector.set_active(False)
            if self.image_display_side is not None:
                self.ax_side.set_xlim(0, self.image_display_side.get_array().shape[1])
                self.ax_side.set_ylim(self.image_display_side.get_array().shape[0], 0)
            if self.image_display_front is not None:
                self.ax_front.set_xlim(0, self.image_display_front.get_array().shape[1])
                self.ax_front.set_ylim(self.image_display_front.get_array().shape[0], 0)
            plt.draw()

        reset_button.on_clicked(reset_callback)

        # Function to handle zoom area selection
        def onselect(eclick, erelease):
            x_min, x_max = sorted([eclick.xdata, erelease.xdata])
            y_min, y_max = sorted([eclick.ydata, erelease.ydata])
            self.ax_side.set_xlim(x_min, x_max)
            self.ax_side.set_ylim(y_max, y_min)
            plt.draw()
            self.zoom_rect_selector.set_active(False)

        # Function to save current limb states
        def save_current_labels():
            frame_info = self.image_files[self.current_index]
            frame_num = frame_info['frame_num']
            subdir = frame_info['subdir']
            key = (frame_num, subdir)
            self.labels[key] = self.limb_states.copy()

        self.save_current_labels = save_current_labels  # Make accessible outside label_images

        # Initial display
        update_image()

        plt.show()
        # After the GUI is closed, destroy the Tkinter root
        self.root.destroy()

    def save_labels(self):
        # Convert self.labels dict to DataFrame
        labels_list = []
        for key, limb_states in self.labels.items():
            frame_num, subdir = key
            row = {
                'Frame': frame_num,
                'Subdirectory': subdir  # Include subdirectory
            }
            # Map limb_states to appropriate values
            def map_state(v):
                if v is True:
                    return 1
                elif v is False:
                    return 0
                elif v is None:
                    return 'unknown'
                else:
                    return 'unknown'
            row.update({k: map_state(v) for k, v in limb_states.items()})
            labels_list.append(row)
        labels_df = pd.DataFrame(labels_list)
        # Remove duplicates, keeping the last entry
        labels_df = labels_df.drop_duplicates(subset=['Frame', 'Subdirectory'], keep='last')
        # Sort labels by subdir and frame number
        labels_df = labels_df.sort_values(['Subdirectory', 'Frame'])
        # Save to CSV
        labels_df.to_csv(self.output_file, index=False)
        print(f"Labels saved to {self.output_file}")


def main():
    base_directory_side = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Side"
    base_directory_front = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\Front"
    # List of subdirectories to include
    subdirectories_to_include = [
        "HM_20230306_APACharRepeat_FAA-1035243_None_side_1",
        "HM_20230306_APACharRepeat_FAA-1035245_R_side_1"
        "HM_20230309_APACharRepeat_FAA-1035297_R_side_1",
        "HM_20230306_APACharRepeat_FAA-1035244_L_side_1",
        "HM_20230307_APAChar_FAA-1035302_LR_side_1",
        "HM_20230308_APACharRepeat_FAA-1035244_L_side_1",
        "HM_20230319_APACharExt_FAA-1035245_R_side_1",
        "HM_20230326_APACharExt_FAA-1035246_LR_side_1",
        "HM_20230404_APACharExt_FAA-1035299_None_side_1",
        "HM_20230412_APACharExt_FAA-1035302_LR_side_1",
        "HM_20230309_APACharRepeat_FAA-1035301_R_side_1",
        "HM_20230325_APACharExt_FAA-1035249_R_side_1"
    ]
    output_file = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round4_Oct24\LimbStuff\limb_labels.csv"

    # Pass the output_file to the ImageLabeler instance
    labeler = ImageLabeler(base_directory_side, base_directory_front, subdirectories_to_include, output_file)
    if labeler.num_images == 0:
        print("No images found in the specified subdirectories.")
        return

    labeler.label_images()


if __name__ == '__main__':
    main()
