"""Iteratively optimizes camera calibration coordinates to minimize reprojection error."""
import helpers.MultiCamLabelling_config as opt_config
from helpers.config import *
from helpers.CalibrateCams import BasicCalibration
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import cv2
import os
import sys
import matplotlib.pyplot as plt
from pycalib.calib import triangulate
import pickle
import time
import warnings
from scipy.optimize import OptimizeWarning
import contextlib
import subprocess


class FileLogger:
    def __init__(self, file_path):
        self.file = open(file_path, 'w')

    def write(self, message):
        self.file.write(message)
        self.file.flush()  # Ensure immediate write

    def flush(self):
        pass  # Needed for compatibility with print's flush behavior

    def close(self):
        self.file.close()

class optimize:
    def __init__(self, calibration_coords, extrinsics, intrinsics, base_path_name, parent_instance):
        self.calibration_coords = calibration_coords
        self.P = None
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.parent = parent_instance
        self.iteration_count = 0
        #self.reference_data = []
        self.body_part_points = {}
        self.error_history = []  # Store error values during optimization
        self.base_path_name = base_path_name

    def optimise_calibration(self, debugging):
        dir = os.path.dirname(self.parent.files['side'])
        pos = np.where(np.array(os.path.basename(self.parent.files['side']).split('_')) == 'side')[0][0]
        filename_base = '_'.join(os.path.basename(self.parent.files['side']).split('_')[:pos])
        filename = filename_base + '_calibration_data.pkl'

        if debugging:
            # retrieve saved calibration data if present, otherwise calculate and save it
            if filename in os.listdir(dir): # look for calibration data in dir
                with open(os.path.join(dir, filename), 'rb') as f:
                    calibration_data = pickle.load(f)
            else:
                calibration_data, _ = self.optimise()
                with open(os.path.join(dir, filename), 'wb') as f:
                    pickle.dump(calibration_data, f)

        else:
            # calculate calibration data and save it
            calibration_data, _ = self.optimise()
            with open(os.path.join(dir, filename), 'wb') as f:
                pickle.dump(calibration_data, f)

        return calibration_data

    def optimise(self):
        reference_points = ['Nose', 'EarL', 'EarR', 'ForepawToeR', 'ForepawToeL', 'HindpawToeR',
                            'HindpawKnuckleR', 'Back1', 'Back6', 'Tail1', 'Tail12',
                            'StartPlatR', 'StepR', 'StartPlatL', 'StepL', 'TransitionR', 'TransitionL']
        #self.reference_data = self.get_data_for_optimisation(reference_points)
        self.body_part_points = self.convert_df_to_dict(reference_points)

        reference_indexes = list(self.body_part_points.keys())

        #self.plot_selected_frames(reference_points)

        initial_total_error, initial_errors = self.compute_reprojection_error(reference_points, reference_indexes, weighted=True)
        print(f"Initial total reprojection error for {reference_points}: \n{initial_total_error}")

        initial_flat_points = self.flatten_calibration_points()
        args = (reference_points, reference_indexes)

        bounds = [(initial_flat_points[i] - 5.0, initial_flat_points[i] + 5.0) for i in range(len(initial_flat_points))]

        print("Optimizing calibration points...")
        # Start the timer
        start_time = time.time()

        result = minimize(self.objective_function, initial_flat_points, args=args, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 100000, 'ftol': 1e-6, 'gtol': 1e-6, 'disp':True}) # 100000, 1e-15, 1e-15

        # End the timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds.")

        optimized_points = self.reshape_calibration_points(result.x)

        min_index = np.sort(reference_indexes).min()
        max_index = np.sort(reference_indexes).max()
        self.plot_original_vs_optimized(min_index, optimized_points, self.calibration_coords, 'start')
        self.plot_original_vs_optimized(max_index, optimized_points, self.calibration_coords, 'end')

        # for label, views in optimized_points.items():
        #     for view, point in views.items():
        #         self.calibration_coords[label][view] = point
        self.calibration_coords = optimized_points

        calibration_data = self.recalculate_camera_parameters()
        self.extrinsics = calibration_data['extrinsics']

        new_total_error, new_errors = self.compute_reprojection_error(reference_points, reference_indexes, weighted=True)
        print(f"New total reprojection error for {reference_points}: \n{new_total_error}")
        print(f"Reprojection error improvement: {initial_total_error - new_total_error}, as a percentage: {((initial_total_error - new_total_error) / initial_total_error) * 100:.2f}%")

        return calibration_data, new_total_error

    def convert_df_to_dict(self, reference_points):
        reference_data = self.get_data_for_optimisation(reference_points)

        reference_data_dict = {}
        for view, df in reference_data.items():
            # Iterate through each row in the dataframe
            for index, row in df.iterrows():
                # Initialize body part entry in the output dictionary if not present
                if index not in reference_data_dict:
                    reference_data_dict[index] = {}

                # Iterate through each body part (columns except 'original_index')
                for bodypart in df.columns.get_level_values('bodyparts').unique():
                    # Skip 'original_index' and any columns that don't have x, y coords
                    if bodypart == 'original_index':
                        continue

                    x_coord = row[bodypart]['x']
                    y_coord = row[bodypart]['y']

                    # Initialize body part dictionary if not present
                    if bodypart not in reference_data_dict[index]:
                        reference_data_dict[index][bodypart] = {}

                    # Set the coordinate tuple
                    reference_data_dict[index][bodypart][view] = (x_coord, y_coord) if pd.notnull(x_coord) and pd.notnull(
                        y_coord) else None
        return reference_data_dict

    def plot_original_vs_optimized(self, frame_number, optimized_points, original_points, suffix):
        """
        Plots the original and optimized calibration labels for each view (side, front, overhead),
        and the error curve during optimization as a fourth subplot.
        """
        # reference_data = self.get_data_for_optimisation(reference_points)
        # real_frame_number = reference_data['side'].iloc(axis=0)[frame_number].loc['original_index'].values[0].astype(
        #     int)
        real_frame_number = frame_number

        # Get video paths
        day = os.path.basename(self.parent.files['side']).split('_')[1]
        video_path = "\\".join([paths['video_folder'], day])
        video_paths = {
            "side": os.path.join(video_path, os.path.basename(self.parent.files['side']).replace(vidstuff['scorers']['side'],'').replace('.h5', '.avi')),
            "front": os.path.join(video_path, os.path.basename(self.parent.files['front']).replace(vidstuff['scorers']['front'],'').replace('.h5', '.avi')),
            "overhead": os.path.join(video_path, os.path.basename(self.parent.files['overhead']).replace(vidstuff['scorers']['overhead'], '').replace('.h5', '.avi'))
        }

        # Check if the video files exist
        for view, video_path in video_paths.items():
            if not os.path.exists(video_path):
                print(f"Video file not found for {view} view at {video_path}")
                return

        # Initialize video capture for each view
        caps = {view: cv2.VideoCapture(video_path) for view, video_path in video_paths.items()}

        # Plot original vs optimized points for each view and error curve
        fig, axs = plt.subplots(4, 1, figsize=(20, 20))

        # Flags for controlling the legend labels
        original_plotted = False
        optimized_plotted = False

        for i, (view, cap) in enumerate(caps.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_number)
            ret, frame = cap.read()

            if not ret:
                print(f"Error: Could not read frame {real_frame_number} from the video for {view} view.")
                continue

            axs[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Plot original points in red
            for label in original_points.loc(axis=1)['bodyparts'].unique():
                orig_x = original_points[np.logical_and(original_points['bodyparts'] == label, original_points['coords'] == 'x')][view]
                orig_y = original_points[np.logical_and(original_points['bodyparts'] == label, original_points['coords'] == 'y')][view]
                axs[i].scatter(orig_x, orig_y, c='r', marker='+', s=20, label='Original' if not original_plotted else "")  # Only label once
                original_plotted = True  # Set flag to True

            # Plot optimized points in blue
            for label in optimized_points.loc(axis=1)['bodyparts'].unique():
                opt_x = optimized_points[np.logical_and(optimized_points['bodyparts'] == label, optimized_points['coords'] == 'x')][view]
                opt_y = optimized_points[np.logical_and(optimized_points['bodyparts'] == label, optimized_points['coords'] == 'y')][view]
                axs[i].scatter(opt_x, opt_y, c='b', marker='+', s=20, label='Optimized' if not optimized_plotted else "")  # Only label once
                optimized_plotted = True

            axs[i].set_title(f'{view.capitalize()} View - Frame {frame_number}')
            axs[i].axis('off')

        # Add a legend for color coding in the first subplot
        axs[0].legend(loc='upper right')

        # Plot the error curve in the fourth subplot
        axs[3].plot(self.error_history, color='g', label='Total Error')
        axs[3].set_ylim([self.error_history[-1] - 200, self.error_history[0] + 200])
        axs[3].set_title("Optimization Error Curve")
        axs[3].set_xlabel("Iteration")
        axs[3].set_ylabel("Total Error")
        axs[3].legend()

        plt.tight_layout()

        # save the figure
        fig.savefig(f"{self.base_path_name}_OptimizationResults_{suffix}.png")
        fig.savefig(f"{self.base_path_name}_OptimizationResults_{suffix}.svg")
        plt.close(fig)

        # Release the video capture objects
        for cap in caps.values():
            cap.release()

    def plot_selected_frames(self, reference_points):
        """
        Plots the selected frames for each camera view with the respective reference data coordinates
        as scatter points.
        """
        reference_data = self.get_data_for_optimisation(reference_points)

        day =os.path.basename(self.parent.files['side']).split('_')[1]
        video_path = "\\".join([paths['video_folder'], day])
        video_paths = {
            "side": os.path.join(video_path, os.path.basename(self.parent.files['side']).replace(vidstuff['scorers']['side'],'').replace('.h5', '.avi')),
            "front": os.path.join(video_path, os.path.basename(self.parent.files['front']).replace(vidstuff['scorers']['front'],'').replace('.h5', '.avi')),
            "overhead": os.path.join(video_path, os.path.basename(self.parent.files['overhead']).replace(vidstuff['scorers']['overhead'],'').replace('.h5', '.avi'))
        }

        # Check if the video files exist
        for view, video_path in video_paths.items():
            if not os.path.exists(video_path):
                print(f"Video file not found for {view} view at {video_path}")
                return

        # Initialize video capture for each view
        caps = {view: cv2.VideoCapture(video_path) for view, video_path in video_paths.items()}

        for pos, frame_number in enumerate(reference_data['side'].index):
            real_frame_number = reference_data['side'].iloc(axis=0)[pos].loc['original_index'].values[0].astype(int)

            fig, axs = plt.subplots(3, 1, figsize=(20, 15))

            for i, (view, cap) in enumerate(caps.items()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_number)
                ret, frame = cap.read()

                if not ret:
                    print(f"Error: Could not read frame {real_frame_number} from the video for {view} view.")
                    continue

                reference_row = reference_data[view].loc[frame_number]

                axs[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                for label in reference_row.index.get_level_values('bodyparts').unique():
                    if label != 'original_index':
                        axs[i].scatter(reference_row.loc[label, 'x'], reference_row.loc[label, 'y'], c='r', s=10)

                axs[i].set_title(f'{view.capitalize()} View - Frame {frame_number}')
                axs[i].axis('off')

            plt.tight_layout()
            plt.show()

        # Release the video capture objects
        for cap in caps.values():
            cap.release()


    def flatten_calibration_points(self):
        """
        Flattens the calibration coordinates into a 1D numpy array for optimization.
        """
        #flat_points = self.calibration_coords[['side', 'front', 'overhead']].values.flatten()
        flat_points = self.calibration_coords[['front', 'overhead']].values.flatten()
        return np.array(flat_points, dtype=float)

    def objective_function(self, flat_points, *args):
        reference_points = args[0]  # Extract reference points from args
        frame_indices = args[1]  # Extract frame indices from args

        # Reshape the flat_points (front + overhead) back into the calibration coordinates
        calibration_points = self.reshape_calibration_points(flat_points)

        # Estimate extrinsics using all points, including the fixed side points
        temp_extrinsics = self.estimate_extrinsics(calibration_points)

        total_error, _ = self.compute_reprojection_error(reference_points, frame_indices, temp_extrinsics,
                                                         weighted=True)

        # Store the error value for plotting later
        self.error_history.append(total_error)

        # print("Total error: %s" %total_error)
        return total_error

    def estimate_extrinsics(self, calibration_points):
        """
        Estimates the camera extrinsics using the reshaped calibration coordinates DataFrame.
        """
        calibration_coordinates = calibration_points[['bodyparts', 'coords', 'side', 'front', 'overhead']]

        calib = BasicCalibration(calibration_coordinates)
        cameras_extrinsics = calib.estimate_cams_pose()
        return cameras_extrinsics

    def reshape_calibration_points(self, flat_points):
        """
        Reshapes a 1D numpy array back into the calibration coordinates DataFrame format.
        """
        reshaped_coords = self.calibration_coords.copy()
        # reshaped_coords['side'] = flat_points[0::3]
        # reshaped_coords['front'] = flat_points[1::3]
        # reshaped_coords['overhead'] = flat_points[2::3]

        # Restore the original side calibration points
        reshaped_coords['side'] = self.calibration_coords['side']

        # Use the flat_points to update the front and overhead points
        reshaped_coords['front'] = flat_points[0::2]
        reshaped_coords['overhead'] = flat_points[1::2]
        return reshaped_coords

    def recalculate_camera_parameters(self):
        """
        Recalculates camera parameters based on the updated calibration coordinates.
        """
        calibration_coordinates = self.calibration_coords[['bodyparts', 'coords', 'side', 'front', 'overhead']]

        calib = BasicCalibration(calibration_coordinates)
        cameras_extrinsics = calib.estimate_cams_pose()
        cameras_intrinsics = calib.cameras_intrinsics
        belt_points_WCS = calib.belt_coords_WCS
        belt_points_CCS = calib.belt_coords_CCS

        calibration_data = {
            'extrinsics': cameras_extrinsics,
            'intrinsics': cameras_intrinsics,
            'belt points WCS': belt_points_WCS,
            'belt points CCS': belt_points_CCS
        }
        return calibration_data

    def compute_reprojection_error(self, labels, frame_indices, extrinsics=None, weighted=False):
        errors = {label: {"side": 0, "front": 0, "overhead": 0} for label in labels}
        cams = ["side", "front", "overhead"]
        total_error = 0

        for frame_index in frame_indices:
            frame_index = int(frame_index)  # Ensure frame_index is an integer
            for label in labels:
                point_3d = self.triangulate(label, extrinsics, frame_index)
                if point_3d is not None:
                    point_3d = point_3d[:3]
                    projections = self.project_to_view(point_3d, extrinsics)

                    for view, frame in zip(cams, [frame_index, frame_index, frame_index]):
                        if self.body_part_points[frame][label][view] is not None:
                            original_x, original_y = self.body_part_points[frame][label][view]
                            if view in projections:
                                projected_x, projected_y = projections[view]
                                error = np.sqrt(
                                    (projected_x - original_x) ** 2 + (projected_y - original_y) ** 2)
                                if weighted:
                                    weight = opt_config.REFERENCE_LABEL_WEIGHTS[view].get(label, 1.0)
                                    error *= weight
                                errors[label][view] = error
                                total_error += error
        return total_error, errors


    def project_to_view(self, point_3d, extrinsics=None):
        projections = {}
        for view in ["side", "front", "overhead"]:
            if extrinsics is None:
                extrinsics = self.extrinsics
            if extrinsics[view] is not None:
                CCS_repr, _ = cv2.projectPoints(
                    point_3d,
                    cv2.Rodrigues(extrinsics[view]['rotm'])[0],
                    extrinsics[view]['tvec'],
                    self.intrinsics[view],
                    np.array([]),
                )
                projections[view] = CCS_repr[0].flatten()
        return projections

    def triangulate(self, label, extrinsics=None, frame_index=None):
        P = []
        coords = []

        frame_mapping = {'side': frame_index, 'front': frame_index, 'overhead': frame_index}
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

    def get_p(self, view, extrinsics=None, return_value=False):
        if extrinsics is None:
            extrinsics = self.extrinsics

        # Camera intrinsics
        K = self.intrinsics[view]

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

    def get_data_for_optimisation(self, reference_points):
        self.DataframeCoors = self.parent.DataframeCoors

        visibility_mask = {
            'side': self.DataframeCoors['side'].loc(axis=1)[reference_points, 'likelihood'] > pcutoff,
            'front': self.DataframeCoors['front'].loc(axis=1)[reference_points, 'likelihood'] > pcutoff,
            'overhead': self.DataframeCoors['overhead'].loc(axis=1)[reference_points, 'likelihood'] > pcutoff
        }
        # Initialize combined visibility DataFrame with all True
        combined_visibility = pd.DataFrame(True, index=visibility_mask[next(iter(visibility_mask))].index,
                                           columns=visibility_mask[next(iter(visibility_mask))].columns.levels[0])

        # Combine visibility across all views
        for view_mask in visibility_mask.values():
            combined_visibility = combined_visibility & view_mask

        data = self.DataframeCoors

        visible_counts = combined_visibility.sum(axis=1)
        ordered_frames = visible_counts.sort_values(ascending=False)
        # mask of more than 6 visible points
        high_visibility_frames = ordered_frames[visible_counts > 9]
        visible_index = high_visibility_frames.index[:int(len(high_visibility_frames.index) * 0.5)]
        data_visible = {view: data[view].loc(axis=0)[visible_index] for view in data.keys()}

        # order dfs by side's x, select random 10 frames, then order by side's y, select random 10 frames
        nose_mask = self.create_combined_visibility_mask(data_visible, 'Nose', pcutoff) #0.95
        tail12_mask = self.create_combined_visibility_mask(data_visible, 'Tail12', pcutoff) #0.95
        hindpaw_toeR_mask = self.create_combined_visibility_mask(data_visible, 'HindpawToeR', pcutoff) #0.95
        earR_mask = self.create_combined_visibility_mask(data_visible, 'EarR', pcutoff) #0.95

        indexes = []
        indexes.append(
            self.get_index_snapshots(data_visible['side'].loc(axis=1)['Nose', 'x'][nose_mask].sort_values().index,
                                     [0.1, 0.5, 0.99, 0.999]))
        indexes.append(
            self.get_index_snapshots(data_visible['side'].loc(axis=1)['Nose', 'y'][nose_mask].sort_values().index,
                                     [0.1, 0.5, 0.9, 0.99]))
        indexes.append(
            self.get_index_snapshots(data_visible['side'].loc(axis=1)['Tail12', 'y'][tail12_mask].sort_values().index,
                                     [0.1, 0.2, 0.85, 0.95, 0.99]))
        indexes.append(
            self.get_index_snapshots(data_visible['front'].loc(axis=1)['Nose', 'x'][nose_mask].sort_values().index,
                                     [0.01, 0.3, 0.5, 0.7, 0.99]))
        indexes.append(
            self.get_index_snapshots(data_visible['front'].loc(axis=1)['Tail12', 'x'][tail12_mask].sort_values().index,
                                     [0.01, 0.3, 0.99]))
        indexes.append(self.get_index_snapshots(
            data_visible['front'].loc(axis=1)['HindpawToeR', 'x'][hindpaw_toeR_mask].sort_values().index,
            [0.01, 0.99]))
        indexes.append(self.get_index_snapshots(
            data_visible['front'].loc(axis=1)['HindpawToeR', 'y'][hindpaw_toeR_mask].sort_values().index,
            [0.01, 0.99]))
        indexes.append(
            self.get_index_snapshots(data_visible['side'].loc(axis=1)['EarR', 'y'][earR_mask].sort_values().index,
                                     [0.01, 0.99]))

        # check for door positions too
        door_mask = np.logical_and.reduce((data['side'].loc(axis=1)['Door', 'likelihood'] > pcutoff,
                                           data['front'].loc(axis=1)['Door', 'likelihood'] > pcutoff,
                                           data['overhead'].loc(axis=1)['Door', 'likelihood'] > pcutoff))
        indexes.append(self.get_index_snapshots(data['side'].loc(axis=1)['Door', 'y'][door_mask].sort_values().index,
                                                [0.001, 0.005, 0.01, 0.9, 0.95]))

        # remove any duplicates
        flattened_list = [item for sublist in indexes for item in sublist]
        unique_items = set(flattened_list)
        unique_list = list(unique_items)

        reference_data = {view: data[view].loc(axis=0)[unique_list].loc(axis=1)[reference_points + ['Door']] for view in
                          data.keys()}
        for view in reference_data.keys():
            mask = reference_data[view].xs(axis=1, level=1, key='likelihood') > pcutoff #0.99
            reference_data[view] = reference_data[view][mask]
            reference_data[view].drop('likelihood', axis=1, level=1, inplace=True)
            reference_data[view].loc(axis=1)['original_index'] = self.DataframeCoors[view].loc(axis=0)[unique_list].loc(axis=1)['original_index'].values

        return reference_data

    def show_side_view_frames(self, frames, reference_data):
        """
        Displays all frames extracted for the side view.
        """

        # frames = reference_data['side'].index
        video_path = r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230306\HM_20230306_APACharRepeat_FAA-1035243_None_side_1.avi"
        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found at {video_path}")
        else:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)

            # Iterate over the frames
            for frame_number in frames:
                reference_row = reference_data['side'].loc(axis=0)[frame_number]
                mask = reference_row['likelihood'] > 0.95
                reference_row = reference_row[mask]

                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                # Check if the frame is captured correctly
                if not ret:
                    print(f"Error: Could not read frame {frame_number} from the video.")
                    continue  # Skip to the next frame if the current frame is not read correctly

                # Display the frame using Matplotlib
                plt.figure(figsize=(10, 6))
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.scatter(reference_row['x'], reference_row['y'], c='r', s=1)
                plt.title(f'Side View - Frame {frame_number}')
                plt.axis('off')
                plt.show()

            # Release the video capture object
            cap.release()

    def create_combined_visibility_mask(self, data_visible, bodypart, pcutoff):
        side_mask = data_visible['side'].loc(axis=1)[bodypart, 'likelihood'] > pcutoff
        front_mask = data_visible['front'].loc(axis=1)[bodypart, 'likelihood'] > pcutoff
        overhead_mask = data_visible['overhead'].loc(axis=1)[bodypart, 'likelihood'] > pcutoff

        # Check visibility in at least two views (side & front, side & overhead, overhead & front) or all three
        combined_mask = (side_mask & front_mask) | (side_mask & overhead_mask) | (overhead_mask & front_mask)
        return combined_mask

    def get_index_snapshots(self, frames, positions):
        total_frames = len(frames)
        indexes = []
        for p in positions:
            idx = frames[int(total_frames * p)]
            indexes.append(idx)
        return indexes
