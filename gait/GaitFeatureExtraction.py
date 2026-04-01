"""Extracts kinematic features from labelled frames for gait classifier training."""
import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import time
from tqdm import tqdm  # Added for progress bar
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from helpers.config import *

class FeatureExtractor:
    def __init__(self, data, fps):
        """
        Initialize the FeatureExtractor with data as a DataFrame.
        """
        self.data = data
        self.fps = fps  # Frames per second of the video/data
        self.features_df = None

    def model_label_to_data_label(self, model_label):
        parts = model_label.split('_')
        if len(parts) == 2:
            paw_with_side, joint = parts
            if paw_with_side.endswith('R') or paw_with_side.endswith('L'):
                paw = paw_with_side[:-1]  # 'Forepaw' or 'Hindpaw'
                side = paw_with_side[-1]  # 'R' or 'L'
                data_label = f"{paw}{joint}{side}"
                return data_label
            else:
                return model_label  # Unexpected format
        else:
            return model_label  # Return as is if unexpected format

    def extract_features(self, frames_to_process):
        joints = ['Toe', 'Knuckle', 'Ankle', 'Knee']
        paws = ['ForepawR', 'ForepawL', 'HindpawR', 'HindpawL']
        coords = ['x', 'z']
        time_offsets = np.array([-10, -5, -1, 0, 1, 5, 10])
        delta_t = 1 / self.fps  # Time difference between consecutive frames
        frames = np.array(frames_to_process)
        indices = self.data.index.values  # All frame indices in the data

        # Create a grid of frames and offsets
        frame_grid, offset_grid = np.meshgrid(frames, time_offsets, indexing='ij')
        t_grid = frame_grid + offset_grid  # Shape: (num_frames, num_offsets)

        # Flatten t_grid for easy indexing
        t_flat = t_grid.flatten()

        # Create t_minus and t_plus grids for velocity calculations
        t_minus_flat = t_flat - 1
        t_plus_flat = t_flat + 1

        # Identify valid indices within the data index range
        valid_indices = self.data.index.values
        data_index_min = valid_indices[0]
        data_index_max = valid_indices[-1]

        # Create masks for valid indices
        t_flat_mask = (t_flat >= data_index_min) & (t_flat <= data_index_max)
        t_minus_flat_mask = (t_minus_flat >= data_index_min) & (t_minus_flat <= data_index_max)
        t_plus_flat_mask = (t_plus_flat >= data_index_min) & (t_plus_flat <= data_index_max)

        # Prepare lists of body parts and columns
        bodyparts = ['Nose', 'Tail1', 'Tail12']
        body_data_coords = [(bp, coord) for bp in bodyparts for coord in ['x']]
        body_feature_coords = body_data_coords  # Assuming labels are the same

        # Prepare paw coordinates
        paw_data_coords = []
        paw_feature_coords = []
        for paw_name in paws:
            for joint in joints:
                model_label = f"{paw_name}_{joint}"  # e.g., 'ForepawR_Toe'
                data_label = self.model_label_to_data_label(model_label)  # e.g., 'ForepawToeR'
                for coord in coords:
                    paw_data_coords.append((data_label, coord))
                    paw_feature_coords.append((model_label, coord))

        # Combine all required bodypart and paw coordinates
        all_data_coords = body_data_coords + paw_data_coords
        all_feature_coords = body_feature_coords + paw_feature_coords

        # Collect all unique times required
        all_times = np.unique(np.concatenate([t_flat, t_minus_flat, t_plus_flat]))
        all_times = all_times[(all_times >= data_index_min) & (all_times <= data_index_max)]

        # Reindex data to include all required times, filling missing times with NaN
        data_reindexed = self.data.reindex(all_times, fill_value=np.nan)

        # Extract positions using data labels
        position_data = data_reindexed.loc[all_times, all_data_coords]

        # Initialize position arrays with NaN
        position_t = pd.DataFrame(np.nan, index=range(len(t_flat)), columns=position_data.columns)
        position_t_minus = pd.DataFrame(np.nan, index=range(len(t_minus_flat)), columns=position_data.columns)
        position_t_plus = pd.DataFrame(np.nan, index=range(len(t_plus_flat)), columns=position_data.columns)

        # Fill valid positions for t_flat
        valid_t_flat_indices = np.where(t_flat_mask)[0]
        valid_t_flat_times = t_flat[valid_t_flat_indices].astype(int)
        position_t.iloc[valid_t_flat_indices] = position_data.loc[valid_t_flat_times].values

        # Fill valid positions for t_minus_flat
        valid_t_minus_indices = np.where(t_minus_flat_mask)[0]
        valid_t_minus_times = t_minus_flat[valid_t_minus_indices].astype(int)
        position_t_minus.iloc[valid_t_minus_indices] = position_data.loc[valid_t_minus_times].values

        # Fill valid positions for t_plus_flat
        valid_t_plus_indices = np.where(t_plus_flat_mask)[0]
        valid_t_plus_times = t_plus_flat[valid_t_plus_indices].astype(int)
        position_t_plus.iloc[valid_t_plus_indices] = position_data.loc[valid_t_plus_times].values

        # Compute velocities
        velocity = (position_t_plus.values - position_t_minus.values) / (2 * delta_t)

        # Reshape position and velocity data back to (num_frames, num_offsets, num_features)
        num_frames = len(frames)
        num_offsets = len(time_offsets)
        num_features = position_t.shape[1]
        position_t_reshaped = position_t.values.reshape(num_frames, num_offsets, num_features)
        velocity_reshaped = velocity.reshape(num_frames, num_offsets, num_features)

        # Prepare a dictionary to collect all features
        features_dict = {}

        # Collect position and velocity features
        for feature_idx, (model_label, coord) in enumerate(all_feature_coords):
            for offset_idx, offset in enumerate(time_offsets):
                # Position features
                position_feature_name = f"{model_label}_{coord}_t{offset}"
                position_values = position_t_reshaped[:, offset_idx, feature_idx]
                features_dict[position_feature_name] = position_values

                # Velocity features (for appropriate body parts)
                if (model_label != 'Tail12' and coord == 'x') or model_label not in bodyparts:
                    velocity_feature_name = f"{model_label}_{coord}_velocity_t{offset}"
                    velocity_values = velocity_reshaped[:, offset_idx, feature_idx]
                    features_dict[velocity_feature_name] = velocity_values

        # Collect angle features
        for paw_name in paws:
            for angle_joints in [('Toe', 'Knuckle', 'Ankle'), ('Knuckle', 'Ankle', 'Knee')]:
                angle_feature_base_name = f"{paw_name}_{angle_joints[0]}_{angle_joints[1]}_{angle_joints[2]}_angle"
                for offset_idx, offset in enumerate(time_offsets):
                    angle_feature_name = f"{angle_feature_base_name}_t{offset}"
                    # Initialize a full array with NaNs
                    angle_feature = np.full(num_frames, np.nan)
                    t_indices = t_grid[:, offset_idx]
                    valid_mask = (t_indices >= data_index_min) & (t_indices <= data_index_max)
                    t_valid = t_indices[valid_mask].astype(int)

                    # Extract coordinates for joints
                    model_joint1_label = f"{paw_name}_{angle_joints[0]}"  # e.g., 'ForepawR_Toe'
                    model_joint2_label = f"{paw_name}_{angle_joints[1]}"
                    model_joint3_label = f"{paw_name}_{angle_joints[2]}"
                    data_joint1_label = self.model_label_to_data_label(model_joint1_label)
                    data_joint2_label = self.model_label_to_data_label(model_joint2_label)
                    data_joint3_label = self.model_label_to_data_label(model_joint3_label)

                    coords_needed = [
                        (data_joint1_label, 'x'), (data_joint1_label, 'z'),
                        (data_joint2_label, 'x'), (data_joint2_label, 'z'),
                        (data_joint3_label, 'x'), (data_joint3_label, 'z')
                    ]

                    # Check if all required data labels exist
                    if all([label in data_reindexed.columns.get_level_values(0) for label in
                            [data_joint1_label, data_joint2_label, data_joint3_label]]):
                        # Extract data for valid times
                        angle_data = data_reindexed.loc[t_valid, coords_needed].values.reshape(-1, 3, 2)
                        # Compute angles, handling NaNs
                        angles = self.calculate_angle_vectorized(angle_data[:, 0, :], angle_data[:, 1, :],
                                                                 angle_data[:, 2, :])
                        angle_feature[valid_mask] = angles
                    else:
                        # If any data label is missing, angle_feature remains NaN
                        pass

                    # Add angle feature to the dictionary
                    features_dict[angle_feature_name] = angle_feature

        # Create features_df from the features_dict with index name 'Frame'
        features_df = pd.DataFrame(features_dict, index=frames)
        features_df.index.name = 'Frame'
        self.features_df = features_df

    # def extract_features(self, frames_to_process):
    #     # Define joints and paws
    #     joints = ['Toe', 'Knuckle', 'Ankle', 'Knee']
    #     paws = {
    #         'ForepawR': {'Toe': 'ForepawToeR', 'Knuckle': 'ForepawKnuckleR', 'Ankle': 'ForepawAnkleR',
    #                      'Knee': 'ForepawKneeR'},
    #         'ForepawL': {'Toe': 'ForepawToeL', 'Knuckle': 'ForepawKnuckleL', 'Ankle': 'ForepawAnkleL',
    #                      'Knee': 'ForepawKneeL'},
    #         'HindpawR': {'Toe': 'HindpawToeR', 'Knuckle': 'HindpawKnuckleR', 'Ankle': 'HindpawAnkleR',
    #                      'Knee': 'HindpawKneeR'},
    #         'HindpawL': {'Toe': 'HindpawToeL', 'Knuckle': 'HindpawKnuckleL', 'Ankle': 'HindpawAnkleL',
    #                      'Knee': 'HindpawKneeL'}
    #     }
    #     coords = ['x', 'z']
    #     time_offsets = [-10, -5, 0, 5, 10] #todo added in extra time offsets
    #     delta_t = 1 / self.fps  # Time difference between consecutive frames
    #
    #     features_list = []
    #     indices = self.data.index
    #
    #     for frame in frames_to_process:
    #         frame_features = {'Frame': frame}
    #         for offset in time_offsets:
    #             t = frame + offset
    #             if t in indices:
    #                 # For velocity calculation at time t + offset
    #                 t_minus = t - 1
    #                 t_plus = t + 1
    #                 for bodypart in ['Nose', 'Tail1', 'Tail12']:
    #                     # get x velocity
    #                     if bodypart != 'Tail12':
    #                         if t_minus in indices and t_plus in indices:
    #                             x_minus = self.data.loc[t_minus, (bodypart, 'x')]
    #                             x_plus = self.data.loc[t_plus, (bodypart, 'x')]
    #                             x_velocity = (x_plus - x_minus) / (2 * delta_t)
    #                             x_velocity_feature_name = f"{bodypart}_x_velocity_t{offset}"
    #                             frame_features[x_velocity_feature_name] = x_velocity
    #                     # get x position #todo added this too
    #                     if t in indices:
    #                         x_pos = self.data.loc[t, (bodypart, 'x')]
    #                         x_pos_feature_name = f"{bodypart}_x_t{offset}"
    #                         frame_features[x_pos_feature_name] = x_pos
    #
    #                 for paw_name, paw_joints in paws.items():
    #                     for joint in joints:
    #                         joint_label = paw_joints.get(joint)
    #                         if joint_label is None:
    #                             continue  # Skip if joint is not available
    #
    #                         for coord in coords:
    #                             # Position features at time t + offset
    #                             pos = self.data.loc[t, (joint_label, coord)]
    #                             feature_name = f"{paw_name}_{joint}_{coord}_t{offset}"
    #                             frame_features[feature_name] = pos
    #
    #                             # Velocity features at time t + offset
    #                             if t_minus in indices and t_plus in indices:
    #                                 pos_minus = self.data.loc[t_minus, (joint_label, coord)]
    #                                 pos_plus = self.data.loc[t_plus, (joint_label, coord)]
    #                                 velocity = (pos_plus - pos_minus) / (2 * delta_t)
    #                                 velocity_feature_name = f"{paw_name}_{joint}_{coord}_velocity_t{offset}"
    #                                 frame_features[velocity_feature_name] = velocity
    #                             else:
    #                                 # Assign NaN if neighboring frames are not available
    #                                 velocity_feature_name = f"{paw_name}_{joint}_{coord}_velocity_t{offset}"
    #                                 frame_features[velocity_feature_name] = np.nan
    #
    #                     # Angle features at time t + offset
    #                     for angle_joints in [('Toe', 'Knuckle', 'Ankle'), ('Knuckle', 'Ankle', 'Knee')]:
    #                         joint1_label = paw_joints.get(angle_joints[0])
    #                         joint2_label = paw_joints.get(angle_joints[1])
    #                         joint3_label = paw_joints.get(angle_joints[2])
    #
    #                         if joint1_label and joint2_label and joint3_label:
    #                             coord1 = self.data.loc[t, (joint1_label, ['x', 'z'])].values.astype(float)
    #                             coord2 = self.data.loc[t, (joint2_label, ['x', 'z'])].values.astype(float)
    #                             coord3 = self.data.loc[t, (joint3_label, ['x', 'z'])].values.astype(float)
    #
    #                             angle = self.calculate_angle(coord1, coord2, coord3)
    #                             angle_feature_name = f"{paw_name}_{angle_joints[0]}_{angle_joints[1]}_{angle_joints[2]}_angle_t{offset}"
    #                             frame_features[angle_feature_name] = angle
    #                         else:
    #                             angle_feature_name = f"{paw_name}_{angle_joints[0]}_{angle_joints[1]}_{angle_joints[2]}_angle_t{offset}"
    #                             frame_features[angle_feature_name] = np.nan
    #             else:
    #                 # Assign NaN for all features at time t + offset if t + offset is out of bounds
    #                 for bodypart in ['Nose', 'Tail1', 'Tail12']:
    #                     x_velocity_feature_name = f"{bodypart}_x_velocity_t{offset}"
    #                     frame_features[x_velocity_feature_name] = np.nan
    #                 for paw_name, paw_joints in paws.items():
    #                     for joint in joints:
    #                         if paw_joints.get(joint) is None:
    #                             continue
    #                         for coord in coords:
    #                             feature_name = f"{paw_name}_{joint}_{coord}_t{offset}"
    #                             frame_features[feature_name] = np.nan
    #                             velocity_feature_name = f"{paw_name}_{joint}_{coord}_velocity_t{offset}"
    #                             frame_features[velocity_feature_name] = np.nan
    #                         for angle_joints in [('Toe', 'Knuckle', 'Ankle'), ('Knuckle', 'Ankle', 'Knee')]:
    #                             angle_feature_name = f"{paw_name}_{angle_joints[0]}_{angle_joints[1]}_{angle_joints[2]}_angle_t{offset}"
    #                             frame_features[angle_feature_name] = np.nan
    #
    #         features_list.append(frame_features)
    #
    #     # Convert list of dicts to DataFrame
    #     self.features_df = pd.DataFrame(features_list).set_index('Frame')

    def calculate_angle_vectorized(self, point1_array, point2_array, point3_array):
        # point1_array, point2_array, point3_array are arrays of shape (N, 2)
        vector1 = point1_array - point2_array
        vector2 = point3_array - point2_array
        dot_product = np.einsum('ij,ij->i', vector1, vector2)
        norm_product = np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2, axis=1)
        # Avoid division by zero
        cos_theta = np.divide(dot_product, norm_product, out=np.full_like(dot_product, np.nan), where=norm_product != 0)
        # Clip values to avoid invalid values due to numerical errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_theta))
        return angles

    # def calculate_angle(self, point1, point2, point3):
    #     # Calculate angle at point2 between point1 and point3
    #     vector1 = point1 - point2
    #     vector2 = point3 - point2
    #     norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    #     if norm_product == 0:
    #         return np.nan
    #     cos_theta = np.dot(vector1, vector2) / norm_product
    #     angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    #     return np.degrees(angle)


class RunClassifier:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.limb_data = self.load_limb_data()
        self.coordinate_data = self.load_coordinate_data()
        smoothed_data_path = os.path.join(self.base_dir, 'smoothed_data.h5')
        if os.path.exists(smoothed_data_path):
            print(f"Loading smoothed data from {smoothed_data_path}")
            self.smoothed_coordinate_data = pd.read_hdf(smoothed_data_path, key='smoothed_data')
        else:
            print("Smoothed data not found. Computing smoothed data...")
            self.smoothed_coordinate_data = self.smooth_data()
            # Save the smoothed data
            self.smoothed_coordinate_data.to_hdf(smoothed_data_path, key='smoothed_data')
            print(f"Smoothed data saved to {smoothed_data_path}")
        self.features_df = None  # Will store the final features

    def load_limb_data(self):
        data_path = os.path.join(self.base_dir, "limb_labels.csv")
        data = pd.read_csv(data_path)
        return data

    def load_coordinate_data(self):
        subdirs = self.limb_data['Subdirectory'].unique()
        subdirs = [subdir.replace('_side_1', '') for subdir in subdirs]

        coords = []
        for subdir in subdirs:
            filename = subdir + '_mapped3D.h5'
            filepath = os.path.join(self.base_dir, "data", filename)
            if os.path.exists(filepath):
                crd_data = pd.read_hdf(filepath, key='real_world_coords')
                limb_crds = crd_data[['ForepawToeR', 'ForepawKnuckleR', 'ForepawAnkleR', 'ForepawKneeR',
                                      'ForepawToeL', 'ForepawKnuckleL', 'ForepawAnkleL', 'ForepawKneeL',
                                      'HindpawToeR', 'HindpawKnuckleR', 'HindpawAnkleR', 'HindpawKneeR',
                                      'HindpawToeL', 'HindpawKnuckleL', 'HindpawAnkleL', 'HindpawKneeL',
                                      'Nose','Tail1','Tail12']].copy()
                # Remove 'y' coordinates
                limb_crds = limb_crds.drop('y', axis=1, level=1)
                # Add filename as an index, preserving the original index as FrameIdx
                limb_crds.loc[:, 'Filename'] = filename
                limb_crds.loc[:, 'FrameIdx'] = limb_crds.index
                limb_crds.set_index(['Filename', 'FrameIdx'], inplace=True)
                # Name columns with bodyparts and coordinates
                limb_crds.columns = pd.MultiIndex.from_tuples([(col[0], col[1]) for col in limb_crds.columns],
                                                              names=['bodyparts', 'coords'])
                coords.append(limb_crds)
            else:
                print(f"Error: File {filename} does not exist.")
        if len(coords) > 0:
            flat_coords = pd.concat(coords)
            return flat_coords
        else:
            raise ValueError("Error: No coordinate files found.")

    def smooth_data(self):
        # Start the timer
        start_time = time.time()
        print("Starting data smoothing...")

        # Get the unique limbparts and coords
        limbparts = self.coordinate_data.columns.get_level_values('bodyparts').unique()
        coords = self.coordinate_data.columns.get_level_values('coords').unique()

        # Create an empty list to hold smoothed data for each group
        smoothed_data_list = []

        # Reset index to have 'Filename' and 'FrameIdx' as columns
        coordinate_data_reset = self.coordinate_data.reset_index()

        total_files = coordinate_data_reset['Filename'].nunique()
        file_counter = 0

        # Process each 'Filename' group separately
        for filename, group in coordinate_data_reset.groupby('Filename'):
            file_counter += 1
            print(f"Processing file {file_counter}/{total_files}: {filename}")

            # Start timing for this file
            file_start_time = time.time()

            # Set 'FrameIdx' as the index
            group = group.set_index('FrameIdx')
            # Drop 'Filename' as it's now constant for this group
            group = group.drop(columns='Filename', level=0)

            total_limbparts = len(limbparts)
            limbpart_counter = 0

            # Interpolate and smooth each limbpart and coord
            for limbpart in limbparts:
                limbpart_counter += 1
                #print(f"  Processing limbpart {limbpart_counter}/{total_limbparts}: {limbpart}")

                for coord in coords:
                    limbpart_coords = group[(limbpart, coord)].copy()

                    # Check if the Series is not all NaNs
                    if limbpart_coords.notnull().any():
                        # Interpolate missing values
                        interpolated = limbpart_coords.interpolate(
                            method='spline',
                            order=3,
                            limit=20,
                            limit_direction='both'
                        )

                        # Apply Gaussian smoothing
                        smoothed = gaussian_filter1d(interpolated.values, sigma=2)

                        # Assign back to group
                        group.loc[:, (limbpart, coord)] = smoothed
                    else:
                        # If all values are NaN, keep them as NaN
                        group.loc[:, (limbpart, coord)] = np.nan

            # End timing for this file
            file_end_time = time.time()
            file_duration = file_end_time - file_start_time
            print(f"Finished processing {filename} in {file_duration:.2f} seconds.")

            # Add 'Filename' back as a column
            group['Filename'] = filename
            # Reset index to include 'FrameIdx'
            group.reset_index(inplace=True)
            # Set index to ['Filename', 'FrameIdx']
            group.set_index(['Filename', 'FrameIdx'], inplace=True)
            # Append the group to the list
            smoothed_data_list.append(group)

        # Concatenate all smoothed groups
        smoothed_data = pd.concat(smoothed_data_list)
        # Ensure that columns are in the same order as self.coordinate_data.columns
        smoothed_data = smoothed_data.loc[:, self.coordinate_data.columns]

        # End the timer
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"Data smoothing completed in {total_duration:.2f} seconds.")

        return smoothed_data

    def collect_features(self):
        print("Starting feature extraction...")
        # Create an empty list to store features from all files
        features_list = []

        # Ensure that 'Subdirectory' in limb_data matches 'Filename' in smoothed_coordinate_data
        # Map 'Subdirectory' to 'Filename'
        self.limb_data['Filename'] = self.limb_data['Subdirectory'].apply(
            lambda x: x.replace('_side_1', '') + '_mapped3D.h5')

        # Process each unique 'Filename' with progress bar
        filenames = self.limb_data['Filename'].unique()
        for filename in tqdm(filenames, desc="Collecting features"):
            print(f"Processing file: {filename}")
            # Get frames to process for this filename
            limb_data_subset = self.limb_data[self.limb_data['Filename'] == filename].copy()
            frames_to_process = limb_data_subset['Frame'].unique()
            frames_to_process = frames_to_process.astype(int).tolist()

            # Get the data for this filename
            try:
                data = self.smoothed_coordinate_data.xs(filename, level='Filename')
            except KeyError:
                print(f"Warning: Data for {filename} not found in smoothed_coordinate_data.")
                continue  # Skip this filename if data not found

            # Create FeatureExtractor with data and actual fps
            feature_extractor = FeatureExtractor(data=data, fps=247)
            # Extract features
            feature_extractor.extract_features(frames_to_process)
            # Get the features DataFrame
            features_df = feature_extractor.features_df
            # Add 'Filename' as a column
            features_df['Filename'] = filename
            # Merge with limb_data to get labels or additional info
            # Ensure 'Frame' is an integer
            features_df.reset_index(inplace=True)
            features_df['Frame'] = features_df['Frame'].astype(int)
            limb_data_subset['Frame'] = limb_data_subset['Frame'].astype(int)
            # Merge on ['Filename', 'Frame']
            merged_df = pd.merge(features_df, limb_data_subset, on=['Filename', 'Frame'], how='left')
            # remove 'Subdirectory' column # todo ?????
            # if 'Subdirectory' in merged_df.columns:
            #     merged_df = merged_df.drop(columns='Subdirectory')
            # Append to the list
            features_list.append(merged_df)

        # Concatenate all features
        if features_list:
            self.features_df = pd.concat(features_list, ignore_index=True)
            # Set index to ['Filename', 'Frame']
            self.features_df.set_index(['Filename', 'Frame'], inplace=True)
            print("Feature extraction completed.")
        else:
            print("No features were extracted.")

    def save_features(self, output_path):
        if self.features_df is not None:
            # Drop the 'Subdirectory' column if it exists
            if 'Subdirectory' in self.features_df.columns:
                self.features_df = self.features_df.drop(columns='Subdirectory')
            # Save to CSV (MultiIndex will be saved as columns)
            self.features_df.to_csv(output_path)
            print(f"Features saved to {output_path}")
        else:
            print("No features to save.")


def main():
    base_directory = r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round4_Oct24\LimbStuff"
    classifier = RunClassifier(base_directory)
    classifier.collect_features()
    # Save features to a CSV file
    output_features_path = os.path.join(base_directory, 'extracted_features.csv')
    classifier.save_features(output_features_path)


if __name__ == '__main__':
    main()
