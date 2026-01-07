import os, cv2, sys, re, glob
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pycalib.calib import triangulate, triangulate_Npts
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count, shared_memory
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(r'C:\Users\hmorl\Projects\VideoAnalysis_Aug2022')

from Helpers.utils_3d_reconstruction import CameraData, BeltPoints
from Helpers.CalibrateCams import BasicCalibration
from Helpers import utils
from Helpers.Config_23 import *
from Helpers import OptimizeCalibration
from Helpers.ConditionsFinder import BaseConditionFiles

########################################################################################################################
# Prepare the data for mapping in parallel
########################################################################################################################
def process_part_wrapper(args):
    """
    Wrapper function for parallel processing of body parts.
    :param args:
    :return:
    """
    try:
        (bidx, body_part), shared_data = args

        # Reconstruct shared arrays
        side_coords_shm = shared_memory.SharedMemory(name=shared_data['side_coords_name'])
        side_coords = np.ndarray(shared_data['side_coords_shape'], dtype=shared_data['side_coords_dtype'], buffer=side_coords_shm.buf)

        front_coords_shm = shared_memory.SharedMemory(name=shared_data['front_coords_name'])
        front_coords = np.ndarray(shared_data['front_coords_shape'], dtype=shared_data['front_coords_dtype'], buffer=front_coords_shm.buf)

        overhead_coords_shm = shared_memory.SharedMemory(name=shared_data['overhead_coords_name'])
        overhead_coords = np.ndarray(shared_data['overhead_coords_shape'], dtype=shared_data['overhead_coords_dtype'], buffer=overhead_coords_shm.buf)

        # Extract other data
        cameras_extrinsics = shared_data['cameras_extrinsics']
        cameras_intrinsics = shared_data['cameras_intrinsics']
        P_gt = shared_data['P_gt']
        Nc = shared_data['Nc']

        # Process the body part
        result_data = process_body_part(
            bidx, body_part,
            side_coords, front_coords, overhead_coords,
            cameras_extrinsics, cameras_intrinsics, P_gt, Nc
        )

        # Close shared memory
        side_coords_shm.close()
        front_coords_shm.close()
        overhead_coords_shm.close()

        return (body_part, result_data)
    except Exception as e:
        print(f"Error processing {args[0][1]}: {e}")
        return (args[0][1], None)

def process_body_part(bidx, body_part, side_coords, front_coords, overhead_coords,
                      cameras_extrinsics, cameras_intrinsics, P_gt, Nc):
    coords_2d_all, likelihoods = get_coords_and_likelihoods(
        bidx, side_coords, front_coords, overhead_coords)

    coords_2d, likelihoods, P_gt_bp, Nc_bp, empty_cameras = find_empty_cameras(
        coords_2d_all, likelihoods, P_gt, Nc)

    result = triangulate_points(
        coords_2d, likelihoods, Nc_bp, P_gt_bp,
        cameras_extrinsics, cameras_intrinsics, coords_2d_all
    )

    return result  # Return the result dictionary


def get_coords_and_likelihoods(bidx, side_coords, front_coords, overhead_coords):
    coords_2d_all = np.array([side_coords[:, bidx, :2], front_coords[:, bidx, :2], overhead_coords[:, bidx, :2]])
    likelihoods = np.array([side_coords[:, bidx, 2], front_coords[:, bidx, 2], overhead_coords[:, bidx, 2]])
    return coords_2d_all, likelihoods

def find_empty_cameras(coords_2d, likelihoods, P_gt, Nc):
    empty_cameras = np.where(np.all(np.all(np.isnan(coords_2d), axis=2), axis=1))
    if len(empty_cameras) > 0:
        coords_2d = np.delete(coords_2d, empty_cameras, axis=0)
        likelihoods = np.delete(likelihoods, empty_cameras, axis=0)
        Nc_bp = len(coords_2d)
        P_gt_bp = np.delete(P_gt, empty_cameras, axis=0)
    else:
        Nc_bp = Nc
        P_gt_bp = P_gt
    return coords_2d, likelihoods, P_gt_bp, Nc_bp, empty_cameras

def triangulate_points(coords_2d, likelihoods, Nc_bp, P_gt_bp, cameras_extrinsics,
                       cameras_intrinsics, coords_2d_all):
    real_world_coords = []
    side_err = []
    front_err = []
    overhead_err = []
    side_repr = []
    front_repr = []
    overhead_repr = []

    triangulation_data = prepare_triangulation_data(coords_2d, likelihoods, P_gt_bp, Nc_bp)

    # Initialize empty array for real-world coordinates with NaNs for missing frames
    Np = coords_2d.shape[1]  # Number of points to triangulate
    real_world_coords = np.full((Np, 3), np.nan)

    # Perform triangulation for each camera pair and 3-camera case
    real_world_coords_sidexfront = triangulate_for_pair(triangulation_data['coords_sidexfront_batch'],
                                                        triangulation_data['P_sidexfront_batch'])

    real_world_coords_frontxoverhead = triangulate_for_pair(triangulation_data['coords_frontxoverhead_batch'],
                                                            triangulation_data['P_frontxoverhead_batch'])

    real_world_coords_sidexoverhead = triangulate_for_pair(triangulation_data['coords_sidexoverhead_batch'],
                                                           triangulation_data['P_sidexoverhead_batch'])

    real_world_coords_3cam = triangulate_for_pair(triangulation_data['coords_2d_3cam_batch'],
                                                  triangulation_data['P_matrices_3cam_batch'])

    # Reinsert the triangulated coordinates back to the original frame order
    reassemble_in_original_order(real_world_coords, real_world_coords_sidexfront,
                                 triangulation_data['triangulated_indices_sidexfront'])

    reassemble_in_original_order(real_world_coords, real_world_coords_frontxoverhead,
                                 triangulation_data['triangulated_indices_frontxoverhead'])

    reassemble_in_original_order(real_world_coords, real_world_coords_sidexoverhead,
                                 triangulation_data['triangulated_indices_sidexoverhead'])

    reassemble_in_original_order(real_world_coords, real_world_coords_3cam,
                                 triangulation_data['triangulated_indices_3cam'])

    # Project back to cameras and calculate reprojection errors
    project_back_to_cameras_vectorized(real_world_coords, cameras_extrinsics, cameras_intrinsics,
                                       coords_2d_all,
                                       side_err, front_err, overhead_err,
                                       side_repr, front_repr, overhead_repr)

    # Finalize the coordinates with reprojection data
    (real_world_coords_final, side_repr_arr, front_repr_arr, overhead_repr_arr,
     side_err_arr, front_err_arr, overhead_err_arr) = finalize_coords(
        real_world_coords, side_repr, front_repr, overhead_repr,
        side_err, front_err, overhead_err)

    # Prepare the result dictionary
    result = {
        'real_world_coords': real_world_coords_final,
        'side_repr': side_repr_arr,
        'front_repr': front_repr_arr,
        'overhead_repr': overhead_repr_arr,
        'side_err': side_err_arr,
        'front_err': front_err_arr,
        'overhead_err': overhead_err_arr,
    }

    return result

def prepare_triangulation_data(coords_2d, likelihoods, P_gt_bp, Nc_bp):
    coords_sidexfront = []
    coords_frontxoverhead = []
    coords_sidexoverhead = []

    P_sidexfront = []
    P_frontxoverhead = []
    P_sidexoverhead = []

    coords_2d_3cam = []
    P_matrices_3cam = []

    triangulated_indices_sidexfront = []
    triangulated_indices_frontxoverhead = []
    triangulated_indices_sidexoverhead = []
    triangulated_indices_3cam = []

    Np = coords_2d.shape[1]  # Number of points to triangulate
    for point_idx in range(Np):
        if Nc_bp < 3:
            continue
        else:
            conf = np.where(likelihoods[:, point_idx] > pcutoff)[0]  # Get cameras with valid data
            Nc_conf = len(conf)

            if Nc_conf == 2:
                if set(conf) == {0, 1}:  # side and front
                    coords_sidexfront.append(coords_2d[conf, point_idx, :])
                    P_sidexfront.append(np.array([P_gt_bp[i] for i in conf]))
                    triangulated_indices_sidexfront.append(point_idx)
                elif set(conf) == {1, 2}:  # front and overhead
                    coords_frontxoverhead.append(coords_2d[conf, point_idx, :])
                    P_frontxoverhead.append(np.array([P_gt_bp[i] for i in conf]))
                    triangulated_indices_frontxoverhead.append(point_idx)
                elif set(conf) == {0, 2}:  # side and overhead
                    coords_sidexoverhead.append(coords_2d[conf, point_idx, :])
                    P_sidexoverhead.append(np.array([P_gt_bp[i] for i in conf]))
                    triangulated_indices_sidexoverhead.append(point_idx)
            elif Nc_conf == 3:
                coords_2d_3cam.append(coords_2d[:, point_idx, :])
                P_matrices_3cam.append(np.array(P_gt_bp))
                triangulated_indices_3cam.append(point_idx)

    return {
        'coords_sidexfront_batch': stack_if_not_empty(coords_sidexfront),
        'P_sidexfront_batch': stack_if_not_empty(P_sidexfront),
        'coords_frontxoverhead_batch': stack_if_not_empty(coords_frontxoverhead),
        'P_frontxoverhead_batch': stack_if_not_empty(P_frontxoverhead),
        'coords_sidexoverhead_batch': stack_if_not_empty(coords_sidexoverhead),
        'P_sidexoverhead_batch': stack_if_not_empty(P_sidexoverhead),
        'coords_2d_3cam_batch': stack_if_not_empty(coords_2d_3cam),
        'P_matrices_3cam_batch': stack_if_not_empty(P_matrices_3cam),
        'triangulated_indices_sidexfront': triangulated_indices_sidexfront,
        'triangulated_indices_frontxoverhead': triangulated_indices_frontxoverhead,
        'triangulated_indices_sidexoverhead': triangulated_indices_sidexoverhead,
        'triangulated_indices_3cam': triangulated_indices_3cam
    }

def stack_if_not_empty(data_list):
    if len(data_list) > 0:
        return np.stack(data_list)
    else:
        return None

def triangulate_for_pair(coords_batch, P_batch):
    if coords_batch is None:
        return []

    real_world_coords = []
    for frame_idx in range(len(coords_batch)):
        coords_2d_frame = np.expand_dims(coords_batch[frame_idx], axis=1)  # Shape (2, 1, 2) or (3, 1, 2)
        P_frame = P_batch[frame_idx]  # Shape (2, 3, 4) or (3, 3, 4)

        # Use triangulate_Npts from pycalib.calib
        real_world_coord = triangulate_Npts(coords_2d_frame, P_frame)
        real_world_coords.append(real_world_coord.flatten()[:3])  # Extract XYZ coordinates

    return real_world_coords

def reassemble_in_original_order(real_world_coords, triangulated_coords, triangulated_indices):
    for idx, point_idx in enumerate(triangulated_indices):
        real_world_coords[point_idx, :] = triangulated_coords[idx]

def project_back_to_cameras_vectorized(real_world_coords, cameras_extrinsics, cameras_intrinsics,
                                       coords_2d_all,
                                       side_err, front_err, overhead_err,
                                       side_repr, front_repr, overhead_repr):
    Np = real_world_coords.shape[0]

    # Precompute the rotation and translation for each camera
    cam_params = {}
    for cam in ['side', 'front', 'overhead']:
        cam_params[cam] = {
            'rvec': cv2.Rodrigues(cameras_extrinsics[cam]['rotm'])[0],
            'tvec': cameras_extrinsics[cam]['tvec'],
            'intrinsic': cameras_intrinsics[cam]
        }

    # Project the world coordinates for each camera
    for cam_idx, cam in enumerate(['side', 'front', 'overhead']):
        repr_list = []
        err_list = []

        # Project each point
        projected_points, _ = cv2.projectPoints(
            real_world_coords.reshape(-1, 1, 3),
            cam_params[cam]['rvec'],
            cam_params[cam]['tvec'],
            cam_params[cam]['intrinsic'],
            None
        )
        projected_points = projected_points.reshape(-1, 2)

        # Valid points
        valid_mask = ~np.isnan(coords_2d_all[cam_idx, :, 0])

        # Reprojection error
        errors = np.linalg.norm(projected_points[valid_mask] - coords_2d_all[cam_idx, valid_mask], axis=1)

        # Append to lists
        repr_list.extend(projected_points[valid_mask])
        err_list.extend(errors)

        # Assign to appropriate lists
        if cam == 'side':
            side_repr.extend(repr_list)
            side_err.extend(err_list)
        elif cam == 'front':
            front_repr.extend(repr_list)
            front_err.extend(err_list)
        elif cam == 'overhead':
            overhead_repr.extend(repr_list)
            overhead_err.extend(err_list)

def finalize_coords(real_world_coords, side_repr, front_repr, overhead_repr,
                    side_err, front_err, overhead_err):
    real_world_coords = np.array(real_world_coords)
    side_repr_arr = np.array(side_repr)
    front_repr_arr = np.array(front_repr)
    overhead_repr_arr = np.array(overhead_repr)
    side_err_arr = np.array(side_err)
    front_err_arr = np.array(front_err)
    overhead_err_arr = np.array(overhead_err)

    return (real_world_coords, side_repr_arr, front_repr_arr, overhead_repr_arr,
            side_err_arr, front_err_arr, overhead_err_arr)


import numpy as np


def find_component_borders(cluster_labels, leniency=2):
    """
    Find the frames where two components border each other with some leniency for small gaps.

    Args:
        cluster_labels: The array of GMM cluster assignments.
        leniency: The number of frames to tolerate between bordering components.

    Returns:
        border_frames: A list of frame indices where two components border each other.
    """
    border_frames = []

    # Find number of components
    n_components = len(np.unique(cluster_labels))

    # Find 1st and last frame of each component
    first_frames = []
    last_frames = []
    for i in range(n_components):
        component_frames = np.where(cluster_labels == i)[0]
        first_frames.append(component_frames[0])
        last_frames.append(component_frames[-1])

    # Find the frames where any components border each other with leniency
    gaps = [(f - l, f, l) for f in first_frames for l in last_frames]

    # Identify the gaps that are within the leniency threshold
    for gap, first_frame, last_frame in gaps:
        if np.abs(gap) <= leniency:
            # find the middle frames between the two components
            middle = np.round(np.mean([first_frame, last_frame])).astype(int)
            border_frames.append(middle)  # Record the frame where components border

    # Flatten the list of borders to a unique set of frame indices
    border_frames = list(set(border_frames))

    return border_frames


def fit_gmm_for_label_view(args):
    """
    Fit GMM for a specific label and view, return the cluster labels and borders.

    Args:
        args: A tuple containing (view, label, coords_filtered, leniency, n_components_range)

    Returns:
        view: The camera view
        label: The calibration label
        frame_indices: Indices of the frames used in GMM
        cluster_labels: Cluster assignments for each frame
        border_frames: List of frame indices where components change
    """
    view, label, coords_filtered, leniency, n_components_range = args
    #print(f"Analyzing {label} in {view} view...")

    # Prepare the calibration data (x, y positions)
    X = np.column_stack((coords_filtered[(label, 'x')].values, coords_filtered[(label, 'y')].values))
    frame_indices = coords_filtered.index.values  # Get the frame indices

    # Fit GMM with different numbers of components and calculate BIC
    bic = []
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(X)
        bic.append(gmm.bic(X))

    # Detect the elbow point using BIC
    knee_locator = KneeLocator(n_components_range, bic, curve='convex', direction='decreasing')
    best_n_components = knee_locator.elbow
    #print(f"Best number of components for {label} in {view}: {best_n_components}")

    if best_n_components is None:
        #print(f"No elbow detected for {label} in {view}. Skipping...")
        return view, label, frame_indices, [], []
    else:
        # Fit GMM with the best number of components
        gmm_best = GaussianMixture(n_components=best_n_components, covariance_type='full', random_state=42)
        gmm_best.fit(X)

        # Get cluster assignments
        cluster_labels = gmm_best.predict(X)

        # Find borders where components change
        border_frames = find_component_borders(cluster_labels, leniency)
        border_frames_real = coords_filtered.iloc[border_frames].index.values

        return view, label, frame_indices, cluster_labels, border_frames_real


########################################################################################################################

class MapExperiment:
    def __init__(self, DataframeCoor_side, DataframeCoor_front, DataframeCoor_overhead, belt_coords, snapshot_paths):
        self.DataframeCoors = {'side': DataframeCoor_side, 'front': DataframeCoor_front, 'overhead': DataframeCoor_overhead}

        self.cameras = CameraData(snapshot_paths)
        self.cameras_specs = self.cameras.specs
        self.cameras_intrinsics = self.cameras.intrinsic_matrices

        self.belt_pts = BeltPoints(belt_coords)
        self.belt_coords_CCS = self.belt_pts.coords_CCS
        self.belt_coords_WCS = self.belt_pts.coords_WCS

    def plot_CamCoorSys(self):
        return self.belt_pts.plot_CCS(self.cameras)

    def plot_WorldCoorSys(self):
        return self.belt_pts.plot_WCS()

    def estimate_pose(self):
        cameras_extrinsics = self.cameras.compute_cameras_extrinsics(self.belt_coords_WCS, self.belt_coords_CCS)
        self.print_reprojection_errors(cameras_extrinsics)
        return cameras_extrinsics

    def estimate_pose_with_guess(self):
        cameras_extrinsics_ini_guess = self.cameras.compute_cameras_extrinsics(
            self.belt_coords_WCS,
            self.belt_coords_CCS,
            use_extrinsics_ini_guess=True
        )
        self.print_reprojection_errors(cameras_extrinsics_ini_guess, with_guess=True)
        return cameras_extrinsics_ini_guess

    def print_reprojection_errors(self, cameras_extrinsics, with_guess=False):
        if with_guess:
            print('Reprojection errors (w/ initial guess):')
        else:
            print('Reprojection errors:')
        for cam, data in cameras_extrinsics.items():
            print(f'{cam}: {data["repr_err"]}')

    def plot_cam_locations_and_pose(self, cameras_extrinsics):
        fig, ax = self.belt_pts.plot_WCS()
        for cam in self.cameras.specs:
            vec_WCS_to_CCS, rot_cam_opencv = self.get_camera_vectors(cameras_extrinsics, cam)
            self.add_camera_pose(ax, cam, vec_WCS_to_CCS, rot_cam_opencv)
        return fig, ax

    def get_camera_vectors(self, cameras_extrinsics, cam):
        cob_cam_opencv = cameras_extrinsics[cam]['rotm'].T
        vec_WCS_to_CCS = -cob_cam_opencv @ cameras_extrinsics[cam]['tvec']
        return vec_WCS_to_CCS, cameras_extrinsics[cam]['rotm']

    def add_camera_pose(self, ax, cam, vec_WCS_to_CCS, rot_cam_opencv):
        ax.scatter(*vec_WCS_to_CCS, s=50, c="b", marker=".", linewidth=0.5, alpha=1)
        ax.text(*vec_WCS_to_CCS.flatten(), s=cam, c="b")
        for row, color in zip(rot_cam_opencv, ["r", "g", "b"]):
            ax.quiver(
                *vec_WCS_to_CCS.flatten(), *row, color=color,
                length=500, arrow_length_ratio=0, normalize=True, linewidth=2
            )
        ax.axis("equal")


class GetSingleExpData:
    def __init__(self, side_file, front_file, overhead_file):
        self.files = {'side': side_file, 'front': front_file, 'overhead': overhead_file}

        # retrieve raw data, format, and match frames across the 3 camera views
        self.DataframeCoors = self.load_and_clean_all_data()

        self.extrinsics = None
        self.intrinsics = None
        self.belt_coords_CCS = None

    def load_and_clean_all_data(self):
        dataframes = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            dataframes[view] = self.get_and_format_data(self.files[view], view)

        aligned_videos = self.align_dfs(dataframes)
        return aligned_videos

    def get_and_format_data(self, file, view):
        df = pd.read_hdf(file)
        try:
            df = df.loc(axis=1)[vidstuff['scorers'][view]].copy()
        except:
            df = df.loc(axis=1)[vidstuff['scorers'][f'{view}_new']].copy()
        return df

    def load_timestamps(self, view):
        timestamp_path = self.files[view].replace(vidstuff['scorers'][view], '_Timestamps').replace('.h5', '.csv')
        # timestamp_path = utils.Utils().Get_timestamps_from_analyse_file(self.files[view], view)
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

    def adjust_frames(self):
        timestamps = {'side': [], 'front': [], 'overhead': []}
        timestamps_adj = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            timestamps[view] = self.zero_timestamps(self.load_timestamps(view))
            if view != 'side':
                timestamps_adj[view] = self.adjust_timestamps(timestamps['side'], timestamps[view])
        timestamps_adj['side'] = timestamps['side']['Timestamp'].astype(float)
        return timestamps_adj

    def match_frames(self):
        timestamps = self.adjust_frames()

        buffer_ns = int(4.04e+6)  # Frame duration in nanoseconds

        # Ensure the timestamps are sorted
        dfs = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            timestamps[view] = timestamps[view].sort_values().reset_index(drop=True)
            dfs[view] = pd.DataFrame({'Timestamp': timestamps[view], 'Frame_number_%s' %view: range(len(timestamps[view]))})

        # Perform asof merge to find the closest matching frames within the buffer
        matched_front = pd.merge_asof(dfs['side'], dfs['front'], on='Timestamp', direction='nearest', tolerance=buffer_ns,
                                      suffixes=('_side', '_front'))
        matched_all = pd.merge_asof(matched_front, dfs['overhead'], on='Timestamp', direction='nearest',
                                    tolerance=buffer_ns, suffixes=('_side', '_overhead'))

        # Handle NaNs explicitly by setting unmatched frames to -1
        matched_frames = matched_all[['Frame_number_side', 'Frame_number_front', 'Frame_number_overhead']].applymap(
            lambda x: int(x) if pd.notnull(x) else -1).values.tolist()

        return matched_frames

    def align_dfs(self, dfs):
        matched_frames = self.match_frames()

        # find first index row where all frames are in positive time
        start = None
        for idx, it in enumerate(matched_frames):
            if np.all(np.array(it) > 0):
                start = idx
                break
        matched_frames = matched_frames[start:]

        frames = {'side': [], 'front': [], 'overhead': []}
        aligned_dfs = {'side': [], 'front': [], 'overhead': []}
        for vidx, view in enumerate(vidstuff['cams']):
            frames[view] = [frame[vidx] for frame in matched_frames]
            aligned_dfs[view] = dfs[view].iloc[frames[view]].reset_index(drop=False).rename(columns={'index': 'original_index'})

        return aligned_dfs


    ###################################################################################################################
    # Map the camera views to the real world coordinates for each video file
    ###################################################################################################################
    def map(self):
        # Get file path for saving data and associated metadata
        name_idx = os.path.basename(self.files['side']).split('_').index('side')
        base_file_name = '_'.join(os.path.basename(self.files['side']).split('_')[:name_idx])
        base_path_name = os.path.join(os.path.dirname(self.files['side']), base_file_name)

        # Detect camera shifts
        calibration_labels = ['StartPlatR', 'TransitionL', 'StartPlatL', 'TransitionR']  # Use your calibration labels
        print('Detecting camera movement...')
        segments = self.detect_camera_movement(calibration_labels)

        if len(segments) > 1:
            print(f"Camera movement detected. Processing data in {len(segments)} segments.")
        else:
            print("No camera movement detected. Processing data as a single segment.")

        # Initialize lists to store results
        all_real_world_coords = []
        all_repr_error = []
        all_repr = []

        for idx, (start_frame, end_frame) in enumerate(segments):
            print(f"Processing segment {idx + 1}/{len(segments)}: frames {start_frame} to {end_frame}")

            # Extract data for this segment
            segment_indices = range(start_frame, end_frame + 1)

            # Get calibration coordinates for this segment
            calibration_coords = self.get_belt_coords(segment_indices)

            # Perform calibration
            calib_obj = BasicCalibration(calibration_coords)
            cameras_extrinsics = calib_obj.estimate_cams_pose()
            cameras_intrinsics = calib_obj.cameras_intrinsics

            # calib_obj.plot_cam_pose(cameras_extrinsics)

            # Create optimize instance and perform optimization
            optimise = OptimizeCalibration.optimize(calibration_coords, cameras_extrinsics, cameras_intrinsics, base_path_name, self)
            new_calibration_data = optimise.optimise_calibration(debugging=True)

            # Update the calibration parameters
            self.extrinsics = new_calibration_data['extrinsics']
            self.intrinsics = new_calibration_data['intrinsics']
            self.belt_coords_CCS = new_calibration_data['belt points CCS']

            #calib_obj.plot_cam_pose(self.extrinsics)

            # Get real-world data for this segment
            results = self.get_realworld_coords(segment_indices)

            # Append results
            all_real_world_coords.append(results['real_world_coords'])
            all_repr_error.append(results['repr_error'])
            all_repr.append(results['repr'])

        # Concatenate all segments' results
        real_world_coords = pd.concat(all_real_world_coords)
        repr_error = pd.concat(all_repr_error)
        repr = pd.concat(all_repr)

        # Save the results as an HDF5 file
        file_path = base_path_name + '_mapped3D.h5'

        # Store the dataframes into a single HDF5 file with different keys
        with pd.HDFStore(file_path) as store:
            store.put('real_world_coords', real_world_coords)
            store.put('repr_error', repr_error)
            store.put('repr', repr)

        print(f"Data saved to HDF5 file at {file_path}")

    def plot_belt_coords(self):
        """
        For 243 on 20230306, test to check dlc coords vs my enhanced manual coords
        :return:
        """
        reference_belt_coords = pd.read_csv(r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\Manual_Labelling\CameraCalibration\HM_20230306_APACharRepeat_FAA-1035243_None_1\calibration_labels_enhanced - copy.csv")
        current_belt_coords = self.get_belt_coords()

        video_paths = [
            r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230306\HM_20230306_APACharRepeat_FAA-1035243_None_side_1.avi",
            r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230306\HM_20230306_APACharRepeat_FAA-1035243_None_front_1.avi",
            r"X:\hmorley\Dual-belt_APAs\videos\Round_3\20230306\HM_20230306_APACharRepeat_FAA-1035243_None_overhead_1.avi"
        ]
        frame_number = 16977
        frames = [self.read_frame_from_video(path, frame_number) for path in video_paths]

        for i, (frame, camera_view) in enumerate(zip(frames, vidstuff['cams'])):
            plt.figure(figsize=(10, 8))
            # Plot df_new coordinates in blue
            self.plot_coordinates_on_image(frame, current_belt_coords, camera_view, color='blue', label='current')
            # Overlay df_enhanced coordinates in red
            self.plot_coordinates_on_image(frame, reference_belt_coords, camera_view, color='red', label='enhanced')
            plt.title(f'Camera {camera_view.capitalize()} - Blue: current, Red: enhanced')
            plt.axis('off')
            plt.show()

    def read_frame_from_video(self, video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
        else:
            raise ValueError(f"Could not read frame {frame_number} from video {video_path}")

    def plot_coordinates_on_image(self, image, df, camera_view, color, label):
        plt.imshow(image)
        # Filter the dataframe for the current camera view and group by body parts
        df_camera = df[['bodyparts', 'coords', camera_view]].pivot(index='bodyparts', columns='coords',
                                                                   values=camera_view)

        # Plot points only relevant to the specific camera view
        for bodypart, row in df_camera.iterrows():
            x, y = row['x'], row['y']  # Get x and y coordinates for the body part
            plt.scatter(x, y, color=color, alpha=0.8, s=4)
            plt.text(x, y, bodypart, fontsize=9, color=color)

    def get_belt_coords(self, segment_indices=None):
        masks = {'side': [], 'front': [], 'overhead': []}
        belt_coords = {'side': [], 'front': [], 'overhead': []}
        means = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            if segment_indices is not None:
                df_view = self.DataframeCoors[view].iloc[segment_indices]
            else:
                df_view = self.DataframeCoors[view]
            masks[view] = np.all(df_view.loc(axis=1)[
                                     ['StartPlatR', 'StepR', 'StartPlatL', 'StepL', 'TransitionR',
                                      'TransitionL'], 'likelihood'] > 0.999, axis=1)
            if sum(masks[view]) < 1000:
                no_transitionL_mask = np.all(df_view.loc(axis=1)[
                                        ['StartPlatR', 'StepR', 'StartPlatL', 'StepL', 'TransitionR'], 'likelihood'] > 0.999, axis=1)
                if sum(no_transitionL_mask) >= 1000:
                    print("Low confidence calibration label for TransitionL detected in %s. Using low confidence TransitionL for now, will be optimised but check quality." %view)
                    masks[view] = no_transitionL_mask
                else:
                    print("Low confidence calibration labels detected, attempting to use pcutoff = 0.9...")
                    masks[view] = np.all(df_view.loc(axis=1)[
                                             ['StartPlatR', 'StepR', 'StartPlatL', 'StepL', 'TransitionR',
                                              'TransitionL'], 'likelihood'] > 0.9, axis=1)
                    if sum(masks[view]) < 100:
                        low_conf_mask = np.all(df_view.loc(axis=1)[
                                             ['StartPlatR', 'StepR', 'StartPlatL', 'StepL', 'TransitionR',
                                              'TransitionL'], 'likelihood'] > 0.8, axis=1)
                        raise ValueError("Insufficient high confidence calibration labels detected!!! Only %s frames available out of %s. At pcutoff = 0.8, %s frames available." %(sum(masks[view]), len(masks[view]), sum(low_conf_mask)))
            belt_coords[view] = df_view.loc(axis=1)[
                ['StartPlatR', 'StepR', 'StartPlatL', 'StepL', 'TransitionR', 'TransitionL'], ['x', 'y']][masks[view]]

            means[view] = belt_coords[view].median(axis=0)
            std = belt_coords[view].std(axis=0)
            if view == 'side':
                for label in std.index.get_level_values('bodyparts').unique():
                    if np.any(std.loc[label] > 2):
                        print(f"!!!!!Warning: high std for {label} in {view} view: {std.loc[label]}!!!!!")

        # Concatenate the mean values
        belt_coords_df = pd.concat([means['side'], means['front'], means['overhead']], axis=1)
        belt_coords_df.columns = vidstuff['cams']

        # Add door coordinates
        door_coords = self.get_door_coords(segment_indices)
        belt_coords_df.reset_index(inplace=True, drop=False)
        coords = pd.concat([belt_coords_df, door_coords], axis=0).reset_index(drop=True)

        return coords

    def get_door_coords(self, segment_indices=None):
        masks = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            if segment_indices is not None:
                df_view = self.DataframeCoors[view].iloc[segment_indices]
            else:
                df_view = self.DataframeCoors[view]
            masks[view] = df_view.loc(axis=1)['Door', 'likelihood'] > pcutoff

        mask = masks['side'] & masks['front'] & masks['overhead']

        sem_factor = 100
        door_present = {'side': [], 'front': [], 'overhead': []}
        door_closed_masks = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            df_view = self.DataframeCoors[view].iloc[segment_indices] if segment_indices is not None else \
            self.DataframeCoors[view]
            door_present[view] = df_view.loc(axis=1)['Door', ['x', 'y']][mask]
            door_closed_masks[view] = np.logical_and(
                door_present[view]['Door', 'y'] < door_present[view]['Door', 'y'].mean() + door_present[view][
                    'Door', 'y'].sem() * sem_factor,
                door_present[view]['Door', 'y'] > door_present[view]['Door', 'y'].mean() - door_present[view][
                    'Door', 'y'].sem() * sem_factor)

        closed_mask = door_closed_masks['side'] & door_closed_masks['front'] & door_closed_masks['overhead']

        means = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            means[view] = door_present[view][closed_mask].mean(axis=0)

        # Concatenate the mean values
        door_coords = pd.concat([means['side'], means['front'], means['overhead']], axis=1)
        door_coords.columns = ['side', 'front', 'overhead']
        door_coords.reset_index(inplace=True, drop=False)

        return door_coords

    def detect_camera_movement(self, calibration_labels):
        """
        Detects camera movement by analyzing shifts in 2D calibration label positions across all cameras.
        """
        data = self.DataframeCoors  # Assuming DataframeCoors contains your data

        coords_filtered_all_views = {}

        for view in ['side', 'front', 'overhead']:
            coords = data[view].loc[:, (calibration_labels, ['x', 'y', 'likelihood'])]

            # Step 1: Create a mask with a pcutoff of 0.999
            valid_frames_mask = np.ones(len(coords), dtype=bool)
            for calib_label in calibration_labels:
                valid_frames_mask &= coords[(calib_label, 'likelihood')] >= 0.999

            # Check if there are more than 1001 valid frames
            if np.sum(valid_frames_mask) > 1001:
                # Use the stricter mask (0.999 pcutoff)
                final_valid_frames_mask = valid_frames_mask
            else:
                # Step 2: Use a less strict mask with a pcutoff of 0.9
                valid_frames_mask_09 = np.ones(len(coords), dtype=bool)
                for calib_label in calibration_labels:
                    valid_frames_mask_09 &= coords[(calib_label, 'likelihood')] >= 0.9
                final_valid_frames_mask = valid_frames_mask_09

            # Apply the final valid frames mask
            coords_filtered_all_views[view] = data[view].loc[
                final_valid_frames_mask, (calibration_labels, ['x', 'y'])].copy()

            # Smoothing the coordinates
            for label in calibration_labels:
                x_data = coords_filtered_all_views[view][(label, 'x')].values
                y_data = coords_filtered_all_views[view][(label, 'y')].values

                # Ensure window_length is valid and smaller than the data size
                window_len = min(1001, len(x_data) if len(x_data) % 2 != 0 else len(x_data) - 1)

                coords_filtered_all_views[view][(label, 'x')] = savgol_filter(x_data, window_length=window_len,
                                                                              polyorder=3)
                coords_filtered_all_views[view][(label, 'y')] = savgol_filter(y_data, window_length=window_len,
                                                                              polyorder=3)

        # Prepare arguments for multiprocessing
        tasks = []
        for view, coords_filtered in coords_filtered_all_views.items():
            for label in calibration_labels:
                args = (view, label, coords_filtered, 5, range(1, 10))
                tasks.append(args)

        # Run GMM in parallel using ProcessPoolExecutor
        all_results = []  # Collect all results for plotting
        all_borders = {}
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(fit_gmm_for_label_view, args) for args in tasks]
            for future in as_completed(futures):
                view, label, frame_indices, cluster_labels, borders = future.result()
                all_results.append((view, label, frame_indices, cluster_labels))
                if view not in all_borders:
                    all_borders[view] = {}
                all_borders[view][label] = borders

        # Collect all border frames
        border_frames_list = []
        for view, labels_borders in all_borders.items():
            for label, borders in labels_borders.items():
                border_frames_list.extend(borders)

            # Remove duplicates and sort
            unique_border_frames = sorted(set(border_frames_list))

            # Refine unique border frames
            refined_border_frames = []
            window_size = 5000  # Define your window size for grouping borders

            # Group border frames if at least 3 occur within a 5000-frame window and calculate the average
            current_window = []
            for frame in unique_border_frames:
                if not current_window or (frame - current_window[-1] <= window_size):
                    current_window.append(frame)
                else:
                    if len(current_window) >= 3:  # Check if there are at least 3 borders in the current window
                        refined_border_frames.append(int(np.mean(current_window)))  # Take the average of the borders
                    current_window = [frame]

            # Check the last window
            if len(current_window) >= 3:
                refined_border_frames.append(int(np.mean(current_window)))

        # Define segments based on movement frames
        frames = data['side'].index  # Assuming 'side' has the frames
        segments = []
        last_frame = frames[0]
        for movement_frame in refined_border_frames:
            segments.append((last_frame, movement_frame - 1))
            last_frame = movement_frame
        segments.append((last_frame, frames[-1]))

        # Return segments and all_results for plotting
        return segments

    def plot_gmm_results(self, all_results, all_borders):
        """
        Plots the cluster labels over frames for each view and calibration label, including borders.

        Args:
            all_results: List of tuples containing (view, label, frame_indices, cluster_labels)
            all_borders: Dictionary containing borders for each view and label
        """
        for view, label, frame_indices, cluster_labels in all_results:
            if len(cluster_labels) == 0:
                continue  # Skip if no clusters
            plt.figure(figsize=(12, 6))
            plt.scatter(frame_indices, cluster_labels, c=cluster_labels, cmap='viridis', s=10)
            plt.title(f'GMM Clusters Over Frames - View: {view}, Label: {label}')
            plt.xlabel('Frame Index')
            plt.ylabel('Cluster Label')
            plt.colorbar(label='Cluster')

            # Plot vertical lines for borders
            borders = all_borders.get(view, {}).get(label, [])
            for border in borders:
                plt.axvline(frame_indices[border], color='red', linestyle='--')

            plt.show()

    ####################################################################################################################
    # Triangulate the 3D coordinates of the common body parts in the same order for all 3 camera views
    ####################################################################################################################
    def get_realworld_coords(self, segment_indices):
        # Initial setup
        DataframeCoors_segment = {}
        for view in vidstuff['cams']:
            DataframeCoors_segment[view] = self.DataframeCoors[view].iloc[segment_indices]

        labels, side_coords, front_coords, overhead_coords = self.prepare_data(DataframeCoors_segment)
        self.side_coords = side_coords
        self.front_coords = front_coords
        self.overhead_coords = overhead_coords

        K, R_gt, t_gt, P_gt, Nc = self.get_camera_params(self.extrinsics, self.intrinsics)
        length = side_coords.shape[0]
        self.real_world_coords_allparts, self.repr_error_allparts, self.repr_allparts = self.setup_dataframes(labels,
                                                                                                              length)

        # Set the index to be the frame indices from the original data
        frame_indices = DataframeCoors_segment['side'].index
        self.real_world_coords_allparts.index = frame_indices
        self.repr_error_allparts.index = frame_indices
        self.repr_allparts.index = frame_indices

        # Set up shared memory for large arrays
        side_coords_shm = shared_memory.SharedMemory(create=True, size=side_coords.nbytes)
        shared_side_coords = np.ndarray(side_coords.shape, dtype=side_coords.dtype, buffer=side_coords_shm.buf)
        shared_side_coords[:] = side_coords[:]

        front_coords_shm = shared_memory.SharedMemory(create=True, size=front_coords.nbytes)
        shared_front_coords = np.ndarray(front_coords.shape, dtype=front_coords.dtype, buffer=front_coords_shm.buf)
        shared_front_coords[:] = front_coords[:]

        overhead_coords_shm = shared_memory.SharedMemory(create=True, size=overhead_coords.nbytes)
        shared_overhead_coords = np.ndarray(overhead_coords.shape, dtype=overhead_coords.dtype, buffer=overhead_coords_shm.buf)
        shared_overhead_coords[:] = overhead_coords[:]

        # Prepare shared data
        shared_data = {
            'side_coords_shape': side_coords.shape,
            'side_coords_dtype': side_coords.dtype,
            'side_coords_name': side_coords_shm.name,

            'front_coords_shape': front_coords.shape,
            'front_coords_dtype': front_coords.dtype,
            'front_coords_name': front_coords_shm.name,

            'overhead_coords_shape': overhead_coords.shape,
            'overhead_coords_dtype': overhead_coords.dtype,
            'overhead_coords_name': overhead_coords_shm.name,

            'cameras_extrinsics': self.extrinsics,
            'cameras_intrinsics': self.intrinsics,
            'P_gt': P_gt,
            'Nc': Nc,
        }

        # Prepare arguments for multiprocessing
        args_list = [((bidx, body_part), shared_data) for bidx, body_part in enumerate(labels['all'])]

        # Start multiprocessing pool
        print('---------------------------------------------------------------\nStarting triangulation...')
        start_time_total = time.time()

        num_processes = min(6, cpu_count())
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_part_wrapper, args_list)

        # Process results
        for res in results:
            body_part, result_data = res
            self.store_results(body_part, result_data)

        total_time = time.time() - start_time_total
        print(f"Total triangulation time: {total_time:.2f} seconds\n---------------------------------------------------------------")

        # Clean up shared memory
        side_coords_shm.close()
        side_coords_shm.unlink()
        front_coords_shm.close()
        front_coords_shm.unlink()
        overhead_coords_shm.close()
        overhead_coords_shm.unlink()

        # Return results
        results = {
            'real_world_coords': self.real_world_coords_allparts,
            'repr_error': self.repr_error_allparts,
            'repr': self.repr_allparts
        }
        return results

    def prepare_data(self, DataframeCoors_segment):
        labels = self.find_common_bodyparts(DataframeCoors_segment)
        side_coords, front_coords, overhead_coords = self.get_common_camera_arrays(labels, DataframeCoors_segment)
        return labels, side_coords, front_coords, overhead_coords

    def find_common_bodyparts(self, DataframeCoors_segment):
        bodyparts = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            bodyparts[view] = self.get_unique_bodyparts(DataframeCoors_segment[view])

        common_bodyparts = self.get_intersected_bodyparts(
            bodyparts['side'], bodyparts['front'], bodyparts['overhead'])
        all_bodyparts = self.get_all_bodyparts(
            bodyparts['side'], bodyparts['front'], bodyparts['overhead'], common_bodyparts)
        # remove 'original_index' from all_bodyparts
        all_bodyparts.remove('original_index')
        labels = self.create_labels_dict(
            bodyparts['side'], bodyparts['front'], bodyparts['overhead'], all_bodyparts, common_bodyparts)
        return labels

    def get_unique_bodyparts(self, dataframe):
        return dataframe.columns.get_level_values('bodyparts').unique()

    def get_intersected_bodyparts(self, side_bodyparts, front_bodyparts, overhead_bodyparts):
        sidexfront = np.intersect1d(side_bodyparts, front_bodyparts)
        sidexoverhead = np.intersect1d(side_bodyparts, overhead_bodyparts)
        frontxoverhead = np.intersect1d(front_bodyparts, overhead_bodyparts)
        return list(set(sidexfront) & set(sidexoverhead) & set(frontxoverhead))

    def get_all_bodyparts(self, side_bodyparts, front_bodyparts, overhead_bodyparts, common_bodyparts):
        sidexfront = np.intersect1d(side_bodyparts, front_bodyparts)
        sidexoverhead = np.intersect1d(side_bodyparts, overhead_bodyparts)
        frontxoverhead = np.intersect1d(front_bodyparts, overhead_bodyparts)
        return list(set(sidexfront) | set(sidexoverhead) | set(frontxoverhead))

    def create_labels_dict(self, side_bodyparts, front_bodyparts, overhead_bodyparts, all_bodyparts, common_bodyparts):
        side = list(set(side_bodyparts) & set(all_bodyparts))
        front = list(set(front_bodyparts) & set(all_bodyparts))
        overhead = list(set(overhead_bodyparts) & set(all_bodyparts))
        if set(all_bodyparts) == set(label_list_World):
            all_bodyparts = label_list_World
        else:
            raise ValueError('The labels in all_bodyparts are not the same as label_list_World')
        return {
            'all': all_bodyparts, 'allcommon': common_bodyparts, 'side': side,
            'front': front, 'overhead': overhead
        }

    def get_common_camera_arrays(self, labels, DataframeCoors_segment):
        data = {'side': [], 'front': [], 'overhead': []}
        for view in vidstuff['cams']:
            data[view] = DataframeCoors_segment[view].to_numpy()

        num_rows = DataframeCoors_segment['side'].shape[0]

        # Mapping of current labels to their column indices in the common label list
        label_to_index = {'side': {}, 'front': {}, 'overhead': {}}
        coords = {'side': [], 'front': [], 'overhead': []}

        for view in vidstuff['cams']:
            for idx, label in enumerate(labels[view]):
                pos = labels['all'].index(label)
                label_to_index[view][label] = pos
            # create empty array with shape (num_rows, num_labels, 3) filled with NaNs
            coords[view] = np.full((num_rows, len(labels['all']), 3), np.nan, dtype=data[view].dtype)

        # Fill in the data for existing labels in their new positions for each camera view
        for idx, label in enumerate(labels['all']):
            for view in vidstuff['cams']:
                if label in labels[view]:
                    pos = label_to_index[view][label]
                    original_pos_mask = DataframeCoors_segment[view].columns.get_loc(label)
                    original_pos = np.where(original_pos_mask)[0]
                    coords[view][:, pos, :] = data[view][:, original_pos]

        return coords['side'], coords['front'], coords['overhead']

    def get_camera_params(self, cameras_extrinsics, cameras_intrinsics):
        # Camera intrinsics
        K = [cameras_intrinsics[cam] for cam in cameras_intrinsics]

        # Camera poses: cameras are at the vertices of a hexagon
        R_gt = [cameras_extrinsics[cam]['rotm'] for cam in cameras_extrinsics]
        t_gt = [cameras_extrinsics[cam]['tvec'] for cam in cameras_extrinsics]
        P_gt = [np.dot(K[i], np.hstack((R_gt[i], t_gt[i]))) for i in range(len(K))]
        Nc = len(K)

        self.P_gt = P_gt
        self.Nc = Nc

        return K, R_gt, t_gt, P_gt, Nc

    def setup_dataframes(self, labels, length):
        # Initialize the columns for the DataFrames based on body parts and their coordinates
        multi_column = pd.MultiIndex.from_product([labels['all'], ['x', 'y', 'z']])
        multi_column_err = pd.MultiIndex.from_product([labels['all'], ['side', 'front', 'overhead']])
        multi_column_repr = pd.MultiIndex.from_product([labels['all'], ['side', 'front', 'overhead'], ['x', 'y']])

        # Initialize DataFrames with the correct number of rows (length) and multi-level columns
        real_world_coords_allparts = pd.DataFrame(index=range(length), columns=multi_column)
        repr_error_allparts = pd.DataFrame(index=range(length), columns=multi_column_err)
        repr_allparts = pd.DataFrame(index=range(length), columns=multi_column_repr)

        return real_world_coords_allparts, repr_error_allparts, repr_allparts

    def find_empty_cameras(self, coords_2d, likelihoods, P_gt, Nc):
        empty_cameras = np.where(np.all(np.all(np.isnan(coords_2d), axis=2), axis=1))
        if len(empty_cameras) > 0:
            coords_2d = np.delete(coords_2d, empty_cameras, axis=0)
            likelihoods = np.delete(likelihoods, empty_cameras, axis=0)
            Nc_bp = len(coords_2d)
            P_gt_bp = np.delete(P_gt, empty_cameras, axis=0)
        else:
            Nc_bp = Nc
            P_gt_bp = P_gt
        return coords_2d, likelihoods, P_gt_bp, Nc_bp, empty_cameras

    def parallel_triangulate(self, body_part_data):
        """
        This function will be used for parallel triangulation of a single body part.
        It will call `process_body_part` which updates the class-wide result dataframes directly.
        """
        bidx, body_part, side_coords, front_coords, overhead_coords, cameras_extrinsics, cameras_intrinsics, P_gt, Nc, real_world_coords_allparts, repr_error_allparts, repr_allparts = body_part_data

        # Directly call process_body_part which updates the result dataframes
        self.process_body_part(
            bidx, body_part,
            side_coords, front_coords, overhead_coords,
            cameras_extrinsics, cameras_intrinsics,
            P_gt, Nc, real_world_coords_allparts,
            repr_error_allparts, repr_allparts
        )

    def store_results(self, body_part, result_data):
        # Extract data from result_data
        real_world_coords = result_data['real_world_coords']
        side_repr = result_data['side_repr']
        front_repr = result_data['front_repr']
        overhead_repr = result_data['overhead_repr']
        side_err = result_data['side_err']
        front_err = result_data['front_err']
        overhead_err = result_data['overhead_err']

        # Update real_world_coords_allparts
        if real_world_coords is None or len(real_world_coords.shape) == 1:
            self.real_world_coords_allparts[body_part, 'x'] = np.nan
            self.real_world_coords_allparts[body_part, 'y'] = np.nan
            self.real_world_coords_allparts[body_part, 'z'] = np.nan
        else:
            self.real_world_coords_allparts[body_part, 'x'] = real_world_coords[:, 0]
            self.real_world_coords_allparts[body_part, 'y'] = real_world_coords[:, 1]
            self.real_world_coords_allparts[body_part, 'z'] = real_world_coords[:, 2]

        # Update repr_allparts
        self.repr_allparts[body_part, 'side', 'x'] = np.squeeze(side_repr[:, 0])
        self.repr_allparts[body_part, 'side', 'y'] = np.squeeze(side_repr[:, 1])
        self.repr_allparts[body_part, 'front', 'x'] = np.squeeze(front_repr[:, 0])
        self.repr_allparts[body_part, 'front', 'y'] = np.squeeze(front_repr[:, 1])
        self.repr_allparts[body_part, 'overhead', 'x'] = np.squeeze(overhead_repr[:, 0])
        self.repr_allparts[body_part, 'overhead', 'y'] = np.squeeze(overhead_repr[:, 1])

        # Update repr_error_allparts
        self.repr_error_allparts[body_part, 'side'] = side_err
        self.repr_error_allparts[body_part, 'front'] = front_err
        self.repr_error_allparts[body_part, 'overhead'] = overhead_err

    def plot_2d_prep_frame(self, view, frame_number):
        # get video paths
        day = os.path.basename(self.files['side']).split('_')[1]
        video_path = "\\".join([paths['video_folder'], day])

        # Determine the correct video file based on the view
        if view == 'side':
            video_file = os.path.join(video_path, os.path.basename(self.files['side']).replace(vidstuff['scorers']['side'],'').replace('.h5', '.avi'))
        elif view == 'front':
            video_file = os.path.join(video_path, os.path.basename(self.files['front']).replace(vidstuff['scorers']['front'],'').replace('.h5', '.avi'))
        elif view == 'overhead':
            video_file = os.path.join(video_path, os.path.basename(self.files['overhead']).replace(vidstuff['scorers']['overhead'], '').replace('.h5', '.avi'))
        else:
            raise ValueError('Invalid view')

        exp_day = video_file.split("\\")[-2] #if view != 'front' else video_file.split("\\")[-3]
        if '_Pre_' in video_file:
            video_file_tag = '_'.join(video_file.split("\\")[-1].split("_")[0:6])
        else:
            video_file_tag = '_'.join(video_file.split("\\")[-1].split("_")[0:5])

        video_dir = os.path.join(paths['video_folder'], exp_day)
        video_files = [f for f in os.listdir(video_dir) if f.startswith(video_file_tag) and f.endswith('.avi')]

        # Select the video file corresponding to the specified view
        video_file = next((f for f in video_files if view in f), None)
        if not video_file:
            raise FileNotFoundError(f"No video file found for view {view}")

        video_path = os.path.join(video_dir, video_file)

        # get the original frame number
        original_frame_number = self.DataframeCoors[view].loc[frame_number, 'original_index'].values.astype(int)[0]

        # Capture the specific frame from the video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Error reading frame {original_frame_number} from {video_path}")

        return frame

    def plot_2d_skeleton_overlay_on_frame_fromOriginal_singleView(self, frame_number, labels, view):
        # plot the video frame
        frame = self.plot_2d_prep_frame(view, frame_number)

        data_file = self.DataframeCoors[view]

        # # minus 10 from the y of overhead view
        # if view == 'overhead':
        #     data_file.loc(axis=1)['StartPlatL', 'y'] = data_file.loc(axis=1)['StartlatL', 'y'] - 7

        # Plot the 2D skeleton overlay on the frame
        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for i, body_part in enumerate(labels['all']):
            if body_part in labels[view]:
                conf = data_file.loc[frame_number, (body_part, 'likelihood')]
                if conf > pcutoff:
                    x = data_file.loc[frame_number, (body_part, 'x')]
                    y = data_file.loc[frame_number, (body_part, 'y')]
                    if np.isfinite(x) and np.isfinite(y):
                        ax.scatter(x, y, color=cmap(i / len(labels['all'])), s=6, label=body_part, zorder=100)
                        # ax.text(x, y, body_part, fontsize=8, color='r')
        # draw lines for skeleton connections underneath the scatter points
        # for start, end in micestuff['skeleton']:
        #     sx = data_file.loc[frame_number, (start, 'x')]
        #     sy = data_file.loc[frame_number, (start, 'y')]
        #     ex = data_file.loc[frame_number, (end, 'x')]
        #     ey = data_file.loc[frame_number, (end, 'y')]
        #     ax.plot([sx, ex], [sy, ey], 'grey', linewidth=0.7, zorder=0)

        ax.axis('off')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
        plt.title("Original")
        plt.show()
        return fig, ax

    def plot_2d_skeleton_overlay_on_frame_fromOriginal_allViews(self, frame_number, labels):
        for view in ['side', 'front', 'overhead']:
            self.plot_2d_skeleton_overlay_on_frame_fromOriginal_singleView(frame_number, labels, view)

    def plot_2d_skeleton_overlay_on_frame_fromReprojection_singleView(self, repr_allparts, frame_number, labels, view):
        # plot the video frame
        frame = self.plot_2d_prep_frame(view, frame_number)

        # get the relevant columns from repr_allparts for the view
        repr = repr_allparts.xs(view, level=1, axis=1)

        # Plot the 2D skeleton overlay on the frame
        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for i, body_part in enumerate(labels['all']):
            if body_part in labels[view]:
                x = repr.loc[frame_number, (body_part, 'x')]
                y = repr.loc[frame_number, (body_part, 'y')]
                if np.isfinite(x) and np.isfinite(y):
                    ax.scatter(x, y, color=cmap(i / len(labels['all'])), s=6, label=body_part, zorder=100)
                    # ax.text(x, y, body_part, fontsize=8, color='r')
        #belt_coords_hardcoded = self.TEMP_belt_coords()
        belt_coords = self.belt_coords_CCS[view]

        for belt_coord in belt_coords:
            ax.scatter(belt_coord[0], belt_coord[1], color='red', marker='x', s=6, zorder=100) # taken out labels, but if want the relative positions check self.points_str2int from utils_3d_reconstruction.py

        # draw lines for skeleton connections underneath the scatter points
        for start, end in micestuff['skeleton']:
            sx, sy = repr.loc[frame_number, (start, 'x')], repr.loc[frame_number, (start, 'y')]
            ex, ey = repr.loc[frame_number, (end, 'x')], repr.loc[frame_number, (end, 'y')]
            ax.plot([sx, ex], [sy, ey], 'grey', linewidth=0.7, zorder=0)
        ax.axis('off')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
        plt.title("Reprojection")
        plt.show()
        return fig, ax

    def plot_2d_skeleton_overlay_on_frame_fromReprojection_allViews(self, repr_allparts, frame_number, labels):
        for view in ['side', 'front', 'overhead']:
            self.plot_2d_skeleton_overlay_on_frame_fromReprojection_singleView(repr_allparts, frame_number, labels,
                                                                               view)

    def plot_3d_mouse(self, real_world_coords_allparts, labels, frame):
        # plt 3d scatter of all body parts at frame 500 and colour them based on the body part using viridis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('viridis')
        coords = real_world_coords_allparts.loc[frame]
        for i, body_part in enumerate(labels['all']):
            # scatter with small marker size
            ax.scatter(coords[body_part, 'x'], coords[body_part, 'y'], coords[body_part, 'z'],
                       label=body_part, color=cmap(i / len(labels['all'])), s=10)
        # Draw lines for each connection
        for start, end in micestuff['skeleton']:
            sx, sy, sz = coords[(start, 'x')], coords[(start, 'y')], coords[(start, 'z')]
            ex, ey, ez = coords[(end, 'x')], coords[(end, 'y')], coords[(end, 'z')]
            ax.plot([sx, ex], [sy, ey], [sz, ez], 'gray')  # Draw line in gray
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)

        ## set axes as equal scales so each tick on each axis represents the same space
        #ax.axis('equal')
        #ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1 for x:y:z
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set the limits for each axis to simulate equal aspect ratio
        x_limits = [coords.xs('x', level=1).min(), coords.xs('x', level=1).max()]
        y_limits = [coords.xs('y', level=1).min(), coords.xs('y', level=1).max()]
        z_limits = [coords.xs('z', level=1).min(), coords.xs('z', level=1).max()]

        # Calculate the range for each axis
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]

        # Set the maximum range for all axes
        max_range = max(x_range, y_range, z_range)

        # Center the limits around the mean to make axes "equal"
        ax.set_xlim([x_limits[0] - (max_range - x_range) / 2, x_limits[1] + (max_range - x_range) / 2])
        ax.set_ylim([y_limits[0] - (max_range - y_range) / 2, y_limits[1] + (max_range - y_range) / 2])
        ax.set_zlim([z_limits[0] - (max_range - z_range) / 2, z_limits[1] + (max_range - z_range) / 2])


    def plot_3d_video(self, real_world_coords_allparts, labels, fps, video_name):
        temp = real_world_coords_allparts# .loc[100:] #.loc[400:]
        temp = temp.reset_index(drop=True)
        n_frames = len(temp)  # number of frames

        # Example setup
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=10, azim=0)
        # Setting up a colormap
        n_labels = len(labels['all'])
        viridis = plt.get_cmap('viridis')  # Alternatively, you can use: viridis = plt.cm.get_cmap('viridis')
        colors = viridis(np.linspace(0, 1, n_labels))
        # colors = colormaps['viridis'](np.linspace(0, 1, n_labels))
        # Define belts (fixed in position, change visibility instead of adding/removing)
        belt1_verts = [[(0, 0, 0), (470, 0, 0), (470, 53.5, 0), (0, 53.5, 0)]]
        belt2_verts = [[(471, 0, 0), (600, 0, 0), (600, 53.5, 0), (471, 53.5, 0)]]
        belt1 = Poly3DCollection(belt1_verts, facecolors='blue', edgecolors='none', alpha=0.2)
        belt2 = Poly3DCollection(belt2_verts, facecolors='blue', edgecolors='none', alpha=0.2)
        ax.add_collection3d(belt1)
        ax.add_collection3d(belt2)
        # Initialize lines for parts and skeleton lines
        lines = {part: ax.plot([], [], [], 'o-', ms=2, label=part, color=colors[i])[0] for i, part in
                 enumerate(labels['all'])}
        skeleton_lines = {pair: ax.plot([], [], [], 'black', linewidth=0.5)[0] for pair in micestuff['skeleton']}

        def init():
            ax.set_xlim(0, 600)
            ax.set_ylim(0, 53.5)
            ax.set_zlim(0, 53.5)
            ax.set_box_aspect([600 / 53.5, 1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            belt1.set_visible(True)
            belt2.set_visible(True)
            return [belt1, belt2] + list(lines.values()) + list(skeleton_lines.values())

        def update(frame):
            # Adjust visibility or other properties if needed
            belt1.set_visible(True)
            belt2.set_visible(True)
            for part, line in lines.items():
                x = np.array([temp.loc[frame, (part, 'x')]])
                y = np.array([temp.loc[frame, (part, 'y')]])
                z = np.array([temp.loc[frame, (part, 'z')]])
                line.set_data(x, y)
                line.set_3d_properties(z)
            for (start, end), s_line in skeleton_lines.items():
                xs = np.array([temp.loc[frame, (start, 'x')],
                      temp.loc[frame, (end, 'x')]])
                ys = np.array([temp.loc[frame, (start, 'y')],
                      temp.loc[frame, (end, 'y')]])
                zs = np.array([temp.loc[frame, (start, 'z')],
                      temp.loc[frame, (end, 'z')]])
                s_line.set_data(xs, ys)
                s_line.set_3d_properties(zs)
            ax.view_init(elev=10, azim=frame * 360 / n_frames)
            return [belt1, belt2] + list(lines.values()) + list(skeleton_lines.values())

        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)
        ani.save('%s.mp4'%video_name, writer='ffmpeg', fps=fps, dpi=300) #30fps
        plt.close(fig)  # Close the figure to avoid displaying it inline if running in a notebook


class GetALLRuns:
    def __init__(self, files=None, directory=None, overwrite=False):
        self.files = files
        self.directory = directory
        self.overwrite = overwrite

    def GetFiles(self):
        files = utils.Utils().GetlistofH5files(self.files,
                                               self.directory)  # gets dictionary of side, front and overhead files

        # Check if there are the same number of files for side, front and overhead before running run identification (which is reliant on all 3)
        if len(files['Side']) == len(files['Front']) == len(files['Overhead']):
            utils.Utils().checkFilenamesMouseID(
                files)  # before proceeding, check that mouse names are correctly labeled

        manual_file_name = os.path.join(paths['filtereddata_folder'], 'bad_frontcam_labels.csv')
        manual_labels = pd.read_csv(manual_file_name)
        manual_labels_vidnames = manual_labels.loc(axis=1)['video_name'].values
        manual_labels_by_video = {
            vid: df.sort_values('frame_number').reset_index(drop=True)
            for vid, df in manual_labels.groupby('video_name')
        }

        for j in range(0, len(files['Side'])):  # all csv files from each cam are same length so use side for all
            match = re.search(r'FAA-(\d+)', files['Side'][j])
            mouseID = match.group(1)
            pattern = "*%s*_mapped3D.h5" % mouseID
            dir = os.path.dirname(files['Side'][j])
            date = os.path.basename(files['Side'][j]).split('_')[1]

            if not glob.glob(os.path.join(dir, pattern)) or self.overwrite:
                try:
                    print(f"###############################################################"
                          f"\nMapping data for {mouseID}...\n###############################################################")
                    getdata = GetSingleExpData(files['Side'][j], files['Front'][j], files['Overhead'][j])

                    # check if data needs it, if so overwrite the front TransitionL labels with manual labels
                    vid_name = date + '_' + mouseID[-3:]
                    if vid_name in manual_labels_vidnames:
                        print(f"Overwriting front camera labels for {vid_name} with manual labels...")
                        df_view = getdata.DataframeCoors['front']
                        manual_df = manual_labels_by_video[vid_name]

                        # Build a full-length Series of frame-index-mapped manual data
                        manual_df = manual_df[['frame_number', 'x', 'y']].copy()
                        manual_df = manual_df.set_index('frame_number').sort_index()

                        # Reindex to all frames in df_view and forward-fill
                        reindexed_manual = manual_df.reindex(df_view.index, method='ffill').fillna(method='bfill')

                        # Apply filled values to the front view's TransitionL
                        df_view.loc[:, ('TransitionL', 'x')] = reindexed_manual['x'].values
                        df_view.loc[:, ('TransitionL', 'y')] = reindexed_manual['y'].values
                        df_view.loc[:, ('TransitionL', 'likelihood')] = pcutoff

                    getdata.map()
                except Exception as e:
                    print(f"Error processing {mouseID}: {e}")
            else:
                print(f"Data for {mouseID} already exists. Skipping...")

        print('All experiments have been mapped to real-world coordinates and saved.')

class GetDirsFromConditions(BaseConditionFiles):
    def process_final_directory(self, directory):
        # Just call GetALLRuns
        GetALLRuns(directory=directory, overwrite=self.overwrite).GetFiles()

# class GetDirsFromConditions:
#     def __init__(self, exp=None, speed=None, repeat_extend=None, exp_wash=None, day=None, vmt_type=None, vmt_level=None, prep=None, overwrite=False):
#         self.exp, self.speed, self.repeat_extend, self.exp_wash, self.day, self.vmt_type, self.vmt_level, self.prep, self.overwrite = (
#             exp, speed, repeat_extend, exp_wash, day, vmt_type, vmt_level, prep, overwrite)
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
#         subdirs = [d for d in os.listdir(current_path)
#                    if os.path.isdir(os.path.join(current_path, d))]
#
#         # Remove 'bin' from the list of subdirectories
#         subdirs = [sd for sd in subdirs if sd.lower() != 'bin']
#
#         # If we still have subdirectories (other than 'bin'), recurse
#         if subdirs:
#             print(f"Subdirectories found in {current_path}: {subdirs}")
#             for subdir in subdirs:
#                 full_subdir_path = os.path.join(current_path, subdir)
#                 self._process_subdirectories(full_subdir_path)
#         else:
#             # No subdirs (or only 'bin'), treat as final directory
#             print(f"Final directory: {current_path}")
#             try:
#                 GetALLRuns(directory=current_path, overwrite=self.overwrite).GetFiles()
#             except Exception as e:
#                 print(f"Error processing directory {current_path}: {e}")

def main():
    # Get all data
    #GetALLRuns(directory=directory).GetFiles()
    ### maybe instantiate first to protect entry point of my script

    # GetDirsFromConditions(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day1',
    #                       overwrite=False).get_dirs()
    # GetDirsFromConditions(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day2',
    #                       overwrite=False).get_dirs()
    # GetDirsFromConditions(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp', day='Day3',
    #                       overwrite=False).get_dirs()
    print("Analysing Repeats...")
    # GetDirsFromConditions(exp='APAChar', speed='LowHigh', repeat_extend='Repeats', exp_wash='Exp',
    #                       overwrite=True).get_dirs()
    # print("Analysing Extended: LowHigh...")
    GetDirsFromConditions(exp='APAChar', speed='LowHigh', repeat_extend='Extended', overwrite=False).get_dirs()
    print("Analysing Extended: LowMid...")
    GetDirsFromConditions(exp='APAChar', speed='LowMid', repeat_extend='Extended', overwrite=False).get_dirs()
    print("Analysing Extended: HighLow...")
    GetDirsFromConditions(exp='APAChar', speed='HighLow', repeat_extend='Extended', overwrite=False).get_dirs()
    print("Analysing PerceptionTest...")
    GetDirsFromConditions(exp='PerceptionTest', overwrite=False).get_dirs()

    # GetDirsFromConditions(exp='APAChar', speed='LowHigh', repeat_extend='Extended', day='Day2', overwrite=False).get_dirs()

if __name__ == "__main__":
    # directory = input("Enter the directory path: ")
    # main(directory)
    main()