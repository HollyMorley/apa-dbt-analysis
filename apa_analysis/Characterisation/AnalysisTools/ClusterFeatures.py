"""Group correlated features into clusters using k-means clustering."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt

from apa_analysis.config import global_settings
from apa_analysis.Characterisation import General_utils as gu
from apa_analysis.Characterisation import Plotting_utils as pu

def get_global_feature_matrix(feature_data, global_fs_mouse_ids, stride_number, stride_data, phase1, phase2, smooth=False):
    """
    Build the global runs-by-features matrix from the provided mouse IDs.
    Returns the transposed feature matrix (rows = features, columns = runs).
    """
    all_runs_data = []
    if stride_number == 'all':
        # Assume stride_data is a dict with stride numbers as keys.
        for sn in global_settings['stride_numbers']:
            for mouse in global_fs_mouse_ids:
                _, run_data, _, _, mask_phase1, mask_phase2 = gu.select_runs_data(
                    mouse, sn, feature_data, stride_data, phase1, phase2)
                run_data = run_data.T # todo added this as transpose was done in the function but not here originally
                if smooth:
                    run_data_smooth = medfilt(run_data, kernel_size=3)
                    run_data = pd.DataFrame(run_data_smooth, index=run_data.index, columns=run_data.columns)
                all_runs_data.append(run_data)
    else:
        for mouse in global_fs_mouse_ids:
            _, run_data, _, _, mask_phase1, mask_phase2 = gu.select_runs_data(
                mouse, stride_number, feature_data, stride_data, phase1, phase2)
            run_data = run_data.T # todo added this as transpose was done in the function but not here originally
            if smooth:
                run_data_smooth = medfilt(run_data, kernel_size=3)
                run_data = pd.DataFrame(run_data_smooth, index=run_data.index, columns=run_data.columns)
            all_runs_data.append(run_data)

    if not all_runs_data:
        raise ValueError("No run data found for global clustering.")

    global_data = pd.concat(all_runs_data, axis=0)
    # Transpose so that rows are features, columns are runs.
    feature_matrix = global_data.T
    return feature_matrix

