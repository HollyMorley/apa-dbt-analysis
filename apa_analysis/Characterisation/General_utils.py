"""Shared utility functions for data loading, normalization, and masking across characterisation analyses."""
import os
import pandas as pd
import inspect
from typing import Tuple
import itertools
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from apa_analysis.config import (
    global_settings
)
from helpers.config import *


def initialize_experiment(condition, exp, day, compare_condition, settings_to_log,
                          base_save_dir_no_c, condition_specific_settings) -> Tuple:
    # Collect stride data and create directories.
    stride_data, stride_data_compare = collect_stride_data(condition, exp, day, compare_condition)
    base_save_dir, base_save_dir_condition = set_up_save_dir(
        condition, exp, base_save_dir_no_c
    )
    script_name = os.path.basename(inspect.getfile(inspect.currentframe()))
    log_settings(settings_to_log, base_save_dir, script_name)

    # collect feature data from each mouse and stride
    feature_data = {}
    feature_data_compare = {}
    if global_settings['stride_numbers'] == [0]:
        strides_to_get = [-1, 0]
    else:
        strides_to_get = global_settings["stride_numbers"]
    for stride in strides_to_get:
        for mouse_id in condition_specific_settings[condition]['global_fs_mouse_ids']:
            # Load and preprocess data for each mouse.
            filtered_data_df = load_and_preprocess_data(mouse_id, stride, condition, exp, day, measures_list_manual_reduction)
            feature_data[(stride, mouse_id)] = filtered_data_df
        for mouse_id in condition_specific_settings[compare_condition]['global_fs_mouse_ids']:
            filtered_data_comparison_df = load_and_preprocess_data(mouse_id, stride, compare_condition, exp, day, measures_list_manual_reduction)
            feature_data_compare[(stride, mouse_id)] = filtered_data_comparison_df

    # Edit the feature data based on my manual feature reduction
    feature_data_df = pd.concat(feature_data, axis=0)
    feature_data_df = process_features(feature_data_df)

    feature_data_compare = pd.concat(feature_data_compare, axis=0)
    feature_data_compare = process_features(feature_data_compare)

    return feature_data_df, feature_data_compare, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition


def process_features(df):
    # Replace back heights with their mean.
    back_cols = [col for col in df.columns if col.startswith('back_height|')]
    df['back_height_mean'] = df[back_cols].mean(axis=1)
    df.drop(columns=back_cols, inplace=True)

    # Replace tail heights with their mean.
    tail_cols = [col for col in df.columns if col.startswith('tail_height|')]
    df['tail_height_mean'] = df[tail_cols].mean(axis=1)
    df.drop(columns=tail_cols, inplace=True)

    # Replace double, triple, and quadruple support with an average support value.
    double_name = [col for col in df.columns if col.startswith('double_support')]
    triple_name = [col for col in df.columns if col.startswith('triple_support')]
    quadruple_name = [col for col in df.columns if col.startswith('quadruple_support')]
    average_support_val = (2 * df[double_name].values + 3 * df[triple_name].values + 4 * df[
        quadruple_name].values) / 100
    df['average_support_val'] = average_support_val
    df.drop(columns=double_name + triple_name + quadruple_name, inplace=True)

    distance_sw_name = 'distance_from_midline|step_phase:0, all_vals:False, full_stride:False, buffer_size:0'
    distance_st_name = 'distance_from_midline|step_phase:1, all_vals:False, full_stride:False, buffer_size:0'
    df.loc(axis=1)[distance_st_name, distance_sw_name] = df.loc(axis=1)[distance_st_name, distance_sw_name].abs()

    return df

def load_stride_data(stride_data_path):
    stride_data = pd.read_hdf(stride_data_path, key='stride_info')
    return stride_data

def collect_stride_data(condition, exp, day, compare_condition):
    stride_data_path = None
    stride_data_path_compare = None
    if exp == 'Extended':
        stride_data_path = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\MEASURES_StrideInfo.h5")
        if compare_condition != 'None':
            stride_data_path_compare = os.path.join(paths['filtereddata_folder'], f"{compare_condition}\\{exp}\\MEASURES_StrideInfo.h5")
    elif exp == 'Repeats':
        stride_data_path = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\{day}\\MEASURES_StrideInfo.h5")
        if compare_condition != 'None':
            stride_data_path_compare = os.path.join(paths['filtereddata_folder'], f"{compare_condition}\\{exp}\\{day}\\MEASURES_StrideInfo.h5")
    stride_data = load_stride_data(stride_data_path)
    stride_data_compare = load_stride_data(stride_data_path_compare)
    return stride_data, stride_data_compare

def set_up_save_dir(condition, exp, base_save_dir_no_c):
    if global_settings["pca_CombineAllConditions"]:
        base_save_dir_no_c += '_allpca'

    if global_settings["use_LH_pcs"] and global_settings["use_LH_reg_model"]:
        base_save_dir_no_c += '_LHmodels'
    elif global_settings["use_LH_pcs"]:
        base_save_dir_no_c += '_LHpcsonly'

    if global_settings['normalise_to_LH_wash']:
        base_save_dir_no_c += '_LhWnrm'
    elif global_settings['normalise_wash_nullspace']:
        base_save_dir_no_c += '_WnrmNullspace'

    pcs_total= global_settings['pcs_to_show']
    pcs_using = global_settings['pcs_to_use']
    strides = str(global_settings['stride_numbers']).replace(' ', '').replace(',','').replace('[','').replace("'", "").replace(']','')
    phase_comp = str(global_settings['phases']).replace(' ', '').replace(',','').replace('[','').replace("'", "").replace(']','')

    if global_settings['residuals']:
        base_save_dir = base_save_dir_no_c + '_res' + f'_{strides}_{phase_comp}' #+ f'-PCStot={pcs_total}-PCSuse={pcs_using}'
    else:
        base_save_dir = base_save_dir_no_c + f'_{strides}_{phase_comp}' #+ f'-PCStot={pcs_total}-PCSuse={pcs_using}'
    base_save_dir_condition = os.path.join(base_save_dir, f'{condition}_{exp}')
    return base_save_dir, base_save_dir_condition

def log_settings(settings, log_dir, script_name):
    """
    Save the provided settings (a dict) to a timestamped log file.
    Also include the name of the running script and the current date.
    """
    # check if a log file already exists, if so delete it
    # to do above
    os.makedirs(log_dir, exist_ok=True)

    # Delete pre-existing log files starting with 'settings_log_'
    for filename in os.listdir(log_dir):
        if filename.startswith("settings_log_") and filename.endswith(".txt"):
            os.remove(os.path.join(log_dir, filename))

    # Get the current datetime as string
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Build the content of the log file.
    log_content = f"Script: {script_name}\nDate: {datetime.datetime.now()}\n\nSettings:\n"
    for key, value in settings.items():
        log_content += f"{key}: {value}\n"

    # Define the log file name.
    log_file = os.path.join(log_dir, f"settings_log_{now_str}.txt")
    with open(log_file, "w") as f:
        f.write(log_content)
    print(f"Settings logged to {log_file}")

def load_and_preprocess_data(mouse_id, stride_number, condition, exp, day, measures):
    """
    Load data for the specified mouse and preprocess it by selecting desired features and
    interpolating for missing values
    """
    if exp == 'Extended':
        filepath = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\MEASURES_single_kinematics_runXstride.h5")
    elif exp == 'Repeats':
        filepath = os.path.join(paths['filtereddata_folder'], f"{condition}\\{exp}\\Wash\\Exp\\{day}\\MEASURES_single_kinematics_runXstride.h5")
    else:
        raise ValueError(f"Unknown experiment type: {exp}")
    data_allmice = pd.read_hdf(filepath, key='single_kinematics')

    try:
        data = data_allmice.loc[mouse_id]
    except KeyError:
        raise ValueError(f"Mouse ID {mouse_id} not found in the dataset.")

    # # Build desired columns using the simplified build_desired_columns function
    # measures = measures_list_feature_reduction

    col_names = []
    for feature in measures.keys():
        if any(measures[feature]):
            if feature != 'signed_angle':
                for param in itertools.product(*measures[feature].values()):
                    param_names = list(measures[feature].keys())
                    formatted_params = ', '.join(f"{key}:{value}" for key, value in zip(param_names, param))
                    col_names.append((feature, formatted_params))
            else:
                for combo in measures['signed_angle'].keys():
                    col_names.append((feature, combo))
        else:
            col_names.append((feature, 'no_param'))

    col_names_trimmed = []
    for c in col_names:
        if np.logical_and('full_stride:True' in c[1], 'step_phase:None' not in c[1]):
            pass
        elif np.logical_and('full_stride:False' in c[1], 'step_phase:None' in c[1]):
            pass
        else:
            col_names_trimmed.append(c)

    selected_columns = col_names_trimmed


    filtered_data = data.loc[:, selected_columns]

    separator = '|'
    # Collapse MultiIndex columns to single-level strings including group info.
    filtered_data.columns = [
        f"{measure}{separator}{params}" if params != 'no_param' else f"{measure}"
        for measure, params in filtered_data.columns
    ]

    try:
        filtered_data = filtered_data.xs(stride_number, level='Stride', axis=0)
    except KeyError:
        raise ValueError(f"Stride number {stride_number} not found in the data.")

    # interpolate missing values
    filtered_data_interp = filtered_data.interpolate(method='linear', limit_direction='both')

    return filtered_data_interp

def normalize_df(df):
    normalize_mean = []
    normalize_std = []
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        normalize_mean.append(mean)
        normalize_std.append(std)
    normalize_std = np.array(normalize_std)
    normalize_mean = np.array(normalize_mean)
    return df, normalize_mean, normalize_std

def normalize_Xdr(Xdr):
    normalize_mean = []
    normalize_std = []
    for row in range(Xdr.shape[0]):
        mean = np.mean(Xdr[row, :])
        std = np.std(Xdr[row, :])
        Xdr[row, :] = (Xdr[row, :] - mean) / std # normalise each pcs's data
        normalize_mean.append(mean)
        normalize_std.append(std)
    normalize_std = np.array(normalize_std)
    normalize_mean = np.array(normalize_mean)
    return Xdr, normalize_mean, normalize_std

def get_mask_p1_p2(data, p1, p2):
    mask_p1 = data.index.get_level_values('Run').isin(
        expstuff['condition_exp_runs']['APAChar']['Extended'][p1])
    mask_p2 = data.index.get_level_values('Run').isin(
        expstuff['condition_exp_runs']['APAChar']['Extended'][p2])
    return mask_p1, mask_p2

def get_runs(scaled_data_df, stride_data, mouse_id, stride_number, phase1, phase2, ignore_stepping_limb=False):
    mask_phase1, mask_phase2 = get_mask_p1_p2(scaled_data_df, phase1, phase2)

    if not mask_phase1.any():
        raise ValueError(f"No runs found for phase '{phase1}'.")
    if not mask_phase2.any():
        raise ValueError(f"No runs found for phase '{phase2}'.")

    run_numbers_phase1 = scaled_data_df.index[mask_phase1]
    run_numbers_phase2 = scaled_data_df.index[mask_phase2]
    run_numbers = list(run_numbers_phase1) + list(run_numbers_phase2)

    # Determine stepping limbs.
    if not ignore_stepping_limb:
        stepping_limbs = [determine_stepping_limbs(stride_data, mouse_id, run, stride_number)
                          for run in run_numbers]
    else:
        stepping_limbs = None

    return run_numbers, stepping_limbs, mask_phase1, mask_phase2

def determine_stepping_limbs(stride_data, mouse_id, run, stride_number):
    """
    Determine the stepping limb (ForepawL or ForepawR) for a given MouseID, Run, and Stride.

    Parameters:
        stride_data (pd.DataFrame): Stride data DataFrame.
        mouse_id (str): Identifier for the mouse.
        run (str/int): Run identifier.
        stride_number (int): Stride number.

    Returns:
        str: 'ForepawL' or 'ForepawR' indicating the stepping limb.
    """
    paws = stride_data.loc(axis=0)[mouse_id, run].xs('SwSt_discrete', level=1, axis=1).isna().any().index[
        stride_data.loc(axis=0)[mouse_id, run].xs('SwSt_discrete', level=1, axis=1).isna().any()]
    if len(paws) > 1:
        raise ValueError(f"Multiple paws found for Mouse {mouse_id}, Run {run}.")
    else:
        return paws[0]

def select_runs_data(mouse_id, stride_number, feature_data, stride_data, phase1, phase2, ignore_stepping_limb=False):
    try:
        scaled_data_df = feature_data.loc(axis=0)[stride_number, mouse_id]
        # Get runs and stepping limbs for each phase.
        run_numbers, stepping_limbs, mask_phase1, mask_phase2 = get_runs(scaled_data_df, stride_data, mouse_id,
                                                                         stride_number, phase1, phase2, ignore_stepping_limb=ignore_stepping_limb)

        # Select only runs from the two phases in feature data
        selected_mask = mask_phase1 | mask_phase2
        selected_scaled_data_df = scaled_data_df.loc[selected_mask].T

        return scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2
    except KeyError:
        raise ValueError(f"Stride: {stride_number} and mouse: {mouse_id} not found in the data.")


def create_mouse_save_directory(base_dir, mouse_id, stride_number, phase1, phase2):
    """
    Create a directory path based on the settings to save plots.

    Parameters:
        base_dir (str): Base directory where plots will be saved.
        mouse_id (str): Identifier for the mouse.
        stride_number (int): Stride number used in analysis.
        phase1 (str): First experimental phase.
        phase2 (str): Second experimental phase.

    Returns:
        str: Path to the directory where plots will be saved.
    """
    # Construct the directory name
    dir_name = f"Mouse_{mouse_id}_Stride_{stride_number}_Compare_{phase1}_vs_{phase2}"
    save_path = os.path.join(base_dir, dir_name)

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    return save_path

def get_pc_weights(pca_pred, stride):
    mouse_stride_preds = [pred for pred in pca_pred if pred.stride == stride]
    mouse_names = [pred.mouse_id for pred in mouse_stride_preds]

    pc_weights = []
    for mouse_pred in mouse_stride_preds:
        pc_weights.append(mouse_pred.pc_weights[0]) if mouse_pred.pc_weights.ndim == 2 else pc_weights.append(mouse_pred.pc_weights)
    pc_weights = np.array(pc_weights)

    # dataframe with names
    pc_weights_df = pd.DataFrame(pc_weights, index=mouse_names, columns=np.arange(global_settings["pcs_to_use"]) + 1)

    return pc_weights_df


def get_and_save_pcs_of_interest(pca_pred, stride_numbers, savedir, conditions=None, reglda='reg', accmse='acc', lesion_significance=False, LH_pred=None):
    if reglda == 'reg':
        from apa_analysis.Characterisation.AnalysisTools import Regression as src
    elif reglda == 'lda':
        from apa_analysis.Characterisation.AnalysisTools import LDA as src

    all_pcs_of_interest = defaultdict(list)
    all_pcs_of_interest_criteria = []  # list to collect per-stride DataFrames

    for s in stride_numbers:
        PC_sigs, mean_accs, mouse_uniform, num_mice, total_mice = src.calculate_PC_prediction_significances(pca_pred, s, conditions=conditions, mice_thresh_percent=0.75, accmse=accmse, lesion_significance=lesion_significance)
        uniformity_display = np.array([f"{num}/{total_mice}" for num in num_mice])

        # if reglda == 'reg':
        if accmse == 'acc':
            of_interest = np.logical_and.reduce((mean_accs >= 0.6, PC_sigs <= 0.05, mouse_uniform))
        elif accmse == 'mse':
            of_interest = np.logical_and.reduce((mean_accs <= 0.2, PC_sigs <= 0.05, mouse_uniform)) # todo choose better mse cut-off
        # elif reglda == 'lda':
        #     of_interest = np.logical_and(PC_sigs <= 0.05, mouse_uniform)

        pcs_of_interest = np.where(of_interest)[0] + 1  # Convert to 1-indexed
        print(f"Stride {s}: PCs of interest: {pcs_of_interest}")
        all_pcs_of_interest[s] = pcs_of_interest

        # if reglda == 'reg':
        if accmse == 'acc':
            pcs_of_interest_criteria_df = pd.DataFrame(
                [PC_sigs, mean_accs, uniformity_display],
                index=['PC_sigs', 'mean_accs', 'uniformity'],
                columns=np.arange(len(PC_sigs)) + 1  # PC numbers as column labels
            )
        elif accmse == 'mse':
            pcs_of_interest_criteria_df = pd.DataFrame(
                [PC_sigs, mean_accs, uniformity_display],
                index=['PC_sigs', 'mean_mse', 'uniformity'],
                columns=np.arange(global_settings["pcs_to_use"]) + 1
            )
        # elif reglda == 'lda':
        #     pcs_of_interest_criteria_df = pd.DataFrame(
        #         [PC_sigs, mouse_uniform],
        #         index=['PC_sigs', 'uniformity'],
        #         columns=np.arange(global_settings["pcs_to_use"]) + 1
        #     )

        if LH_pred is not None:
            LH_weights_df = get_pc_weights(LH_pred, s)
            current_weights_df = get_pc_weights(pca_pred, s)

            # Match shared mice
            shared_mice = LH_weights_df.index.intersection(current_weights_df.index)
            LH_weights_df = LH_weights_df.loc[shared_mice]
            current_weights_df = current_weights_df.loc[shared_mice]

            # Calculate per mouse per PC sign
            LH_signs = np.sign(LH_weights_df)
            current_signs = np.sign(current_weights_df)

            # Check where signs differ per mouse per PC
            sign_diff = LH_signs != current_signs

            # Count number of mice with different signs per PC
            num_switch = sign_diff.sum(axis=0)
            total_mice = len(shared_mice)

            # Format as 'num/total' string per PC
            num_switch_str = [f"{n}/{total_mice}" for n in num_switch]

            # Add as a new row to dataframe
            pcs_of_interest_criteria_df.loc['Sign switch'] = num_switch_str

        pcs_of_interest_criteria_df.columns.name = 'PCs'
        all_pcs_of_interest_criteria.append(pcs_of_interest_criteria_df)

    criteria_multi_df = pd.concat(all_pcs_of_interest_criteria, keys=stride_numbers, names=['stride', 'measure'])

    rows = []
    for stride, pc_array in all_pcs_of_interest.items():
        if len(pc_array) == 0:
            rows.append((stride, np.nan))
        else:
            for pc in pc_array:
                rows.append((stride, pc))
    all_pcs_of_interest_df = pd.DataFrame(rows, columns=['stride', 'pc_of_interest'])
    all_pcs_of_interest_df.set_index(['stride'], inplace=True)

    # Save the CSV files.
    all_pcs_of_interest_df.to_csv(os.path.join(savedir, 'pcs_of_interest.csv'))
    criteria_multi_df.to_csv(os.path.join(savedir, 'pcs_of_interest_criteria.csv'))

    # For each stride, create a table figure with formatted values and bold the rows for PCs of interest.
    for s in stride_numbers:
        # Extract the criteria for the current stride and transpose: rows are PC numbers, columns are measures.
        stride_criteria = criteria_multi_df.loc(axis=0)[s].T

        # Define a helper function to format values: two decimals for floats; others converted to string.
        def format_val(val):
            if isinstance(val, (float, np.floating)):
                if val >= .01:
                    return f"{val:.3f}"
                else:
                    return f"<.001"
            else:
                return str(val)

        formatted_df = stride_criteria.applymap(format_val)
        # if reglda == 'reg':
        if accmse == 'acc':
            formatted_df.columns = ['pval', 'acc', 'uniformity'] if LH_pred is None else ['pval', 'acc', 'uniformity', 'Sign switch']
        elif accmse == 'mse':
            formatted_df.columns = ['pval', 'mse', 'uniformity']
        # elif reglda == 'lda':
        #     formatted_df.columns = ['pval', 'uniformity']

        # Create the figure and table.
        column_width = 1  # adjust as needed for clarity
        fig_width = column_width * (len(formatted_df.columns) + 1)  # +1 for row labels
        fig, ax = plt.subplots(figsize=(fig_width, 4))

        ax.axis('tight')
        ax.axis('off')

        tbl = ax.table(
            cellText=formatted_df.values.tolist(),
            colLabels=formatted_df.columns,
            rowLabels=formatted_df.index,
            loc='center'
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.2, 1.2)

        # Bold the rows where the PC is in pcs_of_interest for this stride.
        # Note: our table rows are labeled with PCs as 1-indexed (since columns were defined with np.arange(...) + 1).
        pcs_interest = [pc for pc in all_pcs_of_interest[s]]
        # Loop over table cells. Data and row label cells have row index >= 0.
        for (row, col), cell in tbl.get_celld().items():
            if row >= 0:
                # The row label is available in the cell at (row, -1). Compare it to pcs_interest.
                try:
                    label_cell = tbl[(row, -1)]
                    label_text = label_cell.get_text().get_text()
                    # Convert label to integer if possible.
                    try:
                        label_val = int(label_text)
                    except ValueError:
                        label_val = None
                except KeyError:
                    label_val = None

                if label_val in pcs_interest:
                    cell.get_text().set_fontweight('bold')

        plt.subplots_adjust(left=0.3, right=0.7)

        # Save the figure as an SVG file.
        plt.savefig(os.path.join(savedir, f'pcs_of_interest_criteria_table_{s}.svg'), format='svg', dpi=300)
        plt.savefig(os.path.join(savedir, f'pcs_of_interest_criteria_table_{s}.png'), format='png', dpi=300)
        plt.close(fig)

    return all_pcs_of_interest_df, criteria_multi_df

def compute_residuals(group, s, savedir):
    plot_dir = os.path.join(savedir, 'Speed')
    os.makedirs(plot_dir, exist_ok=True)

    speeds = group.loc(axis=1)['walking_speed|bodypart:Tail1, speed_correct:True']
    res = pd.DataFrame(index=group.index, columns=group.columns)
    for col in group.columns:
        safe_name = short_names.get(col, col)

        # Get the x values from the speed column and y values from the current feature.
        x = speeds.values
        y = group[col].values

        # Compute the linear regression parameters.
        m, c = np.polyfit(x, y, 1)
        predicted = m * x + c

        # Store the residuals (difference between actual and predicted values).
        res[col] = y - predicted

        # Sort x (and corresponding predicted values) for a smooth line plot.
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        pred_sorted = predicted[sort_idx]

        # Create the plot for the current feature.
        plt.figure(figsize=(4, 3))
        plt.scatter(x, y, label='Data points', s=10)
        plt.plot(x_sorted, pred_sorted, color='red', label='Fitted regression line')
        plt.xlabel('Speed', fontsize=7)
        plt.ylabel(col, fontsize=7)
        plt.title(f"{safe_name}\n{s}",fontsize=7)
        plt.tick_params(axis='both', which='both', labelsize=7)
        # plt.legend()
        plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.2)

        plt.savefig(os.path.join(plot_dir, f'Speed_regression_{safe_name}_{s}.png'), format='png')
        plt.savefig(os.path.join(plot_dir, f'Speed_regression_{safe_name}_{s}.svg'), format='svg')
        plt.close()

    #res.drop(columns=['walking_speed|bodypart:Tail1, speed_correct:True'], inplace=True)

    return res


