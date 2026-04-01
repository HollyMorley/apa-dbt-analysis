import os
import re
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import medfilt
from sklearn.cluster import KMeans
import ast
from joblib import Parallel, delayed
import random
from collections import Counter
from tqdm import tqdm
import datetime
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from scipy.optimize import curve_fit
import hashlib
from scipy.signal import savgol_filter
from curlyBrace import curlyBrace

from helpers.config import *
from apa_analysis.config import global_settings
from apa_analysis.Legacy_methods.LogisticRegression import compute_lasso_regression
from apa_analysis.Legacy_methods.FeatureSelection import rfe_feature_selection, random_forest_feature_selection
from apa_analysis.Legacy_methods.SignificanceTesting import ShufflingTest_ComparePhases, ShufflingTest_CompareConditions


@dataclass
class PredictionData:
    mouse_id: str
    x_vals: List[float]
    smoothed_scaled_pred: np.ndarray
    group_id: Optional[int] = None  # None if individual; set for grouped predictions

    def as_tuple(self) -> Tuple:
        """Return a tuple representation for backward compatibility."""
        return (self.mouse_id, self.x_vals, self.smoothed_scaled_pred, self.group_id)

    # Add __getitem__ so that indexing works like with a tuple.
    def __getitem__(self, index):
        if index == 0:
            return self.mouse_id
        elif index == 1:
            return self.x_vals
        elif index == 2:
            return self.smoothed_scaled_pred
        elif index == 3:
            return self.group_id
        else:
            raise IndexError("Index out of range for PredictionData")

@dataclass
class FeatureWeights:
    mouse_id: str
    feature_weights: pd.Series
    group_id: Optional[int] = None  # None if individual; set for grouped predictions

    def as_tuple(self) -> Tuple:
        """Return a tuple representation for backward compatibility."""
        return (self.mouse_id, self.feature_weights, self.group_id)

    def __getitem__(self, index):
        if index == 0:
            return self.mouse_id
        elif index == 1:
            return self.feature_weights
        elif index == 2:
            return self.group_id
        else:
            raise IndexError("Index out of range for FeatureWeights")

@dataclass
class ContributionData:
    mouse_id: str
    unique_feature_contribution: pd.Series
    unique_pc_contribution: pd.Series
    single_feature_contribution: pd.Series
    single_pc_contribution: pd.Series
    group_id: Optional[int] = None

    def as_tuple(self) -> Tuple:
        """Return a tuple representation for backward compatibility."""
        return (self.mouse_id, self.unique_feature_contribution,
                self.unique_pc_contribution, self.single_feature_contribution,
                self.single_pc_contribution, self.group_id)

        # Add __getitem__ so that indexing works like with a tuple.

    def __getitem__(self, index):
        if index == 0:
            return self.mouse_id
        elif index == 1:
            return self.unique_feature_contribution
        elif index == 2:
            return self.unique_pc_contribution
        elif index == 3:
            return self.single_feature_contribution
        elif index == 4:
            return self.single_pc_contribution
        else:
            raise IndexError("Index out of range for ContributionData")


def make_safe_feature_name(feature, max_length=50):
    # Replace disallowed characters and remove commas.
    safe_feature = re.sub(r'[<>:"/\\|?*]', '_', feature)
    safe_feature = re.sub(r'\s+', '_', safe_feature)
    safe_feature = re.sub(r',', '', safe_feature)

    # If the safe_feature is too long, truncate it and append a hash.
    if len(safe_feature) > max_length:
        hash_suffix = hashlib.md5(safe_feature.encode()).hexdigest()[:6]
        safe_feature = safe_feature[:max_length - 7] + '_' + hash_suffix
    return safe_feature

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
    Load data for the specified mouse and preprocess it by selecting desired features,
    imputing missing values, and standardizing.
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

    filtered_data_imputed = filtered_data.fillna(filtered_data.mean())

    if filtered_data_imputed.isnull().sum().sum() > 0:
        print("Warning: There are still missing values after imputation.")

    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(filtered_data_imputed)
    # scaled_data_df = pd.DataFrame(scaled_data, index=filtered_data_imputed.index,
    #                               columns=filtered_data_imputed.columns)
    return filtered_data_imputed

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


def select_runs_data(mouse_id, stride_number, feature_data, stride_data, phase1, phase2):
    try:
        scaled_data_df = feature_data.loc(axis=0)[stride_number, mouse_id]
        # Get runs and stepping limbs for each phase.
        run_numbers, stepping_limbs, mask_phase1, mask_phase2 = get_runs(scaled_data_df, stride_data, mouse_id,
                                                                         stride_number, phase1, phase2)

        # Select only runs from the two phases in feature data
        selected_mask = mask_phase1 | mask_phase2
        selected_scaled_data_df = scaled_data_df.loc[selected_mask].T

        return scaled_data_df, selected_scaled_data_df, run_numbers, stepping_limbs, mask_phase1, mask_phase2
    except KeyError:
        raise ValueError(f"Stride: {stride_number} and mouse: {mouse_id} not found in the data.")

def unified_feature_selection(feature_data_df, y, c, method='regression', cv=5, n_iterations=100, fold_assignment=None, save_file=None, overwrite_FeatureSelection=False):
    """
    Unified feature selection function that can be used both in local (single-mouse)
    and global (aggregated) cases.

    Parameters:
      - feature_data_df: DataFrame with features as rows and samples as columns.
      - y: target vector (e.g. binary labels)
      - method: 'rfecv', 'rf', or 'regression'
      - cv: number of folds for cross-validation (if applicable)
      - n_iterations: number of shuffles for regression-based selection
      - fold_assignment: optional pre-computed fold assignment dictionary for regression-based selection.
      - save_file: if provided, a file path to load/save results.

    Returns:
      - selected_features: the list (or index) of selected features.
      - results: For 'regression' method, a DataFrame of per-feature results; otherwise, None.
    """
    # Check if a results file exists and we are not overwriting.
    if save_file is not None and os.path.exists(save_file) and not overwrite_FeatureSelection:
        if method == 'regression':
            all_feature_accuracies_df = pd.read_csv(save_file, index_col=0)
            # Convert the string representation back into dictionaries.
            all_feature_accuracies_df['iteration_diffs'] = all_feature_accuracies_df['iteration_diffs'].apply(
                ast.literal_eval)
            print("Global feature selection results loaded from file.")
            selected_features = all_feature_accuracies_df[all_feature_accuracies_df['significant']].index
            return selected_features, all_feature_accuracies_df
        else:
            df = pd.read_csv(save_file)
            # Assuming selected features were saved in a column 'selected_features'
            selected_features = df['selected_features'].tolist()
            print("Global feature selection results loaded from file.")
            return selected_features, None

    # Compute feature selection using the chosen method.
    if method == 'rfecv':
        print("Running RFECV for feature selection.")
        selected_features, rfecv_model = rfe_feature_selection(feature_data_df, y, cv=cv, min_features_to_select=5, C=c)
        print(f"RFECV selected {rfecv_model.n_features_} features.")
        results = None
    elif method == 'rf':
        print("Running Random Forest for feature selection.")
        selected_features, rf_model = random_forest_feature_selection(feature_data_df, y)
        print(f"Random Forest selected {len(selected_features)} features.")
        results = None
    elif method == 'regression':
        N = feature_data_df.shape[1]
        if fold_assignment is None:
            indices = list(range(N))
            random.shuffle(indices)
            fold_assignment = {i: (j % cv + 1) for j, i in enumerate(indices)}
        features = list(feature_data_df.index)
        results = Parallel(n_jobs=-1)(
            delayed(process_single_feature)(
                feature,
                feature_data_df.loc[feature].values,
                fold_assignment,
                y,
                list(range(N)),
                cv,
                n_iterations
            )
            for feature in tqdm(features, desc="Unified regression-based feature selection")
        )
        all_feature_accuracies = dict(results)
        all_feature_accuracies_df = pd.DataFrame.from_dict(all_feature_accuracies, orient='index')
        # Mark features as significant if the 99th percentile of shuffled differences is below zero.
        all_feature_accuracies_df['significant'] = 0 > all_feature_accuracies_df['iteration_diffs'].apply(
            lambda d: np.percentile(list(d.values()), 99)
        )
        selected_features = all_feature_accuracies_df[all_feature_accuracies_df['significant']].index
        results = all_feature_accuracies_df
    else:
        raise ValueError("Unknown method specified for feature selection.")

    # Save results if a file path is provided.
    if save_file is not None:
        if method == 'regression':
            results.to_csv(save_file)
        else:
            # Save the list of selected features in a simple CSV.
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            pd.DataFrame({'selected_features': selected_features}).to_csv(save_file, index=False)

    if method == 'regression':
        return selected_features, results
    else:
        return selected_features, None

def process_single_feature(feature, X, fold_assignment, y_reg, run_numbers, nFolds, n_iterations):
    fold_true_accuracies = []
    iteration_diffs_all = {i: [] for i in range(n_iterations)}
    #knn = KNeighborsClassifier(n_neighbors=5)
    # find custom weights based on class imbalance
    class_counts = Counter(y_reg)
    n_samples = len(y_reg)
    custom_weights = {
        cls: n_samples / (len(class_counts) * count)
        for cls, count in class_counts.items()
    }
    #knn = ImbalancedKNN(n_neighbors=5, weights=custom_weights)

    # Loop over folds
    for fold in range(1, nFolds + 1):
        test_mask = np.array([fold_assignment[run] == fold for run in run_numbers])
        train_mask = ~test_mask

        # Training on the training set
        X_fold_train = X[train_mask].reshape(1, -1) # Get feature values across training runs in current fold
        y_reg_fold_train = y_reg[train_mask]  #  Create y (regression target) - 1 for phase1, 0 for phase2 - for this fold

        w, _ = compute_lasso_regression(X_fold_train, y_reg_fold_train) # Run logistic regression on single feature to get weights
        #w, _ = compute_regression(X_fold_train, y_reg_fold_train) # Run logistic regression on single feature to get weights
        #knn.fit(X_fold_train.T, y_reg_fold_train)

        # Testing on the test set
        X_fold_test = X[test_mask].reshape(1, -1) # Get feature values across test runs in current fold
        y_reg_fold_test = y_reg[test_mask] # Create y (regression target) - 1 for phase1, 0 for phase2 - for this fold
        y_pred = np.dot(w, X_fold_test) # Get accuracy from test set
        y_pred[y_pred > 0] = 1 # change y_pred +ves to 1 and -ves to 0
        y_pred[y_pred < 0] = 0 # change y_pred +ves to 1 and -ves to 0
        #y_pred = knn.predict(X_fold_test.T) ## add in weights for
        feature_accuracy_test = balanced_accuracy(y_reg_fold_test.T, y_pred.T) # Get balanced accuracy from test set
        fold_true_accuracies.append(feature_accuracy_test)


        # For each iteration: shuffle and compute difference in accuracy.
        for i in range(n_iterations):
            X_shuffled = X.copy()
            random.shuffle(X_shuffled)

            X_shuffled_fold_train = X_shuffled[train_mask].reshape(1, -1)
            # Run logistic regression on shuffled data
            w, _ = compute_lasso_regression(X_shuffled_fold_train, y_reg_fold_train)
            #w, _ = compute_regression(X_shuffled_fold_train, y_reg_fold_train)
            # knn.fit(X_shuffled_fold_train.T, y_reg_fold_train)

            X_shuffled_fold_test = X_shuffled[test_mask].reshape(1, -1)
            y_pred_shuffle = np.dot(w, X_shuffled_fold_test)
            y_pred_shuffle[y_pred_shuffle > 0] = 1
            y_pred_shuffle[y_pred_shuffle < 0] = 0
            # y_pred_shuffle = knn.predict(X_shuffled_fold_test.T)
            shuffled_feature_accuracy_test = balanced_accuracy(y_reg_fold_test.T, y_pred_shuffle.T)

            # Difference between true and shuffled accuracy.
            feature_diff = shuffled_feature_accuracy_test - feature_accuracy_test #feature_accuracy_test - shuffled_feature_accuracy_test
            iteration_diffs_all[i].append(feature_diff)

    # Average differences across folds
    avg_feature_diffs = {i: np.mean(iteration_diffs_all[i]) for i in range(n_iterations)}
    avg_true_accuracy = np.mean(fold_true_accuracies)

    return feature, {"true_accuracy": avg_true_accuracy, "iteration_diffs": avg_feature_diffs}

def normalize_Xdr(Xdr):
    normalize_mean = []
    normalize_std = []
    for row in range(Xdr.shape[0]):
        mean = np.mean(Xdr[row, :])
        std = np.std(Xdr[row, :])
        # Xdr[row, :] = Xdr[row, :]/np.max(np.abs(Xdr[row, :]))
        Xdr[row, :] = (Xdr[row, :] - mean) / std # normalise each pcs's data
        normalize_mean.append(mean)
        normalize_std.append(std)
    normalize_std = np.array(normalize_std)
    normalize_mean = np.array(normalize_mean)
    return Xdr, normalize_mean, normalize_std

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

def balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy: average of sensitivity and specificity.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        float: balanced accuracy score
    """
    # Calculate true positives and true negatives
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))

    # Calculate false positives and false negatives
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    # Calculate sensitivity (true positive rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Balanced accuracy is the average of sensitivity and specificity
    return (sensitivity + specificity)/2

def shuffle_single(feature, raw_features):
    shuffled = raw_features.copy()
    for col in raw_features.columns:
        if col != feature:
            shuffled[col] = np.random.permutation(shuffled[col].values)
    return shuffled

def shuffle_unique(feature, raw_features):
    shuffled = raw_features.copy()
    shuffled.loc(axis=0)[feature] = np.random.permutation(shuffled.loc(axis=0)[feature].values)
    return shuffled

def plot_feature_accuracy(single_cvaccuracy, mouseID, save_path, x_label_offset_multiple=0.95, title_suffix="Single_Feature_cvaccuracy"):
    """
    Plots the single-feature model accuracy values.

    Parameters:
        single_cvaccuracy (dict): Mapping of feature names to accuracy values.
        save_path (str): Directory where the plot will be saved.
        title_suffix (str): Suffix for the plot title and filename.
    """
    df = pd.DataFrame(list(single_cvaccuracy.items()), columns=['Feature', 'cvaccuracy'])
    if not df['Feature'].str.contains('PC').all():
        df['Display'] = df['Feature'].apply(lambda x: short_names.get(x, x))
        df['cluster'] = df['Feature'].map(manual_clusters['cluster_mapping'])
        df = df.dropna(subset=['cluster'])
        df['cluster'] = df['cluster'].astype(int)

        order_map = {feat: idx for idx, feat in enumerate(manual_clusters['cluster_mapping'].keys())}
        df['order'] = df['Feature'].map(order_map)
        df = df.sort_values(by='order').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, max(8, int(len(df) * 0.3))))
    if not df['Feature'].str.contains('PC').all():
        sns.barplot(data=df, x='cvaccuracy', y='Display', palette='viridis')
    else:
        sns.barplot(data=df, x='cvaccuracy', y='Feature', palette='viridis')
    plt.xlim(0.5, 1.0)
    plt.title(f'{mouseID}\nSingle Feature Model accuracy ' + title_suffix)
    plt.xlabel('accuracy')
    plt.ylabel('')
    plt.tight_layout()

    lower_x_lim = df['cvaccuracy'].min()
    upper_x_lim = df['cvaccuracy'].max()
    x_range = upper_x_lim - lower_x_lim

    if not df['Feature'].str.contains('PC').all():
        for i, cl in enumerate(sorted(df['cluster'].unique())):
            group_indices = df.index[df['cluster'] == cl].tolist()
            x_pos = lower_x_lim - x_range * x_label_offset_multiple
            y_positions = group_indices
            y0 = min(y_positions) - 0.05
            y1 = max(y_positions) + 0.05
            k_r = 0.1
            span = abs(y1 - y0)
            desired_depth = 0.1  # or any value that gives you the uniform look you want
            k_r_adjusted = desired_depth / span if span != 0 else k_r

            # Alternate the int_line_num value for every other cluster:
            base_line_num = 2
            int_line_num = base_line_num + 5 if i % 2 else base_line_num

            cluster_label = [k for k, v in manual_clusters['cluster_values'].items() if v == cl][0]

            add_vertical_brace_curly(ax, y0, y1, x_pos, k_r=k_r_adjusted, int_line_num=int_line_num,
                                     xoffset=0.2, label=cluster_label, rot_label=90)
        plt.subplots_adjust(left=0.35)
    plt.savefig(os.path.join(save_path, f"Single_Feature_cvaccuracy_{title_suffix}.png"), dpi=300)
    plt.close()

def plot_unique_delta_accuracy(unique_delta_accuracy, mouseID, save_path, x_label_offset_multiple=0.65, title_suffix="Unique_Δaccuracy"):
    """
    Plots the unique contribution (Δaccuracy) for each feature.

    Parameters:
        unique_delta_accuracy (dict): Mapping of feature names to unique Δaccuracy.
        save_path (str): Directory where the plot will be saved.
        title_suffix (str): Suffix for the plot title and filename.
    """

    df = pd.DataFrame(list(unique_delta_accuracy.items()), columns=['Feature', 'Unique_Δaccuracy'])
    if not df['Feature'].str.contains('PC').all():
        df['Display'] = df['Feature'].apply(lambda x: short_names.get(x, x))
        df['cluster'] = df['Feature'].map(manual_clusters['cluster_mapping'])
        df = df.dropna(subset=['cluster'])
        df['cluster'] = df['cluster'].astype(int)

        order_map = {feat: idx for idx, feat in enumerate(manual_clusters['cluster_mapping'].keys())}
        df['order'] = df['Feature'].map(order_map)
        df = df.sort_values(by='order').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, max(8, int(len(df) * 0.3))))
    if not df['Feature'].str.contains('PC').all():
        sns.barplot(data=df, x='Unique_Δaccuracy', y='Display', palette='magma')
    else:
        sns.barplot(data=df, x='Unique_Δaccuracy', y='Feature', palette='magma')
    plt.title(f'{mouseID}\nUnique Feature Contributions (Δaccuracy) ' + title_suffix)
    plt.xlabel('Δaccuracy')
    plt.ylabel('')
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()

    lower_x_lim = df['Unique_Δaccuracy'].min()
    upper_x_lim = df['Unique_Δaccuracy'].max()
    x_range = upper_x_lim - lower_x_lim

    plt.xlim(lower_x_lim - x_range*0.1, upper_x_lim + x_range*0.1)

    if not df['Feature'].str.contains('PC').all():
        for i, cl in enumerate(sorted(df['cluster'].unique())):
            group_indices = df.index[df['cluster'] == cl].tolist()
            x_pos = lower_x_lim - x_range * x_label_offset_multiple
            y_positions = group_indices
            y0 = min(y_positions) - 0.05
            y1 = max(y_positions) + 0.05
            k_r = 0.1
            span = abs(y1 - y0)
            desired_depth = 0.1  # or any value that gives you the uniform look you want
            k_r_adjusted = desired_depth / span if span != 0 else k_r

            # Alternate the int_line_num value for every other cluster:
            base_line_num = 2
            int_line_num = base_line_num + 5 if i % 2 else base_line_num

            cluster_label = [k for k, v in manual_clusters['cluster_values'].items() if v == cl][0]

            add_vertical_brace_curly(ax, y0, y1, x_pos, k_r=k_r_adjusted, int_line_num=int_line_num,
                                     xoffset=0, label=cluster_label, rot_label=90)
        plt.subplots_adjust(left=0.35)

    plt.savefig(os.path.join(save_path, f"Unique_delta_accuracy_{title_suffix}.png"), dpi=300)
    plt.close()

def plot_run_prediction(scaled_data_df, run_pred, run_pred_smoothed, save_path, mouse_id, phase1, phase2, stride_number, scale_suffix, dataset_suffix):
    # median filter smoothing on run_pred
    #run_pred_smoothed = medfilt(run_pred[0], kernel_size=5)

    # plot run prediction
    plt.figure(figsize=(8, 6))
    plt.plot(scaled_data_df.index, run_pred[0], color='lightblue', ls='--', label='Prediction')
    plt.plot(scaled_data_df.index, run_pred_smoothed, color='blue', ls='-', label='Smoothed Prediction')
    # Exp phases
    plt.vlines(x=9.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='red', linestyle='--')
    plt.vlines(x=109.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='red', linestyle='--')
    # Days
    plt.vlines(x=39.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)
    plt.vlines(x=79.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)
    plt.vlines(x=39.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)
    plt.vlines(x=119.5, ymin=run_pred[0].min(), ymax=run_pred[0].max(), color='black', linestyle='--', alpha=0.5)

    # plot a shaded box over x=60 to x=110 and x=135 to x=159, ymin to ymax
    plt.fill_between(x=range(60, 110), y1=run_pred[0].min(), y2=run_pred[0].max(), color='gray', alpha=0.1)
    plt.fill_between(x=range(135, 160), y1=run_pred[0].min(), y2=run_pred[0].max(), color='gray', alpha=0.1)

    plt.title(f'Run Prediction for Mouse {mouse_id} - {phase1} vs {phase2}')
    plt.xlabel('Run Number')
    plt.ylabel('Prediction')

    legend_elements = [Line2D([0], [0], color='red', linestyle='--', label='Experimental Phases'),
                          Line2D([0], [0], color='black', linestyle='--', label='Days'),
                          Patch(facecolor='gray', edgecolor='black', alpha=0.1, label='Training Portion'),
                          Line2D([0], [0], color='lightblue', label='Prediction', linestyle='--'),
                          Line2D([0], [0], color='blue', label='Smoothed Prediction', linestyle='-')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(False)
    plt.gca().yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"Run_Prediction_{phase1}_vs_{phase2}_stride{stride_number}_{scale_suffix}_{dataset_suffix}.png"), dpi=300)
    plt.close()

def plot_aggregated_run_predictions(aggregated_data: List[PredictionData],
                                    save_dir: str, phase1: str, phase2: str,
                                    stride_number: int, condition_label: str,
                                    normalization_method: str = 'maxabs'):
    plt.figure(figsize=(10, 8))

    # Collect common x-axis values.
    all_x_vals = []
    for data in aggregated_data:
        all_x_vals.extend(data.x_vals)
    global_min_x = min(all_x_vals)
    global_max_x = max(all_x_vals)
    common_npoints = max(len(data.x_vals) for data in aggregated_data)
    common_x = np.linspace(global_min_x, global_max_x, common_npoints)

    plt.axvspan(9.5, 109.5, color='lightblue', alpha=0.2)

    interpolated_curves = []
    for data in aggregated_data:
        mouse_id = data.mouse_id
        x_vals = data.x_vals
        smoothed_pred = data.smoothed_scaled_pred

        # Normalize the curve.
        if normalization_method == 'zscore':
            mean_val = np.mean(smoothed_pred)
            std_val = np.std(smoothed_pred)
            normalized_curve = (smoothed_pred - mean_val) / std_val if std_val != 0 else smoothed_pred
        elif normalization_method == 'maxabs':
            max_abs = max(abs(smoothed_pred.min()), abs(smoothed_pred.max()))
            normalized_curve = smoothed_pred / max_abs if max_abs != 0 else smoothed_pred
        else:
            normalized_curve = smoothed_pred

        interp_curve = np.interp(common_x, x_vals, normalized_curve)
        interpolated_curves.append(interp_curve)

        plt.plot(common_x, interp_curve, label=f'Mouse {mouse_id}', alpha=0.3, color='grey')

    all_curves_array = np.vstack(interpolated_curves)
    mean_curve = np.mean(all_curves_array, axis=0)
    plt.plot(common_x, mean_curve, color='black', linewidth=2, label='Mean Curve')

    #plt.vlines(x=[9.5, 109.5], ymin=-1, ymax=1, color='red', linestyle='-')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(f'Aggregated {normalization_method.upper()} Scaled Run Predictions for {phase1} vs {phase2}, stride {stride_number}\n{condition_label}')
    plt.xlabel('Run Number')
    plt.ylabel('Normalized Prediction (Smoothed)')
    if normalization_method == 'maxabs':
        plt.ylim(-1, 1)
    #plt.legend(loc='upper right')
    plt.grid(False)
    #plt.gca().yaxis.grid(True)
    plt.tight_layout()

    save_path_full = os.path.join(save_dir,
                                  f"Aggregated_{normalization_method.upper()}_Run_Predictions_{phase1}_vs_{phase2}_stride{stride_number}_{condition_label}.png")
    plt.savefig(save_path_full, dpi=300)
    plt.close()


def plot_aggregated_run_predictions_by_group(aggregated_data: List[PredictionData],
                                             save_dir: str, phase1: str, phase2: str,
                                             stride_number: int, condition_label: str,
                                             normalization_method: str = 'maxabs'):
    plt.figure(figsize=(10, 8))

    # Collect common x-axis values and group IDs.
    all_x_vals = []
    group_ids = []
    for data in aggregated_data:
        all_x_vals.extend(data.x_vals)
        group_ids.append(data.group_id)

    global_min_x = min(all_x_vals)
    global_max_x = max(all_x_vals)
    common_npoints = max(len(data.x_vals) for data in aggregated_data)
    common_x = np.linspace(global_min_x, global_max_x, common_npoints)

    # Define a color mapping for groups.
    unique_groups = sorted(set(group_ids))
    cmap = plt.get_cmap("tab10")
    group_color_dict = {group: cmap(i) for i, group in enumerate(unique_groups)}

    interpolated_curves = []
    for data in aggregated_data:
        # Access PredictionData fields directly.
        mouse_id = data.mouse_id
        x_vals = data.x_vals
        smoothed_pred = data.smoothed_scaled_pred
        group_id = data.group_id

        # Normalize curve based on the chosen method.
        if normalization_method == 'zscore':
            mean_val = np.mean(smoothed_pred)
            std_val = np.std(smoothed_pred)
            normalized_curve = (smoothed_pred - mean_val) / std_val if std_val != 0 else smoothed_pred
        elif normalization_method == 'maxabs':
            max_abs = max(abs(smoothed_pred.min()), abs(smoothed_pred.max()))
            normalized_curve = smoothed_pred / max_abs if max_abs != 0 else smoothed_pred
        else:
            normalized_curve = smoothed_pred

        interp_curve = np.interp(common_x, x_vals, normalized_curve)
        interpolated_curves.append(interp_curve)

        plt.plot(common_x, interp_curve, label=f'Mouse {mouse_id} (Group {group_id})',
                 alpha=0.3, color=group_color_dict[group_id])

    all_curves_array = np.vstack(interpolated_curves)
    mean_curve = np.mean(all_curves_array, axis=0)
    plt.plot(common_x, mean_curve, color='black', linewidth=2, label='Mean Curve')

    plt.vlines(x=[9.5, 109.5], ymin=-1, ymax=1, color='red', linestyle='-')
    plt.title(
        f'Aggregated {normalization_method.upper()} Scaled Run Predictions for {phase1} vs {phase2}, stride {stride_number}\n{condition_label}')
    plt.xlabel('Run Number')
    plt.ylabel('Normalized Prediction (Smoothed)')

    if normalization_method == 'maxabs':
        plt.ylim(-1, 1)

    plt.legend(loc='upper right')
    plt.grid(False)
    plt.gca().yaxis.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir,
                             f"Aggregated_{normalization_method.upper()}_Run_Predictions_ByGroup_{phase1}_vs_{phase2}_stride{stride_number}_{condition_label}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def cluster_weights(w, loadings_df, save_path, mouse_id, phase1, phase2, n_clusters=2):
    """
    Cluster the regression weights (w, from PCA space) using KMeans and plot the original clustering,
    including the cluster centroids.

    This function:
      1. Clusters the regression weight vector (w) in PCA space.
      2. Plots a scatter of PCA component indices vs. regression weights, colored by cluster,
         and overlays the centroids (each centroid is labeled with its corresponding cluster number).
      3. Transforms the regression weights back to feature space (via loadings_df) to assign features
         to clusters and saves this mapping to CSV for later use.

    Parameters:
      - w: Regression weight vector from PCA space (numpy array of shape (n_components,)).
      - loadings_df: DataFrame of PCA loadings (rows: features, columns: components).
      - save_path: Directory to save the clustering results and visualization.
      - mouse_id, phase1, phase2: Identifiers used in filenames.
      - n_clusters: Number of clusters to form (default: 2).

    Returns:
      - cluster_df: DataFrame mapping each feature to its weight and assigned cluster label.
      - kmeans: The fitted KMeans model (from clustering w).
    """
    # --- Step 1: Cluster the regression weights (w) in PCA space ---
    w_2d = w.reshape(-1, 1)  # KMeans expects 2D data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(w_2d)
    cluster_labels_pca = kmeans.labels_  # one label per PCA component

    # --- Step 2: Visualize the original clustering of regression weights ---
    plt.figure(figsize=(8, 6))
    component_indices = np.arange(len(w))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k in range(n_clusters):
        color = colors[k % len(colors)]
        mask = (cluster_labels_pca == k)
        plt.scatter(component_indices[mask], w[mask], color=color, label=f'Cluster {k}', s=100, alpha=0.8)
        # Compute centroid for this cluster:
        centroid_x = np.mean(component_indices[mask])
        centroid_y = np.mean(w[mask])
        plt.scatter(centroid_x, centroid_y, color=color, marker='X', s=200, edgecolor='black',
                    label=f'Centroid {k}')
    plt.xlabel('PCA Component Index')
    plt.ylabel('Regression Weight')
    plt.title(f'Regression Weights Clustering for Mouse {mouse_id} ({phase1} vs {phase2})')
    plt.legend()
    vis_path = os.path.join(save_path, f'cluster_regression_weights_{mouse_id}_{phase1}_vs_{phase2}.png')
    plt.savefig(vis_path, dpi=300)
    plt.close()

    # --- Step 3: Map the clustering back into feature space ---
    # Compute feature-space weights as the weighted combination of PCA loadings.
    feature_weights = loadings_df.dot(w).squeeze()
    if isinstance(feature_weights, pd.DataFrame):
        feature_weights = feature_weights.iloc[:, 0]

    n_features = loadings_df.shape[0]
    cluster_scores = np.zeros((n_features, n_clusters))
    for j in range(len(w)):
        cluster_idx = cluster_labels_pca[j]
        cluster_scores[:, cluster_idx] += loadings_df.iloc[:, j].values * w[j]

    # Assign each feature to the cluster whose (absolute) score is largest.
    feature_cluster = np.argmax(np.abs(cluster_scores), axis=1)

    # --- Step 4: Create and save a DataFrame with the clustering results ---
    cluster_df = pd.DataFrame({
        'feature': loadings_df.index,
        'weight': feature_weights,
        'cluster': feature_cluster
    })

    cluster_file = os.path.join(save_path, f'cluster_weights_{mouse_id}_{phase1}_vs_{phase2}.csv')
    cluster_df.to_csv(cluster_file, index=False)
    print(f"Cluster weights saved to: {cluster_file}")

    return cluster_df, kmeans


def plot_weights_in_feature_space(feature_weights, save_path, mouse_id, phase1, phase2, stride_number, condition, x_label_offset=0.5):
    """
    Plot the feature-space weights as a bar plot with feature names on the x-axis.
    """
    # Create a DataFrame for plotting and sort for easier visualization
    df = pd.DataFrame({'Feature': feature_weights.index, 'weight': feature_weights.values})
    df['Display'] = df['Feature'].apply(lambda x: short_names.get(x, x))
    df['cluster'] = df['Feature'].map(manual_clusters['cluster_mapping'])
    df = df.dropna(subset=['cluster'])
    df['cluster'] = df['cluster'].astype(int)

    order_map = {feat: idx for idx, feat in enumerate(manual_clusters['cluster_mapping'].keys())}
    df['order'] = df['Feature'].map(order_map)
    df = df.sort_values(by='order').reset_index(drop=True)

    #sort df by weight
    #df = df.sort_values(by='weight', ascending=False)
    fig, ax = plt.subplots(figsize=(14, max(8, int(len(df) * 0.3))))
    sns.barplot(x='weight', y='Display', data=df, palette='viridis')
    plt.xlabel('Weight Value')
    plt.ylabel('')
    plt.title(f'Feature Weights in Original Space for Mouse {mouse_id} ({phase1} vs {phase2})')
    plt.tight_layout()

    lower_x_lim = df['weight'].min()
    upper_x_lim = df['weight'].max()
    x_range = upper_x_lim - lower_x_lim

    if not df['Feature'].str.contains('PC').all():
        for i, cl in enumerate(sorted(df['cluster'].unique())):
            group_indices = df.index[df['cluster'] == cl].tolist()
            x_pos = lower_x_lim - x_range * x_label_offset
            y_positions = group_indices
            y0 = min(y_positions) - 0.05
            y1 = max(y_positions) + 0.05
            k_r = 0.1
            span = abs(y1 - y0)
            desired_depth = 0.1  # or any value that gives you the uniform look you want
            k_r_adjusted = desired_depth / span if span != 0 else k_r

            # Alternate the int_line_num value for every other cluster:
            base_line_num = 2
            int_line_num = base_line_num + 0.5 if i % 2 else base_line_num

            cluster_label = [k for k, v in manual_clusters['cluster_values'].items() if v == cl][0]

            add_vertical_brace_curly(ax, y0, y1, x_pos, k_r=k_r_adjusted, int_line_num=int_line_num,
                                     xoffset=0.2, label=cluster_label, rot_label=90)
        plt.subplots_adjust(left=0.35)
        plt.xlim(lower_x_lim, upper_x_lim)

    plot_file = os.path.join(save_path, f'feature_space_weights_{mouse_id}_{phase1}_vs_{phase2}_stride{stride_number}_{condition}.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Vertical feature-space weights plot saved to: {plot_file}")

def get_top_features(contrib_dict, stride_features, phase1, phase2, stride_number, n_features, quartile, average):
    """
    Get the top n features (out of the specifically selected features for that stride) based on the mean of their weights across mice.
    :param weights_dict:
    :param stride_features:
    :param phase1:
    :param phase2:
    :param stride_number:
    :param n_features:
    :return:
    """
    filtered_contribution = {
        mouse_id: (contribs.single_feature_contribution )
        for (mouse_id, p1, p2, s), contribs in contrib_dict.items()
        if p1 == phase1 and p2 == phase2 and s == stride_number
    }
    contrib_df = pd.DataFrame(filtered_contribution)
    # filter contribs_df by stride_features
    contrib_df = contrib_df.loc[stride_features]

    # find mean of feature contribs
    if average == 'mean':
        average_contribs = contrib_df.mean(axis=1)
    elif average == 'median':
        average_contribs = contrib_df.median(axis=1)

    # find top quartile of features by absolute contribution, preserving original sign
    threshold = average_contribs.abs().quantile(quartile)
    top_features_quartile = average_contribs[average_contribs.abs() >= threshold]

    # find top n features of absolute contribution, preserving original sign
    top_features_top10 = average_contribs.nlargest(n_features)
    #top_features_top10 = top_features_top10 * median_contribs.loc[top_features_top10.index].apply(np.sign)
    return top_features_top10, top_features_quartile, average_contribs


def select_top_quantile_features(feature_weights: pd.Series, quantile: float = 0.9):
    # Compute the threshold based on the absolute values of the feature weights
    threshold = feature_weights.abs().quantile(quantile)
    # Filter the features that have an absolute loading greater than or equal to the threshold
    selected_features = feature_weights[feature_weights.abs() >= threshold]
    return selected_features

def get_top_feature_data(data_dict, phase1, phase2, stride_number, top_features)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the top features for phase 1 and phase 2.
    :param data_dict:
    :param phase1:
    :param phase2:
    :param stride_number:
    :param top_features:
    :return: dictionary of top features for phase 1 and phase 2
    """
    # Create an empty dictionary to hold the new DataFrames.
    data = {}

    for key, df in data_dict.items():
        # Unpack the key.
        mouse, phase1, phase2, stride_num = key
        if phase1 == phase1 and phase2 == phase2 and stride_num == stride_number:
            # Create the new grouping key.
            new_key = (phase1, phase2, stride_num)

            # Make a copy of the DataFrame so we don't modify the original.
            df_copy = df.copy()
            # Create a MultiIndex for the rows: level 0 is the mouse ID, level 1 is the original run index.
            df_copy.index = pd.MultiIndex.from_product([[mouse], df_copy.index], names=['mouse', 'Run'])

            # If we already have a DataFrame for this group, concatenate the new data.
            if new_key in data:
                data[new_key] = pd.concat([data[new_key], df_copy])
            else:
                data[new_key] = df_copy

    phase1_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase1]
    phase2_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase2]

    # Get feature data for phase 1 and phase 2.
    phase1_mask = data[phase1, phase2, stride_number].index.get_level_values('Run').isin(phase1_runs)
    phase2_mask = data[phase1, phase2, stride_number].index.get_level_values('Run').isin(phase2_runs)

    top_features_phase1 = data[phase1, phase2, stride_number][phase1_mask].loc(axis=1)[top_features.index]
    top_features_phase2 = data[phase1, phase2, stride_number][phase2_mask].loc(axis=1)[top_features.index]

    return (top_features_phase1, top_features_phase2)

def plot_top_feature_phase_comparison(top_feature_data, base_save_dir, phase1, phase2, stride_number, condition_label, connect_mice):
    top_features_phase1, top_features_phase2 = top_feature_data

    mean_mouse_phase1 = top_features_phase1.groupby(level='mouse').mean()
    mean_mouse_phase2 = top_features_phase2.groupby(level='mouse').mean()

    name_exclusions = ['buffer_size:0','all_vals:False','full_stride:False','step_phase:None']

    for f in top_features_phase1.columns:
        # remove name exclusions
        feature_name_bits = f.split(', ')
        feature_name_to_keep = []
        for name_bit in feature_name_bits:
            if name_bit not in name_exclusions:
                feature_name_to_keep.append(name_bit)
        feature_name = ', '.join(feature_name_to_keep)

        fig, ax = plt.subplots(figsize=(4, 6))

        feature_mean_p1 = mean_mouse_phase1[f].mean()
        feature_mean_p2 = mean_mouse_phase2[f].mean()

        feature_sem_p1 = mean_mouse_phase1[f].sem()
        feature_sem_p2 = mean_mouse_phase2[f].sem()

        p_val, Eff_obs, EffNull_vals = ShufflingTest_ComparePhases(Obs_p1=top_features_phase1.loc(axis=1)[f],
                                                                   Obs_p2=top_features_phase2.loc(axis=1)[f],
                                                                   mouseObs_p1=mean_mouse_phase1[f],
                                                                   mouseObs_p2=mean_mouse_phase2[f],
                                                                   phase1=phase1,
                                                                   phase2=phase2,
                                                                   type='mean')

        #stat, p_val = stats.ttest_rel(mean_mouse_phase1[f], mean_mouse_phase2[f])
        # Optionally, adjust the p-value threshold or use multiple stars for very small p-values:
        if p_val < 0.001:
            significance = "***"
        elif p_val < 0.01:
            significance = "**"
        elif p_val < 0.05:
            significance = "*"  # you can use "**" or "***" for lower p-values if desired
        else:
            significance = "n.s."  # not significant

        # plot mouse means as line with phases along x axis and feature values along y axis. scatter plot for each mouse
        for mouse in mean_mouse_phase1.index:
            if connect_mice == False:
                ax.scatter([1], mean_mouse_phase1.loc[mouse, f], color='blue', s=100, alpha=0.5)
                ax.scatter([2], mean_mouse_phase2.loc[mouse, f], color='red', s=100, alpha=0.5)
            else:
                ax.plot([1, 2], [mean_mouse_phase1.loc[mouse, f], mean_mouse_phase2.loc[mouse, f]], color='black')
                # add label next to phase 2 point
                ax.text(2.05, mean_mouse_phase2.loc[mouse, f], mouse, fontsize=6, verticalalignment='center')


        # plot mean of feature values for each phase
        if connect_mice == False:
            ax.errorbar([1, 2], [feature_mean_p1, feature_mean_p2],
                        color='black', linestyle='--', marker='o', markersize=5, yerr=[feature_sem_p1*1.645, feature_sem_p2*1.645], capsize=5)

            # Compute ymax from the errorbars as before.
            ymax = max(feature_mean_p1 + feature_sem_p1*1.645, feature_mean_p2 + feature_sem_p2*1.645) + 0.2
            h = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            # Draw bracket
            ax.plot([1, 1, 2, 2], [ymax, ymax + h, ymax + h, ymax], lw=1.5, c='k')
            # Add significance text using the computed p-value:
            ax.text(1.5, ymax + h, significance, ha='center', va='bottom', color='k', fontsize=16)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([phase1, phase2])
        ax.set_xlim(0.5, 2.5)
        ax.set_ylabel(feature_name)
        if connect_mice == False:
            ax.set_title(f'Stride {stride_number}\nShuffling Test, Err=90% CI')
        else:
            ax.set_title(f'Stride {stride_number}')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        save_dir = os.path.join(base_save_dir, f'top_feature_descriptives\\stride{stride_number}')
        os.makedirs(save_dir, exist_ok=True)
        safe_feature_name = make_safe_feature_name(feature_name)
        save_path = os.path.join(save_dir, f'{safe_feature_name}__{phase1}vs{phase2}_stride{stride_number}_{condition_label}_cm{connect_mice}.png')
        fig.savefig(save_path, dpi=300)
        plt.close()

def plot_top_feature_phase_comparison_connected_means(top_feature_data, base_save_dir, phase1, phase2, stride_number, condition_label):
    top_features_phase1, top_features_phase2 = top_feature_data

    mean_mouse_phase1 = top_features_phase1.groupby(level='mouse').mean()
    mean_mouse_phase2 = top_features_phase2.groupby(level='mouse').mean()

    for f in top_features_phase1.columns:
        display_name = short_names.get(f, f)
        fig, ax = plt.subplots(figsize=(4, 6))

        feature_mean_p1 = mean_mouse_phase1[f].mean()
        feature_mean_p2 = mean_mouse_phase2[f].mean()

        feature_sem_p1 = mean_mouse_phase1[f].sem()
        feature_sem_p2 = mean_mouse_phase2[f].sem()

        p_val, Eff_obs, EffNull_vals = ShufflingTest_ComparePhases(Obs_p1=top_features_phase1.loc(axis=1)[f],
                                                                   Obs_p2=top_features_phase2.loc(axis=1)[f],
                                                                   mouseObs_p1=mean_mouse_phase1[f],
                                                                   mouseObs_p2=mean_mouse_phase2[f],
                                                                   phase1=phase1,
                                                                   phase2=phase2,
                                                                   type='mean')

        #stat, p_val = stats.ttest_rel(mean_mouse_phase1[f], mean_mouse_phase2[f])
        # Optionally, adjust the p-value threshold or use multiple stars for very small p-values:
        if p_val < 0.001:
            significance = "***"
        elif p_val < 0.01:
            significance = "**"
        elif p_val < 0.05:
            significance = "*"  # you can use "**" or "***" for lower p-values if desired
        else:
            significance = "n.s."  # not significant

        # plot mouse means as line with phases along x axis and feature values along y axis. scatter plot for each mouse
        for mouse in mean_mouse_phase1.index:
            ax.plot([1, 2], [mean_mouse_phase1.loc[mouse, f], mean_mouse_phase2.loc[mouse, f]], color='grey')
            # add label next to phase 2 point
            ax.text(2.05, mean_mouse_phase2.loc[mouse, f], mouse, fontsize=6, verticalalignment='center', color='grey')

        # plot median of feature values for each phase
        ax.errorbar([1, 2], [feature_mean_p1, feature_mean_p2],
                    color='black', linestyle='--', marker='o', markersize=5, yerr=[feature_sem_p1*1.645, feature_sem_p2*1.645], capsize=5)

        # Compute ymax from the errorbars as before.
        ymax = max(feature_mean_p1 + feature_sem_p1*1.645, feature_mean_p2 + feature_sem_p2*1.645) + 0.2
        h = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        # Draw bracket
        ax.plot([1, 1, 2, 2], [ymax, ymax + h, ymax + h, ymax], lw=1.5, c='k')
        # Add significance text using the computed p-value:
        ax.text(1.5, ymax + h, significance, ha='center', va='bottom', color='k', fontsize=16)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([phase1, phase2])
        ax.set_xlim(0.5, 2.5)
        ax.set_ylabel(display_name)
        ax.set_title(f'Stride {stride_number}\nShuffling Test (mean), Err=90% CI')
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        save_dir = os.path.join(base_save_dir, f'top_feature_descriptives\\stride{stride_number}')
        os.makedirs(save_dir, exist_ok=True)
        safe_feature_name = make_safe_feature_name(display_name)
        save_path = os.path.join(save_dir, f'{safe_feature_name}__{phase1}vs{phase2}_stride{stride_number}_{condition_label}_mean.png')
        fig.savefig(save_path, dpi=300)
        plt.close()

def plot_common_across_strides_top_features(top_feats, real_dict, condition_label, base_save_dir):
    # find common top_feats between strides
    all_top_feats = []
    all_top_feats_data = []
    for (phase1, phase2, stride), top_feats_list in top_feats.items():
        all_top_feats.append(list(top_feats_list.index))
    common_top_feats = list(set.intersection(*map(set, all_top_feats)))

    # plot the common features across all strides
    common_top_feats_real_data = {}
    for (phase1, phase2, stride), data in real_dict.items():
        p1, p2 = data
        top_p1 = p1.loc(axis=1)[common_top_feats]
        top_p2 = p2.loc(axis=1)[common_top_feats]
        common_top_feats_real_data[(phase1, phase2, stride)] = [top_p1, top_p2]

    for (phase1, phase2, stride), data in common_top_feats_real_data.items():
        p1, p2 = data
        mean_mouse_p1 = p1.groupby(level='mouse').mean()
        mean_mouse_p2 = p2.groupby(level='mouse').mean()

        n_feats = len(common_top_feats)
        # Create a single row of subplots, one for each feature
        fig, axes = plt.subplots(1, n_feats, figsize=(2 * n_feats, 5))
        if n_feats == 1:
            axes = [axes]

        fig.suptitle(f'Stride {stride}: {phase1} vs {phase2}', fontsize=16)

        for i, f in enumerate(common_top_feats):
            ax = axes[i]

            # Calculate means and SEM (using 90% CI: SEM*1.645)
            feature_mean_p1 = mean_mouse_p1[f].mean()
            feature_mean_p2 = mean_mouse_p2[f].mean()
            feature_sem_p1 = mean_mouse_p1[f].sem() * 1.645
            feature_sem_p2 = mean_mouse_p2[f].sem() * 1.645

            # Compute p-value and significance using your shuffling test.
            p_val, _, _ = ShufflingTest_ComparePhases(
                p1.loc(axis=1)[f], p2.loc(axis=1)[f],
                mean_mouse_p1[f], mean_mouse_p2[f],
                phase1, phase2, type='mean'
            )
            if p_val < 0.001:
                significance = "***"
            elif p_val < 0.01:
                significance = "**"
            elif p_val < 0.05:
                significance = "*"
            else:
                significance = "n.s."

            # Offsets so that scatter points don't overlap
            offset = 0.1

            # Plot each mouse's data as scatter points and connect them with a line
            for mouse in mean_mouse_p1.index:
                val1 = mean_mouse_p1.loc[mouse, f]
                val2 = mean_mouse_p2.loc[mouse, f]
                # ax.scatter(1 - offset, val1, color='blue', alpha=0.5)
                # ax.scatter(2 + offset, val2, color='red', alpha=0.5)
                ax.plot([1 - offset, 2 + offset], [val1, val2], color='grey', linewidth=1, alpha=0.7)
                # Optionally add the mouse label next to the phase2 point
                #ax.text(2.05, val2, mouse, fontsize=6, verticalalignment='center', color='grey')

            # Plot mean markers with error bars (90% CI)
            ax.errorbar([1 - offset, 2 + offset],
                        [feature_mean_p1, feature_mean_p2],
                        yerr=[feature_sem_p1, feature_sem_p2],
                        color='black', linestyle='--', marker='o', markersize=5, capsize=5)

            # Determine y position for significance bracket (above the highest errorbar)
            ymin = min(feature_mean_p1 - feature_sem_p1, feature_mean_p2 - feature_sem_p2)
            ymax = max(feature_mean_p1 + feature_sem_p1, feature_mean_p2 + feature_sem_p2)
            yrange = ymax - ymin
            sig_y = ymax + 0.2 * yrange
            h = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            # Draw bracket using the x positions (1-offset and 2+offset)
            ax.plot([1 - offset, 1 - offset, 2 + offset, 2 + offset],
                    [sig_y, sig_y + h, sig_y + h, sig_y], lw=1.5, c='k')
            # Place significance text centered above the bracket.
            ax.text((1 - offset + 2 + offset) / 2, sig_y + h, significance,
                    ha='center', va='bottom', color='k', fontsize=16)

            # Set x-axis ticks and labels for the phases
            ax.set_xticks([1, 2])
            ax.set_xticklabels([phase1, phase2])
            ax.set_xlim(0.8, 2.2)

            # Remove grid lines and subplot borders
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # Instead of a subplot title, put the feature name as an extra label at the bottom.
           # ax.text(0.5, -0.25, f, transform=ax.transAxes, ha='center', va='center', fontsize=8, rotation=45)
            # Optionally, remove the y-axis label if not needed.
            ax.set_ylabel(f)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        save_dir = os.path.join(base_save_dir, f'top_feature_descriptives\\stride{stride}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,
                                 f'CommonFeaturesByStride__{phase1}vs{phase2}_stride{stride}_{condition_label}_mean.png')
        fig.savefig(save_path, dpi=300)
        plt.close()

        # Define your stride color mapping
        stride_color_mapping = {
            -3: plt.cm.Blues(0.2),
            -2: plt.cm.Blues(0.45),
            -1: plt.cm.Blues(0.7),
            0: plt.cm.Blues(0.99)
        }

        # Create a new figure for aggregated phase means across strides
        n_feats = len(common_top_feats)
        fig, axes = plt.subplots(1, n_feats, figsize=(2 * n_feats, 5), sharey=False)
        if n_feats == 1:
            axes = [axes]

        # For consistency, extract a representative phase1 and phase2 label
        # (Assuming they are the same across strides; if not, adjust accordingly.)
        # Here we take the first key from common_top_feats_real_data:
        first_key = next(iter(common_top_feats_real_data))
        phase1_label, phase2_label, _ = first_key

        for i, f in enumerate(common_top_feats):
            ax = axes[i]
            # For each stride, plot the phase means with connecting lines
            # (Note: if a given stride is missing data for a feature, you might skip it.)
            for (phase1, phase2, stride), data in common_top_feats_real_data.items():
                p1, p2 = data
                mean_mouse_p1 = p1.groupby(level='mouse').mean()
                mean_mouse_p2 = p2.groupby(level='mouse').mean()
                # Compute the mean for the feature in both phases
                med1 = mean_mouse_p1[f].mean()
                med2 = mean_mouse_p2[f].mean()
                color = stride_color_mapping.get(stride, 'black')
                ax.plot([1, 2], [med1, med2], color=color, marker='o', markersize=5, label=f'Stride {stride}')

            # Set x-axis ticks to represent the two phases
            ax.set_xticks([1, 2])
            ax.set_xticklabels([phase1_label, phase2_label])
            ax.set_xlim(0.8, 2.2)
            # Remove grid lines and all subplot borders
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Place the feature name at the bottom (as if it were an x-axis label)
            ax.set_ylabel(f)

        # Add a legend to the first subplot (avoid duplicate legends)
        axes[0].legend(fontsize=8, loc='upper left')
        fig.suptitle(f'Common Top Feature Meadians Across Strides: {phase1_label} vs {phase2_label}', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        save_dir = os.path.join(base_save_dir, f'top_feature_descriptives')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,
                                 f'CommonFeaturesByStride__{phase1}vs{phase2}_allstrides_{condition_label}_mean.png')
        fig.savefig(save_path, dpi=300)
        plt.close()


def plot_top_feature_phase_comparison_differences(top_feature_data, base_save_dir, phase1, phase2, stride_number,
                                                  condition_label):
    top_features_phase1, top_features_phase2 = top_feature_data

    mean_mouse_phase1 = top_features_phase1.groupby(level='mouse').mean()
    mean_mouse_phase2 = top_features_phase2.groupby(level='mouse').mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    sorted_features = sorted(top_features_phase1.columns)
    len_features = len(sorted_features)

    display_names = []
    for fidx, f in enumerate(sorted_features):
        display_name = short_names.get(f, f)
        display_names.append(display_name)

        # Calculate the difference between the two phases.
        feature_phase_diff = mean_mouse_phase2[f] - mean_mouse_phase1[f]
        # scale max abs
        feature_phase_diff = feature_phase_diff / max(abs(feature_phase_diff.min()), abs(feature_phase_diff.max()))
        feature_phase_diff_mean = feature_phase_diff.mean()
        feature_phase_diff_sem = feature_phase_diff.sem()

        # Introduce jitter along x for each point
        jitter = np.random.uniform(-0.1, 0.1, len(feature_phase_diff))
        x_values = np.full(len(feature_phase_diff), fidx) + jitter

        ax.scatter(x_values, feature_phase_diff, color='red', alpha=0.5)
        ax.scatter([fidx], feature_phase_diff_mean, color='black')
        ax.errorbar([fidx], feature_phase_diff_mean, yerr=feature_phase_diff_sem*1.645, color='black', capsize=5)

    ax.axhline(0, color='black', linestyle='--', alpha=0.2)

    # Set x-axis ticks to each feature and rotate labels
    ax.set_xticks(np.arange(len_features))
    ax.set_xticklabels(display_names, rotation=45, ha='right')

    # Adjust x-axis limits so that all labels are fully in frame.
    ax.set_xlim(-0.5, len_features - 0.5)
    ax.set_ylabel('Mean Feature Difference (max abs)')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f'Feature Differences between {phase2} - {phase1} - Stride {stride_number}\nErr=90% CI')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.7)

    save_dir = os.path.join(base_save_dir, f'top_feature_descriptives\\stride{stride_number}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
                             f'FeatureDifferences__{phase2}-{phase1}_stride{stride_number}_{condition_label}.png')
    fig.savefig(save_path, dpi=300)
    plt.close()

def plot_top_feature_phase_comparison_differences_BothConditions(top_feature_data, top_feature_data_compare, base_save_dir, phase1, phase2, stride_number, condition_label, compare_condition_label, suffix):
    # Unpack & group
    p1, p2 = top_feature_data
    p1c, p2c = top_feature_data_compare

    mean1 = p1.groupby(level='mouse').mean()
    mean2 = p2.groupby(level='mouse').mean()
    mean1c = p1c.groupby(level='mouse').mean()
    mean2c = p2c.groupby(level='mouse').mean()

    name_exclusions = ['buffer_size:0', 'all_vals:False', 'full_stride:False', 'step_phase:None']

    # find common features
    features = mean1.columns.intersection(mean1c.columns)

    # Compute phase‑differences and scale
    diff = (mean2.loc(axis=1)[features] - mean1.loc(axis=1)[features])
    diffc = (mean2c.loc(axis=1)[features] - mean1c.loc(axis=1)[features])
    scaled = diff.div(diff.abs().max(axis=0))
    scaledc = diffc.div(diffc.abs().max(axis=0))

    # Restrict both DataFrames to the same mice
    common_mice = diff.index.intersection(diffc.index)
    diff = diff.loc[common_mice]
    diffc = diffc.loc[common_mice]
    scaled = scaled.loc[common_mice]
    scaledc = scaledc.loc[common_mice]
    p1 = p1.loc[common_mice]
    p2 = p2.loc[common_mice]
    p1c = p1c.loc[common_mice]
    p2c = p2c.loc[common_mice]

    means = scaled.mean()
    sems = scaled.sem()
    means_comp = scaledc.mean()
    sems_comp = scaledc.sem()

    # significance testing
    pvals = {}
    for f in features:
        if len(common_mice) < 2:
            pvals[f] = np.nan
        else:
            p_val, _, _ = ShufflingTest_CompareConditions(
                Obs_p1=p1.loc(axis=1)[f],
                Obs_p2=p2.loc(axis=1)[f],
                Obs_p1c=p1c.loc(axis=1)[f],
                Obs_p2c=p2c.loc(axis=1)[f],
                pdiff_Obs=diff[f],
                pdiff_c_Obs=diffc[f],
                phase1=phase1,
                phase2=phase2,
                type='mean')
            pvals[f] = p_val
            # pvals[f] = stats.ttest_rel(scaled[f], scaled_comp[f]).pvalue

    # Plot
    x = np.arange(len(features))
    width = 0.15
    fig, ax = plt.subplots(figsize=(len(features) * (width + 1.5), 8))
    ax.bar(x - width / 2, means, width, yerr=sems*1.645, label=condition_label, color='navy')
    ax.bar(x + width / 2, means_comp, width, yerr=sems_comp*1.645, label=compare_condition_label, color='crimson')
    short_names = []
    for feat in features:
        name_bits = feat.split(', ')
        name_to_keep = [bit for bit in name_bits if bit not in name_exclusions]
        short_names.append(', '.join(name_to_keep))
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=-45, ha='left', )
    # Significance stars
    ylim = ax.get_ylim()
    height = ylim[1] - ylim[0]
    for i, feat in enumerate(features):
        p = pvals.get(feat, np.nan)
        if np.isnan(p) or p >= 0.05:
            continue
        # Choose star‑string
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        else:
            sig = '*'
        # Determine bracket vertical position
        y1 = means[feat] + sems[feat]*1.645
        y2 = means_comp[feat] + sems_comp[feat]*1.645
        y = max(y1, y2)
        h = 0.05 * height
        # Draw bracket
        ax.plot(
            [x[i] - width / 2, x[i] - width / 2, x[i] + width / 2, x[i] + width / 2],
            [y + h, y + 1.5 * h, y + 1.5 * h, y + h],
            lw=1.5, c='k'
        )
        # Add significance text
        ax.text(
            x[i], y + 2 * h + 0.01 * height,
            sig, ha='center', va='bottom', fontsize=12
        )
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Scaled Phase Difference')
    ax.set_title(f'{phase2}–{phase1} Phase Difference by Feature_{suffix}\nShuffling Test, err= 90% CI, Only common mice:\n{list(common_mice)}')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.45)
    # Save
    save_dir = os.path.join(base_save_dir, f'top_feature_descriptives/stride{stride_number}')
    os.makedirs(save_dir, exist_ok=True)
    fname = f'PhaseDiff_BothConditions__{phase1}vs{phase2}_stride{stride_number}_{suffix}.png'
    fig.savefig(os.path.join(save_dir, fname), dpi=300)
    plt.close()


def get_back_data(data_dict, norm, phase1, phase2, stride_number):
    # Filter weights for the current phase pair.
    filtered_weights = {
        mouse_id: (weights.feature_weights if hasattr(weights, "feature_weights") else weights)
        for (mouse_id, p1, p2, s), weights in data_dict.items()
        if p1 == phase1 and p2 == phase2 and s == stride_number
    }
    data = pd.concat(filtered_weights, axis=0)
    data.index.names = ['mouse', 'Run']

    phase1_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase1]
    phase2_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase2]

    phase1_mask = data.index.get_level_values('Run').isin(phase1_runs)
    phase2_mask = data.index.get_level_values('Run').isin(phase2_runs)

    back_phase1 = data[phase1_mask]
    back_phase2 = data[phase2_mask]

    # filter by 'back_height' and 'step_phase:0' snippets
    back_phase1 = back_phase1.filter(like='back_height', axis=1)
    back_phase1 = back_phase1.filter(like='step_phase:0', axis=1)
    back_phase2 = back_phase2.filter(like='back_height', axis=1)
    back_phase2 = back_phase2.filter(like='step_phase:0', axis=1)

    # Reverse normalisation
    back_phase1_original, back_phase2_original = back_phase1.copy(), back_phase2.copy()
    for back_feat in back_phase1.columns:
        back_phase1_original[back_feat] = back_phase1_original[back_feat] * norm.loc['std', back_feat] + norm.loc['mean', back_feat]
        back_phase2_original[back_feat] = back_phase2_original[back_feat] * norm.loc['std', back_feat] + norm.loc['mean', back_feat]

    return (back_phase1, back_phase2), (back_phase1_original, back_phase2_original)


def plot_back_phase_comparison(back_data, base_save_dir, phase1, phase2, stride_number, condition_label):
    back_phase1, back_phase2 = back_data

    mean_mouse_phase1 = back_phase1.groupby(level='mouse').mean()
    mean_mouse_phase2 = back_phase2.groupby(level='mouse').mean()

    mean_mouse_difference =  mean_mouse_phase2 - mean_mouse_phase1

    mean_phase1 = mean_mouse_phase1.mean()
    mean_phase2 = mean_mouse_phase2.mean()
    mean_difference = mean_mouse_difference.mean()

    if len(mean_mouse_difference.columns) != 12:
        raise ValueError(f"Expected 12 columns in mean_mouse_difference, but got {len(mean_mouse_difference.columns)}")
    # reduce feature names by looking for 'back_label:' and finding string after this which should follow the pattern Back(number)
    back_names = [col.split('back_label:')[1].split(',')[0] for col in mean_mouse_difference.columns]
    back_numbers = [int(name.split('Back')[1]) for name in back_names]
    back_numbers_sorted = sorted(back_numbers, reverse=True)
    back_names_numbers_sorted = {name: x for x, name in sorted(zip(back_numbers, mean_mouse_difference.columns), reverse=True)}

    mouse_colours = assign_mouse_colors('viridis')

    save_dir = os.path.join(base_save_dir, f'top_feature_descriptives\\stride{stride_number}')
    os.makedirs(save_dir, exist_ok=True)

    def plot_backs(xnames,y,ymean,title,ylabel,filename):
        y_sorted = y.loc(axis=1)[xnames.keys()]
        ymean_sorted_vals = ymean[xnames.keys()].values
        x_sorted_vals = list(xnames.values())

        fig, ax = plt.subplots(figsize=(14, 6))
        for mouse in y.index:
            ax.plot(x_sorted_vals, y_sorted.loc(axis=0)[mouse], alpha=0.3, label=mouse, color=mouse_colours[mouse])
        ax.plot(x_sorted_vals, ymean_sorted_vals, color='black', linestyle='--', label='Mean Phase 1')
        ax.grid(False)
        ax.set_title(title)
        ax.set_xticks(x_sorted_vals)
        ax.set_xticklabels(x_sorted_vals)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Tail <- Back positions -> Head')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        fig.tight_layout()
        ax.invert_xaxis()
        save_path = os.path.join(save_dir,filename)
        fig.savefig(save_path, dpi=300)
        plt.close()

    def plot_backs_phase_difference(xnames, y1_allruns, y2_allruns, y1, y2, ymean1, ymean2, phase1, phase2, title, ylabel, filename):
        # Choose hatch patterns
        phase1_pattern = '' if phase1 == 'APA2' else '/'
        phase2_pattern = '' if phase2 == 'APA2' else '/'

        # Reverse order so Back12→…→Back1
        x_sorted_names = list(xnames.keys())
        back_positions = list(xnames.values())
        x = np.arange(len(x_sorted_names))
        width = 0.35

        # Compute means & SEMs in exactly that order
        ymean1_vals = ymean1[x_sorted_names].values
        ysem1_vals = y1.loc(axis=1)[x_sorted_names].sem(axis=0).values
        ymean2_vals = ymean2[x_sorted_names].values
        ysem2_vals = y2.loc(axis=1)[x_sorted_names].sem(axis=0).values

        # significance
        pvals = []
        common = y1.index.intersection(y2.index)
        for col in x_sorted_names:
            if len(common) < 2:
                pvals.append(np.nan)
            else:
                a = y1[col].loc[common]
                b = y2[col].loc[common]
                p_val, _, _ = ShufflingTest_ComparePhases(
                    Obs_p1=y1_allruns[col].loc[common],
                    Obs_p2=y2_allruns[col].loc[common],
                    mouseObs_p1=a,
                    mouseObs_p2=b,
                    phase1=phase1,
                    phase2=phase2,
                    type='mean'
                )
                pvals.append(p_val)
                #pvals.append(stats.ttest_rel(a, b).pvalue)

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot bars
        ax.bar(x - width / 2, ymean1_vals, width, yerr=ysem1_vals*1.645,
               facecolor=('black' if phase1_pattern == '' else 'white'),
               edgecolor='black', hatch=phase1_pattern, label=phase1)
        ax.bar(x + width / 2, ymean2_vals, width, yerr=ysem2_vals*1.645,
               facecolor=('black' if phase2_pattern == '' else 'white'),
               edgecolor='black', hatch=phase2_pattern, label=phase2)

        # Significance brackets
        ylim = ax.get_ylim()
        h = 0.05 * (ylim[1] - ylim[0])
        for i, p in enumerate(pvals):
            if p < 0.05:
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                y_top = max(ymean1_vals[i] + ysem1_vals[i], ymean2_vals[i] + ysem2_vals[i])
                ax.plot([x[i] - width / 2, x[i] - width / 2, x[i] + width / 2, x[i] + width / 2],
                        [y_top + h, y_top + 1.5 * h, y_top + 1.5 * h, y_top + h], color='k')
                ax.text(x[i], y_top + 2 * h + 0.01 * (ylim[1] - ylim[0]), sig, ha='center')

        ax.set_xticks(x)
        ax.set_xticklabels(back_positions)
        ax.set_ylim(bottom=22)
        ax.set_xlabel('Tail <- Back positions -> Head')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()

    plot_backs(back_names_numbers_sorted, mean_mouse_phase1, mean_phase1, f'Back Height Phase Comparison - {phase1}', 'Back Height', f'BackHeightP1__{phase1}vs{phase2}_stride{stride_number}_{condition_label}.png')
    plot_backs(back_names_numbers_sorted, mean_mouse_phase2, mean_phase2, f'Back Height Phase Comparison - {phase2}', 'Back Height', f'BackHeightP2__{phase1}vs{phase2}_stride{stride_number}_{condition_label}.png')
    plot_backs(back_names_numbers_sorted, mean_mouse_difference, mean_difference, f'Back Height Phase Difference - {phase2} - {phase1}', 'Back Height Difference', f'BackHeightDifferenceP1vsP2__{phase2}-{phase1}_stride{stride_number}_{condition_label}.png')

    plot_backs_phase_difference(back_names_numbers_sorted, back_phase1, back_phase2, mean_mouse_phase1, mean_mouse_phase2, mean_phase1, mean_phase2, phase1, phase2, f'Back Height Comparison - {phase2} vs {phase1}\nShuffling Test, err=90% CI', 'Back Height (mm)', f'BackHeightP1vsP2_Bar__{phase1}vs{phase2}_stride{stride_number}_{condition_label}.png')













def plot_aggregated_feature_weights_byFeature(weights_dict, sorted_features, feature_cluster_assignments, save_path,
                                              phase1, phase2, stride_number, condition_label):
    """
    Plot aggregated feature-space weights across mice for a specific phase pair,
    summarizing the mean (with error bars) for each feature while overlaying individual mouse lines.
    Also draws brackets on the left indicating the cluster each feature belongs to.

    Parameters:
      - weights_dict: dict where keys are tuples (mouse_id, phase1, phase2, stride_number)
                      and values are pandas Series of feature weights.
      - sorted_features: list of features sorted by cluster.
      - feature_cluster_assignments: dict mapping feature names to their cluster assignments.
      - save_path: directory to save the resulting plot.
      - phase1, phase2: phase names.
      - stride_number: stride number.
      - condition_label: additional label for the plot.
    """
    # Filter weights for the current phase pair.
    filtered_weights = {
        mouse_id: (weights.feature_weights if hasattr(weights, "feature_weights") else weights)
        for (mouse_id, p1, p2, s), weights in weights_dict.items()
        if p1 == phase1 and p2 == phase2 and s == stride_number
    }

    display_names = []
    for f in sorted_features:
        display_names.append(short_names.get(f, f))

    if not filtered_weights:
        print(f"No weights found for {phase1} vs {phase2}.")
        return

    # Combine into a DataFrame (rows = features, columns = mouse_ids)
    weights_df = pd.DataFrame(filtered_weights)
    weights_df = weights_df.loc[sorted_features]

    # Scale the weights so they are comparable (optional)
    weights_df = weights_df / weights_df.abs().max()

    # Create numeric y positions and use these for plotting.
    y_positions = np.arange(len(sorted_features))

    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot a faint line for each mouse.
    for mouse in weights_df.columns:
        ax.plot(weights_df[mouse].values, y_positions, alpha=0.3,
                marker='o', markersize=3, linestyle='-', label=mouse)

    # Compute summary statistics: mean and standard error (SEM) for each feature.
    mean_weights = weights_df.mean(axis=1)
    std_weights = weights_df.std(axis=1)
    sem = std_weights / np.sqrt(len(weights_df.columns))

    # Plot the mean with error bars.
    ax.errorbar(mean_weights, y_positions, xerr=std_weights, fmt='o-', color='black',
                label='Mean ± STD', linewidth=2, capsize=4)

    # Add a vertical reference line at 0.
    ax.axvline(x=0, color='red', linestyle='--')

    # Set y-ticks to use feature names.
    ax.set_yticks(y_positions)
    ax.set_yticklabels(display_names)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Feature')
    ax.set_title(
        f'Aggregated Feature Space Weights Across Mice ({phase1} vs {phase2}), stride {stride_number}\n{condition_label}')

    # ---- New: Draw cluster brackets on the left ----
    # Group features by their cluster assignments.
    clusters = {}
    for idx, feat in enumerate(sorted_features):
        clust = feature_cluster_assignments.get(feat)
        clusters.setdefault(clust, []).append(idx)

    # Get current x-limits to determine bracket position.
    x_min, x_max = ax.get_xlim()
    offset = 0.05 * (x_max - x_min)  # horizontal offset for bracket

    for clust, indices in clusters.items():
        # Determine the vertical span with a little margin.
        y_bottom = indices[0] - 0.4
        y_top = indices[-1] + 0.4
        y_mid = (y_bottom + y_top) / 2

        # Draw vertical line (the bracket line) on the left of the plot.
        ax.plot([x_min - offset, x_min - offset], [y_bottom, y_top], color='black', lw=1)
        # Draw horizontal ticks at the top and bottom.
        ax.plot([x_min - offset, x_min - offset * 2], [y_top, y_top], color='black', lw=1)
        ax.plot([x_min - offset, x_min - offset * 2], [y_bottom, y_bottom], color='black', lw=1)
        # Add a text label (e.g., "Cluster 1") centered along the bracket.
        ax.text(x_min - offset - 0.01 * (x_max - x_min), y_mid, f'Cluster {clust}',
                va='center', ha='right', fontsize=10)

    plt.tight_layout()
    plt.legend(title='Mouse ID / Summary', loc='upper right')

    output_file = os.path.join(save_path,
                               f'aggregated_feature_weights_{phase1}_vs_{phase2}_stride{stride_number}_{condition_label}.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Aggregated feature weights plot saved to: {output_file}")


def plot_aggregated_feature_weights_byRun(weights_dict, raw_features, top_features, save_dir, phase1, phase2, stride_number, condition_label):
    # Filter weights for the current phase pair.
    filtered_weights = {
        mouse_id: (weights.feature_weights if hasattr(weights, "feature_weights") else weights)
        for (mouse_id, p1, p2, s), weights in weights_dict.items()
        if p1 == phase1 and p2 == phase2 and s == stride_number
    }

    if not filtered_weights:
        print(f"No weights found for {phase1} vs {phase2}.")
        return

    weights_df = pd.DataFrame(filtered_weights)
    feature_contrib_dict = {}
    for mouse in weights_df.columns:
        # raw_features for a given mouse is stored with key (mouse, phase1, phase2, stride_number)
        # Assuming raw_features[...] is a DataFrame of shape (runs, features), we transpose so that
        # rows become features and columns become run numbers.
        raw_features_mouse = raw_features[(mouse, phase1, phase2, stride_number)].T  # shape: (n_features, n_runs)
        weights = weights_df[mouse]  # Series with index = features

        # Ensure the order of features is the same.
        if not weights.index.equals(raw_features_mouse.index):
            raise ValueError("Feature order mismatch between weights and raw features.")

        # Multiply each weight with its corresponding feature values.
        per_run_contributions = weights.values.reshape(-1, 1) * raw_features_mouse.values
        # Transpose so that each row corresponds to a run and each column to a feature.
        per_run_contributions = per_run_contributions.T  # shape: (n_runs, n_features)
        # Create a DataFrame with run numbers as the index and feature names as the columns.
        df_mouse = pd.DataFrame(per_run_contributions,
                                index=raw_features_mouse.columns,
                                columns=weights.index)

        # Append this mouse's data for each feature.
        for feature in top_features:
            col = df_mouse[feature]
            max_val = col.abs().max()
            # Only scale if the maximum absolute value is nonzero
            if max_val != 0:
                col_scaled = col / max_val
            else:
                col_scaled = col
            if feature not in feature_contrib_dict:
                feature_contrib_dict[feature] = pd.DataFrame()
            feature_contrib_dict[feature][mouse] = col_scaled

    # Plotting each feature.
    for feature, df_feature in feature_contrib_dict.items():
        plt.figure(figsize=(12, 12))
        traces = {}

        for mouse in df_feature.columns:
            trace = df_feature[mouse].values
            traces[mouse] = trace
            plt.plot(df_feature.index, trace, label=f'Mouse {mouse}', alpha=0.3, color='grey')

        # Create a DataFrame from the traces and compute the mean summary line.
        df_traces = pd.DataFrame(traces, index=df_feature.index)
        mean_line = df_traces.mean(axis=1)

        # --- Interpolation ---
        # Ensure the index is numeric for interpolation.
        x_numeric = pd.to_numeric(mean_line.index, errors='coerce')
        mean_line.index = x_numeric
        # Interpolate missing values linearly.
        mean_line_interp = mean_line.interpolate(method='linear')

        # --- Savitzky-Golay Smoothing ---
        # Set a window length (must be odd and <= length of data) and polynomial order.
        window_length = 10  # Adjust as needed.
        if window_length > len(mean_line_interp):
            window_length = len(mean_line_interp) // 2 * 2 + 1  # ensure odd and not too long
        polyorder = 2  # Adjust polynomial order as needed.
        mean_line_smooth = savgol_filter(mean_line_interp, window_length=window_length, polyorder=polyorder)

        # Plot the smoothed mean line.
        plt.plot(mean_line_interp.index, mean_line_smooth, label='Smoothed Mean', color='black', linewidth=2,
                 linestyle='-')

        plt.vlines(x=[9.5, 109.5], ymin=df_traces.min().min(), ymax=df_traces.max().max(), color='red', linestyle='--')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title(f'Feature: {feature} ({phase1} vs {phase2}, stride {stride_number})')
        plt.xlabel('Run Number')
        plt.ylabel('Weight')
        plt.legend()
        plt.tight_layout()
        # Save the plot for the feature.
        out_dir = os.path.join(save_dir, 'feature_weights_by_run')
        os.makedirs(out_dir, exist_ok=True)
        safe_feature = make_safe_feature_name(feature)
        feature_filename = os.path.join(out_dir, f"{safe_feature}_{phase1}{phase2}_{stride_number}.png")
        plt.savefig(feature_filename, dpi=300)
        plt.close()




def plot_aggregated_feature_weights_comparison(weights_dict1, weights_dict2, save_path, phase1, phase2, cond1_label,
                                               cond2_label):
    """
    Plot the average (with SEM error bars) aggregated feature weights for two conditions on the same plot.
    Features are ordered by the absolute mean weight (descending) of condition 1.

    Parameters:
      - weights_dict1: dict for condition 1 (keys: (mouse_id, phase1, phase2), values: pandas Series of feature weights)
      - weights_dict2: dict for condition 2 (same format as weights_dict1)
      - save_path: directory where the resulting plot is saved.
      - phase1, phase2: phase names (for the plot title).
      - cond1_label, cond2_label: labels for condition 1 and condition 2.
    """
    # Extract aggregated weights for each condition.
    def aggregate_weights(weights_dict):
        # Filter weights for the given phase pair.
        filtered = {
            mouse_id: weights
            for (mouse_id, p1, p2), weights in weights_dict.items()
            if p1 == phase1 and p2 == phase2
        }
        if not filtered:
            raise ValueError("No weights found for the specified phase pair.")
        # Build a DataFrame (rows = features, columns = mouse IDs)
        df = pd.DataFrame(filtered).sort_index()
        # Scale weights (optional; here we keep as-is; remove or adjust scaling as needed)
        df = df / df.abs().max()
        return df

    df1 = aggregate_weights(weights_dict1)
    df2 = aggregate_weights(weights_dict2)

    # Compute mean and SEM for each feature
    mean1 = df1.mean(axis=1)
    sem1 = df1.std(axis=1) / np.sqrt(df1.shape[1])
    mean2 = df2.mean(axis=1)
    sem2 = df2.std(axis=1) / np.sqrt(df2.shape[1])

    # Order features by descending absolute mean from condition 1
    ordered_features = mean1.abs().sort_values(ascending=False).index.tolist()

    # Reorder statistics accordingly.
    mean1 = mean1.loc[ordered_features]
    sem1 = sem1.loc[ordered_features]
    mean2 = mean2.loc[ordered_features]
    sem2 = sem2.loc[ordered_features]

    # Create the plot.
    fig, ax = plt.subplots(figsize=(10, len(ordered_features) * 0.3 + 3))

    # Plot condition 1: horizontal errorbar (mean ± SEM)
    ax.errorbar(mean1, ordered_features, xerr=sem1, fmt='o-', color='blue',
                label=f'{cond1_label} Mean ± SEM', capsize=3, linewidth=2)

    # Plot condition 2: horizontal errorbar (mean ± SEM)
    ax.errorbar(mean2, ordered_features, xerr=sem2, fmt='s-', color='green',
                label=f'{cond2_label} Mean ± SEM', capsize=3, linewidth=2)

    # Vertical reference line at 0
    ax.axvline(x=0, color='red', linestyle='--')

    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Feature')
    ax.set_title(f'Comparison of Aggregated Feature Weights ({phase1} vs {phase2})')
    plt.legend(loc='upper right')
    plt.tight_layout()

    output_file = os.path.join(save_path,
                               f'aggregated_feature_weights_comparison_{phase1}_vs_{phase2}_{cond1_label}_vs_{cond2_label}.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Comparison plot saved to: {output_file}")


def plot_aggregated_feature_weights_by_group(weights_dict, mouse_to_group, save_path, phase1, phase2, stride_number,
                                             condition_label):
    """
    Create separate aggregated feature weight plots for each group.

    Parameters:
      - weights_dict: dict with keys (mouse_id, phase1, phase2, stride_number)
                      and values being pandas Series of feature weights.
      - mouse_to_group: dict mapping mouse_id (str) to its group ID (int)
      - save_path: directory to save the plots
      - phase1, phase2, stride_number, condition_label: used for filtering and in titles.
    """
    grouped_weights = {}
    for (mouse_id, p1, p2, s), weights in weights_dict.items():
        if p1 == phase1 and p2 == phase2 and s == stride_number:
            group = mouse_to_group.get(mouse_id)
            if group is not None:
                # Use the underlying Series stored in the FeatureWeights object.
                grouped_weights.setdefault(group, {})[mouse_id] = weights.feature_weights if hasattr(weights, "feature_weights") else weights

    # Now make a plot for each group.
    for group, weights_by_mouse in grouped_weights.items():
        # Create a DataFrame (features as rows, columns = mouse_ids in the group)
        weights_df = pd.DataFrame(weights_by_mouse).sort_index()
        # Optionally scale the weights so they are comparable:
        weights_df = weights_df / weights_df.abs().max()

        fig, ax = plt.subplots(figsize=(15, 15))
        # Plot each mouse’s weights as a faint line:
        for mouse in weights_df.columns:
            ax.plot(weights_df[mouse].values, weights_df.index, alpha=0.3,
                    marker='o', markersize=3, linestyle='-', label=mouse)
        # Compute mean and standard error (SEM) per feature:
        mean_weights = weights_df.mean(axis=1)
        std_weights = weights_df.std(axis=1)
        sem = std_weights / np.sqrt(len(weights_df.columns))

        # Plot the mean with error bars:
        ax.errorbar(mean_weights, weights_df.index, xerr=sem, fmt='o-', color='black',
                    label='Mean ± SEM', linewidth=2, capsize=4)

        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Feature')
        ax.set_title(
            f'Aggregated Feature Weights for Group {group}\n({phase1} vs {phase2}), stride {stride_number}\n{condition_label}')
        plt.tight_layout()
        plt.legend(title='Mouse ID / Summary', loc='upper right')

        output_file = os.path.join(save_path,
                                   f'aggregated_feature_weights_Group{group}_{phase1}_vs_{phase2}_stride{stride_number}_{condition_label}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Aggregated feature weights plot for group {group} saved to: {output_file}")


def plot_aggregated_raw_features(raw_features_dict, save_path, phase1, phase2, stride_number):
    """
    Plot aggregated raw features across mice for a specific phase pair,
    summarizing the mean (with error bars) for each feature while overlaying individual mouse lines.

    Parameters:
      - raw_features_dict: dict where keys are tuples (mouse_id, phase1, phase2)
                           and values are pandas DataFrame of raw features.
      - save_path: directory to save the resulting plot.
      - phase1, phase2: phase names to include in the plot title.
    """
    # Filter raw features for the current phase pair.
    filtered_features = {
        mouse_id: features
        for (mouse_id, p1, p2, s), features in raw_features_dict.items()
        if p1 == phase1 and p2 == phase2 and s == stride_number
    }

    if not filtered_features:
        print(f"No raw features found for {phase1} vs {phase2}.")
        return

    # Combine into a DataFrame (rows = features, columns = mouse_ids)
    features_df = pd.concat(filtered_features, axis=0).sort_index()

    for feature in features_df.columns:
        feature_df = features_df[feature]
        # make mousid the column
        feature_df = feature_df.unstack(level=0)
        feature_df = feature_df.apply(pd.to_numeric, errors='coerce')
        #feature_df = feature_df.applymap(lambda x: x.filled(np.nan) if hasattr(x, "filled") else x)

        #smooth the data with median filter
        #feature_df = feature_df.apply(lambda x: medfilt(x, kernel_size=5), axis=0)

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot a faint line for each mouse
        for mouse in feature_df.columns:
            ydata = feature_df.loc(axis=1)[mouse].values
            if np.ma.is_masked(ydata):
                ydata = np.ma.filled(ydata, np.nan)
            ax.plot(feature_df.index, ydata, alpha=0.3,
                    marker='o', markersize=3, linestyle='-', label=mouse)

        # Compute summary statistics: mean and standard error (SEM) for each feature
        mean_features = feature_df.mean(axis=1)
        std_features = feature_df.std(axis=1)
        sem = std_features / np.sqrt(len(feature_df.columns))

        # Plot the mean with error bars
        ax.errorbar(feature_df.index, mean_features, xerr=sem, fmt='o-', color='black',
                    label='Mean ± SEM', linewidth=2, capsize=4)

        # Compute the global values for this feature (flattening across all mice)
        all_values = feature_df.values.flatten()
        all_values = all_values[~np.isnan(all_values)]  # remove any NaNs

        # Compute the first and third quartiles and the IQR
        Q1 = np.percentile(all_values, 25)
        Q3 = np.percentile(all_values, 75)
        IQR = Q3 - Q1

        # Define lower and upper bounds (1.5 times the IQR below Q1 and above Q3)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Set the y-axis limits using these bounds
        ax.set_ylim(lower_bound, upper_bound)

        # Now draw vertical lines using the filtered bounds:
        ax.vlines(x=[9.5, 109.5], ymin=lower_bound, ymax=upper_bound, color='black', linestyle='--')

        ax.set_xlabel('Run')
        ax.set_ylabel(f'{feature}')
        ax.set_title(f'Aggregated {feature} Across Mice ({phase1} vs {phase2}), stride {stride_number}')
        plt.tight_layout()
        plt.legend(title='Mouse ID / Summary', loc='upper right')
        plt.grid(False)
        plt.gca().yaxis.grid(True)

        filename = f"{feature}_{phase1}_vs_{phase2}_stride{stride_number}"
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        safe_filename = re.sub(r'\s+', '_', safe_filename)

        subdir = os.path.join(save_path, 'aggregated_raw_features')
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        output_file = os.path.join(subdir, f'{safe_filename}.png')
        output_file = r'\\?\{}'.format(output_file)
        plt.savefig(output_file)
        plt.close()


def cluster_regression_weights_across_mice(aggregated_feature_weights, phase_pair, save_dir, n_clusters=2,
                                           aggregate_strides=False):
    """
    Clusters regression weight vectors across mice for a given phase pair
    and labels the points with their mouse IDs.

    Parameters:
      - aggregated_feature_weights: dict with keys (mouse_id, phase1, phase2, stride) and
          value = regression weight vector, either as a dict or a FeatureWeights object.
      - phase_pair: tuple (phase1, phase2, stride_number) specifying the phase comparison.
          If aggregate_strides is True, the third element (stride_number) is ignored.
      - save_dir: directory to save the clustering plot.
      - n_clusters: number of clusters to form.
      - aggregate_strides: bool. If True, regression weights from all strides matching phase1 and phase2
          are aggregated (averaged) per mouse.

    Returns:
      - cluster_df: DataFrame mapping mouse_id to its assigned cluster.
      - kmeans: The fitted KMeans model.
    """
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns

    phase1, phase2, stride_number = phase_pair
    weights_dict = {}  # mouse_id -> list of weight objects (dicts or FeatureWeights)

    # Collect weights for the given phase pair.
    for (mouse_id, p1, p2, s), weights_obj in aggregated_feature_weights.items():
        if p1 == phase1 and p2 == phase2:
            if not aggregate_strides:
                if s != stride_number:
                    continue
            weights_dict.setdefault(mouse_id, []).append(weights_obj)

    if not weights_dict:
        print(f"No regression weights found for phase pair {phase1} vs {phase2} with stride {stride_number}.")
        return None, None

    # Convert all weight objects to dictionaries.
    all_weight_dicts = []
    for w_list in weights_dict.values():
        for w in w_list:
            if hasattr(w, 'feature_weights'):
                all_weight_dicts.append(w.feature_weights)
            else:
                all_weight_dicts.append(w)

    # Compute common keys as the intersection over all weight dictionaries.
    common_keys = set(all_weight_dicts[0].keys())
    for w_dict in all_weight_dicts[1:]:
        common_keys = common_keys.intersection(set(w_dict.keys()))
    common_keys = sorted(common_keys)

    if not common_keys:
        raise ValueError("No common feature keys found across weight dictionaries.")

    # For each mouse, convert each weight dictionary into a numeric vector (using the common keys),
    # and then average the vectors if aggregating across strides.
    mouse_ids = []
    numeric_vectors = []
    for mouse_id, w_list in weights_dict.items():
        mouse_ids.append(mouse_id)
        vectors = []
        for w in w_list:
            if hasattr(w, 'feature_weights'):
                w_dict = w.feature_weights
            else:
                w_dict = w
            try:
                vector = np.array([float(w_dict[k]) for k in common_keys])
            except Exception as e:
                raise ValueError(f"Error converting weights for mouse {mouse_id}: {e}")
            vectors.append(vector)
        if aggregate_strides:
            aggregated_vector = np.mean(np.stack(vectors, axis=0), axis=0)
        else:
            aggregated_vector = vectors[0]
        numeric_vectors.append(aggregated_vector)

    # Stack into a matrix: each row corresponds to one mouse.
    weights_matrix = np.vstack(numeric_vectors)

    # Standardize the weights.
    scaler = StandardScaler()
    weights_matrix_scaled = scaler.fit_transform(weights_matrix)

    # Cluster using KMeans.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(weights_matrix_scaled)

    # Create a DataFrame with the clustering results.
    cluster_df = pd.DataFrame({
        'mouse_id': mouse_ids,
        'cluster': clusters
    })

    # Project to 2D for visualization using PCA.
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(weights_matrix_scaled)

    # Plot the results with annotations for each point.
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=clusters, palette='viridis', s=50, ax=ax)

    for i, mouse in enumerate(mouse_ids):
        ax.text(pcs[i, 0] + 0.02, pcs[i, 1] + 0.02, str(mouse),
                fontsize=9, color='black', weight='bold')

    if aggregate_strides:
        title_str = f"Clustering of Aggregated Regression Weights: {phase1} vs {phase2} (All Strides)"
        file_suffix = "allStrides"
    else:
        title_str = f"Clustering of Regression Weights: {phase1} vs {phase2}, stride {stride_number}"
        file_suffix = f"stride{stride_number}"
    ax.set_title(title_str)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.legend(title="Cluster")

    plot_path = os.path.join(save_dir, f"regression_weights_clustering_{phase1}_vs_{phase2}_{file_suffix}.png")
    plt.savefig(plot_path)
    plt.close()

    return cluster_df, kmeans


def find_cluster_features(w, loadings_df, save_path, mouse_id, phase1, phase2, n_clusters=2):
    """
    Identify and save the features contributing to each cluster.

    Returns:
      - cluster_dict: Dictionary mapping cluster labels to lists of features.
    """
    # Cluster the weights (this call also saves the clustering CSV)
    cluster_df, _ = cluster_weights(w, loadings_df, save_path, mouse_id, phase1, phase2, n_clusters=n_clusters)

    # Create a dictionary grouping features by their cluster label
    cluster_dict = {}
    for cluster in range(n_clusters):
        cluster_features = cluster_df[cluster_df['cluster'] == cluster]['feature'].tolist()
        cluster_dict[f'Cluster {cluster}'] = cluster_features
        print(f"Features in Cluster {cluster} for Mouse {mouse_id} ({phase1} vs {phase2}):")
        print(cluster_features)

    # Save the cluster details to a text file
    output_file = os.path.join(save_path, f'cluster_features_{mouse_id}_{phase1}_vs_{phase2}.txt')
    with open(output_file, 'w') as f:
        for cluster, features in cluster_dict.items():
            f.write(f"{cluster}:\n")
            for feat in features:
                f.write(f"  {feat}\n")
    print(f"Cluster features details saved to: {output_file}")

    return cluster_dict


def plot_clustered_weights(cluster_df, save_path, mouse_id, phase1, phase2):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Sort the DataFrame for better visualization
    cluster_df_sorted = cluster_df.sort_values(['cluster', 'weight'])

    plt.figure(figsize=(12, 8))
    sns.barplot(x='weight', y='feature', hue='cluster', data=cluster_df_sorted, palette='viridis')
    plt.xlabel('Weight Value')
    plt.ylabel('Feature')
    plt.title(f'Clustered Feature Weights for Mouse {mouse_id} ({phase1} vs {phase2})')
    plt.tight_layout()

    plot_file = os.path.join(save_path, f'clustered_feature_weights_{mouse_id}_{phase1}_vs_{phase2}.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Vertical clustered feature weights plot saved to: {plot_file}")


def plot_cluster_loadings_lines(aggregated_cluster_loadings, save_dir):
    """
    For each (phase1, phase2, stride) combination, create a line plot where each mouse
    is represented by a single line that shows its regression loading across clusters.
    If a mouse does not have a value for a given cluster, it is plotted as zero.
    """
    os.makedirs(save_dir, exist_ok=True)

    for key, mouse_cluster_dict in aggregated_cluster_loadings.items():
        phase1, phase2, stride_number = key

        # Compute the union of all cluster IDs for this key.
        all_clusters = set()
        for cl_loadings in mouse_cluster_dict.values():
            all_clusters.update(cl_loadings.keys())
        sorted_clusters = sorted(all_clusters)

        plt.figure(figsize=(10, 6))

        for mouse, cl_loadings in mouse_cluster_dict.items():
            # For each cluster in the union, get the loading, or 0 if missing.
            loadings = [cl_loadings.get(cluster, 0) for cluster in sorted_clusters]
            # scale loadings
            loadings = np.array(loadings) / np.max(np.abs(loadings))
            plt.plot(sorted_clusters, loadings, marker='o', label=mouse)

        plt.title(f"Regression Loadings by Cluster: {phase1} vs {phase2} | Stride {stride_number}")
        plt.xlabel("Cluster")
        plt.ylabel("Regression Loading")
        plt.legend(title="Mouse ID")
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'cluster_loadings_lines_{phase1}_{phase2}_stride{stride_number}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved cluster loading line plot to {save_path}")


def create_save_directory(base_dir, mouse_id, stride_number, phase1, phase2):
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


def load_stride_data(stride_data_path):
    """
    Load stride data from the specified HDF5 file.

    Parameters:
        stride_data_path (str): Path to the stride data HDF5 file.

    Returns:
        pd.DataFrame: Loaded stride data.
    """
    stride_data = pd.read_hdf(stride_data_path, key='stride_info')
    return stride_data

def set_up_save_dir(condition, exp, c, base_save_dir_no_c):
    base_save_dir = base_save_dir_no_c + f'-c={c}'
    base_save_dir_condition = os.path.join(base_save_dir, f'{condition}_{exp}')
    return base_save_dir, base_save_dir_condition


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


def get_runs(scaled_data_df, stride_data, mouse_id, stride_number, phase1, phase2):
    mask_phase1 = scaled_data_df.index.get_level_values('Run').isin(
        expstuff['condition_exp_runs']['APAChar']['Extended'][phase1])
    mask_phase2 = scaled_data_df.index.get_level_values('Run').isin(
        expstuff['condition_exp_runs']['APAChar']['Extended'][phase2])

    if not mask_phase1.any():
        raise ValueError(f"No runs found for phase '{phase1}'.")
    if not mask_phase2.any():
        raise ValueError(f"No runs found for phase '{phase2}'.")

    run_numbers_phase1 = scaled_data_df.index[mask_phase1]
    run_numbers_phase2 = scaled_data_df.index[mask_phase2]
    run_numbers = list(run_numbers_phase1) + list(run_numbers_phase2)

    # Determine stepping limbs.
    stepping_limbs = [determine_stepping_limbs(stride_data, mouse_id, run, stride_number)
                      for run in run_numbers]

    return run_numbers, stepping_limbs, mask_phase1, mask_phase2

def get_pc_run_info(pcs, mask_phase1, mask_phase2, phase1, phase2):
    pcs_phase1 = pcs[mask_phase1]
    pcs_phase2 = pcs[mask_phase2]

    labels_phase1 = np.array([phase1] * pcs_phase1.shape[0])
    labels_phase2 = np.array([phase2] * pcs_phase2.shape[0])
    labels = np.concatenate([labels_phase1, labels_phase2])
    pcs_combined = np.vstack([pcs_phase1, pcs_phase2])

    return pcs_combined, labels, pcs_phase1, pcs_phase2

# def process_compare_condition(feature_data, feature_data_df_compare, mouseIDs_base, mouseIDs_compare, condition, compare_condition, exp, day, stride_data, stride_data_compare, phases,
#                               stride_numbers, base_save_dir_condition, aggregated_save_dir,
#                               global_fs_results, global_pca_results):
#     global_regression_params = {}
#     for phase1, phase2 in itertools.combinations(phases, 2):
#         for stride_number in stride_numbers:
#             selected_features, fs_df = global_fs_results[(phase1, phase2, stride_number)]
#             pca, loadings_df = global_pca_results[(phase1, phase2, stride_number)]
#
#             regression_params = compute_global_regression_model(
#                 feature_data,
#                 mouseIDs_base,
#                 stride_number,
#                 phase1, phase2,
#                 condition, exp, day,
#                 stride_data,
#                 selected_features, loadings_df
#             )
#             global_regression_params[(phase1, phase2, stride_number)] = regression_params
#
#     aggregated_compare_predictions = {}
#     if compare_condition != 'None':
#         compare_mouse_ids = mouseIDs_compare
#         for phase1, phase2 in itertools.combinations(phases, 2):
#             for stride_number in stride_numbers:
#                 # Retrieve regression parameters computed from the base condition.
#                 regression_params = global_regression_params.get((phase1, phase2, stride_number), None)
#                 if regression_params is None:
#                     print(f"No regression model for phase pair {phase1} vs {phase2}, stride {stride_number}.")
#                     continue
#                 w = regression_params['w']
#                 loadings_df = regression_params['loadings_df']
#                 selected_features = regression_params['selected_features']
#
#                 for mouse_id in compare_mouse_ids:
#                     try:
#                         compare_save_path = os.path.join(base_save_dir_condition, f"{compare_condition}_predictions",
#                                                          mouse_id)
#                         os.makedirs(compare_save_path, exist_ok=True)
#
#                         smoothed_scaled_pred, runs = predict_compare_condition(
#                             feature_data_df_compare,
#                             mouse_id, compare_condition, stride_number, phase1, phase2,
#                             selected_features, loadings_df, w, compare_save_path
#                         )
#
#                         #x_vals = np.arange(len(smoothed_scaled_pred))
#                         aggregated_compare_predictions.setdefault((phase1, phase2, stride_number), []).append(
#                             PredictionData(mouse_id=mouse_id, x_vals=runs,
#                                                  smoothed_scaled_pred=smoothed_scaled_pred)
#                         )
#
#
#                     except Exception as e:
#                         print(
#                             f"Error processing compare condition for mouse {mouse_id}, phase pair {phase1} vs {phase2}: {e}")
#
#         # Plot aggregated compare predictions.
#         for (phase1, phase2, stride_number), agg_data in aggregated_compare_predictions.items():
#             plot_aggregated_run_predictions(agg_data, aggregated_save_dir, phase1, phase2, stride_number, condition_label=f"vs_{compare_condition}")

def plot_significant_features(selected_features_accuracies, save_path, selected_features):
    for fidx, feature in enumerate(selected_features):
        plt.figure()
        sns.histplot(selected_features_accuracies.loc[feature, 'iteration_diffs'].values(), bins=20, kde=True)
        plt.axvline(0, color='red', label='True Accuracy')
        plt.title(feature)
        plt.xlabel('Shuffled Accuracy - True Accuracy')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'feature_significances\\feature{fidx}_feature_selection.png'))
        plt.close()

def plot_nonsignificant_features(nonselected_features_accuracies, save_path, nonselected_features):
    for fidx, feature in enumerate(nonselected_features):
        plt.figure()
        sns.histplot(nonselected_features_accuracies.loc[feature, 'iteration_diffs'].values(), bins=20, kde=True)
        plt.axvline(0, color='red', label='True Accuracy')
        plt.title(feature)
        plt.xlabel('Shuffled Accuracy - True Accuracy')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'feature_nonsignificances\\feature{fidx}_feature_selection.png'))
        plt.close()


def plot_multi_stride_predictions(stride_dict: Dict[int, List[PredictionData]],
                                  phase1: str,
                                  phase2: str,
                                  save_dir: str,
                                  condition_label: str,
                                  smooth: bool = False,
                                  smooth_window: int = 21,
                                  normalize: bool = False):
    """
    For each stride in stride_dict (for the given phase pair), compute the mean normalized
    prediction curve across all PredictionData objects (after optional smoothing of individual curves),
    taking into account that the x-axis values may differ. Only the mean lines are plotted,
    with vertical phase indicators. Line colors are fixed from the "Blues" colormap.
    """
    plt.figure(figsize=(10, 8))
    mean_curves = {}  # Stores: stride_number -> (common_x, mean_curve)

    # Fixed color mapping for possible strides (avoid the lightest value).
    stride_color_mapping = {
        -3: plt.cm.Blues(0.2),
        -2: plt.cm.Blues(0.45),
        -1: plt.cm.Blues(0.7),
         0: plt.cm.Blues(0.99)
    }

    # Process each stride
    for stride_number, pred_list in stride_dict.items():
        # Gather all x-axis values from all predictions for this stride.
        all_x_vals = []
        for pred in pred_list:
            all_x_vals.extend(pred.x_vals)
        if not all_x_vals:
            continue  # skip if empty

        global_min_x = min(all_x_vals)
        global_max_x = max(all_x_vals)
        # Use the maximum length among predictions as the number of common points.
        common_npoints = max(len(pred.x_vals) for pred in pred_list)
        common_x = np.linspace(global_min_x, global_max_x, common_npoints)

        interpolated_curves = []
        for pred in pred_list:
            smoothed_pred = pred.smoothed_scaled_pred

            # Normalize the prediction curve according to the chosen method.
            max_abs = max(abs(smoothed_pred.min()), abs(smoothed_pred.max()))
            normalized_curve = smoothed_pred / max_abs if max_abs != 0 else smoothed_pred

            # Optionally smooth the normalized curve.
            if smooth:
               # normalized_curve = smooth_curve(normalized_curve, smooth_window)
                normalized_curve = medfilt(normalized_curve, kernel_size=smooth_window)

            # Interpolate to the common x-axis.
            interp_curve = np.interp(common_x, pred.x_vals, normalized_curve)
            interpolated_curves.append(interp_curve)

        if interpolated_curves:
            mean_curve = np.mean(np.vstack(interpolated_curves), axis=0)
            if normalize:
                mean_curve = mean_curve / np.max(np.abs(mean_curve))
            mean_curves[stride_number] = (common_x, mean_curve)

    # Plot each stride's mean curve using the fixed color mapping.
    for stride_number, (common_x, mean_curve) in sorted(mean_curves.items()):
        color = stride_color_mapping.get(stride_number, plt.cm.Blues(0.6))
        plt.plot(common_x, mean_curve, linewidth=2, label=f"Stride {stride_number}", color=color)

    # Add vertical phase indicators.
    plt.vlines(x=[9.5, 109.5], ymin=-1, ymax=1, color='red', linestyle='-')
    plt.vlines(x=[39.5, 79.5, 119.5], ymin=-1, ymax=1, color='gray', alpha=0.6, linestyle='--')
    plt.title(f"Aggregated Scaled Multi-Stride Predictions for {phase1} vs {phase2}\n{condition_label}\nSmooth={smooth}, Window={smooth_window}, Normalize={normalize}")
    plt.xlabel("Run Number")
    plt.ylabel("Normalized Prediction (Smoothed)")
    plt.ylim(-1, 1)
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.gca().yaxis.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir,
                             f"Aggregated_MultiStride_Predictions_{phase1}_vs_{phase2}_{condition_label}_smooth={smooth}-sw{smooth_window}-normalize{normalize}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_multi_stride_predictions_difference(stride_dict: Dict[int, List[PredictionData]],
                                             phase1: str,
                                             phase2: str,
                                             save_dir: str,
                                             condition_label: str,
                                             smooth: bool = False,
                                             smooth_window: int = 21,
                                             normalize: bool = False,):
    """
    For each phase pair (phase1 vs phase2), for each stride in stride_dict:
      - Interpolate each PredictionData to a fixed common x-axis of 160 points.
      - Optionally smooth each normalized prediction using a moving average.
      - Compute the mean normalized prediction curve.
    Then compute the difference between consecutive stride mean curves (current minus previous)
    and plot these difference curves with vertical phase indicators (at 9.5 and 109.5).
    """
    plt.figure(figsize=(10, 8))
    mean_curves = {}  # Stores: stride_number -> (common_x, mean_curve)
    fixed_npoints = 160  # Fixed number of x-axis points

    # Compute mean curves for each stride using interpolation.
    for stride_number, pred_list in stride_dict.items():
        # Gather all x-values from predictions.
        all_x_vals = []
        for pred in pred_list:
            all_x_vals.extend(pred.x_vals)
        if not all_x_vals:
            continue

        global_min_x = min(all_x_vals)
        global_max_x = max(all_x_vals)
        common_x = np.linspace(global_min_x, global_max_x, fixed_npoints)

        interpolated_curves = []
        for pred in pred_list:
            smoothed_pred = pred.smoothed_scaled_pred
            # Normalize the prediction curve.
            max_abs = max(abs(smoothed_pred.min()), abs(smoothed_pred.max()))
            normalized_curve = smoothed_pred / max_abs if max_abs != 0 else smoothed_pred

            # Optionally smooth the normalized curve.
            if smooth:
                normalized_curve = medfilt(normalized_curve, kernel_size=smooth_window)

            # Interpolate to the common x-axis.
            interp_curve = np.interp(common_x, pred.x_vals, normalized_curve)
            interpolated_curves.append(interp_curve)
        if interpolated_curves:
            mean_curve = np.mean(np.vstack(interpolated_curves), axis=0)
            if normalize:
                mean_curve = mean_curve / np.max(np.abs(mean_curve))
            mean_curves[stride_number] = (common_x, mean_curve)

    # Compute differences between consecutive stride mean curves.
    sorted_strides = sorted(mean_curves.keys())
    difference_curves = {}  # Key: (prev_stride, curr_stride) -> (common_x, diff_curve)
    for i in range(1, len(sorted_strides)):
        prev_stride = sorted_strides[i - 1]
        curr_stride = sorted_strides[i]
        common_x_prev, mean_curve_prev = mean_curves[prev_stride]
        common_x_curr, mean_curve_curr = mean_curves[curr_stride]
        # Since we forced a fixed common_x, we can use that.
        diff_curve = mean_curve_curr - mean_curve_prev
        difference_curves[(prev_stride, curr_stride)] = (common_x_curr, diff_curve)

    # Set up a fixed color mapping for the difference curves using 'PuOr'.
    cmap = plt.get_cmap("RdPu")
    diff_keys = sorted(difference_curves.keys())
    colors = {key: cmap(0.3 + 0.4 * i / max(len(diff_keys) - 1, 1)) for i, key in enumerate(diff_keys)}

    # Plot each difference curve.
    for key, (common_x, diff_curve) in sorted(difference_curves.items()):
        prev_stride, curr_stride = key
        plt.plot(common_x, diff_curve, linewidth=2,
                 label=f"Diff (Stride {curr_stride} - {prev_stride})",
                 color=colors[key])

    # Add vertical phase indicators.
    plt.vlines(x=[9.5, 109.5], ymin=-1, ymax=1, color='red', linestyle='-')
    plt.vlines(x=[39.5, 79.5, 119.5], ymin=-1, ymax=1, color='gray', alpha=0.6, linestyle='--')
    plt.title(f"Difference Between Consecutive Stride Mean Predictions for {phase1} vs {phase2}\n{condition_label}\nSmooth={smooth}, Window={smooth_window}")
    plt.xlabel("Run Number")
    plt.ylabel("Difference in Normalized Prediction")
    plt.ylim(-1, 1)
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.gca().yaxis.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir,
                             f"Aggregated_MultiStride_Differences_{phase1}_vs_{phase2}_{condition_label}_smooth={smooth}-sw{smooth_window}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_regression_loadings_PC_space_across_mice(global_pca_results, aggregated_feature_weights, aggregated_save_dir, phase1, phase2, stride_number, condition_label):
    pca, loadings_df = global_pca_results[(phase1, phase2, stride_number)]

    plt.figure(figsize=(10,8))
    # Loop through each mouse's feature weights for the current phase and stride.
    for (mouse_id, p1, p2, s), ftr_wght in aggregated_feature_weights.items():
        if p1 == phase1 and p2 == phase2 and s == stride_number:
            # ftr_wght.feature_weights is a Series with feature names as the index.
            common_features = loadings_df.index.intersection(ftr_wght.feature_weights.index)
            if len(common_features) == 0:
                continue
            loadings_sub = loadings_df.loc[common_features]
            weights_sub = ftr_wght.feature_weights.loc[common_features]
            # Project feature-space weights into PC space.
            pc_weights = loadings_sub.T.dot(weights_sub)
            # Normalize by maximum absolute value.
            max_abs = np.abs(pc_weights).max()
            if max_abs != 0:
                pc_weights_norm = pc_weights / max_abs
            else:
                pc_weights_norm = pc_weights

            # Sort the PC indices (if numeric or digit strings)
            keys = pc_weights_norm.index
            norm_weights = pc_weights_norm
            #sorted_keys = sorted(pc_weights_norm.index, key=lambda x: int(x) if str(x).isdigit() else x)
            #sorted_norm_weights = [pc_weights_norm[k] for k in sorted_keys]
            plt.plot(keys, norm_weights, marker='o', label=f"Mouse {mouse_id}")
    plt.xlabel("Principal Component")
    plt.ylabel("Normalized Regression Weight (PC space)")
    plt.title(f"Normalized PC Regression Loadings for {phase1} vs {phase2}, Stride {stride_number}\n{condition_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(aggregated_save_dir,
                             f"Normalized_PC_regression_loadings_{phase1}_{phase2}_stride{stride_number}_{condition_label}.png"))
    plt.close()

def plot_even_odd_PCs_across_mice(even_ws, odd_ws, aggregated_save_dir, phase1, phase2, stride_number, condition_label):
    # Plot even vs odd for each mouse for this phase pair and stride.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # First, gather unique mouse IDs for the current phase/stride.
    unique_mouse_ids = sorted({mouse_id for (mouse_id, p1, p2, s) in even_ws.keys()
                               if p1 == phase1 and p2 == phase2 and s == stride_number})

    # Create a color mapping using a colormap with enough distinct colours.
    cmap = plt.get_cmap("tab20")
    color_mapping = {mouse_id: cmap(i / len(unique_mouse_ids)) for i, mouse_id in enumerate(unique_mouse_ids)}

    # Track which mouse IDs have been added to the legend.
    plotted_mice = set()

    # Loop through even_ws keys and filter by phase and stride.
    for key, evenW in even_ws.items():
        mouse_id, p1, p2, s = key
        if p1 == phase1 and p2 == phase2 and s == stride_number:
            oddW = odd_ws[key]

            # Ensure arrays are 2D.
            evenW = np.atleast_2d(np.array(evenW))
            oddW = np.atleast_2d(np.array(oddW))

            # Extract the first 3 PCs.
            even_PC1, even_PC2, even_PC3 = evenW[:, 0], evenW[:, 1], evenW[:, 2]
            odd_PC1, odd_PC2, odd_PC3 = oddW[:, 0], oddW[:, 1], oddW[:, 2]

            # Assign the precomputed color.
            color = color_mapping[mouse_id]

            # Only add labels once per mouse.
            label_even = f"{mouse_id} Even" if mouse_id not in plotted_mice else None
            label_odd = f"{mouse_id} Odd" if mouse_id not in plotted_mice else None
            plotted_mice.add(mouse_id)

            ax.scatter(even_PC1, even_PC2, even_PC3, marker='o', label=label_even, color=color)
            ax.scatter(odd_PC1, odd_PC2, odd_PC3, marker='^', label=label_odd, color=color)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(f"Aggregated Even vs Odd Weights for {phase1} vs {phase2}, stride {stride_number}\n{condition_label}")
    plt.tight_layout()
    plt.savefig(os.path.join(aggregated_save_dir,
                             f"Aggregated_even_vs_odd_w_{phase1}_{phase2}_stride{stride_number}_{condition_label}.png"))
    plt.close()

def rank_within_vs_between_differences(even_ws, odd_ws,aggregated_save_dir, phase1, phase2, stride_number, condition_label):
    # Collect even and odd weights for all mice in this (phase1, phase2, stride_number) group.
    even_dict = {}
    odd_dict = {}
    for key, evenW in even_ws.items():
        mouse_id, p1, p2, s = key
        if p1 == phase1 and p2 == phase2 and s == stride_number:
            # Ensure the weight vector is 1D (if it was saved as 1D or single point).
            even_dict[mouse_id] = np.atleast_1d(np.array(evenW))
            odd_dict[mouse_id] = np.atleast_1d(np.array(odd_ws[key]))

    # Total number of mice in this group.
    num_mice = len(even_dict)
    print(f"Total mice for {phase1} vs {phase2}, stride {stride_number}: {num_mice}")

    # For each mouse, compute the within difference (dwi) and between differences (di).
    ranks = {}  # will store the rank for each mouse
    for mouse_i, even_i in even_dict.items():
        odd_i = odd_dict[mouse_i]
        # Compute the within-mouse difference (dwi) as the Euclidean norm
        dwi = np.linalg.norm(even_i - odd_i)

        # Compute the between-mouse differences: difference between this mouse's even weight and each other mouse's even weight.
        between_diffs = []
        for mouse_j, even_j in even_dict.items():
            if mouse_j == mouse_i:
                continue
            dij = np.linalg.norm(even_i - even_j)
            between_diffs.append(dij)

        # Combine all differences: (total mice - 1) between differences + 1 within difference.
        all_diffs = between_diffs + [dwi]

        # Rank the within difference relative to all differences.
        # Sorting in ascending order; the smallest difference gets rank 1.
        sorted_diffs = sorted(all_diffs)
        # Find the rank (1-indexed)
        rank = sorted_diffs.index(dwi) + 1
        ranks[mouse_i] = rank

    # Print or further process the resulting ranks.
    for mouse, rank in ranks.items():
        print(f"Mouse {mouse} rank: {rank} (out of {num_mice})")

    save_path = os.path.join(aggregated_save_dir,
                             f"Within_vs_between_mouse_rank_table_EvenOdd_{phase1}_{phase2}_stride{stride_number}_{condition_label}.png")
    save_rank_table(ranks, save_path)


def save_rank_table(ranks: dict, save_path: str):
    """
    Saves a table figure showing the rank for each mouse.

    Parameters:
        ranks (dict): A dictionary mapping mouse_id to rank.
        save_path (str): Path (including filename) to save the figure.
    """
    # Sort the dictionary by mouse_id (or by rank if preferred)
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[0])
    header = ["Mouse", "Rank"]
    cell_text = [[str(mouse), str(rank)] for mouse, rank in sorted_ranks]

    # Determine figure size (adjust height based on the number of mice)
    fig, ax = plt.subplots(figsize=(3, 0.5 + 0.3 * len(cell_text)))
    ax.axis('tight')
    ax.axis('off')

    # Create a table with the cell data and headers
    table = ax.table(cellText=cell_text, colLabels=header, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    fig.tight_layout()

    # Save the figure to the specified path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def compare_within_between_variability(runs_dict: dict, mouse_runs: dict, aggregated_save_dir: str, phase: str, stride_data: pd.DataFrame, phase1: str, phase2: str,
                                       stride_number: int, exp: str, condition_label: str):
    """
    For a given phase and stride, compute the within-mouse variability and between-mouse variability,
    rank the within variability relative to the between variabilities, and plot all run-level PC data in 3D.
    Additionally, compute a global median vector (across all mice), and flag outlier runs based on
    each run's distance to this global median vector. The global median vector is also plotted.

    Parameters:
        runs_dict (dict): Keys are tuples (mouse, p1, p2, s) with run-level PC data (n_runs, n_PCs).
        mouse_runs (dict): Keys are mice, values are the available within phase runs for that mouse
        aggregated_save_dir (str): Directory for saving output figures.
        phase (str): Phase for analysis (assumed at index 1 of key).
        phase1 (str): First phase (for titling).
        phase2 (str): Second phase (for titling).
        stride_number (int): Stride number (assumed at index 3 of key).
        condition_label (str): Label for condition (for titling/filenames).

    Returns:
        within_vars (dict): Average within-mouse variability per mouse.
        between_vars (dict): List of between-mouse distances for each mouse.
        rank (dict): Ranking of within variability per mouse.
    """
    # Filter keys for the given phase and stride.
    mouse_params = [key for key in runs_dict.keys() if key[1] == phase or key[2] == phase and key[3] == stride_number]

    mouse_means = {}
    within_vars = {}

    # Compute the mean PC vector for each mouse and its within variability (using each mouse's own mean).
    for (mouse, p1, p2, s) in mouse_params:
        runs = runs_dict[(mouse, p1, p2, s)]
        mean_vec = np.mean(runs, axis=0)
        mouse_means[mouse] = mean_vec
        distances = np.linalg.norm(runs - mean_vec, axis=1)
        within_vars[mouse] = np.mean(distances)

    print("Total mice for", phase, "stride", stride_number, ":", len(mouse_params))

    # Compute between variability: for each mouse, distances from its mean to every other mouse's mean.
    between_vars = {}
    for mouse_i in mouse_means.keys():
        mean_i = mouse_means[mouse_i]
        other_dists = []
        for mouse_j in mouse_means.keys():
            if mouse_j == mouse_i:
                continue
            mean_j = mouse_means[mouse_j]
            other_dists.append(np.linalg.norm(mean_i - mean_j))
        between_vars[mouse_i] = other_dists

    # Rank the within variability relative to between distances.
    rank = {}
    for mouse, within_var in within_vars.items():
        all_vars = between_vars[mouse] + [within_var]
        sorted_vars = sorted(all_vars)
        rank[mouse] = sorted_vars.index(within_var) + 1
    for mouse, r in rank.items():
        print(f"Mouse {mouse} rank: {r} (out of {len(mouse_means)})")

    # Save rank table as a figure.
    save_path_ranktable = os.path.join(aggregated_save_dir,
                                       f"Within_vs_between_mouse_rank_table_IndividRuns_{phase}({phase1}_{phase2})_stride{stride_number}_{condition_label}.png")
    if rank:
        save_rank_table(rank, save_path_ranktable)  # assumes you have save_rank_table defined

    # Use global outlier detection.
    outlier_runs, threshold, global_stats = find_outlier_runs_global_std(runs_dict, phase, stride_number)

    # Plot all runs for each mouse in 3D, with outliers highlighted.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_mice = sorted({key[0] for key in mouse_params})
    cmap = plt.get_cmap("tab20")
    color_mapping = {mouse: cmap(i / len(unique_mice)) for i, mouse in enumerate(unique_mice)}
    plotted = set()

    # for key in mouse_params:
    #     mouse, p1, p2, s = key
    #     runs = np.array(runs_dict[key])
    #     if runs.ndim == 1:
    #         runs = np.atleast_2d(runs)
    #     label = mouse if mouse not in plotted else None
    #     # Plot all runs.
    #     ax.scatter(runs[:, 0], runs[:, 1], runs[:, 2],
    #                color=color_mapping[mouse],
    #                label=label,
    #                s=50,
    #                alpha=0.8)
    #     # Highlight outliers for this mouse.
    #     if mouse in outlier_runs:
    #         outlier_indices = list(outlier_runs[mouse])
    #         if len(outlier_indices) > 0:
    #             outlier_points = runs[outlier_indices, :]
    #             ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2],
    #                        color=color_mapping[mouse],
    #                        marker='x',
    #                        s=100,
    #                        linewidths=2)
    #     plotted.add(mouse)

    # Plot each mouse's mean (from within mouse data) as a diamond.
    for mouse in unique_mice:
        mean_vec = mouse_means[mouse]
        ax.scatter(mean_vec[0], mean_vec[1], mean_vec[2],
                   color=color_mapping[mouse],
                   marker='D',
                   s=150,
                   edgecolor='k',
                   linewidths=1.5,
                   label=f"{mouse} mean")

    # Plot the global median vector as a black star.
    global_median_vector = global_stats["global_median_vector"]
    ax.scatter(global_median_vector[0], global_median_vector[1], global_median_vector[2],
               color='k',
               marker='*',
               s=200,
               label="Global median")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"{condition_label}: {phase} (stride {stride_number}) - Run-level PC Projections")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    save_path_plot = os.path.join(aggregated_save_dir,
                                  f"Run_PC_projections_{phase}({phase1}_{phase2})_stride{stride_number}_{condition_label}.png")
    plt.savefig(save_path_plot, bbox_inches='tight')
    plt.close()

    # # save the outliers as a csv
    # save_outliers_csv_updated(outlier_runs, mouse_runs, aggregated_save_dir, condition_label, phase1, phase2, stride_number, stride_data)

    return within_vars, between_vars, rank

def plot_pcs_outliers(pc, outlier_runs, p, phase1, phase2, stride, base_save_dir):
    # Plot all runs for each mouse in 3D, with outliers highlighted.
    fig = plt.figure(figsize=(15, 13))
    ax = fig.add_subplot(111, projection='3d')

    unique_mice = sorted({key[0] for key in pc.keys()})
    cmap = plt.get_cmap("tab20")
    color_mapping = {mouse: cmap(i / len(unique_mice)) for i, mouse in enumerate(unique_mice)}
    plotted = set()
    for (mouse, _, _, _) in pc.keys():
        runs = np.array(pc[(mouse, phase1, phase2, stride)])
        if runs.ndim == 1:
            runs = np.atleast_2d(runs)
        label = mouse if mouse not in plotted else None
        # Plot all runs.
        ax.scatter(runs[:, 0], runs[:, 1], runs[:, 2],
                   color=color_mapping[mouse],
                   label=label,
                   s=50,
                   alpha=0.8)
        # Highlight outliers for this mouse.
        if mouse in outlier_runs:
            outlier_indices = list(outlier_runs[mouse])
            if len(outlier_indices) > 0:
                outlier_points = runs[outlier_indices, :]
                ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2],
                           color=color_mapping[mouse],
                           marker='x',
                           s=100,
                           linewidths=2)
        plotted.add(mouse)
    # save
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"Outliers for {p} stride {stride} ({phase1} vs {phase2})\nRun-level PC Projections")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    save_path_plot = os.path.join(base_save_dir,
                                  f"Run_PC_projections_{p}_stride{stride}_({phase1}-{phase2})_outliers.png")
    fig.savefig(save_path_plot, bbox_inches='tight')
    plt.close()


def find_outlier_runs_global_std(runs_dict: dict, phase: str, stride_number: int, factor: float = 2.0):
    """
    Compute a global median vector across all runs (all mice) for the specified phase and stride.
    Then, compute the Euclidean distance of every run to this global median vector.
    Using the distribution of these distances, compute the global median distance and standard deviation.
    Any run whose distance exceeds (global_median_distance + factor * global_std_distance) is flagged as an outlier.

    Parameters:
        runs_dict (dict): Keys are tuples (mouse, p1, p2, s) and values are numpy arrays of shape (n_runs, n_PCs)
                          representing run-level PC data.
        phase (str): The phase to filter by (assumed to be at index 1 of the key).
        stride_number (int): The stride number to filter by (assumed to be at index 3 of the key).
        factor (float): The multiplier for the standard deviation (default is 2.0).

    Returns:
        outliers (dict): Dictionary mapping each mouse to a set of outlier run indices.
        threshold (float): The global threshold (global_median_distance + factor * global_std_distance).
        global_stats (dict): Dictionary containing the global median vector, global_median_distance, and global_std_distance.
    """
    all_runs = []
    runs_by_key = {}

    # Filter runs for the specified phase and stride.
    for key, data in runs_dict.items():
        mouse, p1, p2, s = key
        if p1 == phase or p2==phase and s == stride_number:
            runs = np.array(data)
            if runs.ndim == 1:
                runs = np.atleast_2d(runs)
            runs_by_key[key] = runs
            all_runs.append(runs)

    if not all_runs:
        print("No runs found for phase", phase, "and stride", stride_number)
        return {}, np.nan, {}

    # Stack all runs together.
    all_runs_stacked = np.vstack(all_runs)

    # Compute the global median vector (component-wise median).
    global_median_vector = np.median(all_runs_stacked, axis=0)

    # Compute distances from each run to the global median vector.
    all_distances = np.linalg.norm(all_runs_stacked - global_median_vector, axis=1)
    global_median_distance = np.median(all_distances)
    global_std_distance = np.std(all_distances)

    # Define threshold.
    threshold = global_median_distance + factor * global_std_distance

    # Flag outliers: for each key (each mouse's set of runs), flag runs with distance > threshold.
    outliers = {}
    for key, runs in runs_by_key.items():
        mouse = key[0]
        distances = np.linalg.norm(runs - global_median_vector, axis=1)
        outlier_idx = set(np.where(distances > threshold)[0].tolist())
        if mouse in outliers:
            outliers[mouse].update(outlier_idx)
        else:
            outliers[mouse] = outlier_idx

    print("Global median distance: {:.2f}".format(global_median_distance))
    print("Global std of distances: {:.2f}".format(global_std_distance))
    print("Threshold (median + {}*std): {:.2f}".format(factor, threshold))

    global_stats = {
        "global_median_vector": global_median_vector,
        "global_median_distance": global_median_distance,
        "global_std_distance": global_std_distance
    }

    return outliers, threshold, global_stats


def save_outliers_csv_updated(outlier_runs: dict, base_save_dir_condition: str, condition_label: str,
                              phase1: str, phase2: str, stride_number: int, stride_data: pd.DataFrame, csv_filename: str = "outlier_runs.csv"):
    """
    Update or create a CSV file that contains outlier run indices for various phases and strides.
    Only entries for mice with non-empty outlier run indices are added or updated.

    If the CSV file exists, it is loaded and new entries (identified by a unique key of
    condition_label, mouseID, phase1, phase2, and stride_number) are either updated (if existing)
    or appended. Other entries are preserved.

    Parameters:
        outlier_runs (dict): Dictionary mapping mouse IDs to a set of outlier run indices.
        base_save_dir_condition (str): Directory where the CSV file is located.
        condition_label (str): The condition label.
        phase1 (str): The first phase.
        phase2 (str): The second phase.
        stride_number (int): The stride number.
        csv_filename (str): The name of the CSV file (default "outlier_runs.csv").
    """
    import os, json, pandas as pd

    file_path = os.path.join(base_save_dir_condition, csv_filename)

    # Prepare new rows as a DataFrame, skipping mice with no outliers.
    rows = []
    for mouse, indices in outlier_runs.items():
        if not indices:  # Skip if empty
            continue
        outlier_real_runs = outlier_runs.get(mouse, [])
        real_runs_str = ",".join(str(r) for r in outlier_real_runs)

        frame_numbers = []
        for run in outlier_real_runs:
            try:
                frame = int(stride_data.loc[(mouse, run)].index.get_level_values('FrameIdx')[0])
                frame_numbers.append(frame)
            except Exception as e:
                frame_numbers.append("NA")
        frames_str = json.dumps(frame_numbers)

        rows.append({
            "condition_label": condition_label,
            "mouseID": mouse,
            "phase1": phase1,
            "phase2": phase2,
            "stride_number": stride_number,
            "outlier_run_indices": real_runs_str,
            "frame_numbers": frames_str
        })

    if not rows:
        print("No outlier entries to update.")
        return

    new_df = pd.DataFrame(rows, columns=["condition_label", "mouseID", "phase1", "phase2",
                                          "stride_number", "outlier_run_indices", "frame_numbers"])

    # Load the existing CSV if it exists.
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
    else:
        existing_df = pd.DataFrame(columns=["condition_label", "mouseID", "phase1", "phase2",
                                            "stride_number", "outlier_run_indices", "frame_numbers"])

    # Combine existing and new rows, then drop duplicates based on the key columns.
    key_columns = ["condition_label", "mouseID", "phase1", "phase2", "stride_number"]
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    # Drop duplicates, keeping the last occurrence (i.e. the new row)
    combined_df = combined_df.drop_duplicates(subset=key_columns, keep="last")

    combined_df.to_csv(file_path, index=False)
    print(f"Updated outlier runs CSV: {file_path}")




def exp_growth(x, y0, L, k):
    """
    Exponential growth function:
      y(x) = y0 + L * (1 - exp(-k * x))
    """
    return y0 + L * (1 - np.exp(-k * x))


def exp_growth_derivative(x, y0, L, k):
    """Analytical derivative of the exponential growth model."""
    return L * k * np.exp(-k * x)

# Type alias for a summary dictionary of curves or derivatives.
# The keys in the inner dictionary:
#   'x_vals': np.ndarray
#   'individual': mapping from mouse id (str) to np.ndarray
#   'mean': np.ndarray
#   'sem': np.ndarray
SummaryDict = Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]

# Key type for phase and stride.
KeyType = Tuple[str, str, int]

# Type for the plateau and learning rate dictionaries.
ParameterDict = Dict[KeyType, Dict[str, float]]

# --------------------------------------------------------------------------
# Helper: Delayed exponential function.
# --------------------------------------------------------------------------
def delayed_exp_growth(x, y0, L, k, delay):
    """
    Compute a delayed exponential function.
    For x < delay, returns y0.
    For x >= delay, returns y0 + L*(1 - exp(-k*(x-delay))).
    """
    x = np.array(x)
    result = np.empty_like(x)
    mask = x >= delay
    result[~mask] = y0
    result[mask] = y0 + L * (1 - np.exp(-k * (x[mask] - delay)))
    return result


# --------------------------------------------------------------------------
# Helper: Pre-assign colors for each mouse (across all strides).
# --------------------------------------------------------------------------
def assign_mouse_colors(colourmap) -> Dict[str, any]:
    """Create a dictionary mapping each unique mouse id to a color."""
    all_mice = global_settings["mouse_ids"]
    all_mice = sorted(all_mice)  # stable order
    num = len(all_mice)
    cmap = plt.get_cmap(colourmap)
    colors = cmap(np.linspace(0, 1, num))
    #colors = plt.cm.viridis(np.linspace(0, 1, num))
    return {mouse: colors[i] for i, mouse in enumerate(all_mice)}


# --------------------------------------------------------------------------
# Helper: Exponential function with offset for fitting.
# --------------------------------------------------------------------------
def exp_growth_offset_fit(x, y0, L, k):
    """
    Exponential model with a fixed offset of 10.
    x is assumed to be >= 10.
    Model: y = y0 + L*(1 - exp(-k*(x-10)))
    """
    return y0 + L * (1 - np.exp(-k * (x - 10)))


# --------------------------------------------------------------------------
# Helper: Process a single mouse's prediction data.
# --------------------------------------------------------------------------
def process_mouse_prediction(data, common_x, full_x, stride_number):
    """
    For a given mouse's data, fit two exponential models:
      1. Full model: fit on all APA runs (full_x) using exp_growth.
      2. Offset model: fit on APA runs excluding the first 10 (common_x) using exp_growth_offset_fit.
    Then extrapolate both fits over full_x.

    Returns a dictionary with:
      - 'full_curve': the full model curve over full_x.
      - 'offset_curve': the offset model curve over full_x.
      - 'diff_curve': offset_curve minus full_curve.
      - Also returns fitted parameters (params_full and params_offset) and an interpolation on the fitting window for plotting.

    The data are normalized using the full APA data.
    """
    x_data = np.array(data.x_vals)
    y_data = np.array(data.smoothed_scaled_pred)

    # Mask full APA runs.
    full_mask = np.isin(x_data, full_x)
    x_data_full = x_data[full_mask]
    y_data_full = y_data[full_mask]
    norm = max(abs(y_data_full.min()), abs(y_data_full.max()))
    if norm == 0:
        norm = 1
    y_full_norm = y_data_full / norm

    # Mask for offset fitting (excluding first 10 runs).
    offset_mask = np.isin(x_data, common_x)
    x_data_offset = x_data[offset_mask]
    y_data_offset = y_data[offset_mask] / norm

    # Fit the full model on full APA data.
    p0_full = [y_full_norm[0], np.max(y_full_norm) - y_full_norm[0], 0.1]
    bounds_full = ([-np.inf, 0, 0], [np.inf, np.inf, np.inf])
    params_full, _ = curve_fit(exp_growth, x_data_full, y_full_norm, p0=p0_full, bounds=bounds_full)
    full_curve = exp_growth(full_x, *params_full)

    # Fit the offset model on offset data (x >= 10).
    p0_offset = [y_data_offset[0], np.max(y_data_offset) - y_data_offset[0], 0.1]
    bounds_offset = ([-np.inf, 0, 0], [np.inf, np.inf, np.inf])
    params_offset, _ = curve_fit(exp_growth_offset_fit, x_data_offset, y_data_offset, p0=p0_offset,
                                 bounds=bounds_offset)
    offset_curve = exp_growth_offset_fit(full_x, *params_offset)

    # Difference curve: offset minus full.
    diff_curve = offset_curve - full_curve

    # For individual plotting on the fitting window (common_x), interpolate:
    interp_full = np.interp(common_x, x_data_full, exp_growth(x_data_full, *params_full))
    interp_offset = exp_growth_offset_fit(common_x, *params_offset)

    return {
        'params_full': params_full,
        'full_curve': full_curve,
        'params_offset': params_offset,
        'offset_curve': offset_curve,
        'diff_curve': diff_curve,
        'interp_full': interp_full,
        'interp_offset': interp_offset,
        'norm': norm
    }


# --------------------------------------------------------------------------
# Helper: Plot individual traces for one stride.
# --------------------------------------------------------------------------
def plot_individual_traces(stride_number: int, pred_list: List, common_x, full_x, save_dir: str,
                           mouse_colors: Dict[str, any]):
    """
    For each mouse in the stride, plot:
      - The raw normalized data (for the offset fitting window).
      - The fitted full model (on full APA runs) over the fitting window.
      - The fitted offset model (with a fixed offset of 10) extrapolated over full APA runs.
    """
    plt.figure(figsize=(10, 6))
    for data in pred_list:
        try:
            result = process_mouse_prediction(data, common_x, full_x, stride_number)
            color = mouse_colors.get(data.mouse_id, 'gray')
            # Plot raw data for offset fitting.
            offset_mask = np.isin(np.array(data.x_vals), common_x)
            x_data_offset = np.array(data.x_vals)[offset_mask]
            y_data_offset = np.array(data.smoothed_scaled_pred)[offset_mask] / result['norm']
            plt.plot(x_data_offset, y_data_offset, color=color, ls='--', label=f'Data {data.mouse_id}')
            # Plot the full model curve (on the fitting window).
            plt.plot(common_x, result['interp_full'], color=color, label=f'Full {data.mouse_id}')
            # Plot the offset model curve (extrapolated over full APA).
            plt.plot(full_x, result['offset_curve'], color=color, ls=':', label=f'Offset {data.mouse_id}')
        except Exception as e:
            print(f"Error processing data for mouse {data.mouse_id}: {e}")
    plt.xlabel('Run')
    plt.ylabel('Normalized Prediction')
    plt.title(f'Exponential Fits (Full vs Offset) (Stride {stride_number})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.75)
    trace_save_path = os.path.join(save_dir, f"Exponential_Fit_Stride_{stride_number}_traces.png")
    plt.savefig(trace_save_path, dpi=300, bbox_inches='tight')
    plt.close()


# --------------------------------------------------------------------------
# Helper: Generic function to plot aggregated data.
# --------------------------------------------------------------------------
def plot_aggregated_subplot(x_vals, individual_dict: Dict, ylabel: str, title: str, save_path: str, filename: str):
    keys = list(individual_dict.keys())
    curves = [individual_dict[k] for k in keys]
    curves_array = np.array(curves)
    mean_curve = np.mean(curves_array, axis=0)
    sem_curve = np.std(curves_array, axis=0) / np.sqrt(curves_array.shape[0])
    plt.figure(figsize=(10, 6))
    for key in keys:
        plt.plot(x_vals, individual_dict[key], ls='-', label=f'Mouse {key}')
    plt.plot(x_vals, mean_curve, linewidth=2, label='Mean', color='black')
    plt.fill_between(x_vals, mean_curve - sem_curve, mean_curve + sem_curve, color='black', alpha=0.1)
    plt.xlabel('Run')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    full_save_path = os.path.join(save_path, filename)
    plt.savefig(full_save_path, dpi=300)
    plt.close()
    return mean_curve, sem_curve


# --------------------------------------------------------------------------
# Helper: Plot aggregated results across strides.
# --------------------------------------------------------------------------
def plot_aggregated_results(summary_dict: Dict, ylabel: str, title: str, save_path: str, filename: str,
                            x_axis_key: str):
    plt.figure(figsize=(10, 8))
    for key, summary in sorted(summary_dict.items(), reverse=True):
        x_vals = summary[x_axis_key]
        mean_val = summary['mean']
        sem_val = summary['sem']
        plt.plot(x_vals, mean_val, linewidth=2, label=f"Stride {key[2]}")
        plt.fill_between(x_vals, mean_val - sem_val, mean_val + sem_val, alpha=0.1)
    plt.vlines(x=[9.5, 109.5], ymin=-1, ymax=1, color='red', linestyle='--')
    plt.xlabel('Run')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    full_save_path = os.path.join(save_path, filename)
    plt.savefig(full_save_path, dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# Main function (refactored with two exponential fits and difference calculation)
# --------------------------------------------------------------------------
def fit_exponential_to_prediction(stride_dict: Dict[int, List],
                                  save_dir: str, phase1: str, phase2: str,
                                  condition: str, exp: str) -> Tuple[
    Dict[tuple, dict],
    Dict[tuple, dict],
    Dict[tuple, dict],
    Dict[tuple, dict],
    Dict[tuple, dict]]:
    """
    For each stride, fit two exponential models per mouse:
      - A full model fitted on all APA runs (full_x).
      - An offset model fitted on APA runs excluding the first 10 (common_x) using an offset of 10.
    Then extrapolate both over full_x and compute the difference (offset - full).
    Individual traces and aggregated curves are plotted.
    """
    # Fixed color mapping for aggregated plots (by stride).
    stride_color_mapping = {
        -3: plt.cm.Blues(0.2),
        -2: plt.cm.Blues(0.45),
        -1: plt.cm.Blues(0.7),
        0: plt.cm.Blues(0.99)
    }
    # Pre-assign mouse colors.
    mouse_colors = assign_mouse_colors('viridis')

    plateau_dict = {}
    learning_rate_dict = {}
    summary_curves_dict = {}  # based on full model fits on common_x (interpolated)
    summary_differences_dict = {}  # difference between offset and full curves (both extrapolated over full_x)

    con = condition.split('_')[0]
    apa_runs = expstuff['condition_exp_runs'][con][exp]['APA']
    common_x = apa_runs[10:]  # APA runs excluding the first 10
    full_x = apa_runs  # full APA runs

    # Process each stride.
    for stride_number, pred_list in stride_dict.items():
        plateau_temp = {}
        learning_rate_temp = {}
        individual_full = {}  # full exponential (fitted on full data)
        individual_offset = {}  # offset exponential (fitted on data with x>=10)

        full_list = []  # aggregated full curve on full_x (per mouse)
        offset_list = []  # aggregated offset curve on full_x (per mouse)

        # Plot individual traces for this stride.
        plot_individual_traces(stride_number, pred_list, common_x, full_x, save_dir, mouse_colors)

        for data in pred_list:
            try:
                result = process_mouse_prediction(data, common_x, full_x, stride_number)
                # Save fitted parameters (using offset model's k as representative, for example)
                plateau_temp[data.mouse_id] = result['params_offset'][0]  # you may choose a different value
                learning_rate_temp[data.mouse_id] = result['params_offset'][2]
                individual_full[data.mouse_id] = result['full_curve']
                individual_offset[data.mouse_id] = result['offset_curve']
                full_list.append(result['full_curve'])
                offset_list.append(result['offset_curve'])
            except Exception as e:
                print(f"Error processing data for mouse {data.mouse_id}: {e}")

        # Compute aggregated curves.
        mean_full = np.mean(np.array(full_list), axis=0)
        sem_full = np.std(np.array(full_list), axis=0) / np.sqrt(len(full_list))
        mean_offset = np.mean(np.array(offset_list), axis=0)
        sem_offset = np.std(np.array(offset_list), axis=0) / np.sqrt(len(offset_list))
        # For SEM of difference, compute individual differences first.
        diff_array = np.array(offset_list) - np.array(full_list)
        aggregated_diff = np.mean(diff_array, axis=0)
        sem_diff = np.std(diff_array, axis=0) / np.sqrt(diff_array.shape[0])

        key = (phase1, phase2, stride_number)
        plateau_dict[key] = plateau_temp
        learning_rate_dict[key] = learning_rate_temp
        summary_curves_dict[key] = {
            'x_vals': common_x,
            # note: full model was fitted on full data but we use common_x for aggregated full curve display
            'individual': individual_full,
            'mean': np.interp(common_x, full_x, mean_full),
            'sem': np.interp(common_x, full_x, sem_full)
        }
        summary_differences_dict[key] = {
            'x_vals': full_x,
            'mean': aggregated_diff,
            'sem': sem_diff
        }

        # Plot aggregated difference for this stride.
        plt.figure(figsize=(10, 6))
        plt.plot(full_x, aggregated_diff, linewidth=2, label='Aggregated Difference', color='black')
        plt.fill_between(full_x, aggregated_diff - sem_diff, aggregated_diff + sem_diff, color='black', alpha=0.1)
        plt.xlabel('Run')
        plt.ylabel('Difference (Offset - Full)')
        plt.title(f'Aggregated Difference (Offset - Full Exponential) (Stride {stride_number})')
        plt.legend()
        diff_save_path = os.path.join(save_dir, f"Differences_Exponential_Fit_Stride_{stride_number}.png")
        plt.savefig(diff_save_path, dpi=300)
        plt.close()

    # Aggregated plots across strides.
    plot_aggregated_results(summary_curves_dict, 'Normalized Prediction',
                            f'Aggregated Average Full Exponential Fits Across Strides - APA ({phase1} vs {phase2})',
                            save_dir, f"Aggregated_Average_Exponential_Fit_APA_({phase1}_vs_{phase2}).png", 'x_vals')
    plot_aggregated_results(summary_differences_dict, 'Difference (Offset - Full)',
                            f'Aggregated Average Difference Curves Across Strides - APA ({phase1} vs {phase2})',
                            save_dir, f"Aggregated_Average_Difference_Exponential_Fit_APA_({phase1}_vs_{phase2}).png",
                            'x_vals')

    return summary_curves_dict, summary_differences_dict, plateau_dict, learning_rate_dict, mouse_colors

def add_vertical_brace_curly(ax, y0, y1, x, xoffset, label=None, k_r=0.1, int_line_num=2, fontdict=None, rot_label=0, **kwargs):
    """
    Add a vertical curly brace using the curlyBrace package.
    The brace is drawn at the given x coordinate.
    """
    fig = ax.figure

    fontdict = fontdict or {}
    if 'fontsize' in kwargs:
        fontdict['fontsize'] = kwargs.pop('fontsize')

    p1 = [x, y0]
    p2 = [x, y1]
    # Do not pass the label here.12
    brace = curlyBrace(fig, ax, p1, p2, k_r=k_r, bool_auto=True, str_text=label,
                       int_line_num=int_line_num, fontdict=fontdict or {}, clip_on=False, color='black', **kwargs)
    # if label:
    #     y_center = (y0 + y1) / 2.0
    #     # Place the label to the left of the brace.
    #     ax.text(x - xoffset, y_center, label,
    #             ha="center", va="center", fontsize=12, fontweight="normal",
    #             color='black', clip_on=False, rotation=rot_label, rotation_mode='anchor',
    #             transform=ax.transData)
def add_horizontal_brace_curly(ax, x0, x1, y, label=None, k_r=0.1, int_line_num=2, fontdict=None, **kwargs):
    """
    Add a horizontal curly brace using the curlyBrace package.
    The brace is drawn at the given y coordinate.
    """
    fig = ax.figure

    fontdict = fontdict or {}
    if 'fontsize' in kwargs:
        fontdict['fontsize'] = kwargs.pop('fontsize')

    # Swap p1 and p2 so that the brace opens toward the plot.
    p1 = [x1, y]
    p2 = [x0, y]
    brace = curlyBrace(fig, ax, p2, p1, k_r=k_r, bool_auto=True, str_text=label,
                       int_line_num=int_line_num, fontdict=fontdict or {}, clip_on=False, color='black', **kwargs)
    # if label:
    #     x_center = (x0 + x1) / 2.0
    #     # Adjust the offset so the label appears above the brace.
    #     ax.text(x_center, y - 12 , label,
    #             ha="center", va="bottom", fontsize=12, fontweight="normal", color='black', clip_on=False)


def plot_top_feature_pc_single_contributors(contributions, p1, p2, s, condition_label, save_dir):
    from collections import defaultdict

    # Group series by stride:
    feature_data_by_stride = defaultdict(list)
    pc_data_by_stride = defaultdict(list)

    for key, contr in contributions.items():
        # Key example: ('1035243', 'APA2', 'Wash2', -3)
        stride = key[-1]
        feature_data_by_stride[stride].append(contr.single_feature_contribution)
        pc_data_by_stride[stride].append(contr.single_pc_contribution)

    # Compute the mean contributions for each stride:
    feature_median_by_stride = {}
    pc_median_by_stride = {}
    for stride, series_list in feature_data_by_stride.items():
        # Convert list of Series to DataFrame and compute column-wise mean
        df_features = pd.DataFrame(series_list)
        feature_median_by_stride[stride] = df_features.median()

    for stride, series_list in pc_data_by_stride.items():
        df_pcs = pd.DataFrame(series_list)
        pc_median_by_stride[stride] = df_pcs.median()

    # Plotting: Generate a bar plot for each stride – one for features and one for PCs.
    for stride, median_series in feature_median_by_stride.items():
        plt.figure(figsize=(10, 6))
        median_series.sort_values().plot(kind='bar')
        plt.title(f'Mean Single Feature Contribution for Stride {stride}')
        plt.ylabel('Contribution')
        plt.xlabel('Feature')
        plt.ylim(0.5, 1.1)
        plt.tight_layout()

        save_path = os.path.join(save_dir,f'MeanSingleFeatureContributions__{p1}vs{p2}_stride{s}_{condition_label}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

    for stride, median_series in pc_median_by_stride.items():
        plt.figure(figsize=(8, 6))
        median_series.sort_values().plot(kind='bar')
        plt.title(f'Mean Single PC Contribution for Stride {stride}')
        plt.ylabel('Contribution')
        plt.xlabel('PC')
        plt.ylim(0.5, 1.1)
        plt.tight_layout()

        save_path = os.path.join(save_dir,f'MeanSinglePCContributions__{p1}vs{p2}_stride{s}_{condition_label}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_featureXruns_heatmap(all_feats, stride_numbers, p1, p2, base_save_dir):
    #all_feats = global_data["aggregated_raw_features"]
    for stride in stride_numbers:
        feats = {key: df for key, df in all_feats.items() if key[-1] == stride}


        # Create new keys using only the mouse_id (first element in each key tuple)
        # This yields a list of keys (as strings)
        flat_keys = [str(key[0]) for key in feats.keys()]

        # Concatenate the dataframes vertically. Each piece will get a new index level from flat_keys.
        combined_df = pd.concat(list(feats.values()), axis=0, keys=flat_keys, names=["mouseID", "Run"])
        average_df = combined_df.groupby(axis=0, level='Run').median()

        # Build an ordered list from your `short_names` dictionary:
        ordered_features = [feat for feat in short_names.keys() if feat in average_df.columns]
        # Find any features not in short_names:
        remaining_features = [feat for feat in average_df.columns if feat not in ordered_features]
        # Define the final order as those in short_names order followed by the remaining features.
        final_order = ordered_features + remaining_features

        # Reorder the DataFrame columns.
        ordered_average_df = average_df[final_order]

        # Now rename the columns using the dictionary: if a feature is in short_names, use its value;
        # otherwise leave it unchanged.
        renamed_columns = [short_names.get(feat, feat) for feat in final_order]
        ordered_average_df.columns = renamed_columns

        # --- PREPARE FOR PLOTTING ---
        # Transpose the DataFrame so that rows are features and columns are runs.
        heatmap_data = ordered_average_df.T

        # smooth with medfilt
        smooth = heatmap_data.apply(lambda  x: medfilt(x, kernel_size=3), axis=1)
        # Convert the Series of NumPy arrays into a DataFrame:
        smooth_df = pd.DataFrame(smooth.tolist(), index=heatmap_data.index, columns=heatmap_data.columns)

        # Plot the heatmap
        plt.figure(figsize=(20,20))
        sns.heatmap(smooth_df, cmap='coolwarm', cbar=True)
        plt.axvline(x=10, color='b', linestyle='--')
        plt.axvline(x=110, color='b', linestyle='--')
        plt.xlabel('Trial')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
















