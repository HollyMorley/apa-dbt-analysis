import numpy as np
import pandas as pd
import os
import random
import ast
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from apa_analysis.Legacy_methods import utils_feature_reduction as utils
# from Analysis.Legacy_methods.PCA import compute_global_pca_for_phase


def rfe_feature_selection(selected_scaled_data_df, y, cv=5, min_features_to_select=5, C=1.0):
    """
    Performs feature selection using RFECV with L1-regularized logistic regression.
    Parameters:
      - selected_scaled_data_df: DataFrame with features as rows and samples as columns.
      - y: target vector.
      - cv: number of folds for cross-validation.
      - min_features_to_select: minimum number of features RFECV is allowed to select.
      - C: Inverse regularization strength (higher values reduce regularization).
    """
    # Transpose so that rows are samples and columns are features.
    X = selected_scaled_data_df.T
    estimator = LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=False, C=C)
    rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='balanced_accuracy',
                  min_features_to_select=min_features_to_select)
    rfecv.fit(X, y)
    selected_features = selected_scaled_data_df.index[rfecv.support_]
    print(f"RFECV selected {rfecv.n_features_} features.")
    return selected_features, rfecv

def random_forest_feature_selection(selected_scaled_data_df, y):
    """
    Performs feature selection using a Random Forest to rank features and selects those
    with importance above the median.
    """
    X = selected_scaled_data_df.T  # rows: samples, columns: features
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    threshold = np.median(importances)
    selected_features = selected_scaled_data_df.index[importances > threshold]
    print(f"Random Forest selected {len(selected_features)} features (threshold: {threshold:.4f}).")
    return selected_features, rf

def sequential_feature_selector(X, y):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SequentialFeatureSelector

    # Instantiate your random forest regressor.
    rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42)

    # Set up the sequential feature selector.
    # 'forward' selection means we start with no features and add one at a time.
    # Adjust 'n_features_to_select' to a desired number or use 'auto' to determine automatically.
    sfs = SequentialFeatureSelector(
        rf_estimator,
        n_features_to_select='auto',  # or an integer value
        direction='forward',
        cv=5,  # 5-fold cross-validation
        scoring='r2',  # or another metric appropriate for your regression task
        n_jobs=-1
    )

    # Assume X is your feature matrix and y is your target variable.
    # You might, for example, get X from your `selected_scaled_data_df.T` and define y accordingly.
    sfs.fit(X, y)

    # After fitting, sfs.get_support() returns a boolean mask of selected features.
    selected_features = X.columns[sfs.get_support()]
    print("Selected features:", selected_features)


def global_feature_selection(feature_data, mice_ids, stride_number, phase1, phase2, condition, exp, day, stride_data, save_dir,
                             c, nFolds, n_iterations, overwrite, method='regression'):
    results_file = os.path.join(save_dir, f'global_feature_selection_results_{phase1}_{phase2}_stride{stride_number}.csv')

    aggregated_data_list = []
    aggregated_y_list = []
    total_run_numbers = []

    for mouse_id in mice_ids:
        # Load and preprocess data for each mouse.
        #scaled_data_df = utils.load_and_preprocess_data(mouse_id, stride_number, condition, exp, day)
        scaled_data_df = feature_data.loc(axis=0)[stride_number, mouse_id]
        # Get runs and phase masks.
        run_numbers, _, mask_phase1, mask_phase2 = utils.get_runs(scaled_data_df, stride_data, mouse_id, stride_number,
                                                            phase1, phase2)
        selected_mask = mask_phase1 | mask_phase2
        # Transpose so that rows are features and columns are runs.
        selected_data = scaled_data_df.loc[selected_mask].T
        aggregated_data_list.append(selected_data)
        # Create the regression target.
        y_reg = np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))])
        aggregated_y_list.append(y_reg)
        # (Store run indices as simple integers for each mouse.)
        total_run_numbers.extend(list(range(selected_data.shape[1])))

    # Combine data across mice.
    aggregated_data_df = pd.concat(aggregated_data_list, axis=1)
    aggregated_y = np.concatenate(aggregated_y_list)

    # Call unified_feature_selection (choose method as desired: 'rfecv', 'rf', or 'regression')
    selected_features, fs_results_df = utils.unified_feature_selection(
                                        feature_data_df=aggregated_data_df,
                                        y=aggregated_y,
                                        c=c,
                                        method=method,
                                        cv=nFolds,
                                        n_iterations=n_iterations,
                                        save_file=results_file,
                                        overwrite_FeatureSelection=overwrite
                                    )
    print(f"Global selected features: {selected_features}, \nLength: {len(selected_features)}")
    return selected_features, fs_results_df


# def select_global_features_and_run_global_PCA(mouse_ids, stride_number, phase1, phase2, condition, exp, day, stride_data,
#                                               c, nFolds, n_iterations, overwrite, method, global_fs_dir):
#     from Analysis.Legacy_methods.PCA import compute_global_pca_for_phase
#     print(f"Performing global feature selection for {phase1} vs {phase2}, stride {stride_number}.")
#     # Perform global feature selection.
#     selected_features, fs_df = global_feature_selection(mouse_ids, stride_number, phase1, phase2, condition, exp, day,
#         stride_data, save_dir=global_fs_dir, c=c, nFolds=nFolds, n_iterations=n_iterations, overwrite=overwrite, method=method)
#
#     # Compute global PCA using the selected features.
#     pca, loadings_df = compute_global_pca_for_phase(mouse_ids, stride_number, phase1, phase2, condition, exp, day,
#                                                     stride_data, selected_features)
#     return selected_features, fs_df, pca, loadings_df
#
