import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy
from scipy.signal import medfilt
import matplotlib.pyplot as plt

from apa_analysis.Legacy_methods import utils_feature_reduction as utils

def compute_regression(X, y):
    model = LogisticRegression(penalty='none', fit_intercept=False)
    model.fit(X.T, y)
    w = model.coef_

    y_pred = np.dot(w, X)
    # change y_pred +ves to 1 and -ves to 0
    y_pred[y_pred > 0] = 1
    y_pred[y_pred < 0] = 0

    bal_acc = balanced_accuracy(y.T, y_pred.T)

    return w, bal_acc

def compute_lasso_regression(X, y):
    # Use L1 penalty for Lasso-style logistic regression.
    model = LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=False, C=1.0)
    model.fit(X.T, y)
    w = model.coef_

    y_pred = np.dot(w, X)
    # change y_pred +ves to 1 and -ves to 0
    y_pred[y_pred > 0] = 1
    y_pred[y_pred < 0] = 0

    bal_acc = balanced_accuracy(y.T, y_pred.T)

    return w, bal_acc


def find_unique_and_single_contributions(selected_scaled_data_df, loadings_df, normalize_mean, normalize_std, y_reg, full_accuracy):
    single_all = []
    unique_all = []

    # Shuffle X and run logistic regression
    for fidx, feature in enumerate(selected_scaled_data_df.index):
        #print(f"Shuffling feature {fidx + 1}/{len(selected_scaled_data_df.index)}")
        single_shuffle = utils.shuffle_single(feature, selected_scaled_data_df)
        unique_shuffle = utils.shuffle_unique(feature, selected_scaled_data_df)
        Xdr_shuffled_single = np.dot(loadings_df.T, single_shuffle)
        Xdr_shuffled_single = ((Xdr_shuffled_single.T - normalize_mean) / normalize_std).T
        Xdr_shuffled_unique = np.dot(loadings_df.T, unique_shuffle)
        Xdr_shuffled_unique = ((Xdr_shuffled_unique.T - normalize_mean) / normalize_std).T

        _, single_accuracy = compute_regression(Xdr_shuffled_single, y_reg)
        _, unique_accuracy = compute_regression(Xdr_shuffled_unique, y_reg)
        unique_contribution = full_accuracy - unique_accuracy
        single_all.append(single_accuracy)
        unique_all.append(unique_contribution)
    single_all = np.array(single_all)
    unique_all = np.array(unique_all)
    single_all_dict = dict(zip(selected_scaled_data_df.index, single_all.flatten()))
    unique_all_dict = dict(zip(selected_scaled_data_df.index, unique_all.flatten()))

    return single_all_dict, unique_all_dict

def find_unique_and_single_contributions_pcs(pc_df, normalize_mean, normalize_std, y_reg, full_accuracy):
    single_all = []
    unique_all = []

    # Iterate over each PC (treated as a feature)
    for pc in pc_df.columns:
        # Shuffle only the current PC while keeping the other PCs intact
        single_shuffle = utils.shuffle_single(pc, pc_df.T)
        unique_shuffle = utils.shuffle_unique(pc, pc_df.T)

        # For PCs, we work directly with the scores (no loadings multiplication)
        X_shuffled_single = ((single_shuffle.T - normalize_mean) / normalize_std)
        X_shuffled_unique = ((unique_shuffle.T - normalize_mean) / normalize_std)

        # Note: The regression expects data in shape (n_features, n_samples)
        _, single_accuracy = compute_regression(X_shuffled_single.values.T, y_reg)
        _, unique_accuracy = compute_regression(X_shuffled_unique.values.T, y_reg)
        unique_contribution = full_accuracy - unique_accuracy

        single_all.append(single_accuracy)
        unique_all.append(unique_contribution)

    single_all_dict = dict(zip(pc_df.columns, single_all))
    unique_all_dict = dict(zip(pc_df.columns, unique_all))

    return single_all_dict, unique_all_dict

    # shuffle_all = utils.shuffle_single(feature, selected_scaled_data_df)
    # shuffle_all = utils.shuffle_unique(feature, shuffle_all)
    # shuffle_all = np.dot(loadings_df.T, shuffle_all)
    # shuffle_all = ((shuffle_all.T - normalize_mean) / normalize_std).T
    # _, full_accuracy_shuffled = compute_regression(shuffle_all, y_reg, selected_scaled_data_df, cv=5)
    #
    #
def find_full_shuffle_accuracy(selected_scaled_data_df, loadings_df, normalize_mean, normalize_std, y_reg, full_accuracy):
    feature = selected_scaled_data_df.index[0]

    shuffle_all = utils.shuffle_single(feature, selected_scaled_data_df)
    shuffle_all = utils.shuffle_unique(feature, shuffle_all)
    shuffle_all = np.dot(loadings_df.T, shuffle_all)
    shuffle_all = ((shuffle_all.T - normalize_mean) / normalize_std).T
    _, full_accuracy_shuffled = compute_regression(shuffle_all, y_reg)
    return full_accuracy_shuffled


def fit_regression_model(loadings_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2):
    # Transform X (scaled feature data) to Xdr (PCA space) - ie using the loadings from PCA
    Xdr = np.dot(loadings_df.T, reduced_feature_selected_data_df)

    # Normalize X # todo this is the second normalisation!!!
    Xdr, normalize_mean, normalize_std = utils.normalize_Xdr(Xdr)

    # Create y (regression target) - 1 for phase1, 0 for phase2
    y_reg = np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))])

    # Run logistic regression on the full model
    w, full_accuracy = compute_regression(Xdr, y_reg)
    print(f"Full model accuracy: {full_accuracy:.3f}")

    # return w, y_reg, full_accuracy
    return w, normalize_mean, normalize_std, y_reg, full_accuracy
    # return w, y_reg, full_accuracy


def predict_runs(loadings_df, reduced_feature_data_df, normalize_mean, normalize_std, w, save_path, mouse_id, phase1, phase2, stride_number, condition_name, plot_pred):
    # Apply the full model to all runs (scaled and unscaled)
    all_trials_dr = np.dot(loadings_df.T, reduced_feature_data_df.T)
    all_trials_dr = ((all_trials_dr.T - normalize_mean) / normalize_std).T # pc wise normalization
    run_pred = np.dot(w, np.dot(loadings_df.T, reduced_feature_data_df.T))
    run_pred_scaled = np.dot(w, all_trials_dr)

    # Compute smoothed scaled predictions for aggregation.
    kernel_size = 5
    padded_run_pred = np.pad(run_pred[0], pad_width=kernel_size, mode='reflect')
    padded_run_pred_scaled = np.pad(run_pred_scaled[0], pad_width=kernel_size, mode='reflect')
    smoothed_pred = medfilt(padded_run_pred, kernel_size=kernel_size)
    smoothed_scaled_pred = medfilt(padded_run_pred_scaled, kernel_size=kernel_size)
    smoothed_pred = smoothed_pred[kernel_size:-kernel_size]
    smoothed_scaled_pred = smoothed_scaled_pred[kernel_size:-kernel_size]

    # Plot run prediction
    if plot_pred:
        utils.plot_run_prediction(reduced_feature_data_df, run_pred, smoothed_pred, save_path, mouse_id, phase1, phase2, stride_number,
                                  scale_suffix="", dataset_suffix=condition_name)
        utils.plot_run_prediction(reduced_feature_data_df, run_pred_scaled, smoothed_scaled_pred, save_path, mouse_id, phase1, phase2, stride_number,
                                  scale_suffix="scaled", dataset_suffix=condition_name)
    return smoothed_scaled_pred, run_pred_scaled

def regression_feature_contributions(loadings_df, reduced_feature_selected_data_df, mouse_id, phase1, phase2, condition, stride_number, save_path, normalize_mean, normalize_std, y_reg, full_accuracy):
    # Shuffle features and run logistic regression to find unique contributions and single feature contributions
    single_all_dict, unique_all_dict = find_unique_and_single_contributions(reduced_feature_selected_data_df,
                                                                            loadings_df, normalize_mean,
                                                                            normalize_std, y_reg, full_accuracy)

    # Find full shuffled accuracy (one shuffle may not be enough)
    full_shuffled_accuracy = find_full_shuffle_accuracy(reduced_feature_selected_data_df, loadings_df,
                                                        normalize_mean, normalize_std, y_reg, full_accuracy)
    print(f"Full model shuffled accuracy: {full_shuffled_accuracy:.3f}")

    # Plot unique and single feature contributions
    utils.plot_unique_delta_accuracy(unique_all_dict, mouse_id, save_path, title_suffix=f"{phase1}_vs_{phase2}_stride{stride_number}_{condition}")
    utils.plot_feature_accuracy(single_all_dict, mouse_id, save_path, title_suffix=f"{phase1}_vs_{phase2}_stride{stride_number}_{condition}")
    return single_all_dict, unique_all_dict


def regression_pc_contributions(pc_df, mouse_id, phase1, phase2, condition, stride_number, save_path,
                                normalize_mean, normalize_std, y_reg, full_accuracy):
    # Compute single and unique contributions for each PC
    single_all_dict, unique_all_dict = find_unique_and_single_contributions_pcs(pc_df, normalize_mean,
                                                                                normalize_std, y_reg, full_accuracy)

    # # Compute full shuffled accuracy for the PCs
    # full_shuffled_accuracy = find_full_shuffle_accuracy_pc(pc_df, normalize_mean, normalize_std, y_reg, full_accuracy)
    # print(f"Full model shuffled accuracy for PCs: {full_shuffled_accuracy:.3f}")

    # Plot the results (reuse your plotting utilities)
    utils.plot_unique_delta_accuracy(unique_all_dict, mouse_id, save_path,
                                     title_suffix=f"{phase1}_vs_{phase2}_stride{stride_number}_{condition}_PCs")
    utils.plot_feature_accuracy(single_all_dict, mouse_id, save_path,
                                title_suffix=f"{phase1}_vs_{phase2}_stride{stride_number}_{condition}_PCs")
    return single_all_dict, unique_all_dict


def run_regression(loadings_df, pcs_p1p2_df, reduced_feature_data_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2, mouse_id, phase1, phase2, stride_number, save_path, condition, plot_pred=True, plot_weights=True):
    from Analysis.Tools.config import global_settings
    w, normalize_mean, normalize_std, y_reg, full_accuracy = fit_regression_model(loadings_df, reduced_feature_selected_data_df, mask_phase1, mask_phase2)

    # Trim the weights and normalization parameters to the number of PCs to use
    w = np.array(w[0][:global_settings['pcs_to_use']]).reshape(1, -1)
    normalize_mean = normalize_mean[:global_settings['pcs_to_use']]
    normalize_std = normalize_std[:global_settings['pcs_to_use']]
    loadings_df = loadings_df.iloc(axis=1)[:global_settings['pcs_to_use']].copy()

    # Compute feature contributions
    single_f, unique_f = regression_feature_contributions(loadings_df, reduced_feature_selected_data_df, mouse_id, phase1, phase2, condition, stride_number, save_path, normalize_mean, normalize_std, y_reg, full_accuracy)
    single_pc, unique_pc = regression_pc_contributions(pcs_p1p2_df, mouse_id, phase1, phase2, condition, stride_number, save_path, normalize_mean, normalize_std, y_reg, full_accuracy)

    # Compute feature-space weights for this mouse
    feature_weights = loadings_df.dot(w.T).squeeze()

    if plot_weights:
        # Plot the weights in the original feature space
        utils.plot_weights_in_feature_space(feature_weights, save_path, mouse_id, phase1, phase2, stride_number, condition)

    # Predict runs using the full model
    smoothed_scaled_pred, _ = predict_runs(loadings_df, reduced_feature_data_df, normalize_mean, normalize_std, w, save_path, mouse_id, phase1, phase2, stride_number, condition, plot_pred)

    return smoothed_scaled_pred, feature_weights, w , normalize_mean, normalize_std, single_f, unique_f, single_pc, unique_pc

# def predict_compare_condition(feature_data_compare, mouse_id, compare_condition, stride_number, phase1, phase2, selected_features, loadings_df, w, save_path):
#     # Retrieve reduced feature data for the comparison condition
#     # _, comparison_selected_scaled_data, _, _, _, _ = select_runs_data(mouse_id, stride_number, compare_condition, exp, day, stride_data_compare, phase1, phase2)
#     #comparison_scaled_data = utils.load_and_preprocess_data(mouse_id, stride_number, compare_condition, exp, day)
#     comparison_scaled_data = feature_data_compare.loc(axis=0)[stride_number, mouse_id]
#     comparison_reduced_feature_data_df = comparison_scaled_data.loc(axis=1)[selected_features]
#     runs = list(comparison_reduced_feature_data_df.index)
#
#     # Transform X (scaled feature data) to Xdr (PCA space) - ie using the loadings from PCA
#     Xdr = np.dot(loadings_df.T, comparison_reduced_feature_data_df.T)
#     # Normalize X
#     Xdr, normalize_mean, normalize_std = utils.normalize(Xdr)
#
#     save_path_compare = os.path.join(save_path, f"vs_{compare_condition}")
#     # prefix path wth \\?\ to avoid Windows path length limit
#     save_path_compare = "\\\\?\\" + save_path_compare
#     os.makedirs(save_path_compare, exist_ok=True)
#     smoothed_scaled_pred, _ = predict_runs(loadings_df, comparison_reduced_feature_data_df, normalize_mean, normalize_std, w, save_path_compare, mouse_id, phase1, phase2, stride_number, compare_condition)
#
#     return smoothed_scaled_pred, runs


# def compute_global loadings_df}_regression_model(feature_data, global_mouse_ids, stride_number, phase1, phase2, condition, exp, day, stride_data,
# #                                     selected_features, loadings_df):
# #     aggregated_data_list = []
# #     y_list = []
# #     for mouse_id in global_mouse_ids:
# #         #scaled_data_df = utils.load_and_preprocess_data(mouse_id, stride_number, condition, exp, day)
# #         scaled_data_df = feature_data.loc(axis=0)[stride_number, mouse_id]
# #         # Get phase masks and runs.
# #         run_numbers, _, mask_phase1, mask_phase2 = utils.get_runs(scaled_data_df, stride_data, mouse_id, stride_number,
# #                                                             phase1, phase2)
# #         selected_mask = mask_phase1 | mask_phase2
# #         selected_data = scaled_data_df.loc[selected_mask][selected_features]
# #         aggregated_data_list.append(selected_data)
# #         # Create labels: 1 for phase1, 0 for phase2.
# #         y_list.append(np.concatenate([np.ones(np.sum(mask_phase1)), np.zeros(np.sum(mask_phase2))]))
# #
# #     # Combine all data across mice.
# #     global_data_df = pd.concat(aggregated_data_list)
# #     y_global = np.concatenate(y_list)
# #
# #     # Project aggregated data into PCA space using the global loadings.
# #     Xdr = np.dot(loadings_df.T, global_data_df.T)
# #     Xdr, norm_mean, norm_std = utils.normalize(Xdr)
# #
# #     # Compute regression weights (using your chosen regression function).
# #     w, full_accuracy = compute_regression(Xdr, y_global)
# #     print(f"Global regression model accuracy for {phase1} vs {phase2}: {full_accuracy:.3f}")
# #
# #     return {'w': w, 'norm_mean': norm_mean, 'norm_std': norm_std, 'selected_features': selected_features,
# #             'loadings_df':

def plot_LOO_regression_accuracies(mouse_accuracies, phase1, phase2, stride_number, base_save_dir_condition):
    # Plot the LOO accuracies for this (phase1, phase2, stride) combination.
    plt.figure(figsize=(8, 4))
    mice = list(mouse_accuracies.keys())
    accuracies = list(mouse_accuracies.values())
    plt.bar(mice, accuracies, color="skyblue")
    plt.axhline(0.8, color="red", linestyle="--", label="Threshold (0.8)")
    plt.xlabel("Mouse ID")
    plt.ylabel("LOO Accuracy")
    plt.title(f"LOO Accuracies: {phase1} vs {phase2}, Stride {stride_number}")
    plt.legend()
    plot_path = os.path.join(base_save_dir_condition, "LeaveOneOut", f"LOO_{phase1}_{phase2}_stride{stride_number}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved LOO accuracy plot for {phase1} vs {phase2}, stride {stride_number} to {plot_path}")



