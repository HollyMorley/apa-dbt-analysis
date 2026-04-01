import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from apa_analysis.Legacy_methods import utils_feature_reduction as utils

def perform_pca(scaled_data_df, n_components):
    """
    Perform PCA on the standardized data.
    """
    pca = PCA(n_components=n_components)
    pca.fit(scaled_data_df)
    pcs = pca.transform(scaled_data_df)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=scaled_data_df.columns,
                               columns=[f'PC{i + 1}' for i in range(n_components)])
    return pca, pcs, loadings_df


def plot_pca(pca, pcs, labels, p1, p2, stride, stepping_limbs, run_numbers, mouse_id, condition_label, save_path):
    """
    Create and save 2D and 3D PCA scatter plots.
    """
    n_pc = pcs.shape[1]
    df_plot = pd.DataFrame(pcs, columns=[f'PC{i + 1}' for i in range(n_pc)])
    df_plot['Condition'] = labels
    df_plot['SteppingLimb'] = stepping_limbs
    df_plot['Run'] = run_numbers

    markers_all = {'ForepawL': 'X', 'ForepawR': 'o'}
    unique_limbs = df_plot['SteppingLimb'].unique()
    current_markers = {}
    for limb in unique_limbs:
        if limb in markers_all:
            current_markers[limb] = markers_all[limb]
        else:
            raise ValueError(f"No marker defined for stepping limb: {limb}")

    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by PC1: {explained_variance[0] * 100:.2f}%")
    print(f"Explained variance by PC2: {explained_variance[1] * 100:.2f}%")
    if pca.n_components_ >= 3:
        print(f"Explained variance by PC3: {explained_variance[2] * 100:.2f}%")

    # 2D Scatter
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_plot,
        x='PC1',
        y='PC2',
        hue='Condition',
        style='SteppingLimb',
        markers=current_markers,
        s=100,
        alpha=0.7
    )
    plt.title(f'PCA: PC1 vs PC2 for Mouse {mouse_id}')
    plt.xlabel(f'PC1 ({explained_variance[0] * 100:.1f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1] * 100:.1f}%)')
    plt.legend(title='Condition & Stepping Limb', bbox_to_anchor=(1.05, 1), loc=2)
    plt.grid(True)
    for _, row in df_plot.iterrows():
        plt.text(row['PC1'] + 0.02, row['PC2'] + 0.02, str(row['Run']), fontsize=8, alpha=0.7)
    padding_pc1 = (df_plot['PC1'].max() - df_plot['PC1'].min()) * 0.05
    padding_pc2 = (df_plot['PC2'].max() - df_plot['PC2'].min()) * 0.05
    plt.xlim(df_plot['PC1'].min() - padding_pc1, df_plot['PC1'].max() + padding_pc1)
    plt.ylim(df_plot['PC2'].min() - padding_pc2, df_plot['PC2'].max() + padding_pc2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"PCA_2D_Mouse_{mouse_id}_{p1}vs{p2}_{stride}_{condition_label}.png"), dpi=300)
    plt.close()

    # 3D Scatter (if available)
    if pca.n_components_ >= 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        palette = sns.color_palette("bright", len(df_plot['Condition'].unique()))
        conditions_unique = df_plot['Condition'].unique()
        for idx, condition in enumerate(conditions_unique):
            subset = df_plot[df_plot['Condition'] == condition]
            ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'],
                       label=condition, color=palette[idx], alpha=0.7, s=50, marker='o')
            for _, row in subset.iterrows():
                ax.text(row['PC1'] + 0.02, row['PC2'] + 0.02, row['PC3'] + 0.02,
                        str(row['Run']), fontsize=8, alpha=0.7)
        ax.set_xlabel(f'PC1 ({explained_variance[0] * 100:.1f}%)')
        ax.set_ylabel(f'PC2 ({explained_variance[1] * 100:.1f}%)')
        ax.set_zlabel(f'PC3 ({explained_variance[2] * 100:.1f}%)')
        ax.set_title(f'3D PCA for Mouse {mouse_id}')
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc=2)
        padding_pc1 = (df_plot['PC1'].max() - df_plot['PC1'].min()) * 0.05
        padding_pc2 = (df_plot['PC2'].max() - df_plot['PC2'].min()) * 0.05
        padding_pc3 = (df_plot['PC3'].max() - df_plot['PC3'].min()) * 0.05
        ax.set_xlim(df_plot['PC1'].min() - padding_pc1, df_plot['PC1'].max() + padding_pc1)
        ax.set_ylim(df_plot['PC2'].min() - padding_pc2, df_plot['PC2'].max() + padding_pc2)
        ax.set_zlim(df_plot['PC3'].min() - padding_pc3, df_plot['PC3'].max() + padding_pc3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"PCA_3D_Mouse_{mouse_id}_{p1}vs{p2}_{stride}_{condition_label}.png"), dpi=300)
        plt.close()


def plot_scree(pca, p1, p2, stride, condition, save_path):
    """
    Plot and save the scree plot.
    """
    from Analysis.Tools.config import (global_settings)
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue', label='Individual Explained Variance')
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, 's--', linewidth=2, color='red', label='Cumulative Explained Variance')
    plt.title(f'Scree Plot with Cumulative Explained Variance\n{p1} vs {p2} - {condition} - {stride}\n'
              f'Num chosen PCs: {global_settings["pcs_to_use"]}')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.ylim(0, 1.05)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"Scree_Plot_{p1}_{p2}_{stride}_{condition}.png"), dpi=300)
    plt.close()


def cross_validate_pca(scaled_data_df, save_path, n_folds=10):
    """
    Perform PCA on folds of the data. For each fold, perform PCA with n_components equal to
    the number of features (or the number of training samples if lower) and record the explained variance ratio.
    This version also plots the cumulative explained variance for each fold and adds a horizontal line at 80%.

    Parameters:
        scaled_data_df (pd.DataFrame): The standardized data.
        save_path (str): Directory where the plot will be saved.
        n_folds (int): Number of cross-validation folds.

    Returns:
        fold_explained_variances (list of np.ndarray): A list containing the explained variance
                                                       ratios for each fold.
    """
    num_features = scaled_data_df.shape[1]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_explained_variances = []

    plt.figure(figsize=(40, 15))

    # For tracking the maximum number of components across folds.
    max_n_components = 0

    # Loop through folds, applying PCA on the training split of each fold.
    for fold_idx, (train_index, test_index) in enumerate(kf.split(scaled_data_df)):
        X_train = scaled_data_df.iloc[train_index]
        current_n_components = min(num_features, X_train.shape[0])
        max_n_components = max(max_n_components, current_n_components)
        #print(f"Fold {fold_idx + 1}: {current_n_components} components")

        pca = PCA(n_components=current_n_components)
        pca.fit(X_train)
        explained = pca.explained_variance_ratio_
        fold_explained_variances.append(explained)

        # Plot individual explained variance for this fold.
        plt.plot(range(1, current_n_components + 1), explained, marker='o', label=f'Fold {fold_idx + 1} EV')

        # Compute and plot cumulative explained variance for this fold.
        cumulative_explained = np.cumsum(explained)
        plt.plot(range(1, current_n_components + 1), cumulative_explained, marker='s', linestyle='--',
                 label=f'Fold {fold_idx + 1} Cumulative')

    # Add a horizontal line at 80% explained variance.
    plt.axhline(y=0.8, color='k', linestyle=':', linewidth=2, label='80% Threshold')

    plt.xlabel('Principal Component Number')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Variance Explained by Principal Components across Folds')

    # Set x-axis ticks to show every other number.
    plt.xticks(range(1, max_n_components + 1, 2))
    plt.yticks(np.linspace(0,1,21))

    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "PCA_CV_Scree_Plot.png"), dpi=300)
    plt.close()

    return fold_explained_variances

def plot_average_variance_explained_across_folds(fold_variances, p1, p2, s):
    # Determine the minimum number of components across folds.
    min_components = min(len(arr) for arr in fold_variances)

    # Trim each fold's explained variance array to min_components.
    trimmed_variances = [arr[:min_components] for arr in fold_variances]

    # Compute the average explained variance ratio across folds.
    avg_variance = np.mean(np.vstack(trimmed_variances), axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, min_components + 1), avg_variance, marker='s', color='black')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Average Explained Variance Ratio')
    plt.title('Average PCA Scree Plot across Folds')
    plt.xticks(range(1, min_components + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Average_PCA_CV_Scree_Plot_{p1}vs{p2}_{s}.png", dpi=300)
    plt.close()

def compute_global_pca_for_phase(feature_data, feature_data_compare, mouseIDs_condition, mouseIDs_compare, stride_number, phase1, phase2,
                                 stride_data, stride_data_compare, selected_features, combine_conditions,
                                 n_components):
    """
    Aggregates data from all mice in global_mouse_ids (using only runs for phase1 and phase2),
    restricts to the globally selected features, and computes PCA.
    """
    aggregated_data = []
    if combine_conditions == True:
        data_mouse_pairs = [(feature_data, mouseIDs_condition, stride_data), (feature_data_compare, mouseIDs_compare, stride_data_compare)]
    else:
        data_mouse_pairs = [(feature_data, mouseIDs_condition, stride_data)]
    for data, mouseIDs, stride_d in data_mouse_pairs:
        for mouse_id in mouseIDs:
            # scaled_data_df = utils.load_and_preprocess_data(mouse_id, stride_number, condition, exp, day)
            scaled_data_df = data.loc(axis=0)[stride_number, mouse_id]
            # Get run masks for the two phases.
            run_numbers, stepping_limbs, mask_phase1, mask_phase2 = utils.get_runs(scaled_data_df, stride_d, mouse_id, stride_number, phase1, phase2)
            # Select only runs corresponding to the phases.
            selected_mask = mask_phase1 | mask_phase2
            selected_data = scaled_data_df.loc[selected_mask]
            # Restrict to the globally selected features.
            reduced_data = selected_data[selected_features]
            aggregated_data.append(reduced_data)
    # Concatenate all runs (rows) across mice.
    global_data = pd.concat(aggregated_data)
    # Compute PCA on the aggregated data.
    # check if n_components is less than the number of features
    if n_components > global_data.shape[1]:
        n_components = global_data.shape[1]
    pca, pcs, loadings_df = perform_pca(global_data, n_components=n_components)
    return pca, loadings_df

def compute_pca_allStrides(feature_data, feature_data_compare, mouseIDs_condition, mouseIDs_compare,
                           stride_numbers, phase1, phase2,
                           stride_data, stride_data_compare, selected_features, combine_conditions,
                           n_components):
    aggregated_data = []
    if combine_conditions:
        data_mouse_pairs = [(feature_data, mouseIDs_condition, stride_data),
                            (feature_data_compare, mouseIDs_compare, stride_data_compare)]
    else:
        data_mouse_pairs = [(feature_data, mouseIDs_condition, stride_data)]

    for data, mouseIDs, stride_d in data_mouse_pairs:
        for stride_number in stride_numbers:
            for mouse_id in mouseIDs:
                scaled_data_df = data.loc(axis=0)[stride_number, mouse_id]
                run_numbers, stepping_limbs, mask_phase1, mask_phase2 = utils.get_runs(scaled_data_df, stride_d,
                                                                                       mouse_id, stride_number, phase1,
                                                                                       phase2)
                selected_mask = mask_phase1 | mask_phase2
                selected_data = scaled_data_df.loc[selected_mask]
                reduced_data = selected_data[selected_features]
                aggregated_data.append(reduced_data)

    global_data = pd.concat(aggregated_data)
    if n_components > global_data.shape[1]:
        n_components = global_data.shape[1]
    pca, pcs, loadings_df = perform_pca(global_data, n_components=n_components)
    return pca, loadings_df




