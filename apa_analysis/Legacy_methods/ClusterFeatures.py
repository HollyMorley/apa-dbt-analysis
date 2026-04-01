import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
# from curlyBrace import curlyBrace
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import FastICA
from sklearn.model_selection import KFold
from scipy.signal import medfilt

from apa_analysis.Legacy_methods import utils_feature_reduction as utils
from apa_analysis.config import global_settings




def cross_validate_ica_reconstruction(feature_matrix, n_components_range=range(1, 11), n_splits=5):
    """
    For each candidate number of components, perform KFold CV on the feature matrix and compute the average
    reconstruction error (mean squared error) when reconstructing held-out data from the ICA decomposition.

    Parameters:
      - feature_matrix: DataFrame with features as rows and runs as columns.
      - n_components_range: Range of candidate numbers of ICA components.
      - n_splits: Number of folds for cross-validation.

    Returns:
      - avg_errors: dict mapping candidate n_components to their average reconstruction error.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    reconstruction_errors = {n: [] for n in n_components_range}

    for n in n_components_range:
        for train_idx, test_idx in kf.split(feature_matrix):
            train_data = feature_matrix.iloc[train_idx]
            test_data = feature_matrix.iloc[test_idx]

            # Fit ICA on training data.
            ica_cv = FastICA(n_components=n, random_state=42)
            ica_cv.fit(train_data)

            # Transform test data and reconstruct.
            S_test = ica_cv.transform(test_data)
            # Reconstruct the test data using the mixing matrix.
            reconstructed_test = np.dot(S_test, ica_cv.mixing_.T)

            # Compute reconstruction error.
            error = mean_squared_error(test_data, reconstructed_test)
            reconstruction_errors[n].append(error)

    avg_errors = {n: np.mean(errors) for n, errors in reconstruction_errors.items()}
    return avg_errors

# def cluster_features_with_ica_cv(global_fs_mouse_ids, stride_number, condition, exp, day,
#                                  stride_data, phase1, phase2,
#                                  n_components_range=range(1, 15),
#                                  n_splits=10,
#                                  save_file=None):
#     """
#     Build the global runs-by-features matrix, perform cross-validated ICA to determine the optimal
#     number of independent components (based on reconstruction error), and assign each feature to the component
#     where it has the highest absolute mixing coefficient.
#
#     Parameters:
#       - n_components_range: Candidate numbers of ICA components.
#       - n_splits: Number of folds for cross-validation.
#       - save_file: If provided, the resulting mapping is saved to this file.
#
#     Returns:
#       - cluster_mapping: dict mapping feature names to component labels.
#       - feature_matrix: The original feature matrix (features as rows).
#       - ica: The ICA object fit on the entire feature matrix with the optimal number of components.
#       - optimal_n: The selected number of independent components.
#       - avg_errors: Dict mapping candidate n_components to their average reconstruction error.
#     """
#     # Build the global feature matrix (features as rows, runs as columns)
#     feature_matrix = get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data,
#                                                phase1, phase2, smooth=True)
#
#     # Cross-validate reconstruction error for candidate numbers of components.
#     avg_errors = cross_validate_ica_reconstruction(feature_matrix, n_components_range, n_splits)
#
#     # Select the number of components with the lowest reconstruction error.
#     optimal_n = min(avg_errors, key=avg_errors.get)
#
#     print(f"ICA CV reconstruction errors: {avg_errors}")
#     print(f"Optimal number of ICA components selected: {optimal_n}")
#
#     # Perform ICA on the entire feature matrix using the optimal number of components.
#     ica = FastICA(n_components=optimal_n, max_iter=10000, tol=0.1, random_state=42)
#     ica.fit(feature_matrix)
#
#     # Retrieve the mixing matrix.
#     A_ = ica.mixing_
#
#     # Assign each feature to the component where it has the highest absolute mixing coefficient.
#     cluster_mapping = {}
#     for i, feature in enumerate(feature_matrix.index):
#         cluster_mapping[feature] = int(np.argmax(np.abs(A_[i])))
#
#     if save_file is not None:
#         os.makedirs(os.path.dirname(save_file), exist_ok=True)
#         joblib.dump(cluster_mapping, save_file)
#         print(f"Feature clustering (using ICA CV) done and saved to {save_file} with optimal_n={optimal_n}.")
#
#     return cluster_mapping, feature_matrix, ica, optimal_n, avg_errors


def cross_validate_pca_explained_variance(feature_matrix, n_components_range=range(1, 11), n_splits=5):
    """
    For each candidate number of components, perform KFold CV on the feature matrix (features as rows)
    and compute the average cumulative explained variance from the PCA fit on each training split.

    Returns a dict mapping each candidate n_components to its average cumulative explained variance.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    explained_var = {n: [] for n in n_components_range}

    for n in n_components_range:
        for train_index, _ in kf.split(feature_matrix):
            train_data = feature_matrix.iloc[train_index]
            # Fit PCA on the training fold.
            pca_cv = PCA(n_components=n, random_state=42)
            pca_cv.fit(train_data)
            explained_var[n].append(np.sum(pca_cv.explained_variance_ratio_))

    avg_explained = {n: np.mean(vals) for n, vals in explained_var.items()}
    return avg_explained


# def cluster_features_with_pca_cv(global_fs_mouse_ids, stride_number, condition, exp, day,
#                                  stride_data, phase1, phase2,
#                                  n_components_range=range(1, 15),
#                                  n_splits=10, variance_threshold=0.8,
#                                  save_file=None):
#     """
#     Build the global runs-by-features matrix, perform cross-validated PCA to determine the optimal
#     number of PCs (as the smallest number whose average cumulative explained variance exceeds a threshold),
#     and assign each feature to the PC (cluster) where it has the highest absolute loading.
#
#     Parameters:
#       - n_components_range: Candidate numbers of components to try.
#       - n_splits: Number of folds for cross-validation.
#       - variance_threshold: Minimum cumulative explained variance (e.g. 0.9 for 90%) required.
#       - save_file: If provided, the cluster mapping will be saved to this file.
#
#     Returns:
#       - cluster_mapping: dict mapping feature names to cluster labels.
#       - feature_matrix: The original feature matrix (features as rows).
#       - pca: The PCA object fit on the entire feature matrix using the optimal n_components.
#       - optimal_n: The selected number of principal components.
#       - avg_explained: Dict mapping candidate n_components to their average explained variance.
#     """
#     # Build the global feature matrix (features as rows, runs as columns)
#     feature_matrix = get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data,
#                                                phase1, phase2, smooth=True)
#
#     # Cross-validate to compute average cumulative explained variance for candidate n_components.
#     avg_explained = cross_validate_pca_explained_variance(feature_matrix, n_components_range, n_splits)
#
#     # Select the smallest n_components that meets or exceeds the threshold.
#     valid_candidates = [n for n, ev in avg_explained.items() if ev >= variance_threshold]
#     optimal_n = min(valid_candidates) if valid_candidates else max(n_components_range)
#
#     print(f"Cross-validation results (avg cumulative explained variance): {avg_explained}")
#     print(f"Optimal number of PCs selected: {optimal_n}")
#
#     # Perform PCA on the entire feature matrix using the optimal number of components.
#     pca = PCA(n_components=optimal_n, random_state=42)
#     pca.fit(feature_matrix)
#
#     # Retrieve the loadings (each row corresponds to a feature, columns to PCs)
#     loadings = pca.components_.T
#
#     # Assign each feature to the PC where it has the highest absolute loading.
#     cluster_mapping = {}
#     for i, feature in enumerate(feature_matrix.index):
#         cluster_mapping[feature] = int(np.argmax(np.abs(loadings[i])))
#
#     if save_file is not None:
#         os.makedirs(os.path.dirname(save_file), exist_ok=True)
#         joblib.dump(cluster_mapping, save_file)
#         print(f"Feature clustering (using PCA CV) done and saved to {save_file} using optimal_n={optimal_n}.")
#
#     return cluster_mapping, feature_matrix, pca, optimal_n, avg_explained

def find_feature_clusters(feature_data, feature_data_compare, condition_mouseIDs, compare_condition_mice, stride_number, stride_data, stride_data_compare, phase1, phase2, save_dir, combine_conditions, method='kmeans'):
    if method == 'kmeans':
        # find k
        optimal_k, avg_sil_scores = cross_validate_k_clusters_folds(feature_data,
                                                                    feature_data_compare,
                                                                    condition_mouseIDs,
                                                                    compare_condition_mice,
                                                                    stride_number,
                                                                    stride_data,
                                                                    stride_data_compare,
                                                                    phase1,
                                                                    phase2,
                                                                    combine_conditions=combine_conditions)
    cluster_save_file = os.path.join(save_dir, f'feature_clusters_{phase1}_vs_{phase2}_stride{stride_number}.pkl')
    # cluster features
    cluster_mapping, feature_matrix = cluster_features_run_space(feature_data,
                                                                 feature_data_compare,
                                                                 condition_mouseIDs,
                                                                 compare_condition_mice,
                                                                 stride_number,
                                                                 stride_data,
                                                                 stride_data_compare,
                                                                 phase1,
                                                                 phase2,
                                                                 n_clusters=optimal_k,
                                                                 save_file=cluster_save_file,
                                                                 method=method,
                                                                 combine_conditions=combine_conditions)
    return cluster_mapping, feature_matrix

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
                #data = utils.load_and_preprocess_data(mouse, sn, condition, exp, day)
                data = feature_data.loc(axis=0)[sn, mouse]
                run_numbers, _, mask_phase1, mask_phase2 = utils.get_runs(data, stride_data, mouse, sn, phase1, phase2)
                selected_mask = mask_phase1 | mask_phase2
                run_data = data.loc[selected_mask]
                if smooth:
                    run_data_smooth = medfilt(run_data, kernel_size=3)
                    run_data = pd.DataFrame(run_data_smooth, index=run_data.index, columns=run_data.columns)
                all_runs_data.append(run_data)
    else:
        for mouse in global_fs_mouse_ids:
            #data = utils.load_and_preprocess_data(mouse, stride_number, condition, exp, day)
            data = feature_data.loc(axis=0)[stride_number, mouse]
            # todo smooth features across runs
            # Get runs for the two phases; here we use phase1 and phase2 for example.
            run_numbers, _, mask_phase1, mask_phase2 = utils.get_runs(data, stride_data, mouse, stride_number, phase1,
                                                                      phase2)
            selected_mask = mask_phase1 | mask_phase2
            run_data = data.loc[selected_mask]
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

#
# def cross_validate_db_score_folds(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data, phase1,
#                                   phase2, k_range=range(2, 11), n_splits=10, n_init=10):
#     """
#     Build the global feature matrix and perform k-fold cross-validation to select the optimal number
#     of clusters using the Davies–Bouldin score.
#
#     For each fold, KMeans is fitted on the training subset, and the DB score is computed on the test subset.
#
#     Returns:
#       - avg_db_scores: dict mapping each k to its average Davies–Bouldin score across folds.
#     """
#     # Build the global feature matrix (features as rows, runs as columns)
#     feature_matrix = get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data,
#                                                phase1, phase2)
#
#     # Prepare KFold cross-validation over features (rows of feature_matrix)
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     db_scores = {k: [] for k in k_range}
#
#     for train_idx, test_idx in tqdm(list(kf.split(feature_matrix)), total=n_splits, desc="Folds"):
#         train_data = feature_matrix.iloc[train_idx]
#         test_data = feature_matrix.iloc[test_idx]
#
#         for k in k_range:
#             # Use built-in n_init for efficiency.
#             kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
#             kmeans.fit(train_data)
#             # Predict test labels.
#             test_labels = kmeans.predict(test_data)
#             # Compute the Davies–Bouldin score on the test data.
#             db_score = davies_bouldin_score(test_data, test_labels)
#             db_scores[k].append(db_score)
#
#     # Average DB scores across folds for each candidate k.
#     avg_db_scores = {k: np.mean(scores) for k, scores in db_scores.items()}
#     for k, score in avg_db_scores.items():
#         print(f"k={k}, Average Davies–Bouldin Score (CV): {score:.3f}")
#     return avg_db_scores

# def cross_validate_inertia_folds(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data, phase1,
#                                  phase2, k_range=range(2, 11), n_splits=10, n_init=10):
#     """
#     Build the global feature matrix and perform k-fold cross-validation to evaluate the average inertia
#     (within-cluster sum-of-squares) for each candidate number of clusters k.
#     Returns:
#       - avg_inertia: dict mapping each k to its average inertia across folds.
#     """
#     # Build the global feature matrix (features as rows, runs as columns)
#     feature_matrix = get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day, stride_data,
#                                                phase1, phase2)
#
#     # Prepare KFold cross-validation over features (rows of feature_matrix)
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     inertia_scores = {k: [] for k in k_range}
#
#     for train_idx, test_idx in tqdm(list(kf.split(feature_matrix)), total=n_splits, desc="Folds"):
#         train_data = feature_matrix.iloc[train_idx]
#         test_data = feature_matrix.iloc[test_idx]
#
#         for k in k_range:
#             # Use built-in n_init for efficiency.
#             kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
#             kmeans.fit(train_data)
#             # Compute inertia on the test set by assigning test_data to the nearest cluster center.
#             # (This is a simple approximation; you could also compute it on the training set.)
#             test_labels = kmeans.predict(test_data)
#             inertia = kmeans.inertia_  # inertia from the training set.
#             inertia_scores[k].append(inertia)
#
#     # Average inertia scores across folds for each candidate k.
#     avg_inertia = {k: np.mean(scores) for k, scores in inertia_scores.items()}
#     for k, inertia in avg_inertia.items():
#         print(f"k={k}, Average Inertia (CV): {inertia:.3f}")
#     return avg_inertia

def cross_validate_k_clusters_folds(feature_data, feature_data_compare, condition_mice, compare_condition_mice, stride_number, stride_data, stride_data_compare, phase1,
                                    phase2, combine_conditions,
                                    k_range=range(2, 11), n_splits=10, n_init=10
                                    ):
    """
    Build the global feature matrix and perform k-fold cross-validation to select the optimal number
    of clusters. In each fold, the clustering model is fit on the training subset of features and then
    applied to the test subset to compute the silhouette score.

    Parameters:
      - global_fs_mouse_ids: list of mouse IDs for global feature selection.
      - stride_number, condition, exp, day, stride_data, phase1, phase2: parameters used to load data.
      - k_range: range of candidate k values (number of clusters).
      - n_splits: number of folds in the KFold cross-validation.
      - n_init: number of random initializations per fold for stability.

    Returns:
      - optimal_k: the k value with the highest average silhouette score across folds.
      - avg_sil_scores: dict mapping each k to its average silhouette score.
    """
    # Build the global feature matrix (features as rows, runs as columns)
    feature_matrix_condition = get_global_feature_matrix(
        feature_data, condition_mice, stride_number, stride_data, phase1, phase2)

    if combine_conditions == True:
        # Get feature matrix for the compare condition
        feature_matrix_compare = get_global_feature_matrix(
            feature_data_compare, compare_condition_mice, stride_number, stride_data_compare, phase1, phase2)
        # Concatenate the two matrices along the rows (features)
        feature_matrix = pd.concat([feature_matrix_condition, feature_matrix_compare], axis=1)
    else:
        feature_matrix = feature_matrix_condition

    # Prepare KFold cross-validation over features (rows of feature_matrix)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Dictionary to store silhouette scores per candidate k for each fold.
    fold_scores = {k: [] for k in k_range}

    # Wrap the outer loop with tqdm to show fold progress.
    for train_idx, test_idx in tqdm(list(kf.split(feature_matrix)), total=n_splits, desc="Folds"):
        train_data = feature_matrix.iloc[train_idx]
        test_data = feature_matrix.iloc[test_idx]

        for k in k_range:
            # Use built-in n_init instead of manual loop.
            kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            kmeans.fit(train_data)
            test_labels = kmeans.predict(test_data)
            score = silhouette_score(test_data, test_labels)
            fold_scores[k].append(score)

    # Average silhouette scores across folds for each candidate k.
    avg_sil_scores = {k: np.mean(scores_list) for k, scores_list in fold_scores.items()}
    for k, score in avg_sil_scores.items():
        print(f"k={k}, Average Silhouette Score (CV): {score:.3f}")

    # Select the k with the highest average silhouette score.
    optimal_k = max(avg_sil_scores, key=avg_sil_scores.get)
    print(f"Optimal k determined to be: {optimal_k}")
    return optimal_k, avg_sil_scores

# def cross_validate_k_clusters_folds_pca(global_fs_mouse_ids, stride_number, condition, exp, day,
#                                         stride_data, phase1, phase2,
#                                         n_components=10,
#                                         k_range=range(2, 11),
#                                         n_splits=10, n_init=10):
#     """
#     Build the global feature matrix, apply PCA to reduce its dimensionality, and then perform
#     k-fold cross-validation to select the optimal number of clusters using the silhouette score.
#
#     Parameters:
#       - global_fs_mouse_ids, stride_number, condition, exp, day, stride_data, phase1, phase2:
#           parameters used to load data.
#       - n_components: number of PCA components to retain.
#       - k_range: range of candidate k values.
#       - n_splits: number of folds in the KFold cross-validation.
#       - n_init: number of initializations for KMeans.
#
#     Returns:
#       - optimal_k: the k value with the highest average silhouette score across folds.
#       - avg_sil_scores: dict mapping each k to its average silhouette score.
#     """
#     # Build the global feature matrix (rows = features, columns = runs)
#     feature_matrix = get_global_feature_matrix(global_fs_mouse_ids, stride_number, condition, exp, day,
#                                                stride_data, phase1, phase2)
#     # Apply PCA to reduce dimensionality
#     pca = PCA(n_components=n_components)
#     X_reduced = pca.fit_transform(feature_matrix.values)
#     # Reconstruct a DataFrame with the same feature index.
#     X_reduced_df = pd.DataFrame(X_reduced, index=feature_matrix.index)
#
#     # Prepare KFold cross-validation over the features (rows of X_reduced_df)
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     fold_scores = {k: [] for k in k_range}
#
#     # Wrap the outer loop with tqdm to show progress.
#     for train_idx, test_idx in tqdm(list(kf.split(X_reduced_df)), total=n_splits, desc="PCA Folds"):
#         train_data = X_reduced_df.iloc[train_idx]
#         test_data = X_reduced_df.iloc[test_idx]
#
#         for k in k_range:
#             kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
#             kmeans.fit(train_data)
#             test_labels = kmeans.predict(test_data)
#             score = silhouette_score(test_data, test_labels)
#             fold_scores[k].append(score)
#
#     # Average the silhouette scores over folds for each candidate k.
#     avg_sil_scores = {k: np.mean(scores) for k, scores in fold_scores.items()}
#     for k, score in avg_sil_scores.items():
#         print(f"k={k}, Average Silhouette Score (CV, PCA): {score:.3f}")
#
#     # Select the k with the highest average silhouette score.
#     optimal_k = max(avg_sil_scores, key=avg_sil_scores.get)
#     print(f"Optimal k based on silhouette score (after PCA) determined to be: {optimal_k}")
#     return optimal_k, avg_sil_scores


def cluster_features_run_space(feature_data, feature_data_compare, condition_mouseIDs, compare_condition_mouseIDs, stride_number, stride_data, stride_data_compare, phase1, phase2,
                               n_clusters, save_file, method, combine_conditions):
    """
    Build the global runs-by-features matrix, transpose it so that rows are features,
    cluster the features using k-means with n_clusters, and save the mapping.
    Returns:
      cluster_mapping: dict mapping feature names to cluster labels.
    """
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    # Get feature matrices for both conditions:
    feature_matrix_condition = get_global_feature_matrix(
        feature_data, condition_mouseIDs, stride_number, stride_data,
        phase1, phase2, smooth=True
    )
    if combine_conditions == True:
        feature_matrix_compare = get_global_feature_matrix(
            feature_data_compare, compare_condition_mouseIDs, stride_number, stride_data_compare,
            phase1, phase2, smooth=True
        )
        # Concatenate the two matrices along the rows (features)
        feature_matrix = pd.concat([feature_matrix_condition, feature_matrix_compare], axis=1)
    else:
        feature_matrix = feature_matrix_condition

    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(feature_matrix)
        # Mapping: feature -> cluster label
        cluster_mapping = dict(zip(feature_matrix.index, kmeans.labels_))

    # joblib.dump(cluster_mapping, save_file)
    # print(f"Feature clustering done and saved to {save_file} using k={n_clusters}.")
    # Adjust save filename to reflect the "all" mode.
    save_file_adjusted = os.path.join(os.path.dirname(save_file),
                                      f'feature_clusters_{phase1}_vs_{phase2}_stride{"all" if stride_number == "all" else stride_number}.pkl')
    joblib.dump(cluster_mapping, save_file_adjusted)
    print(f"Feature clustering done and saved to {save_file_adjusted} using k={n_clusters}.")

    return cluster_mapping, feature_matrix


def plot_feature_clustering(feature_matrix, cluster_mapping, p1, p2, s, save_dir):
    """
    Projects the features into 2D using PCA and plots them colored by cluster.
    Each point represents a feature (from the rows of feature_matrix).
    """
    # Perform PCA on the feature matrix (rows are features)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(feature_matrix.values)

    # Get cluster label for each feature
    features = feature_matrix.index
    clusters = [cluster_mapping.get(feat, -1) for feat in features]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap="gist_rainbow", s=50)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Feature Clustering (PCA Projection)")
    plt.colorbar(scatter, label="Cluster")

    # Optionally annotate features (can be crowded if many features)
    # for i, feat in enumerate(features):
    #     plt.annotate(feat, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=6, alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"feature_clustering_{p1}_vs_{p2}_stride{s}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved feature clustering plot to {save_path}")


from mpl_toolkits.mplot3d import Axes3D  # explicitly import Axes3D


def plot_feature_clustering_3d(feature_matrix, cluster_mapping, p1, p2, s, save_dir):
    """
    Projects the features into 3D using PCA and plots them colored by cluster.
    Each point represents a feature (from the rows of feature_matrix).
    """
    # Perform PCA with 3 components
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(feature_matrix.values)

    # Get cluster labels for each feature; default to -1 if not mapped.
    features = feature_matrix.index
    clusters = [cluster_mapping.get(feat, -1) for feat in features]

    # Create a new figure and explicitly add a 3D Axes using Axes3D.
    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig)  # explicitly create a 3D axes instance

    # Scatter plot in 3D
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                         c=clusters, cmap="gist_rainbow", s=50)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title("3D Feature Clustering (PCA Projection)")

    # Add a colorbar
    fig.colorbar(scatter, ax=ax, label="Cluster")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"feature_clustering_3D_{p1}_vs_{p2}_stride{s}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved 3D feature clustering plot to {save_path}")


def plot_feature_clusters_chart(cluster_mapping, sorted_features, p1, p2, s, save_dir):
    """
    Creates and saves a chart that arranges feature names by their assigned cluster.
    Each column represents one cluster.
    """
    # Build clusters in the order that features appear in sorted_features.
    clusters = {}
    for feat in sorted_features:
        cl = cluster_mapping.get(feat, -1)
        clusters.setdefault(cl, []).append(feat)

    # The keys in clusters will now be in the order they first appear in sorted_features.
    cluster_order = list(clusters.keys())
    k = len(cluster_order)
    # Determine the maximum number of features in any cluster.
    max_features = max(len(features) for features in clusters.values())

    # Build table data with one column per cluster (in cluster_order).
    table_data = []
    for i in range(max_features):
        row = []
        for cl in cluster_order:
            feats = clusters[cl]
            if i < len(feats):
                row.append(feats[i])
            else:
                row.append("")
        table_data.append(row)

    # Create the figure and table.
    fig, ax = plt.subplots(figsize=(k * 8, max_features * 0.5 + 1))
    ax.axis("tight")
    ax.axis("off")
    col_labels = [f"Cluster {cl}" for cl in cluster_order]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.tight_layout()
    save_path = os.path.join(save_dir, f"feature_clusters_chart_{p1}_vs_{p2}_stride{s}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved feature clusters chart to {save_path}")


# def add_vertical_brace_curly(ax, y0, y1, x, label=None, k_r=0.1, int_line_num=2, fontdict=None, **kwargs):
#     """
#     Add a vertical curly brace using the curlyBrace package.
#     The brace is drawn at the given x coordinate.
#     """
#     fig = ax.figure
#     p1 = [x, y0]
#     p2 = [x, y1]
#     # Do not pass the label here.12
#     brace = curlyBrace(fig, ax, p1, p2, k_r=k_r, bool_auto=True, str_text='',
#                        int_line_num=int_line_num, fontdict=fontdict or {}, clip_on=False, color='black', **kwargs)
#     if label:
#         y_center = (y0 + y1) / 2.0
#         # Place the label to the left of the brace.
#         ax.text(x - 14, y_center, label,
#                 ha="right", va="center", fontsize=12, fontweight="normal", color='black', clip_on=False)
#
# def add_horizontal_brace_curly(ax, x0, x1, y, label=None, k_r=0.1, int_line_num=2, fontdict=None, **kwargs):
#     """
#     Add a horizontal curly brace using the curlyBrace package.
#     The brace is drawn at the given y coordinate.
#     """
#     fig = ax.figure
#     # Swap p1 and p2 so that the brace opens toward the plot.
#     p1 = [x1, y]
#     p2 = [x0, y]
#     brace = curlyBrace(fig, ax, p1, p2, k_r=k_r, bool_auto=True, str_text='',
#                        int_line_num=int_line_num, fontdict=fontdict or {}, clip_on=False, color='black', **kwargs)
#     if label:
#         x_center = (x0 + x1) / 2.0
#         # Adjust the offset so the label appears above the brace.
#         ax.text(x_center, y - 12 , label,
#                 ha="center", va="bottom", fontsize=12, fontweight="normal", color='black', clip_on=False)

def plot_corr_matrix_sorted_by_cluster(data_df, sorted_features, cluster_mapping, save_dir, filename="corr_matrix_by_cluster.png"):
    """
    Plot the correlation matrix of features (from data_df) sorted by their cluster assignment.
    Instead of labeling every feature, this version adds curly-brace annotations (via curlyBrace)
    outside the top and left edges to indicate the extent of each cluster.

    Parameters:
      data_df (pd.DataFrame): DataFrame with samples as rows and features as columns.
      cluster_mapping (dict): Mapping from feature names to cluster IDs.
      save_dir (str): Directory to save the plot.
      filename (str): Name of the output plot file.
    """
    # Compute the correlation matrix (assuming features are columns)
    corr = data_df.T.corr()

    # Sort features by cluster assignment (defaulting to -1 if missing)
    #sorted_features = sorted(corr.columns, key=lambda f: cluster_mapping.get(f, -1))
    corr_sorted = corr.loc[sorted_features, sorted_features]

    # Compute cluster boundaries based on sorted order.
    cluster_boundaries = {}
    for idx, feat in enumerate(sorted_features):
        cl = cluster_mapping.get(feat, -1)
        if cl not in cluster_boundaries:
            cluster_boundaries[cl] = {"start": idx, "end": idx}
        else:
            cluster_boundaries[cl]["end"] = idx

    # Create the heatmap without individual feature tick labels.
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr_sorted, annot=False, fmt=".2f", cmap="coolwarm",
                     xticklabels=False, yticklabels=False, vmin=-1, vmax=1, center=0)
    ax.set_title("Correlation Matrix Sorted by Cluster", pad=60)
    # ensure legend spans 0-1
   # ax.collections[0].colorbar.set_ticks([0, 0.5, 1])

    # For each cluster, adjust boundaries by 0.5 (to align with cell edges).
    for cl, bounds in cluster_boundaries.items():
        # Define boundaries in data coordinates.
        x0, x1 = bounds["start"] - 0.5, bounds["end"] + 0.5
        y0, y1 = bounds["start"] - 0.5, bounds["end"] + 0.5
        # Add a vertical curly brace along the left side.
        utils.add_vertical_brace_curly(ax, y0, y1, x=-0.5, label=f"Cluster {cl}", k_r=0.1)
        # Add a horizontal curly brace along the top.
        utils.add_horizontal_brace_curly(ax, x0, x1, xoffset=14, y=-0.5, label=f"Cluster {cl}", k_r=0.1)

    plt.subplots_adjust(top=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

def plot_corr_matrix_sorted_manually(data_df, save_dir, filename="corr_matrix_by_manualcluster.png"):
    """
    Plot the correlation matrix of features (from data_df) sorted by their cluster assignment.
    Instead of labeling every feature, this version adds curly-brace annotations (via curlyBrace)
    outside the top and left edges to indicate the extent of each cluster.

    Parameters:
      data_df (pd.DataFrame): DataFrame with samples as rows and features as columns.
      cluster_mapping (dict): Mapping from feature names to cluster IDs.
      save_dir (str): Directory to save the plot.
      filename (str): Name of the output plot file.
    """
    from helpers.config import manual_clusters

    cluster_names = {v: k for k, v in manual_clusters['cluster_values'].items()}

    sorted_features = list(manual_clusters['cluster_mapping'].keys())

    # Compute the correlation matrix (assuming features are columns)
    corr = data_df.T.corr()

    # Sort features by cluster assignment (defaulting to -1 if missing)
    #sorted_features = sorted(corr.columns, key=lambda f: cluster_mapping.get(f, -1))
    corr_sorted = corr.loc[sorted_features, sorted_features]

    # Compute cluster boundaries based on sorted order.
    cluster_boundaries = {}
    for idx, feat in enumerate(sorted_features):
        cl = manual_clusters['cluster_mapping'].get(feat, -1)
        if cl not in cluster_boundaries:
            cluster_boundaries[cl] = {"start": idx, "end": idx}
        else:
            cluster_boundaries[cl]["end"] = idx

    # Create the heatmap without individual feature tick labels.
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr_sorted, annot=False, fmt=".2f", cmap="coolwarm",
                     xticklabels=False, yticklabels=False, vmin=-1, vmax=1, center=0)
    ax.set_title("Correlation Matrix Sorted by Cluster", pad=60)
    # ensure legend spans 0-1
   # ax.collections[0].colorbar.set_ticks([0, 0.5, 1])

    # For each cluster, adjust boundaries by 0.5 (to align with cell edges).
    for i, (cl, bounds) in enumerate(cluster_boundaries.items()):
        # Define boundaries in data coordinates.
        x0, x1 = bounds["start"], bounds["end"]
        y0, y1 = bounds["start"], bounds["end"]

        k_r = 0.1
        span = abs(y1 - y0)
        desired_depth = 0.1  # or any value that gives you the uniform look you want
        k_r_adjusted = desired_depth / span if span != 0 else k_r

        # Alternate the int_line_num value for every other cluster:
        base_line_num = 2
        int_line_num = base_line_num + 4 if i % 2 else base_line_num

        fs = 6

        # Add a vertical curly brace along the left side.
        utils.add_vertical_brace_curly(ax, y0, y1, x=-0.5, xoffset=1, label=cluster_names.get(cl, f"Cluster {cl}"),
                                       k_r=k_r_adjusted, int_line_num=int_line_num, fontsize=fs)
        # Add a horizontal curly brace along the top.
        utils.add_horizontal_brace_curly(ax, x0, x1, y=-0.5, label=cluster_names.get(cl, f"Cluster {cl}"),
                                         k_r=k_r_adjusted*-1, int_line_num=int_line_num, fontsize=fs)

    plt.subplots_adjust(top=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()


