"""Plots for feature clustering results."""
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns

from apa_analysis.Characterisation import Plotting_utils as pu

def plot_corr_matrix_sorted_manually(data_df, save_dir, filename):
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
        pu.add_vertical_brace_curly(ax, y0, y1, x=-0.5, xoffset=1, label=cluster_names.get(cl, f"Cluster {cl}"),
                                       k_r=k_r_adjusted, int_line_num=int_line_num, fontsize=fs)
        # Add a horizontal curly brace along the top.
        pu.add_horizontal_brace_curly(ax, x0, x1, y=-0.5, label=cluster_names.get(cl, f"Cluster {cl}"),
                                         k_r=k_r_adjusted*-1, int_line_num=int_line_num, fontsize=fs)

    plt.subplots_adjust(top=0.92)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    # save as png
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300, format="png")
    # save as svg
    plt.savefig(os.path.join(save_dir, f"{filename}.svg"), dpi=300, format="svg")
    plt.close()