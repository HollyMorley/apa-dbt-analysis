import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def compute_mouse_similarity(cluster_loadings_dict):
    """
    Given a dictionary mapping mouse IDs to their cluster loading dictionaries,
    compute a similarity matrix (using cosine similarity) between mice.

    Parameters:
      - cluster_loadings_dict: dict, where keys are mouse IDs and values are dicts
          mapping cluster IDs to loadings.

    Returns:
      - sim_df: A DataFrame with mouse IDs as both rows and columns containing
          the pairwise cosine similarity between loading vectors.
      - loading_matrix: A DataFrame where each row corresponds to a mouse and
          columns are clusters (with missing clusters filled with 0).
    """
    # Determine the union of all clusters across mice.
    all_clusters = set()
    for loadings in cluster_loadings_dict.values():
        all_clusters.update(loadings.keys())
    all_clusters = sorted(all_clusters)

    # Build a loading matrix: each row is a mouse and each column is a cluster.
    loading_data = {}
    for mouse, loadings in cluster_loadings_dict.items():
        loading_vector = [loadings.get(cluster, 0) for cluster in all_clusters]
        loading_data[mouse] = loading_vector
    loading_matrix = pd.DataFrame(loading_data, index=all_clusters).T

    # Compute cosine similarity between mice (rows).
    sim_matrix = cosine_similarity(loading_matrix.values)
    sim_df = pd.DataFrame(sim_matrix, index=loading_matrix.index, columns=loading_matrix.index)
    return sim_df, loading_matrix

def get_loading_matrix_from_nested(aggregated_cluster_loadings, key):
    """
    Given an aggregated_cluster_loadings dictionary that maps a key (e.g., (phase1, phase2, stride))
    to a dictionary mapping mouse IDs to their cluster loading dictionaries,
    this function returns a DataFrame for that specific key.

    The DataFrame rows correspond to mouse IDs, and the columns to cluster IDs.
    Missing cluster values are filled with 0.

    Parameters:
      - aggregated_cluster_loadings: dict, with keys like (phase1, phase2, stride)
        and values as dictionaries: {mouse_id: {cluster: loading, ...}, ...}
      - key: a tuple (phase1, phase2, stride) for which to extract the loading matrix.

    Returns:
      - loading_df: DataFrame with mouse IDs as rows and clusters as columns.
    """
    if key not in aggregated_cluster_loadings:
        raise ValueError(f"Key {key} not found in aggregated_cluster_loadings.")
    mouse_dict = aggregated_cluster_loadings[key]

    # Determine the union of all clusters present across all mice.
    all_clusters = set()
    for loadings in mouse_dict.values():
        all_clusters.update(loadings.keys())
    sorted_clusters = sorted(all_clusters)

    # Build the loading matrix.
    data = {}
    for mouse, cl_dict in mouse_dict.items():
        data[mouse] = [cl_dict.get(cluster, 0) for cluster in sorted_clusters]

    loading_df = pd.DataFrame.from_dict(data, orient='index', columns=sorted_clusters)
    return loading_df


def pool_mice_by_similarity(sim_df, threshold=0.8):
    """
    Given a similarity matrix DataFrame (sim_df) with mouse IDs as both rows and columns,
    create groups (connected components) of mice with similarity above the threshold.

    Parameters:
      - sim_df: DataFrame where sim_df.loc[i, j] is the cosine similarity between mouse i and mouse j.
      - threshold: similarity threshold for pooling (default 0.8)

    Returns:
      - groups: dict where keys are group numbers (e.g., 1, 2, ...) and values are lists of mouse IDs in that group.
    """
    # Create an undirected graph with each mouse as a node.
    G = nx.Graph()
    for mouse in sim_df.index:
        G.add_node(mouse)

    # Add an edge between two mice if their similarity is above or equal to the threshold.
    for i in sim_df.index:
        for j in sim_df.columns:
            if i != j and sim_df.loc[i, j] >= threshold:
                G.add_edge(i, j)

    # Find connected components in the graph.
    components = list(nx.connected_components(G))

    # Assign a numeric group id to each component.
    groups = {}
    for idx, comp in enumerate(components, start=1):
        groups[idx] = list(comp)

    return groups

def plot_similarity_matrix_threshold(sim_df, threshold=0.5, title="Mouse Loading Similarity (values > 0.5)", save_file=None):
    # Create a copy and mask values below or equal to the threshold.
    masked_sim_df = sim_df.copy()
    masked_sim_df[masked_sim_df <= threshold] = np.nan

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(masked_sim_df, annot=True, cmap="viridis", fmt=".2f",
                     mask=masked_sim_df.isna(), cbar_kws={'label': 'Cosine Similarity'})
    plt.title(title)
    plt.xlabel("Mouse ID")
    plt.ylabel("Mouse ID")
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()

