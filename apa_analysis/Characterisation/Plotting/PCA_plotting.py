"""Plots for PCA results, component loadings, and variance explained."""
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from scipy.stats import wilcoxon


from helpers.config import *
from apa_analysis.config import (global_settings)
from apa_analysis.Characterisation import General_utils as gu
from apa_analysis.Characterisation import Plotting_utils as pu

def plot_scree(pca, p1, p2, stride, condition, save_path, fs=7):
    """
    Plot and save the scree plot.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'o-', markersize=2, linewidth=1, color='blue', label='Individual Explained Variance')
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, 's--', markersize=2, linewidth=1, color='red', label='Cumulative Explained Variance')
    ax.set_title(stride, fontsize=fs)
    ax.set_xlabel('Principal Component', fontsize=fs)
    ax.set_ylabel('Explained Variance Ratio', fontsize=fs)
    # xtick range with every 10th label
    ax.set_xlim(0, len(pca.explained_variance_ratio_) + 1)
    ax.set_xticks(np.arange(0, len(pca.explained_variance_ratio_) + 1, 10))
    ax.set_xticklabels(np.arange(0, len(pca.explained_variance_ratio_) + 1, 10), fontsize=fs)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(axis='x', which='major', bottom=True, top=False, length=2, width=1)
    ax.tick_params(axis='x', which='minor', bottom=True, top=False, length=1, width=1)
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    ax.set_yticklabels(np.arange(0, 1.1, 0.25), fontsize=fs)
    ax.legend(loc='best')
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"Scree_Plot_{p1}_{p2}_{stride}_{condition}.png"), dpi=300)
    plt.savefig(os.path.join(save_path, f"Scree_Plot_{p1}_{p2}_{stride}_{condition}.svg"), dpi=300)
    plt.close()


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
    # print(f"Explained variance by PC1: {explained_variance[0] * 100:.2f}%")
    # print(f"Explained variance by PC2: {explained_variance[1] * 100:.2f}%")
    # if pca.n_components_ >= 3:
    #     print(f"Explained variance by PC3: {explained_variance[2] * 100:.2f}%")

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
    plt.savefig(os.path.join(save_path, f"PCA_2D_Mouse_{mouse_id}_{p1}vs{p2}_{stride}_{condition_label}.svg"), dpi=300)
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
        plt.savefig(os.path.join(save_path, f"PCA_3D_Mouse_{mouse_id}_{p1}vs{p2}_{stride}_{condition_label}.svg"), dpi=300)
        plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def pca_plot_feature_loadings(pca_data, phases, save_path, fs=7):
    if len(pca_data) == 1 and pca_data[0].phase[0] == phases[0] and pca_data[0].phase[1] == phases[1]:
        pca_loadings = pca_data[0].pca_loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()
    else:
        raise ValueError("Not expecting more PCA data than for APA2 and Wash2 now!")

    # build display names
    display_names = [short_names.get(f, f) for f in pca_loadings.index]

    # build heatmap DataFrame: rows=PCs, columns=features
    heatmap_df = pca_loadings.copy()
    heatmap_df.index = pca_loadings.index  # original feature keys
    heatmap_df.columns = [f"PC{idx + 1}" for idx in range(heatmap_df.shape[1])]
    heatmap_df.columns.name = "Principal Component"
    heatmap_df.index = display_names     # pretty feature labels
    heatmap_df = heatmap_df.T            # now rows=PCs, cols=features

    # --- Raw loadings plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_df,
        cmap='coolwarm',
        cbar_kws={'label': 'Loading'},
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )
    ax.set_title(f'PCA Feature Loadings: {phases[0]} vs {phases[1]}', fontsize=fs)
    ax.set_xlabel('Features', fontsize=fs)
    ax.set_ylabel('Principal Component', fontsize=fs)
    ax.tick_params(axis='x', rotation=90, labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        fn = f'PCA_feature_Loadings_{phases[0]}vs{phases[1]}_raw.{ext}'
        fig.savefig(os.path.join(save_path, fn), dpi=300)
    plt.close(fig)

    # prepare a colormap corresponding to the positive half of the original coolwarm
    full_cmap = plt.cm.get_cmap('coolwarm', 256)
    half = np.linspace(0.5, 1.0, 128)
    pos_cmap = ListedColormap(full_cmap(half))

    # compute maximum absolute loading for scaling
    max_abs = heatmap_df.abs().values.max()

    # --- Absolute loadings plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_df.abs(),
        cmap=pos_cmap,
        vmin=0,
        vmax=max_abs,
        cbar_kws={'label': 'Absolute Loading'},
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )
    ax.set_title(f'PCA Absolute Feature Loadings: {phases[0]} vs {phases[1]}', fontsize=fs)
    ax.set_xlabel('Features', fontsize=fs)
    ax.set_ylabel('Principal Component', fontsize=fs)
    ax.tick_params(axis='x', rotation=90, labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        fn = f'PCA_feature_Loadings_{phases[0]}vs{phases[1]}_absolute.{ext}'
        fig.savefig(os.path.join(save_path, fn), dpi=300)
    plt.close(fig)



def plot_top_features_per_PC(pca_data, feature_data, feature_data_notscaled, phases, stride_numbers, condition, save_path, n_top_features=5, fs=7, feature_data_LH=None):
    """
    Find the top features which load onto each principal component.
    """
    if len(pca_data) == 1 and pca_data[0].phase[0] == phases[0] and pca_data[0].phase[1] == phases[1]:
        pca_loadings = pca_data[0].pca_loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()
    else:
        raise ValueError("Not expecting more PCA data than for APA2 and Wash2 now!")

    top_features = {}
    for pc in pca_loadings.columns:
        top_features[pc] = pca_loadings.loc(axis=1)[pc].abs().nlargest(n_top_features).index.tolist()

    for s in stride_numbers:
        feats = feature_data.loc(axis=0)[s]
        feats_raw = feature_data_notscaled.loc(axis=0)[s]
        feats_LH = feature_data_LH.loc(axis=0)[s] if feature_data_LH is not None else None

        for pc in pca_loadings.columns:
            top_feats_pc = top_features[pc]
            top_feats_loadings = pca_loadings.loc(axis=1)[pc].loc(axis=0)[top_feats_pc]
            top_feats_data = feats.loc(axis=1)[top_feats_pc]
            top_feats_data_LH = feats_LH.loc(axis=1)[top_feats_pc] if feature_data_LH is not None else None
            top_feats_display_names = [short_names.get(f, f) for f in top_feats_pc]

            mask_p1, mask_p2 = gu.get_mask_p1_p2(top_feats_data, phases[0], phases[1])
            feats_p1 = top_feats_data.loc(axis=0)[mask_p1]
            feats_p2 = top_feats_data.loc(axis=0)[mask_p2]
            # feats_raw_p1 = feats_raw.loc(axis=0)[mask_p1]
            # feats_raw_p2 = feats_raw.loc(axis=0)[mask_p2]
            if feature_data_LH is not None:
                mask_p1_LH, mask_p2_LH = gu.get_mask_p1_p2(top_feats_data_LH, phases[0], phases[1])
                feats_p1_LH = top_feats_data_LH.loc(axis=0)[mask_p1_LH]
                feats_p2_LH = top_feats_data_LH.loc(axis=0)[mask_p2_LH]
            else:
                feats_p1_LH = None
                feats_p2_LH = None

            plot_top_feat_descriptives(feats_p1, feats_p2, top_feats_pc, top_feats_loadings, pc, phases, s,
                                       top_feats_display_names, save_path, fs=fs, feats_p1_LH=feats_p1_LH, feats_p2_LH=feats_p2_LH)
    return top_features

            # # Plot the raw features
            # common_x = np.arange(160)
            # fig, axs = plt.subplots(n_top_features, 1, figsize=(4, 7))
            # for i, feat in enumerate(top_feats_pc):
            #     mice_feats = np.zeros((len(condition_specific_settings[condition]['global_fs_mouse_ids']), len(common_x)))
            #     for midx, mouse_id in enumerate(condition_specific_settings[condition]['global_fs_mouse_ids']):
            #         interpolated_data = np.interp(common_x, feats.loc(axis=0)[mouse_id].loc(axis=1)[feat].index,
            #                                         feats.loc(axis=0)[mouse_id].loc(axis=1)[feat].values)
            #         smoothed_data = median_filter(interpolated_data, size=11)
            #
            #         ms = pu.get_marker_style_mice(mouse_id)
            #         axs[i].plot(common_x, smoothed_data, label=mouse_id, alpha=0.3, color='grey', markersize=3, zorder=10, marker=ms, linewidth=0.5)
            #
            #         mice_feats[midx] = smoothed_data
            #     # find the median across mice
            #     median_feats = np.median(mice_feats, axis=0)
            #     axs[i].plot(common_x, median_feats, alpha=0.7, color='black', zorder=10, linewidth=1)

def plot_top_feat_descriptives_3way(feats_list, top_feats_pc, top_feats_loadings, pc, phases, s, top_feats_display_names, save_path, fs=7, conditions=False):
    assert len(feats_list) == len(phases), "Each phase must have a corresponding feature dataframe"

    # Compute per-mouse means
    feats_permouse_medians = [f.groupby(level=0).mean() for f in feats_list]
    shared_mice = list(set.intersection(*[set(f.index) for f in feats_permouse_medians]))
    feats_permouse_medians = [f.loc[shared_mice] for f in feats_permouse_medians]

    data_per_phase = [[f[feat].values for feat in top_feats_pc] for f in feats_permouse_medians]

    # Get phase colors
    if not conditions:
        colors = [pu.get_color_phase(ph) for ph in phases]
    else:
        colors = [pu.get_color_speedpair(ph) for ph in phases]
    dark_colors = [pu.darken_color(c, 0.7) for c in colors]

    boxprops_list = [dict(facecolor=c, color=c) for c in colors]
    medianprops_list = [dict(color=dc, linewidth=2) for dc in dark_colors]
    whiskerprops_list = [dict(color=dc, linewidth=1.5, linestyle='-') for dc in dark_colors]

    x = np.arange(len(top_feats_pc))
    width = 0.2
    bar_multiple = 0.6
    shifts = np.linspace(-width, width, len(phases))
    positions_list = [x + shift for shift in shifts]

    fig, axs = plt.subplots(4, 1, figsize=(6, 10))

    ### Subplot 0: Loadings
    axs[0].bar(x, top_feats_loadings, width * bar_multiple, alpha=0.7, color='k')
    axs[0].set_ylabel('Feature Loadings', fontsize=fs)
    axs[0].set_ylim(-0.6, 0.6)
    axs[0].set_yticks(np.arange(-0.5, 0.6, 0.5))
    axs[0].set_xticks([])

    ### Subplot 1: Z-scored Feature Values
    for pos, data, boxprops, medprops, whiskprops in zip(positions_list, data_per_phase, boxprops_list, medianprops_list, whiskerprops_list):
        axs[2].boxplot(data, positions=pos, widths=width * bar_multiple,
                      patch_artist=True, boxprops=boxprops,
                      medianprops=medprops, whiskerprops=whiskprops,
                      showcaps=False, showfliers=False)

    for midx in shared_mice:
        vals = [f.loc[midx][top_feats_pc].values for f in feats_permouse_medians]
        axs[2].plot(positions_list, vals, 'o-', alpha=0.3, color='grey', markersize=3, zorder=10)

    axs[2].set_ylabel('Z-scored Feature', fontsize=fs)
    axs[2].set_xticklabels([])
    # axs[2].set_xticks(x)
    # axs[2].set_xticklabels(top_feats_display_names, fontsize=fs, rotation=90)


    handles = [mpatches.Patch(color=c, label=ph) for c, ph in zip(colors, phases)]
    axs[2].legend(handles=handles, fontsize=fs, loc='upper right', bbox_to_anchor=(1.2, 1), title='Phase', title_fontsize=fs)

    ### Subplot 2: PC Projection
    weighted_features = [[feature * loading for feature, loading in zip(data, top_feats_loadings.values)] for data in data_per_phase]
    for pos, data, boxprops, medprops, whiskprops in zip(positions_list, weighted_features, boxprops_list, medianprops_list, whiskerprops_list):
        axs[1].boxplot(data, positions=pos, widths=width * bar_multiple,
                      patch_artist=True, boxprops=boxprops,
                      medianprops=medprops, whiskerprops=whiskprops,
                      showcaps=False, showfliers=False)

    # Convert weighted features into DataFrames for consistent mouse indexing
    weighted_dfs = [
        pd.DataFrame(np.column_stack(data), index=shared_mice, columns=top_feats_pc)
        for data in weighted_features
    ]

    for midx in shared_mice:
        vals = [df.loc[midx].values for df in weighted_dfs]
        axs[1].plot(positions_list, vals, 'o-', alpha=0.3, color='grey', markersize=3, zorder=10)

    axs[1].set_ylabel('PC Projection', fontsize=fs)
    axs[1].set_ylim(-0.6, 0.6)
    axs[1].set_yticks(np.arange(-0.5, 0.6, 0.5))
    axs[1].set_yticklabels(np.arange(-0.5, 0.6, 0.5), fontsize=fs)
    axs[1].set_xticklabels([])

    ### Subplot 3: Relative Feature Differences (LowHigh - others)
    base = feats_permouse_medians[0]  # Assume LowHigh is first
    diffs = [base[feat] - other[feat] for other in feats_permouse_medians[1:] for feat in top_feats_pc]
    diffs = np.reshape(diffs, (len(feats_permouse_medians) - 1, len(top_feats_pc), -1))
    labels = [f'{phases[0]}-{ph}' for ph in phases[1:]]
    for i, d in enumerate(diffs):
        axs[3].boxplot(d.T, positions=x + (i - 0.5) * 0.2, widths=0.15,
                       patch_artist=True, boxprops=dict(facecolor='white', edgecolor=dark_colors[i + 1]),
                       medianprops=dict(color=dark_colors[i + 1]), whiskerprops=dict(color=dark_colors[i + 1]))

    axs[3].set_ylabel('Relative Feature Diff.', fontsize=fs)
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(top_feats_display_names, fontsize=fs, rotation=90)
    axs[3].legend(
        [mpatches.Patch(edgecolor=dark_colors[i + 1], facecolor='white', label=labels[i]) for i in range(len(labels))],
        labels, fontsize=fs, loc='upper right', title='Diffs rel. to LowHigh')

    for ax in axs:
        if ax != axs[0] and ax != axs[1]:
            ax.set_ylim(-1.4, 1.4)
            ax.set_yticks(np.arange(-1, 1.1, 1))
            ax.set_yticklabels(np.arange(-1, 1.1, 1), fontsize=fs)
        ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.4)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(False)
        ax.set_xlim(-0.5, len(top_feats_pc) - 0.5)
        ax.tick_params(axis='y', which='both', left=True, labelsize=fs)
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.tick_params(axis='y', which='minor', length=4, width=1, color='k')
        if i != 3:
            ax.tick_params(axis='x', which='both', bottom=False, top=False)

    plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.21, hspace=0.1)
    plt.suptitle(pc, fontsize=fs)

    plt.savefig(os.path.join(save_path, f'PCA_top_features_{"vs".join(phases)}_stride{s}_{pc}.png'), dpi=300)
    plt.savefig(os.path.join(save_path, f'PCA_top_features_{"vs".join(phases)}_stride{s}_{pc}.svg'), dpi=300)
    plt.close()


def plot_top_feat_descriptives(feats_p1, feats_p2, top_feats_pc, top_feats_loadings, pc, phases, s,
                               top_feats_display_names, save_path, fs=7, conditions=False, feats_p1_LH=None, feats_p2_LH=None):
    feats_permouse_medians_p1 = feats_p1.groupby(level=0).mean()
    feats_permouse_medians_p2 = feats_p2.groupby(level=0).mean()

    if feats_p1_LH is not None and feats_p2_LH is not None:
        feats_permouse_medians_p1_LH = feats_p1_LH.groupby(level=0).mean()
        feats_permouse_medians_p2_LH = feats_p2_LH.groupby(level=0).mean()

    shared_mice = feats_permouse_medians_p1.index.intersection(feats_permouse_medians_p2.index)

    feats_permouse_medians_p1 = feats_permouse_medians_p1.loc[shared_mice]
    feats_permouse_medians_p2 = feats_permouse_medians_p2.loc[shared_mice]

    # Prepare data lists trimmed to shared mice
    data_p1 = [feats_permouse_medians_p1[feat].values for feat in top_feats_pc]
    data_p2 = [feats_permouse_medians_p2[feat].values for feat in top_feats_pc]

    # Get the phase colours and darker versions for the median and whiskers
    if not conditions:
        p1_color = pu.get_color_phase(phases[0])
        p2_color = pu.get_color_phase(phases[1])
    else:
        p1_color = pu.get_color_speedpair(phases[0])
        p2_color = pu.get_color_speedpair(phases[1])

    dark_color_p1 = pu.darken_color(p1_color, 0.35, to_black=True)
    dark_color_p2 = pu.darken_color(p2_color, 0.35, to_black=True)

    x = np.arange(len(top_feats_pc))
    width = 0.35
    bar_multiple = 0.6
    positions_p1 = x - width / 2
    positions_p2 = x + width / 2

    mean_scatter_size = 25
    scatter_size = 8
    line_scatter_length = 0.7

    # Create a figure with 3 subplots (loadings, phase values, and phase difference)
    fig, axs = plt.subplots(3, 1, figsize=(4, 6))

    # ### Subplot 0: Feature Loadings
    # axs[0].bar(x, top_feats_loadings, width * bar_multiple, alpha=0.7, color='k')
    # axs[0].set_xticks([])
    # axs[0].set_ylabel('Feature loading onto PC', fontsize=fs)
    # axs[0].set_ylim(-0.6, 0.6)
    # axs[0].set_yticks(np.arange(-0.5, 0.6, 0.5))

    # Compute p-values per feature for phase differences
    pvals = []
    for feat in top_feats_pc:
        vals_p1 = feats_permouse_medians_p1[feat].values
        vals_p2 = feats_permouse_medians_p2[feat].values
        stat, p = wilcoxon(vals_p1, vals_p2)
        pvals.append(p)

    ### Subplot 1: Phase Z-scored Feature Values
    for i, feat in enumerate(top_feats_pc):
        # Scatter all mice, phase 1
        axs[0].scatter([positions_p1[i]] * len(data_p1[i]), data_p1[i],
                       color=p1_color, edgecolor='none', alpha=0.7, s=scatter_size, zorder=10, label=phases[0] if i == 0 else "")
        # Scatter all mice, phase 2
        axs[0].scatter([positions_p2[i]] * len(data_p2[i]), data_p2[i],
                       color=p2_color, edgecolor='none', alpha=0.7, s=scatter_size, zorder=10, label=phases[1] if i == 0 else "")

        # Overlay means as larger scatter point (phase colour, larger size)
        # axs[0].scatter(positions_p1[i], np.mean(data_p1[i]), color='k', edgecolor='none', s=mean_scatter_size, zorder=20)
        # axs[0].scatter(positions_p2[i], np.mean(data_p2[i]), color='k', edgecolor='none', s=mean_scatter_size, zorder=20)
        axs[0].plot([positions_p1[i] - (width / 2) * line_scatter_length, positions_p1[i] + (width / 2) * line_scatter_length],
                    [np.mean(data_p1[i]), np.mean(data_p1[i])], color=dark_color_p1, linewidth=2, zorder=20, solid_capstyle='projecting')
        axs[0].plot([positions_p2[i] - (width / 2) * line_scatter_length, positions_p2[i] + (width / 2) * line_scatter_length],
                    [np.mean(data_p2[i]), np.mean(data_p2[i])], color=dark_color_p2, linewidth=2, zorder=20, solid_capstyle='projecting')

    # Connect paired values for each mouse across phases (lines)
    for midx in feats_permouse_medians_p1.index:
        if midx in feats_permouse_medians_p2.index:
            axs[0].plot([positions_p1, positions_p2],
                        [feats_permouse_medians_p1.loc[midx], feats_permouse_medians_p2.loc[midx]],
                        '-', alpha=0.8, color='gray', linewidth=0.5, zorder=1)

    pu.add_significance_stars(axs[0], positions_p1, positions_p2, data_p1, data_p2, pvals, fs=fs)

    p1_patch = mpatches.Patch(color=p1_color, label=f'{phases[0]}')
    p2_patch = mpatches.Patch(color=p2_color, label=f'{phases[1]}')
    axs[0].set_xticklabels('')
    axs[0].set_ylabel('Feature magnitude (z)', fontsize=fs)
    # legend labels
    axs[0].legend(handles=[p1_patch, p2_patch], fontsize=fs, loc='upper right', bbox_to_anchor=(1.2, 1),
                  title='Phase', title_fontsize=fs)

    ### Subplot 2: Projection of features on PCA
    weighted_features_p1 = [feature * loading for feature, loading in zip(data_p1, top_feats_loadings.values)]
    weighted_features_p2 = [feature * loading for feature, loading in zip(data_p2, top_feats_loadings.values)]

    # Convert the lists of weighted feature arrays into DataFrames.
    # Each column corresponds to a feature and each row to a mouse.
    weighted_df_p1 = pd.DataFrame(np.column_stack(weighted_features_p1),
                                  index=shared_mice,
                                  columns=top_feats_pc)
    weighted_df_p2 = pd.DataFrame(np.column_stack(weighted_features_p2),
                                  index=shared_mice,
                                  columns=top_feats_pc)

    pvals_projection = []
    for feat in top_feats_pc:
        vals_p1 = weighted_df_p1[feat].values
        vals_p2 = weighted_df_p2[feat].values
        stat, p = wilcoxon(vals_p1, vals_p2)
        pvals_projection.append(p)

    for i, feat in enumerate(top_feats_pc):
        # Scatter all mice, phase 1
        axs[1].scatter([positions_p1[i]] * len(weighted_df_p1[feat]), weighted_df_p1[feat].values,
                       color=p1_color, edgecolor='none', alpha=0.7, s=scatter_size, zorder=10)
        # Scatter all mice, phase 2
        axs[1].scatter([positions_p2[i]] * len(weighted_df_p2[feat]), weighted_df_p2[feat].values,
                       color=p2_color, edgecolor='none', alpha=0.7, s=scatter_size, zorder=10)

        # Overlay means
        # axs[1].scatter(positions_p1[i], np.mean(weighted_df_p1[feat]), color='k', edgecolor='none', s=mean_scatter_size, zorder=20)
        # axs[1].scatter(positions_p2[i], np.mean(weighted_df_p2[feat]), color='k', edgecolor='none', s=mean_scatter_size, zorder=20)
        axs[1].plot([positions_p1[i] - (width / 2) * line_scatter_length, positions_p1[i] + (width / 2) * line_scatter_length],
                    [np.mean(weighted_df_p1[feat]), np.mean(weighted_df_p1[feat])], color=dark_color_p1, linewidth=2, zorder=20, solid_capstyle='projecting')
        axs[1].plot([positions_p2[i] - (width / 2) * line_scatter_length, positions_p2[i] + (width / 2) * line_scatter_length],
                    [np.mean(weighted_df_p2[feat]), np.mean(weighted_df_p2[feat])], color=dark_color_p2, linewidth=2, zorder=20, solid_capstyle='projecting')

    # Connect paired values for each mouse (lines)
    for midx in weighted_df_p1.index:
        axs[1].plot([positions_p1, positions_p2],
                    [weighted_df_p1.loc[midx].values, weighted_df_p2.loc[midx].values],
                    '-', alpha=0.8, color='gray', linewidth=0.5, zorder=1)

    pu.add_significance_stars(axs[1], positions_p1, positions_p2, weighted_features_p1, weighted_features_p2,
                              pvals_projection, fs=fs)

    axs[1].set_ylabel('Projection onto PC', fontsize=fs)
    axs[1].set_ylim(-0.6, 0.6)
    axs[1].set_yticks(np.arange(-0.5, 0.6, 0.5))
    axs[1].set_yticklabels(np.arange(-0.5, 0.6, 0.5), fontsize=fs)
    axs[1].set_xticklabels('')

    ### Subplot 3: Phase Difference (p2 - p1)
    # Compute the per-mouse differences for each feature
    feats_diff = feats_permouse_medians_p2 - feats_permouse_medians_p1
    # Create a list of arrays (one per feature) for the differences
    data_diff = [feats_diff[feat].values for feat in top_feats_pc]
    data_diff_mean = np.mean(data_diff, axis=1)

    feats_diff_LH = feats_permouse_medians_p2_LH - feats_permouse_medians_p1_LH if feats_p1_LH is not None and feats_p2_LH is not None else None
    if feats_diff_LH is not None:
        # reduce to the same mice as feats_diff
        feats_diff_LH = feats_diff_LH.loc[shared_mice]
        data_diff_LH = [feats_diff_LH[feat].values for feat in top_feats_pc]
        data_diff_mean_LH = np.mean(data_diff_LH, axis=1)
        # Find if sign changes across features
        sign_flip = np.sign(data_diff_mean_LH) != np.sign(data_diff_mean)

    # Choose a neutral color for the phase differences
    diff_color = "#888888"

    # Scatter plot for phase differences (no boxplot)
    for i, feat in enumerate(top_feats_pc):
        diff_vals = feats_diff[feat].values
        # Plot individual mouse differences with some jitter for clarity
        x_jittered = np.random.normal(loc=x[i], scale=0.04, size=len(diff_vals))
        axs[2].scatter(x_jittered, diff_vals, color='k', edgecolor='none', alpha=0.7, s=scatter_size, zorder=10)
        # Overlay the mean as a large colored point (grey)
        # axs[2].scatter(x[i], np.mean(diff_vals), color='k', edgecolor='none', s=mean_scatter_size, zorder=20)
        axs[2].plot([x[i] - width * line_scatter_length, x[i] + width * line_scatter_length],
                    [np.mean(diff_vals), np.mean(diff_vals)], color='k', linewidth=2, zorder=20, solid_capstyle='projecting')


    # Optionally annotate 'FLIP'
    if feats_diff_LH is not None:
        for i, flipped in enumerate(sign_flip):
            if flipped:
                axs[2].text(x[i], axs[2].get_ylim()[1] * 0.9, 'FLIP', ha='center', va='top', color='red',
                            fontsize=fs - 1)

    axs[2].set_xticks(x)
    axs[2].set_xticklabels(top_feats_display_names, fontsize=fs, rotation=90)
    axs[2].set_ylabel(f'{phases[-1]} - {phases[0]} (z)', fontsize=fs)

    for ax in axs:
        #ax.set_xticks(x)
        if ax != axs[1]: #and ax != axs[0]:
            ax.set_ylim(-1.4, 1.4)
            ax.set_yticks(np.arange(-1, 1.1, 1))
            ax.set_yticklabels(np.arange(-1, 1.1, 1), fontsize=fs)
        ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.4)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(False)
        #ax.tick_params(axis='y', labelsize=fs)
        ax.set_xlim(-0.5, len(top_feats_pc) - 0.5)
        ax.tick_params(axis='y', which='both', left=True, labelsize=fs)
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.tick_params(axis='y', which='minor', length=4, width=1, color='k')

    plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.18, hspace=0.1) # to fit all feature names in bottom = 0.33

    # title
    plt.suptitle(pc, fontsize=fs)

    plt.savefig(os.path.join(save_path, f'PCA_top_features_{phases[0]}vs{phases[1]}_stride{s}_{pc}.png'), dpi=300)
    plt.savefig(os.path.join(save_path, f'PCA_top_features_{phases[0]}vs{phases[1]}_stride{s}_{pc}.svg'), dpi=300)
    plt.close()









