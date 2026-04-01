"""Analyse which strides before the gait transition carry anticipatory postural adjustment signals."""
import os
import pandas as pd
import pickle
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

from apa_analysis.config import (global_settings, condition_specific_settings)
from apa_analysis.Characterisation import General_utils as gu
from apa_analysis.Characterisation import Plotting_utils as pu
from helpers.config import *

# load LH pred data
LH_MultiFeatPath = PLOTS_ROOT + r"\LH_res_-3-2-1_APA2Wash2\APAChar_LowHigh_Extended\MultiFeaturePredictions"
LH_preprocessed_data_file_path = PLOTS_ROOT + r"\LH_res_-3-2-1_APA2Wash2\preprocessed_data_APAChar_LowHigh.pkl"
LH_stride_0_preprocessed_data_file_path = PLOTS_ROOT + r"\LH_LHpcsonly_res_0_APA2Wash2\preprocessed_data_APAChar_LowHigh.pkl"
LH_pred_path = f"{LH_MultiFeatPath}\\pca_predictions_APAChar_LowHigh.pkl"
LH_pca_path = f"{LH_MultiFeatPath}\\pca_APAChar_LowHigh.pkl"
with open(LH_preprocessed_data_file_path, 'rb') as f:
    data = pickle.load(f)
    LH_preprocessed_data = data['feature_data'] # this is the normalised! :)
with open(LH_stride_0_preprocessed_data_file_path, 'rb') as f:
    data = pickle.load(f)
    LH_stride0_preprocessed_data = data['feature_data']
with open(LH_pred_path, 'rb') as f:
    LH_pred_data = pickle.load(f)
with open(LH_pca_path, 'rb') as f:
    LH_pca_data = pickle.load(f)


class WhenAPA:
    def __init__(self, LH_feature_data, LH_feature_data_s0, LH_pred_data, LH_pca_data, base_dir):
        self.LH_feature_data = LH_feature_data
        self.LH_feature_data_s0 = LH_feature_data_s0
        self.LH_pred = LH_pred_data
        self.LH_pca = LH_pca_data
        self.base_dir = base_dir
        self.strides = [-1, -2, -3]

    def plot_accuracy_of_each_stride_model(self, fs=7):
        import scipy.stats

        # Collect accuracy data
        all_stride_accs = {}
        all_stride_cv_accs = {}
        for s in self.strides:
            stride_mice_names = [pred.mouse_id for pred in self.LH_pred if
                                 pred.stride == s]

            stride_accs = [pred.cv_acc for pred in self.LH_pred if
                           pred.stride == s]
            accs_df = pd.DataFrame(stride_accs, index=stride_mice_names)

            stride_null_accs = [pred.null_acc_circ for pred in self.LH_pred if
                                pred.stride == s]
            null_df = pd.DataFrame(stride_null_accs, index=stride_mice_names)

            delta_accs = accs_df - null_df
            delta_accs = delta_accs.mean(axis=1)

            all_stride_accs[s] = delta_accs
            all_stride_cv_accs[s] = accs_df.mean(axis=1)

        all_stride_accs_by_stride = pd.concat(all_stride_accs).reset_index()
        all_stride_accs_by_stride.columns = ['Stride', 'Mouse', 'Accuracy']
        all_stride_accs_by_stride['Stride_abs'] = all_stride_accs_by_stride[
            'Stride'].abs()
        df = all_stride_accs_by_stride

        for s in self.strides:
            stride_acc = all_stride_cv_accs[s]
            stride_acc_mean = stride_acc.mean()
            print(f"Stride {s}: Mean CV Accuracy = {stride_acc_mean:.3f}")

        stride_order = sorted(df['Stride_abs'].unique())
        palette = {s: pu.get_color_stride(-s) for s in stride_order}

        fig, ax = plt.subplots(figsize=(2, 3))

        means = []
        cis_lower = []
        cis_upper = []
        p_values = []

        # Calculate means, CIs, and significance vs chance
        for s in stride_order:
            accs = df[df['Stride_abs'] == s]['Accuracy']
            mean_val = accs.mean()
            sem = scipy.stats.sem(accs)
            ci_range = sem * scipy.stats.t.ppf((1 + 0.95) / 2, len(accs) - 1)

            means.append(mean_val)
            cis_lower.append(mean_val - ci_range)
            cis_upper.append(mean_val + ci_range)

            t_stat, p_value = ttest_1samp(accs, 0, alternative='greater')
            p_values.append(p_value)
            print(
                f"Stride {int(-s)}: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")

            # --- Pairwise comparisons between strides (paired) ---
            from itertools import combinations
            pairwise_results = []
            for (i, s1), (j, s2) in combinations(enumerate(stride_order), 2):
                accs1 = df[df['Stride_abs'] == s1].set_index('Mouse')[
                    'Accuracy']
                accs2 = df[df['Stride_abs'] == s2].set_index('Mouse')[
                    'Accuracy']
                # Align by mouse to ensure correct pairing
                common_mice = accs1.index.intersection(accs2.index)
                t_stat, p_val = scipy.stats.ttest_rel(accs1.loc[common_mice],
                                                      accs2.loc[common_mice])
                pairwise_results.append((i, j, p_val))
                print(
                    f"Stride {int(-s1)} vs {int(-s2)}: paired t-statistic = {t_stat:.3f}, p-value = {p_val:.3f}")

        # Jitter individual points
        for i, s in enumerate(stride_order):
            stride_color = palette[s]
            accs = df[df['Stride_abs'] == s]['Accuracy']
            jitter = np.random.uniform(-0.1, 0.1, size=len(accs))
            x_jittered = i + jitter
            ax.scatter(x_jittered, accs, marker='o', color=stride_color, s=7,
                       alpha=0.6, zorder=1)

        x_pos = np.arange(len(stride_order))

        # Plot mean points only (no lines)
        ax.scatter(x_pos, means, color='black', s=10, zorder=3)

        # Plot error bars (CI)
        ax.errorbar(
            x_pos, means,
            yerr=[np.array(means) - np.array(cis_lower),
                  np.array(cis_upper) - np.array(means)],
            fmt='none',
            ecolor='black',
            elinewidth=1,
            capsize=4,
            zorder=2
        )

        # Add significance stars (vs chance)
        for i, (p_value, mean_val) in enumerate(zip(p_values, means)):
            star = ''
            if p_value < 0.001 and mean_val > 0:
                star = '***'
            elif p_value < 0.01 and mean_val > 0:
                star = '**'
            elif p_value < 0.05 and mean_val > 0:
                star = '*'

            if star:
                y_max = max(cis_upper[i],
                            df[df['Stride_abs'] == stride_order[i]][
                                'Accuracy'].max())
                ax.text(i, y_max + 0.02, star, ha='center', va='bottom',
                        fontsize=fs)

        # --- Draw pairwise comparison brackets ---
        def _get_star_label(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            return 'n.s.'

        # Starting height for brackets (above the highest point + vs-chance stars)
        y_data_max = max(
            max(cis_upper),
            df['Accuracy'].max()
        )
        bracket_base = y_data_max + 0.12  # clear the vs-chance stars
        bracket_step = 0.08  # vertical spacing between stacked brackets

        for level, (i, j, p_val) in enumerate(pairwise_results):
            star_label = _get_star_label(p_val)
            y_bracket = bracket_base + level * bracket_step
            ax.plot(
                [i, i, j, j],
                [y_bracket - 0.015, y_bracket, y_bracket, y_bracket - 0.015],
                color='black', linewidth=0.8
            )
            ax.text(
                (i + j) / 2, y_bracket + 0.005, star_label,
                ha='center', va='bottom', fontsize=fs
            )

        # Chance line
        ax.axhline(y=0, color='gray', linestyle='--')

        # Y axis formatting — account for brackets
        y_vals = df['Accuracy']
        data_min = y_vals.min()
        y_hi_needed = bracket_base + len(
            pairwise_results) * bracket_step + 0.08
        span = y_hi_needed - data_min
        pad = span * 0.1
        y_lo = data_min - pad
        y_hi = y_hi_needed
        y_lo = np.floor(y_lo * 10) / 10
        y_hi = np.ceil(y_hi * 10) / 10
        yticks = np.linspace(y_lo, y_hi, 5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{t:.2f}" for t in yticks], fontsize=fs)

        ax.set_xlim(-0.5, len(stride_order) - 0.5)
        ax.set_xticks(range(len(stride_order)))
        ax.set_xticklabels([-s for s in stride_order], fontsize=fs)
        ax.set_xlabel('Stride', fontsize=fs)
        ax.set_ylabel('Delta CV Accuracy', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.invert_xaxis()
        fig.tight_layout()

        savepath = os.path.join(self.base_dir, 'WhenAPA_StrideModelAccuracy')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight',
                    dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight',
                    dpi=300)
        plt.close()

    def _compute_corr_matrix(self, df1, df2):
        """Pearson‐r matrix for two [runs × PCs] DataFrames."""
        n = global_settings['pcs_to_use']
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    M[i, j], _ = pearsonr(df1.iloc[:, i], df2.iloc[:, j])
                except:
                    M[i, j] = np.nan
        return M

    def _compute_mean_r(self, baseline_stride, compare_stride, run_type, pcs_byStride_interpolated, pcs_to_plot=None, eps=1e-6):
        """Returns the mean Pearson‐r matrix (after Fisher transform) for one stride comparison and run_type."""
        pcs_base = pcs_byStride_interpolated[baseline_stride]
        pcs_cmp = pcs_byStride_interpolated[compare_stride]
        mice = pcs_base.index.get_level_values(0).unique()
        zs = []
        for midx in mice:
            pc1 = pcs_base.loc[midx]
            pc2 = pcs_cmp.loc[midx]
            if run_type == 'APAlate':
                runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
                pc1 = pc1.loc[runs]
                pc2 = pc2.loc[runs]
            elif run_type == 'Washlate':
                runs = expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']
                pc1 = pc1.loc[runs]
                pc2 = pc2.loc[runs]
            C = self._compute_corr_matrix(pc1, pc2)
            if pcs_to_plot is not None:
                C = C[np.ix_(pcs_to_plot, pcs_to_plot)]
            C_clipped = np.clip(C, -1 + eps, 1 - eps)
            zs.append(np.arctanh(C_clipped))
        return np.tanh(np.nanmean(zs, axis=0))

    def _compute_stats(self, baseline_stride, compare_stride,
                             run_type, pcs_byStride_interpolated, pcs_to_plot=None, eps=1e-6):
        """
        Returns (mean_delta, stars) where
          mean_delta[i,j] = average over mice of (r_cmp[i,j] - r_base[i,j])
          stars[i,j]     = '', '*', '**', or '***' depending on p-value
        of a one-sample t-test that Δr ≠ 0.
        """
        pcs_base = pcs_byStride_interpolated[baseline_stride]
        pcs_cmp = pcs_byStride_interpolated[compare_stride]
        mice = pcs_base.index.get_level_values(0).unique()

        n_full = global_settings['pcs_to_use']
        # If pcs_to_plot is None, use all
        pcs_to_plot = pcs_to_plot if pcs_to_plot is not None else list(range(n_full))
        n = len(pcs_to_plot)
        deltas_restricted = []

        for midx in mice:
            pc1 = pcs_base.loc[midx]
            pc2 = pcs_cmp.loc[midx]
            if run_type == 'APAlate':
                runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
                pc1 = pc1.loc[runs]
                pc2 = pc2.loc[runs]
            elif run_type == 'Washlate':
                runs = expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']
                pc1 = pc1.loc[runs]
                pc2 = pc2.loc[runs]

            # compute and clip both matrices
            Cb = np.clip(self._compute_corr_matrix(pc1, pc1), -1 + eps, 1 - eps)
            Cc = np.clip(self._compute_corr_matrix(pc1, pc2), -1 + eps, 1 - eps)
            Cb = Cb[np.ix_(pcs_to_plot, pcs_to_plot)]
            Cc = Cc[np.ix_(pcs_to_plot, pcs_to_plot)]
            deltas_restricted.append(Cc)

        deltas_restricted = np.stack(deltas_restricted, axis=0) # shape (n_mice, n_pcs, n_pcs)
        mean_delta = np.nanmean(deltas_restricted, axis=0)

        n = mean_delta.shape[0]
        pvals = []
        positions = []
        # Collect all p-values
        for i in range(n):
            for j in range(n):
                vals = deltas_restricted[:, i, j]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 1:
                    _, p = ttest_1samp(vals, 0.0)
                else:
                    p = 1.0
                pvals.append(p)
                positions.append((i, j))

        # Apply Holm-Bonferroni correction
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='holm')

        stars = np.full((n, n), '', dtype=object)
        for idx, (i, j) in enumerate(positions):
            p = pvals_corrected[idx]
            if reject[idx]:
                if p < 0.001:
                    stars[i, j] = '***'
                elif p < 0.01:
                    stars[i, j] = '**'
                elif p < 0.05:
                    stars[i, j] = '*'
            else:
                stars[i, j] = ''

        return mean_delta, stars


    def _compute_delta_stats(self, baseline_stride, compare_stride,
                             run_type, pcs_byStride_interpolated, pcs_to_plot=None, eps=1e-6):
        """
        Returns (mean_delta, stars) where
          mean_delta[i,j] = average over mice of (r_cmp[i,j] - r_base[i,j])
          stars[i,j]     = '', '*', '**', or '***' depending on p-value
        of a one-sample t-test that Δr ≠ 0.
        """
        pcs_base = pcs_byStride_interpolated[baseline_stride]
        pcs_cmp = pcs_byStride_interpolated[compare_stride]
        mice = pcs_base.index.get_level_values(0).unique()

        n_full = global_settings['pcs_to_use']
        # If pcs_to_plot is None, use all
        pcs_to_plot = pcs_to_plot if pcs_to_plot is not None else list(range(n_full))
        n = len(pcs_to_plot)
        deltas_restricted = []

        for midx in mice:
            pc1 = pcs_base.loc[midx]
            pc2 = pcs_cmp.loc[midx]
            if run_type == 'APAlate':
                runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
                pc1 = pc1.loc[runs]
                pc2 = pc2.loc[runs]
            elif run_type == 'Washlate':
                runs = expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']
                pc1 = pc1.loc[runs]
                pc2 = pc2.loc[runs]

            # compute and clip both matrices
            Cb = np.clip(self._compute_corr_matrix(pc1, pc1), -1 + eps, 1 - eps)
            Cc = np.clip(self._compute_corr_matrix(pc1, pc2), -1 + eps, 1 - eps)
            Cb = Cb[np.ix_(pcs_to_plot, pcs_to_plot)]
            Cc = Cc[np.ix_(pcs_to_plot, pcs_to_plot)]
            deltas_restricted.append(Cc - Cb)

        deltas_restricted = np.stack(deltas_restricted, axis=0) # shape (n_mice, n_pcs, n_pcs)
        mean_delta = np.nanmean(deltas_restricted, axis=0)

        n = mean_delta.shape[0]
        pvals = []
        positions = []
        # Collect all p-values
        for i in range(n):
            for j in range(n):
                vals = deltas_restricted[:, i, j]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 1:
                    _, p = ttest_1samp(vals, 0.0)
                else:
                    p = 1.0
                pvals.append(p)
                positions.append((i, j))

        # Apply Holm-Bonferroni correction
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='holm')

        stars = np.full((n, n), '', dtype=object)
        for idx, (i, j) in enumerate(positions):
            p = pvals_corrected[idx]
            if reject[idx]:
                if p < 0.001:
                    stars[i, j] = '***'
                elif p < 0.01:
                    stars[i, j] = '**'
                elif p < 0.05:
                    stars[i, j] = '*'
            else:
                stars[i, j] = ''

        return mean_delta, stars

    def _plot_heatmap(self, mat, label, run_type, fs=7, suffix="", pcs_to_plot=None):
        """Generic heatmap plotting + saving."""
        if pcs_to_plot is None:
            pcs_to_plot = list(range(global_settings['pcs_to_use']))
        n = len(pcs_to_plot)
        xl = [f"PC{pc + 1} ({label})" for pc in pcs_to_plot]
        yl = [f"PC{pc + 1} (-1)" for pc in pcs_to_plot]

        if pcs_to_plot is None or len(pcs_to_plot) == global_settings['pcs_to_use']:
            figsize = (10, 6)  # old fixed size for full heatmaps
        else:
            n = len(pcs_to_plot)
            figsize = (5,4)  # smaller for subset
        fig, ax = plt.subplots(figsize=figsize)
        heatmap = sns.heatmap(mat, vmin=-1, vmax=1, cmap='coolwarm',
                    xticklabels=xl, yticklabels=yl,
                    cbar_kws={'label': 'Pearson Correlation'}, ax=ax)
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Pearson Correlation', fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
        ax.set_title(f"{'Δ ' if suffix else ''}Stride -1 vs {label}  ({run_type})", fontsize=fs)
        ax.tick_params(labelsize=fs)
        ax.figure.axes[-1].tick_params(labelsize=fs)
        plt.tight_layout()
        if pcs_to_plot is not None:
            fname = f"CorrPCs_{run_type}_Stride{label}{suffix}_PCs{pcs_to_plot}"
        else:
            fname = f"CorrPCs_{run_type}_Stride{label}{suffix}"
        for ext in ('png', 'svg'):
            plt.savefig(os.path.join(self.base_dir, fname + f".{ext}"),
                        bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_density_grid(self, baseline_stride, compare_stride, pcs_byStride_interpolated, fs=7, bins=30):
        """Plot density grids (2D histograms) for APA2 and Wash2 separately for stride comparison."""
        apa_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
        wash_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']
        apa_color = pu.get_color_phase('APA2')
        wash_color = pu.get_color_phase('Wash2')
        apa_cmap = pu.make_triple_cmap(apa_color)
        apa_cmap = pu.make_cmap_with_white_bottom(apa_cmap)
        wash_cmap = pu.make_triple_cmap(wash_color)
        wash_cmap = pu.make_cmap_with_white_bottom(wash_cmap)

        pcs_base = pcs_byStride_interpolated[baseline_stride]
        pcs_cmp = pcs_byStride_interpolated[compare_stride]
        mice = pcs_base.index.get_level_values(0).unique()

        apa_x = np.zeros((global_settings['pcs_to_use'], len(mice), len(apa_runs)))
        apa_y = np.zeros((global_settings['pcs_to_use'], len(mice), len(apa_runs)))
        wash_x = np.zeros((global_settings['pcs_to_use'], len(mice), len(wash_runs)))
        wash_y = np.zeros((global_settings['pcs_to_use'], len(mice), len(wash_runs)))

        for pc_idx in range(global_settings['pcs_to_use']):
            for midx, mouse in enumerate(mice):
                pc1 = pcs_base.loc[mouse]
                pc2 = pcs_cmp.loc[mouse]

                pc1a = pc1.loc(axis=0)[apa_runs].iloc[:, pc_idx]
                pc2a = pc2.loc(axis=0)[apa_runs].iloc[:, pc_idx]
                pc1w = pc1.loc(axis=0)[wash_runs].iloc[:, pc_idx]
                pc2w = pc2.loc(axis=0)[wash_runs].iloc[:, pc_idx]

                apa_x[pc_idx, midx, :] = pc1a.values
                apa_y[pc_idx, midx, :] = pc2a.values
                wash_x[pc_idx, midx, :] = pc1w.values
                wash_y[pc_idx, midx, :] = pc2w.values

        PCs = np.array([1, 3, 7])
        for pc_name in PCs:
            pc_idx = pc_name - 1  # zero-based index
            compare_pcs = PCs

            for compare_pc in compare_pcs:
                compare_pc_idx = compare_pc - 1

                fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
                ax_apa, ax_wash = axs

                # Gather flattened data
                apa_data_x = apa_x[pc_idx, :, :].flatten()
                apa_data_y = apa_y[compare_pc_idx, :, :].flatten()
                wash_data_x = wash_x[pc_idx, :, :].flatten()
                wash_data_y = wash_y[compare_pc_idx, :, :].flatten()

                # Calculate axis limits (rounding to nearest 5)
                def round_down_to_5(x):
                    return np.floor(x / 5) * 5

                def round_up_to_5(x):
                    return np.ceil(x / 5) * 5

                all_x = np.concatenate([apa_data_x, wash_data_x])
                all_y = np.concatenate([apa_data_y, wash_data_y])
                x_min, x_max = round_down_to_5(all_x.min()), round_up_to_5(all_x.max())
                y_min, y_max = round_down_to_5(all_y.min()), round_up_to_5(all_y.max())

                # Plot APA density
                h_apa = ax_apa.hist2d(apa_data_x, apa_data_y, bins=bins, range=[[x_min, x_max], [y_min, y_max]],
                                      cmap=apa_cmap)
                # plot equality line
                ax_apa.plot([x_min, x_max], [x_min, x_max], color='black', linestyle='--', linewidth=0.5)
                ax_apa.set_title('APAlate', fontsize=fs)
                ax_apa.set_xlabel(f'PC{pc_name} ({baseline_stride})', fontsize=fs)
                ax_apa.set_ylabel(f'PC{compare_pc} ({compare_stride})', fontsize=fs)
                ax_apa.spines['top'].set_visible(False)
                ax_apa.spines['right'].set_visible(False)
                ax_apa.set_xlim(x_min, x_max)
                ax_apa.set_ylim(y_min, y_max)
                ax_apa.set_xticks(np.arange(x_min, x_max + 1, 5))
                ax_apa.set_yticks(np.arange(y_min, y_max + 1, 5))
                ax_apa.tick_params(axis='x',labelsize=fs)
                ax_apa.yaxis.set_major_locator(plt.MultipleLocator(5))  # ticks every 5 units
                # ax_apa.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
                ax_apa.tick_params(axis='y', labelsize=fs)

                # Plot Wash density
                h_wash = ax_wash.hist2d(wash_data_x, wash_data_y, bins=bins, range=[[x_min, x_max], [y_min, y_max]],
                                        cmap=wash_cmap)
                # plot equality line
                ax_wash.plot([x_min, x_max], [x_min, x_max], color='black', linestyle='--', linewidth=0.5)
                ax_wash.set_title('Washlate', fontsize=fs)
                ax_wash.set_xlabel(f'PC{pc_name} ({baseline_stride})', fontsize=fs)
                # y-label only on left plot
                ax_wash.tick_params(axis='y', labelleft=True, labelsize=fs)
                ax_wash.tick_params(axis='x', labelsize=fs)
                ax_wash.spines['top'].set_visible(False)
                ax_wash.spines['right'].set_visible(False)
                ax_wash.set_xlim(x_min, x_max)
                ax_wash.set_ylim(y_min, y_max)
                ax_wash.set_xticks(np.arange(x_min, x_max + 1, 5))

                ax_wash.yaxis.set_major_locator(plt.MultipleLocator(5))  # ticks every 5 units
                # ax_wash.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

                fig.tight_layout()

                fname = f"DensityGrid_APA_vs_Wash_PC{pc_name}xPC{compare_pc}_Stride{compare_stride}"
                for ext in ('png', 'svg'):
                    fig.savefig(os.path.join(self.base_dir, fname + f".{ext}"),
                                bbox_inches='tight', dpi=300)
                plt.close(fig)

    def _plot_scatter(self, baseline_stride, compare_stride, pcs_byStride_interpolated, fs=7):
        """Scatter PC means (APA2 vs Wash2) for one stride comparison."""
        apa_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
        wash_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']
        apa_color = pu.get_color_phase('APA2')
        wash_color = pu.get_color_phase('Wash2')

        pcs_base = pcs_byStride_interpolated[baseline_stride]
        pcs_cmp = pcs_byStride_interpolated[compare_stride]
        mice = pcs_base.index.get_level_values(0).unique()

        apa_x = np.zeros((global_settings['pcs_to_use'], len(mice), len(apa_runs)))
        apa_y = np.zeros((global_settings['pcs_to_use'], len(mice), len(apa_runs)))
        wash_x = np.zeros((global_settings['pcs_to_use'], len(mice), len(wash_runs)))
        wash_y = np.zeros((global_settings['pcs_to_use'], len(mice), len(wash_runs)))
        for pc_idx in range(global_settings['pcs_to_use']):
            for midx, mouse in enumerate(mice):
                pc1 = pcs_base.loc[mouse]
                pc2 = pcs_cmp.loc[mouse]

                pc1a = pc1.loc(axis=0)[apa_runs].iloc[:, pc_idx]
                pc2a = pc2.loc(axis=0)[apa_runs].iloc[:, pc_idx]
                pc1w = pc1.loc(axis=0)[wash_runs].iloc[:, pc_idx]
                pc2w = pc2.loc(axis=0)[wash_runs].iloc[:, pc_idx]

                apa_x[pc_idx, midx, :] = pc1a.values
                apa_y[pc_idx, midx, :] = pc2a.values
                wash_x[pc_idx, midx, :] = pc1w.values
                wash_y[pc_idx, midx, :] = pc2w.values

        PCs = np.array([1,3,7])
        for pc_name in PCs:
            pc_idx = pc_name - 1  # Convert to zero-based index
            compare_pcs = PCs

            for compare_pc in compare_pcs:
                compare_pc_idx = compare_pc - 1  # Convert to zero-based index

                fig, ax = plt.subplots(figsize=(2, 2))
                ax.scatter(apa_x[pc_idx, :, :].flatten(), apa_y[compare_pc_idx, :, :].flatten(),
                           marker='o', s=3, label='APAlate', alpha=0.1, color=apa_color, linewidth=0)
                ax.scatter(wash_x[pc_idx, :, :].flatten(), wash_y[compare_pc_idx, :, :].flatten(),
                            marker='o', s=3, label='Washlate', alpha=0.1, color=wash_color, linewidth=0)
                ax.set_xlabel(f'PC{pc_name} ({baseline_stride})', fontsize=fs)
                ax.set_ylabel(f'PC{compare_pc} ({compare_stride})', fontsize=fs)
                # Compute min/max of the data for x and y
                x_data = np.concatenate([apa_x[pc_idx, :, :].flatten(), wash_x[pc_idx, :, :].flatten()])
                y_data = np.concatenate([apa_y[compare_pc_idx, :, :].flatten(), wash_y[compare_pc_idx, :, :].flatten()])

                def round_down_to_5(x):
                    return np.floor(x / 5) * 5

                def round_up_to_5(x):
                    return np.ceil(x / 5) * 5

                x_min, x_max = round_down_to_5(x_data.min()), round_up_to_5(x_data.max())
                y_min, y_max = round_down_to_5(y_data.min()), round_up_to_5(y_data.max())

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                # Set ticks at multiples of 5 within the limits
                ax.set_xticks(np.arange(x_min, x_max + 1, 5))
                ax.set_yticks(np.arange(y_min, y_max + 1, 5))
                # ax.set_xlim(-1, 1)
                # ax.set_ylim(-1, 1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend(bbox_to_anchor=(1, 1), loc='upper left', frameon=False, fontsize=fs)
                ax.tick_params(labelsize=fs)
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.85)
                fname = f"ScatterAPA_vs_Wash_PC{pc_name}xPC{compare_pc}_Stride{compare_stride}"
                for ext in ('png', 'svg'):
                    fig.savefig(os.path.join(self.base_dir, fname + f".{ext}"),
                                bbox_inches='tight', dpi=300)
                plt.close()

    def _plot_delta_heatmap(self, mat, stars, label, run_type, fs=7, suffix="", pcs_to_plot=None):
        if pcs_to_plot is None:
            pcs_to_plot = list(range(global_settings['pcs_to_use']))
        n = len(pcs_to_plot)
        xl = [f"PC{pc + 1} ({label})" for pc in pcs_to_plot]
        yl = [f"PC{pc + 1} (-1)" for pc in pcs_to_plot]

        if pcs_to_plot is None or len(pcs_to_plot) == global_settings['pcs_to_use']:
            figsize = (10, 6)
        else:
            figsize = (5, 4)
        fig, ax = plt.subplots(figsize=figsize)
        heatmap = sns.heatmap(
            mat, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=xl, yticklabels=yl, cbar_kws={'label': 'Δ Pearson r'},
            annot=False, ax=ax
        )
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Δ Pearson r', fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)# set fontsize here

        for i in range(n):
            for j in range(n):
                star = stars[i, j]
                if star:
                    ax.text(j + 0.5, i + 0.5, star,
                            ha='center', va='center', fontsize=fs, color='black')

        ax.set_title(f"Δ Stride -1 vs {label}  ({run_type})", fontsize=fs)
        ax.tick_params(labelsize=fs)
        ax.figure.axes[-1].tick_params(labelsize=fs)
        plt.tight_layout()

        if pcs_to_plot is not None:
            fname = f"CorrPCs_{run_type}_Stride{label}{suffix}_PCs{pcs_to_plot}"
        else:
            fname = f"CorrPCs_{run_type}_Stride{label}{suffix}_delta"
        for ext in ('png', 'svg'):
            plt.savefig(os.path.join(self.base_dir, fname + f".{ext}"),
                        bbox_inches='tight', dpi=300)
        plt.close()

    def plot_corr_pcs_heatmap(self, fs=7, pcs_to_plot=None):
        pca = self.LH_pca[0].pca
        pcs_byStride = {}
        pcs_byStride_interpolated = {}
        for s in self.strides:
            stride_feature_data = self.LH_feature_data.loc(axis=0)[s]
            mice_names = stride_feature_data.index.get_level_values('MouseID').unique()

            pcs_byMouse = {}
            pcs_byMouse_interpolated = {}
            for midx in mice_names:
                pc_df = pd.DataFrame(
                    index=np.arange(160),
                    columns=[f'PC{i + 1}' for i in range(global_settings['pcs_to_use'])]
                )
                pc_interp_df = pc_df.copy()

                mouse_data = stride_feature_data.loc[midx]
                pcs = pca.transform(mouse_data)[:, :global_settings['pcs_to_use']]
                runs = mouse_data.index.get_level_values('Run').unique()

                pcs_interp = np.array([
                    np.interp(np.arange(160), runs, pcs[:, i])
                    for i in range(global_settings['pcs_to_use'])
                ]).T

                pc_df.loc[runs, :] = pcs
                pc_interp_df.loc[:, :] = pcs_interp

                pcs_byMouse[midx] = pc_df
                pcs_byMouse_interpolated[midx] = pc_interp_df

            pcs_byStride[s] = pd.concat(pcs_byMouse)
            pcs_byStride_interpolated[s] = pd.concat(pcs_byMouse_interpolated)

        pcs_to_plot = pcs_to_plot or list(range(global_settings['pcs_to_use']))
        # 1) raw heatmaps + scatters
        for rt in ['APAlate', 'Washlate']: # 'All runs',
            for s in (-1, -2, -3):
                mean_r = self._compute_mean_r(-1, s, rt, pcs_byStride_interpolated, pcs_to_plot=pcs_to_plot)
                self._plot_heatmap(mean_r, s, rt, fs=fs, pcs_to_plot=pcs_to_plot)
                self._plot_scatter(-1, s, pcs_byStride_interpolated,
                                   fs=fs)  # You may want to modify _plot_scatter similarly
                self._plot_density_grid(-1, s, pcs_byStride_interpolated, fs=fs, bins=60)

            mean_full2, stars_full2 = self._compute_stats(-1, -2, rt, pcs_byStride_interpolated, pcs_to_plot=pcs_to_plot)
            mean_full3, stars_full3 = self._compute_stats(-1, -3, rt, pcs_byStride_interpolated, pcs_to_plot=pcs_to_plot)
            mean_full1, stars_full1 = self._compute_stats(-1, -1, rt, pcs_byStride_interpolated, pcs_to_plot=pcs_to_plot)


            mean2, stars2 = self._compute_delta_stats(-1, -2, rt, pcs_byStride_interpolated, pcs_to_plot=pcs_to_plot)
            mean3, stars3 = self._compute_delta_stats(-1, -3, rt, pcs_byStride_interpolated, pcs_to_plot=pcs_to_plot)

            self._plot_delta_heatmap(mean2, stars2, '-2 minus -1', rt, fs=fs, suffix='_Delta2', pcs_to_plot=pcs_to_plot)
            self._plot_delta_heatmap(mean3, stars3, '-3 minus -1', rt, fs=fs, suffix='_Delta3', pcs_to_plot=pcs_to_plot)

    def plot_line_pcs_apa_vs_wash(self, fs=7):
        apa_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
        wash_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']
        strides = [-3, -2, -1]

        # Precompute PCA projections for each stride and mouse
        pcs_byStride = {}
        pca = self.LH_pca[0].pca
        for s in strides:
            if s != 0:
                data = self.LH_feature_data.loc(axis=0)[s]
            else:
                data = self.LH_feature_data_s0.loc(axis=0)[s]

            pcs_byMouse = {}
            for midx in data.index.get_level_values('MouseID').unique():
                mouse_df = data.loc[midx]
                projected = pca.transform(mouse_df)[:, :global_settings['pcs_to_use']]
                pc_df = pd.DataFrame(
                    projected,
                    index=mouse_df.index.get_level_values('Run'),
                    columns=[f'PC{i+1}' for i in range(global_settings['pcs_to_use'])]
                )
                pcs_byMouse[midx] = pc_df
            pcs_byStride[s] = pcs_byMouse

        # Helper to map p-value to star string
        def p2star(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return ''

        # Build delta (APA-Wash) matrix and p-value matrix for all PCs and strides
        pc_labels = [f'PC{i+1}' for i in range(global_settings['pcs_to_use'])]
        delta_df = pd.DataFrame(index=pc_labels, columns=strides, dtype=float)
        pmat = pd.DataFrame(index=pc_labels, columns=strides, dtype=float)

        for pc_idx, label in enumerate(pc_labels):
            for s in strides:
                apa_vals, wash_vals = [], []
                for midx, pc_df in pcs_byStride[s].items():
                    apa_vals.append(
                        pc_df.loc[pc_df.index.isin(apa_runs), label].mean()
                    )
                    wash_vals.append(
                        pc_df.loc[pc_df.index.isin(wash_runs), label].mean()
                    )
                diffs = np.array(apa_vals) - np.array(wash_vals)
                delta_df.loc[label, s] = np.nanmean(diffs)
                _, p = ttest_1samp(diffs, 0.0, alternative='two-sided', nan_policy='omit')
                # _, p = ttest_rel(apa_vals, wash_vals, alternative='greater', nan_policy='omit')
                pmat.loc[label, s] = p

        # Map p-values to stars for heatmap
        star_df = pmat.applymap(p2star)
        # Create combined annotation strings with star above diff value
        annot = delta_df.round(2).astype(str)
        for r in delta_df.index:
            for c in delta_df.columns:
                star = star_df.loc[r, c]
                val = annot.loc[r, c]
                if star:
                    annot.loc[r, c] = f"{star}\n{val}"
                else:
                    annot.loc[r, c] = val

        fig, ax = plt.subplots(figsize=(4, 6))
        heatmap = sns.heatmap(
            delta_df,
            annot=annot,
            fmt="",
            cmap='coolwarm',
            center=0,
            cbar_kws={'label': 'Delta (APA - Wash)'},
            vmin=-2, vmax=2,
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            annot_kws={"size": fs},
        )

        # Set font size of colorbar label
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Delta (APA - Wash)', fontsize=fs)

        # Set font size for colorbar tick labels
        cbar.ax.tick_params(labelsize=fs)

        # Set axis labels, title, and ticks font size
        ax.set_xlabel('Stride', fontsize=fs)
        ax.set_ylabel('PC', fontsize=fs)
        ax.set_title('APA - Wash', fontsize=fs)
        ax.tick_params(labelsize=fs)

        plt.tight_layout()
        heatmap_base = os.path.join(self.base_dir, 'DeltaHeatmap_APA_Wash')
        fig.savefig(f"{heatmap_base}.png", bbox_inches='tight', dpi=300)
        fig.savefig(f"{heatmap_base}.svg", bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Line plots for each PC with CI and stars
        stride_labels = strides
        for pc_idx, label in enumerate(pc_labels):
            apa_means, wash_means, delta_means = [], [], []
            apa_CIs, wash_CIs, delta_CIs = [], [], []
            stars = []

            for s in strides:
                apa_vals = []
                wash_vals = []
                for midx, pc_df in pcs_byStride[s].items():
                    apa_vals.append(
                        pc_df.loc[pc_df.index.isin(apa_runs), label].mean()
                    )
                    wash_vals.append(
                        pc_df.loc[pc_df.index.isin(wash_runs), label].mean()
                    )
                diffs = np.array(apa_vals) - np.array(wash_vals)

                apa_means.append(np.nanmean(apa_vals))
                wash_means.append(np.nanmean(wash_vals))
                delta_means.append(np.nanmean(diffs))
                apa_CIs.append(np.nanstd(apa_vals)/np.sqrt(len(apa_vals))*1.96)
                wash_CIs.append(np.nanstd(wash_vals)/np.sqrt(len(wash_vals))*1.96)
                delta_CIs.append(np.nanstd(diffs)/np.sqrt(len(diffs))*1.96)
                stars.append(p2star(pmat.loc[label, s]))

            fig, (ax_delta, ax_lines) = plt.subplots(2, 1, figsize=(3, 4), sharex=True,
                                           gridspec_kw={'height_ratios': [1, 2]})

            # Top subplot: Delta with error bars + stars
            ax_delta.errorbar(
                stride_labels, delta_means, yerr=delta_CIs,
                color='teal', marker='o', ms=5, linestyle='--', linewidth=1, capsize=3, label='Delta APA–Wash'
            )
            ax_delta.set_ylabel('Delta (mean ± 95% CI)', fontsize=fs)
            ax_delta.tick_params(axis='y', labelsize=fs)
            ax_delta.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_delta.spines['top'].set_visible(False)
            ax_delta.spines['right'].set_visible(False)
            ax_delta.spines['bottom'].set_visible(False)
            ax_delta.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax_delta.set_title(label, fontsize=fs)

            # Set y-axis limits/ticks for DELTA (multiples of 2, outward rounding)
            delta_min = np.min([m - ci for m, ci in zip(delta_means, delta_CIs)])
            delta_max = np.max([m + ci for m, ci in zip(delta_means, delta_CIs)])
            delta_min_tick = self.outward_round(delta_min, 2, 'down')
            delta_max_tick = self.outward_round(delta_max, 2, 'up')
            ax_delta.set_ylim(delta_min_tick, delta_max_tick)
            ax_delta.set_yticks(np.arange(delta_min_tick, delta_max_tick + 1, 2))

            # Add stars above error bars on delta plot
            for x, d_mean, d_ci, star in zip(stride_labels, delta_means, delta_CIs, stars):
                y = d_mean + d_ci + 0.02
                ax_delta.text(x, y, star, ha='center', va='bottom', fontsize=fs)

            # Bottom subplot: APA & Wash with error bars
            ax_lines.errorbar(
                stride_labels, apa_means, yerr=apa_CIs,
                color=pu.get_color_phase('APA2'), marker='o', ms=5, linewidth=1, capsize=3, label='APA'
            )
            ax_lines.errorbar(
                stride_labels, wash_means, yerr=wash_CIs,
                color=pu.get_color_phase('Wash2'), marker='o', ms=5, linewidth=1, capsize=3, label='Wash'
            )
            ax_lines.set_xlabel('Stride', fontsize=fs)
            ax_lines.set_xticks(stride_labels)
            ax_lines.set_xticklabels([str(s) for s in stride_labels], fontsize=fs)
            ax_lines.set_ylabel('PC value (mean ± 95% CI)', fontsize=fs)
            ax_lines.tick_params(axis='both', labelsize=fs)
            ax_lines.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax_lines.spines['top'].set_visible(False)
            ax_lines.legend(fontsize=fs, loc='best')

            # Set y-axis limits/ticks for lines (multiples of 1, outward rounding)
            all_y = np.concatenate([
                [m - ci for m, ci in zip(apa_means, apa_CIs)],
                [m + ci for m, ci in zip(apa_means, apa_CIs)],
                [m - ci for m, ci in zip(wash_means, wash_CIs)],
                [m + ci for m, ci in zip(wash_means, wash_CIs)],
            ])
            lines_min = np.min(all_y)
            lines_max = np.max(all_y)
            lines_min_tick = self.outward_round(lines_min, 1, 'down')
            lines_max_tick = self.outward_round(lines_max, 1, 'up')
            ax_lines.set_ylim(lines_min_tick, lines_max_tick)
            ax_lines.set_yticks(np.arange(lines_min_tick, lines_max_tick + 1, 1))

            fig.tight_layout()

            # Save line plot
            base = os.path.join(self.base_dir, f'LinePlot_{label}_Delta_APA_Wash')
            fig.savefig(f"{base}.png", bbox_inches='tight', dpi=300)
            fig.savefig(f"{base}.svg", bbox_inches='tight', dpi=300)
            plt.close(fig)

    def outward_round(self, val, base, direction):
        """
        Rounds val outward to the nearest multiple of base.
        direction: 'up' for max, 'down' for min.
        """
        if direction == 'up':
            return base * np.ceil(val / base)
        elif direction == 'down':
            return base * np.floor(val / base)
        else:
            raise ValueError("direction must be 'up' or 'down'")

    def plot_pcs_timeseries_by_stride(self, fs=7, smooth_window=15):
        """
        Plot PC timeseries across trials for each stride, one plot per PC.
        Each plot: x = trial, y = PC value (mean across mice), line = stride.
        """
        from scipy.ndimage import median_filter

        pca = self.LH_pca[0].pca
        goal_runs = np.arange(160)
        strides = self.strides if hasattr(self, 'strides') else [-1, -2, -3]
        pc_labels = [f'PC{i + 1}' for i in range(global_settings['pcs_to_use'])]

        # --- For each stride, collect smoothed and normalized PCs for all mice ---
        # Format: {stride: {mouse: (160 x n_pc array)}}
        pcs_byStride_byMouse = {}
        for s in strides:
            if s != 0:
                data = self.LH_feature_data.loc(axis=0)[s]
            else:
                data = self.LH_feature_data_s0.loc(axis=0)[s]
            pcs_byMouse = {}
            for midx in data.index.get_level_values('MouseID').unique():
                mouse_df = data.loc[midx]
                pcs = pca.transform(mouse_df)[:, :global_settings['pcs_to_use']]
                run_vals = mouse_df.index.get_level_values('Run')
                # Interpolate to full 160 trials
                pcs_interp = np.vstack([
                    np.interp(goal_runs, run_vals, pcs[:, i])
                    for i in range(pcs.shape[1])
                ]).T  # shape: 160 x n_pcs
                # Optionally smooth and normalise each PC trace
                pcs_smooth = median_filter(pcs_interp, size=(smooth_window, 1), mode='nearest')
                # Normalise each PC trace independently (per mouse, per PC)
                max_abs = np.max(np.abs(pcs_smooth), axis=0)
                pcs_norm = pcs_smooth / (max_abs + 1e-10)
                pcs_byMouse[midx] = pcs_norm
            pcs_byStride_byMouse[s] = pcs_byMouse

        apa1_color = pu.get_color_phase('APA1')
        apa2_color = pu.get_color_phase('APA2')
        wash1_color = pu.get_color_phase('Wash1')
        wash2_color = pu.get_color_phase('Wash2')

        # --- For each PC, plot all strides (average across mice, with shaded SEM) ---
        for pc_idx, pc_label in enumerate(pc_labels):

            fig, ax = plt.subplots(figsize=(5, 5))

            boxy = 1
            height = 0.02
            patch1 = plt.axvspan(xmin=10, xmax=60, ymin=boxy, ymax=boxy + height, color=apa1_color, lw=0)
            patch2 = plt.axvspan(xmin=60, xmax=110, ymin=boxy, ymax=boxy + height, color=apa2_color, lw=0)
            patch3 = plt.axvspan(xmin=110, xmax=135, ymin=boxy, ymax=boxy + height, color=wash1_color, lw=0)
            patch4 = plt.axvspan(xmin=135, xmax=160, ymin=boxy, ymax=boxy + height, color=wash2_color, lw=0)
            patch1.set_clip_on(False)
            patch2.set_clip_on(False)
            patch3.set_clip_on(False)
            patch4.set_clip_on(False)

            for daybreak in [40,80, 120]:
                ax.axvline(daybreak, color='grey', linestyle='--', linewidth=0.5)
            ax.axvline(10, color=apa1_color, linestyle='-', linewidth=0.5)
            ax.axvline(110, color=wash1_color, linestyle='-', linewidth=0.5)
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)

            for s in strides:
                # Collect (mice x trials) for this PC and stride
                pcs_all_mice = []
                for pcs_norm in pcs_byStride_byMouse[s].values():
                    pcs_all_mice.append(pcs_norm[:, pc_idx])
                pcs_all_mice = np.stack(pcs_all_mice)  # [n_mice, 160]
                mean_trace = pcs_all_mice.mean(axis=0)
                sem_trace = pcs_all_mice.std(axis=0, ddof=1) / np.sqrt(pcs_all_mice.shape[0])
                color = pu.get_color_stride(s)
                label = f'Stride {abs(s)}'
                # Plot with shaded error
                ax.plot(goal_runs + 1, mean_trace, color=color, label=label, linewidth=2)
                ax.fill_between(goal_runs + 1, mean_trace - sem_trace, mean_trace + sem_trace,
                                color=color, alpha=0.16, linewidth=0)



            ax.set_xlabel('Trial number', fontsize=fs)
            ax.set_ylabel(f'Normalised {pc_label}', fontsize=fs)
            ax.set_title(f'{pc_label} timeseries by stride', fontsize=fs)
            ax.set_xlim(1, 160)
            ax.set_xticks([0, 10, 40, 60, 80, 110, 120, 135, 160])
            ax.set_xticklabels(['0', '10', '40', '60', '80', '110', '120', '135', '160'], fontsize=fs)
            ax.set_ylim(-1, 1)
            ax.set_yticks(np.arange(-1, 1.1, 0.5))
            ax.set_yticklabels(np.arange(-1, 1.1, 0.5), fontsize=fs)
            ax.tick_params(axis='both', labelsize=fs)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(fontsize=fs, frameon=False)
            plt.tight_layout()
            savepath = os.path.join(self.base_dir, f'Timeseries_{pc_label}_byStride')
            plt.savefig(f"{savepath}.png", bbox_inches='tight', dpi=300)
            plt.savefig(f"{savepath}.svg", bbox_inches='tight', dpi=300)
            plt.close(fig)

    def plot_stride_times(self, fs=7):
        fps=247
        stride_data, _ = gu.collect_stride_data('APAChar_LowHigh', 'Extended', None, 'APAChar_HighLow')
        mice = condition_specific_settings['APAChar_LowHigh']['global_fs_mouse_ids']
        strides = [0, -1, -2, -3]  # Transition and preceding strides

        stride_times_per_mouse = {mouse: {} for mouse in mice}

        for mouse in mice:
            stride_stance_byDay_times = pd.DataFrame(index=np.arange(0,160), columns=strides, dtype=float)
            mouse_stride_data = stride_data.loc[mouse]

            # Find stride 0 stance frames (transition points)
            stride0_mask = mouse_stride_data.loc(axis=1)['Stride_no'] == 0
            stride0_data = mouse_stride_data.loc[stride0_mask]

            # Use ForepawL and ForepawR data together, whichever is not NaN
            stride0_stance_mask =  (stride0_data.loc[:, ('ForepawL', 'SwSt_discrete')] == locostuff['swst_vals_2025']['st']) | (stride0_data.loc[:, ('ForepawR', 'SwSt_discrete')] == locostuff['swst_vals_2025']['st'])
            stride0_stance_frames = stride0_data[stride0_stance_mask].index.get_level_values('FrameIdx').to_numpy()
            stride0_runs = stride0_data.loc[stride0_stance_mask].index.get_level_values('Run').unique()
            assert len(stride0_runs) == len(stride0_stance_frames), "Mismatch between runs and stance frames for stride 0"
            stride_stance_byDay_times.loc[stride0_runs, 0] = stride0_stance_frames / fps # in seconds

            # Get stride -1, -2, -3 stance frames relative to stride 0
            for s in strides[1:]:
                stride_mask = mouse_stride_data.loc(axis=1)['Stride_no'] == s
                stride_s_data = mouse_stride_data.loc[stride_mask]

                stride_s_stance_mask = (stride_s_data.loc[:, ('ForepawL', 'SwSt_discrete')] == locostuff['swst_vals_2025']['st']) | (stride_s_data.loc[:, ('ForepawR', 'SwSt_discrete')] == locostuff['swst_vals_2025']['st'])
                stride_s_stance_frames = stride_s_data[stride_s_stance_mask].index.get_level_values('FrameIdx').to_numpy()
                stride_s_runs = stride_s_data.loc[stride_s_stance_mask].index.get_level_values('Run').unique()
                assert len(stride_s_runs) == len(stride_s_stance_frames), f"Mismatch between runs and stance frames for stride {s}"
                stride_stance_byDay_times.loc[stride_s_runs, s] = stride_s_stance_frames / fps  # in seconds

            # trim stride_stance_byDay_times to only APA2 runs
            apa_runs = expstuff['condition_exp_runs']['APAChar']['Extended']['APA2']
            stride_stance_byDay_times = stride_stance_byDay_times.loc[apa_runs]

            # Store stride times for this mouse
            stride_times_per_mouse[mouse] = stride_stance_byDay_times

        all_mice_stride_diffs_times = {mouse: {} for mouse in mice}
        for mouse in mice:
            stride_diffs_times = pd.DataFrame(index=np.arange(60,110), columns=strides[1:], dtype=float)
            for s in strides[1:]:
                # Calculate time differences relative to stride 0
                run = np.array(stride_times_per_mouse[mouse][s].index)
                stride_diffs_times.loc[run,s] = stride_times_per_mouse[mouse][s] - stride_times_per_mouse[mouse][0]
            all_mice_stride_diffs_times[mouse] = stride_diffs_times
        # Concatenate all mice stride differences into a single DataFrame

        all_mice_stride_diffs_times = pd.concat(all_mice_stride_diffs_times)
        all_mice_stride_diffs_times.index.names = ['MouseID', 'Run']
        mice_mean_stride_diffs_times = all_mice_stride_diffs_times.groupby(level='MouseID').mean()
        mean_stride_diffs_times = mice_mean_stride_diffs_times.mean(axis=0)
        std_stride_diffs_times = mice_mean_stride_diffs_times.std(axis=0)
        # Calculate 95% CI
        ci_stride_diffs_times = std_stride_diffs_times / np.sqrt(len(mice)) * 1.96

        # Plotting
        fig, ax = plt.subplots(figsize=(3, 2))
        for mouse in mice:
            mouse_color = pu.get_color_mice(mouse)
            mouse_marker = pu.get_marker_style_mice(mouse)
            ax.plot(mice_mean_stride_diffs_times.columns, mice_mean_stride_diffs_times.loc[mouse],
                       color=mouse_color, marker=mouse_marker, linestyle='-', alpha=0.8, markersize=3, label= mouse)

        ax.errorbar(mean_stride_diffs_times.index + 0.15, mean_stride_diffs_times.values,
                    yerr=ci_stride_diffs_times.values, fmt='o-', color='k', capsize=3,
                    elinewidth=1, markersize=4, linewidth=0, zorder=10)

        # print the mean values for each stride above the error bars
        y_max = mice_mean_stride_diffs_times.max().max() + ci_stride_diffs_times.max()
        y_min = mice_mean_stride_diffs_times.min().min() - ci_stride_diffs_times.min()
        for i, (stride, mean_time) in enumerate(mean_stride_diffs_times.items()):
            ax.text(stride + 0.15, y_max, f"mean={mean_time:.2f}", ha='center', va='bottom', fontsize=fs-1)

        y_min_rounded = np.floor(y_min * 10) / 10  # Round down to nearest 0.1
        ax.set_ylim(0, y_min_rounded)
        ax.set_xticks(strides[1:])
        ax.set_xticklabels(strides[1:], fontsize=fs)
        ax.set_xlabel('Stride', fontsize=fs)
        ax.set_ylabel('Time from Transition (s)', fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(
            fontsize=fs, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, ncol=1
        )

        fig.subplots_adjust(left=0.2, bottom=0.2, right=0.7, top=0.95)

        savepath = os.path.join(self.base_dir, 'StrideTimes')
        fig.savefig(f"{savepath}.png", bbox_inches='tight', dpi=300)
        fig.savefig(f"{savepath}.svg", bbox_inches='tight', dpi=300)




def main():
    save_dir = r"H:\Characterisation_v2\WhenAPA_corrections"
    os.path.exists(save_dir) or os.makedirs(save_dir)

    # Initialize the WhenAPA class with LH prediction data
    when_apa = WhenAPA(LH_preprocessed_data, LH_stride0_preprocessed_data, LH_pred_data, LH_pca_data, save_dir)

    # when_apa.plot_stride_times()

    # Plot the accuracy of each stride model
    when_apa.plot_accuracy_of_each_stride_model()
    when_apa.plot_corr_pc_weights_heatmap()
    when_apa.plot_corr_pcs_heatmap(pcs_to_plot=None)
    when_apa.plot_corr_pcs_heatmap(pcs_to_plot=[0,2,6])
    when_apa.plot_line_pcs_apa_vs_wash()
    when_apa.plot_pcs_timeseries_by_stride()


if __name__ == '__main__':
    main()




