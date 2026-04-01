"""Analyse how anticipatory postural adjustments develop over the course of the experiment."""
import os
import pandas as pd
import pickle
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
from scipy.stats import ttest_rel
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


from apa_analysis.config import (global_settings)
from apa_analysis.Characterisation import Plotting_utils as pu
from helpers.config import *
from helpers.utils import Utils
from apa_analysis.Characterisation.AnalysisTools import Regression as reg

condition = 'LowMid'

if condition == 'LowHigh':
    # load LH pred data
    LH_MultiFeatPath = PLOTS_ROOT + r"\LH_res_-3-2-1_APA2Wash2\APAChar_LowHigh_Extended\MultiFeaturePredictions"
    LH_preprocessed_data_file_path = PLOTS_ROOT + r"\LH_res_-3-2-1_APA2Wash2\preprocessed_data_APAChar_LowHigh.pkl"
    LH_stride_0_preprocessed_data_file_path = PLOTS_ROOT + r"\LH_LHpcsonly_res_0_APA2Wash2\preprocessed_data_APAChar_LowHigh.pkl"
    LH_pred_path = f"{LH_MultiFeatPath}\\pca_predictions_APAChar_LowHigh.pkl"
    LH_pca_path = f"{LH_MultiFeatPath}\\pca_APAChar_LowHigh.pkl"
# elif condition == 'LowMid':


with open(LH_preprocessed_data_file_path, 'rb') as f:
    data = pickle.load(f)
    LH_preprocessed_data = data['feature_data']
with open(LH_stride_0_preprocessed_data_file_path, 'rb') as f:
    data = pickle.load(f)
    LH_stride0_preprocessed_data = data['feature_data']
with open(LH_pred_path, 'rb') as f:
    LH_pred_data = pickle.load(f)
with open(LH_pca_path, 'rb') as f:
    LH_pca_data = pickle.load(f)

class Learning:
    def __init__(self, feature_data, feature_data_s0, pred_data, pca_data, save_dir, disturb_pred_file: str):
        self.LH_feature_data = feature_data
        self.LH_feature_data_s0 = feature_data_s0
        self.LH_pred = LH_pred_data
        self.LH_pca = LH_pca_data
        self.base_dir = save_dir
        self.strides = [-1, -2, -3]

        self.all_learners_learning = {}
        self.all_learners_extinction = {}
        self.fast_learning = {}
        self.slow_learning = {}
        self.fast_extinction = {}
        self.slow_extinction = {}

        with open(disturb_pred_file, 'rb') as f:
            self.disturb_pred_data = pickle.load(f)

    def get_disturb_preds(self):
        """
        Returns: { mouse_id: (x_vals, y_pred_array) }
        for stride==0 entries in self.disturb_pred_data
        """
        out = {}
        for p in self.disturb_pred_data:
            if p.stride == 0:
                x = np.array(list(p.x_vals))
                y = p.y_pred[0]
                out[p.mouse_id] = (x, y)
        return out


    def get_pcs(self, s=-1):
        pca = self.LH_pca[0].pca
        stride_feature_data = self.LH_feature_data.loc(axis=0)[s]
        mice_names = stride_feature_data.index.get_level_values('MouseID').unique()

        pcs_bymouse = {}
        for midx in mice_names:
            # Get pcs from feature data: # n runs x pcs
            mouse_data = stride_feature_data.loc[midx]
            pcs = pca.transform(mouse_data)
            pcs = pcs[:, :global_settings['pcs_to_use']]
            run_vals = mouse_data.index.get_level_values('Run').unique()
            pcs_bymouse[midx] = {'pcs': pcs, 'run_vals': run_vals}
        return pcs_bymouse

    def get_preds(self, pcwise, s=-1):
        pcs_bymouse = self.get_pcs(s=s)

        stride_feature_data = self.LH_feature_data.loc(axis=0)[s]
        mice_names = stride_feature_data.index.get_level_values('MouseID').unique()
        goal_runs = np.arange(160)

        # get pcs from feature data and pca for each mouse
        if pcwise:
            preds_byPC_bymouse = {f'PC{i + 1}': {} for i in range(global_settings['pcs_to_use'])}
        else:
            preds_byPC_bymouse = {}

        for midx in mice_names:
            pcs = pcs_bymouse[midx]['pcs']
            run_vals = pcs_bymouse[midx]['run_vals']

            # Get regression weights: 1 x pcs
            pc_pred_weights = [pred.pc_weights for pred in self.LH_pred if pred.stride == s and pred.mouse_id == midx][
                0]

            if pcwise:
                for pc_idx in range(min(global_settings['pcs_to_use'], pc_pred_weights.shape[1])):
                    pc_weights = pc_pred_weights[0][pc_idx]
                    y_pred = np.dot(pcs, pc_weights.T).squeeze()
                    y_pred_pc = y_pred[:, pc_idx]

                    y_pred_interp = np.interp(goal_runs, run_vals, y_pred_pc)

                    # normalise with max abs
                    max_abs = max(abs(y_pred_interp.min()), abs(y_pred_interp.max()))
                    y_pred_interp_norm = y_pred_interp / max_abs

                    preds_byPC_bymouse[f'PC{pc_idx + 1}'][midx] = y_pred_interp_norm
            else:
                # Get overall prediction for the mouse
                y_pred = np.dot(pcs, pc_pred_weights.T).squeeze()
                y_pred_interp = np.interp(goal_runs, run_vals, y_pred)
                # normalise with max abs
                max_abs = max(abs(y_pred_interp.min()), abs(y_pred_interp.max()))
                y_pred_interp_norm = y_pred_interp / max_abs
                preds_byPC_bymouse[midx] = y_pred_interp_norm
        return preds_byPC_bymouse

    def fit_pcwise_regression_model(self, chosen_pcs, s=-1):
        pcs_bymouse = self.get_pcs(s=s)

        mice_names = [midx for midx in pcs_bymouse.keys()]

        goal_runs = np.arange(160)

        pc_reg_models = {f'PC{pc}': {} for pc in chosen_pcs}
        for pc in chosen_pcs:
            pc_index = pc - 1  # Convert to zero-based index
            for midx in mice_names:
                current_pcs = pcs_bymouse[midx]['pcs'][:, pc_index]
                current_run_vals = pcs_bymouse[midx]['run_vals']

                apa_mask = np.isin(current_run_vals, expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'])
                wash_mask = np.isin(current_run_vals, expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2'])

                pcs_apa = current_pcs[apa_mask]
                pcs_wash = current_pcs[wash_mask]
                Xdr = np.concatenate([pcs_apa, pcs_wash])
                Xdr = Xdr.reshape(1, -1)  # Reshape to 2D array for regression

                Xdr_long = current_pcs.reshape(1, -1)  # Reshape to 2D array for regression

                y_reg = np.concatenate([np.ones_like(pcs_apa), np.zeros_like(pcs_wash)])

                null_acc_circ = reg.compute_null_accuracy_circular(Xdr_long, y_reg, apa_mask, wash_mask)

                num_folds = 10
                w, bal_acc, cv_acc, w_folds = reg.compute_regression(Xdr, y_reg, folds=num_folds)

                pred = np.dot(Xdr_long.T, w).squeeze()

                # Store the regression model for this PC and mouse
                pc_reg_models[f'PC{pc}'][midx] = {
                    'x_vals': current_run_vals,
                    'y_pred': pred,
                    'weights': w,
                    'balanced_accuracy': bal_acc,
                    'cv_accuracy': cv_acc,
                    'w_folds': w_folds,
                    'null_accuracy_circ': null_acc_circ
                }

        return pc_reg_models

    def save_pvalues_csv(self, savepath_base, comparisons, pvals, pvals_corr=None, test_type=None, extra_cols=None):
        """
        Save p-values to CSV.
        - savepath_base: Path without extension (e.g. .../plotname)
        - comparisons: List of comparison labels (e.g. ['day1_base_vs_post5', ...])
        - pvals: List of raw p-values.
        - pvals_corr: List of corrected p-values (optional).
        - test_type: List or single string describing test (optional).
        - extra_cols: dict of {colname: list}, optional extra columns.
        """
        data = {
            'comparison': comparisons,
            'pval': pvals,
        }
        if pvals_corr is not None:
            data['pval_corrected'] = pvals_corr
        if test_type is not None:
            if isinstance(test_type, list):
                data['test_type'] = test_type
            else:
                data['test_type'] = [test_type] * len(comparisons)
        if extra_cols:
            for k, v in extra_cols.items():
                data[k] = v
        df = pd.DataFrame(data)
        df.to_csv(savepath_base + ".csv", index=False)

    def plot_total_predictions_x_trial(self, fast_slow=None, s=-1, fs=7, smooth_window=3):
        # Get the total (not PC-wise) predictions for each mouse
        preds_by_mouse = self.get_preds(pcwise=False, s=s)

        # Decide which mice to plot
        speed_ordered_mice = list(self.all_learners_learning.keys())
        if fast_slow == 'fast':
            mice_names = self.fast_learning.keys()
        elif fast_slow == 'slow':
            mice_names = self.slow_learning.keys()
        else:
            mice_names = speed_ordered_mice

        goal_runs = np.arange(160)

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        pu.plot_phase_bars()

        # Loop over mice
        for midx in mice_names:
            y = preds_by_mouse[midx]
            # interpolate if you need (should already be length 160)
            # smooth
            y_smooth = median_filter(y, size=smooth_window, mode='nearest')
            # normalize
            max_abs = np.max(np.abs(y_smooth))
            y_norm = y_smooth / max_abs

            # styling
            c = pu.get_color_mice(midx, speedordered=speed_ordered_mice)
            ls = pu.get_line_style_mice(midx)

            ax.plot(goal_runs + 1, y_norm, color=c, linestyle=ls, linewidth=1, label=str(midx))

        # Formatting exactly like your other plots
        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Normalised Total Prediction', fontsize=fs)
        title = 'Total predictions'
        if fast_slow is not None:
            title += f' ({fast_slow})'
        ax.set_title(title, fontsize=fs)
        ax.set_xlim(0, 160)
        ax.set_xticks([10, 60, 110, 135, 160])
        ax.set_xticklabels(['10','60','110','135','160'], fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=fs, frameon=False)

        plt.tight_layout()

        # Save
        fname = f'TotalPreds_Stride{s}'
        if fast_slow:
            fname = f'TotalPreds_{fast_slow}_Stride{s}'
        savepath = os.path.join(self.base_dir, fname)
        plt.savefig(f"{savepath}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{savepath}.svg", dpi=300, bbox_inches='tight')
        plt.close()


    def plot_line_important_pcs_preds_x_trial(self, chosen_pcs, fast_slow=None, s=-1, fs=7, smooth_window=3):
        pc_reg_models = self.fit_pcwise_regression_model(chosen_pcs, s=s)
        goal_runs = np.arange(160)

        speed_ordered_mice = list(self.all_learners_learning.keys())

        # Select mouse IDs
        if fast_slow == 'fast':
            mice_names = self.fast_learning.keys()
        elif fast_slow == 'slow':
            mice_names = self.slow_learning.keys()
        else:
            mice_names = speed_ordered_mice

        fig, ax = plt.subplots(figsize=(6, 4))
        pu.plot_phase_bars()
        for pc in chosen_pcs:
            fig_pc, ax_pc = plt.subplots(figsize=(6, 4))
            pc_index = pc - 1  # Convert to zero-based index
            pc_data = pc_reg_models[f'PC{pc}']

            pc_preds_df = pd.DataFrame(index=goal_runs, columns=mice_names)
            for midx in mice_names:
                current_preds = pc_data[midx]['y_pred']
                current_run_vals = pc_data[midx]['x_vals']

                # Interpolate to match goal runs
                current_preds_interp = np.interp(goal_runs, current_run_vals, current_preds)
                # smooth
                current_preds_smooth = median_filter(current_preds_interp, size=smooth_window, mode='nearest')
                # normalise with max abs
                max_abs = max(abs(current_preds_smooth.min()), abs(current_preds_smooth.max()))
                current_preds_norm = current_preds_smooth / max_abs

                mouse_color = pu.get_color_mice(midx, speedordered=speed_ordered_mice)
                mouse_ls = pu.get_line_style_mice(midx)

                ax_pc.plot(goal_runs + 1, current_preds_norm, color=mouse_color, linestyle=mouse_ls, linewidth=1, label=f'PC{pc} - {midx}')

                pc_preds_df[midx] = current_preds_norm

            # Format and save individual PC plot
            ax_pc.set_xlabel('Trial number', fontsize=fs)
            ax_pc.set_ylabel('Normalised Prediction', fontsize=fs)
            ax_pc.set_title(f'PC{pc} predictions', fontsize=fs)
            ax_pc.set_xlim(0, 160)
            ax_pc.set_xticks([10, 60, 110, 135, 160])
            ax_pc.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
            ax_pc.tick_params(axis='both', which='major', labelsize=fs)
            ax_pc.spines['top'].set_visible(False)
            ax_pc.spines['right'].set_visible(False)
            ax_pc.legend(fontsize=fs, frameon=False)
            plt.tight_layout()
            # --- Save individual PC plot ---
            if fast_slow is not None:
                savepath = os.path.join(self.base_dir, f'PC{pc}_preds_{fast_slow}_Stride{s}')
            else:
                savepath = os.path.join(self.base_dir, f'PC{pc}_preds_Stride{s}')
            plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
            plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
            plt.close(fig_pc)

            # --- compute mean and 95% CI across mice ---
            n_mice = pc_preds_df.shape[1]
            mean_series = pc_preds_df.mean(axis=1)
            sem_series = pc_preds_df.std(axis=1, ddof=1) / np.sqrt(n_mice)
            ci_mult = t.ppf(0.975, df=n_mice - 1)  # two-tailed 95%
            ci_series = sem_series * ci_mult

            pc_color = pu.get_color_pc(pc_index)
            # plot shaded CI
            ax.fill_between(goal_runs + 1,
                            mean_series - ci_series,
                            mean_series + ci_series,
                            color=pc_color,
                            alpha=0.08,
                            linewidth=0)
            # plot the mean line
            ax.plot(goal_runs + 1,
                    mean_series,
                    color=pc_color,
                    linewidth=1,
                    label=f'PC{pc}')

        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Normalised Prediction', fontsize=fs)
        ax.set_xlim(0, 160)
        ax.set_xticks([10, 60, 110, 135, 160])
        ax.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=fs, frameon=False)
        plt.tight_layout()
        # --- Save ---
        if fast_slow is not None:
            savepath = os.path.join(self.base_dir, f'PC_preds_{chosen_pcs}_{fast_slow}_Stride{s}')
        else:
            savepath = os.path.join(self.base_dir, f'PC_preds_{chosen_pcs}_Stride{s}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()




    def plot_line_important_pcs_x_trial(self, chosen_pcs, fast_slow=None, s=-1, fs=7, smooth_window=15):
        pcs_bymouse = self.get_pcs(s=s)

        speed_ordered_mice = list(self.all_learners_learning.keys())

        # Select mouse IDs
        if fast_slow == 'fast':
            mice_names = self.fast_learning.keys()
        elif fast_slow == 'slow':
            mice_names = self.slow_learning.keys()
        else:
            mice_names = [midx for midx in speed_ordered_mice]

        goal_runs = np.arange(160)

        fig, ax = plt.subplots(figsize=(3, 3))
        pu.plot_phase_bars()

        for pc in chosen_pcs:
            fig_pc, ax_pc = plt.subplots(figsize=(4, 4))

            pc_index = pc - 1  # Convert to zero-based index
            pcs_df = pd.DataFrame(columns=goal_runs, index=mice_names)
            for midx in mice_names:
                current_pcs = pcs_bymouse[midx]['pcs'][:, pc_index]
                current_run_vals = pcs_bymouse[midx]['run_vals']

                # Interpolate to match goal runs
                current_pcs_interp = np.interp(goal_runs, current_run_vals, current_pcs)
                # smooth
                current_pcs_smooth = median_filter(current_pcs_interp, size=smooth_window, mode='nearest')
                # normalise with max abs
                max_abs = max(abs(current_pcs_smooth.min()), abs(current_pcs_smooth.max()))
                current_pcs_norm = current_pcs_smooth / max_abs
                pcs_df.loc(axis=0)[midx] = current_pcs_norm

                mouse_color = pu.get_color_mice(midx, speedordered=speed_ordered_mice)
                mouse_ls = pu.get_line_style_mice(midx)
                ax_pc.plot(goal_runs + 1, current_pcs_norm, color=mouse_color, linestyle=mouse_ls, linewidth=1, label=f'PC{pc} - {midx}')

            # Format and save individual PC plot
            ax_pc.set_xlabel('Trial number', fontsize=fs)
            ax_pc.set_ylabel('Normalised PC', fontsize=fs)
            ax_pc.set_title(f'PC{pc} values', fontsize=fs)
            ax_pc.set_xlim(0, 160)
            ax_pc.set_xticks([10, 60, 110, 135, 160])
            ax_pc.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
            ax_pc.tick_params(axis='both', which='major', labelsize=fs)
            ax_pc.spines['top'].set_visible(False)
            ax_pc.spines['right'].set_visible(False)
            ax_pc.legend(fontsize=fs, frameon=False)
            plt.tight_layout()
            # --- Save individual PC plot ---
            if fast_slow is not None:
                savepath = os.path.join(self.base_dir, f'PC{pc}_vals_{fast_slow}_Stride{s}')
            else:
                savepath = os.path.join(self.base_dir, f'PC{pc}_vals_Stride{s}')
            plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
            plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
            plt.close(fig_pc)

            # --- compute mean and 95% CI across mice ---
            n_mice = pcs_df.shape[0]
            mean_vals = pcs_df.mean(axis=0)
            sem_vals = pcs_df.std(axis=0, ddof=1) / np.sqrt(n_mice)
            ci_mult = t.ppf(0.975, df=n_mice - 1)
            ci_vals = sem_vals * ci_mult

            pc_color = pu.get_color_pc(pc, chosen_pcs=True)
            # shaded CI
            if pc == 3 or pc ==7: # flip y values upside down
                mean_vals = -mean_vals
                ci_vals = -ci_vals
            ax.fill_between(goal_runs + 1,
                            mean_vals - ci_vals,
                            mean_vals + ci_vals,
                            color=pc_color,
                            alpha=0.08,
                            linewidth=0)
            # mean line
            label = f'PC{pc}' if pc == 1 else f'PC{pc} flipped'

            ax.plot(goal_runs + 1,
                    mean_vals,
                    color=pc_color,
                    linewidth=0.5,
                    label=f'PC{pc}')

        apa1_color = pu.get_color_phase('APA1')
        wash1_color = pu.get_color_phase('Wash1')
        for daybreak in [40, 80, 120]:
            ax.axvline(daybreak, color='grey', linestyle='--', linewidth=0.5)
        ax.axvline(10, color=apa1_color, linestyle='-', linewidth=0.5)
        ax.axvline(110, color=wash1_color, linestyle='-', linewidth=0.5)

        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Normalised PC', fontsize=fs)
        ax.set_ylim(-1,1)
        ax.set_yticks(np.arange(-1, 1.1, 0.5))
        ax.set_yticklabels(np.arange(-1, 1.1, 0.5), fontsize=fs)
        ax.set_xlim(0, 160)
        ax.set_xticks([0, 10, 60, 110, 135, 160])
        ax.set_xticklabels(['0', '10', '60', '110', '135', '160'], fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=fs, frameon=False)
        plt.tight_layout()
        # --- Save ---
        if fast_slow is not None:
            savepath = os.path.join(self.base_dir, f'PC_vals_{chosen_pcs}_{fast_slow}_Stride{s}')
        else:
            savepath = os.path.join(self.base_dir, f'PC_vals_{chosen_pcs}_Stride{s}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()





    def plot_line_important_pc_predictions_x_trial(self, chosen_pcs, s=-1, fs=7, smooth_window=15):
        preds_byPC_bymouse = self.get_preds(pcwise=True, s=s)

        fig, ax = plt.subplots(figsize=(6, 4))
        pu.plot_phase_bars()

        for idx, pc_idx in enumerate(chosen_pcs):
            pc_name = f'PC{pc_idx}'
            preds_all_mice = preds_byPC_bymouse.get(pc_name, {})

            # Convert to DataFrame for easy mean calculation
            preds_df = pd.DataFrame.from_dict(preds_all_mice, orient='index')  # rows=mice, cols=runs

            # Calculate mean across mice (ignore NaNs)
            preds_mean = preds_df.mean(axis=0)

            # Smooth
            #preds_mean_smooth = pd.Series(preds_mean).rolling(window=smooth_window, center=True, min_periods=1).mean()

            # Smooth median filter
            preds_mean_smooth = median_filter(preds_mean, size=smooth_window, mode='nearest')

            # --- Plot ---
            pc_color = pu.get_color_pc(pc_idx - 1, n_pcs=global_settings['pcs_to_use'])
            ax.plot(np.arange(160)[10:], preds_mean_smooth[10:], color=pc_color, linewidth=1, label=pc_name)

        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Normalised Prediction', fontsize=fs)
        ax.set_title(f'Smooth window={smooth_window}', fontsize=fs)
        ax.set_xlim(0, 160)
        ax.set_xticks([10,60,110,135,160])
        ax.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=fs, frameon=False)

        plt.tight_layout()

        # --- Save ---
        savepath = os.path.join(self.base_dir, f'Predictions_perPC_{chosen_pcs}_Stride{s}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

    def find_learned_trials(self, smoothing=None, phase='learning', fast_threshold=15, learned_threshold=5, fs=7):
        preds = self.get_preds(pcwise=False, s=-1)
        if phase == 'learning':
            phase_mask = np.isin(np.arange(160), expstuff['condition_exp_runs']['APAChar']['Extended']['APA'])
        elif phase == 'extinction':
            phase_mask = np.isin(np.arange(160), expstuff['condition_exp_runs']['APAChar']['Extended']['Washout'])
        phase_preds = {
            mouse_id: preds[mouse_id][phase_mask] for mouse_id in preds.keys()
        }

        fig, ax = plt.subplots(figsize=(15, 10))
        # Plot predictions for each mouse
        smooth_window = 3
        for mouse_id, pred in phase_preds.items():
            smooth_pred = median_filter(pred, size=smooth_window, mode='nearest')
            ax.plot(np.arange(len(pred))+1, smooth_pred, label=mouse_id, marker='o')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Smoothed Prediction', fontsize=fs)
        ax.set_title(f'Smooth window={smooth_window}', fontsize=fs)
        savepath = os.path.join(self.base_dir, f'SMOOTHED_Predictions_{phase}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

        # Find trial where prediction is above 0 for threshold runs
        learned_trials_xMice = {}
        for mouse_id, pred in phase_preds.items():
            if smoothing:
                # Apply smoothing
                pred = median_filter(pred, size=smoothing, mode='nearest')
            if phase == 'learning':
                learned_trials = np.where(pred > 0)[0] + 1  # +1 to convert from index to trial number
            elif phase == 'extinction':
                learned_trials = np.where(pred < 0)[0] + 1
            learned_blocks = Utils().find_blocks(learned_trials, gap_threshold=1, block_min_size=learned_threshold)
            if len(learned_blocks) > 0:
                learned_trial = learned_blocks[0][0]  # Take the first block's first trial
                learned_trials_xMice[mouse_id] = learned_trial
            else:
                learned_trials_xMice[mouse_id] = None  # No plateau found for this mouse

        setattr(self, f'learned_trials_{phase}', learned_trials_xMice)

        # pick out fast and slow learners relative to 'fast_threshold'
        fast_learners = {mouse_id: trial for mouse_id, trial in learned_trials_xMice.items() if trial is not None and trial <= fast_threshold}
        slow_learners = {mouse_id: trial for mouse_id, trial in learned_trials_xMice.items() if trial is not None and trial > fast_threshold}

        # get top 3
        fast_learners_sorted = dict(sorted(fast_learners.items(), key=lambda item: item[1]))
        slow_learners_sorted = dict(sorted(slow_learners.items(), key=lambda item: item[1]))
        all_learners_sorted = dict(sorted(learned_trials_xMice.items(), key=lambda item: item[1]))

        fast_learners_top3 = {k: v for k, v in list(fast_learners_sorted.items())[-3:]}
        slow_learners_top3 = {k: v for k, v in list(slow_learners_sorted.items())[-3:]}

        setattr(self, f'fast_{phase}', fast_learners_top3)
        setattr(self, f'slow_{phase}', slow_learners_top3)
        setattr(self, f'all_learners_{phase}', all_learners_sorted)

        # --- Prepare ordered table data with group labels merged ---
        table_rows = []

        # Fast learners block
        fast_mouse_ids = list(fast_learners_sorted.keys())
        for idx, mouse_id in enumerate(fast_mouse_ids):
            table_rows.append([mouse_id, fast_learners_sorted[mouse_id]])

        # Slow learners block
        slow_mouse_ids = list(slow_learners_sorted.keys())
        for idx, mouse_id in enumerate(slow_mouse_ids):
            table_rows.append([mouse_id, slow_learners_sorted[mouse_id]])

        # --- Plot table ---
        fig, ax = plt.subplots(figsize=(4, 2 + len(table_rows) * 0.2))
        ax.axis('off')

        table = ax.table(cellText=table_rows,
                         colLabels=['Mouse ID', f'Trials to {phase.capitalize()}'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(fs)
        plt.tight_layout()

        # --- Save ---
        savepath = os.path.join(self.base_dir, f'LearnedTrialsTable_{phase}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_pc_correlations_heatmap(self, pcs=[1, 3, 7], s=-1, fs=7):
        """
        Plot correlation between pairs of PCs (e.g. PC1 vs PC3, PC1 vs PC7, PC3 vs PC7),
        separately for APA2 and Wash2 phases.
        Only lower triangle including diagonal is plotted (upper triangle left blank).

        Args:
            pcs (list): list of PCs to include in correlations
            fs (int): font size for labels
        """
        pcs_sorted = sorted(pcs)
        n = len(pcs_sorted)

        fig, axes = plt.subplots(n, n, figsize=(1.5 * n, 1.5 * n), sharex=True, sharey=True)

        pcs_bymouse = self.get_pcs(s=s)
        mice = list(pcs_bymouse.keys())

        phases = {
            'APA2': expstuff['condition_exp_runs']['APAChar']['Extended']['APA2'],
            'Wash2': expstuff['condition_exp_runs']['APAChar']['Extended']['Wash2']
        }

        comp_labels = []
        pvals = []
        for i, y_pc in enumerate(pcs_sorted):
            for j, x_pc in enumerate(pcs_sorted):
                ax = axes[i, j]

                if j > i:
                    # Upper triangle: hide axes and remove ticks
                    ax.axis('off')
                    continue

                if x_pc == y_pc:
                    # # Diagonal: label PC number
                    # ax.text(0.5, 0.5, f'PC{x_pc}', ha='center', va='center', fontsize=fs * 1.5)
                    # ax.set_xticks([])
                    # ax.set_yticks([])
                    # ax.spines['top'].set_visible(False)
                    # ax.spines['right'].set_visible(False)
                    # ax.spines['bottom'].set_visible(False)
                    # ax.spines['left'].set_visible(False)
                    ax.axis('off')
                    continue

                corrs_phase = {phase: [] for phase in phases.keys()}

                for mouse in mice:
                    mouse_data = pcs_bymouse[mouse]['pcs']
                    mouse_runs = pcs_bymouse[mouse]['run_vals']

                    for phase_name, phase_runs in phases.items():
                        phase_mask = np.isin(mouse_runs, phase_runs)

                        x_vals = mouse_data[phase_mask, x_pc - 1]
                        y_vals = mouse_data[phase_mask, y_pc - 1]

                        if len(x_vals) > 1 and len(y_vals) > 1:
                            r, _ = pearsonr(x_vals, y_vals)
                            corrs_phase[phase_name].append(r)
                        else:
                            corrs_phase[phase_name].append(np.nan)

                apa_corrs = np.array(corrs_phase['APA2'])
                wash_corrs = np.array(corrs_phase['Wash2'])

                apa_mean = np.nanmean(apa_corrs)
                wash_mean = np.nanmean(wash_corrs)

                apa_std = np.nanstd(apa_corrs)
                apa_ci = 1.96 * apa_std / np.sqrt(len(apa_corrs))
                wash_std = np.nanstd(wash_corrs)
                wash_ci = 1.96 * wash_std / np.sqrt(len(wash_corrs))


                x_positions = [0, 1]
                jitter_strength = 0.05
                ax.scatter(np.random.normal(x_positions[0], jitter_strength, size=len(apa_corrs)), apa_corrs,
                           color=pu.get_color_phase('APA2'), alpha=0.7, label='APA2' if (i == n - 1 and j == 0) else "",
                           s=3)
                ax.scatter(np.random.normal(x_positions[1], jitter_strength, size=len(wash_corrs)), wash_corrs,
                           color=pu.get_color_phase('Wash2'), alpha=0.7,
                           label='Wash2' if (i == n - 1 and j == 0) else "",
                           s=3)

                # plot mean and error bars
                ax.errorbar(0, apa_mean, yerr=apa_ci, fmt='o', color='k', capsize=5, label='APA2 mean' if (i == n - 1 and j == 0) else "", ms=3)
                ax.errorbar(1, wash_mean, yerr=wash_ci, fmt='o', color='k', capsize=5, label='Wash2 mean' if (i == n - 1 and j == 0) else "", ms=3)

                # check significance between APA2 and Wash2
                tstat, pval = ttest_rel(apa_corrs, wash_corrs, nan_policy='omit')
                if pval < 0.001:
                    significance = '***'
                elif pval < 0.01:
                    significance = '**'
                elif pval < 0.05:
                    significance = '*'
                else:
                    significance = 'ns'

                # plot significance with lines across the two points
                max_y = max(apa_corrs.max(), wash_corrs.max(), apa_mean + apa_ci, wash_mean + wash_ci)
                sig_y = max_y + 0.05 * (max_y - min(apa_corrs.min(), wash_corrs.min()))
                ax.plot([0, 1], [sig_y, sig_y], color='k', linewidth=1)
                ax.text(0.5, sig_y + 0.02, significance, ha='center', va='bottom', fontsize=fs)

                ax.set_xlim(-0.5, 1.5)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(['APA2', 'Wash2'], fontsize=fs)
                ax.set_ylim(-1, 1)
                ax.tick_params(axis='both', which='major', labelsize=fs)

                if j == 0:
                    ax.set_ylabel(f'PC{y_pc} corr', fontsize=fs)
                if i == n - 1:
                    ax.set_xlabel(f'PC{x_pc}', fontsize=fs)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                comp_labels.append(f'APA2_vs_Wash2_PC{y_pc}_vs_PC{x_pc}')
                pvals.append(pval)  # pval already calculated in your code

        # Add legend only once in lower-left subplot
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label='APA2', markerfacecolor=pu.get_color_phase('APA2'),
                       markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Wash2', markerfacecolor=pu.get_color_phase('Wash2'),
                       markersize=8),
        ]
        fig.legend(handles=handles, loc='lower left', fontsize=fs, frameon=False, bbox_to_anchor=(0.4, 0.5))

        plt.tight_layout(rect=[0, 0, 0.95, 1])

        savepath = os.path.join(self.base_dir, 'PC_correlations_heatmap')
        plt.savefig(f"{savepath}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{savepath}.svg", dpi=300, bbox_inches='tight')

        self.save_pvalues_csv(
            savepath_base=savepath,
            comparisons=comp_labels,
            pvals=pvals,
            test_type='paired t-test'
        )

        plt.close()

    def plot_fast_vs_slow_learners_pcs(self, fast_slow, chosen_pcs, s=-1, fs=7, smooth_window=15):
        preds_byPC_bymouse = self.get_preds(pcwise=True, s=s)

        # Select mouse IDs
        if fast_slow == 'fast':
            selected_mice = self.fast_learning.keys()
        elif fast_slow == 'slow':
            selected_mice = self.slow_learning.keys()
        else:
            raise ValueError("fast_slow must be 'fast' or 'slow'")

        fig, ax = plt.subplots(figsize=(6, 4))

        apa1_color = pu.get_color_phase('APA1')
        apa2_color = pu.get_color_phase('APA2')
        wash1_color = pu.get_color_phase('Wash1')
        wash2_color = pu.get_color_phase('Wash2')
        boxy = 1
        height = 0.02
        patch1 = plt.axvspan(xmin=9.5, xmax=59.5, ymin=boxy, ymax=boxy + height, color=apa1_color, lw=0)
        patch2 = plt.axvspan(xmin=59.5, xmax=109.5, ymin=boxy, ymax=boxy + height, color=apa2_color, lw=0)
        patch3 = plt.axvspan(xmin=109.5, xmax=134.5, ymin=boxy, ymax=boxy + height, color=wash1_color, lw=0)
        patch4 = plt.axvspan(xmin=134.5, xmax=159.5, ymin=boxy, ymax=boxy + height, color=wash2_color, lw=0)
        patch1.set_clip_on(False)
        patch2.set_clip_on(False)
        patch3.set_clip_on(False)
        patch4.set_clip_on(False)

        goal_runs = np.arange(160)

        for idx, pc_idx in enumerate(chosen_pcs):
            pc_name = f'PC{pc_idx}'

            # Extract predictions only for selected mice
            preds_selected_mice = {mouse: preds for mouse, preds in preds_byPC_bymouse[pc_name].items() if
                                   mouse in selected_mice}

            # Convert to DataFrame for easy mean calculation
            preds_df = pd.DataFrame.from_dict(preds_selected_mice, orient='index')

            # Calculate mean across mice (ignore NaNs)
            preds_mean = preds_df.mean(axis=0)

            # Smooth with median filter
            preds_mean_smooth = median_filter(preds_mean, size=smooth_window, mode='nearest')

            # Plot
            pc_color = pu.get_color_pc(pc_idx - 1, n_pcs=global_settings['pcs_to_use'])
            ax.plot(goal_runs[10:], preds_mean_smooth[10:], color=pc_color, linewidth=1, label=pc_name)

        ax.set_xlabel('Trial number', fontsize=fs)
        ax.set_ylabel('Normalised Prediction', fontsize=fs)
        ax.set_title(f'{fast_slow.capitalize()} learners (smooth window={smooth_window})',
                     fontsize=fs)
        ax.set_xlim(0, 160)
        ax.set_xticks([10, 60, 110, 135, 160])
        ax.set_xticklabels(['10', '60', '110', '135', '160'], fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=fs, frameon=False)

        plt.tight_layout()

        # --- Save ---
        savepath = os.path.join(self.base_dir, f'Predictions_{fast_slow}_learners_perPC_{chosen_pcs}_Stride{s}')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_prediction_delta(self):
        preds_byPC_bymouse = self.get_preds(pcwise=False, s=-1)

        # Calculate deltas
        delta_preds = {}
        for mouse_id, preds in preds_byPC_bymouse.items():
            smooth_preds = median_filter(preds, size=10, mode='nearest')
            delta_preds[mouse_id] = np.diff(smooth_preds)

        delta_df = pd.DataFrame.from_dict(delta_preds, orient='index')

        # plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for mouse_id, delta in delta_df.iterrows():
            ax.plot(np.arange(len(delta)) + 1, delta, label=mouse_id, marker='o')
        plt.close()


    def plot_prediction_per_day(self, fs=7):
        preds_byPC_bymouse = self.get_preds(pcwise=False, s=-1)

        # smooth preds
        smooth_window = 3

        # Convert to DataFrame for easy plotting
        preds_df = pd.DataFrame.from_dict(preds_byPC_bymouse, orient='index')
        preds_df.index.name = 'Mouse ID'

        day_chunk_size = 5
        chunk_dividers = [10, 40, 80, 110, 120]
        day_runs = [np.arange(chunk - day_chunk_size, chunk + day_chunk_size * 2) for chunk in chunk_dividers]
        day_dividers = [40, 80, 120]

        apa1_color = pu.get_color_phase('APA1')
        apa2_color = pu.get_color_phase('APA2')
        wash1_color = pu.get_color_phase('Wash1')
        wash2_color = pu.get_color_phase('Wash2')

        speed_ordered_mice = list(self.all_learners_learning.keys())

        fig2, ax2 = plt.subplots(figsize=(8, 3)) #was 5,3

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

        all_chunk_means = {}
        post5_means_days = {}

        # Collect all baseline vs post5 and baseline vs post10 p-values to correct together
        pvals_baseline = []
        # Store raw p-values to map back after correction
        chunk_pvals = []

        for day_idx, day in enumerate(day_runs):
            data = preds_df.loc(axis=1)[day]
            data = data.apply(lambda x: median_filter(x, size=smooth_window, mode='nearest'))

            baseline_window = day[:day_chunk_size]
            post5_window = day[day_chunk_size:day_chunk_size * 2]
            post10_window = day[day_chunk_size * 2:day_chunk_size * 3]

            baseline_means = data.loc[:, baseline_window].mean(axis=1)
            post5_means = data.loc[:, post5_window].mean(axis=1)
            post5_means_days[day_idx] = post5_means
            post10_means = data.loc[:, post10_window].mean(axis=1)

            all_chunk_means[day_idx] = (baseline_means, post5_means, post10_means)

            # Significance tests
            tstat_a, pval_a = ttest_rel(baseline_means, post5_means)
            tstat_b, pval_b = ttest_rel(baseline_means, post10_means)

            pvals_baseline.extend([pval_a, pval_b])
            chunk_pvals.append((pval_a, pval_b))

            print(f"Chunk {day_idx + 1} comparison a (-5 vs +5): p={pval_a:.3f}")
            print(f"Chunk {day_idx + 1} comparison b (-5 vs +6-10): p={pval_b:.3f}")

        # Apply Holm correction for all baseline comparisons (post5 and post10 together)
        reject_baseline, pvals_baseline_corr, _, _ = multipletests(pvals_baseline, alpha=0.05, method='fdr_bh')
        # Map corrected p-values back to chunks
        corrected_chunk_pvals = [(pvals_baseline_corr[i * 2], pvals_baseline_corr[i * 2 + 1]) for i in
                                 range(len(day_runs))]

        # Post5 between chunks comparisons
        post5_means_df = pd.DataFrame(post5_means_days)
        post5_pairs = [(0, 1), (0, 2), (1, 2)]
        pvals_post5_between = []
        for pair in post5_pairs:
            tstat, pval = ttest_rel(post5_means_df.loc(axis=1)[pair[0]], post5_means_df.loc(axis=1)[pair[1]])
            pvals_post5_between.append(pval)
        reject_post5_between, pvals_post5_between_corr, _, _ = multipletests(pvals_post5_between, alpha=0.05,
                                                                             method='fdr_bh')

        # Now proceed to plotting, using corrected p-values

        for day_idx, day in enumerate(day_runs):
            baseline_means, post5_means, post10_means = all_chunk_means[day_idx]
            x_baseline = day[:day_chunk_size].mean() + 1
            x_post5 = day[day_chunk_size:day_chunk_size * 2].mean() + 1
            x_post10 = day[day_chunk_size * 2:day_chunk_size * 3].mean() + 1

            xpos = np.array([x_baseline, x_post5, x_post10])
            data_to_plot = [baseline_means, post5_means, post10_means]

            xpos_counter = 0
            for x, d in zip(xpos, data_to_plot):
                for mouse in speed_ordered_mice:
                    marker = pu.get_marker_style_mice(mouse)
                    color = pu.get_color_mice(mouse, speedordered=speed_ordered_mice)
                    label = mouse if day_idx == 0 and xpos_counter < len(speed_ordered_mice) else ""
                    ax2.scatter([x], d[mouse], color=color, marker=marker, s=15, alpha=0.6, linewidth=0, label=label)
                    xpos_counter += 1

            means = [d.mean() for d in data_to_plot]
            ax2.plot(xpos, means, color='k', linewidth=1)

            # Add brackets and stars with corrected p-values
            y_max = max([d.max() for d in data_to_plot]) + 0.05
            line_height = 0.02

            # baseline vs post5
            pval_a_corr = corrected_chunk_pvals[day_idx][0]
            if pval_a_corr < 0.001:
                stars = '***'
            elif pval_a_corr < 0.01:
                stars = '**'
            elif pval_a_corr < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'
            ax2.plot([x_baseline, x_baseline, x_post5, x_post5],
                     [y_max, y_max + line_height, y_max + line_height, y_max],
                     lw=0.8, c='k')
            ax2.text((x_baseline + x_post5) / 2, y_max + line_height + 0.01, stars, ha='center', fontsize=fs)

            # baseline vs post10
            pval_b_corr = corrected_chunk_pvals[day_idx][1]
            if pval_b_corr < 0.001:
                stars = '***'
            elif pval_b_corr < 0.01:
                stars = '**'
            elif pval_b_corr < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'
            y_max_b = y_max + 0.07
            ax2.plot([x_baseline, x_baseline, x_post10, x_post10],
                     [y_max_b, y_max_b + line_height, y_max_b + line_height, y_max_b],
                     lw=0.8, c='k')
            ax2.text((x_baseline + x_post10) / 2, y_max_b + line_height + 0.01, stars, ha='center', fontsize=fs)

        # Plot post5-between-chunks comparisons with corrected p-values
        for idx, (pair, pval_corr) in enumerate(zip(post5_pairs, pvals_post5_between_corr)):
            if pval_corr < 0.001:
                stars = '***'
            elif pval_corr < 0.01:
                stars = '**'
            elif pval_corr < 0.05:
                stars = '*'
            else:
                stars = 'n.s.'

            y_max = max(post5_means_df.max()) + 0.15 * (idx + 1)
            x1 = day_runs[pair[0]][5:10].mean() + 1
            x2 = day_runs[pair[1]][5:10].mean() + 1

            ax2.plot([x1, x1, x2, x2],
                     [y_max, y_max + 0.02, y_max + 0.02, y_max],
                     lw=0.8, c='r')
            ax2.text((x1 + x2) / 2, y_max + 0.03, stars, ha='center', fontsize=fs, color='r')

        ax2.vlines(x=day_dividers, ymin=-1, ymax=1, color='grey', linestyle='--', linewidth=0.5)
        ax2.vlines(x=10, ymin=-1, ymax=1, color=apa1_color, linestyle='-', linewidth=0.5)
        ax2.vlines(x=110, ymin=-1, ymax=1, color=wash1_color, linestyle='-', linewidth=0.5)

        # Finalise plot
        ax2.set_xlabel('Trial number', fontsize=fs)
        ax2.set_ylabel('Mean Prediction', fontsize=fs)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlim(0, 160)
        ax2.set_xticks([0, 10, 40, 60, 80, 110, 120, 135, 160])
        ax2.set_ylim(-1, 1)
        ax2.tick_params(axis='both', which='major', labelsize=fs)
        ax2.legend(fontsize=fs, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

        save_path2 = os.path.join(self.base_dir, 'Predictions_WindowMeans_Significance_ScatterTrueX')
        plt.savefig(f"{save_path2}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{save_path2}.svg", format='svg', bbox_inches='tight', dpi=300)

        # Build comparison labels for within-day (baseline vs post5/post10)
        within_labels = []
        for day_idx in range(len(day_runs)):
            within_labels.append(f'day{day_idx + 1}_baseline_vs_post5')
            within_labels.append(f'day{day_idx + 1}_baseline_vs_post10')

        # Build labels for between-day comparisons (post5 between chunks)
        between_labels = []
        for idx, pair in enumerate(post5_pairs):
            between_labels.append(f'post5_day{pair[0] + 1}_vs_day{pair[1] + 1}')

        # Concatenate all info
        all_labels = within_labels + between_labels
        all_pvals = pvals_baseline + pvals_post5_between
        all_pvals_corr = list(pvals_baseline_corr) + list(pvals_post5_between_corr)
        all_types = ['paired t-test'] * len(within_labels) + ['paired t-test'] * len(between_labels)

        # Save to CSV (save_path is your plot file base name, e.g. .../Predictions_WindowMeans_Significance_Scatter_PC1)
        self.save_pvalues_csv(
            savepath_base=save_path2,
            comparisons=all_labels,
            pvals=all_pvals,
            pvals_corr=all_pvals_corr,
            test_type=all_types,
        )

        plt.close()

    def plot_learning_by_extinction_scatter(self, fs=7):
        learning_trials = getattr(self, 'learned_trials_learning', {})
        extinction_trials = getattr(self, 'learned_trials_extinction', {})

        speed_ordered_mice = list(self.all_learners_learning.keys())

        learn_x_extinct_df = pd.DataFrame({
            'Mouse ID': list(learning_trials.keys()),
            'Learning Trial': list(learning_trials.values()),
            'Extinction Trial': [extinction_trials.get(mouse, np.nan) for mouse in learning_trials.keys()]
        })

        # sort into speed_ordered_mice order
        learn_x_extinct_df = learn_x_extinct_df.set_index('Mouse ID').reindex(speed_ordered_mice).reset_index()
        # make MouseID the index
        learn_x_extinct_df.set_index('Mouse ID', inplace=True)

        learn_extinct_diff = learn_x_extinct_df['Extinction Trial'] - learn_x_extinct_df['Learning Trial']
        _, p = ttest_1samp(learn_extinct_diff, 0)

        fig, (ax, ax_diff) = plt.subplots(1, 2, figsize=(4, 3), gridspec_kw={'width_ratios': [3, 1]})

        for mouse_id in learn_x_extinct_df.index:
            grp = learn_x_extinct_df.loc[mouse_id]
            mkr = pu.get_marker_style_mice(mouse_id)
            col = pu.get_color_mice(mouse_id, speedordered=speed_ordered_mice)
            ax.scatter(
                grp['Learning Trial'],
                grp['Extinction Trial'],
                marker=mkr,
                s=35,
                color=col,
                label=str(mouse_id),
                linewidth=0,
            )

        # plot equality line to show where learning and extinction trials are equal given unequal scale
        max_val = min(learn_x_extinct_df['Learning Trial'].max(), learn_x_extinct_df['Extinction Trial'].max())
        ax.plot([0, max_val], [0, max_val], color='grey', linestyle='--', linewidth=0.5)

        ax.set_xlabel('Learning Time (Trials)', fontsize=fs)
        ax.set_ylabel('Extinction Time (Trials)', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.legend(fontsize=fs, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(0, 5 * round(xmax/5))
        ax.set_ylim(0, 5 * round(ymax/5))
        ax.set_xticks(np.arange(0, xmax + 1, 20))
        ax.set_yticks(np.arange(0, ymax + 1, 5))

        # Scatter plot difference (right)
        for mouse in learn_extinct_diff.index:
            diff = learn_extinct_diff[mouse]
            mkr = pu.get_marker_style_mice(mouse)
            col = pu.get_color_mice(mouse, speedordered=speed_ordered_mice)
            jitter = np.random.normal(0, 0.01, size=1)  # small jitter for visibility
            ax_diff.scatter(1 + jitter, diff, marker=mkr, s=35, edgecolor=col, facecolor='none', linewidth=1)
        # plot mean difference and 95% CI
        mean_diff = learn_extinct_diff.mean()
        ci = 1.96 * learn_extinct_diff.std() / np.sqrt(len(learn_extinct_diff))
        ax_diff.errorbar(
            [1], mean_diff, yerr=ci, fmt='o', color='black', markersize=5, capsize=3, label='Mean ± 95% CI', elinewidth=0.5
        )

        # Format difference plot
        ax_diff.set_xlim(0.95, 1.05)

        ymin, ymax = ax_diff.get_ylim()
        ymin_rounded = 5 * round(ymin / 5)
        ymax_rounded = 5 * round(ymax / 5)
        ax_diff.set_ylim(ymin_rounded, ymax_rounded)
        ax_diff.set_yticks(np.arange(ymin_rounded, ymax_rounded + 1, 10))

        ax_diff.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax_diff.set_ylabel('Extinction - Learning Trials', fontsize=fs)
        ax_diff.tick_params(axis='x', labelsize=fs)
        ax_diff.tick_params(axis='y', labelsize=fs)
        ax_diff.spines['top'].set_visible(False)
        ax_diff.spines['right'].set_visible(False)

        # Set x-ticks as mouse ids for clarity, but you can tweak if too crowded
        ax_diff.set_xticks([1])
        ax_diff.set_xticklabels(['Diff'], fontsize=fs)

        # Add significance text
        if p < 0.05:
            # Convert p-value to stars
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            else:
                stars = '*'
            sig_text = stars
        else:
            sig_text = 'n.s.'

        # Place significance text above scatter plot
        ylim = ax_diff.get_ylim()
        ax_diff.text(
            0.5, 1.1, sig_text,
            ha='center', va='top', fontsize=fs, transform=ax_diff.transAxes
        )

        plt.tight_layout()

        savepath = os.path.join(self.base_dir, 'Learning_vs_Extinction_Scatter')
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)

        self.save_pvalues_csv(
            savepath_base=savepath,
            comparisons=['Extinction_minus_Learning_mean'],
            pvals=[p],
            test_type='1-sample t-test'
        )

        plt.close()

    def plot_disturbance_by_prediction_interpolated(self,
                                                    phase: str = 'APA2',
                                                    desired_diameter: int = 3):
        """
        One plot of Low vs High APA tertiles → mean disturbance,
        coloring each mouse by fast (blue), slow (red), or other (gray).
        """
        # 1) grab APA & disturbance preds
        apa_preds = self.get_preds(pcwise=False, s=-1)  # {mouse: 160-array}
        disturb_preds = self.get_disturb_preds()  # {mouse: (x_vals, y_vals)}

        # 2) which mice?
        mice = list(apa_preds.keys())
        fast = set(self.fast_learning.keys())
        slow = set(self.slow_learning.keys())

        # 3) which trial-indices belong to this phase?
        runs = np.array(expstuff['condition_exp_runs']
                        ['APAChar']['Extended'][phase])

        # 4) set up figure
        fig, ax = plt.subplots(figsize=(4, 4))
        low_vals, high_vals, colors, diffs = [], [], [], []

        for m in mice:
            # --- APA for this mouse ---
            y_apa = apa_preds[m]
            trials = np.arange(len(y_apa))
            phase_mask = np.isin(trials, runs)
            x_apa = trials[phase_mask]
            y_apa = y_apa[phase_mask]

            # --- interp disturbance to the APA trial points ---
            x_dist, y_dist = disturb_preds[m]
            y_dist_on_apa = np.interp(x_apa, x_dist, y_dist)

            # --- tertiles of APA strength ---
            order = np.argsort(y_apa)
            third = len(order) // 3
            bot_idx = order[:third]
            top_idx = order[-third:]

            bot_mean = y_dist_on_apa[bot_idx].mean()
            top_mean = y_dist_on_apa[top_idx].mean()

            low_vals.append(bot_mean)
            high_vals.append(top_mean)
            diff = bot_mean - top_mean
            diffs.append(diff)

            # choose color
            if m in fast:
                c = 'blue'
            elif m in slow:
                c = 'red'
            else:
                c = 'gray'
            colors.append(c)

            # draw the per‐mouse connector
            ax.plot([1, 2], [bot_mean, top_mean],
                    marker='o', markersize=desired_diameter,
                    color=c, alpha=0.5)

        # 5) scatter the Δ at x=3
        s = np.pi * (desired_diameter / 2) ** 2
        jit = np.random.normal(0, 0.02, size=len(diffs))
        ax.scatter(3 + jit, diffs, s=s,
                   c=colors, edgecolors='none', alpha=0.7)

        # 6) legend handles
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color='blue', label='Fast learners'),
            mpatches.Patch(color='red', label='Slow learners'),
            mpatches.Patch(color='gray', label='Others'),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=8)

        # 7) styling & save
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Low', 'High', 'Δ'], fontsize=9)
        ax.set_ylabel('Disturbance prediction', fontsize=10)
        ax.set_title(f"{phase} — all mice", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        fname = f"Disturb_vs_Pred_interp_{phase}_all"
        fig.savefig(os.path.join(self.base_dir, fname + '.png'), dpi=300)
        fig.savefig(os.path.join(self.base_dir, fname + '.svg'), dpi=300)
        plt.close(fig)

    def mean_window(self, preds, start, window_size=5):
        # Extract window trials
        window_trials = np.arange(start, start + window_size)
        vals = preds[window_trials]
        vals_smoothed = median_filter(vals, size=3, mode='nearest')
        return np.mean(vals_smoothed)

    def compute_phase_slope_differences(self, chosen_pcs=[1, 3, 7], smooth_window=3):
        """
        Compute difference per mouse between +5 and +10 trial windows for
        learning (phase start ~10) and extinction (phase start ~110).
        Also compute this for each chosen PC.

        Returns:
            dict: {
                'learning': {mouse_id: delta_value, ...},
                'extinction': {mouse_id: delta_value, ...},
                'learning_pc': {pc: {mouse_id: delta_value, ...}, ...},
                'extinction_pc': {pc: {mouse_id: delta_value, ...}, ...}
            }
        """
        result = {
            'learning': {},
            'extinction': {},
            'learning_pc': {pc: {} for pc in chosen_pcs},
            'extinction_pc': {pc: {} for pc in chosen_pcs},
        }

        # Trial windows for +5 and +10 means (adjust if needed)
        # For example, +5 window is 5 trials starting at phase start (10 or 110)
        # +10 window is next 5 trials after +5 window
        window_size = 5
        learning_start = 10
        extinction_start = 110

        # Get total preds (non-PC-wise)
        total_preds = self.get_preds(pcwise=False, s=-1)
        mice = total_preds.keys()

        # Helper to compute mean over window with optional smoothing


        # Compute delta = mean(+5 window) - mean(+10 window) per mouse for learning and extinction
        for m in mice:
            preds = total_preds[m]
            # if smooth_window:
            #     preds = median_filter(preds, size=smooth_window, mode='nearest')

            # learning delta
            learning_5 = self.mean_window(preds, learning_start) # smoothing inside mean_window
            learning_10 = self.mean_window(preds, learning_start + window_size)
            result['learning'][m] = abs(learning_10 - learning_5)

            # extinction delta
            extinction_5 = self.mean_window(preds, extinction_start)
            extinction_10 = self.mean_window(preds, extinction_start + window_size)
            result['extinction'][m] = abs(extinction_10 - extinction_5)

        # Now do same for each PC
        pc_preds = self.get_preds(pcwise=True, s=-1)
        for pc in chosen_pcs:
            pc_key = f'PC{pc}'
            pc_data = pc_preds.get(pc_key, {})
            for m in mice:
                preds = pc_data.get(m, None)
                if preds is None:
                    continue
                if smooth_window:
                    preds = median_filter(preds, size=smooth_window, mode='nearest')

                learning_5 = self.mean_window(preds, learning_start)
                learning_10 = self.mean_window(preds, learning_start + window_size)
                result['learning_pc'][pc][m] = abs(learning_10 - learning_5)

                extinction_5 = self.mean_window(preds, extinction_start)
                extinction_10 = self.mean_window(preds, extinction_start + window_size)
                result['extinction_pc'][pc][m] = abs(extinction_10 - extinction_5)

        self.phase_slope_differences = result

    def plot_learning_extinction_slope(self, chosen_pc=None, fs=7):
        if not chosen_pc:
            learning_slopes = self.phase_slope_differences['learning']
            extinction_slopes = self.phase_slope_differences['extinction']
        else:
            learning_slopes = self.phase_slope_differences['learning_pc'].get(chosen_pc, {})
            extinction_slopes = self.phase_slope_differences['extinction_pc'].get(chosen_pc, {})

        speed_ordered_mice = list(self.all_learners_learning.keys())
        fig, ax = plt.subplots(figsize=(2, 2))
        diffs = np.full_like(list(learning_slopes.values()), np.nan, dtype=float)
        for midx, mouse_id in enumerate(speed_ordered_mice):
            diff = learning_slopes[mouse_id] - extinction_slopes.get(mouse_id, 0)
            diffs[midx] = diff
            marker = pu.get_marker_style_mice(mouse_id)
            ax.plot(
                [1, 2], [learning_slopes[mouse_id], extinction_slopes[mouse_id]],
                marker=marker, markersize=3, linewidth=1, color=pu.get_color_mice(mouse_id, speedordered=speed_ordered_mice),
            )
            ax.scatter(3, diff, marker=marker, s=15, color=pu.get_color_mice(mouse_id, speedordered=speed_ordered_mice))

        # test against 0
        tstat, pval = ttest_1samp(diffs, 0)
        if pval < 0.001:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        elif pval < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'

        max_y = max(np.nanmax(diffs), 0.1)
        max_y_rounded = 0.1 * np.ceil(max_y / 0.1)
        ax.text(3, max_y_rounded, stars, fontsize=fs, ha='center')

        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Learning', 'Extinction', 'Diff'], fontsize=fs)

        self.set_dynamic_yticks(ax)

        ax.set_ylabel('Absolute Slope', fontsize=fs)
        ax.set_title(f"PC{chosen_pc}" if chosen_pc else "", fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        savepath = os.path.join(self.base_dir, f'Learning_vs_Extinction_Slope_PC{chosen_pc}' if chosen_pc else 'Learning_vs_Extinction_Slope')
        plt.tight_layout()
        plt.savefig(f"{savepath}.png", format='png', bbox_inches='tight', dpi=300)
        plt.savefig(f"{savepath}.svg", format='svg', bbox_inches='tight', dpi=300)

        # Save p-value for the difference test (across all mice)
        self.save_pvalues_csv(
            savepath_base=savepath,
            comparisons=[f'Learning_vs_Extinction_diff{"_PC" + str(chosen_pc) if chosen_pc else ""}'],
            pvals=[pval],
            test_type='1-sample t-test'
        )
        plt.close()

    def set_dynamic_yticks(self, ax, bottom_buffer=False):
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin

        # Decide decimal places and tick step
        if yrange >= 50:
            dp = 0
            tick_step = 10
        elif yrange >= 10:
            dp = 0
            tick_step = 5
        elif yrange >= 1:
            dp = 2
            tick_step = 0.25
        elif yrange >= 0.1:
            dp = 1
            tick_step = 0.2
        else:
            dp = 3
            tick_step = 0.01

        # Round limits
        ymin_rounded = tick_step * np.floor(ymin / tick_step)
        ymax_rounded = tick_step * np.ceil(ymax / tick_step)

        # Generate ticks and format them
        if bottom_buffer:
            ticks = np.arange(ymin_rounded - tick_step, ymax_rounded + tick_step, tick_step)
        else:
            ticks = np.arange(ymin_rounded, ymax_rounded + tick_step, tick_step)
        ax.set_ylim(ymin_rounded, ymax_rounded)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.{dp}f}" for tick in ticks])

        return dp

    def get_slope_arrays(self, pcs=[1, 3, 7]):
        """
        Return mean slopes per PC for learning and extinction as arrays.
        Uses self.phase_slope_differences, expects it to be computed.
        """
        learning_means = []
        extinction_means = []

        for pc in pcs:
            learning_dict = self.phase_slope_differences['learning_pc'].get(pc, {})
            extinction_dict = self.phase_slope_differences['extinction_pc'].get(pc, {})

            # Convert dict values to arrays (only mice that appear in both)
            common_mice = set(learning_dict.keys()).intersection(extinction_dict.keys())
            learning_vals = np.array([learning_dict[m] for m in common_mice])
            extinction_vals = np.array([extinction_dict[m] for m in common_mice])

            learning_means.append(np.nanmean(learning_vals))
            extinction_means.append(np.nanmean(extinction_vals))

        return np.array(learning_means), np.array(extinction_means)

    def get_slopes_across_chunks(self, pcs=[1, 3, 7], fs=7):
        # calculate slopes for each 5 trial chunk
        chunk_size = 10
        # Get total preds (non-PC-wise)
        total_preds = self.get_preds(pcwise=False, s=-1)
        mice = total_preds.keys()

        mouse_chunk_slopes = {mouse: [] for mouse in mice}
        for midx, mouse in enumerate(mice):
            preds = total_preds[mouse]

            # instantiate array to hold slopes, should be as long as len(preds) // chunk_size
            mouse_chunk_means = np.zeros((len(preds) - chunk_size) // chunk_size + 1)
            for chunk in range(0, len(preds) - chunk_size + 1, chunk_size):
                chunk_preds = preds[chunk:chunk + chunk_size]
                #chunk_preds_smoothed = median_filter(chunk_preds, size=3, mode='nearest')
                mouse_chunk_means[chunk // chunk_size] = np.mean(chunk_preds)
            # find slopes between chunks
            mouse_chunk_slope = np.diff(mouse_chunk_means)
            mouse_chunk_slopes[mouse] = mouse_chunk_slope

        # Now plot the slopes for each mouse
        # for mouse, slopes in mouse_chunk_slopes.items():
        #     fig, ax = plt.subplots(figsize=(8, 5))
        #     x = (np.arange(len(mouse_chunk_slopes[next(iter(mouse_chunk_slopes))])) + 1) * chunk_size + chunk_size# chunk indices
        #     ax.plot(x[:5], slopes[:5], marker='o', label=mouse, linewidth=1,
        #             color=pu.get_color_mice(mouse, speedordered=list(mice)))
        # ax.set_xticks(x)

        mouse_chunk_positives = {mouse: [] for mouse in mice}
        for mouse, slopes in mouse_chunk_slopes.items():
            # find first positive value
            first_positive = np.argmax(slopes[:5] > 0)
            if first_positive < 5:
                positive_idx = first_positive
                mouse_chunk_positives[mouse] = (positive_idx + 1) * chunk_size + chunk_size
            else:
                mouse_chunk_positives[mouse] = np.nan

        self.mouse_chunk_positives = mouse_chunk_positives

    # Helper to mark significance
    def significance_marker(self, p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'n.s.'

    # Original function modified to add t-tests and significance annotations
    def plot_slopes_all_pcs(self, pcs=[1, 3, 7], custom_chunks=False, fs=10):
        learning_pc = self.phase_slope_differences['learning_pc']
        extinction_pc = self.phase_slope_differences['extinction_pc']
        speed_ordered_mice = list(self.all_learners_learning.keys())

        fig, axes = plt.subplots(1, 2, figsize=(3, 3), sharey=True)
        ax1, ax2 = axes
        x = np.arange(len(pcs)) + 1

        learning_data, extinction_data = [], []

        # Learning subplot
        for mouse in speed_ordered_mice:
            yvals = [learning_pc.get(pc, {}).get(mouse, np.nan) for pc in pcs]
            learning_data.append(yvals)
            mouse_marker = pu.get_marker_style_mice(mouse)
            ax1.plot(x, yvals, marker=mouse_marker, label=mouse, linewidth=1,
                     color=pu.get_color_mice(mouse, speedordered=speed_ordered_mice), ms=3)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'PC{pc}' for pc in pcs], fontsize=fs)
        ax1.set_xlabel('Learning slopes', fontsize=fs)
        ax1.set_ylabel('Absolute Slope', fontsize=fs)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Extinction subplot
        for mouse in speed_ordered_mice:
            yvals = [extinction_pc.get(pc, {}).get(mouse, np.nan) for pc in pcs]
            extinction_data.append(yvals)
            mouse_marker = pu.get_marker_style_mice(mouse)
            ax2.plot(x, yvals, marker=mouse_marker, label=mouse, linewidth=1,
                     color=pu.get_color_mice(mouse, speedordered=speed_ordered_mice), ms=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'PC{pc}' for pc in pcs], fontsize=fs)
        ax2.set_xlabel('Extinction slopes', fontsize=fs)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Perform paired t-tests and annotate
        for data, ax in zip([learning_data, extinction_data], [ax1, ax2]):
            data = np.array(data)
            ylim = ax.get_ylim()
            y_max = ylim[1]
            y_range = ylim[1] - ylim[0]

            comparisons = [(0, 1), (1, 2), (0, 2)]
            for idx, (i, j) in enumerate(comparisons):
                vals_i, vals_j = data[:, i], data[:, j]
                t_stat, p_val = ttest_rel(vals_i, vals_j, nan_policy='omit')

                y_pos = y_max + (0.05 + 0.1 * idx) * y_range
                ax.plot([x[i], x[j]], [y_pos, y_pos], color='k', linewidth=1)
                ax.text(np.mean([x[i], x[j]]), y_pos + 0.01 * y_range, self.significance_marker(p_val),
                        ha='center', fontsize=fs)

            ax.set_ylim(ylim[0], y_max + 0.25 * y_range)  # Extend y-axis for annotations

        self.set_dynamic_yticks(ax1, bottom_buffer=False)
        # self.set_dynamic_yticks(ax2, bottom_buffer=True)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticks(ax1.get_yticks())
        # ax2.tick_params(axis='y', labelleft=True)  # If you want labels on both

        plt.tight_layout()
        savepath = os.path.join(self.base_dir, 'Slopes_Learning_Extinction_AllPCs_Scatter')
        plt.savefig(f"{savepath}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{savepath}.svg", dpi=300, bbox_inches='tight')

        comparisons, pvals = [], []
        for ax_data, tag in zip([learning_data, extinction_data], ['learning', 'extinction']):
            arr = np.array(ax_data)
            for (i, j) in [(0, 1), (1, 2), (0, 2)]:
                comparisons.append(f'{tag}_PC{pcs[i]}_vs_PC{pcs[j]}')
                _, p = ttest_rel(arr[:, i], arr[:, j], nan_policy='omit')
                pvals.append(p)
        self.save_pvalues_csv(
            savepath_base=savepath,  # as used for your figure
            comparisons=comparisons,
            pvals=pvals,
            test_type='paired t-test'
        )

        plt.close()

    def compute_daybreak_slopes(self, chosen_pcs=[1, 3, 7], smooth_window=3, window_size=5):
        """
        Compute slopes for windows at day breaks (40 and 80) same way as phase slopes.
        Returns dict: {'day40': {pc: {mouse: slope}}, 'day80': {pc: {mouse: slope}}}
        """
        day40_start = 40
        day80_start = 80

        result = {'day40': {pc: {} for pc in chosen_pcs}, 'day80': {pc: {} for pc in chosen_pcs}}

        pc_preds = self.get_preds(pcwise=True, s=-1)
        mice = list(self.LH_feature_data.index.get_level_values('MouseID').unique())

        def mean_window(preds, start):
            return np.mean(preds[start:start + window_size])

        for pc in chosen_pcs:
            pc_key = f'PC{pc}'
            pc_data = pc_preds.get(pc_key, {})
            for m in mice:
                preds = pc_data.get(m, None)
                if preds is None:
                    continue
                if smooth_window:
                    preds = median_filter(preds, size=smooth_window, mode='nearest')

                slope_40 = abs(mean_window(preds, day40_start + window_size) - mean_window(preds, day40_start))
                slope_80 = abs(mean_window(preds, day80_start + window_size) - mean_window(preds, day80_start))

                result['day40'][pc][m] = slope_40
                result['day80'][pc][m] = slope_80

        self.daybreak_slopes = result

    def plot_slopes_subtracted_daybreak(self, pcs=[1, 3, 7], daybreak='day40', fs=10):
        if not hasattr(self, 'phase_slope_differences'):
            raise RuntimeError("Compute phase slopes first by calling compute_phase_slope_differences()")
        if not hasattr(self, 'daybreak_slopes'):
            raise RuntimeError("Compute daybreak slopes first by calling compute_daybreak_slopes()")

        learning_pc = self.phase_slope_differences['learning_pc']
        extinction_pc = self.phase_slope_differences['extinction_pc']
        db_pc = self.daybreak_slopes.get(daybreak, {})
        speed_ordered_mice = list(self.all_learners_learning.keys())

        fig, axes = plt.subplots(1, 2, figsize=(3, 3), sharey=True)
        ax1, ax2 = axes
        x = np.arange(len(pcs)) + 1

        learning_data, extinction_data = [], []

        # Learning adjusted
        for mouse in speed_ordered_mice:
            vals = [learning_pc.get(pc, {}).get(mouse, np.nan) - db_pc.get(pc, {}).get(mouse, 0)
                    for pc in pcs]
            learning_data.append(vals)
            ax1.plot(x, vals, marker='o', linewidth=1,
                     color=pu.get_color_mice(mouse, speedordered=speed_ordered_mice), ms=3)
        ax1.set_xticks(x);
        ax1.set_xticklabels([f'PC{pc}' for pc in pcs], fontsize=fs)
        ax1.set_xlabel(f'Learning minus\n{daybreak}', fontsize=fs)
        ax1.set_ylabel('Slope (adjusted)', fontsize=fs)
        ax1.spines['top'].set_visible(False);
        ax1.spines['right'].set_visible(False)

        # Extinction adjusted
        for mouse in speed_ordered_mice:
            vals = [extinction_pc.get(pc, {}).get(mouse, np.nan) - db_pc.get(pc, {}).get(mouse, 0)
                    for pc in pcs]
            extinction_data.append(vals)
            ax2.plot(x, vals, marker='o', linewidth=1,
                     color=pu.get_color_mice(mouse, speedordered=speed_ordered_mice), ms=3)
        ax2.set_xticks(x);
        ax2.set_xticklabels([f'PC{pc}' for pc in pcs], fontsize=fs)
        ax2.set_xlabel(f'Extinction minus\n{daybreak}', fontsize=fs)
        ax2.spines['top'].set_visible(False);
        ax2.spines['right'].set_visible(False)

        # Annotate paired t-tests
        for data, ax in zip([learning_data, extinction_data], [ax1, ax2]):
            arr = np.array(data)
            ylim = ax.get_ylim();
            y_max = ylim[1];
            y_range = y_max - ylim[0]
            comps = [(0, 1), (1, 2), (0, 2)]
            for idx, (i, j) in enumerate(comps):
                _, p = ttest_rel(arr[:, i], arr[:, j], nan_policy='omit')
                y = y_max + (0.05 + idx * 0.1) * y_range
                ax.plot([x[i], x[j]], [y, y], 'k-')
                ax.text((x[i] + x[j]) / 2, y + 0.01 * y_range, self.significance_marker(p),
                        ha='center', fontsize=fs)
            ax.set_ylim(ylim[0], y_max + 0.25 * y_range)  # Extend y-axis for annotations

        self.set_dynamic_yticks(ax1, bottom_buffer=False)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticks(ax1.get_yticks())

        plt.tight_layout()
        savepath = os.path.join(self.base_dir, f'Slopes_Learning_Extinction_Subtracted_{daybreak}_Scatter')
        plt.savefig(f"{savepath}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{savepath}.svg", dpi=300, bbox_inches='tight')

        comparisons, pvals = [], []
        for data, tag in zip([learning_data, extinction_data], ['learning', 'extinction']):
            arr = np.array(data)
            for (i, j) in [(0, 1), (1, 2), (0, 2)]:
                comparisons.append(f'{tag}_PC{pcs[i]}_vs_PC{pcs[j]}')
                _, p = ttest_rel(arr[:, i], arr[:, j], nan_policy='omit')
                pvals.append(p)
        self.save_pvalues_csv(
            savepath_base=savepath,  # as used for your figure
            comparisons=comparisons,
            pvals=pvals,
            test_type='paired t-test'
        )

        plt.close()

    def plot_prediction_per_day_pcwise(self, chosen_pcs=[1, 3, 7], fs=7, smooth_window=3):
        preds_byPC_bymouse = self.get_preds(pcwise=True, s=-1)  # dict: {PCx: {mouse: preds}}
        speed_ordered_mice = list(self.all_learners_learning.keys())

        # These chunks are the same as your total prediction
        day_chunk_size = 5
        chunk_dividers = [10, 40, 80, 110, 120]
        day_runs = [np.arange(chunk - day_chunk_size, chunk + day_chunk_size * 2) for chunk in chunk_dividers]
        day_dividers = [40, 80, 120]

        apa1_color = pu.get_color_phase('APA1')
        apa2_color = pu.get_color_phase('APA2')
        wash1_color = pu.get_color_phase('Wash1')
        wash2_color = pu.get_color_phase('Wash2')

        for pc in chosen_pcs:
            pcname = f'PC{pc}'
            preds_df = pd.DataFrame.from_dict(preds_byPC_bymouse[pcname], orient='index')
            preds_df.index.name = 'Mouse ID'

            fig, ax = plt.subplots(figsize=(5, 3))
            # Plot phase backgrounds
            boxy = 1;
            height = 0.02
            ax.axvspan(10, 60, ymin=boxy, ymax=boxy + height, color=apa1_color, lw=0)
            ax.axvspan(60, 110, ymin=boxy, ymax=boxy + height, color=apa2_color, lw=0)
            ax.axvspan(110, 135, ymin=boxy, ymax=boxy + height, color=wash1_color, lw=0)
            ax.axvspan(135, 160, ymin=boxy, ymax=boxy + height, color=wash2_color, lw=0)

            all_chunk_means = {}
            post5_means_days = {}
            pvals_baseline = []
            chunk_pvals = []

            for day_idx, day in enumerate(day_runs):
                data = preds_df.loc[:, day]
                data = data.apply(lambda x: median_filter(x, size=smooth_window, mode='nearest'))

                baseline_window = day[:day_chunk_size]
                post5_window = day[day_chunk_size:day_chunk_size * 2]
                post10_window = day[day_chunk_size * 2:day_chunk_size * 3]

                baseline_means = data.loc[:, baseline_window].mean(axis=1)
                post5_means = data.loc[:, post5_window].mean(axis=1)
                post10_means = data.loc[:, post10_window].mean(axis=1)

                post5_means_days[day_idx] = post5_means
                all_chunk_means[day_idx] = (baseline_means, post5_means, post10_means)

                # Significance tests
                tstat_a, pval_a = ttest_rel(baseline_means, post5_means)
                tstat_b, pval_b = ttest_rel(baseline_means, post10_means)
                pvals_baseline.extend([pval_a, pval_b])
                chunk_pvals.append((pval_a, pval_b))

            reject_baseline, pvals_baseline_corr, _, _ = multipletests(pvals_baseline, alpha=0.05, method='fdr_bh')
            corrected_chunk_pvals = [(pvals_baseline_corr[i * 2], pvals_baseline_corr[i * 2 + 1]) for i in
                                     range(len(day_runs))]

            post5_means_df = pd.DataFrame(post5_means_days)
            post5_pairs = [(0, 1), (0, 2), (1, 2)]
            pvals_post5_between = []
            for pair in post5_pairs:
                tstat, pval = ttest_rel(post5_means_df.loc[:, pair[0]], post5_means_df.loc[:, pair[1]])
                pvals_post5_between.append(pval)
            reject_post5_between, pvals_post5_between_corr, _, _ = multipletests(pvals_post5_between, alpha=0.05,
                                                                                 method='fdr_bh')

            # Now plot, exactly as in your base function
            for day_idx, day in enumerate(day_runs):
                baseline_means, post5_means, post10_means = all_chunk_means[day_idx]
                x_baseline = day[:day_chunk_size].mean() + 1
                x_post5 = day[day_chunk_size:day_chunk_size * 2].mean() + 1
                x_post10 = day[day_chunk_size * 2:day_chunk_size * 3].mean() + 1
                xpos = np.array([x_baseline, x_post5, x_post10])
                data_to_plot = [baseline_means, post5_means, post10_means]

                for x, d in zip(xpos, data_to_plot):
                    for mouse in speed_ordered_mice:
                        marker = pu.get_marker_style_mice(mouse)
                        color = pu.get_color_mice(mouse, speedordered=speed_ordered_mice)
                        ax.scatter([x], d[mouse], color=color, marker=marker, s=15, alpha=0.6, linewidth=0)

                means = [d.mean() for d in data_to_plot]
                ax.plot(xpos, means, color='k', linewidth=2)

                # Brackets/stars
                y_max = max([d.max() for d in data_to_plot]) + 0.05
                line_height = 0.02
                pval_a_corr = corrected_chunk_pvals[day_idx][0]
                pval_b_corr = corrected_chunk_pvals[day_idx][1]
                for pval, (x1, x2), offset in zip([pval_a_corr, pval_b_corr],
                                                  [(x_baseline, x_post5), (x_baseline, x_post10)], [0, 0.07]):
                    if pval < 0.001:
                        stars = '***'
                    elif pval < 0.01:
                        stars = '**'
                    elif pval < 0.05:
                        stars = '*'
                    else:
                        stars = 'n.s.'
                    y_here = y_max + offset
                    ax.plot([x1, x1, x2, x2], [y_here, y_here + line_height, y_here + line_height, y_here], lw=0.8,
                            c='k')
                    ax.text((x1 + x2) / 2, y_here + line_height + 0.01, stars, ha='center', fontsize=fs)

            # Plot post5-between-chunks comparisons
            for idx, (pair, pval_corr) in enumerate(zip(post5_pairs, pvals_post5_between_corr)):
                if pval_corr < 0.001:
                    stars = '***'
                elif pval_corr < 0.01:
                    stars = '**'
                elif pval_corr < 0.05:
                    stars = '*'
                else:
                    stars = 'n.s.'
                y_max = max(post5_means_df.max()) + 0.15 * (idx + 1)
                x1 = day_runs[pair[0]][5:10].mean() + 1
                x2 = day_runs[pair[1]][5:10].mean() + 1
                ax.plot([x1, x1, x2, x2], [y_max, y_max + 0.02, y_max + 0.02, y_max], lw=0.8, c='r')
                ax.text((x1 + x2) / 2, y_max + 0.03, stars, ha='center', fontsize=fs, color='r')

            ax.vlines(x=day_dividers, ymin=-1, ymax=1, color='grey', linestyle='--', linewidth=0.5)
            ax.vlines(x=10, ymin=-1, ymax=1, color=apa1_color, linestyle='-', linewidth=0.5)
            ax.vlines(x=110, ymin=-1, ymax=1, color=wash1_color, linestyle='-', linewidth=0.5)

            ax.set_xlabel('Trial number', fontsize=fs)
            ax.set_ylabel(f'Mean Prediction (PC{pc})', fontsize=fs)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(1, 160)
            ax.set_xticks([10, 40, 60, 80, 110, 120, 135, 160])
            ax.set_ylim(-1, 1)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.legend(fontsize=fs, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

            save_path = os.path.join(self.base_dir, f'Predictions_WindowMeans_Significance_Scatter_PC{pc}')
            plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
            plt.savefig(f"{save_path}.svg", format='svg', bbox_inches='tight', dpi=300)

            # Build comparison labels for within-day (baseline vs post5/post10)
            within_labels = []
            for day_idx in range(len(day_runs)):
                within_labels.append(f'day{day_idx + 1}_baseline_vs_post5')
                within_labels.append(f'day{day_idx + 1}_baseline_vs_post10')

            # Build labels for between-day comparisons (post5 between chunks)
            between_labels = []
            for idx, pair in enumerate(post5_pairs):
                between_labels.append(f'post5_day{pair[0] + 1}_vs_day{pair[1] + 1}')

            # Concatenate all info
            all_labels = within_labels + between_labels
            all_pvals = pvals_baseline + pvals_post5_between
            all_pvals_corr = list(pvals_baseline_corr) + list(pvals_post5_between_corr)
            all_types = ['paired t-test'] * len(within_labels) + ['paired t-test'] * len(between_labels)

            # Save to CSV (save_path is your plot file base name, e.g. .../Predictions_WindowMeans_Significance_Scatter_PC1)
            self.save_pvalues_csv(
                savepath_base=save_path,
                comparisons=all_labels,
                pvals=all_pvals,
                pvals_corr=all_pvals_corr,
                test_type=all_types,
            )

            plt.close()

    def plot_pcvalues_per_day_pcwise(self, chosen_pcs=[1, 3, 7], fs=7, smooth_window=3):
        pcs_bymouse = self.get_pcs(s=-1)  # {mouse: {'pcs': [n_runs x n_pcs], 'run_vals': ...}}
        speed_ordered_mice = list(self.all_learners_learning.keys())

        day_chunk_size = 5
        chunk_dividers = [10, 40, 80, 110, 120]
        day_runs = [np.arange(chunk - day_chunk_size, chunk + day_chunk_size * 2) for chunk in chunk_dividers]
        day_dividers = [40, 80, 120]

        apa1_color = pu.get_color_phase('APA1')
        apa2_color = pu.get_color_phase('APA2')
        wash1_color = pu.get_color_phase('Wash1')
        wash2_color = pu.get_color_phase('Wash2')

        # Build PC-by-mouse DataFrames
        for pc in chosen_pcs:
            # Build DataFrame: rows=mice, cols=trials (0-159)
            pc_df = pd.DataFrame(index=speed_ordered_mice, columns=np.arange(160))
            for midx in speed_ordered_mice:
                pcs = pcs_bymouse[midx]['pcs'][:, pc - 1]
                run_vals = pcs_bymouse[midx]['run_vals']
                pcs_interp = np.interp(np.arange(160), run_vals, pcs)
                pcs_smooth = median_filter(pcs_interp, size=smooth_window, mode='nearest')
                max_abs = np.max(np.abs(pcs_smooth))
                pc_df.loc[midx] = pcs_smooth / max_abs if max_abs != 0 else pcs_smooth

            fig, ax = plt.subplots(figsize=(5, 3))
            boxy = 1;
            height = 0.02
            ax.axvspan(10, 60, ymin=boxy, ymax=boxy + height, color=apa1_color, lw=0)
            ax.axvspan(60, 110, ymin=boxy, ymax=boxy + height, color=apa2_color, lw=0)
            ax.axvspan(110, 135, ymin=boxy, ymax=boxy + height, color=wash1_color, lw=0)
            ax.axvspan(135, 160, ymin=boxy, ymax=boxy + height, color=wash2_color, lw=0)

            all_chunk_means = {}
            post5_means_days = {}
            pvals_baseline = []
            chunk_pvals = []

            for day_idx, day in enumerate(day_runs):
                data = pc_df.loc[:, day].astype(float)
                data = data.apply(lambda x: median_filter(x, size=smooth_window, mode='nearest'))

                baseline_window = day[:day_chunk_size]
                post5_window = day[day_chunk_size:day_chunk_size * 2]
                post10_window = day[day_chunk_size * 2:day_chunk_size * 3]

                baseline_means = data.loc[:, baseline_window].mean(axis=1)
                post5_means = data.loc[:, post5_window].mean(axis=1)
                post10_means = data.loc[:, post10_window].mean(axis=1)

                post5_means_days[day_idx] = post5_means
                all_chunk_means[day_idx] = (baseline_means, post5_means, post10_means)

                tstat_a, pval_a = ttest_rel(baseline_means, post5_means)
                tstat_b, pval_b = ttest_rel(baseline_means, post10_means)
                pvals_baseline.extend([pval_a, pval_b])
                chunk_pvals.append((pval_a, pval_b))

            reject_baseline, pvals_baseline_corr, _, _ = multipletests(pvals_baseline, alpha=0.05, method='fdr_bh')
            corrected_chunk_pvals = [(pvals_baseline_corr[i * 2], pvals_baseline_corr[i * 2 + 1]) for i in
                                     range(len(day_runs))]

            post5_means_df = pd.DataFrame(post5_means_days)
            post5_pairs = [(0, 1), (0, 2), (1, 2)]
            pvals_post5_between = []
            for pair in post5_pairs:
                tstat, pval = ttest_rel(post5_means_df.loc[:, pair[0]], post5_means_df.loc[:, pair[1]])
                pvals_post5_between.append(pval)
            reject_post5_between, pvals_post5_between_corr, _, _ = multipletests(pvals_post5_between, alpha=0.05,
                                                                                 method='fdr_bh')

            for day_idx, day in enumerate(day_runs):
                baseline_means, post5_means, post10_means = all_chunk_means[day_idx]
                x_baseline = day[:day_chunk_size].mean() + 1
                x_post5 = day[day_chunk_size:day_chunk_size * 2].mean() + 1
                x_post10 = day[day_chunk_size * 2:day_chunk_size * 3].mean() + 1
                xpos = np.array([x_baseline, x_post5, x_post10])
                data_to_plot = [baseline_means, post5_means, post10_means]

                for x, d in zip(xpos, data_to_plot):
                    for mouse in speed_ordered_mice:
                        marker = pu.get_marker_style_mice(mouse)
                        color = pu.get_color_mice(mouse, speedordered=speed_ordered_mice)
                        ax.scatter([x], d[mouse], color=color, marker=marker, s=15, alpha=0.6, linewidth=0)

                means = [d.mean() for d in data_to_plot]
                ax.plot(xpos, means, color='k', linewidth=2)

                # Brackets/stars
                y_max = max([d.max() for d in data_to_plot]) + 0.05
                line_height = 0.02
                pval_a_corr = corrected_chunk_pvals[day_idx][0]
                pval_b_corr = corrected_chunk_pvals[day_idx][1]
                for pval, (x1, x2), offset in zip([pval_a_corr, pval_b_corr],
                                                  [(x_baseline, x_post5), (x_baseline, x_post10)], [0, 0.07]):
                    if pval < 0.001:
                        stars = '***'
                    elif pval < 0.01:
                        stars = '**'
                    elif pval < 0.05:
                        stars = '*'
                    else:
                        stars = 'n.s.'
                    y_here = y_max + offset
                    ax.plot([x1, x1, x2, x2], [y_here, y_here + line_height, y_here + line_height, y_here], lw=0.8,
                            c='k')
                    ax.text((x1 + x2) / 2, y_here + line_height + 0.01, stars, ha='center', fontsize=fs)

            for idx, (pair, pval_corr) in enumerate(zip(post5_pairs, pvals_post5_between_corr)):
                if pval_corr < 0.001:
                    stars = '***'
                elif pval_corr < 0.01:
                    stars = '**'
                elif pval_corr < 0.05:
                    stars = '*'
                else:
                    stars = 'n.s.'
                y_max = max(post5_means_df.max()) + 0.15 * (idx + 1)
                x1 = day_runs[pair[0]][5:10].mean() + 1
                x2 = day_runs[pair[1]][5:10].mean() + 1
                ax.plot([x1, x1, x2, x2], [y_max, y_max + 0.02, y_max + 0.02, y_max], lw=0.8, c='r')
                ax.text((x1 + x2) / 2, y_max + 0.03, stars, ha='center', fontsize=fs, color='r')

            ax.vlines(x=day_dividers, ymin=-1, ymax=1, color='grey', linestyle='--', linewidth=0.5)
            ax.vlines(x=10, ymin=-1, ymax=1, color=apa1_color, linestyle='-', linewidth=0.5)
            ax.vlines(x=110, ymin=-1, ymax=1, color=wash1_color, linestyle='-', linewidth=0.5)

            ax.set_xlabel('Trial number', fontsize=fs)
            ax.set_ylabel(f'Mean PC Value (PC{pc})', fontsize=fs)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(1, 160)
            ax.set_xticks([10, 40, 60, 80, 110, 120, 135, 160])
            ax.set_ylim(-1, 1)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.legend(fontsize=fs, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

            save_path = os.path.join(self.base_dir, f'PCvals_WindowMeans_Significance_Scatter_PC{pc}')
            plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
            plt.savefig(f"{save_path}.svg", format='svg', bbox_inches='tight', dpi=300)

            # Build comparison labels for within-day (baseline vs post5/post10)
            within_labels = []
            for day_idx in range(len(day_runs)):
                within_labels.append(f'day{day_idx + 1}_baseline_vs_post5')
                within_labels.append(f'day{day_idx + 1}_baseline_vs_post10')

            # Build labels for between-day comparisons (post5 between chunks)
            between_labels = []
            for idx, pair in enumerate(post5_pairs):
                between_labels.append(f'post5_day{pair[0] + 1}_vs_day{pair[1] + 1}')

            # Concatenate all info
            all_labels = within_labels + between_labels
            all_pvals = pvals_baseline + pvals_post5_between
            all_pvals_corr = list(pvals_baseline_corr) + list(pvals_post5_between_corr)
            all_types = ['paired t-test'] * len(within_labels) + ['paired t-test'] * len(between_labels)

            # Save to CSV (save_path is your plot file base name, e.g. .../Predictions_WindowMeans_Significance_Scatter_PC1)
            self.save_pvalues_csv(
                savepath_base=save_path,
                comparisons=all_labels,
                pvals=all_pvals,
                pvals_corr=all_pvals_corr,
                test_type=all_types,
            )

            plt.close()


def main():
    save_dir = r"H:\Characterisation_v2\Learning_1"
    os.path.exists(save_dir) or os.makedirs(save_dir)

    chosen_pcs = [1, 3, 7]
    other_pcs = [5, 6, 8]
    chosen_pcs_extended = [1, 3, 5, 6, 7, 8]

    # Initialize the WhenAPA class with LH prediction data
    learning = Learning(LH_preprocessed_data, LH_stride0_preprocessed_data, LH_pred_data, LH_pca_data, save_dir,
                        disturb_pred_file=r"H:\Characterisation\LH_subtract_res_0_APA1APA2-PCStot=60-PCSuse=12\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowHigh.pkl"
                        )

    learning.find_learned_trials(smoothing=None, phase='learning')
    learning.find_learned_trials(smoothing=None, phase='extinction')
    print("Fast learners in learning phase:", learning.fast_learning)
    print("Slow learners in learning phase:", learning.slow_learning)
    print("Fast learners in extinction phase:", learning.fast_extinction)
    print("Slow learners in extinction phase:", learning.slow_extinction)

    learning.plot_learning_by_extinction_scatter()

    learning.plot_prediction_per_day_pcwise(chosen_pcs=[1, 3, 7])
    learning.plot_pcvalues_per_day_pcwise(chosen_pcs=[1, 3, 7])

    learning.plot_prediction_delta()
    learning.plot_prediction_per_day(fs=7)

    learning.compute_phase_slope_differences(chosen_pcs=[1, 3, 7], smooth_window=3)
    learning.compute_daybreak_slopes(chosen_pcs=[1, 3, 7], smooth_window=3)

    learning.plot_pc_correlations_heatmap(pcs=chosen_pcs, s=-1, fs=7)
    learning.get_slopes_across_chunks(pcs=chosen_pcs, fs=7)

    learning.plot_learning_extinction_slope(chosen_pc=None)
    for pc in chosen_pcs:
        learning.plot_learning_extinction_slope(chosen_pc=pc)


    # learning.plot_slopes_all_pcs(pcs=chosen_pcs, custom_chunks=True)
    learning.plot_slopes_all_pcs(pcs=chosen_pcs)
    learning.plot_slopes_subtracted_daybreak(pcs=chosen_pcs, daybreak='day40')
    learning.plot_slopes_subtracted_daybreak(pcs=chosen_pcs, daybreak='day80')

    learning.plot_total_predictions_x_trial(smooth_window=3)
    learning.plot_total_predictions_x_trial(fast_slow='fast', smooth_window=3)
    learning.plot_total_predictions_x_trial(fast_slow='slow', smooth_window=3)

    learning.plot_line_important_pcs_x_trial(chosen_pcs=chosen_pcs, smooth_window=10)
    learning.plot_line_important_pcs_x_trial(fast_slow='fast', chosen_pcs=chosen_pcs, smooth_window=10)
    learning.plot_line_important_pcs_x_trial(fast_slow='slow', chosen_pcs=chosen_pcs, smooth_window=10)

    learning.plot_line_important_pcs_preds_x_trial(chosen_pcs=chosen_pcs, smooth_window=10)
    learning.plot_line_important_pcs_preds_x_trial(fast_slow='fast', chosen_pcs=chosen_pcs, smooth_window=10)
    learning.plot_line_important_pcs_preds_x_trial(fast_slow='slow', chosen_pcs=chosen_pcs, smooth_window=10)

    # repeat with extended PCs
    learning.plot_line_important_pcs_x_trial(chosen_pcs=other_pcs, smooth_window=10)
    learning.plot_line_important_pcs_x_trial(fast_slow='fast', chosen_pcs=other_pcs, smooth_window=10)
    learning.plot_line_important_pcs_x_trial(fast_slow='slow', chosen_pcs=other_pcs, smooth_window=10)

    # learning.plot_line_important_pcs_preds_x_trial(chosen_pcs=other_pcs, smooth_window=10)
    # learning.plot_line_important_pcs_preds_x_trial(fast_slow='fast', chosen_pcs=other_pcs, smooth_window=10)
    # learning.plot_line_important_pcs_preds_x_trial(fast_slow='slow', chosen_pcs=other_pcs, smooth_window=10)




    learning.fit_pcwise_regression_model(chosen_pcs=chosen_pcs)

    for phase in ['APA1','APA2','APA']:
        learning.plot_disturbance_by_prediction_interpolated(phase)


if __name__ == '__main__':
    main()
