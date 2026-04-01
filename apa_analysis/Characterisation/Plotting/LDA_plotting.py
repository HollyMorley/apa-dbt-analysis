"""Plots for LDA classification results."""
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import numpy as np
import matplotlib.ticker as ticker
import os
import pandas as pd
from scipy.ndimage import median_filter

from apa_analysis.Characterisation import Plotting_utils as pu
from apa_analysis.config import (global_settings)

def plot_lda_weights(lda_data, s, cond1, cond2, exp, save_dir, fs=7):
    apa_weights = [lda.weights for lda in lda_data if lda.stride == s and lda.phase == 'apa' and lda.conditions == [f"APAChar_{cond1}", f"APAChar_{cond2}"]]
    apa_weights = np.array(apa_weights)[:, :global_settings["pcs_to_plot"]]
    mice_names = [lda.mouse_id for lda in lda_data if lda.stride == s and lda.phase == 'apa' and lda.conditions == [f"APAChar_{cond1}", f"APAChar_{cond2}"]]

    fig, ax = plt.subplots(figsize=(4, 8))

    w_norm = np.zeros_like(apa_weights)
    for midx, mouse_weights in enumerate(apa_weights):
        mouse_name = mice_names[midx]

        # max/abs normalisation
        max_abs = np.abs(mouse_weights).max()
        w_mouse_norm = mouse_weights / max_abs if max_abs != 0 else mouse_weights
        w_norm[midx] = w_mouse_norm

        ms = pu.get_marker_style_mice(mouse_name)
        ls='-'

        ax.plot(mouse_weights, np.arange(1, len(mouse_weights)+1),
                marker=ms, markersize=4, linestyle=ls, linewidth=0.5, label=mouse_name, color='k')

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.4)

    ax.set_xlabel("Normalized PC Weights", fontsize=fs)
    ax.set_ylabel("PC", fontsize=fs)
    ax.set_xlim(-1.2, 1.2)
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_xticklabels(np.arange(-1, 1.1, 0.5), fontsize=fs)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.tick_params(axis='x', which='major', bottom=True, top=False, length=4, width=1)
    ax.tick_params(axis='x', which='minor', bottom=True, top=False, length=2, width=1)
    ax.set_ylim(0, global_settings["pcs_to_plot"]+1)
    ax.set_yticks(np.arange(1, global_settings["pcs_to_plot"]+1))
    ax.set_yticklabels(np.arange(1, global_settings["pcs_to_plot"]+1), fontsize=fs)
    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title=f"Mouse ID", title_fontsize=fs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.grid(False)

    ax.invert_yaxis()

    plt.subplots_adjust(right=0.7, top=0.95)

    fig.savefig(os.path.join(save_dir, f"Normalized_PC_lda_weights_{cond1}_{cond2}_stride{s}_{exp}.png"), dpi=300)
    fig.savefig(os.path.join(save_dir, f"Normalized_PC_lda_weights_{cond1}_{cond2}_stride{s}_{exp}.svg"), dpi=300)
    plt.close()

# def plot_prediction_histogram(lda_data, s, conditions, exp, save_dir, fs=7):
#     apa_preds = [lda.y_pred for lda in lda_data if lda.stride == s and lda.phase == 'apa']
#     x_vals = [lda.x_vals for lda in lda_data if lda.stride == s and lda.phase == 'apa']
#     mice_names = [lda.mouse_id for lda in lda_data if lda.stride == s and lda.phase == 'apa']
#     common_x = np.arange(0, 100, 1)
#
#     preds_df = pd.DataFrame(index=common_x, columns=mice_names, dtype=float)
#     for midx, mouse_preds in enumerate(apa_preds):
#         mouse_name = mice_names[midx]
#         preds_df.loc[x_vals[midx], mouse_name] = mouse_preds
#
#     all_preds = {}
#     for i, cond in enumerate(conditions):
#         trial_inds = np.arange(i * 50, (i + 1) * 50)
#         preds_cond = preds_df.loc[trial_inds].values.flatten()
#         preds_cond = preds_cond[~np.isnan(preds_cond)]
#         all_preds[cond] = preds_cond
#
#     fig, ax = plt.subplots(figsize=(2, 2))
#     bins = np.linspace(-1, 1, 20)
#
#     for cond in conditions:
#         color = pu.get_color_speedpair(cond.split('_')[-1])
#         hist_vals, _ = np.histogram(all_preds[cond], bins=bins)
#         smoothed_hist = gaussian_filter1d(hist_vals, sigma=1.5)
#         ax.plot(bins[:-1], smoothed_hist, label=cond.split('_')[-1], color=color, linewidth=1.5, linestyle='-')
#
#     ax.set_xlabel('Prediction Score', fontsize=fs)
#     ax.set_ylabel('Count', fontsize=fs)
#     ax.legend(fontsize=fs - 1, loc='upper right')
#
#     # X-axis: labels + minor ticks
#     ax.set_xlim(-1.2, 1.2)
#     ax.set_xticks(np.arange(-1, 1.1, 0.5))
#     ax.set_xticklabels(np.arange(-1, 1.1, 0.5), fontsize=fs)
#     ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
#     ax.tick_params(axis='x', which='minor', bottom=True, length=2, width=1, color='k')
#     ax.tick_params(axis='x', which='major', bottom=True, length=4, width=1)
#
#     # Y-axis: font size + minor ticks
#     ax.tick_params(axis='y', labelsize=fs)
#     ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
#     ax.tick_params(axis='y', which='minor', length=2, width=1, color='k')
#
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     plt.grid(False)
#     plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
#
#     fname = f"LDA_PredictionHistogram_stride{s}_{'_vs_'.join(conditions)}_{exp}"
#     save_path_full = os.path.join(save_dir, fname)
#     plt.savefig(f"{save_path_full}.png", dpi=300)
#     plt.savefig(f"{save_path_full}.svg", dpi=300)
#     plt.close()


def plot_prediction_per_trial(lda_data, s, cond1, cond2, exp, save_dir, smooth_kernel=3, normalise=True, fs=7):
    apa_preds = [lda.y_pred for lda in lda_data if lda.stride == s and lda.phase == 'apa' and lda.conditions == [f"APAChar_{cond1}", f"APAChar_{cond2}"]]
    x_vals = [lda.x_vals for lda in lda_data if lda.stride == s and lda.phase == 'apa' and lda.conditions == [f"APAChar_{cond1}", f"APAChar_{cond2}"]]
    mice_names = [lda.mouse_id for lda in lda_data if lda.stride == s and lda.phase == 'apa' and lda.conditions == [f"APAChar_{cond1}", f"APAChar_{cond2}"]]
    common_x = np.arange(0, 100, 1)

    preds_df = pd.DataFrame(index=common_x, columns=mice_names, dtype=float)
    for midx, mouse_preds in enumerate(apa_preds):
        mouse_name = mice_names[midx]
        preds_df.loc[x_vals[midx], mouse_name] = mouse_preds

    fig,ax = plt.subplots(figsize=(5, 5))

    LH_color = pu.get_color_speedpair(cond1)
    other_color = pu.get_color_speedpair(cond2)

    boxy = 1
    height = 0.02
    patch1 = plt.axvspan(xmin=0, xmax=50, ymin=boxy, ymax=boxy+height, color=LH_color, lw=0)
    patch2 = plt.axvspan(xmin=50, xmax=100, ymin=boxy, ymax=boxy+height, color=other_color, lw=0)
    patch1.set_clip_on(False)
    patch2.set_clip_on(False)

    interp_preds = pd.DataFrame(index=common_x, columns=mice_names, dtype=float)
    for mouse_name in mice_names:
        pred = preds_df[mouse_name].values  # shape: (len(common_x),)

        # Interpolate missing values before smoothing
        pred_interp = pd.Series(pred, index=common_x).interpolate(limit_direction='both').values

        # Apply smoothing
        smoothed_preds = median_filter(pred_interp, size=smooth_kernel,
                                       mode='nearest') if smooth_kernel > 1 else pred_interp

        # Normalize if requested
        if normalise and not np.all(np.isnan(smoothed_preds)):
            max_abs = max(abs(np.nanmin(smoothed_preds)), abs(np.nanmax(smoothed_preds)))
            smoothed_preds = smoothed_preds / max_abs if max_abs != 0 else smoothed_preds

        # Store final prediction
        interp_preds[mouse_name] = smoothed_preds
        ax.plot(common_x, smoothed_preds, linewidth=0.6, alpha=0.3, color='grey', label=f'Mouse {mouse_name}')

    mean_preds = interp_preds.mean(axis=1)
    ax.plot(common_x, mean_preds, color='black', linewidth=1, label='Mean Curve')

    ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.axvline(50, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

    ax.set_title(s, fontsize=fs, pad=10)
    ax.set_xlabel('Trial', fontsize=fs)
    ax.set_ylabel('Normalized Prediction', fontsize=fs)
    ax.set_ylim(-1, 1)
    ax.set_yticks(np.arange(-0.5, 0.6, 0.5))
    ax.set_yticklabels(np.arange(-0.5, 0.6, 0.5), fontsize=fs)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xticklabels(np.arange(0, 101, 20), fontsize=fs)
    ax.set_xlim(0, 100)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis='x', which='major', bottom=True, top=False, length=4, width=1)
    ax.tick_params(axis='x', which='minor', bottom=True, top=False, length=2, width=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.grid(False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.1)

    save_path_full = os.path.join(save_dir,
                                  f"LDA_run_predictions_{cond1}_vs_{cond2}_stride{s}_{exp}")
    plt.savefig(f"{save_path_full}.png", dpi=300)
    plt.savefig(f"{save_path_full}.svg", dpi=300)
    plt.close()
    return mean_preds, interp_preds

def plot_prediction_discrete_conditions(interp_preds, s, cond1, cond2, exp, save_dir, fs=7):
    fig, ax = plt.subplots(figsize=(2, 3))

    cond1_vals = interp_preds.loc[:50].mean(axis=0)
    cond2_vals = interp_preds.loc[50:].mean(axis=0)

    # Boxplot overlay
    data_to_plot = [cond1_vals.values, cond2_vals.values]
    ax.boxplot(data_to_plot, positions=[0, 1], widths=0.3, patch_artist=True, zorder=1,
               boxprops=dict(facecolor='lightgrey', color='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))

    # Paired lines for each mouse
    for midx in interp_preds.columns:
        ax.plot([cond1_vals[midx], cond2_vals[midx]], linestyle='--', marker='o', markersize=3, zorder=2,
                linewidth=1, alpha=0.3, color='grey')

    ax.set_xticks([0, 1])
    ax.set_xticklabels([cond1, cond2])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # change all font sizes to 7
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel('Condition', fontsize=fs)
    plt.ylabel('Prediction Score', fontsize=fs)
    plt.title(s, fontsize=fs)

    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)

    save_path_full = os.path.join(save_dir, f"LDA_mean_predictions_{cond1}_vs_{cond2}_stride{s}_{exp}")
    plt.savefig(f"{save_path_full}.png", dpi=300)
    plt.savefig(f"{save_path_full}.svg", dpi=300)
    plt.close()













