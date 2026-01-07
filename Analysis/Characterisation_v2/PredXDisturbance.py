import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_1samp

def p_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

for stride_to_measure in [-3, -2, -1]:

    savedir = rf"H:\Characterisation_v2\PredXDisturbance_3\stride_{stride_to_measure}"
    os.makedirs(savedir, exist_ok=True)


    with open(r"H:\Characterisation\LH_subtract_res_0_APA1APA2-PCStot=60-PCSuse=12\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowHigh.pkl", 'rb') as f:
        disturb_pred = pickle.load(f)
    with open(r"H:\Characterisation\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowHigh.pkl", 'rb') as f:
        apa_pred = pickle.load(f)
    # with open(r"H:\Characterisation_v2\LH_res_0_APA1APA2\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowHigh.pkl", 'rb') as f:
    #     disturb_pred = pickle.load(f)
    # with open(r"H:\Characterisation_v2\LH_res_-3-2-1_APA2Wash2\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowHigh.pkl", 'rb') as f:
    #     apa_pred = pickle.load(f)


    apa_y_pred = [pred.y_pred[0] for pred in apa_pred if pred.stride == stride_to_measure]
    apa_x_vals = [np.array(list(pred.x_vals)) for pred in apa_pred if pred.stride == stride_to_measure]
    apa_mouse_ids = [pred.mouse_id for pred in apa_pred if pred.stride == stride_to_measure]

    disturb_y_pred = [pred.y_pred[0] for pred in disturb_pred if pred.stride == 0]
    disturb_x_vals = [np.array(list(pred.x_vals)) for pred in disturb_pred if pred.stride == 0]
    disturb_mouse_ids = [pred.mouse_id for pred in disturb_pred if pred.stride == 0]

    apa_runs = np.arange(10, 110)
    apa1_runs = np.arange(10,60)
    apa2_runs = np.arange(60,110)
    wash1_runs = np.arange(110,135)
    wash2_runs = np.arange(135,160)

    for phase in ['APA1', 'APA2', 'Wash1', 'Wash2', 'APA']:
        if phase == 'APA1':
            apa_runs = apa1_runs
            disturb_runs = apa1_runs
        elif phase == 'APA2':
            apa_runs = apa2_runs
            disturb_runs = apa2_runs
        elif phase == 'Wash1':
            apa_runs = wash1_runs
            disturb_runs = wash1_runs
        elif phase == 'Wash2':
            apa_runs = wash2_runs
            disturb_runs = wash2_runs
        elif phase == 'APA':
            apa_runs = apa_runs
            disturb_runs = apa_runs
        else:
            raise ValueError(f"Unknown phase: {phase}")

        desired_diameter = 3  # points

        low_vals = []
        high_vals = []
        fig, ax = plt.subplots(figsize=(2, 2))
        #ax.xticks([1, 2], ['Low', 'High'])
        for midx, mouseID in enumerate(apa_mouse_ids):

            # Find the top and bottom third of the APA predictions
            apa_y_pred_midx = apa_y_pred[midx]
            apa_x_vals_midx = apa_x_vals[midx]
            apa_runs_mask = np.isin(apa_x_vals_midx, apa_runs)
            apa_y_pred_midx = apa_y_pred_midx[apa_runs_mask]
            apa_y_pred_df = pd.DataFrame(apa_y_pred_midx, index=apa_x_vals_midx[apa_runs_mask])
            # find top and bottom third
            apa_y_pred_df_sorted = apa_y_pred_df.sort_values(by=0, ascending=False)
            top_idxs = apa_y_pred_df_sorted.index[:len(apa_y_pred_df_sorted)//3]
            bottom_idxs = apa_y_pred_df_sorted.index[-len(apa_y_pred_df_sorted)//3:]

            # Get the corresponding disturbance predictions
            disturb_y_pred_midx = disturb_y_pred[midx]
            disturb_x_vals_midx = disturb_x_vals[midx]
            disturb_runs_mask = np.isin(disturb_x_vals_midx, disturb_runs)
            disturb_y_pred_midx = disturb_y_pred_midx[disturb_runs_mask]
            disturb_x_vals_midx = disturb_x_vals_midx[disturb_runs_mask]
            disturb_y_pred_df = pd.DataFrame(disturb_y_pred_midx, index=disturb_x_vals_midx)

            # find which APA top/bottom runs also exist in the disturb frame
            common_top_idxs = np.intersect1d(top_idxs, disturb_y_pred_df.index)
            common_bottom_idxs = np.intersect1d(bottom_idxs, disturb_y_pred_df.index)

            top_apa_y_pred = apa_y_pred_df.loc[common_top_idxs]
            bottom_apa_y_pred = apa_y_pred_df.loc[common_bottom_idxs]
            # now pull only those matching disturb values
            top_disturb_y_pred = disturb_y_pred_df.loc[common_top_idxs].iloc[:, 0]
            bottom_disturb_y_pred = disturb_y_pred_df.loc[common_bottom_idxs].iloc[:, 0]

            bottom_mean = bottom_disturb_y_pred.mean()
            top_mean = top_disturb_y_pred.mean()

            low_vals.append(bottom_mean)
            high_vals.append(top_mean)

            plt.plot([1,2], [bottom_mean, top_mean], marker='o', markersize=desired_diameter, color='black', alpha=0.3)

            # jitter = 0.05
            # # top (High)
            # x_top = np.random.normal(2, jitter, size=len(top_disturb_y_pred))
            # plt.scatter(x_top, top_disturb_y_pred, alpha=0.3)
            #
            # # bottom (Low)
            # x_bot = np.random.normal(1, jitter, size=len(bottom_disturb_y_pred))
            # plt.scatter(x_bot, bottom_disturb_y_pred, alpha=0.3)



        # — then, immediately after the end of that midx‐loop (but before plt.show()) —
        s = desired_diameter ** 2  # ≈ 25
        # or for literal circle area:
        s = np.pi * (desired_diameter / 2) ** 2  # ≈ 19.6
        jitter = .02
        low_vals = np.array(low_vals)
        high_vals = np.array(high_vals)
        deltas = low_vals - high_vals
        ax.scatter(np.random.normal(3, jitter, size=len(deltas)), deltas, s=s)

        # one-sample t-test vs 0
        tstat, pval = ttest_1samp(deltas, 0, nan_policy='omit')
        stars = p_to_stars(pval)

        # add mean marker for delta
        mean_delta = np.nanmean(deltas)
        ax.scatter([3], [mean_delta], s=s * 2.5, edgecolor='k', linewidth=0.8, zorder=3)

        y_min, y_max = -2, 2  # keep your chosen limits
        ax.set_ylim(y_min, y_max)
        span = y_max - y_min
        y_for_star = max(mean_delta, 0) + 0.07 * span
        y_for_star = min(y_for_star, y_max - 0.05 * span)  # keep inside frame
        if stars:
            ax.text(3, y_for_star, stars, ha='center', va='bottom', fontsize=9)

        # parts = plt.violinplot(
        #     [low_vals, high_vals],
        #     positions=[1, 2],
        #     widths=0.6,
        #     showextrema=False
        # )
        # for pc in parts['bodies']:
        #     pc.set_alpha(0.3)
        ax.set_title(phase, fontsize=7)
        ax.set_xlabel('APA prediction strength', fontsize=7)
        ax.set_ylabel('Disturbance', fontsize=7)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Low', 'High', 'Low-High'], fontsize=7)
        # set y tick font size to 7
        ax.set_ylim(-2, 2) ################## was (-0.5, 2)
        ax.set_yticks([0, 2]) ################### was ([0, 2])
        ax.set_yticklabels([0, '2'], fontsize=7)
        # ax.set_yticks([0, 1])
        # ax.set_yticklabels([0, 1], fontsize=7)
        #plt.xticks([1, 2], ['Low', 'High'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)

        fig.savefig(os.path.join(savedir, f"Disturbance_Prediction_{phase}.png"), dpi=300)
        fig.savefig(os.path.join(savedir, f"Disturbance_Prediction_{phase}.svg"), dpi=300)
        plt.close()









print('Done')
