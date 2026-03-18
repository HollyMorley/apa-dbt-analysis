import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from scipy.signal import medfilt
import matplotlib.ticker as ticker
import os
import pickle
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
from matplotlib.ticker import FuncFormatter
import re
import matplotlib.patches as mpatches
from scipy import stats

from Helpers.Config_23 import *

from Helpers import utils
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import Plotting_utils as pu


def extract_bodypart_labels(feature_name, bodypart_names):
    # Sort by length descending so Back12 is checked before Back1, etc.
    sorted_names = sorted(bodypart_names, key=len, reverse=True)
    matches = []
    used_ranges = []

    idx = 0
    while len(matches) < 2 and idx < len(feature_name):
        found = False
        for bp in sorted_names:
            if feature_name.startswith(bp, idx):
                matches.append(bp)
                idx += len(bp)
                found = True
                break
        if not found:
            idx += 1  # move forward in string if no match at current position

    # If two matches found, return in order of appearance
    if len(matches) >= 2:
        return matches[0], matches[1]
    elif len(matches) == 1:
        return matches[0], ''
    else:
        return '', ''


def plot_literature_parallels(feature_data, stride, phases, savedir, fs=7):
    features = ['stride_length|speed_correct:True', 'cadence', 'walking_speed|bodypart:Tail1, speed_correct:True',
                'bos_stancestart|ref_or_contr:ref, y_or_euc:y', 'bos_stancestart|ref_or_contr:contr, y_or_euc:y',
                'back_skew|step_phase:1, all_vals:False, full_stride:False, buffer_size:0',
                'signed_angle|Back1Back12_side_zref_swing_peak']
    for feature in features:
        # Plot a TS of feature and also phase difference
        single_feat = feature_data.loc(axis=0)[stride].loc(axis=1)[feature]
        # Get the mask for each phase
        mask_p1, mask_p2 = gu.get_mask_p1_p2(single_feat, 'APA1', 'APA2')
        featsp1 = single_feat.loc(axis=0)[mask_p1]
        featsp2 = single_feat.loc(axis=0)[mask_p2]
        # make mice a column
        featsp1 = featsp1.unstack(level=0)
        featsp2 = featsp2.unstack(level=0)

        feat_name = short_names.get(feature, feature)

        # Plot the feature
        # fig, axs = plt.subplots(1, 2, figsize=(5, 2))
        fig, axs = plt.subplots(
            1, 2,
            figsize=(3, 2),
            gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.2}
        )

        apa1_color = pu.get_color_phase('APA1')
        apa2_color = pu.get_color_phase('APA2')
        wash1_color = pu.get_color_phase('Wash1')
        wash2_color = pu.get_color_phase('Wash2')

        boxy = 1
        height = 0.02
        patch1 = axs[0].axvspan(xmin=10, xmax=60, ymin=boxy, ymax=boxy + height, color=apa1_color, lw=0)
        patch2 = axs[0].axvspan(xmin=60, xmax=110, ymin=boxy, ymax=boxy + height, color=apa2_color, lw=0)
        patch3 = axs[0].axvspan(xmin=110, xmax=135, ymin=boxy, ymax=boxy + height, color=wash1_color, lw=0)
        patch4 = axs[0].axvspan(xmin=135, xmax=160, ymin=boxy, ymax=boxy + height, color=wash2_color, lw=0)
        patch1.set_clip_on(False)
        patch2.set_clip_on(False)
        patch3.set_clip_on(False)
        patch4.set_clip_on(False)

        mice = single_feat.index.get_level_values(level='MouseID').unique().tolist()
        common_x = np.arange(0, 160)

        # plot time series
        mice_data = np.zeros((len(mice), len(common_x)))
        for midx, m in enumerate(mice):
            mouse_feat = single_feat.loc(axis=0)[m]
            mouse_feat_interp = np.interp(common_x+1, mouse_feat.index.get_level_values(level='Run'), mouse_feat.values)
            mouse_feat_smooth = medfilt(mouse_feat_interp, kernel_size=15)
            #axs[0].plot(common_x, mouse_feat_smooth, alpha=0.2, color='grey')
            mice_data[midx, :] = mouse_feat_smooth
        mice_mean = np.mean(mice_data, axis=0)
        mice_sem = np.std(mice_data, axis=0) / np.sqrt(len(mice))
        axs[0].plot(common_x+1, mice_mean, color='black', label='Mean', linewidth=1)
        axs[0].fill_between(common_x, mice_mean - mice_sem, mice_mean + mice_sem, color='black', alpha=0.2)

        axs[0].axvline(x=10, color='k', linestyle='--', alpha=0.2)
        axs[0].axvline(x=60, color='k', linestyle='--', alpha=0.2)
        axs[0].axvline(x=110, color='k', linestyle='--', alpha=0.2)
        axs[0].axvline(x=135, color='k', linestyle='--', alpha=0.2)

        axs[0].grid(False)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].set_ylim(-1.2, 1.2)
        axs[0].set_yticks([-1, 0, 1])
        axs[0].tick_params(axis='x', which='major', bottom=True, top=False, length=4, width=1)
        axs[0].tick_params(axis='x', which='minor', bottom=True, top=False, length=2, width=1)
        axs[0].set_yticklabels([-1, 0, 1], fontsize=fs)
        axs[0].set_ylabel(f"{feat_name} (z-scored)", fontsize=fs)
        axs[0].set_xlabel('Run', fontsize=fs)
        axs[0].set_xlim(0, 160)
        axs[0].set_xticks([0, 50, 100, 150])
        axs[0].set_xticklabels([0, 50, 100, 150], fontsize=fs)
        axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(10))

        axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # where the ticks go
        axs[0].tick_params(axis='y', which='minor',
                           left=True,  # ← use left / right for the y‑axis
                           right=False,
                           length=2, width=1)
        axs[0].tick_params(axis='y', which='major',
                           left=True, right=False,
                           length=4, width=1)

        # plt differences between phase
        p1_means = np.zeros((len(mice)))
        p2_means = np.zeros((len(mice)))
        for midx, m in enumerate(mice):
            m_p1 = featsp1.loc(axis=1)[m]
            m_p2 = featsp2.loc(axis=1)[m]
            p1_mean = m_p1.mean()
            p2_mean = m_p2.mean()
            p1_means[midx] = p1_mean
            p2_means[midx] = p2_mean
            axs[1].plot([0, 1], [p1_mean, p2_mean], 'o-', alpha=0.3, color='grey', markersize=3, zorder=10)

        # plot boxplots
        # Get the phase colours and darker versions for the median and whiskers
        p1_color = pu.get_color_phase(phases[0])
        p2_color = pu.get_color_phase(phases[1])
        dark_color_p1 = pu.darken_color(p1_color, 0.7)
        dark_color_p2 = pu.darken_color(p2_color, 0.7)

        # Boxplot properties for phases
        boxprops_p1 = dict(facecolor=p1_color, color=p1_color)
        boxprops_p2 = dict(facecolor=p2_color, color=p2_color)
        medianprops_p1 = dict(color=dark_color_p1, linewidth=2)
        whiskerprops_p1 = dict(color=dark_color_p1, linewidth=1.5, linestyle='-')
        medianprops_p2 = dict(color=dark_color_p2, linewidth=2)
        whiskerprops_p2 = dict(color=dark_color_p2, linewidth=1.5, linestyle='-')

        x = np.array([0.5])
        width = 0.35
        bar_multiple = 0.6
        positions_p1 = np.array([0])#x - width / 2
        positions_p2 = np.array([1])#x + width / 2

        axs[1].boxplot(p1_means, positions=positions_p1, widths=width*bar_multiple,
                       patch_artist=True, boxprops=boxprops_p1,
                       medianprops=medianprops_p1, whiskerprops=whiskerprops_p1, showcaps=False, showfliers=False)
        axs[1].boxplot(p2_means, positions=positions_p2, widths=width*bar_multiple,
                          patch_artist=True, boxprops=boxprops_p2,
                          medianprops=medianprops_p2, whiskerprops=whiskerprops_p2, showcaps=False, showfliers=False)

        axs[1].set_xticks([0, 1])
        axs[1].set_xticklabels(
            [r'APA$_{\mathrm{end}}$', r'Wash$_{\mathrm{end}}$'],
            fontsize=fs
        )
        axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # where the ticks go
        # axs[1].tick_params(axis='x', which='minor',
        #                     bottom=True, top=False,
        #                     length=2, width=1)
        axs[1].tick_params(axis='x', which='major',
                            bottom=True, top=False,
                            length=4, width=1)
        axs[1].set_xlim(-0.5, 1.5)
        axs[1].set_ylim(-1, 1)
        axs[1].set_yticks([-1, 0, 1])
        axs[1].set_yticklabels([-1, 0, 1], fontsize=fs)
        axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # where the ticks go
        axs[1].tick_params(axis='y', which='minor',
                            left=True,  # ← use left / right for the y‑axis
                            right=False,
                            length=2, width=1)
        axs[1].tick_params(axis='y', which='major',
                            left=True, right=False,
                            length=4, width=1)

        axs[1].grid(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)

        fig.subplots_adjust(left=0.15, right=0.99, top=0.9, bottom=0.2)

        fig.savefig(os.path.join(savedir, f"{feat_name}_{stride}_{phases[0]}_{phases[1]}.png"), dpi=300)
        fig.savefig(os.path.join(savedir, f"{feat_name}_{stride}_{phases[0]}_{phases[1]}.svg"), dpi=300)
        plt.close(fig)




def plot_angles(feature_data_notscaled, phases, stride, savedir):
    features_to_plot = ['signed_angle|ToeKnuckle_ipsi_side_zref_swing_mean',
                        'signed_angle|ToeKnuckle_ipsi_side_zref_swing_peak',
                        'signed_angle|ToeKnuckle_contra_side_zref_swing_mean',
                        'signed_angle|ToeKnuckle_contra_side_zref_swing_peak',
                        'signed_angle|ToeAnkle_ipsi_side_zref_swing_mean',
                        'signed_angle|ToeAnkle_ipsi_side_zref_swing_peak',
                        'signed_angle|ToeAnkle_contra_side_zref_swing_mean',
                        'signed_angle|ToeAnkle_contra_side_zref_swing_peak',
                        'signed_angle|Back1Back12_side_zref_swing_mean',
                        'signed_angle|Back1Back12_side_zref_swing_peak',
                        'signed_angle|Tail1Tail12_side_zref_swing_mean',
                        'signed_angle|Tail1Tail12_side_zref_swing_peak',
                        'signed_angle|NoseBack1_side_zref_swing_mean',
                        'signed_angle|NoseBack1_side_zref_swing_peak',
                        'signed_angle|Back1Back12_overhead_xref_swing_mean',
                        'signed_angle|Back1Back12_overhead_xref_swing_peak',
                        'signed_angle|Tail1Tail12_overhead_xref_swing_mean',
                        'signed_angle|Tail1Tail12_overhead_xref_swing_peak',
                        'signed_angle|NoseBack1_overhead_xref_swing_mean',
                        'signed_angle|NoseBack1_overhead_xref_swing_peak'
                        ]
    for feature in features_to_plot:
        data_notscaled = feature_data_notscaled.loc(axis=1)[feature]
        # Get the mask for each phase
        mask_p1, mask_p2 = gu.get_mask_p1_p2(data_notscaled, phases[0], phases[1])
        plot_angle_polar(angle_p1=data_notscaled[mask_p1],angle_p2=data_notscaled[mask_p2],
                         p1=phases[0], p2=phases[1], stride=stride, feature=feature, savedir=savedir)

def plot_angle_polar(angle_p1, angle_p2 ,p1, p2, stride, feature, savedir):
    feature_name = angle_p1.name
    bodypart_names = ['Toe', 'Knuckle', 'Ankle', 'Back1', 'Back12', 'Tail1', 'Tail12', 'Nose']
    short_feat = short_names.get(feature, feature)

    angle_p1.index = angle_p1.index.set_names(['Stride', 'MouseID', 'FrameIdx'])
    angle_p2.index = angle_p2.index.set_names(['Stride', 'MouseID', 'FrameIdx'])

    # filter by stride
    angle_p1 = angle_p1.loc(axis=0)[stride]
    angle_p2 = angle_p2.loc(axis=0)[stride]

    if 'ToeKnuckle' in feature_name or 'ToeAnkle' in feature_name:
        # Flip the angles as they were accidentally calculated Knuckle-Toe and Ankle-Toe
        angle_p1 = -angle_p1
        angle_p2 = -angle_p2
    if 'overhead_xref' in feature_name:
        # so 0 degrees is facing forward, not backward
        angle_p1 = -angle_p1
        angle_p2 = -angle_p2

    # Group by run (assuming the run is on level=1) to compute the average across all observations per run
    angle_p1_avg = angle_p1.groupby(level='MouseID').mean()
    angle_p2_avg = angle_p2.groupby(level='MouseID').mean()
    # Convert from degrees to radians as polar histograms require radians
    theta_p1 = np.deg2rad(angle_p1_avg.values)
    theta_p2 = np.deg2rad(angle_p2_avg.values)
    # Define the bins for the polar histogram
    # Here we create 20 equal bins spanning from 0 to 2*pi
    num_bins = 180
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    # Compute histograms: counts of run averages falling within each bin
    hist_p1, _ = np.histogram(theta_p1, bins=bins)
    hist_p2, _ = np.histogram(theta_p2, bins=bins)
    # The width of each bin, needed for the bar plot
    width = bins[1] - bins[0]
    # Create a figure with two polar subplots side by side
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(5, 5))
    # Plot Phase 1: each bar's length corresponds to the number of runs in that angular bin
    p1_col = pu.get_color_phase(p1)
    p2_col = pu.get_color_phase(p2)
    ax.bar(bins[:-1], hist_p1, width=width, bottom=0.0, align='edge', linewidth=0, color=p1_col, alpha=0.6, label=p1)
    ax.set_title(short_feat)
    # Plot Phase 2
    ax.bar(bins[:-1], hist_p2, width=width, bottom=0.0, align='edge', linewidth=0, color=p2_col, alpha=0.6, label=p2)

    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(['–180°', '–90°', '0°', '90°', '180°'])

    # Extract labels automatically
    bp1, bp2 = extract_bodypart_labels(feature_name, bodypart_names)

    # Center label
    ax.text(0, 0, bp2, ha='center', va='center', fontsize=12, fontweight='bold', color='dimgray')

    # Outer label at 0 radians
    r = ax.get_rmax() * 1.1
    ax.text(0, r, bp1, ha='center', va='bottom', fontsize=12, color='dimgray')

    if 'yaw' in short_feat:
        ax.set_theta_zero_location("E")
    elif 'pitch' in short_feat:
        ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    # ax.set_thetamin(180)
    # ax.set_thetamax(0)
    plt.legend()
    save_path = os.path.join(savedir, f"polar_histogram_{short_feat}_{p1}_{p2}_{stride}")
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.savefig(f"{save_path}.svg", dpi=300)
    plt.close()

def plot_limb_positions(raw_data, phases, savedir, fs=7):
    forelimb_parts = ['ForepawToe', 'ForepawKnuckle', 'ForepawAnkle', 'ForepawKnee']
    y_midline = structural_stuff['belt_width'] / 2

    for phase in phases:
        # Get data from last stride before transition
        for midx, mouse in enumerate(raw_data.keys()):
            mouse_data = raw_data[mouse]
            # Drop the 'Day' index level
            mouse_data = mouse_data.droplevel('Day', axis=0)
            mouse_data = mouse_data.loc(axis=0)[expstuff['condition_exp_runs']['APAChar']['Extended'][phase]]
            for run in mouse_data.index.get_level_values('Run').unique():
                run_data = mouse_data.loc(axis=0)[run]
                transition_idx = run_data.index.get_level_values(level='FrameIdx')[run_data.index.get_level_values('RunStage') == 'Transition'][0]
                transition_paw = run_data.loc(axis=1)['initiating_limb'].loc(axis=0)['Transition', transition_idx]

                # Get the last stride before transition
                stance_periods_mask = run_data.loc(axis=0)['RunStart'].loc(axis=1)[transition_paw,'SwSt_discrete'] == locostuff['swst_vals_2025']['st']
                stance_periods_idxs = run_data.loc(axis=0)['RunStart'].index.get_level_values(level='FrameIdx')[stance_periods_mask]
                last_stance_idx = stance_periods_idxs[-1]
                stride_data = run_data.loc(axis=0)['RunStart',np.arange(last_stance_idx,transition_idx)]

                fig, ax = plt.subplots(figsize=(5, 2.5))
                column_multi = pd.MultiIndex.from_product([forelimb_parts, ['x', 'y', 'z']], names=['BodyPart', 'Coord'])
                run_coords = pd.DataFrame(index=stride_data.index.get_level_values(level='FrameIdx'), columns=column_multi)
                for fidx, frame in enumerate(stride_data.index.get_level_values('FrameIdx')):
                    frame_data = stride_data.loc(axis=0)['RunStart',frame]
                    transition_paw_side = transition_paw.split('Forepaw')[1]  # 'R' or 'L'

                    transition_paw_limbparts = [l+transition_paw_side for l in forelimb_parts]
                    x = frame_data.loc[transition_paw_limbparts, 'x'].values
                    y = frame_data.loc[transition_paw_limbparts, 'y'].values
                    z = frame_data.loc[transition_paw_limbparts, 'z'].values

                    if transition_paw_side == 'L':
                        # Need to flip the y coords to mirror the left side to the right side
                        mirrored_y = 2 * y_midline - y
                        y = mirrored_y

                    # Plot the limb positions
                    ax.plot(x, z, marker='o', markersize=2, color='grey', alpha=0.5, linewidth=0.5)
                    # Store the coordinates for the current frame
                    col_x = [(part, 'x') for part in forelimb_parts]
                    col_y = [(part, 'y') for part in forelimb_parts]
                    col_z = [(part, 'z') for part in forelimb_parts]

                    run_coords.loc[frame, col_x] = x
                    run_coords.loc[frame, col_y] = y
                    run_coords.loc[frame, col_z] = z

                run_coords_mice_mean = run_coords.mean(axis=0)
                # Plot the mean limb positions
                ax.plot(run_coords_mice_mean.loc(axis=0)[forelimb_parts, 'x'], run_coords_mice_mean.loc(axis=0)[forelimb_parts, 'z'],
                        marker='o', markersize=3, color='black', linewidth=1, label='Mean')


def plot_toe_distance_to_transition(raw_data, mouse_runs, phases, savedir, fs=7, n_interp=100, stsw='st', bodypart='Toe', raw=False):
    # plot the position of the leading toe at the start of the stride and the end of the stride in APA and Wash phases
    mice = ['1035243', '1035244', '1035245', '1035246', '1035250', '1035297', '1035299', '1035301']
    y_midline = structural_stuff['belt_width'] / 2
    labels = ['Toe']

    if raw:
        fig, axs = plt.subplots(len(mice), 1, figsize=(4, 10), sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(4, 2))

    all_arr = {}
    stride_lengths_all = {}
    for phase in phases:
        phase_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase]
        n_phase_runs = len(phase_runs)
        n_mice = len(mice)
        arr = np.full((n_mice, n_phase_runs, 3, 2), np.nan)

        for m, mouse in enumerate(mice):
            mouse_data = raw_data[mouse].droplevel('Day', axis=0)
            for r, run in enumerate(phase_runs):
                if run not in mouse_runs[mouse]:
                    # print(f"Run {run} not in mouse {mouse} runs, skipping.")
                    continue

                run_data = mouse_data.loc(axis=0)[run]

                transition_idx = run_data.index.get_level_values(level='FrameIdx')[
                    run_data.index.get_level_values('RunStage') == 'Transition'][0]
                transition_paw = run_data.loc(axis=1)['initiating_limb'].loc(axis=0)['Transition', transition_idx]

                # check for sliding trials and skip them
                if (run_data.loc(axis=1)['initiating_limb'].str.contains('slid', na=False).any()
                        and run_data.loc(axis=0)['Transition', transition_idx].loc[transition_paw,'SwSt_discrete'] != '1'):
                    print(f"Skipping mouse {mouse} run {run} due to sliding trial.")
                    continue

                stance_periods_mask = run_data.loc(axis=0)['RunStart'].loc(axis=1)[transition_paw, 'SwSt'] == \
                                      locostuff['swst_vals_2025']['st']
                stance_periods_idxs = run_data.loc(axis=0)['RunStart'].index.get_level_values(level='FrameIdx')[
                    stance_periods_mask]
                stance_chunks = utils.Utils().find_blocks(stance_periods_idxs, gap_threshold=6, block_min_size=6)
                last_stance_idx = stance_chunks[-1][0]

                stride_data = run_data.loc(axis=0)['RunStart', np.arange(last_stance_idx, transition_idx)]
                bp_side = transition_paw.split('Forepaw')[1]

                if bodypart == 'Toe':
                    toe_name = 'ForepawToe' + bp_side
                    x = stride_data.loc(axis=1)[toe_name, 'x'].values
                    y = stride_data.loc(axis=1)[toe_name, 'y'].values
                    z = stride_data.loc(axis=1)[toe_name, 'z'].values
                elif bodypart == 'Tailbase':
                    x = stride_data.loc(axis=1)['Tail1', 'x'].values
                    y = stride_data.loc(axis=1)['Tail1', 'y'].values
                    z = stride_data.loc(axis=1)['Tail1', 'z'].values

                # if x[0] > 460:
                #     print(f"Mouse {mouse} run {run}")

                if bp_side == 'L':
                    # Need to flip the y coords to mirror the left side to the right side
                    mirrored_y = 2 * y_midline - y
                    y = mirrored_y

                if stsw == 'st':
                    first_x = x[0]
                    first_y = y[0]
                    first_z = z[0]
                elif stsw == 'sw':
                    # get swing start frame
                    sw_start_mask = stride_data.loc(axis=1)[transition_paw, 'SwSt_discrete'].values == locostuff['swst_vals_2025']['sw']
                    sw_start_idxs = stride_data.index.get_level_values(level='FrameIdx')[sw_start_mask]
                    if len(sw_start_idxs) == 0:
                        continue
                    elif len(sw_start_idxs) == 1:
                        sw_start_idx = sw_start_idxs[0]
                        sw_start_position = np.where(stride_data.index.get_level_values(level='FrameIdx') == sw_start_idx)[0][0]

                        first_x = x[sw_start_position]
                        first_y = y[sw_start_position]
                        first_z = z[sw_start_position]
                    else:
                        # print(f"skipped run {run} for mouse {mouse} because multiple swing starts found")
                        continue

                    # first_x = x[-1]
                    # first_y = y[-1]
                    # first_z = z[-1]
                last_x = x[-1]
                last_y = y[-1]
                last_z = z[-1]

                arr[m, r, 0, 0] = first_x # shape (n_mice, n_phase_runs, axes, position)
                arr[m, r, 0, 1] = last_x
                arr[m, r, 1, 0] = first_y
                arr[m, r, 1, 1] = last_y
                arr[m, r, 2, 0] = first_z
                arr[m, r, 2, 1] = last_z
        all_arr[phase] = arr

        if not raw:
            first_mean_x = np.nanmean(arr[:, :, 0, 0], axis=(0, 1))
            first_mean_y = np.nanmean(arr[:, :, 1, 0], axis=(0, 1))
            first_mean_z = np.nanmean(arr[:, :, 2, 0], axis=(0, 1))
            last_mean_x = np.nanmean(arr[:, :, 0, 1], axis=(0, 1))
            last_mean_y = np.nanmean(arr[:, :, 1, 1], axis=(0, 1))
            last_mean_z = np.nanmean(arr[:, :, 2, 1], axis=(0, 1))

            print(f"{phase} {bodypart} {stsw} first (x,y,z): ({first_mean_x:.1f}, {first_mean_y:.1f}, {first_mean_z:.1f})"
                    f" last (x,y,z): ({last_mean_x:.1f}, {last_mean_y:.1f}, {last_mean_z:.1f})")

            # plot x by y
            color = pu.get_color_phase(phase)
            ax.scatter(first_mean_x, first_mean_y, color=color, label='Start', marker='x', s=10, zorder=5)
            ax.scatter(last_mean_x, last_mean_y, color=color, label='End', marker='x', s=10, zorder=5)
            ax.plot([470,470], [0,50], color='k', linestyle='--', linewidth=0.5, zorder=0)
        else:
            num_mice = arr.shape[0]
            color = pu.get_color_phase(phase)
            dark_color = pu.darken_color(color, 0.5)

            # Calculate mouse means for this phase
            mouse_mean_first_x = np.nanmean(arr[:, :, 0, 0], axis=1) # shape (num_mice,)
            # mouse_mean_first_y = np.nanmean(arr[:, :, 1, 0], axis=1) # shape (num_mice,)
            mouse_mean_last_x = np.nanmean(arr[:, :, 0, 1], axis=1) # shape (num_mice,)
            # mouse_mean_last_y = np.nanmean(arr[:, :, 1, 1], axis=1) # shape (num_mice,)

            stride_lengths_mice = np.full_like(mouse_mean_first_x, np.nan)
            for mouse in range(num_mice):
                first_xs = arr[mouse, :, 0, 0] # shape (n_phase_runs,)
                last_xs = arr[mouse, :, 0, 1]

                first_xs_valid = first_xs[~np.isnan(first_xs)]
                last_xs_valid = last_xs[~np.isnan(last_xs)]

                ### Plot violin plots of start and end positions for each mouse on belt
                interval = 50/4
                position = interval if phase == phases[0] else 50 - interval
                positions = [position, position]

                jitter_strength = 2
                axs[mouse].scatter(first_xs, np.random.normal(position, jitter_strength, size=first_xs.shape), edgecolors='none', facecolors='k', label='Start', marker='.', s=4, alpha=0.5)
                axs[mouse].scatter(last_xs, np.random.normal(position, jitter_strength, size=last_xs.shape), edgecolors='k', facecolors='none', linewidths=0.3, label='End', marker='.', s=4, alpha=0.5)

                parts = axs[mouse].violinplot([first_xs_valid, last_xs_valid],
                                              positions=positions,
                                              vert=False,
                                              widths=15,
                                              showmeans=True,
                                              showmedians=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.3)
                    # pc.set_edgecolor(color)

                # Colour the mean lines
                parts['cmeans'].set_color('k')
                parts['cmeans'].set_linewidth(1)
                parts['cmeans'].set_zorder(10)

                # Colour other elements
                for partname in ('cbars', 'cmins', 'cmaxes'):
                    parts[partname].set_edgecolor(dark_color)
                    parts[partname].set_linewidth(0.5)

                axs[mouse].plot([470, 470], [0, 50], color='k', linestyle='--', linewidth=0.5, zorder=0)
                axs[mouse].set_title(f"{mice[mouse]}", fontsize=fs)

                ### Plot stride length plot (x-axis= phases, y-axis=stride length, data=mice means across runs)
                mouse_stride_lengths = last_xs - first_xs
                mouse_mean_stride_length = np.nanmean(mouse_stride_lengths)
                stride_lengths_mice[mouse] = mouse_mean_stride_length
            stride_lengths_all[phase] = stride_lengths_mice

    if not raw:
        stride_lengths_all_arr = np.array([stride_lengths_all[phases[0]], stride_lengths_all[phases[1]]]).T  # shape (num_mice, 2)
        fig, ax = plt.subplots(figsize=(4, 2.5))
        for mouse in range(stride_lengths_all_arr.shape[0]):
            ax.plot([0, 1], stride_lengths_all_arr[mouse, :], 'o-', alpha=0.3, color='grey', markersize=3, zorder=10)
        # plot boxplots without outer box
        p1_color = pu.get_color_phase(phases[0])
        p2_color = pu.get_color_phase(phases[1])
        boxprops_p1 = dict(facecolor=p1_color, color=p1_color)
        boxprops_p2 = dict(facecolor=p2_color, color=p2_color)
        medianprops_p1 = dict(color='k', linewidth=2)
        medianprops_p2 = dict(color='k', linewidth=2)
        x = np.array([0.5])

        t, p = stats.wilcoxon(stride_lengths_all_arr[:,0], stride_lengths_all_arr[:,1])




    def set_formatting(ax, violin):
        if not violin:
            ax.set_ylim(0, 50)
            ax.set_yticks(np.arange(0, 51, 25))
            ax.set_yticklabels(np.arange(0, 51, 25), fontsize=fs)
        else:
            ax.set_ylim(0, 50)
            ax.set_yticks(np.arange(0, 51, 25))
            ax.set_yticklabels(np.arange(0, 51, 25), fontsize=fs)
        ax.set_ylabel('Y (mm)', fontsize=fs)

        ax.set_xlim(0, 600)
        ax.set_xticks([0,470, 600])
        ax.set_xticklabels([0, 470, 600], fontsize=fs)
        ax.set_xlabel('X (mm)', fontsize=fs)
        ax.set_aspect('equal', adjustable='box')

    if raw:
        for mouse in range(num_mice):
            set_formatting(axs[mouse], violin=True)
        savepath = os.path.join(savedir, f"toe_distance_to_transition_{phases[0]}_{phases[1]}_{stsw}_{bodypart}_all_mice")

    else:
        set_formatting(ax, violin=False)
        savepath = os.path.join(savedir, f"toe_distance_to_transition_{phases[0]}_{phases[1]}_{stsw}_{bodypart}")

    fig.savefig(f"{savepath}.png", dpi=400)
    fig.savefig(f"{savepath}.svg", dpi=400)







def plot_toe_trajectory_real_distance(raw_data, mouse_runs, phases, savedir, fs=7, n_interp=100):
    mice = ['1035243', '1035244', '1035245', '1035246', '1035250', '1035297', '1035299', '1035301']
    labels = ['Toe', 'TransitionR', 'TransitionL']

    all_arr = {}
    for phase in phases:
        phase_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase]
        n_phase_runs = len(phase_runs)
        n_mice = len(mice)
        n_bps = 3  # Only Toe, TransitionR and TransitionL
        arr = np.full((n_mice, n_phase_runs, n_bps, 3, n_interp), np.nan)

        for m, mouse in enumerate(mice):
            mouse_data = raw_data[mouse].droplevel('Day', axis=0)
            for r, run in enumerate(phase_runs):
                if run not in mouse_runs[mouse]:
                    print(f"Run {run} not in mouse {mouse} runs, skipping.")
                    continue

                run_data = mouse_data.loc(axis=0)[run]

                transition_idx = run_data.index.get_level_values(level='FrameIdx')[run_data.index.get_level_values('RunStage') == 'Transition'][0]
                transition_paw = run_data.loc(axis=1)['initiating_limb'].loc(axis=0)['Transition', transition_idx]
                stance_periods_mask = run_data.loc(axis=0)['RunStart'].loc(axis=1)[transition_paw, 'SwSt'] == locostuff['swst_vals_2025']['st']
                stance_periods_idxs = run_data.loc(axis=0)['RunStart'].index.get_level_values(level='FrameIdx')[stance_periods_mask]
                stance_chunks = utils.Utils().find_blocks(stance_periods_idxs, gap_threshold=10, block_min_size=10)
                last_stance_idx = stance_chunks[-1][0]

                stride_data = run_data.loc(axis=0)['RunStart', np.arange(last_stance_idx, transition_idx)]

                bp_side = transition_paw.split('Forepaw')[1]
                ref_bp = 'ForepawToe' + bp_side  # e.g., 'ForepawToeR'

                for b, bp in enumerate(labels):
                    if bp == 'Toe':
                        bp_name = ref_bp
                    else:
                        bp_name = bp

                    x = stride_data.loc(axis=1)[bp_name, 'x'].values
                    y = stride_data.loc(axis=1)[bp_name, 'y'].values # dont actually use y here BUT if i did would need to mirror everything!
                    z = stride_data.loc(axis=1)[bp_name, 'z'].values

                    n_pts = len(x)

                    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any():
                        continue

                    # Interpolate to n_interp points
                    x_interp = interp1d(np.linspace(0, 1, n_pts), x, kind='linear')(np.linspace(0, 1, n_interp))
                    y_interp = interp1d(np.linspace(0, 1, n_pts), y, kind='linear')(np.linspace(0, 1, n_interp))
                    z_interp = interp1d(np.linspace(0, 1, n_pts), z, kind='linear')(np.linspace(0, 1, n_interp))

                    arr[m, r, b, 0, :] = x_interp
                    arr[m, r, b, 1, :] = y_interp
                    arr[m, r, b, 2, :] = z_interp
        all_arr[phase] = arr

        mean_x = np.nanmean(arr[:, :, :, 0, :], axis=(0, 1))
        mean_z = np.nanmean(arr[:, :, :, 2, :], axis=(0, 1))

        mean_TransitionR_x = np.nanmean(mean_x[1, :])
        mean_TransitionR_z = np.nanmean(mean_z[1, :])
        mean_TransitionL_x = np.nanmean(mean_x[2, :])
        mean_TransitionL_z = np.nanmean(mean_z[2, :])

        mean_transition = np.array([[mean_TransitionR_x, mean_TransitionL_x],
                                    [mean_TransitionR_z, mean_TransitionL_z]])


        # Now plot the mean toe trajectory for each phase
        fig, ax = plt.subplots(figsize=(2.5, 2))
        # plot toe trajectory
        color = pu.get_color_phase(phase)
        ax.plot(mean_x[0, :], mean_z[0, :], color=color, label='Toe', linewidth=1)
        # plot TransitionR and TransitionL
        ax.scatter(mean_TransitionR_x, mean_TransitionR_z, color='k', label='TransitionR', marker='x', s=50, zorder=5)

        ax.set_ylim(-0.4, 4)
        ax.set_yticks(np.arange(0, 4.1, 2))
        ax.set_yticklabels(np.arange(0, 4.1, 2), fontsize=fs)
        ax.set_ylabel('Z (mm)', fontsize=fs)
        ax.set_xlim(390,500)
        ax.set_xticks(np.arange(400, 501, 25))
        ax.set_xticklabels(np.arange(400, 501, 25), fontsize=fs)
        ax.set_xlabel('X (mm)', fontsize=fs)
        ax.tick_params(axis='both', which='major', length=4, width=1, labelsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
        # plt.tight_layout()

        save_path = os.path.join(savedir, f"toe_trajectory_{phase}")
        fig.savefig(f"{save_path}.png", dpi=400)
        fig.savefig(f"{save_path}.svg", dpi=400)

        plt.close(fig)

    # # Add extra plot for the stride lengths across all runs and mice
    #
    # stride_lengths_phases = {phase: None for phase in phases}
    # for phase in phases:
    #     toe_x = all_arr[phase][:, :, 0, 0, :]  # shape: (n_mice, n_runs, n_interp)
    #     stride_lengths = toe_x[:, :, -1] - toe_x[:, :, 0]  # shape: (n_mice, n_runs)
    #     # mean_stride_lengths = np.nanmean(stride_lengths, axis=(1)) # shape: (n_mice,)
    #     stride_lengths_phases[phase] = stride_lengths
    #
    #
    # fig, ax = plt.subplots(figsize=(2.5, 2))
    # mouse_medians_byPhase = [np.nanmedian(stride_lengths_phases[phase], axis=1) for phase in phases]
    # mouse_medians_byPhase = np.array(mouse_medians_byPhase).T  # shape: (n_mice, n_phases)
    # phase_means = np.nanmedian(mouse_medians_byPhase, axis=0)
    #
    # for m in range(mouse_medians_byPhase.shape[0]):
    #     ax.plot([0, 1], mouse_medians_byPhase[m, :], color='gray', alpha=0.5, linewidth=0.8)
    # ax.scatter([0, 1], phase_means, color='k')
    #
    # # wilcoxon test
    # stat, p_value = stats.wilcoxon(mouse_medians_byPhase[:,0], mouse_medians_byPhase[:,1])
    #
    # # Determine significance stars
    # if p_value < 0.001:
    #     stars = '***'
    # elif p_value < 0.01:
    #     stars = '**'
    # elif p_value < 0.05:
    #     stars = '*'
    # else:
    #     stars = 'ns'
    # # Plot stars above the data
    # if stars != 'ns':
    #     y_pos = max(phase_means) * 1.1
    #     ax.text(0.5, y_pos, stars,
    #             ha='center', va='bottom', fontsize=10)
    #


    # n_mice = stride_lengths_phases[phases[0]].shape[0]
    # x_positions = np.arange(n_mice)
    # offset = 0.15  # offset between phases for the same mouse
    #
    # # Store max values for each mouse to position significance stars
    # max_values = np.zeros(n_mice)
    #
    # for i, phase in enumerate(phases):
    #     color = pu.get_color_phase(phase)
    #     x_pos = x_positions + (i - 0.5) * offset  # shift left/right for each phase
    #
    #     for mouse_idx in range(n_mice):
    #         mouse_data = stride_lengths_phases[phase][mouse_idx, :]
    #         mouse_median = np.nanmedian(mouse_data)
    #
    #
    #         # Remove NaNs for this mouse
    #         valid_data = mouse_data[~np.isnan(mouse_data)]
    #
    #         if len(valid_data) > 0:
    #             # Scatter individual runs
    #             ax.scatter(np.full(len(valid_data), x_pos[mouse_idx]),
    #                        valid_data,
    #                        color=color,
    #                        alpha=0.3,
    #                        s=20)
    #
    #             # Plot mean
    #             # mean_val = np.nanmean(valid_data)
    #             median_val = np.nanmedian(valid_data)
    #             # sem_val = np.nanstd(valid_data) / np.sqrt(len(valid_data))
    #             # ci_val = 1.96 * np.nanstd(valid_data) / np.sqrt(len(valid_data))
    #             # Bootstrap confidence interval
    #             n_bootstrap = 10000
    #             bootstrap_medians = []
    #             for _ in range(n_bootstrap):
    #                 resample = np.random.choice(valid_data, size=len(valid_data), replace=True)
    #                 bootstrap_medians.append(np.median(resample))
    #
    #             ci_lower = np.percentile(bootstrap_medians, 2.5)
    #             ci_upper = np.percentile(bootstrap_medians, 97.5)
    #             ci_val = median_val - ci_lower  # for symmetric error bar, use larger of the two
    #
    #
    #             ax.plot(x_pos[mouse_idx], median_val, 'o',
    #                     color='k', markersize=8, markeredgecolor='white',
    #                     markeredgewidth=0.5)
    #
    #             # Plot SEM
    #             ax.errorbar(x_pos[mouse_idx], median_val, yerr=ci_val,
    #                         color='k', capsize=3, linewidth=1.5)
    #
    #             # Track max value for star placement
    #             max_values[mouse_idx] = max(max_values[mouse_idx], median_val + ci_val)
    #
    # # Add significance tests between phases for each mouse
    # for mouse_idx in range(n_mice):
    #     data_phase1 = stride_lengths_phases[phases[0]][mouse_idx, :]
    #     data_phase2 = stride_lengths_phases[phases[1]][mouse_idx, :]
    #
    #     # Remove NaNs
    #     valid_data1 = data_phase1[~np.isnan(data_phase1)]
    #     valid_data2 = data_phase2[~np.isnan(data_phase2)]
    #
    #     if len(valid_data1) > 1 and len(valid_data2) > 1:
    #         # Perform t-test
    #         # t_stat, p_value = stats.ttest_rel(valid_data1, valid_data2)
    #         stat, p_value = stats.wilcoxon(valid_data1, valid_data2)
    #
    #         # Determine significance stars
    #         if p_value < 0.001:
    #             stars = '***'
    #         elif p_value < 0.01:
    #             stars = '**'
    #         elif p_value < 0.05:
    #             stars = '*'
    #         else:
    #             stars = 'ns'
    #
    #         # Plot stars above the data
    #         if stars != 'ns':
    #             y_pos = max_values[mouse_idx] * 1.1
    #             ax.text(x_positions[mouse_idx], y_pos, stars,
    #                     ha='center', va='bottom', fontsize=10)
    #
    # ax.set_xlabel('Mouse')
    # ax.set_ylabel('Stride length')
    # ax.set_xticks(x_positions)
    # ax.set_xticklabels([f'{i + 1}' for i in range(n_mice)])
    #
    # plt.tight_layout()
    # plt.show()
    #








def plot_limb_positions_average(raw_data, mouse_runs, phases, savedir, fs=7, n_interp=100):
    forelimb_parts = ['ForepawToe', 'ForepawKnuckle', 'ForepawAnkle', 'ForepawKnee']
    mice = ['1035243', '1035244', '1035245', '1035246', '1035250', '1035297', '1035299', '1035301']
    coords = ['x', 'y', 'z']
    y_midline = structural_stuff['belt_width'] / 2

    all_arr = {}
    real_mid_strides = {phase: [] for phase in phases}  # Will hold [n_runs x n_bps x 2] (x, z) for each phase

    for phase in phases:
        phase_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase]
        n_phase_runs = len(phase_runs)
        n_mice = len(mice)
        n_bps = len(forelimb_parts)
        arr = np.full((n_mice, n_phase_runs, n_bps, 3, n_interp), np.nan)
        transitionR = np.full((n_mice, n_phase_runs, 3), np.nan)  # For TransitionR and TransitionL
        transitionL = np.full((n_mice, n_phase_runs, 3), np.nan)  # For TransitionR and TransitionL
        x_starts = []
        x_ends = []

        for m, mouse in enumerate(mice):
            mouse_data = raw_data[mouse].droplevel('Day', axis=0)
            for r, run in enumerate(phase_runs):
                if run not in mouse_runs[mouse]:
                    print(f"Run {run} not in mouse {mouse} runs, skipping.")
                    continue

                run_data = mouse_data.loc(axis=0)[run]
                # if sum(run_data.index.get_level_values('RunStage') == 'Transition') == 0:
                #     continue
                transition_idx = run_data.index.get_level_values(level='FrameIdx')[run_data.index.get_level_values('RunStage') == 'Transition'][0]
                transition_paw = run_data.loc(axis=1)['initiating_limb'].loc(axis=0)['Transition', transition_idx]
                stance_periods_mask = run_data.loc(axis=0)['RunStart'].loc(axis=1)[transition_paw, 'SwSt'] == locostuff['swst_vals_2025']['st']
                stance_periods_idxs = run_data.loc(axis=0)['RunStart'].index.get_level_values(level='FrameIdx')[stance_periods_mask]
                stance_chunks = utils.Utils().find_blocks(stance_periods_idxs, gap_threshold=10, block_min_size=10)
                last_stance_idx = stance_chunks[-1][0]

                # transition_idx = run_data.index.get_level_values(level='FrameIdx')[run_data.index.get_level_values('RunStage') == 'Transition'][0]
                # transition_paw = run_data.loc(axis=1)['initiating_limb'].loc(axis=0)['Transition', transition_idx]
                # stance_periods_mask = run_data.loc(axis=0)['RunStart'].loc(axis=1)[transition_paw, 'SwSt_discrete'] == locostuff['swst_vals_2025']['st']
                # stance_periods_idxs = run_data.loc(axis=0)['RunStart'].index.get_level_values(level='FrameIdx')[stance_periods_mask]
                # last_stance_idx = stance_periods_idxs[-1]
                stride_data = run_data.loc(axis=0)['RunStart', np.arange(last_stance_idx, transition_idx)]

                bp_side = transition_paw.split('Forepaw')[1]  # 'L' or 'R'

                # Get the true 50% timepoint joint positions (x, z) for each run
                xz_joint = []
                n_frames = stride_data.shape[0]
                t_idx = int(round(n_frames / 2)) if n_frames > 0 else 0
                for bp in forelimb_parts:
                    bp_name = bp + bp_side
                    try:
                        x = stride_data.loc(axis=1)[bp_name, 'x'].values
                        z = stride_data.loc(axis=1)[bp_name, 'z'].values
                        # pick midpoint of stride
                        x_val = x[t_idx] if t_idx < len(x) else np.nan
                        z_val = z[t_idx] if t_idx < len(z) else np.nan
                        xz_joint.append([x_val, z_val])
                    except Exception:
                        xz_joint.append([np.nan, np.nan])
                real_mid_strides[phase].append(xz_joint)

                # 1. Reference for x: get from initiating paw at stride start and end
                ref_bp = 'ForepawToe' + bp_side  # e.g., 'ForepawToeR'
                x_ref_all = stride_data.loc(axis=1)[ref_bp, 'x'].values
                x_start = x_ref_all[0]
                x_end = x_ref_all[-1]

                x_starts.append(x_start)
                x_ends.append(x_end)

                # 2. Stack all joints to get overall mean y/z for run (use a list or array)
                y_all_joints = []
                z_all_joints = []
                for bp in forelimb_parts:
                    bp_name = bp + bp_side
                    try:
                        y = stride_data.loc(axis=1)[bp_name, 'y'].values
                        z = stride_data.loc(axis=1)[bp_name, 'z'].values
                        if bp_side == 'L':
                            y = 2 * y_midline - y
                        y_all_joints.append(y)
                        z_all_joints.append(z)
                    except KeyError:
                        continue
                y_all_joints = np.concatenate(y_all_joints)
                z_all_joints = np.concatenate(z_all_joints)
                y_center = np.mean(y_all_joints)
                z_center = np.mean(z_all_joints)

                for b, bp in enumerate(forelimb_parts):
                    bp_name = bp + bp_side
                    x = stride_data.loc(axis=1)[bp_name, 'x'].values
                    y = stride_data.loc(axis=1)[bp_name, 'y'].values
                    z = stride_data.loc(axis=1)[bp_name, 'z'].values

                    if bp_side == 'L':
                        y = 2 * y_midline - y

                    x_TR = stride_data.loc(axis=1)['TransitionR', 'x'].values
                    x_TL = stride_data.loc(axis=1)['TransitionL', 'x'].values
                    y_TR = stride_data.loc(axis=1)['TransitionR', 'y'].values
                    y_TL = stride_data.loc(axis=1)['TransitionL', 'y'].values
                    z_TR = stride_data.loc(axis=1)['TransitionR', 'z'].values
                    z_TL = stride_data.loc(axis=1)['TransitionL', 'z'].values

                    if bp_side == 'L':
                        y_TR = 2 * y_midline - y_TR
                        y_TL = 2 * y_midline - y_TL

                    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any():
                        continue

                    n_pts = len(x)

                    # --- NORMALISE ---
                    # x: relative to stride start/end of reference toe/ankle
                    x_norm = (x - x_start) / (x_end - x_start) * 100
                    # y/z: center by whole run mean
                    y_norm = y - y_center
                    z_norm = z - z_center

                    x_TR_norm = (x_TR - x_start) / (x_end - x_start) * 100
                    x_TL_norm = (x_TL - x_start) / (x_end - x_start) * 100
                    y_TR_norm = y_TR - y_center
                    y_TL_norm = y_TL - y_center
                    z_TR_norm = z_TR - z_center
                    z_TL_norm = z_TL - z_center

                    interp_x = interp1d(np.linspace(0, 1, n_pts), x_norm, kind='linear')(np.linspace(0, 1, n_interp))
                    interp_y = interp1d(np.linspace(0, 1, n_pts), y_norm, kind='linear')(np.linspace(0, 1, n_interp))
                    interp_z = interp1d(np.linspace(0, 1, n_pts), z_norm, kind='linear')(np.linspace(0, 1, n_interp))

                    arr[m, r, b, 0, :] = interp_x
                    arr[m, r, b, 1, :] = interp_y
                    arr[m, r, b, 2, :] = interp_z

                    # summarise over time in the window (or mid-window if you prefer)
                    TR_mean = np.array([np.mean(x_TR_norm), np.mean(y_TR_norm), np.mean(z_TR_norm)])
                    TL_mean = np.array([np.mean(x_TL_norm), np.mean(y_TL_norm), np.mean(z_TL_norm)])

                    # Assign ipsi/contra based on leading paw
                    if bp_side == 'R':
                        ipsi_mean = TR_mean
                        contra_mean = TL_mean
                    else:  # 'L'
                        ipsi_mean = TL_mean
                        contra_mean = TR_mean

                    # Now store them back into the original R/L arrays so plotting code doesn't change
                    transitionR[m, r, :] = ipsi_mean
                    transitionL[m, r, :] = contra_mean

                stride_len = x_end - x_start
                if 'stride_lengths' not in locals():
                    stride_lengths = {ph: [] for ph in phases}
                stride_lengths[phase].append(stride_len)

        mean_stride_len = np.nanmean(stride_lengths[phase])

        x_starts_mean = np.nanmean(np.array(x_starts))
        x_ends_mean = np.nanmean(np.array(x_ends))
        x_distance = x_ends_mean - x_starts_mean

        # 1. Compute the average x and z for each joint at each time point
        mean_x = np.nanmean(arr[:, :, :, 0, :], axis=(0, 1)) * x_distance / 100 # shape: [n_bps, n_interp]
        mean_y = np.nanmean(arr[:, :, :, 1, :], axis=(0, 1))  # shape: [n_bps, n_interp]
        mean_z = np.nanmean(arr[:, :, :, 2, :], axis=(0, 1))  # shape: [n_bps, n_interp]

        mean_TR_x = np.nanmean(transitionR[:, :, 0], axis=(0, 1)) * x_distance / 100  # shape: [n_interp]
        mean_TR_y = np.nanmean(transitionR[:, :, 1], axis=(0, 1))  # shape: [n_interp]
        mean_TR_z = np.nanmean(transitionR[:, :, 2], axis=(0, 1))  # shape: [n_interp]
        mean_TL_x = np.nanmean(transitionL[:, :, 0], axis=(0, 1)) * x_distance / 100  # shape: [n_interp]
        mean_TL_y = np.nanmean(transitionL[:, :, 1], axis=(0, 1))  # shape: [n_interp]
        mean_TL_z = np.nanmean(transitionL[:, :, 2], axis=(0, 1))  # shape: [n_interp]

        # 2. Stick-figure sequence plot (trajectory through the stride)
        base_color = pu.get_color_phase(phase)
        cmap = pu.make_phase_cmap(base_color, light=0.80, dark=0.9)  # tweak to taste
        norm = plt.Normalize(0, n_interp - 1)

        fig, ax = plt.subplots(figsize=(6, 2))
        # Plot stick figures for each timepoint (optional: stride through in steps for less clutter)
        for t in range(n_interp):
            xs = mean_x[:, t]
            zs = mean_z[:, t]
            color = cmap(norm(t))
            ax.plot(xs, zs, marker='.', color=color, alpha=0.4, linewidth=0.5, markersize=1, zorder=1)
        # Plot the mean transition points
        transitions = np.array([np.mean([mean_TR_x, mean_TL_x]),
                                np.mean([mean_TR_z, mean_TL_z])])
        ax.scatter(transitions[0], transitions[1], marker='o', color='black', alpha=0.6, linewidth=1, s=2, zorder=2)

        desired_mm = 10  # or px, depending on your units
        # How big is this in normalised units?
        if mean_stride_len > 0:
            scale_bar_norm = 100 * (desired_mm / mean_stride_len)
        else:
            scale_bar_norm = 0
        # Draw scale bar (10 mm)
        sb_x_start = -40  # or pick a corner that works for your plot
        sb_y = -9.5  # just below your lowest stick
        ax.hlines(sb_y, sb_x_start, sb_x_start + scale_bar_norm, color='black', linewidth=2)
        ax.text(sb_x_start + scale_bar_norm / 2, sb_y - 1.5, '10 mm', ha='center', va='top', fontsize=fs)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15)
        cbar.set_label("Relative time in stride (%)", fontsize=fs)

        ax.set_title(phase, fontsize=fs)
        ax.set_xlabel('x (normalised)', fontsize=fs)
        ax.set_ylabel('z (centred)', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.set_ylim(-10, 10)
        ax.set_yticks(np.arange(-10, 11, 5))
        ax.set_yticklabels(np.arange(-10, 11, 5), fontsize=fs)
        ax.set_xlim(-50, 115)
        ax.set_xticks(np.arange(-50, 101, 50))
        ax.set_xticklabels(np.arange(-50, 101, 50), fontsize=fs)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        save_path = os.path.join(savedir, f"stick_trajectory_{phase}_color")
        plt.savefig(f"{save_path}.png", dpi=400)
        plt.savefig(f"{save_path}.svg", dpi=400)
        plt.close(fig)

        # # plot 3d version    ######### if restore this would need to mirror y!!!!
        # # 3D plot of the stick figure trajectory
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111, projection='3d')
        # for t in range(n_interp):
        #     xs = mean_x[:, t]
        #     ys = mean_y[:, t]
        #     zs = mean_z[:, t]
        #     ax.plot(xs, ys, zs, marker='.', color='k', alpha=0.4, linewidth=0.5, markersize=1, zorder=1)
        # ax.set_xlim(np.nanmin(mean_x), np.nanmax(mean_x))
        # ax.set_ylim(np.nanmin(mean_y), np.nanmax(mean_y))
        # ax.set_zlim(np.nanmin(mean_z), np.nanmax(mean_z))
        #
        # ax.set_title(phase, fontsize=fs)
        # ax.set_xlabel('x (normalised)', fontsize=fs)
        # ax.set_ylabel('y (centred)', fontsize=fs)
        # ax.set_zlabel('z (centred)', fontsize=fs)
        # ax.tick_params(axis='both', which='major', labelsize=fs)
        # # Calculate the ranges for each axis
        # x_range = np.nanmax(mean_x) - np.nanmin(mean_x)
        # y_range = np.nanmax(mean_y) - np.nanmin(mean_y)
        # z_range = np.nanmax(mean_z) - np.nanmin(mean_z)
        # ax.set_box_aspect([x_range, y_range, z_range])  # Matches real proportions!
        # #        plt.tight_layout()
        # plt.savefig(f"{savedir}/stick_trajectory_3d_{phase}.png", dpi=400)
        # plt.close(fig)

        # 3. Mean stick-figure (average posture across stride)
        # Recenter x to the toe at each timepoint before averaging
        arr_toe_x = arr[:, :, 0, 0, :]  # [mouse, run, time] -- toe's x
        # Subtract the toe x at each timepoint from all bodyparts for each run/mouse
        arr_x_centered = arr[:, :, :, 0, :] - arr_toe_x[:, :, np.newaxis, :]
        # Now avg_x gives you mean shape with toe at x=0 for each timepoint
        mean_x_centered = np.nanmean(arr_x_centered, axis=(0, 1))  # shape: [n_bps, n_interp]
        avg_x = np.nanmean(mean_x_centered, axis=1)
        avg_z = np.nanmean(mean_z, axis=1)
        std_x = np.nanstd(mean_x_centered, axis=1)
        std_z = np.nanstd(mean_z, axis=1)


        fig, ax = plt.subplots(figsize=(2.5, 2))
        ax.plot(avg_x, avg_z, marker='o', color='black', linewidth=1, markersize=2, zorder=2)
        # Plot error bars for std deviation
        eb = ax.errorbar(avg_x, avg_z, xerr=std_x, yerr=std_z, fmt='none', ecolor='grey', elinewidth=0.5, capsize=2, zorder=1)
        for bar in eb[2]:
            bar.set_linestyle('--')

        ax.set_title(phase, fontsize=fs)
        ax.set_xlabel('x (normalised)', fontsize=fs)
        ax.set_ylabel('z (centred)', fontsize=fs)
        ax.set_ylim(-8,8)
        ax.set_yticks(np.arange(-8, 9, 8))
        ax.set_yticklabels(np.arange(-8, 9, 8), fontsize=fs)
        ax.set_xlim(-30, 10)
        ax.set_xticks(np.arange(-30, 11, 10))
        ax.set_xticklabels(np.arange(-30, 11, 10), fontsize=fs)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        plt.tight_layout()
        save_path = os.path.join(savedir, f"stick_shape_{phase}")
        plt.savefig(f"{save_path}.png", dpi=400)
        plt.savefig(f"{save_path}.svg", dpi=400)
        plt.close(fig)

        all_arr[phase] = arr

    fig, ax = plt.subplots(figsize=(3, 4))
    colors = {'APA2': 'blue', 'Wash2': 'red'}
    for phase in phases:
        arr_mid = np.array(real_mid_strides[phase])  # shape: [n_runs, n_bps, 2]
        mean_x = np.nanmean(arr_mid[:, :, 0], axis=0)
        mean_z = np.nanmean(arr_mid[:, :, 1], axis=0)
        ax.plot(mean_x, mean_z, marker='o', color=colors.get(phase, 'k'), linewidth=2, markersize=6, label=phase)
        for i, bp in enumerate(forelimb_parts):
            ax.text(mean_x[i], mean_z[i], bp, fontsize=8, color=colors.get(phase, 'k'))
    ax.set_xlabel('x (real units)', fontsize=fs)
    ax.set_ylabel('z (real units)', fontsize=fs)
    ax.legend()
    ax.set_title('Mean stick figure at 50% stride (real units)', fontsize=fs)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "stick_shape_50pct_real.png"), dpi=400)
    plt.savefig(os.path.join(savedir, "stick_shape_50pct_real.svg"), dpi=400)
    plt.close(fig)


def plot_nose_back_tail_averages(raw_data, mouse_runs, phases, savedir, fs=7, n_interp=100):
    body_parts = [
        'Nose', 'EarR', 'EarL',
        'Back1', 'Back2', 'Back3', 'Back4', 'Back5', 'Back6', 'Back7', 'Back8', 'Back9', 'Back10', 'Back11', 'Back12',
        'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7', 'Tail8', 'Tail9', 'Tail10', 'Tail11', 'Tail12',
    ]
    ymidline = structural_stuff['belt_width'] / 2
    mice = ['1035243', '1035244', '1035245', '1035246', '1035250', '1035297', '1035299', '1035301']

    # --- 1. Calculate mean back/tail lengths and nose/ear offsets from Back1 ---
    back_lengths, tail_lengths = [], []
    nose_offsets, earr_offsets, earl_offsets = [], [], []
    for mouse in mice:
        mouse_data = raw_data[mouse].droplevel('Day', axis=0)
        for phase in phases:
            phase_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase]
            for run in phase_runs:
                if run not in mouse_runs[mouse]:
                    continue
                run_data = mouse_data.loc(axis=0)[run]
                try:
                    back1_x = run_data.loc(axis=1)['Back1', 'x'].values
                    back12_x = run_data.loc(axis=1)['Back12', 'x'].values
                    tail1_x = run_data.loc(axis=1)['Tail1', 'x'].values
                    tail12_x = run_data.loc(axis=1)['Tail12', 'x'].values
                    nose_x = run_data.loc(axis=1)['Nose', 'x'].values
                    earr_x = run_data.loc(axis=1)['EarR', 'x'].values
                    earl_x = run_data.loc(axis=1)['EarL', 'x'].values

                    back_lengths.append(np.nanmean(np.abs(back12_x - back1_x)))
                    tail_lengths.append(np.nanmean(np.abs(tail12_x - tail1_x)))
                    nose_offsets.append(np.nanmean(back1_x - nose_x))
                    earr_offsets.append(np.nanmean(back1_x - earr_x))
                    earl_offsets.append(np.nanmean(back1_x - earl_x))
                except Exception:
                    continue

    mean_back_length = np.nanmean(back_lengths)
    mean_tail_length = np.nanmean(tail_lengths)
    mean_nose_offset = np.nanmean(nose_offsets)
    mean_earr_offset = np.nanmean(earr_offsets)
    mean_earl_offset = np.nanmean(earl_offsets)

    # --- 2. Set up template x positions ---
    n_back = 12
    n_tail = 12
    back_xs = np.linspace(0, mean_back_length, n_back)      # Back1 = 0, Back12 = mean_back_length
    tail_xs = np.linspace(mean_back_length, mean_back_length + mean_tail_length, n_tail)
    bp_xvals = {
        'Nose': 0 + mean_nose_offset,
        'EarR': 0 + mean_earr_offset,
        'EarL': 0 + mean_earl_offset,
    }
    for i in range(n_back):
        bp_xvals[f'Back{i + 1}'] = back_xs[i]
    for i in range(n_tail):
        bp_xvals[f'Tail{i + 1}'] = tail_xs[i]

    all_arr = {}
    stride_lengths = {ph: [] for ph in phases}
    phase_shapes = {}
    phase_transition_means = {}  # will hold {'R': [x,y,z], 'L': [x,y,z], 'avg': [x,y,z]} per phase

    for phase in phases:
        phase_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase]
        n_phase_runs = len(phase_runs)
        n_mice = len(mice)
        n_bps = len(body_parts)
        arr = np.full((n_mice, n_phase_runs, n_bps, 3, n_interp), np.nan)

        # store TransitionR/L relative positions per run (to Back1 at transition frame; y centered; z raw)
        ipsi_rel = []
        contra_rel = []

        for m, mouse in enumerate(mice):
            mouse_data = raw_data[mouse].droplevel('Day', axis=0)
            for r, run in enumerate(phase_runs):
                print(f"Processing mouse {mouse}, run {run}, phase {phase}")
                if run not in mouse_runs[mouse]:
                    print(f"Run {run} not in mouse {mouse} runs, skipping.")
                    continue

                run_data = mouse_data.loc(axis=0)[run]
                transition_idx = run_data.index.get_level_values(level='FrameIdx')[run_data.index.get_level_values('RunStage') == 'Transition'][0]
                transition_paw = run_data.loc(axis=1)['initiating_limb'].loc(axis=0)['Transition', transition_idx]
                stance_periods_mask = run_data.loc(axis=0)['RunStart'].loc(axis=1)[transition_paw, 'SwSt'] == locostuff['swst_vals_2025']['st']
                stance_periods_idxs = run_data.loc(axis=0)['RunStart'].index.get_level_values(level='FrameIdx')[stance_periods_mask]
                stance_chunks = utils.Utils().find_blocks(stance_periods_idxs, gap_threshold=10, block_min_size=10)
                last_stance_idx = stance_chunks[-1][0]
                stride_data = run_data.loc(axis=0)['RunStart', np.arange(last_stance_idx, transition_idx)]

                transition_paw_side = transition_paw.split('Forepaw')[1]  # 'L' or 'R'

                # Reference: Back12 x at stride start/end
                ref_bp = 'Back12'
                try:
                    x_ref_all = stride_data.loc(axis=1)[ref_bp, 'x'].values
                except KeyError:
                    continue
                x_start = x_ref_all[0]
                x_end = x_ref_all[-1]

                stride_len = x_end - x_start
                stride_lengths[phase].append(stride_len)

                # y centering reference for this stride window (consistent with body normalization)
                #y_ref = np.mean(np.mean(stride_data.loc(axis=1)[['TransitionR', 'TransitionL'], 'y'].values, axis=1))
                y_ref = ymidline

                # collect transition R/L positions relative to Back1 at the transition frame
                back1_x_t = np.mean(stride_data.loc(axis=1)['Back1','x'].values)

                # TransitionR
                tR_x = np.mean(run_data.loc(axis=1)['TransitionR', 'x'].values)
                tR_y = np.mean(run_data.loc(axis=1)['TransitionR', 'y'].values)
                tR_z = np.mean(run_data.loc(axis=1)['TransitionR', 'z'].values)

                # TransitionL
                tL_x = np.mean(run_data.loc(axis=1)['TransitionL', 'x'].values)
                tL_y = np.mean(run_data.loc(axis=1)['TransitionL', 'y'].values)
                tL_z = np.mean(run_data.loc(axis=1)['TransitionL', 'z'].values)

                if transition_paw_side == 'L':
                    tR_y = 2 * ymidline - tR_y
                    tL_y = 2 * ymidline - tL_y

                # TR_rel = (tR_x - back1_x_t, tR_y - y_ref, tR_z)
                # TL_rel = (tL_x - back1_x_t, tL_y - y_ref, tL_z)
                TR_rel = (back1_x_t - tR_x, y_ref - tR_y, tR_z)
                TL_rel = (back1_x_t - tL_x, y_ref - tL_y, tL_z)

                if transition_paw_side == 'R':
                    ipsi_rel.append(TR_rel)
                    contra_rel.append(TL_rel)
                else:
                    ipsi_rel.append(TL_rel)
                    contra_rel.append(TR_rel)

                # fill arrays for body parts
                lateral_swap = {'EarR': 'EarL', 'EarL': 'EarR'}
                for b, bp in enumerate(body_parts):
                    bp_fetch = bp
                    if transition_paw_side == 'L' and bp in lateral_swap:
                        bp_fetch = lateral_swap[bp]  # relabel for mirrored trials

                    try:
                        x = stride_data.loc(axis=1)[bp_fetch, 'x'].values
                        y = stride_data.loc(axis=1)[bp_fetch, 'y'].values
                        z = stride_data.loc(axis=1)[bp_fetch, 'z'].values
                    except KeyError:
                        continue
                    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any():
                        continue
                    n_pts = len(x)

                    if transition_paw_side == 'L': # mirror y for left side
                        y = 2 * ymidline - y

                    # fixed template x; interpolate y/z as before
                    x_fixed = bp_xvals[bp]
                    z_norm = z  # already relative to belt surface
                    interp_x = np.full(n_interp, x_fixed)
                    interp_y = interp1d(np.linspace(0, 1, n_pts), y - y_ref, kind='linear')(np.linspace(0, 1, n_interp))
                    interp_z = interp1d(np.linspace(0, 1, n_pts), z_norm, kind='linear')(np.linspace(0, 1, n_interp))
                    arr[m, r, b, 0, :] = interp_x
                    arr[m, r, b, 1, :] = interp_y
                    arr[m, r, b, 2, :] = interp_z

        # means across runs and mice
        mean_x = np.nanmean(arr[:, :, :, 0, :], axis=(0, 1))
        mean_z = np.nanmean(arr[:, :, :, 2, :], axis=(0, 1))
        mean_y = np.nanmean(arr[:, :, :, 1, :], axis=(0, 1))
        std_x = np.nanstd(arr[:, :, :, 0, :], axis=(0, 1))
        std_z = np.nanstd(arr[:, :, :, 2, :], axis=(0, 1))

        # store mean shape for this phase for summary plot
        avg_x = np.array([bp_xvals[bp] for bp in body_parts])
        avg_z = np.nanmean(mean_z, axis=1)
        avg_y = np.nanmean(mean_y, axis=1)
        phase_shapes[phase] = (avg_x, avg_y, avg_z)

        # compute per-phase transition means for R, L, and average
        mean_tx_ipsi = np.nanmean(np.array(ipsi_rel), axis=0)
        mean_tx_contra = np.nanmean(np.array(contra_rel), axis=0)
        mean_tx_avg = np.nanmean(np.vstack([mean_tx_ipsi, mean_tx_contra]), axis=0)

        phase_transition_means[phase] = {'ipsi': mean_tx_ipsi, 'contra': mean_tx_contra, 'avg': mean_tx_avg}

        # per-phase plot of mean shape (side view x vs z) with averaged transition marker
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(avg_x, avg_z, marker='o', color='black', linewidth=2, markersize=4, zorder=2)
        for i, bp in enumerate(body_parts):
            ax.text(avg_x[i], avg_z[i], bp, fontsize=fs-1, ha='right', va='center', color='black', zorder=3)

        tx = phase_transition_means[phase]['avg']
        if not np.isnan(tx).any():
            ax.scatter(tx[0], tx[2], s=60, marker='*', color='magenta', linewidths=0, zorder=5, label='Transition (avg)')
            ax.legend(fontsize=fs-1, loc='best')

        ax.set_title(f"{phase} (mean shape)", fontsize=fs)
        ax.set_xlabel('x (mm, fixed template)', fontsize=fs)
        ax.set_ylabel('z', fontsize=fs)
        ax.invert_xaxis()
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        plt.tight_layout()
        save_path = os.path.join(savedir, f"body_shape_{phase}")
        plt.savefig(f"{save_path}.png", dpi=400)
        plt.savefig(f"{save_path}.svg", dpi=400)
        plt.close(fig)

        all_arr[phase] = arr

    # --- Plot APA and Wash phases together on the same figure ---
    for type in ['side', 'front', 'top']:
        fig, ax = plt.subplots(figsize=(6, 2))
        for p in phases:
            avg_x, avg_y, avg_z = phase_shapes[p]
            if type == 'side':
                x_plot = avg_x
                y_plot = avg_z
            elif type == 'front':
                x_plot = avg_y
                y_plot = avg_z
            elif type == 'top':
                x_plot = avg_x
                y_plot = avg_y
            color = pu.get_color_phase(p)
            ax.plot(x_plot[[0,1]], y_plot[[0,1]], marker='o', color=color, linewidth=0.5, markersize=1, label=p)
            ax.plot(x_plot[[0,2]], y_plot[[0,2]], marker='o', color=color, linewidth=0.5, markersize=1)
            ax.plot(x_plot[[1,2]], y_plot[[1,2]], marker='o', color=color, linewidth=0.5, markersize=1)
            ax.plot(x_plot[[0,3]], y_plot[[0,3]], marker='o', color=color, linewidth=0.5, markersize=1)
            ax.plot(x_plot[[1,3]], y_plot[[1,3]], marker='o', color=color, linewidth=0.5, markersize=1)
            ax.plot(x_plot[[2,3]], y_plot[[2,3]], marker='o', color=color, linewidth=0.5, markersize=1)
            ax.plot(x_plot[3:], y_plot[3:], marker='o', color=color, linewidth=0.5, markersize=1)

            # overlay transitions
            txs = phase_transition_means[p]
            if type == 'side':
                tx = txs['avg']
                if not np.isnan(tx).any():
                    ax.scatter(tx[0], tx[2], s=50, marker='.', color=color, linewidths=0, zorder=6)
            elif type == 'front':
                ax.plot([txs['ipsi'][1], txs['contra'][1]], [txs['ipsi'][2], txs['contra'][2]], 'o-', color=color, linewidth=0.5, markersize=1, zorder=5)
            elif type == 'top':
                ax.plot([txs['ipsi'][0], txs['contra'][0]], [txs['ipsi'][1], txs['contra'][1]], 'o-', color=color, linewidth=0.5, markersize=1, zorder=5)


        if type == 'side':
            ax.set_xlabel('x (mm, fixed template)', fontsize=fs)
            ax.set_ylabel('z', fontsize=fs)
            ax.set_xlim(-60, 150)
            ax.set_xticks(np.arange(-50, 151, 25))
            ax.set_xticklabels(np.arange(-50, 151, 25), fontsize=fs)
            ax.set_ylim(-5, 40)
            ax.set_yticks(np.arange(0, 41, 20))
            ax.set_yticklabels(np.arange(0, 41, 20), fontsize=fs)
            ax.invert_xaxis()
        elif type == 'front':
            ax.set_xlabel('y (centered)', fontsize=fs)
            ax.set_ylabel('z', fontsize=fs)
            ax.set_xlim(-30, 30)
            ax.set_xticks(np.arange(-25, 26, 25))
            ax.set_xticklabels(np.arange(-25, 26, 25), fontsize=fs)
            ax.set_ylim(-5, 40)
            ax.set_yticks(np.arange(0, 41, 20))
            ax.set_yticklabels(np.arange(0, 41, 20), fontsize=fs)
        elif type == 'top':
            ax.set_xlabel('x (mm, fixed template)', fontsize=fs)
            ax.set_ylabel('y (centered)', fontsize=fs)
            ax.set_xlim(-60, 150)
            ax.set_xticks(np.arange(-50, 151, 25))
            ax.set_xticklabels(np.arange(-50, 151, 25), fontsize=fs)
            ax.set_ylim(-30, 30)
            ax.set_yticks(np.arange(-25, 26, 25))
            ax.set_yticklabels(np.arange(-25, 26, 25), fontsize=fs)
            ax.invert_xaxis()

        ax.set_aspect('equal')
        # dedup legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        plt.tight_layout()
        save_path = os.path.join(savedir, f"body_shape_APA_vs_Wash{type.capitalize()}")
        plt.savefig(f"{save_path}.png", dpi=400)
        plt.savefig(f"{save_path}.svg", dpi=400)
        plt.close(fig)

    # 3D Plot of the average body shape with TransitionR/L markers
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    for phase in phases:
        avg_x, avg_y, avg_z = phase_shapes[phase]
        color = pu.get_color_phase(phase)
        xs = avg_x * -1
        ys = avg_y
        zs = avg_z

        ax.plot(xs[[0, 1]], ys[[0, 1]], zs[[0, 1]], 'o-', color=color, linewidth=0.5, markersize=1, label=phase, zorder=10)
        ax.plot(xs[[0, 2]], ys[[0, 2]], zs[[0, 2]], 'o-', color=color, linewidth=0.5, markersize=1)
        ax.plot(xs[[1, 2]], ys[[1, 2]], zs[[1, 2]], 'o-', color=color, linewidth=0.5, markersize=1)
        ax.plot(xs[[0, 3]], ys[[0, 3]], zs[[0, 3]], 'o-', color=color, linewidth=0.5, markersize=1)
        ax.plot(xs[[1, 3]], ys[[1, 3]], zs[[1, 3]], 'o-', color=color, linewidth=0.5, markersize=1)
        ax.plot(xs[[2, 3]], ys[[2, 3]], zs[[2, 3]], 'o-', color=color, linewidth=0.5, markersize=1)
        ax.plot(xs[3:], ys[3:], zs[3:], 'o-', color=color, linewidth=0.5, markersize=1)

        # overlay TransitionR/L stars in 3D (note x negation)
        txs = phase_transition_means[phase]
        ax.plot([-txs['ipsi'][0], -txs['contra'][0]], [txs['ipsi'][1], txs['contra'][1]], [txs['ipsi'][2], txs['contra'][2]], 'o-', color=color, linewidth=0.5, markersize=1, zorder=15)


    ax.set_xlabel('x (mm, fixed template)', fontsize=fs)
    ax.set_ylabel('y (centered)', fontsize=fs)
    ax.set_zlabel('z (centred)', fontsize=fs)
    ax.set_title('Average body shape (3D)', fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.view_init(elev=5.875059194642063, azim=-25.203643447562627)
    x_range = np.nanmax(avg_x) - np.nanmin(avg_x)
    y_range = np.nanmax(avg_y) - np.nanmin(avg_y)
    z_range = np.nanmax(avg_z) - np.nanmin(avg_z)
    ax.set_box_aspect([x_range, y_range, z_range])
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    # dedup legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=fs)
    plt.tight_layout()
    save_path = os.path.join(savedir, "body_shape_3D")
    plt.savefig(f"{save_path}.png", dpi=400)
    plt.savefig(f"{save_path}.svg", dpi=400)
    plt.close(fig)

    # HEAD PLOTS
    head_labels = ['Nose', 'EarR', 'EarL', 'Back1']

    # Gather for each phase: mean x/y/z for Nose, EarR, EarL, Back1
    head_data = {}
    for phase in phases:
        avg_x, avg_y, avg_z = phase_shapes[phase]
        idxs = [body_parts.index(bp) for bp in head_labels]
        head_xyz = np.stack([
            avg_x[idxs],
            avg_y[idxs],
            avg_z[idxs]
        ], axis=1)  # shape (4, 3)
        head_data[phase] = head_xyz

    # Align all heads so Back1 is at (0, 0, 0)
    for phase in phases:
        base = head_data[phase][3, :]  # Back1
        head_data[phase] = head_data[phase] - base

    # 3D head triangle plots
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')

    for i, phase in enumerate(phases):
        xyz = head_data[phase]
        color = pu.get_color_phase(phase)
        xs, ys, zs = xyz[:3, 0] * -1, xyz[:3, 1], xyz[:3, 2]

        if i == 1:
            zs = zs + 0.02  # small lift to help visibility

        ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), np.append(zs, zs[0]), 'o-', color=color, label=phase, zorder=10 + i)
        verts = [list(zip(xs, ys, zs))]
        poly = Poly3DCollection(verts, color=color, alpha=0.9, zorder=10 + i)
        ax.add_collection3d(poly)
        ax.scatter(0, 0, 0, color='k', marker='x', s=30, alpha=1, zorder=11 + i)

    ax.set_xlabel('x (mm, rel. to Back1)', fontsize=fs)
    ax.set_ylabel('y (mm, rel. to Back1)', fontsize=fs)
    ax.set_zlabel('z (mm, rel. to Back1)', fontsize=fs)
    ax.set_title('Head angle (3D)', fontsize=fs)
    ax.view_init(elev=8.78236413268246, azim=-83.0788319960298)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=fs)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "head_angle_3D.png"), dpi=400)
    plt.savefig(os.path.join(savedir, "head_angle_3D.svg"), dpi=400)
    plt.close(fig)

    print("Saved all plots to", savedir)


def handedness(raw_data):
    mice = raw_data.keys()
    StartingPaws = pd.DataFrame(index=np.arange(0,160), columns=mice)
    TransitioningPaws = pd.DataFrame(index=np.arange(0,160), columns=mice)
    for midx, mouse in enumerate(mice):
        mouse_data = raw_data[mouse]
        initiating_limb = mouse_data.loc(axis=1)['initiating_limb']
        # Stack index level 0 'Day' together to remove this index
        initiating_limb = initiating_limb.droplevel('Day', axis=0)
        runstage_vals = initiating_limb.index.get_level_values('RunStage')
        runstage_series = pd.Series(runstage_vals, index=initiating_limb.index)

        # Find starting paws
        runstart_frames_mask = np.logical_and(runstage_series == 'RunStart',runstage_series.shift(1) == 'TrialStart').values
        runstart_frames = initiating_limb.index.get_level_values('FrameIdx')[runstart_frames_mask]

        starting = initiating_limb.loc(axis=0)[:,:,runstart_frames].droplevel('RunStage', axis=0).droplevel('FrameIdx', axis=0)
        starting_nums = starting.replace({'ForepawR': micestuff['LR']['ForepawToeR'], 'ForepawL': micestuff['LR']['ForepawToeL']})
        StartingPaws[mouse] = starting_nums

        # Find transitioning paws
        transition_frames_mask = np.logical_and(runstage_series == 'Transition', runstage_series.shift(1) == 'RunStart').values
        transition_frames = initiating_limb.index.get_level_values('FrameIdx')[transition_frames_mask]

        transitioning = initiating_limb.loc(axis=0)[:,:,transition_frames].droplevel('RunStage', axis=0).droplevel('FrameIdx', axis=0)
        # Replace 'R' with 1 and 'L' with 0
        transitioning_nums = transitioning.replace({'ForepawR': micestuff['LR']['ForepawToeR'], 'ForepawL': micestuff['LR']['ForepawToeL']})
        TransitioningPaws[mouse] = transitioning_nums

    savedir = os.path.join(r"H:\Characterisation", 'Handedness')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    across_mice_L_R_percentage(TransitioningPaws, savedir)
    start_transition_consistency(StartingPaws, TransitioningPaws, savedir)

def plot_combo_sequence_per_mouse(StartingPaws, TransitioningPaws, savedir):
    import os
    import matplotlib.pyplot as plt

    combo_map = {
        (1, 1): 0,  # Left → Left
        (1, 2): 1,  # Left → Right
        (2, 1): 2,  # Right → Left
        (2, 2): 3  # Right → Right
    }
    combo_labels = [
        'Left start: Left transition',
        'Left start: Right transition',
        'Right start: Left transition',
        'Right start: Right transition'
    ]


    for mouse in StartingPaws.columns:
        start = StartingPaws[mouse]
        trans = TransitioningPaws[mouse]
        n_runs = len(start)

        combo_sequence = []
        valid_indices = []

        for i in range(n_runs):
            if pd.notna(start[i]) and pd.notna(trans[i]):
                key = (int(start[i]), int(trans[i]))
                if key in combo_map:
                    combo_sequence.append(combo_map[key])
                    valid_indices.append(i)

        # Plot
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.step(valid_indices, combo_sequence, where='mid', color='black', linewidth=1)

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(combo_labels, fontsize=7)
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlabel('Run', fontsize=7)
        ax.set_title(mouse, fontsize=7)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        plt.tight_layout()

        save_path = os.path.join(savedir, f"{mouse}_ComboStepPlot")
        plt.savefig(f"{save_path}.png", dpi=300)
        plt.savefig(f"{save_path}.svg", dpi=300)
        plt.close()

def start_transition_consistency(StartingPaws, TransitioningPaws, savedir):
    combo_data = []
    for mouse in StartingPaws.columns:
        df = pd.DataFrame({
            'Start': StartingPaws[mouse],
            'Transition': TransitioningPaws[mouse]
        }).dropna()

        df['Combo'] = df.apply(lambda row: f"{'Left' if row['Start'] == 1 else 'Right'} start: " +
                                           f"{'Left' if row['Transition'] == 1 else 'Right'} transition", axis=1)
        combo_counts = df['Combo'].value_counts(normalize=True)
        for combo, val in combo_counts.items():
            combo_data.append({'Mouse': mouse, 'Combo': combo, 'Proportion': val})

    combo_df = pd.DataFrame(combo_data)
    combo_pivot = combo_df.pivot(index='Mouse', columns='Combo', values='Proportion').fillna(0)

    # Define order and styling
    combo_order = [
        'Left start: Left transition',    # solid black
        'Left start: Right transition',   # black with white hatch
        'Right start: Left transition',   # white with black hatch
        'Right start: Right transition'   # solid white
    ]
    hatches = ['', '////', '////', '']
    facecolors = ['grey', 'grey', 'white', 'white']
    edgecolors = ['black', 'black', 'black', 'black']  # outline to contrast with fill

    fig, ax = plt.subplots(figsize=(5, 2))
    lefts = np.zeros(len(combo_pivot))

    for label, hatch, facecolor, edgecolor in zip(combo_order, hatches, facecolors, edgecolors):
        vals = combo_pivot[label].values if label in combo_pivot.columns else np.zeros(len(combo_pivot))
        ax.barh(combo_pivot.index, vals, left=lefts,
                color=facecolor, edgecolor=edgecolor, hatch=hatch, linewidth=1, label=label)
        lefts += vals

    ax.set_xlabel('Proportion of Runs', fontsize=7)
    ax.set_ylabel('Mouse', fontsize=7)
    ax.set_title('Start vs Transition Paw Combinations', fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.invert_yaxis()  # first mouse on top

    legend_elements = [
        Patch(facecolor=fc, edgecolor=ec, hatch=h, label=lbl, linewidth=1)
        for lbl, h, fc, ec in zip(combo_order, hatches, facecolors, edgecolors)
    ]
    ax.legend(handles=legend_elements, title='Combination',
              bbox_to_anchor=(1.05, 1), loc='upper left',
              frameon=False, fontsize=7, title_fontsize=7)

    plt.tight_layout()

    save_path = os.path.join(savedir, "StartTransition_PawCombination_perMouse")
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.savefig(f"{save_path}.svg", dpi=300)
    plt.close()

def across_mice_L_R_percentage(TransitioningPaws, savedir):
    # Clean
    valid_trans = TransitioningPaws.dropna(how='all', axis=0).dropna(how='all', axis=1)
    rounded_trans = valid_trans.round().astype('Int64')

    # Long format
    df_long = rounded_trans.reset_index(drop=True).melt(var_name='Mouse', value_name='Paw')
    df_long = df_long.dropna()
    df_long['Paw'] = df_long['Paw'].map({1: 'Left', 2: 'Right'})

    # Per-mouse proportions
    mouse_counts = df_long.groupby(['Mouse', 'Paw']).size().unstack(fill_value=0)
    mouse_props = (mouse_counts.T / mouse_counts.sum(axis=1)).T.fillna(0)
    mouse_props_sorted = mouse_props.sort_index(ascending=False)

    # Set up plot
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))

    # Custom stacked bars with solid fill and hatching
    left_vals = mouse_props_sorted['Left']
    right_vals = mouse_props_sorted['Right']
    y_pos = np.arange(len(mouse_props_sorted))

    # Plot left (solid black)
    ax.barh(y_pos, left_vals, color='black', edgecolor='black', label='Left')

    # Plot right (striped black) stacked on top
    ax.barh(y_pos, right_vals, left=left_vals, color='white', hatch='////', edgecolor='black', label='Right')

    # Axis formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mouse_props_sorted.index, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Proportion of Runs', fontsize=7)
    ax.set_ylabel('Mouse', fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)

    # Legend outside
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Left'),
        Patch(facecolor='white', edgecolor='black', hatch='////', label='Right')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=7, title_fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(savedir, f"TransitioningStride_perMouse_LR_proportions")
    plt.savefig(f"{save_path}.png", dpi=300)
    plt.savefig(f"{save_path}.svg", dpi=300)
    plt.close()


def plot_gait_features_corrs(features, stride, phases, savedir, fs=7):
    mice = features.index.get_level_values(1).unique()
    feature_list = ['Stride duration', 'Walking speed', 'Duty factor', 'Cadence',
                    'Propulsion duration', 'Brake duration', 'Swing velocity']
    short_to_long = {v: k for k, v in short_names.items()}

    # tick formatter: no dps when |x| >= 1, else 1 dp; also clean up -0
    def smart_fmt(x, _pos):
        if abs(x) >= 1:
            s = f"{int(round(x))}"
        else:
            s = f"{x:.1f}"
        return "0" if s in ("-0", "-0.0", "0.0") else s

    for feat in feature_list:
        fig, ax = plt.subplots(figsize=(2, 1.5))
        all_means = []
        all_CIs = []

        for mouse in mice:
            long_feat_name = short_to_long[feat]  # short key used in df columns
            data = features.loc(axis=0)[stride, mouse].loc(axis=1)[long_feat_name]

            phase_means = np.zeros(len(phases))
            phase_CIs = np.zeros(len(phases))
            for i, phase in enumerate(phases):
                phase_runs = expstuff['condition_exp_runs']['APAChar']['Extended'][phase]
                phase_data = data.loc(axis=0)[phase_runs[0]:phase_runs[-1]]

                phase_means[i] = np.nanmean(phase_data)
                phase_CIs[i] = np.nanstd(phase_data) / np.sqrt(len(phase_data)) * 1.96  # 95% CI

            all_means.append(phase_means)
            all_CIs.append(phase_CIs)

            mouse_marker = pu.get_marker_style_mice(mouse)
            mouse_colour = pu.get_color_mice(mouse)

            ax.errorbar(phase_means[0], phase_means[1],
                        xerr=phase_CIs[0], yerr=phase_CIs[1],
                        fmt='.', color=mouse_colour, alpha=1,
                        markersize=6, capsize=1, elinewidth=0.5, markeredgewidth=0.5, label=mouse) # markersize was 5


        all_means = np.array(all_means)
        all_CIs = np.array(all_CIs)

        total_mean = np.nanmean(all_means, axis=0)
        print(f"Total means for {feat}:\nAPA: {total_mean[0]:.2f}, Wash: {total_mean[1]:.2f}")

        # plot the overall mean as a large dot
        ax.plot(total_mean[0], total_mean[1], marker='x', color='black', markersize=8, label='Mean across mice', zorder=3)

        # compute independent x and y ranges with 10 percent padding
        x_high = np.nanmax(all_means[:, 0] + all_CIs[:, 0])
        x_low  = np.nanmin(all_means[:, 0] - all_CIs[:, 0])
        y_high = np.nanmax(all_means[:, 1] + all_CIs[:, 1])
        y_low  = np.nanmin(all_means[:, 1] - all_CIs[:, 1])

        x_span = x_high - x_low
        y_span = y_high - y_low

        # choose increments per axis
        def choose_inc(span):
            if span < 1:
                return 0.1
            elif span < 10:
                return 2
            elif span < 100:
                return 25
            elif span < 500:
                return 250
            else:
                return 500

        x_inc = choose_inc(x_span)
        y_inc = choose_inc(y_span)

        # round outward to chosen increments
        x_min = np.floor(x_low / x_inc) * x_inc
        x_max = np.ceil(x_high / x_inc) * x_inc
        y_min = np.floor(y_low / y_inc) * y_inc
        y_max = np.ceil(y_high / y_inc) * y_inc

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # ticks - let Matplotlib place them from our increments
        ax.set_xticks(np.arange(x_min, x_max + 0.5 * x_inc, x_inc))
        ax.set_yticks(np.arange(y_min, y_max + 0.5 * y_inc, y_inc))

        # smart tick formatting for decimals
        ax.xaxis.set_major_formatter(FuncFormatter(smart_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(smart_fmt))

        # draw equality line within visible overlap
        lo = max(x_min, y_min)
        hi = min(x_max, y_max)
        if hi > lo:
            ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=0.5, color='gray', zorder=0)

        ax.set_xlabel(phases[0], fontsize=fs)
        ax.set_ylabel(phases[1], fontsize=fs)
        ax.set_title(feat, fontsize=fs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.set_aspect('auto')  # ensure no forced equal aspect

        # ax.legend(fontsize=fs)

        # plt.tight_layout()
        plt.subplots_adjust(left=0.25, bottom=0.25, right=0.9, top=0.85)
        save_path = os.path.join(savedir, f"corrs_{feat}_{phases[0]}_{phases[1]}")
        plt.savefig(f"{save_path}.png", dpi=400)
        plt.savefig(f"{save_path}.svg", dpi=400)
        plt.close()

def num_strides(raw_data):
    mice = raw_data.keys()
    num_strides = pd.DataFrame(index=np.arange(0,160), columns=mice)
    for midx, mouse in enumerate(mice):
        mouse_data = raw_data[mouse].droplevel('Day', axis=0)
        for run, ridx in enumerate(mouse_data.index.get_level_values('Run').unique()):
            try:
                run_data = mouse_data.loc(axis=0)[run, ['RunStart', 'Transition', 'RunEnd']].droplevel('Run', axis=0)
                transition_idx = run_data.index.get_level_values(level='FrameIdx')[
                run_data.index.get_level_values('RunStage') == 'Transition'][0]
            except:
                print('Cant find transition paw for', mouse, 'run', run)
                continue

            transition_paw = run_data.loc(axis=1)['initiating_limb'].loc(axis=0)[
                    'Transition', transition_idx]

            belt_1_run = run_data.loc(axis=0)['RunStart']
            stance_mask = belt_1_run.loc(axis=1)[transition_paw,'SwSt_discrete'] == locostuff['swst_vals_2025']['st']
            swing_mask = belt_1_run.loc(axis=1)[transition_paw,'SwSt_discrete'] == locostuff['swst_vals_2025']['sw']

            stance_idxs = belt_1_run.index.get_level_values('FrameIdx')[stance_mask]
            swing_idxs = belt_1_run.index.get_level_values('FrameIdx')[swing_mask]

            stance_idxs_rel = stance_idxs - transition_idx
            swing_idxs_rel = swing_idxs - transition_idx







# Import data
with open(r"H:\\Characterisation\\LH_allpca_LhWnrm_res_-3-2-1_APA2Wash2\\preprocessed_data_APAChar_LowHigh.pkl",
          'rb') as f:
    data_LH = pickle.load(f)
print("Loaded data from preprocessed_data_APAChar_LowHigh.pkl")

with open(r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis\FilteredData\Round7_Jan25\APAChar_LowHigh\Extended\allmice.pickle",'rb') as file:
    raw_data = pickle.load(file)

indexes = data_LH['feature_data_notscaled'].loc(axis=0)[-1].index
# If idx has two levels: mouse and run
mouse_to_runs = {}
for mouse, run in indexes:
    mouse_to_runs.setdefault(mouse, []).append(run)

# Convert to np arrays if you want
mouse_to_runs = {mouse: np.array(runs) for mouse, runs in mouse_to_runs.items()}

angle_save_dir = r"H:\Characterisation_v2\Angles"
if not os.path.exists(angle_save_dir):
    os.makedirs(angle_save_dir)

plot_toe_trajectory_real_distance(raw_data, mouse_to_runs, phases=['APA2', 'Wash2'], savedir=angle_save_dir, fs=7, n_interp=100)
plot_toe_distance_to_transition(raw_data, mouse_to_runs, phases=['APA2', 'Wash2'], savedir=angle_save_dir, fs=7, n_interp=100, stsw='st', bodypart='Toe', raw=True)
plot_toe_distance_to_transition(raw_data, mouse_to_runs, phases=['APA2', 'Wash2'], savedir=angle_save_dir, fs=7, n_interp=100, stsw='sw')
plot_toe_distance_to_transition(raw_data, mouse_to_runs, phases=['APA2', 'Wash2'], savedir=angle_save_dir, fs=7, n_interp=100, stsw='st', bodypart='Tailbase')
plot_toe_distance_to_transition(raw_data, mouse_to_runs, phases=['APA2', 'Wash2'], savedir=angle_save_dir, fs=7, n_interp=100, stsw='sw', bodypart='Tailbase')
plot_toe_distance_to_transition(raw_data, mouse_to_runs, phases=['APA2', 'Wash2'], savedir=angle_save_dir, fs=7, n_interp=100, stsw='st', bodypart='Toe')
plot_gait_features_corrs(data_LH['feature_data_notscaled'], stride=-1, phases=['APA2', 'Wash2'], savedir=angle_save_dir, fs=7)
plot_nose_back_tail_averages(raw_data, mouse_to_runs, phases=['APA2', 'Wash2'], savedir=angle_save_dir, fs=7, n_interp=100)
plot_limb_positions_average(raw_data, mouse_to_runs, phases=['APA2', 'Wash2'], savedir=angle_save_dir, fs=7, n_interp=100)

plot_angles(data_LH['feature_data_notscaled'], phases=['APA2', 'Wash2'], stride=-1,
            savedir=angle_save_dir)
handedness(raw_data)
