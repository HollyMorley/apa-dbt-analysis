import pandas as pd
import pickle
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

from Helpers.Config_23 import *

from Helpers import utils
from Analysis.Characterisation_v2 import Plotting_utils as pu
from Analysis.Tools.config import condition_specific_settings

# Maps the short condition names used in this file to config keys
CONDITION_CONFIG_MAP = {
    'Low-High': 'APAChar_LowHigh',
    'Low-Mid': 'APAChar_LowMid',
    'High-Low': 'APAChar_HighLow',
}


def find_strides(stance_idxs, swing_idxs):
    """
    Identify strides as stance onset → next stance onset.
    Returns a list of (stance_start, swing_start, stride_end) tuples.
    """
    strides = []

    # Find transitions: stance → swing → stance
    # Work through stance bouts to find stance onsets
    stance_onsets = []
    swing_onsets = []

    # Detect stance onsets (frame where stance begins, i.e. not preceded by
    # stance)
    for i, idx in enumerate(stance_idxs):
        if i == 0 or idx != stance_idxs[i - 1] + 1:
            stance_onsets.append(idx)

    # Detect swing onsets
    for i, idx in enumerate(swing_idxs):
        if i == 0 or idx != swing_idxs[i - 1] + 1:
            swing_onsets.append(idx)

    stance_onsets = np.array(stance_onsets)
    swing_onsets = np.array(swing_onsets)

    # Pair each stance onset with the next swing onset, then the following
    # stance onset
    for s_on in stance_onsets:
        # Find the first swing onset after this stance onset
        sw_after = swing_onsets[swing_onsets > s_on]
        if len(sw_after) == 0:
            continue
        sw_on = sw_after[0]

        # Next stance onset is optional - include stride either way
        next_st = stance_onsets[stance_onsets > sw_on]
        stride_end = next_st[0] if len(next_st) > 0 else None

        strides.append((s_on, sw_on, stride_end))

    return strides


def check_sitting(run_data):
    all_st_mask = (run_data.loc(axis=1)[
                       ['ForepawR', 'ForepawL', 'HindpawR',
                        'HindpawL'], 'SwSt'] ==
                   locostuff['swst_vals_2025']['st']).all(axis=1)
    all_st_idxs = run_data.index.get_level_values('FrameIdx')[all_st_mask]

    all_st_blocks = utils.Utils().find_blocks(all_st_idxs, gap_threshold=100,
                                              block_min_size=400)

    # If any stance blocks are found in all_st_blocks, return True
    if len(all_st_blocks) > 0:
        return True
    else:
        return False


def num_strides(raw_data):
    stride_counts_all = {}
    num_sitting = 0
    num_bad_transitions = 0
    sitting_log = {}

    for condition, condition_data in raw_data.items():
        allowed_mice = \
            condition_specific_settings[CONDITION_CONFIG_MAP[condition]][
                'global_fs_mouse_ids']
        mice = [m for m in condition_data.keys() if m in allowed_mice]
        num_strides_df = pd.DataFrame(index=np.arange(0, 160), columns=mice)
        sitting_log[condition] = {}

        for mouse in mice:
            mouse_data = condition_data[mouse].droplevel('Day', axis=0)
            sitting_log[condition][mouse] = []

            for ridx in mouse_data.index.get_level_values('Run').unique():
                try:
                    run_data = mouse_data.loc(axis=0)[
                        ridx, ['RunStart', 'Transition', 'RunEnd']].droplevel(
                        'Run', axis=0)
                    transition_idx = \
                        run_data.index.get_level_values(level='FrameIdx')[
                            run_data.index.get_level_values(
                                'RunStage') == 'Transition'][0]
                except:
                    print(
                        f'Cannot find transition for {condition}, {mouse}, '
                        f'run {ridx}')
                    continue

                # Find if all limbs are in stance for extended period
                sitting = check_sitting(run_data)
                if sitting:
                    print(
                        f'Run {ridx} of {mouse} in {condition} appears to be '
                        f'sitting/standing. Skipping stride count.')
                    num_sitting += 1
                    sitting_log[condition][mouse].append(ridx)
                    continue

                transition_paw = \
                    run_data.loc(axis=1)['initiating_limb'].loc(axis=0)[
                        'Transition', transition_idx]

                paw = 'ForepawToeL' if transition_paw == 'ForepawL' else \
                    'ForepawToeR'
                paw_transition_position = \
                    run_data.loc(axis=1)[paw, 'x'].loc(axis=0)[
                        'Transition'].iloc[
                        0]
                if paw_transition_position < expstuff['setup'][
                    'transition_mm']:
                    num_bad_transitions += 1
                    continue

                belt_1_run = run_data.loc(axis=0)['RunStart']
                stance_mask = belt_1_run.loc(axis=1)[
                                  transition_paw, 'SwSt_discrete'] == \
                              locostuff['swst_vals_2025']['st']
                swing_mask = belt_1_run.loc(axis=1)[
                                 transition_paw, 'SwSt_discrete'] == \
                             locostuff['swst_vals_2025']['sw']

                stance_idxs = belt_1_run.index.get_level_values('FrameIdx')[
                    stance_mask]
                swing_idxs = belt_1_run.index.get_level_values('FrameIdx')[
                    swing_mask]

                if belt_1_run.loc(axis=1)['initiating_limb'].iloc[
                    0] == transition_paw:
                    # Check if paw in stance and the frame not already
                    # included in stance_idxs
                    if belt_1_run.loc(axis=1)[transition_paw, 'SwSt'].iloc[
                        0] == locostuff['swst_vals_2025']['st'] and \
                            belt_1_run.index.get_level_values('FrameIdx')[
                                0] not in stance_idxs:
                        # Add first frame to stance idxs
                        stance_idxs = np.insert(stance_idxs, 0,
                                                belt_1_run.index.get_level_values(
                                                    'FrameIdx')[0])

                else:
                    if belt_1_run.loc(axis=1)[transition_paw, 'SwSt'].iloc[
                        0] == locostuff['swst_vals_2025']['st']:
                        # Add first frame to stance idxs
                        stance_idxs = np.insert(stance_idxs, 0,
                                                belt_1_run.index.get_level_values(
                                                    'FrameIdx')[0])

                strides = find_strides(stance_idxs, swing_idxs)
                num_strides_df.loc[ridx, mouse] = len(strides)

        stride_counts_all[condition] = num_strides_df

    print(
        f'Identified {num_sitting} runs with extended sitting/standing '
        f'periods.')
    print(
        f'Identified {num_bad_transitions} runs with suspicious transition '
        f'positions.')

    return stride_counts_all, sitting_log


def plot_num_strides(stride_counts_all):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ax, (condition, num_strides_df) in zip(axes,
                                               stride_counts_all.items()):
        all_counts = num_strides_df.values.flatten()
        all_counts = all_counts[~pd.isnull(all_counts)].astype(float)

        bins = np.arange(all_counts.min(), all_counts.max() + 2)
        ax.hist(all_counts, bins=bins, color='steelblue', edgecolor='white')

        ax.set_xlabel('Number of strides on Belt 1')
        ax.set_title(condition)

    axes[0].set_ylabel('Number of trials')
    plt.suptitle('Distribution of stride counts on Belt 1', y=1.02)
    plt.tight_layout()
    plt.show()

    return fig


def get_suspicious_runs(raw_data, stride_counts_all, above_threshold=None,
                        below_threshold=None):
    suspicious = {}

    for condition, num_strides_df in stride_counts_all.items():
        suspicious[condition] = {}

        for mouse in num_strides_df.columns:
            mouse_data = raw_data[condition][mouse].droplevel('Day', axis=0)
            high_stride_runs = num_strides_df[mouse].dropna()
            if above_threshold and not below_threshold:
                high_stride_runs = high_stride_runs[
                    high_stride_runs.astype(float) > above_threshold]
            elif below_threshold and not above_threshold:
                high_stride_runs = high_stride_runs[
                    high_stride_runs.astype(float) < below_threshold]
            elif above_threshold and below_threshold:
                high_stride_runs = high_stride_runs[
                    (high_stride_runs.astype(float) > above_threshold) &
                    (high_stride_runs.astype(float) < below_threshold)
                    ]

            if high_stride_runs.empty:
                continue

            run_dfs = []
            for run in high_stride_runs.index:
                try:
                    run_data = mouse_data.loc(axis=0)[run]
                    run_data = run_data.copy()
                    run_data['mouse'] = mouse
                    run_data['run'] = run
                    run_data['n_strides'] = high_stride_runs[run]
                    run_dfs.append(run_data)
                except KeyError:
                    print(
                        f'Could not retrieve {condition}, {mouse}, run {run}')
                    continue

            if run_dfs:
                suspicious[condition][mouse] = pd.concat(run_dfs)

    return suspicious


def get_running_speeds(raw_data, fps=247):
    """
    Calculate mean running speed (relative to belt) up to transition,
    aligned to transition frame. Averages within mouse first, then across mice.
    Returns dict of {condition: list of per-mouse mean speed arrays}.
    """
    speeds_all = {}

    for condition, condition_data in raw_data.items():
        belt_speed = belt_speeds[condition]
        allowed_mice = \
            condition_specific_settings[CONDITION_CONFIG_MAP[condition]][
                'global_fs_mouse_ids']
        mouse_means = []

        for mouse, mouse_data in condition_data.items():
            if mouse not in allowed_mice:
                continue
            mouse_data = mouse_data.droplevel('Day', axis=0)
            trial_speeds = []

            for ridx in mouse_data.index.get_level_values('Run').unique():
                try:
                    run_data = mouse_data.loc(axis=0)[
                        ridx, ['RunStart', 'Transition', 'RunEnd']].droplevel(
                        'Run', axis=0)
                    transition_idx = \
                        run_data.index.get_level_values('FrameIdx')[
                            run_data.index.get_level_values(
                                'RunStage') == 'Transition'][0]
                except:
                    continue

                if check_sitting(run_data):
                    continue

                # Exclude runs where transition paw is too far back on the
                # belt, as these likely reflect bad transitions where mouse
                # is not actually running onto belt 2
                transition_paw = \
                    run_data.loc(axis=1)['initiating_limb'].loc(axis=0)[
                        'Transition', transition_idx]
                paw_toe = 'ForepawToeL' if transition_paw == 'ForepawL' else \
                    'ForepawToeR'
                paw_transition_position = \
                    run_data.loc(axis=1)[paw_toe, 'x'].loc(axis=0)[
                        'Transition'].iloc[0]
                if paw_transition_position < expstuff['setup'][
                    'transition_mm']:
                    continue

                belt_1_run = run_data.loc(axis=0)['RunStart']
                frame_idxs = belt_1_run.index.get_level_values('FrameIdx')

                try:
                    tail_x = belt_1_run.loc(axis=1)['Tail1', 'x'].values
                except KeyError:
                    continue

                tail_speed = np.diff(tail_x) * fps
                frame_idxs_diff = frame_idxs[1:]
                tail_speed_rel = tail_speed - belt_speed
                frames_rel = frame_idxs_diff - transition_idx

                trial_speeds.append((frames_rel, tail_speed_rel))

            if not trial_speeds:
                continue

            # Average across trials within this mouse
            mouse_means.append(trial_speeds)

        speeds_all[condition] = mouse_means

    return speeds_all


def plot_running_speeds(speeds_all, window=(-200, 0)):
    condition_colours = {
        'Low-High': 'steelblue',
        'Low-Mid': 'mediumseagreen',
        'High-Low': 'tomato',
    }

    frame_axis = np.arange(window[0], window[1])
    time_axis = frame_axis / 247
    fig, ax = plt.subplots(figsize=(10, 4))

    for condition, mouse_trial_speeds in speeds_all.items():
        per_mouse_means = []

        for trial_speeds in mouse_trial_speeds:
            # Interpolate each trial onto common frame axis
            interpolated = []
            for frames_rel, speed in trial_speeds:
                mask = (frames_rel >= window[0]) & (frames_rel < window[1])
                if mask.sum() < 10:
                    continue
                try:
                    f = interp1d(frames_rel[mask], speed[mask],
                                 bounds_error=False, fill_value=np.nan)
                    interpolated.append(f(frame_axis))
                except:
                    continue

            if not interpolated:
                continue

            # Mean across trials for this mouse
            mouse_mean = np.nanmean(np.array(interpolated), axis=0)
            per_mouse_means.append(mouse_mean)

        if not per_mouse_means:
            continue

        # Mean and SEM across mice
        per_mouse_means = np.array(per_mouse_means)
        grand_mean = np.nanmean(per_mouse_means, axis=0)
        sem = np.nanstd(per_mouse_means, axis=0) / np.sqrt(
            per_mouse_means.shape[0])

        ax.plot(time_axis, grand_mean, color=condition_colours[condition],
                label=condition, linewidth=2)
        ax.fill_between(time_axis, grand_mean - sem, grand_mean + sem,
                        color=condition_colours[condition], alpha=0.2)

    ax.axvline(0, color='k', linestyle='--', linewidth=1, label='Transition')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Time relative to transition (s)')
    ax.set_ylabel('Running speed relative to belt (mm/s)')
    ax.set_title('Running speed on Belt 1 prior to transition')
    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig


def get_stride_positions(raw_data, fps=247, n_strides_back=8):
    positions_all = {}

    for condition, condition_data in raw_data.items():
        allowed_mice = \
        condition_specific_settings[CONDITION_CONFIG_MAP[condition]][
            'global_fs_mouse_ids']
        stride_positions = {s: [] for s in range(-1, -n_strides_back - 1, -1)}

        for mouse, mouse_data in condition_data.items():
            if mouse not in allowed_mice:
                continue
            mouse_data = mouse_data.droplevel('Day', axis=0)
            mouse_stride_positions = {s: [] for s in
                                      range(-1, -n_strides_back - 1, -1)}

            for ridx in mouse_data.index.get_level_values('Run').unique():
                try:
                    run_data = mouse_data.loc(axis=0)[
                        ridx, ['RunStart', 'Transition', 'RunEnd']].droplevel(
                        'Run', axis=0)
                    transition_idx = \
                        run_data.index.get_level_values('FrameIdx')[
                            run_data.index.get_level_values(
                                'RunStage') == 'Transition'][0]
                except:
                    continue

                if check_sitting(run_data):
                    continue

                transition_paw = \
                    run_data.loc(axis=1)['initiating_limb'].loc(axis=0)[
                        'Transition', transition_idx]
                paw_toe = 'ForepawToeL' if transition_paw == 'ForepawL' else \
                    'ForepawToeR'

                # Skip runs where transition paw is too far back on the
                # belt, as these likely reflect bad transitions where mouse
                # is not actually running onto belt 2
                paw_transition_position = \
                    run_data.loc(axis=1)[paw_toe, 'x'].loc(axis=0)[
                        'Transition'].iloc[0]
                if paw_transition_position < expstuff['setup'][
                    'transition_mm']:
                    continue

                belt_1_run = run_data.loc(axis=0)['RunStart']
                frame_idxs = belt_1_run.index.get_level_values('FrameIdx')

                stance_mask = belt_1_run.loc(axis=1)[
                                  transition_paw, 'SwSt_discrete'] == \
                              locostuff['swst_vals_2025']['st']
                swing_mask = belt_1_run.loc(axis=1)[
                                 transition_paw, 'SwSt_discrete'] == \
                             locostuff['swst_vals_2025']['sw']

                stance_idxs = frame_idxs[stance_mask]
                swing_idxs = frame_idxs[swing_mask]

                strides = find_strides(stance_idxs, swing_idxs)

                # Label strides relative to transition:
                # Find which stride is -1 by finding the last stride whose
                # stance onset is before the transition frame
                pre_transition = [(i, s) for i, s in enumerate(strides) if
                                  s[0] < transition_idx]

                if not strides:
                    print(f'{mouse} run {ridx}: NO STRIDES FOUND')
                    continue
                elif not pre_transition:
                    print(
                        f'{mouse} run {ridx}: strides found but none before '
                        f'transition. '
                        f'Stride onset frames: {[s[0] for s in strides]}, '
                        f'transition_idx: {transition_idx}')
                    continue
                else:
                    # Check x position retrieval
                    last_idx = pre_transition[-1][0]
                    s_on = strides[last_idx][0]
                    x_vals = belt_1_run.loc(axis=1)[paw_toe, 'x'].loc[
                        belt_1_run.index.get_level_values(
                            'FrameIdx') == s_on].values
                    if len(x_vals) == 0:
                        print(
                            f'{mouse} run {ridx}: stride -1 found at frame '
                            f'{s_on} but no x position retrieved')

                # The last pre-transition stride is stride -1
                last_idx = pre_transition[-1][0]

                for offset, (s_on, sw_on, stride_end) in enumerate(
                        reversed(strides[:last_idx + 1])):
                    label = -(offset + 1)
                    if label < -n_strides_back:
                        break

                    try:
                        x_pos = belt_1_run.loc(axis=1)[paw_toe, 'x'].loc[
                            belt_1_run.index.get_level_values(
                                'FrameIdx') == s_on].values[0]
                        mouse_stride_positions[label].append(x_pos)
                    except (IndexError, KeyError):
                        continue

            for label in mouse_stride_positions:
                vals = mouse_stride_positions[label]
                stride_positions[label].append(
                    (mouse, np.nanmean(vals) if vals else np.nan))

        positions_all[condition] = stride_positions

    return positions_all


def plot_stride_positions(positions_all, n_strides_back=8):
    stride_labels = list(range(-1, -n_strides_back - 1, -1))

    all_mice = sorted(set(
        mouse_id
        for stride_positions in positions_all.values()
        for pairs in stride_positions.values()
        for mouse_id, _ in pairs
    ))

    fig, axes = plt.subplots(1, len(positions_all),
                             figsize=(5 * len(positions_all), 4))

    for ax, (condition, stride_positions) in zip(axes, positions_all.items()):
        for label in stride_labels:
            pairs = stride_positions[label]
            for mouse_id, val in pairs:
                if np.isnan(val):
                    continue
                ax.scatter(val, label, color=pu.get_color_mice(mouse_id),
                           marker=pu.get_marker_style_mice(mouse_id),
                           edgecolors='none', alpha=0.7, s=40, zorder=3)

            # Mean across mice
            vals = [v for _, v in pairs if not np.isnan(v)]
            if vals:
                ax.scatter(np.nanmean(vals), label, color='black',
                           edgecolors='none', s=80, zorder=4, marker='|')

        ax.axvline(expstuff['setup']['transition_mm'], color='k',
                   linestyle='--', linewidth=1, label='Transition')
        ax.set_xlim(0, 470)
        ax.set_xlabel('Position on belt (mm)')
        ax.set_title(condition)
        ax.set_yticks(stride_labels)
        ax.set_yticklabels([str(s) for s in stride_labels])

    axes[0].set_ylabel('Stride number')

    legend_elements = [Patch(facecolor=pu.get_color_mice(m), label=m)
                       for m in all_mice]
    axes[-1].legend(handles=legend_elements, title='Mouse',
                    bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    plt.suptitle('Stance onset position per stride', y=1.02)
    plt.tight_layout()
    plt.show()

    return fig


files = {
    'Low-High': r"H:\Dual-belt_APAs\analysis\DLC_DualBelt"
                r"\DualBelt_MyAnalysis\FilteredData\Round7_Jan25"
                r"\APAChar_LowHigh\Extended\allmice.pickle",
    'Low-Mid': r"H:\Dual-belt_APAs\analysis\DLC_DualBelt\DualBelt_MyAnalysis"
               r"\FilteredData\Round7_Jan25\APAChar_LowMid\Extended\allmice"
               r".pickle",
    'High-Low': r"H:\Dual-belt_APAs\analysis\DLC_DualBelt"
                r"\DualBelt_MyAnalysis\FilteredData\Round7_Jan25"
                r"\APAChar_HighLow\Extended\allmice.pickle",
}

raw_data = dict.fromkeys(files.keys())
for condition, path in files.items():
    print(f'Processing condition: {condition}')
    with open(path, 'rb') as file:
        raw_data[condition] = pickle.load(file)

stride_counts, sitting_log = num_strides(raw_data)
fig = plot_num_strides(stride_counts)

suspicious_runs = get_suspicious_runs(raw_data, stride_counts,
                                      above_threshold=6)
suspicious_runs_2 = get_suspicious_runs(raw_data, stride_counts,
                                        below_threshold=3)

belt_speeds = {
    'Low-High': expstuff['speeds']['Low'],
    'Low-Mid': expstuff['speeds']['Low'],
    'High-Low': expstuff['speeds']['High'],
}

speeds_all = get_running_speeds(raw_data)
fig = plot_running_speeds(speeds_all, window=(-200, 0))

stride_positions = get_stride_positions(raw_data, n_strides_back=8)
fig = plot_stride_positions(stride_positions, n_strides_back=8)

print(
    "Analysis complete. Check generated figures and suspicious_runs for "
    "details.")
