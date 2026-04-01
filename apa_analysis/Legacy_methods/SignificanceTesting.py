import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


from helpers.config import *


def _shuffle_iteration_phase(all_trials, trial2idx, Obs_matrix, phase1_trials, phase2_trials, type):
    # Use the same shuffling logic as in your original loop
    phase1_trials_null = np.random.choice(all_trials, len(phase1_trials), replace=False)
    phase2_trials_null = all_trials[~np.isin(all_trials, phase1_trials_null)]
    phase2_trials_null = np.random.choice(phase2_trials_null, len(phase2_trials), replace=False)

    idx1 = [trial2idx[t] for t in phase1_trials_null]
    idx2 = [trial2idx[t] for t in phase2_trials_null]

    if type == 'mean':
        mouse_means_p1 = np.nanmean(Obs_matrix[:, idx1], axis=1)
        mouse_means_p2 = np.nanmean(Obs_matrix[:, idx2], axis=1)
        return np.nanmean(mouse_means_p2 - mouse_means_p1)
    elif type == 'median':
        mouse_medians_p1 = np.nanmedian(Obs_matrix[:, idx1], axis=1)
        mouse_medians_p2 = np.nanmedian(Obs_matrix[:, idx2], axis=1)
        return np.nanmean(mouse_medians_p2 - mouse_medians_p1)
    elif type == 'var':
        mouse_vars_p1 = np.nanvar(Obs_matrix[:, idx1], axis=1)
        mouse_vars_p2 = np.nanvar(Obs_matrix[:, idx2], axis=1)
        return np.nanmean(mouse_vars_p2 - mouse_vars_p1)

def _shuffle_iteration_condition(all_trials, trial2idx, Obs_matrix, Obs_matrixc, phase1_trials, phase2_trials, type):
    phase1_null = np.random.choice(all_trials, len(phase1_trials), replace=False)
    phase2_pool = all_trials[~np.isin(all_trials, phase1_null)]
    phase2_null = np.random.choice(phase2_pool, len(phase2_trials), replace=False)

    idx1 = [trial2idx[t] for t in phase1_null]
    idx2 = [trial2idx[t] for t in phase2_null]

    if type == 'mean':
        eff = np.nanmean(Obs_matrix[:, idx2], axis=1) - np.nanmean(Obs_matrix[:, idx1], axis=1)
        effc = np.nanmean(Obs_matrixc[:, idx2], axis=1) - np.nanmean(Obs_matrixc[:, idx1], axis=1)
        return np.nanmean(eff - effc)
    elif type == 'median':
        eff = np.nanmedian(Obs_matrix[:, idx2], axis=1) - np.nanmedian(Obs_matrix[:, idx1], axis=1)
        effc = np.nanmedian(Obs_matrixc[:, idx2], axis=1) - np.nanmedian(Obs_matrixc[:, idx1], axis=1)
        return np.nanmean(eff - effc)
    elif type == 'var':
        eff = np.nanvar(Obs_matrix[:, idx2], axis=1) - np.nanvar(Obs_matrix[:, idx1], axis=1)
        effc = np.nanvar(Obs_matrixc[:, idx2], axis=1) - np.nanvar(Obs_matrixc[:, idx1], axis=1)
        return np.nanmean(eff - effc)


def ShufflingTest_ComparePhases(Obs_p1, Obs_p2, mouseObs_p1, mouseObs_p2, phase1, phase2, type, num_iter=1000) -> [float, float, np.array]:
    """
    Shuffling test to determine if the difference in means between two phases is significant.
    Args:
        Obs_p1: DataFrame, Observations for phase1 for all trials and mice
        Obs_p2: DataFrame, Observations for phase2 for all trials and mice
        meanObs_p1: DataFrame, Mean observations for phase1 for each mouse
        meanObs_p2: DataFrame, Mean observations for phase2 for each mouse
        phase1: str, Name of phase1
        phase2: str, Name of phase2
        num_iter: int, Number of iterations for the shuffling test
    Returns:
        p_value: float, p-value of the shuffling test
        Eff_obs: float, Observed effect size
        EffNull: array, Null effect size distribution
    """

    Eff_obs = (mouseObs_p2 - mouseObs_p1).mean()

    # Identify phase1 and phase2 trials
    phase1_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase1]
    phase2_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase2]
    all_trials = np.concatenate((phase1_trials, phase2_trials))

    Obs_both = pd.concat([Obs_p1, Obs_p2], axis=0)

    Obs_matrix = Obs_both.unstack(level='Run').loc[:, all_trials].values
    trial2idx = {trial: i for i, trial in enumerate(all_trials)}

    # Use joblib to run iterations in parallel, using the same shuffling logic.
    EffNull = Parallel(n_jobs=-1)(
        delayed(_shuffle_iteration_phase)(all_trials, trial2idx, Obs_matrix, phase1_trials, phase2_trials, type)
        for _ in range(num_iter)
    )
    EffNull = np.array(EffNull)

    # Count how many values in Eff^Null are greater than Eff^Obs
    # Two‑tailed p‑value (adding +1 to numerator and denominator for a small‐sample correction)
    p_value = (np.sum(np.abs(EffNull) >= abs(Eff_obs)) + 1) / (num_iter + 1)

    return p_value, Eff_obs, EffNull

def ShufflingTest_CompareConditions(Obs_p1, Obs_p2 ,Obs_p1c ,Obs_p2c ,pdiff_Obs ,pdiff_c_Obs, phase1, phase2, type, num_iter=1000):
    Eff_obs = (pdiff_Obs - pdiff_c_Obs).mean()

    # Identify phase1 and phase2 trials
    phase1_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase1]
    phase2_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase2]
    all_trials = np.concatenate((phase1_trials, phase2_trials))

    # Ensure inputs are Series
    Obs_p1 = Obs_p1.squeeze()
    Obs_p2 = Obs_p2.squeeze()
    Obs_p1c = Obs_p1c.squeeze()
    Obs_p2c = Obs_p2c.squeeze()

    # Build trial×mouse matrices for each condition
    Obs_matrix = pd.concat([Obs_p1, Obs_p2], axis=0).unstack(level='Run').loc[:, all_trials].values
    Obs_matrixc = pd.concat([Obs_p1c, Obs_p2c], axis=0).unstack(level='Run').loc[:, all_trials].values

    trial2idx = {trial: i for i, trial in enumerate(all_trials)}

    # Use joblib to run iterations in parallel, using the same shuffling logic.
    EffNull = Parallel(n_jobs=-1)(
        delayed(_shuffle_iteration_condition)(all_trials, trial2idx, Obs_matrix, Obs_matrixc, phase1_trials, phase2_trials, type)
        for _ in range(num_iter)
    )
    EffNull = np.array(EffNull)

    p_value = (np.sum(np.abs(EffNull) >= abs(Eff_obs)) + 1) / (num_iter + 1)
    return p_value, Eff_obs, EffNull

# def ShufflingTest_ComparePhases_variance(Obs_p1, Obs_p2, varObs_p1, varObs_p2, phase1, phase2, num_iter=1000):
#     """
#     Shuffling test for differences in variances.
#     """
#     Eff_obs = (varObs_p2 - varObs_p1).mean()
#
#     # Identify phase1 and phase2 trials
#     phase1_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase1]
#     phase2_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase2]
#     all_trials = np.concatenate((phase1_trials, phase2_trials))
#
#     # Build trial×mouse matrices for each condition
#     Obs_both = pd.concat([Obs_p1, Obs_p2], axis=0)
#     Obs_matrix = Obs_both.unstack(level='Run').loc[:, all_trials].values
#
#     trial2idx = {trial: i for i, trial in enumerate(all_trials)}
#
#     EffNull = Parallel(n_jobs=-1)(
#         delayed(_shuffle_iteration_phase)(all_trials, trial2idx, Obs_matrix, phase1_trials, phase2_trials, 'var')
#         for _ in range(num_iter)
#     )
#     EffNull = np.array(EffNull)
#
#     p_value = (np.sum(np.abs(EffNull) >= abs(Eff_obs)) + 1) / (num_iter + 1)
#     return p_value, Eff_obs, EffNull
#
# def ShufflingTest_CompareConditions_variance(Obs_p1, Obs_p2, Obs_p1c, Obs_p2c, pdiff_Obs, pdiff_c_Obs, phase1, phase2, num_iter=1000):
#     """
#     Shuffling test for differences in variance differences between conditions.
#     """
#     Eff_obs = (pdiff_Obs - pdiff_c_Obs).mean()
#
#     # Identify phase1 and phase2 trials
#     phase1_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase1]
#     phase2_trials = expstuff['condition_exp_runs']['APAChar']['Extended'][phase2]
#     all_trials = np.concatenate((phase1_trials, phase2_trials))
#
#     # Ensure inputs are Series
#     Obs_p1 = Obs_p1.squeeze()
#     Obs_p2 = Obs_p2.squeeze()
#     Obs_p1c = Obs_p1c.squeeze()
#     Obs_p2c = Obs_p2c.squeeze()
#
#     # Build trial×mouse matrices for each condition
#     Obs_matrix = pd.concat([Obs_p1, Obs_p2], axis=0).unstack(level='Run').loc[:, all_trials].values
#     Obs_matrixc = pd.concat([Obs_p1c, Obs_p2c], axis=0).unstack(level='Run').loc[:, all_trials].values
#
#     trial2idx = {trial: i for i, trial in enumerate(all_trials)}
#
#     EffNull = Parallel(n_jobs=-1)(
#         delayed(_shuffle_iteration_condition)(all_trials, trial2idx, Obs_matrix, Obs_matrixc, phase1_trials, phase2_trials, 'var')
#         for _ in range(num_iter)
#     )
#     EffNull = np.array(EffNull)
#
#     p_value = (np.sum(np.abs(EffNull) >= abs(Eff_obs)) + 1) / (num_iter + 1)
#     return p_value, Eff_obs, EffNull





