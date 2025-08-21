import os
import itertools
import random
import pickle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import seaborn as sns
import pandas as pd
from typing import List
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA

from Helpers.Config_23 import *
from Analysis.Tools.config import (global_settings, condition_specific_settings, instance_settings)
from Analysis.Characterisation_v2 import General_utils as gu
from Analysis.Characterisation_v2 import Plotting_utils as pu
from Analysis.Characterisation_v2 import SingleFeaturePred_utils as sfpu
from Analysis.Characterisation_v2 import MultiFeaturePred_utils as mfpu
from Analysis.Characterisation_v2.AnalysisTools import ClusterFeatures as cf
from Analysis.Characterisation_v2.AnalysisTools import PCA as pca
from Analysis.Characterisation_v2.AnalysisTools import Regression as reg
from Analysis.Characterisation_v2.Plotting import ClusterFeatures_plotting as cfp
from Analysis.Characterisation_v2.Plotting import SingleFeaturePred_plotting as sfpp
from Analysis.Characterisation_v2.Plotting import PCA_plotting as pcap
from Analysis.Characterisation_v2.Plotting import Regression_plotting as regp
# from Analysis.Characterisation_v2.Plotting import GeneralDescriptives_plotting as gdp
from Analysis.Characterisation_v2.AnalysisTools import LDA

sns.set(style="whitegrid")
random.seed(42)
np.random.seed(42)

# base_save_dir_no_c = os.path.join(paths['plotting_destfolder'], f'Characterisation\\LH_res')
base_save_dir_no_c = r"H:\Characterisation_v2\\LH"


def filter_data(data, phase):
    apa_mask, wash_mask = gu.get_mask_p1_p2(data, 'APA2', 'Wash2')
    phase_mask = apa_mask if phase == 'APA2' else wash_mask
    phase_run_vals = data.index[phase_mask]
    data_trimmed = data.loc[phase_run_vals, :]
    return data_trimmed, phase_run_vals

def main(stride_numbers: List[int], phases: List[str],
         condition: str = 'LowHigh', exp: str = 'Extended', day=None, compare_condition: str = 'None',
         settings_to_log: dict = None, residuals: bool = False):

    """
    Initialize experiment (data collection, directories, interpolation, logging). - NOT SCALED YET!!
    """
    print(f"Running {condition} analysis...")
    feature_data_notscaled, feature_data_compare_notscaled, stride_data, stride_data_compare, base_save_dir, base_save_dir_condition = gu.initialize_experiment(
        condition, exp, day, compare_condition, settings_to_log, base_save_dir_no_c, condition_specific_settings)

    # Paths
    global_data_path = os.path.join(base_save_dir, f"global_data_{condition}.pkl")
    preprocessed_data_file_path = os.path.join(base_save_dir, f"preprocessed_data_{condition}.pkl")
    LH_preprocessed_data_file_path = os.path.join(r"H:\Characterisation_v2\LH_res_-3-2-1_APA2Wash2\preprocessed_data_APAChar_LowHigh.pkl") # was "H:\Characterisation\LH_allpca_res_-3-2-1_APA2Wash2\preprocessed_data_APAChar_LowHigh.pkl"
    SingleFeatPath = os.path.join(base_save_dir_condition, 'SingleFeaturePredictions')
    MultiFeatPath = os.path.join(base_save_dir_condition, 'MultiFeaturePredictions')
    LH_MultiFeatPath = r"H:\Characterisation_v2\LH_res_-3-2-1_APA2Wash2\APAChar_LowHigh_Extended\MultiFeaturePredictions" # was "H:\Characterisation\LH_allpca_LhWnrm_res_-3-2-1_APA2Wash2\APAChar_LowHigh_Extended\MultiFeaturePredictions"
    ResidualFeatPath = os.path.join(base_save_dir_condition, 'Residuals')
    SingleResidualFeatPath = os.path.join(SingleFeatPath, 'Residuals')
    MultiResidualFeatPath = os.path.join(MultiFeatPath, 'Residuals')
    MultiTopPCsPath = os.path.join(MultiFeatPath, 'TopPCs')
    MultiBottom9PCsPath = os.path.join(MultiFeatPath, 'Bottom9PCs')
    MultiBottom3PCsPath = os.path.join(MultiFeatPath, 'Bottom3PCs')
    GeneralDescriptivesPath = os.path.join(base_save_dir_condition, 'GeneralDescriptives')

    os.makedirs(SingleFeatPath, exist_ok=True)
    os.makedirs(MultiFeatPath, exist_ok=True)
    if residuals:
        os.makedirs(ResidualFeatPath, exist_ok=True)
        os.makedirs(SingleResidualFeatPath, exist_ok=True)
        os.makedirs(MultiResidualFeatPath, exist_ok=True)
    os.makedirs(MultiTopPCsPath, exist_ok=True)
    os.makedirs(MultiBottom9PCsPath, exist_ok=True)
    os.makedirs(MultiBottom3PCsPath, exist_ok=True)
    os.makedirs(GeneralDescriptivesPath, exist_ok=True)

    print(f"Base save directory: {base_save_dir_condition}")

    idx = pd.IndexSlice

    # Skipping outlier removal
    if not os.path.exists(preprocessed_data_file_path):

        """
                   # -------- Normalize to LH wash per mouse (if applicable) --------
               """
        if inst["condition"] == "APAChar_HighLow" or inst["compare_condition"] == "APAChar_HighLow":
            HL_data = feature_data_notscaled.copy() if inst["condition"] == "APAChar_HighLow" else feature_data_compare_notscaled.copy()

            if global_settings["normalise_to_LH_wash"] and not global_settings["normalise_wash_nullspace"]:
                print("Normalizing HL to LH wash...")
                # load LH data
                with open(LH_preprocessed_data_file_path, 'rb') as f:
                    data = pickle.load(f)
                    feature_data_LH = data['feature_data_notscaled']

                # per mouse and stride, find mean of wash2 and subtract this from current phase
                higher_level_index = HL_data.index.droplevel('Run').drop_duplicates()
                means_LH = pd.DataFrame(index=higher_level_index, columns=HL_data.columns)
                means_current = pd.DataFrame(index=higher_level_index, columns=HL_data.columns)

                for (stride, mouse_id), data in means_current.groupby(level=[0, 1]):
                    # find wash2 runs
                    runs = expstuff["condition_exp_runs"]["APAChar"][exp]["Wash2"]
                    # get mean of wash2 runs
                    wash2_LH_means = feature_data_LH.loc[idx[stride, mouse_id, runs], :].mean()
                    wash2_HL_means = HL_data.loc[idx[stride, mouse_id, runs], :].mean()

                    means_LH.loc[idx[stride, mouse_id], :] = wash2_LH_means
                    means_current.loc[idx[stride, mouse_id], :] = wash2_HL_means

                # subtract each mouse's wash2 mean from all other runs
                for (stride, mouse_id), data in means_current.groupby(level=[0, 1]):
                    # get mean of wash2 runs
                    wash2_LH_means = means_LH.loc[idx[stride, mouse_id], :].values
                    wash2_HL_means = means_current.loc[idx[stride, mouse_id], :].values
                    wash2_diff = wash2_HL_means - wash2_LH_means

                    # subtract from all other runs
                    HL_data.loc[idx[stride, mouse_id, :], :] = HL_data.loc[idx[stride, mouse_id, :], :] - wash2_diff
            else:
                wash2_diff = None

            if inst["condition"] == "APAChar_HighLow":
                feature_data_notscaled = HL_data.copy()
            elif inst["compare_condition"] == "APAChar_HighLow":
                feature_data_compare_notscaled = HL_data.copy()
            else:
                raise ValueError(f"Unknown condition: {inst['condition']} or {inst['compare_condition']}")

            feature_data_notscaled = feature_data_notscaled.astype(float)

        else:
            wash2_diff = None
            if global_settings["normalise_to_LH_wash"]:
                print("NOT normalizing to LH wash as this is LH!")
                pass

        """
            # -------- Scale Data --------
            (Both conditions)
            - Z-score scale data for each mouse and stride
        """
        print("Scaling data...")
        feature_data = feature_data_notscaled.copy()
        feature_data.index.names = ['Stride', 'MouseID', 'Run']
        feature_data_compare = feature_data_compare_notscaled.copy()
        feature_data_compare.index.names = ['Stride', 'MouseID', 'Run']
        Normalize = {}


        feature_names = list(short_names.keys())
        # reorder feature_data by feature_names
        feature_data = feature_data.reindex(columns=feature_names)
        feature_data_compare = feature_data_compare.reindex(columns=feature_names)

        for (stride, mouse_id), data in feature_data.groupby(level=[0, 1]):
            d, normalize_mean, normalize_std = gu.normalize_df(data)
            feature_data.loc[idx[stride, mouse_id, :], :] = d
            norm_df = pd.DataFrame([normalize_mean, normalize_std], columns=feature_names, index=['mean', 'std'])
            Normalize[(stride, mouse_id)] = norm_df
        Normalize_compare = {}
        for (stride, mouse_id), data in feature_data_compare.groupby(level=[0, 1]):
            d, normalize_mean, normalize_std = gu.normalize_df(data)
            feature_data_compare.loc[idx[stride, mouse_id, :], :] = d
            norm_df = pd.DataFrame([normalize_mean, normalize_std], columns=feature_names, index=['mean', 'std'])
            Normalize_compare[(stride, mouse_id)] = norm_df

        # Get average feature values for each feature in each stride (across mice)
        feature_data_average = feature_data.groupby(level=['Stride', 'Run']).median()
        feature_data_compare_average = feature_data_compare.groupby(level=['Stride', 'Run']).median()


        """
            # -------- Feature Clusters --------
            (Main condition)
            For all strides together, plot correlations between features, organised by manual cluster. 
            Save manual clusters for later access.
        """
        print("Clustering features...")
        cluster_mappings = {}
        for p1, p2 in itertools.combinations(phases, 2):
            # features x runs
            feature_matrix = cf.get_global_feature_matrix(feature_data,
                                                       condition_specific_settings[condition]['global_fs_mouse_ids'],
                                                       'all', stride_data, p1, p2, smooth=False)
            cfp.plot_corr_matrix_sorted_manually(feature_matrix, base_save_dir_condition,
                                             f'CorrMatrix_manualclustering_{p1}-{p2}_all')
            for s in stride_numbers:
                cluster_mappings[(p1, p2, s)] = manual_clusters['cluster_mapping']

        """
            # -------- Save everything so far --------
        """
        with open(preprocessed_data_file_path, 'wb') as f:
            pickle.dump({
                'feature_names': feature_names,
                'feature_data': feature_data,
                'feature_data_compare': feature_data_compare,
                'feature_data_average': feature_data_average,
                'feature_data_compare_average': feature_data_compare_average,
                'feature_data_notscaled': feature_data_notscaled,
                'feature_data_compare_notscaled': feature_data_compare_notscaled,
                'stride_data': stride_data,
                'stride_data_compare': stride_data_compare,
                'Normalize': Normalize,
                'Normalize_compare': Normalize_compare,
                'cluster_mappings': cluster_mappings,
                'wash_diff': wash2_diff
            }, f)
    else:
        with open(preprocessed_data_file_path, 'rb') as f:
            data = pickle.load(f)
            feature_names = data['feature_names']
            feature_data = data['feature_data']
            feature_data_compare = data['feature_data_compare']
            feature_data_average = data['feature_data_average']
            feature_data_compare_average = data['feature_data_compare_average']
            feature_data_notscaled = data['feature_data_notscaled']
            feature_data_compare_notscaled = data['feature_data_compare_notscaled']
            stride_data = data['stride_data']
            stride_data_compare = data['stride_data_compare']
            Normalize = data['Normalize']
            Normalize_compare = data['Normalize_compare']
            cluster_mappings = data['cluster_mappings']
            wash2_diff = data['wash_diff']

    """
        # -------- Find residuals --------
        Residual between every feature (excluding walking speed) and walking speed
    """
    if stride_numbers != [0]:
        if residuals:
            if not os.path.exists(os.path.join(ResidualFeatPath, "ResidualData.h5")):
                residual_data = reg.find_residuals(feature_data, stride_numbers, phases, ResidualFeatPath)
            else:
                residual_data = pd.read_hdf(os.path.join(ResidualFeatPath, "ResidualData.h5"))
    else:
        residuals = False

    """
    ------------------ Single feature predictions ------------------
    """
    # -------- Get predictions for each phase combo, mouse, stride and feature
    filename = f'single_feature_predictions_{condition}.pkl'
    if os.path.exists(os.path.join(SingleFeatPath, filename)):
        print("Loading single feature predictions from file...")
        with open(os.path.join(SingleFeatPath, filename), 'rb') as f:
            single_feature_predictions = pickle.load(f)
    else:
        print("Running single feature predictions...")
        single_feature_predictions = sfpu.run_single_feature_regressions(phases, stride_numbers, condition, feature_names,
                                                                   feature_data, stride_data, SingleFeatPath,
                                                                   filename)

    # ------- Get predictions for each residual feature too
    if residuals:
        residual_filename = 'residual_' + filename
        if os.path.exists(os.path.join(SingleResidualFeatPath, residual_filename)):
            print("Loading residual single feature predictions from file...")
            with open(os.path.join(SingleResidualFeatPath, residual_filename), 'rb') as f:
                residual_single_feature_predictions = pickle.load(f)
        else:
            print("Running residual single feature predictions...")
            residual_single_feature_predictions = sfpu.run_single_feature_regressions(phases, stride_numbers, condition,
                                                                       residual_data.columns.tolist(), residual_data, stride_data,
                                                                       SingleResidualFeatPath, residual_filename)

    # -------------------------------------------------------------------------------------------------------
    # Get summary/average predictions for each phase combo, stride and feature, and find the top x features for each phase combo and stride
    filename_summary = f'single_feature_predictions_summary_{condition}.pkl'
    filename_top_feats = f'top_features_{condition}.pkl'

    if os.path.exists(os.path.join(SingleFeatPath, filename_summary)):
        print("Loading single feature predictions summary from file...")
        with open(os.path.join(SingleFeatPath, filename_summary), 'rb') as f:
            single_feature_predictions_summary = pickle.load(f)
        with open(os.path.join(SingleFeatPath, filename_top_feats), 'rb') as f:
            top_feats = pickle.load(f)
    else:
        print("Running single feature predictions summary...")
        single_feature_predictions_summary, top_feats = sfpu.get_summary_and_top_single_feature_data(10,
                                                                                                     phases, stride_numbers,
                                                                                                     feature_names,
                                                                                                     single_feature_predictions)

        print("Plotting single feature predictions...")
        sfpp.plot_featureXruns_heatmap(phases, stride_numbers, feature_names,
                                       single_feature_predictions_summary, 'RunPreds', SingleFeatPath)
        sfpp.plot_featureXruns_heatmap(phases, stride_numbers, feature_names,
                                       feature_data_average, 'RawFeats', SingleFeatPath)

        # Save the top features for each phase combo and stride
        with open(os.path.join(SingleFeatPath, filename_top_feats), 'wb') as f:
            pickle.dump(top_feats, f)
        # Save the single feature predictions
        with open(os.path.join(SingleFeatPath, filename_summary), 'wb') as f:
            pickle.dump(single_feature_predictions_summary, f)

    # -------------------------------------------------------------------------------------------------------
    # Get summary/average predictions for each **RESIDUAL** feature, and find the top x features for each phase combo and stride
    if residuals:
        residual_filename_summary = 'residual_' + filename_summary
        residual_filename_top_feats = 'residual_' + filename_top_feats

        if os.path.exists(os.path.join(SingleResidualFeatPath, residual_filename_summary)):
            print("Loading residual single feature predictions summary from file...")
            with open(os.path.join(SingleResidualFeatPath, residual_filename_summary), 'rb') as f:
                residual_single_feature_predictions_summary = pickle.load(f)
            with open(os.path.join(SingleResidualFeatPath, residual_filename_top_feats), 'rb') as f:
                residual_top_feats = pickle.load(f)
        else:
            print("Running residual single feature predictions summary...")
            residual_single_feature_predictions_summary, residual_top_feats = sfpu.get_summary_and_top_single_feature_data(10,
                                                                                                         phases, stride_numbers,
                                                                                                         residual_data.columns.tolist(),
                                                                                                         residual_single_feature_predictions)
            residual_data_average = residual_data.groupby(level=['Stride', 'Run']).median()

            print("Plotting residual single feature predictions...")
            sfpp.plot_featureXruns_heatmap(phases, stride_numbers, residual_data.columns.tolist(),
                                           residual_single_feature_predictions_summary, 'RunPreds', SingleResidualFeatPath)
            sfpp.plot_featureXruns_heatmap(phases, stride_numbers, residual_data.columns.tolist(),
                                           residual_data_average, 'RawFeats', SingleResidualFeatPath)

            # Save the top features for each phase combo and stride
            with open(os.path.join(SingleResidualFeatPath, residual_filename_top_feats), 'wb') as f:
                pickle.dump(residual_top_feats, f)
            # Save the single feature predictions
            with open(os.path.join(SingleResidualFeatPath, residual_filename_summary), 'wb') as f:
                pickle.dump(residual_single_feature_predictions_summary, f)

    """
    ------------------------- PCA ----------------------
    """
    if global_settings["use_LH_pcs"]:
        filename_pca = f'pca_APAChar_LowHigh.pkl'
        filepath_pca = os.path.join(LH_MultiFeatPath, filename_pca)
    else:
        filename_pca = f'pca_{condition}.pkl'
        filepath_pca = os.path.join(MultiFeatPath, filename_pca)

    if os.path.exists(filepath_pca):
        print(f"Loading PCA from file...\n{filepath_pca}")
        with open(filepath_pca, 'rb') as f:
            pca_all = pickle.load(f)
    else:
        if global_settings["use_LH_pcs"]:
            raise ValueError(f"Cannot find the LowHigh PCA file, which is required when 'use_LH_pcs' is set.\n"
                             "PCA filepath is:\n{filepath_pca}\n")
        else:
            print("Running PCA...")
            if global_settings["pca_CombineAllConditions"]:
                all_feature_data = pd.concat([feature_data, feature_data_compare], axis=0)
                all_stride_data = pd.concat([stride_data, stride_data_compare], axis=0)
                all_condition = f'{condition}_and_{compare_condition}'
                pca_all = pca.pca_main(all_feature_data, all_stride_data, phases, stride_numbers, all_condition, MultiFeatPath)
            else:
                pca_all = pca.pca_main(feature_data, stride_data, phases, stride_numbers, condition, MultiFeatPath)

            # Save PCA results
            with open(os.path.join(MultiFeatPath, filename_pca), 'wb') as f:
                pickle.dump(pca_all, f)

        # Plot how each feature loads onto the PCA components
        pcap.pca_plot_feature_loadings(pca_all, phases, MultiFeatPath)

    LH_feature_data  = feature_data_compare if inst["condition"] != "APAChar_LowHigh" else None
    top_features = pcap.plot_top_features_per_PC(pca_all, feature_data, feature_data_notscaled, phases, stride_numbers, condition, MultiFeatPath, n_top_features=8, feature_data_LH=LH_feature_data)
    # Save the top features for each PC
    with open(os.path.join(MultiFeatPath, f'top_features_per_PC_{condition}.pkl'), 'wb') as f:
        pickle.dump(top_features, f)

    """
    -------------------- PCA/Multi Feature Predictions ----------------------
    """

    filename_pca_pred = f'pca_predictions_{condition}.pkl'
    filepath_pca_pred = os.path.join(MultiFeatPath, filename_pca_pred)

    if os.path.exists(filepath_pca_pred):
        print("Loading PCA predictions from file...")
        with open(filepath_pca_pred, 'rb') as f:
            pca_pred = pickle.load(f)
    else:
        if global_settings["normalise_wash_nullspace"] and not global_settings["normalise_to_LH_wash"] and inst["condition"] == "APAChar_HighLow":
            pca_pred = mfpu.run_pca_regressions(phases, stride_numbers, condition, pca_all,
                                                feature_data, stride_data, MultiFeatPath,
                                                feature_compare_data=feature_data_compare,
                                                feature_compare_stride_data=stride_data_compare
                                                )
        else:
            pca_pred = mfpu.run_pca_regressions(phases, stride_numbers, condition, pca_all, feature_data, stride_data, MultiFeatPath)
        # Save PCA predictions
        with open(os.path.join(MultiFeatPath, filename_pca_pred), 'wb') as f:
            pickle.dump(pca_pred, f)

        # Plot PCA predictions as heatmap
        regp.plot_PCA_pred_heatmap(pca_all, pca_pred, feature_data, stride_data, phases, stride_numbers,condition, MultiFeatPath, cbar_scaling=0.7)

    # ------ PCA for residual features too -----
    if residuals and inst["condition"] == "APAChar_LowHigh":
        residual_filename_pca_pred = 'residual_' + filename_pca_pred
        if os.path.exists(os.path.join(MultiResidualFeatPath, residual_filename_pca_pred)):
            print("Loading PCA predictions from file...")
            with open(os.path.join(MultiResidualFeatPath, residual_filename_pca_pred), 'rb') as f:
                pca_pred_residual = pickle.load(f)
        else:
            if global_settings["normalise_wash_nullspace"] and not global_settings["normalise_to_LH_wash"] and inst["condition"] == "APAChar_HighLow":
                pca_pred_residual = mfpu.run_pca_regressions(phases, stride_numbers, condition, pca_all,
                                                             residual_data, stride_data, MultiResidualFeatPath,
                                                             feature_compare_data=residual_data,
                                                             feature_compare_stride_data=stride_data)
            else:
                pca_pred_residual = mfpu.run_pca_regressions(phases, stride_numbers, condition, pca_all, residual_data, stride_data, MultiResidualFeatPath)
            # Save PCA predictions
            with open(os.path.join(MultiResidualFeatPath, residual_filename_pca_pred), 'wb') as f:
                pickle.dump(pca_pred_residual, f)

            # Plot PCA predictions as heatmap
            regp.plot_PCA_pred_heatmap(pca_all, pca_pred_residual, residual_data, stride_data, phases, stride_numbers,condition, MultiResidualFeatPath, cbar_scaling=0.7)



    """
    ------------------ Interpretations ----------------------
    """
    if inst["condition"] != "APAChar_LowHigh":
        LH_pred_path = os.path.join(LH_MultiFeatPath, 'pca_predictions_APAChar_LowHigh.pkl')
        if os.path.exists(LH_pred_path):
            with open(LH_pred_path, 'rb') as f:
                pca_pred_LH = pickle.load(f)
    else:
        pca_pred_LH = None

    regp.mouse_sign_flip_with_LH(pca_pred, pca_pred_LH, -1, condition, MultiFeatPath)

    print('Features:')
    pcs_of_interest, pcs_of_interest_criteria = gu.get_and_save_pcs_of_interest(pca_pred, stride_numbers, MultiFeatPath, lesion_significance=False, LH_pred=pca_pred_LH)

    if residuals and inst["condition"] == "APAChar_LowHigh":
        print('Residuals:')
        residual_pcs_of_interest, residual_pcs_of_interest_criteria = gu.get_and_save_pcs_of_interest(pca_pred_residual, stride_numbers, MultiResidualFeatPath, lesion_significance=False, LH_pred=pca_pred_LH)


    # --------------- Plot run predicitions ---------------
    stride_mean_preds = defaultdict(list)
    residual_stride_mean_preds = defaultdict(list)
    for s in stride_numbers:
        stride_pred_mean = regp.plot_aggregated_run_predictions(pca_pred, MultiFeatPath, phases[0], phases[1], s, condition, smooth_kernel=3, error_bars=True)
        stride_mean_preds[s] = stride_pred_mean
        regp.plot_regression_loadings_PC_space_across_mice(pca_all, pca_pred, s, phases[0], phases[1], condition, MultiFeatPath)

        # residual
        if residuals and inst["condition"] == "APAChar_LowHigh":
            stride_pred_mean_residual = regp.plot_aggregated_run_predictions(pca_pred_residual, MultiResidualFeatPath, phases[0], phases[1], s, condition, smooth_kernel=3)
            residual_stride_mean_preds[s] = stride_pred_mean_residual
            regp.plot_regression_loadings_PC_space_across_mice(pca_all, pca_pred_residual, s, phases[0], phases[1], condition, MultiResidualFeatPath)


    regp.plot_multi_stride_predictions(stride_mean_preds, phases[0], phases[1], condition, MultiFeatPath, mean_smooth_window=21)
    if residuals and inst["condition"] == "APAChar_LowHigh":
        regp.plot_multi_stride_predictions(residual_stride_mean_preds, phases[0], phases[1], condition, MultiResidualFeatPath, mean_smooth_window=21)

    regp.plot_top3_pcs_run_projections(feature_data, pca_all, stride=-1, condition=condition, save_dir=MultiFeatPath)

    # # if condition == 'APAChar_LowHigh':
    # # ----------- Now predict APA with only the top 3 PCs ---------
    # filename_top3_pca_pred = f'pca_predictions_top3_{condition}.pkl'
    # if os.path.exists(os.path.join(MultiTopPCsPath, filename_top3_pca_pred)):
    #     print("Loading top 3 PCA predictions from file...")
    #     with open(os.path.join(MultiTopPCsPath, filename_top3_pca_pred), 'rb') as f:
    #         pca_pred_top3 = pickle.load(f)
    # else:
    #     print("Running PCA regressions with only the top 3 PCs...")
    #     pca_pred_top3 = mfpu.run_pca_regressions(phases, [-1], condition, pca_all, feature_data, stride_data, MultiTopPCsPath, select_pcs=['PC1','PC3','PC7'], select_pc_type='Top3')
    #     # Save PCA predictions
    #     with open(os.path.join(MultiTopPCsPath, filename_top3_pca_pred), 'wb') as f:
    #         pickle.dump(pca_pred_top3, f)
    #
    # # ----------- Now predict APA with only the remaining PCs ---------
    # filename_bottom9_pca_pred = f'pca_predictions_bottom9_{condition}.pkl'
    # if os.path.exists(os.path.join(MultiBottom9PCsPath, filename_bottom9_pca_pred)):
    #     print("Loading bottom 9 PCA predictions from file...")
    #     with open(os.path.join(MultiBottom9PCsPath, filename_bottom9_pca_pred), 'rb') as f:
    #         pca_pred_bottom_9 = pickle.load(f)
    # else:
    #     print("Running PCA regressions with only the remaining PCs...")
    #     other_pcs = ['PC2', 'PC4', 'PC5', 'PC6', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12']
    #     pca_pred_bottom_9 = mfpu.run_pca_regressions(phases, [-1], condition, pca_all, feature_data, stride_data, MultiBottom9PCsPath, select_pcs=other_pcs, select_pc_type='Bottom9')
    #     # Save PCA predictions
    #     with open(os.path.join(MultiBottom9PCsPath, filename_bottom9_pca_pred), 'wb') as f:
    #         pickle.dump(pca_pred_bottom_9, f)
    #
    # filename_bottom3_pca_pred = f'pca_predictions_bottom3_{condition}.pkl'
    # if os.path.exists(os.path.join(MultiBottom3PCsPath, filename_bottom3_pca_pred)):
    #     print("Loading bottom 3 PCA predictions from file...")
    #     with open(os.path.join(MultiBottom3PCsPath, filename_bottom3_pca_pred), 'rb') as f:
    #         pca_pred_bottom_3 = pickle.load(f)
    # else:
    #     bottom_3 = ['PC5', 'PC6', 'PC8']
    #     pca_pred_bottom_3 = mfpu.run_pca_regressions(phases, [-1], condition, pca_all, feature_data, stride_data, MultiBottom3PCsPath, select_pcs=bottom_3, select_pc_type='Bottom3')
    #     # Save PCA predictions
    #     with open(os.path.join(MultiBottom3PCsPath, filename_bottom3_pca_pred), 'wb') as f:
    #         pickle.dump(pca_pred_bottom_3, f)
    #
    # _ = regp.plot_aggregated_run_predictions(pca_pred_top3, MultiTopPCsPath, phases[0], phases[1], -1, condition, smooth_kernel=3)
    # _ = regp.plot_aggregated_run_predictions(pca_pred_bottom_9, MultiBottom9PCsPath, phases[0], phases[1], -1, condition, smooth_kernel=3)
    # _ = regp.plot_aggregated_run_predictions(pca_pred_bottom_3, MultiBottom3PCsPath, phases[0], phases[1], -1, condition, smooth_kernel=3)

    # # Save and print all the cv accuracies
    # top3_model_cv_acc = reg.find_model_cv_accuracy(pca_pred_top3, -1, phases, MultiTopPCsPath)
    # print(f"Top 3 model CV accuracy for stride -1: {top3_model_cv_acc}")
    # bottom9_model_cv_acc = reg.find_model_cv_accuracy(pca_pred_bottom_9, -1, phases, MultiBottom9PCsPath)
    # print(f"Bottom 9 model CV accuracy for stride -1: {bottom9_model_cv_acc}")
    # bottom3_model_cv_acc = reg.find_model_cv_accuracy(pca_pred_bottom_3, -1, phases, MultiBottom3PCsPath)
    # print(f"Bottom 3 model CV accuracy for stride -1: {bottom3_model_cv_acc}")

    # Save the remainder of the cv accuracies
    full_model_cv_accs = dict.fromkeys(stride_numbers, None)
    for s in stride_numbers:
        full_model_cv_acc = reg.find_model_cv_accuracy(pca_pred, s, phases, MultiFeatPath)
        full_model_cv_accs[s] = full_model_cv_acc
        print(f"Full model CV accuracy for stride {s}: {full_model_cv_acc}")
    if residuals and inst["condition"] == "APAChar_LowHigh":
        res_model_cv_acc = reg.find_model_cv_accuracy(pca_pred_residual, -1, phases, MultiResidualFeatPath)
        print(f"Residual model CV accuracy for stride -1: {res_model_cv_acc}")

    # gdp.plot_literature_parallels(feature_data, -1, phases, GeneralDescriptivesPath)
    # gdp.plot_angles(feature_data_notscaled, phases, -1, GeneralDescriptivesPath)

    print('Done')

#568

if __name__ == "__main__":
    # add flattened LowHigh etc settings to global_settings for log
    global_settings["LowHigh_c"] = condition_specific_settings['APAChar_LowHigh']['c']
    global_settings["HighLow_c"] = condition_specific_settings['APAChar_HighLow']['c']
    global_settings["LowMid_c"] = condition_specific_settings['APAChar_LowMid']['c']
    global_settings["LowHigh_mice"] = condition_specific_settings['APAChar_LowHigh']['global_fs_mouse_ids']
    global_settings["HighLow_mice"] = condition_specific_settings['APAChar_HighLow']['global_fs_mouse_ids']
    global_settings["LowMid_mice"] = condition_specific_settings['APAChar_LowMid']['global_fs_mouse_ids']

    # Combine the settings in a single dict to log.
    settings_to_log = {
        "global_settings": global_settings,
        "instance_settings": instance_settings
    }

    # Run each instance.
    for inst in instance_settings:
        main(
            global_settings["stride_numbers"],
            global_settings["phases"],
            condition=inst["condition"],
            exp=inst["exp"],
            day=inst["day"],
            compare_condition=inst["compare_condition"],
            settings_to_log=settings_to_log,
            residuals= global_settings["residuals"]
        )
