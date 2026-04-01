"""Multi-feature PCA-based logistic regression predictions for phase discrimination."""
import itertools

from helpers.config import *
from apa_analysis.config import global_settings, condition_specific_settings
from apa_analysis.Characterisation import General_utils as gu
from apa_analysis.Characterisation import DataClasses as dc
from apa_analysis.Characterisation.Plotting import PCA_plotting as pcap
from apa_analysis.Characterisation.AnalysisTools import Regression as reg
from apa_analysis.Characterisation.AnalysisTools import LDA


def run_pca_regressions(phases, stride_numbers, condition, pca_data, feature_data, stride_data, save_dir,
                        select_pcs=None, select_pc_type=None, feature_compare_data=None, feature_compare_stride_data=None):
    pca_predictions = []
    for p1, p2 in itertools.combinations(phases, 2):

        if not (len(pca_data) == 1 and pca_data[0].phase == (p1, p2)):
            raise ValueError("Not expecting more PCA data than for APA2 and Wash2 now!")
        else:
            pca = pca_data[0].pca
            pca_loadings = pca_data[0].pca_loadings

        if select_pcs is not None:
            pca_loadings = pca_loadings.loc(axis=1)[select_pcs]

        for s in stride_numbers:
            print(f"Running single feature predictions for {p1}-{p2} on stride {s}...")
            for midx in condition_specific_settings[condition]['global_fs_mouse_ids']:
                save_path = gu.create_mouse_save_directory(save_dir, midx, s, p1, p2)

                # --------- Collect and organise PCA/feature data ---------
                # # Get mouse run data
                featsxruns, featsxruns_phaseruns, run_ns, stepping_limbs, mask_p1, mask_p2 = gu.select_runs_data(
                    midx, s, feature_data, stride_data, p1, p2)
                pcs = pca.transform(featsxruns)

                pcs_p1 = pcs[mask_p1]
                pcs_p2 = pcs[mask_p2]
                pcs_p1p2 = np.vstack([pcs_p1, pcs_p2])

                # Check if need to run LDA on Wash data to find null space
                if feature_compare_data is not None and feature_compare_stride_data is not None:
                    featsxruns_compare, featsxruns_phaseruns_compare, run_ns_compare, stepping_limbs_compare, mask_p1_compare, mask_p2_compare = gu.select_runs_data(
                        midx, s, feature_compare_data, feature_compare_stride_data, p1, p2)
                    pcs_compare = pca.transform(featsxruns_compare)

                    if p2 == 'Wash2':
                        pcs_p2_compare = pcs_compare[mask_p2_compare]

                        pcs_wash = np.vstack([pcs_p2, pcs_p2_compare])
                        labels_wash = np.array([0] * pcs_p2.shape[0] + [1] * pcs_p2_compare.shape[0])

                        pcs_wash_trim = pcs_wash[:, :global_settings['pcs_to_use']]

                        _, lda_w, _, _, _, _ = LDA.compute_lda(pcs_wash_trim, labels_wash, folds=5)
                        lda_w_unit = lda_w / np.linalg.norm(lda_w)
                    else:
                        raise ValueError("Unexpected phase for comparison data. Expected 'Wash2'.")
                else:
                    lda_w_unit = None

                if select_pcs is None:
                    pcs_p1p2 = pcs_p1p2[:, :global_settings['pcs_to_use']] # todo is this correct?

                if select_pcs is not None:
                    if feature_compare_data is not None and feature_compare_stride_data is not None:
                        raise ValueError("Cannot select PCs when using comparison data for null space calculation.")
                    else:
                        select_PCs_as_numbers = [int(pc[-1]) - 1 for pc in select_pcs]
                        pcs_p1 = pcs_p1[:, select_PCs_as_numbers]
                        pcs_p2 = pcs_p2[:, select_PCs_as_numbers]
                        pcs_p1p2 = pcs_p1p2[:, select_PCs_as_numbers]

                labels_phase1 = np.array([p1] * pcs_p1.shape[0])
                labels_phase2 = np.array([p2] * pcs_p2.shape[0])
                labels = np.concatenate([labels_phase1, labels_phase2])


                # --------- Plot PCA projections for mouse ---------
                pcap.plot_pca(pca, pcs_p1p2, labels, p1, p2, s, stepping_limbs, run_ns, midx,
                            condition, save_path)


                # ----------- Run Regression on PCA data and use to predict full run set for mouse -----------
                results = reg.run_regression_on_PCA_and_predict(pca_loadings, pcs_p1p2, featsxruns,
                                                                featsxruns_phaseruns,
                                                                mask_p1, mask_p2,
                                                                midx, p1, p2, s,
                                                                condition, save_path,
                                                                select_pc_type,
                                                                lda_w_unit=lda_w_unit)
                (y_pred, smoothed_y_pred, feature_weights, w_PC, normalize_mean_pc, normalize_std_pc,
                 acc, cv_acc, w_folds, pc_acc, pc_y_preds, null_acc, pc_lesions_cv_acc, pc_lesions_w_folds, null_acc_circ,
                 w_single_pc, bal_acc_single_pc, cv_acc_single_pc, cv_acc_shuffle_single_pc, bal_acc_shuffle_single_pc) = results

                # Skipping w/in vs b/wn mice comparison

                # --------- Find significance of PC predictions ---------

                # Save the regression results
                pca_pred_class = dc.PCAPredictionData(phase=(p1, p2),
                                                      stride=s,
                                                      mouse_id=midx,
                                                      x_vals= featsxruns.index,
                                                      y_pred=y_pred,
                                                      y_pred_smoothed=smoothed_y_pred,
                                                      feature_weights=feature_weights,
                                                      pc_weights=w_PC,
                                                      accuracy=acc,
                                                      cv_acc=cv_acc,
                                                      w_folds=w_folds,
                                                      y_preds_PCwise=pc_y_preds,
                                                      pc_acc=pc_acc,
                                                      null_acc=null_acc,
                                                      null_acc_circ=null_acc_circ,
                                                      pc_lesions_cv_acc=pc_lesions_cv_acc,
                                                      pc_lesions_w_folds=pc_lesions_w_folds,
                                                      w_single_pc=w_single_pc,
                                                      bal_acc_single_pc=bal_acc_single_pc,
                                                      cv_acc_single_pc=cv_acc_single_pc,
                                                      cv_acc_shuffle_single_pc=cv_acc_shuffle_single_pc,
                                                      bal_acc_shuffle_single_pc=bal_acc_shuffle_single_pc
                                                      )
                pca_predictions.append(pca_pred_class)
    return pca_predictions





















