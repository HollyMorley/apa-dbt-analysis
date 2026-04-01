"""Single-feature logistic regression predictions for discriminating experimental phases."""
import pickle
import os
import itertools
import numpy as np
import pandas as pd

from apa_analysis.Characterisation import General_utils as gu
from apa_analysis.Characterisation import DataClasses as dc
from apa_analysis.Characterisation.AnalysisTools import Regression as reg
from apa_analysis.config import condition_specific_settings

def run_single_feature_regressions(phases, stride_numbers, condition, feature_names, feature_data, stride_data,
                                   save_path, filename):
    print("Running single feature predictions...")
    # find how many nans in feature data
    nans = np.isnan(feature_data.values)
    single_feature_predictions = []
    for p1, p2 in itertools.combinations(phases, 2):
        for s in stride_numbers:
            print(f"Running single feature predictions for {p1}-{p2} on stride {s}...")
            for midx in condition_specific_settings[condition]['global_fs_mouse_ids']:
                allfeatsxruns, allfeatsxruns_phaseruns, run_ns, stepping_limbs, mask_p1, mask_p2 = gu.select_runs_data(
                    midx, s, feature_data, stride_data, p1, p2)
                y_reg = np.concatenate([np.ones(np.sum(mask_p1)), np.zeros(np.sum(mask_p2))])
                for f in feature_names:
                    featurexruns_phases = allfeatsxruns_phaseruns.loc(axis=0)[f]
                    X = featurexruns_phases.values.reshape(-1, 1)
                    w, bal_acc, cv_acc, w_folds = reg.compute_regression(X.T, y_reg)

                    # Now predict all runs
                    featurexruns = allfeatsxruns.loc(axis=1)[f]
                    X_all = featurexruns.values.reshape(-1, 1)
                    y_pred = np.dot(w, X_all.T)
                    all_runs = np.array(list(allfeatsxruns.index))

                    feat_pred_class = dc.SinglePredictionData(phase=(p1, p2),
                                                              stride=s,
                                                              feature=f,
                                                              mouse_id=midx,
                                                              run_vals=all_runs,
                                                              w=w,
                                                              y_pred=y_pred,
                                                              acc=bal_acc,
                                                              cv_acc = cv_acc,
                                                              w_folds = w_folds)

                    single_feature_predictions.append(feat_pred_class)

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, filename), 'wb') as f:
        pickle.dump(single_feature_predictions, f)

    return single_feature_predictions

def get_summary_and_top_single_feature_data(number_feats, phases, stride_numbers, feature_names,
                                            single_feature_predictions):
    # Find the top x features for each phase combo and stride
    single_feature_predictions_summary = []
    common_x = np.arange(0, 160)
    top_feats = {}
    for p1, p2 in itertools.combinations(phases, 2):
        for s in stride_numbers:
            feat_acc = {}
            for f in feature_names:
                # Get the predictions for this phase combo and stride
                single_feat_pred = [pred for pred in single_feature_predictions if
                                    pred.phase == (p1, p2) and pred.stride == s and pred.feature == f]

                # Get average for w, acc, y_pred
                average_feat_acc = np.median([pred.acc for pred in single_feat_pred])
                average_feat_cv_acc = np.median([pred.cv_acc for pred in single_feat_pred], axis=0)
                average_w = np.median([pred.w for pred in single_feat_pred])
                interpolated_y_pred = []
                for pred in single_feat_pred:
                    x_vals = np.array(pred.run_vals)
                    y_pred = np.array(pred.y_pred)[0]
                    y_pred_interp = np.interp(common_x, x_vals, y_pred)
                    interpolated_y_pred.append(y_pred_interp)
                average_y_pred = np.mean(interpolated_y_pred, axis=0)

                feat_acc[f] = average_feat_acc

                Feat_pred_summary_class = dc.SingleFeatureDataSummary(phase=(p1, p2),
                                                                      stride=s,
                                                                      feature=f,
                                                                      run_vals=common_x,
                                                                      w=average_w,
                                                                      acc=average_feat_acc,
                                                                      cv_acc=average_feat_cv_acc,
                                                                      y_pred=average_y_pred)
                single_feature_predictions_summary.append(Feat_pred_summary_class)

            feat_acc_df = pd.DataFrame.from_dict(feat_acc, orient='index', columns=['acc'])
            feat_acc_df = feat_acc_df.sort_values(by='acc', ascending=False)
            top_feat_names = feat_acc_df.index[:number_feats].tolist()
            top_feats[((p1, p2), s)] = top_feat_names


    return single_feature_predictions_summary, top_feats

