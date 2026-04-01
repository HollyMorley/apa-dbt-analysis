"""Logistic and linear regression for predicting experimental phase from stride features."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy
from scipy.signal import medfilt
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon, ttest_1samp
import os
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from apa_analysis.Characterisation import General_utils as gu
from apa_analysis.Characterisation.Plotting import Regression_plotting as rp
from apa_analysis.config import condition_specific_settings, global_settings

np.random.seed(1)

def compute_linear_regression(X, y, folds=5):
    model = LinearRegression(fit_intercept=False)

    n_samples = X.shape[1]
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    cv_mse = []
    w_folds = []

    for train_idx, test_idx in kf.split(np.arange(n_samples)):
        model_fold = LinearRegression(fit_intercept=False)
        model_fold.fit(X[:, train_idx].T, y[train_idx])
        w_fold = model_fold.coef_
        y_pred = np.dot(w_fold, X[:, test_idx])
        mse_fold = mean_squared_error(y[test_idx], y_pred.ravel())
        cv_mse.append(mse_fold)
        w_folds.append(w_fold)

    cv_mse = np.array(cv_mse)
    w_folds = np.array(w_folds)

    model.fit(X.T, y)
    w = model.coef_
    y_pred = np.dot(w, X)
    full_mse = mean_squared_error(y, y_pred.ravel())

    return w, full_mse, cv_mse, w_folds
def compute_linear_regression_pcwise_prediction(X, y, w, shuffles=1000):
    #w = w[0]
    num_pcs = X.shape[0]

    pc_mse = np.zeros((num_pcs,))
    null_mse = np.zeros((num_pcs, shuffles))
    y_preds = np.zeros((num_pcs, X.shape[1]))

    for pc in range(num_pcs):
        wpc = w[pc]
        y_pred = np.dot(wpc, X[pc, :])
        y_preds[pc, :] = y_pred
        pc_mse[pc] = mean_squared_error(y, y_pred)

        for idx in range(shuffles):
            x_shuffle = np.random.permutation(X[pc, :].T).T
            y_pred_shuffle = np.dot(wpc, x_shuffle)
            shuffle_mse = mean_squared_error(y, y_pred_shuffle.ravel())
            null_mse[pc, idx] = shuffle_mse

    return pc_mse, y_preds, null_mse

def compute_regression_existing_model(X, y, w_folds):
    # cross-validate
    _splits = w_folds.shape[0]
    n_samples = X.shape[1]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = []

    for fold_idx, (_, test_idx) in enumerate(kf.split(np.arange(n_samples), y)):
        ### start loop through pcs
        # Train using the training columns
        # pick the corresponding fold weights
        w_fold = w_folds[fold_idx]
        # squeeze to shape (n_features,)
        # if w_fold.ndim == 3:
        #     # e.g. (1, n_features) inside
        #     w_fold = np.squeeze(w_fold, axis=0)  # now (1, n_features) -> (n_features,)
        # w_fold = np.squeeze(w_fold)

        y_pred = np.dot(w_fold, X[:, test_idx])
        y_pred[y_pred > 0] = 1
        y_pred[y_pred < 0] = 0

        acc_fold = balanced_accuracy(y[test_idx], y_pred.ravel())
        cv_acc.append(acc_fold)
    cv_acc = np.array(cv_acc)
    return cv_acc

def compute_regression(X, y, folds=5):
    model = LogisticRegression(penalty='l2', fit_intercept=False, solver='liblinear', C=0.5)

    # cross-validate
    n_samples = X.shape[1]
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_acc = []
    w_folds = []
    for train_idx, test_idx in kf.split(np.arange(n_samples), y):
        ### start loop through pcs
        # Create a new model instance for each fold
        model_fold = LogisticRegression(penalty='l2', fit_intercept=False, solver='liblinear', C=0.5)
        # Train using the training columns
        model_fold.fit(X[:, train_idx].T, y[train_idx])


        w_fold = model_fold.coef_
        y_pred = np.dot(w_fold, X[:, test_idx])
        y_pred[y_pred > 0] = 1
        y_pred[y_pred < 0] = 0

        acc_fold = balanced_accuracy(y[test_idx], y_pred.ravel())
        cv_acc.append(acc_fold)
        w_folds.append(w_fold)


    cv_acc = np.array(cv_acc)
    w_folds = np.array(w_folds)

    model.fit(X.T, y)
    w = model.coef_

    y_pred = np.dot(w, X)

    # change y_pred +ves to 1 and -ves to 0
    y_pred[y_pred > 0] = 1
    y_pred[y_pred < 0] = 0

    bal_acc = balanced_accuracy(y.T, y_pred.T)

    # y_pred3 = np.dot(w[0][2], X[2, :])
    # plt.figure()
    # plt.plot(y_pred3)
    # plt.show()

    return w, bal_acc, cv_acc, w_folds

def compute_null_accuracy_circular(Xdr_long, y_reg, mask_p1, mask_p2, num_repeats=100, folds=5, min_shift=25, max_shift=None):
    null_accuracies = []

    n_samples = Xdr_long.shape[1]
    if max_shift is None:
        max_shift = n_samples - 3  # for example, avoid wrapping around entirely

    for i in range(num_repeats):
        Xdr_null = np.zeros_like(Xdr_long)

        # draw shift within desired range (avoiding zero shift)
        # shift = np.random.randint(min_shift, max_shift+1)
        # for pc in range(Xdr_long.shape[0]):
        #     Xdr_null[pc] = np.roll(Xdr_long[pc], shift)

        # Select only p1 and p2 data points after shifting
        # Xdr_null_selected = Xdr_null[:, np.concatenate([np.where(mask_p1)[0], np.where(mask_p2)[0]])]

        shift1 = np.random.randint(min_shift, max_shift + 1)
        shift2 = np.random.randint(min_shift, max_shift + 1)
        mask_p1_shifted = np.roll(mask_p1, shift1)
        mask_p2_shifted = np.roll(mask_p2, shift2)
        while np.sum(mask_p1_shifted & mask_p2_shifted) > 25:  # ensure at least 3 samples in each phase
            shift1 = np.random.randint(min_shift, max_shift + 1)
            shift2 = np.random.randint(min_shift, max_shift + 1)
            mask_p1_shifted = np.roll(mask_p1, shift1)
            mask_p2_shifted = np.roll(mask_p2, shift2)

        # Select only p1 and p2 data points after shifting
        Xdr_null_selected = Xdr_long[:, np.concatenate([np.where(mask_p1_shifted)[0], np.where(mask_p2_shifted)[0]])]

        # Run regression on shifted-selected data
        w_null, bal_acc_null, cv_acc_null, w_folds_null = compute_regression(Xdr_null_selected, y_reg, folds=folds)
        null_accuracies.append(cv_acc_null)

    null_accuracies = np.array(null_accuracies)
    null_accuracies_mean = np.mean(null_accuracies, axis=0)
    return null_accuracies_mean


def compute_regression_lesion(X, y, folds=5, regressor_to_shuffle: int=None):
    # cross-validate
    n_samples = X.shape[1]
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_acc = []
    w_folds = []
    for train_idx, test_idx in kf.split(np.arange(n_samples), y):
        ### start loop through pcs
        # Create a new model instance for each fold
        model_fold = LogisticRegression(penalty='l2', fit_intercept=False, solver='liblinear', C=0.5)
        # Train using the training columns
        if regressor_to_shuffle is not None:
            X_lesion = X.copy()
            X_lesion[regressor_to_shuffle, train_idx] = np.random.permutation(X_lesion[regressor_to_shuffle, train_idx])

        model_fold.fit(X_lesion[:, train_idx].T, y[train_idx])

        w_fold = model_fold.coef_
        y_pred = np.dot(w_fold, X_lesion[:, test_idx])
        y_pred[y_pred > 0] = 1
        y_pred[y_pred < 0] = 0

        acc_fold = balanced_accuracy(y[test_idx], y_pred.ravel())
        cv_acc.append(acc_fold)
        w_folds.append(w_fold)


    cv_acc = np.array(cv_acc)
    w_folds = np.array(w_folds)


    return cv_acc, w_folds

def compute_regression_pcwise_prediction(X, y, w, folds=10, shuffles=1000):
    w = w[0]
    n_samples = X.shape[1]
   # kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    num_pcs = X.shape[0]

    pc_acc = np.zeros((num_pcs,)) # pcs x folds
    null_acc = np.zeros((num_pcs, shuffles)) # pcs x folds
    y_preds = np.zeros((num_pcs, X.shape[1])) # pcs x runs
    for pc in range(num_pcs):
        wpc = w[pc]
        y_pred = np.dot(wpc, X[pc, :])  # wpc is a row vector, X is a column vector
        y_preds[pc, :] = y_pred
        y_pred[y_pred > 0] = 1
        y_pred[y_pred < 0] = 0
        pc_acc[pc] = balanced_accuracy(y, y_pred.ravel())  # not sure why ravel

        for idx in range(shuffles):
            x_shuffle = np.random.permutation(X[pc, :].T).T
            y_pred_shuffle = np.dot(wpc, x_shuffle)
            y_pred_shuffle[y_pred_shuffle > 0] = 1
            y_pred_shuffle[y_pred_shuffle < 0] = 0
            bal_acc = balanced_accuracy(y, y_pred_shuffle)  # not sure why ravel
            null_acc[pc, idx] = bal_acc


    return pc_acc, y_preds, null_acc

def compute_single_pc_regression(X, y, folds, shuffles=100):
    len_pcs = X.shape[0]

    w_x_pcs = np.zeros((len_pcs, folds))
    bal_acc_x_pcs = np.zeros((len_pcs, folds))
    cv_acc_x_pcs = np.zeros((len_pcs, folds))

    cv_acc_shuffle_x_pcs = np.zeros((len_pcs, shuffles, folds))
    bal_acc_shuffle_x_pcs = np.zeros((len_pcs, shuffles, folds))

    for pc in range(len_pcs): # should be pc x run
        Xpc = X[pc, :].reshape(1, -1)  # reshape to 2D array for regression
        w_x_pcs[pc,:], bal_acc_x_pcs[pc,:], cv_acc_x_pcs[pc,:], _ = compute_regression(Xpc, y, folds=folds)

        for i in tqdm(np.arange(shuffles)):
            x_pc_shuffle = np.random.permutation(Xpc.T).T
            _, bal_acc_shuffle_x_pcs[pc,i,:], cv_acc_shuffle_x_pcs[pc,i,:], _ = compute_regression(x_pc_shuffle, y, folds=folds)

    return w_x_pcs, bal_acc_x_pcs, cv_acc_x_pcs, cv_acc_shuffle_x_pcs, bal_acc_shuffle_x_pcs



def run_regression_on_PCA_and_predict(loadings: pd.DataFrame,
                       pcs: np.ndarray,
                       feature_data: pd.DataFrame,
                       selected_feature_data: pd.DataFrame,
                       mask_p1: np.ndarray,
                       mask_p2: np.ndarray,
                       mouse_id: str,
                       p1: str, p2: str, s: int,
                       condition: str,
                       save_path: str,
                       select_pc_type: str = None,
                       lda_w_unit: np.ndarray = None,
                       ) -> tuple:

    # # Fit regression model on PCA data
    # if global_settings['stride_numbers'] == [0]:
    #     print('Picking out every other run for stride 0')
    #     selected_feature_data = selected_feature_data[::2]
    #     mask_p1 = mask_p1[::2]
    #     mask_p2 = mask_p2[::2]
    results = fit_regression_model(loadings, selected_feature_data, feature_data, mask_p1, mask_p2, mouse_id, s, select_pc_type, lda_w_unit=lda_w_unit)
    (w, normalize_mean, normalize_std, y_reg, full_accuracy, cv_acc, w_folds, pc_acc, y_preds, null_acc, pc_lesions_cv_acc, pc_lesions_w_folds, null_acc_circ,
     w_single_pc, bal_acc_single_pc, cv_acc_single_pc, cv_acc_shuffle_single_pc, bal_acc_shuffle_single_pc) = results

    w = np.array(w[0][:global_settings['pcs_to_use']]).reshape(1, -1)
    normalize_mean = normalize_mean[:global_settings['pcs_to_use']]
    normalize_std = normalize_std[:global_settings['pcs_to_use']]
    loadings = loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()

    # Transform regression weights in PCA space to feature space
    feature_weights = loadings.dot(w.T).squeeze()

    pc_weights = pd.Series(w[0], index=loadings.columns)

    # Plot the weights in the original feature space
    rp.plot_weights_in_feature_space(feature_weights, save_path, mouse_id, p1, p2, s, condition)
    rp.plot_weights_in_pc_space(pc_weights, save_path, mouse_id, p1, p2, s, condition)

    # Predict runs using the full model
    smoothed_y_pred, y_pred = predict_runs(loadings, feature_data, normalize_mean, normalize_std, w, #todo check dtypes are correct
                                           save_path, mouse_id, p1, p2, s, condition)

    return y_pred, smoothed_y_pred, feature_weights, w, normalize_mean, normalize_std, full_accuracy, cv_acc, w_folds, pc_acc, y_preds, null_acc,pc_lesions_cv_acc, pc_lesions_w_folds, null_acc_circ,w_single_pc, bal_acc_single_pc, cv_acc_single_pc, cv_acc_shuffle_single_pc, bal_acc_shuffle_single_pc


def fit_regression_model(loadings: pd.DataFrame, selected_feature_data: pd.DataFrame, feature_data: pd.DataFrame,
                         mask_p1: np.ndarray, mask_p2: np.ndarray, mouse_id: str, s: int, select_pc_type: str = None,
                         lda_w_unit: np.ndarray = None):
    # trim pc loadings
    loadings = loadings.iloc(axis=1)[:global_settings['pcs_to_use']].copy()
    # Transform X (scaled feature data) to Xdr (PCA space) - ie using the loadings from PCA
    Xdr = np.dot(loadings.T, selected_feature_data)
    Xdr_long = np.dot(loadings.T, feature_data.T)


    # Normalize X
    _, normalize_mean, normalize_std = gu.normalize_Xdr(Xdr)

    # Create y (regression target) - 1 for phase1, 0 for phase2
    y_reg = np.concatenate([np.ones(np.sum(mask_p1)), np.zeros(np.sum(mask_p2))])

    null_acc_circ = compute_null_accuracy_circular(Xdr_long, y_reg, mask_p1, mask_p2, folds=10)
    print(f"Mean null accuracy: {np.mean(null_acc_circ):.3f}")

    # Run logistic regression on the full model
    if not global_settings["use_LH_reg_model"]:
        num_folds = 10
        w, bal_acc, cv_acc, w_folds = compute_regression(Xdr, y_reg, folds=num_folds)
        pc_acc, y_preds, null_acc = compute_regression_pcwise_prediction(Xdr, y_reg, w)
        w_single_pc, bal_acc_single_pc, cv_acc_single_pc, cv_acc_shuffle_single_pc, bal_acc_shuffle_single_pc = compute_single_pc_regression(Xdr, y_reg, folds=num_folds, shuffles=100)
        pc_lesions_cv_acc = np.zeros((global_settings['pcs_to_use'], num_folds))
        pc_lesions_w_folds = np.zeros((global_settings['pcs_to_use'], num_folds, global_settings['pcs_to_use']))
        for pc in range(global_settings['pcs_to_use']):
            cv_acc_lesion, w_folds_lesion = compute_regression_lesion(Xdr, y_reg, folds=num_folds, regressor_to_shuffle=pc)
            pc_lesions_cv_acc[pc, :] = cv_acc_lesion
            pc_lesions_w_folds[pc, :, :] = w_folds_lesion.squeeze()

        if lda_w_unit is not None:
            # Remove w projected on the lda axis from w
            w = w - np.dot(w, lda_w_unit) * lda_w_unit

            # w_folds and pc_lesions_w_folds are 2D arrays, so we need to iterate over folds
            for fold in range(num_folds):
                w_folds[fold] = w_folds[fold] - np.dot(w_folds[fold], lda_w_unit) * lda_w_unit
            for pc in range(global_settings['pcs_to_use']):
                for fold in range(num_folds):
                    pc_lesions_w_folds[pc, fold] = pc_lesions_w_folds[pc, fold] - np.dot(pc_lesions_w_folds[pc, fold], lda_w_unit) * lda_w_unit


    else:
        # load the regression from the LowHigh model
        multipath = r"H:\Characterisation_v2\LH_res_-3-2-1_APA2Wash2\APAChar_LowHigh_Extended\MultiFeaturePredictions"
        if select_pc_type is None:
            LH_reg_path = os.path.join(multipath, r"pca_predictions_APAChar_LowHigh.pkl")  #r"H:\Characterisation\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_APAChar_LowHigh.pkl"
        elif select_pc_type == "Top3":
            top3_path = os.path.join(multipath, "Top3PCs")
            LH_reg_path = os.path.join(top3_path, r"pca_predictions_top3_APAChar_LowHigh.pkl")   #r"H:\Characterisation\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_top3_APAChar_LowHigh.pkl"
        elif select_pc_type == "TopAnyLen":
            topanylen_path = os.path.join(multipath, "TopAnyPCs")
            LH_reg_path = os.path.join(topanylen_path, r"pca_predictions_top_anylen_APAChar_LowHigh.pkl")
        elif select_pc_type == "Bottom9":
            bottom9_path = os.path.join(multipath, "Bottom9PCs")
            LH_reg_path = os.path.join(bottom9_path, r"pca_predictions_bottom9_APAChar_LowHigh.pkl")  #r"H:\Characterisation\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_bottom9_APAChar_LowHigh.pkl"
        elif select_pc_type == "Bottom3":
            bottom3_path = os.path.join(multipath, "Bottom3PCs")
            LH_reg_path = os.path.join(bottom3_path, r"pca_predictions_bottom3_APAChar_LowHigh.pkl")  #r"H:\Characterisation\LH_res_-3-2-1_APA2Wash2-PCStot=60-PCSuse=12\APAChar_LowHigh_Extended\MultiFeaturePredictions\pca_predictions_bottom3_APAChar_LowHigh.pkl"
        else:
            raise ValueError(f"Unknown select_pc_type: {select_pc_type}")

        with open(LH_reg_path, 'rb') as f:
            pca_pred = pickle.load(f)
        pred = [p for p in pca_pred if p.mouse_id == mouse_id and p.stride == s and p.phase == (global_settings['phases'][0], global_settings['phases'][1])]
        if len(pred) == 0:
            raise ValueError(f"No LH prediction found for mouse {mouse_id} with stride {s} and phase {global_settings['phases'][0]}-{global_settings['phases'][1]}")
        elif len(pred) > 1:
            raise ValueError(f"Multiple LH predictions found for mouse {mouse_id} with stride {s} and phase {global_settings['phases'][0]}-{global_settings['phases'][1]}")
        pred = pred[0]
        # w = pred.pc_weights
        # bal_acc = pred.accuracy
        # cv_acc = pred.cv_acc
        # w_folds = pred.w_folds
        # pc_acc = pred.pc_acc
        # y_preds = pred.y_preds_PCwise
        # null_acc = pred.null_acc
        # pc_lesions_cv_acc = pred.pc_lesions_cv_acc
        # pc_lesions_w_folds = pred.pc_lesions_w_folds
        w = pred.pc_weights
        bal_acc = pred.accuracy
        # need to calculate cv_acc new for this data!!
        cv_acc = compute_regression_existing_model(Xdr, y_reg, pred.w_folds)
        # cv_acc = pred.cv_acc
        w_folds = pred.w_folds
        pc_acc = pred.pc_acc
        y_preds = pred.y_preds_PCwise
        null_acc = pred.null_acc
        pc_lesions_cv_acc = pred.pc_lesions_cv_acc
        pc_lesions_w_folds = pred.pc_lesions_w_folds
        null_acc_circ = pred.null_acc_circ
        w_single_pc = pred.w_single_pc
        bal_acc_single_pc = pred.bal_acc_single_pc
        cv_acc_single_pc = pred.cv_acc_single_pc
        cv_acc_shuffle_single_pc = pred.cv_acc_shuffle_single_pc
        bal_acc_shuffle_single_pc = pred.bal_acc_shuffle_single_pc


    # mean_cv_acc_PCwise = np.mean(cv_acc_PCwise, axis=1)
    # mean_cv_acc_shuffle_PCwise = np.mean(cv_acc_shuffle_PCwise, axis=1)

    # w, full_accuracy, cv_acc = compute_regression(Xdr, y_reg)
    print(f"Full model accuracy: {bal_acc:.3f}")

    return w, normalize_mean, normalize_std, y_reg, bal_acc, cv_acc, w_folds, pc_acc, y_preds, null_acc, pc_lesions_cv_acc, pc_lesions_w_folds, null_acc_circ, w_single_pc, bal_acc_single_pc, cv_acc_single_pc, cv_acc_shuffle_single_pc, bal_acc_shuffle_single_pc

def predict_runs(loadings: pd.DataFrame, feature_data: pd.DataFrame, normalize_mean: float, normalize_std: float,
                 w: np.ndarray, save_path: str, mouse_id: str, p1: str, p2:str, s: int, condition_name: str):
    # Apply the full model to all runs (scaled and unscaled)
    all_trials_dr = np.dot(loadings.T, feature_data.T)

    all_trials_dr = ((all_trials_dr.T - normalize_mean) / normalize_std).T # pc wise normalization
    run_pred = np.dot(w, np.dot(loadings.T, feature_data.T))
    run_pred_scaled = np.dot(w, all_trials_dr)

    # Compute smoothed scaled predictions for aggregation.
    kernel_size = 5
    padded_run_pred = np.pad(run_pred[0], pad_width=kernel_size, mode='reflect')
    padded_run_pred_scaled = np.pad(run_pred_scaled[0], pad_width=kernel_size, mode='reflect')
    smoothed_pred = medfilt(padded_run_pred, kernel_size=kernel_size)
    smoothed_scaled_pred = medfilt(padded_run_pred_scaled, kernel_size=kernel_size)
    smoothed_pred = smoothed_pred[kernel_size:-kernel_size]
    smoothed_scaled_pred = smoothed_scaled_pred[kernel_size:-kernel_size]

    # Plot run prediction
    rp.plot_run_prediction(feature_data, run_pred_scaled, smoothed_scaled_pred, save_path, mouse_id, p1, p2, s,
                              scale_suffix="scaled", dataset_suffix=condition_name)
    return smoothed_scaled_pred, run_pred_scaled

def calculate_PC_prediction_significances(pca_pred, stride, conditions, mice_thresh_percent=0.75, accmse='acc', lesion_significance=False):
    if conditions:
        mouse_stride_preds = [pred for pred in pca_pred if pred.stride == stride and pred.conditions == conditions]
    else:
        mouse_stride_preds = [pred for pred in pca_pred if pred.stride == stride]

    single_pc_model_full_acc = []
    single_pc_model_cv_acc = []
    single_pc_model_null_cv_acc = []
    pc_weights = []
    if lesion_significance:
        pc_lesions_cv_acc = []
    for mouse_pred in mouse_stride_preds:
        single_pc_model_full_acc.append(mouse_pred.bal_acc_single_pc)
        single_pc_model_cv_acc.append(mouse_pred.pc_acc) # pcs   #cv_acc_single_pc) # pcs x fold
        single_pc_model_null_cv_acc.append(mouse_pred.null_acc) # pcs x shuffles     #cv_acc_shuffle_single_pc) # pcs x shuffle x fold
        pc_weights.append(mouse_pred.pc_weights[0]) if mouse_pred.w_single_pc.ndim == 2 else pc_weights.append(mouse_pred.pc_weights)


        if lesion_significance:
            pc_lesions_cv_acc.append(mouse_pred.pc_lesions_cv_acc)

    accuracies_single_pc = np.array(single_pc_model_full_acc)  # mice x pcs x fold
    accuracies_single_pc_cv = np.array(single_pc_model_cv_acc)  # mice x pcs x fold
    accuracies_single_pc_null_cv = np.array(single_pc_model_null_cv_acc)  # mice x pcs x shuffle x fold

    accuracies_single_pc = np.mean(accuracies_single_pc, axis=2)  # mice x pcs
    #accuracies_single_pc_cv = np.mean(accuracies_single_pc_cv, axis=2)  # mice x pcs
    accuracies_single_pc_null_cv = np.mean(accuracies_single_pc_null_cv, axis=2)    #.mean(axis=2)  # mice x pcs

    mean_accs = accuracies_single_pc.mean(axis=0)  # pcs

    pc_weights = np.array(pc_weights)
    pos_counts = (pc_weights > 0).sum(axis=0)
    neg_counts = (pc_weights < 0).sum(axis=0)
    max_counts = np.maximum(pos_counts, neg_counts)
    total_mice_num = len(mouse_stride_preds)
    percent_uniform = max_counts / total_mice_num
    mice_uniform = percent_uniform >= mice_thresh_percent
    # ideal_mice_num  = total_mice_num - mice_thresh
    # counts_more_than_thresh = max_counts >= ideal_mice_num


    if not lesion_significance:
        if accmse == 'acc': # ie higher is better
            delta_acc_by_mouse = accuracies_single_pc_cv - accuracies_single_pc_null_cv
        # elif accmse == 'mse': # ie lower is better
        #     delta_acc_by_mouse = accuracies_full_model.mean(axis=2) - full_model_accuracy

        pc_significances = np.zeros((delta_acc_by_mouse.shape[1],))
        for pc in np.arange(delta_acc_by_mouse.shape[1]):
            pc_acc = delta_acc_by_mouse[:, pc]
            stat = ttest_1samp(pc_acc, 0)
            pc_significances[pc] = stat.pvalue
    # else:
    #     delta_acc_by_mouse_fold = full_model_accuracy[:, np.newaxis, :] - accuracies_lesions_x_pcs
    #     delta_acc_by_mouse = delta_acc_by_mouse_fold.mean(axis=2)  # mice x pcs
    #
    #     pc_significances = np.zeros((global_settings['pcs_to_use'],))
    #     for pc in np.arange(global_settings['pcs_to_use']):
    #         stat = ttest_1samp(delta_acc_by_mouse[:, pc], 0)
    #         pc_significances[pc] = stat.pvalue

    return pc_significances, mean_accs, mice_uniform, max_counts, total_mice_num

def find_residuals(feature_data, stride_numbers, phases, savedir):
    all_residuals = []
    for s in stride_numbers:
        print(f"Stride {s} - finding residuals for {phases[0]} and {phases[1]}")
        stride_features = feature_data.loc(axis=0)[s]
        # mask_p1, mask_p2 = gu.get_mask_p1_p2(stride_features, phases[0], phases[1])
        # phase_stride_features = pd.concat(
        #     [stride_features[mask_p1], stride_features[mask_p2]])
        residuals = stride_features.groupby(level='MouseID').apply(gu.compute_residuals, s, savedir)
        all_residuals.append(residuals)
    all_residuals_df =  pd.concat(all_residuals, keys=stride_numbers, names=['Stride'])

    # save
    all_residuals_df.to_hdf(os.path.join(savedir, "ResidualData.h5"), key="residuals", mode="w")
    return all_residuals_df

def find_model_cv_accuracy(predicition_data, stride, phases, save_dir):
    mice_cv_accs = [pred.cv_acc for pred in predicition_data if pred.stride == stride and pred.phase == (phases[0], phases[1])]
    mice_names = [pred.mouse_id for pred in predicition_data if pred.stride == stride and pred.phase == (phases[0], phases[1])]
    mice_mean_cv_acc = np.zeros(len(mice_cv_accs))
    for m in range(len(mice_cv_accs)):
        mice_mean_cv_acc[m] = np.mean(mice_cv_accs[m])
    # mean_cv_acc = np.mean(mice_mean_cv_acc)

    mice_cv_accs_df = pd.DataFrame(mice_mean_cv_acc, index=mice_names, columns=["MeanCVAcc"])

    # save mean_cv_acc into csv
    save_path = os.path.join(save_dir, f"MiceCVAccs_{phases[0]}_{phases[1]}_stride{stride}.csv")
    mice_cv_accs_df.to_csv(save_path)
    return mice_cv_accs_df




















