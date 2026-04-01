"""Linear discriminant analysis for classifying experimental phases from stride features."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy
from scipy.stats import ttest_1samp

def compute_lda(X, y, folds=5):
    """
    Compute LDA weights and balanced accuracy using cross-validation.
    :param X:
    :param y:
    :param folds:
    :return: w: LDA weights, bal_acc: balanced accuracy, cv_acc: cross-validated accuracy, w_folds: weights for each fold
    """
    lda = LDA()

    # cross validate
    n_samples = X.shape[0]
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_acc = []
    w_folds = []
    for train_idx, test_idx in kf.split(np.arange(n_samples), y):
        ### start loop through pcs
        # Create a new model instance for each fold
        lda_fold = LDA()
        lda_fold.fit(X[train_idx], y[train_idx])

        w_fold = lda_fold.coef_[0]
        y_pred_fold = np.dot(X[test_idx], w_fold) + lda_fold.intercept_[0]  # Linear combination of features and weights
        y_pred_int = y_pred_fold.copy()
        y_pred_int[y_pred_int < 0] = 0
        y_pred_int[y_pred_int > 0] = 1

        acc_fold = balanced_accuracy(y[test_idx], y_pred_int)
        cv_acc.append(acc_fold)
        w_folds.append(w_fold)

    cv_acc = np.array(cv_acc)
    w_folds = np.array(w_folds)

    # Fit the model on the entire dataset
    lda.fit(X, y)
    w = lda.coef_[0]
    intercept = lda.intercept_[0]
    y_pred = np.dot(X, w) + intercept # Linear combination of features and weights
    y_pred_int = y_pred.copy()
    y_pred_int[y_pred_int < 0] = 0
    y_pred_int[y_pred_int > 0] = 1
    bal_acc = balanced_accuracy(y, y_pred_int)

    return y_pred, w, bal_acc, w_folds, cv_acc, intercept

def compute_lda_pcwise(X, y, w, intercept, shuffles=1000):
    n_samples = X.shape[0]
    n_pcs = X.shape[1]

    pc_acc = np.zeros((n_pcs,))  # pcs x folds
    null_acc = np.zeros((n_pcs, shuffles))  # pcs x folds
    y_preds = np.zeros((n_pcs, X.shape[0]))  # pcs x runs

    for pc in range(n_pcs):
        wpc = w[pc]
        y_pred = np.dot(wpc, X[:, pc]) + intercept
        y_preds[pc,:] = y_pred

        y_pred_int = y_pred.copy()
        y_pred_int[y_pred_int < 0] = 0
        y_pred_int[y_pred_int > 0] = 1
        acc = balanced_accuracy(y, y_pred_int)
        pc_acc[pc] = acc

        # Shuffle the labels and compute the accuracy
        for idx in range(shuffles):
            x_shuffle = np.random.permutation(X[:, pc])
            y_pred_shuffle = np.dot(wpc, x_shuffle) + intercept
            y_pred_int_shuffle = y_pred_shuffle.copy()
            y_pred_int_shuffle[y_pred_int_shuffle < 0] = 0
            y_pred_int_shuffle[y_pred_int_shuffle > 0] = 1
            bal_acc = balanced_accuracy(y, y_pred_int_shuffle)
            null_acc[pc, idx] = bal_acc
    return pc_acc, null_acc, y_preds

# def calculate_PC_prediction_significances(lda_data, s, conditions, mice_thresh_percent, accmse='acc', lesion_significance=False): # ignore accmse and lesion_significance here
#     if conditions:
#         mouse_stride_preds = [pred for pred in lda_data if pred.stride == s and pred.conditions == conditions]
#     else:
#         mouse_stride_preds = [pred for pred in lda_data if pred.stride == s]
#
#     accuracies_x_pcs = []
#     accuracies_pcs_x_shuffle = []
#     pc_weights = []
#     for mouse_pred in mouse_stride_preds:
#         accuracies_x_pcs.append(mouse_pred.pc_acc)
#         accuracies_pcs_x_shuffle.append(mouse_pred.null_acc)
#         pc_weights.append(mouse_pred.weights)
#     accuracies_x_pcs = np.array(accuracies_x_pcs)  # mice x pcs
#     accuracies_shuffle_x_pcs = np.array(accuracies_pcs_x_shuffle)  # mice x pcs x shuffle
#     mean_accs = accuracies_x_pcs.mean(axis=0)
#
#     pc_weights = np.array(pc_weights)
#     pos_counts = (pc_weights > 0).sum(axis=0)
#     neg_counts = (pc_weights < 0).sum(axis=0)
#     max_counts = np.maximum(pos_counts, neg_counts)
#     total_mice_num = len(mouse_stride_preds)
#     percent_uniform = max_counts / total_mice_num
#     mice_uniform = percent_uniform >= mice_thresh_percent
#     # ideal_mice_num  = total_mice_num - mice_thresh
#     # counts_more_than_thresh = max_counts >= ideal_mice_num
#
#     delta_acc_by_mouse = accuracies_x_pcs - accuracies_shuffle_x_pcs.mean(axis=2)
#     pc_significances = np.zeros((delta_acc_by_mouse.shape[1]))
#     for pc in np.arange(delta_acc_by_mouse.shape[1]):
#         pc_acc = delta_acc_by_mouse[:, pc]
#         stat = ttest_1samp(pc_acc, 0)
#         pc_significances[pc] = stat.pvalue
#
#     return pc_significances, mean_accs, mice_uniform, max_counts, total_mice_num


