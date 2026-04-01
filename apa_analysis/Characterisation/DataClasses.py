"""Data container classes for organizing feature and stride data across analyses."""
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field

import pandas as pd


# ----------------------------------------------------------
# ----------------- Single Feature Model -------------------
# ----------------------------------------------------------

@dataclass
class SinglePredictionData:
    phase: tuple  # e.g., (p1, p2)
    stride: int
    feature: str
    mouse_id: str
    run_vals: np.ndarray
    w: int
    y_pred: any
    acc: float
    cv_acc: np.ndarray
    w_folds: np.ndarray

@dataclass
class SingleFeatureDataSummary:
    phase: tuple
    stride: int
    feature: str
    run_vals: np.ndarray
    w: int
    y_pred: np.ndarray
    acc: float
    cv_acc: np.ndarray


# ----------------------------------------------------------
# --------------------- Full PCA Model ---------------------
# ----------------------------------------------------------

@dataclass
class PCAData:
    phase: tuple
    stride: any
    pca: object
    pcs: np.ndarray
    pca_loadings: pd.DataFrame

@dataclass
class PCAPredictionData:
    phase: tuple
    stride: int
    mouse_id: str
    x_vals: List[float]
    y_pred: np.ndarray
    y_pred_smoothed: np.ndarray
    feature_weights: pd.Series
    pc_weights: pd.Series
    accuracy: float
    cv_acc: np.ndarray
    w_folds: np.ndarray
    y_preds_PCwise: np.ndarray
    pc_acc: np.ndarray
    null_acc: np.ndarray
    null_acc_circ: np.ndarray
    w_single_pc: np.ndarray
    bal_acc_single_pc: np.ndarray
    cv_acc_single_pc: np.ndarray
    cv_acc_shuffle_single_pc: np.ndarray
    bal_acc_shuffle_single_pc: np.ndarray
    pc_lesions_cv_acc: Optional[np.ndarray] = None  # Optional for lesion analysis
    pc_lesions_w_folds: Optional[np.ndarray] = None  # Optional for lesion analysis


    # cv_acc_PCwise: np.ndarray
    # shuffle_acc: np.ndarray
    # mean_acc_PCwise: np.ndarray
    # mean_acc_shuffle_PCwise: np.ndarray

@dataclass
class FeatureWeights:
    mouse_id: str
    feature_weights: pd.DataFrame

@dataclass
class LDAPredictionData: # --> for condition comparison
    conditions: List[str]
    phase: str
    stride: int
    mouse_id: str
    x_vals: np.ndarray
    y_pred: np.ndarray
    y_preds_pcs: np.ndarray
    weights: np.ndarray
    accuracy: float
    cv_acc: np.ndarray
    w_folds: np.ndarray
    pc_acc: np.ndarray
    null_acc: np.ndarray

@dataclass
class RegressionPredicitonData: # --> for condition comparison
    conditions: List[str]
    phase: str
    stride: int
    mouse_id: str
    x_vals: np.ndarray
    y_pred: np.ndarray
    y_preds_pcs: np.ndarray
    pc_weights: np.ndarray # note this is different to LDA to accomodate existing regression functions
    accuracy: float
    cv_acc: np.ndarray
    w_folds: np.ndarray
    pc_acc: np.ndarray
    null_acc: np.ndarray
    w_single_pc: np.ndarray = None #Optional
    bal_acc_single_pc: np.ndarray = None #Optional
    cv_acc_single_pc: np.ndarray = None #Optional
    cv_acc_shuffle_single_pc: np.ndarray = None #Optional
    bal_acc_shuffle_single_pc: np.ndarray = None #Optional
    pc_lesions_cv_acc: Optional[np.ndarray] = None  # Optional for lesion analysis
    pc_lesions_w_folds: Optional[np.ndarray] = None  # Optional for lesion analysis



