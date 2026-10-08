"""
Evaluation metrics for glaucoma progression prediction.
Covers both regression accuracy and clinical utility.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Any


def compute_regression_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Standard regression metrics for progression rate prediction.
    All values in the same unit as the target (dB/year or %/year).
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    bias = float(np.mean(y_pred - y_true))

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}mae": mae,
        f"{p}rmse": rmse,
        f"{p}r2": r2,
        f"{p}bias": bias,
    }


def compute_clinical_metrics(
    md_pred: np.ndarray,
    md_true: np.ndarray,
    fast_threshold_db_year: float = -1.0,
) -> dict[str, float]:
    """
    Clinical utility metrics:
    - Fast progressor detection: sensitivity/specificity for identifying
      patients with MD rate < threshold (e.g., -1.0 dB/year)
    - Clinically meaningful error: fraction within 0.5 dB/year of truth
    """
    is_fast_true = (md_true < fast_threshold_db_year).astype(int)
    is_fast_pred = (md_pred < fast_threshold_db_year).astype(int)

    tp = int(((is_fast_pred == 1) & (is_fast_true == 1)).sum())
    tn = int(((is_fast_pred == 0) & (is_fast_true == 0)).sum())
    fp = int(((is_fast_pred == 1) & (is_fast_true == 0)).sum())
    fn = int(((is_fast_pred == 0) & (is_fast_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    within_half_db = float(np.mean(np.abs(md_pred - md_true) <= 0.5))

    auc = float("nan")
    if len(np.unique(is_fast_true)) == 2:
        # Lower (more negative) predicted rate → more likely fast progressor
        auc = float(roc_auc_score(is_fast_true, -md_pred))

    return {
        "fast_progressor_sensitivity": sensitivity,
        "fast_progressor_specificity": specificity,
        "fast_progressor_ppv": ppv,
        "fast_progressor_auc": auc,
        "within_half_db_pct": within_half_db,
    }


def compute_all_metrics(
    md_pred: np.ndarray,
    md_true: np.ndarray,
    vfi_pred: np.ndarray,
    vfi_true: np.ndarray,
) -> dict[str, Any]:
    """Compute full evaluation suite."""
    metrics = {}
    metrics.update(compute_regression_metrics(md_pred, md_true, prefix="md"))
    metrics.update(compute_regression_metrics(vfi_pred, vfi_true, prefix="vfi"))
    metrics.update(compute_clinical_metrics(md_pred, md_true))
    return metrics
