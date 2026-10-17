"""Unified evaluation metrics module for classification, clustering, recommendations, and forecasting."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error
)
from src.utils.logger import get_logger

logger = get_logger("evaluation.metrics")


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.50
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics for churn evaluation.

    Args:
        y_true: True binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (1).
        threshold: Decision boundary threshold.

    Returns:
        Dict[str, float]: Dictionary of evaluated metrics.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(loss, 4),
        "accuracy": round(acc, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    return metrics


def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate unsupervised clustering quality metrics.

    Args:
        X: Scaled feature matrix.
        labels: Assigned cluster labels.

    Returns:
        Dict[str, float]: Silhouette and Davies-Bouldin scores.
    """
    sil = float(silhouette_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    return {
        "silhouette_score": round(sil, 4),
        "davies_bouldin_score": round(db, 4),
        "num_clusters": int(len(np.unique(labels)))
    }


def compute_forecast_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> Dict[str, float]:
    """Calculate time-series forecasting regression metrics.

    Args:
        y_true: Ground truth actual values.
        y_pred: Predicted values.

    Returns:
        Dict[str, float]: MAE, RMSE, MAPE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    # Handle zero values in MAPE safely
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = float(mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask]))
    else:
        mape = 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 4)
    }
