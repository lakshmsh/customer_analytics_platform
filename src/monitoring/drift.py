"""Data and Concept Drift Monitoring module for Customer Analytics Platform.

Implements Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
to proactively detect distribution shifts in customer features in production.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger("monitoring.drift")


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current data.

    Thresholds:
    - PSI < 0.10: No significant distribution change (Stable)
    - 0.10 <= PSI < 0.20: Moderate distribution change (Warning)
    - PSI >= 0.20: Significant distribution change (Retrain recommended)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on reference distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] -= epsilon
    bins[-1] += epsilon
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    # Compute frequency counts
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    # Normalize to proportions
    expected_pct = np.clip(expected_counts / len(expected), epsilon, 1.0)
    actual_pct = np.clip(actual_counts / len(actual), epsilon, 1.0)

    # Calculate PSI sum
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi_value, 4))


def run_ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """Run two-sample Kolmogorov-Smirnov test for distribution equivalence."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0, 1.0

    stat, p_val = stats.ks_2samp(expected, actual)
    return float(round(stat, 4)), float(round(p_val, 4))


def evaluate_dataset_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str]
) -> Dict[str, Any]:
    """Evaluate drift metrics across all specified features."""
    results = {}
    drift_detected = False

    for col in features:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
        curr_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values

        if len(ref_vals) == 0 or len(curr_vals) == 0:
            continue

        psi = calculate_psi(ref_vals, curr_vals)
        ks_stat, ks_pval = run_ks_test(ref_vals, curr_vals)

        if psi >= 0.20 or (ks_pval < 0.01 and ks_stat > 0.15):
            status = "Drift Detected"
            drift_detected = True
        elif psi >= 0.10 or ks_pval < 0.05:
            status = "Moderate Shift"
        else:
            status = "Stable"

        results[col] = {
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "status": status
        }

    return {
        "drift_detected": drift_detected,
        "total_features_evaluated": len(results),
        "features": results
    }
