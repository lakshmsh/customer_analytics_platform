"""CLI Script for evaluating Data Drift between baseline feature store and incoming batches."""

import sys
from pathlib import Path
import json
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.monitoring.drift import evaluate_dataset_drift
from src.utils.logger import get_logger

logger = get_logger("scripts.check_drift")


def main():
    logger.info("Initiating Data Drift Analysis...")
    ref_path = Path("data/processed/telco_processed.parquet")
    if not ref_path.exists():
        logger.error(f"Baseline feature store not found at {ref_path}.")
        sys.exit(1)

    ref_df = pd.read_parquet(ref_path)

    # In production, current_df represents a new batch of scored customers.
    # For demonstration/testing, we sample from reference with subtle simulated shift:
    sim_current = ref_df.sample(frac=0.5, random_state=42).copy()
    sim_current["monthly_charges"] = sim_current["monthly_charges"] * 1.05

    eval_features = ["tenure", "monthly_charges", "total_charges", "charges_tenure_ratio"]
    report = evaluate_dataset_drift(ref_df, sim_current, eval_features)

    print("\n" + "=" * 60)
    print(" 📊 FEATURE DRIFT EVALUATION REPORT (PSI & KS-TEST)")
    print("=" * 60)
    for feat, metrics in report["features"].items():
        print(f"Feature: {feat:<24} | PSI: {metrics['psi']:.4f} | Status: {metrics['status']}")

    print("=" * 60)
    if report["drift_detected"]:
        print("⚠️ ACTION REQUIRED: Significant data drift detected. Trigger model retraining.")
    else:
        print("✅ STABLE: No severe drift detected. Models are operating within statistical tolerances.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
