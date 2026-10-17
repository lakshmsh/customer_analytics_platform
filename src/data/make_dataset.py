"""Main ETL Pipeline Orchestrator for Customer Analytics Platform.

Transforms raw datasets into validated interim and processed Parquet feature stores.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.data.ingestion import load_raw_telco, load_raw_online_retail
from src.data.cleaning import clean_telco_churn, clean_online_retail
from src.data.validation import validate_telco_churn, validate_online_retail
from src.features.build_features import build_customer_feature_store
from src.features.time_features import aggregate_daily_demand_revenue
from src.features.encoding import encode_telco_features
from src.utils.logger import get_logger
from src.utils.helpers import timer

logger = get_logger("data.make_dataset")

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


@timer
def run_etl_pipeline() -> Dict[str, Any]:
    """Execute the end-to-end ETL and feature generation pipeline.

    Returns:
        Dict[str, Any]: Summary metrics and artifact metadata.
    """
    logger.info("==================================================")
    logger.info("   STARTING CUSTOMER ANALYTICS ETL PIPELINE       ")
    logger.info("==================================================")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    # ----------------------------------------------------
    # 1. Telco Customer Churn Pipeline
    # ----------------------------------------------------
    logger.info("--- [STEP 1/4] Processing Telco Dataset ---")
    raw_telco = load_raw_telco()
    summary["telco_raw_rows"] = len(raw_telco)

    cleaned_telco = clean_telco_churn(raw_telco)
    summary["telco_cleaned_rows"] = len(cleaned_telco)

    # Validate Telco
    is_valid_telco, telco_val_report = validate_telco_churn(cleaned_telco)
    summary["telco_validation"] = telco_val_report

    # Save Telco Interim
    telco_interim_path = INTERIM_DIR / "telco_cleaned.parquet"
    cleaned_telco.to_parquet(telco_interim_path, index=False)
    logger.info(f"Saved interim Telco dataset: {telco_interim_path}")

    # Encode & Save Telco Processed
    telco_encoded, telco_meta = encode_telco_features(cleaned_telco)
    telco_proc_path = PROCESSED_DIR / "telco_processed.parquet"
    telco_encoded.to_parquet(telco_proc_path, index=False)
    logger.info(f"Saved processed Telco dataset: {telco_proc_path} ({telco_encoded.shape})")

    # ----------------------------------------------------
    # 2. Online Retail II Transaction Pipeline
    # ----------------------------------------------------
    logger.info("--- [STEP 2/4] Processing Online Retail Dataset ---")
    raw_retail = load_raw_online_retail()
    summary["retail_raw_rows"] = len(raw_retail)

    cleaned_retail = clean_online_retail(raw_retail, drop_missing_customers=True, remove_cancellations=True)
    summary["retail_cleaned_rows"] = len(cleaned_retail)

    # Validate Online Retail
    is_valid_retail, retail_val_report = validate_online_retail(cleaned_retail)
    summary["retail_validation"] = retail_val_report

    # Save Retail Interim
    retail_interim_path = INTERIM_DIR / "retail_cleaned.parquet"
    cleaned_retail.to_parquet(retail_interim_path, index=False)
    logger.info(f"Saved interim Retail dataset: {retail_interim_path}")

    # Save Retail Processed Transactions
    retail_proc_path = PROCESSED_DIR / "retail_cleaned_transactions.parquet"
    cleaned_retail.to_parquet(retail_proc_path, index=False)
    logger.info(f"Saved processed Retail transactions: {retail_proc_path} ({cleaned_retail.shape})")

    # ----------------------------------------------------
    # 3. Customer Feature Store (RFM, Churn Target, CLV)
    # ----------------------------------------------------
    logger.info("--- [STEP 3/4] Building Customer Feature Store ---")
    customer_features = build_customer_feature_store(cleaned_retail, inactivity_days_threshold=90)
    summary["unique_customers"] = len(customer_features)
    summary["customer_features_churn_rate"] = float(customer_features["is_churned"].mean())

    customer_features_path = PROCESSED_DIR / "customers_features.parquet"
    customer_features.to_parquet(customer_features_path, index=False)
    logger.info(f"Saved Customer Feature Store: {customer_features_path} ({customer_features.shape})")

    # ----------------------------------------------------
    # 4. Daily Demand & Revenue Time Series
    # ----------------------------------------------------
    logger.info("--- [STEP 4/4] Aggregating Daily Time Series ---")
    daily_ts = aggregate_daily_demand_revenue(cleaned_retail)
    summary["time_series_days"] = len(daily_ts)
    summary["total_revenue"] = float(daily_ts["daily_revenue"].sum())

    daily_ts_path = PROCESSED_DIR / "daily_demand_revenue.parquet"
    daily_ts.to_parquet(daily_ts_path, index=False)
    logger.info(f"Saved Daily Demand & Revenue Series: {daily_ts_path} ({daily_ts.shape})")

    logger.info("==================================================")
    logger.info("   ETL PIPELINE COMPLETED SUCCESSFULLY!           ")
    logger.info("==================================================")
    return summary


if __name__ == "__main__":
    run_etl_pipeline()
