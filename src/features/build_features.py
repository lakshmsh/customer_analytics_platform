"""Build comprehensive customer feature store from cleaned transactional data."""

from typing import Optional
import pandas as pd
import numpy as np
from src.features.rfm import compute_rfm
from src.utils.logger import get_logger
from src.utils.helpers import timer

logger = get_logger("features.build_features")


@timer
def build_customer_feature_store(
    cleaned_transactions: pd.DataFrame,
    inactivity_days_threshold: int = 90,
    reference_date: Optional[str] = None
) -> pd.DataFrame:
    """Build a unified, customer-level feature store for churn modeling and segmentation.

    Args:
        cleaned_transactions: Validated cleaned transactions.
        inactivity_days_threshold: Inactivity days defining churn (default: 90 days).
        reference_date: Cutoff date. If None, max date + 1 day.

    Returns:
        pd.DataFrame: Customer features dataframe.
    """
    logger.info("Building customer feature store...")
    df = cleaned_transactions.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    # 1. Base RFM and basket statistics
    rfm_df = compute_rfm(df, reference_date=reference_date)

    # 2. Additional behavioral features: primary country
    country_df = df.groupby("customer_id")["country"].agg(
        lambda s: s.mode().iloc[0] if not s.empty else "Unknown"
    ).reset_index()

    # 3. Order spacing dynamics (interval between orders)
    logger.info("Computing customer purchase interval dynamics...")
    order_dates = df[["customer_id", "invoice", "invoice_date"]].drop_duplicates()
    order_dates = order_dates.sort_values(["customer_id", "invoice_date"])
    order_dates["prev_order_date"] = order_dates.groupby("customer_id")["invoice_date"].shift(1)
    order_dates["interval_days"] = (order_dates["invoice_date"] - order_dates["prev_order_date"]).dt.days

    interval_stats = order_dates.groupby("customer_id")["interval_days"].agg(
        purchase_interval_mean="mean",
        purchase_interval_std="std"
    ).reset_index()
    interval_stats["purchase_interval_mean"] = interval_stats["purchase_interval_mean"].fillna(0.0).round(1)
    interval_stats["purchase_interval_std"] = interval_stats["purchase_interval_std"].fillna(0.0).round(1)

    # 4. Merge all customer features
    features = rfm_df.merge(country_df, on="customer_id", how="left")
    features = features.merge(interval_stats, on="customer_id", how="left")

    # 5. Define Churn Ground Truth
    # For non-contractual e-commerce, churn is defined as no purchases in the last N days (default 90 days)
    features["is_churned"] = (features["recency_days"] >= inactivity_days_threshold).astype(int)
    churn_rate = features["is_churned"].mean()
    logger.info(
        f"Churn definition applied (>= {inactivity_days_threshold} days inactivity): "
        f"{features['is_churned'].sum():,} / {len(features):,} customers churned ({churn_rate * 100:.2f}%)."
    )

    # 6. Customer Lifetime Value (CLV proxy) & Value Tiers
    # Tier based on monetary percentiles
    p50 = features["monetary_total"].quantile(0.50)
    p80 = features["monetary_total"].quantile(0.80)
    p95 = features["monetary_total"].quantile(0.95)

    def assign_tier(spend: float) -> str:
        if spend >= p95:
            return "VIP"
        elif spend >= p80:
            return "High Value"
        elif spend >= p50:
            return "Medium Value"
        return "Low Value"

    features["value_tier"] = features["monetary_total"].apply(assign_tier)

    logger.info(f"Customer feature store ready: {features.shape[0]:,} rows, {features.shape[1]} features.")
    return features
