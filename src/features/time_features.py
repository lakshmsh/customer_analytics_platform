"""Time-series aggregation and calendar feature extraction module."""

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("features.time_features")


def aggregate_daily_demand_revenue(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactional line items into a daily time series.

    Computes:
    - daily_revenue
    - daily_orders (distinct invoices)
    - daily_items_sold (sum of quantity)
    - unique_buyers (distinct customer_ids)
    - avg_order_value (revenue / orders)
    - calendar features: day_of_week, is_weekend, month, year, day_of_month

    Args:
        transactions_df: Cleaned transactions dataframe.

    Returns:
        pd.DataFrame: Daily aggregated time-series dataframe sorted chronologically.
    """
    logger.info("Aggregating daily demand and revenue time series...")
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["invoice_date"]).dt.date

    daily = df.groupby("date").agg(
        daily_revenue=("total_amount", "sum"),
        daily_orders=("invoice", "nunique"),
        daily_items_sold=("quantity", "sum"),
        unique_buyers=("customer_id", "nunique"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Fill potential missing calendar dates with 0s to maintain continuous time series
    full_range = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_range).fillna({
        "daily_revenue": 0.0,
        "daily_orders": 0,
        "daily_items_sold": 0,
        "unique_buyers": 0
    }).rename_axis("date").reset_index()

    # Derived daily metrics
    daily["avg_order_value"] = (
        daily["daily_revenue"] / daily["daily_orders"].replace(0, np.nan)
    ).fillna(0.0).round(2)

    # Calendar features
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["day_name"] = daily["date"].dt.day_name()
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["day_of_month"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["quarter"] = daily["date"].dt.quarter

    logger.info(f"Daily aggregation complete: {len(daily):,} days from {daily['date'].min().date()} to {daily['date'].max().date()}.")
    return daily
