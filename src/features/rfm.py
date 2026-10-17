"""RFM (Recency, Frequency, Monetary) computation module."""

from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("features.rfm")


def compute_rfm(
    transactions_df: pd.DataFrame,
    reference_date: Optional[datetime | str] = None,
    recency_col: str = "invoice_date",
    monetary_col: str = "total_amount",
    customer_id_col: str = "customer_id",
    invoice_col: str = "invoice"
) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary metrics per customer.

    Args:
        transactions_df: Cleaned transactions dataframe.
        reference_date: Reference timestamp for recency calculation. If None, max date + 1 day is used.
        recency_col: Column containing timestamp of transaction.
        monetary_col: Column containing total spend per line item.
        customer_id_col: Column identifying customer.
        invoice_col: Column identifying distinct invoice/order.

    Returns:
        pd.DataFrame: RFM dataframe with quintile scores and composite RFM segments.
    """
    logger.info("Computing RFM features...")
    df = transactions_df.copy()
    df[recency_col] = pd.to_datetime(df[recency_col])

    if reference_date is None:
        ref_dt = df[recency_col].max() + pd.Timedelta(days=1)
    else:
        ref_dt = pd.to_datetime(reference_date)

    logger.info(f"Using reference date: {ref_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # Group by customer
    rfm = df.groupby(customer_id_col).agg(
        recency_days=(recency_col, lambda dates: (ref_dt - dates.max()).days),
        frequency_orders=(invoice_col, "nunique"),
        monetary_total=(monetary_col, "sum"),
        first_purchase=(recency_col, "min"),
        latest_purchase=(recency_col, "max"),
        total_items_bought=("quantity", "sum") if "quantity" in df.columns else (monetary_col, "count"),
        unique_items_count=("stock_code", "nunique") if "stock_code" in df.columns else (monetary_col, "count")
    ).reset_index()

    # Tenure in days
    rfm["tenure_days"] = (rfm["latest_purchase"] - rfm["first_purchase"]).dt.days
    rfm["avg_basket_value"] = (rfm["monetary_total"] / rfm["frequency_orders"].clip(lower=1)).round(2)
    rfm["avg_basket_size"] = (rfm["total_items_bought"] / rfm["frequency_orders"].clip(lower=1)).round(1)

    # Compute Quintiles (1-5, where 5 is best)
    # Recency: lower is better (5 = most recent, 1 = oldest)
    # Frequency: higher is better (5 = most frequent, 1 = least)
    # Monetary: higher is better (5 = highest spend, 1 = lowest)
    try:
        rfm["r_score"] = pd.qcut(rfm["recency_days"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    except ValueError:
        rfm["r_score"] = pd.cut(rfm["recency_days"].rank(method="first"), bins=5, labels=[5, 4, 3, 2, 1]).astype(int)

    try:
        rfm["f_score"] = pd.qcut(rfm["frequency_orders"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    except ValueError:
        rfm["f_score"] = pd.cut(rfm["frequency_orders"].rank(method="first"), bins=5, labels=[1, 2, 3, 4, 5]).astype(int)

    try:
        rfm["m_score"] = pd.qcut(rfm["monetary_total"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    except ValueError:
        rfm["m_score"] = pd.cut(rfm["monetary_total"].rank(method="first"), bins=5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)

    # Drop intermediate date columns to keep table clean for modeling
    rfm.drop(columns=["first_purchase", "latest_purchase"], inplace=True)

    logger.info(f"RFM features generated for {len(rfm):,} unique customers.")
    return rfm
