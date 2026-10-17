"""Data cleaning module for Customer Analytics Platform.

Applies robust cleaning transformations to both Telco and Online Retail datasets.
Logs all row counts, null imputations, and duplicate removals.
"""

import re
from typing import Tuple
import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.helpers import timer

logger = get_logger("data.cleaning")


def _to_snake_case(name: str) -> str:
    """Convert PascalCase or mixed string into snake_case."""
    name = re.sub(r"[\s\-]+", "_", name.strip())
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


@timer
def clean_telco_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco Customer Churn dataset.

    Steps:
    1. Standardize column names to snake_case.
    2. Convert TotalCharges from string with blank spaces to float, imputing 0.0 for tenure=0.
    3. Strip whitespace from string values.
    4. Create binary target column `churn_label` (1=Yes, 0=No).
    5. Deduplicate rows if any.

    Args:
        df: Raw Telco dataframe.

    Returns:
        pd.DataFrame: Cleaned Telco dataframe.
    """
    initial_rows = len(df)
    logger.info(f"Cleaning Telco dataset. Initial row count: {initial_rows:,}")

    df_clean = df.copy()

    # 1. Standardize column names
    df_clean.columns = [_to_snake_case(c) for c in df_clean.columns]
    if "customer_i_d" in df_clean.columns:
        df_clean.rename(columns={"customer_i_d": "customer_id"}, inplace=True)
    elif "customerid" in df_clean.columns:
        df_clean.rename(columns={"customerid": "customer_id"}, inplace=True)

    # 2. Clean TotalCharges (handle blank spaces)
    blank_total_charges = (df_clean["total_charges"].astype(str).str.strip() == "").sum()
    if blank_total_charges > 0:
        logger.info(f"Detected {blank_total_charges} rows with blank 'total_charges'. Imputing with 0.0 (tenure=0).")
        df_clean["total_charges"] = pd.to_numeric(
            df_clean["total_charges"].astype(str).str.strip().replace("", "0.0"),
            errors="coerce"
        ).fillna(0.0)
    else:
        df_clean["total_charges"] = pd.to_numeric(df_clean["total_charges"], errors="coerce").fillna(0.0)

    # 3. Strip whitespace in string columns
    str_cols = df_clean.select_dtypes(include=["object"]).columns
    for c in str_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip()

    # 4. Standardize binary target
    if "churn" in df_clean.columns:
        df_clean["churn_label"] = (df_clean["churn"].str.lower() == "yes").astype(int)

    # 5. Remove duplicates
    duplicates_count = df_clean.duplicated().sum()
    if duplicates_count > 0:
        logger.warning(f"Found {duplicates_count} duplicate rows in Telco dataset. Removing...")
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 6. Ensure non-negative charges and tenure
    df_clean["tenure"] = df_clean["tenure"].astype(int).clip(lower=0)
    df_clean["monthly_charges"] = df_clean["monthly_charges"].astype(float).clip(lower=0.0)
    df_clean["total_charges"] = df_clean["total_charges"].astype(float).clip(lower=0.0)

    final_rows = len(df_clean)
    logger.info(
        f"Telco cleaning complete: {initial_rows:,} -> {final_rows:,} rows. "
        f"Churn rate: {df_clean['churn_label'].mean() * 100:.2f}%."
    )
    return df_clean


@timer
def clean_online_retail(
    df: pd.DataFrame,
    drop_missing_customers: bool = True,
    remove_cancellations: bool = True
) -> pd.DataFrame:
    """Clean the raw Online Retail II transactional dataset.

    Steps:
    1. Standardize column names (invoice, stock_code, description, quantity, invoice_date, unit_price, customer_id, country).
    2. Format customer_id as clean string identifier (e.g. '13085').
    3. Filter out rows with missing customer_id if drop_missing_customers is True.
    4. Flag or filter cancellations (invoices starting with 'C' or quantity < 0).
    5. Clean unit_price (drop items where unit_price <= 0).
    6. Filter obvious system anomalies/test records.
    7. Compute total_amount = quantity * unit_price.
    8. Deduplicate identical transactions.

    Args:
        df: Raw Online Retail II dataframe.
        drop_missing_customers: Whether to drop rows without customer ID.
        remove_cancellations: Whether to remove cancellation transactions from purchase history.

    Returns:
        pd.DataFrame: Cleaned retail transactions dataframe.
    """
    initial_rows = len(df)
    logger.info(f"Cleaning Online Retail dataset. Initial row count: {initial_rows:,}")

    df_clean = df.copy()

    # 1. Standardize column names
    col_mapping = {
        "Invoice": "invoice",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "Price": "unit_price",
        "Customer ID": "customer_id",
        "Country": "country",
    }
    df_clean.rename(columns=col_mapping, inplace=True)

    # 2. Filter missing customer_id
    null_customers = df_clean["customer_id"].isnull().sum()
    logger.info(f"Rows with null customer_id: {null_customers:,} ({100 * null_customers / initial_rows:.2f}%)")
    if drop_missing_customers:
        df_clean = df_clean.dropna(subset=["customer_id"]).reset_index(drop=True)
        logger.info(f"Dropped rows with missing customer_id. Remaining: {len(df_clean):,}")

    # Format customer_id as clean integer string
    df_clean["customer_id"] = df_clean["customer_id"].astype(int).astype(str)

    # 3. Clean invoice and cancellation tracking
    df_clean["invoice"] = df_clean["invoice"].astype(str).str.strip()
    df_clean["is_cancellation"] = (
        df_clean["invoice"].str.startswith("C") | (df_clean["quantity"] < 0)
    ).astype(int)

    cancellation_count = df_clean["is_cancellation"].sum()
    logger.info(f"Identified {cancellation_count:,} cancellation records.")

    if remove_cancellations:
        df_clean = df_clean[df_clean["is_cancellation"] == 0].reset_index(drop=True)
        logger.info(f"Filtered cancellations. Remaining purchase transactions: {len(df_clean):,}")

    # 4. Clean quantity and unit_price
    df_clean["quantity"] = pd.to_numeric(df_clean["quantity"], errors="coerce").fillna(0).astype(int)
    df_clean["unit_price"] = pd.to_numeric(df_clean["unit_price"], errors="coerce").fillna(0.0).astype(float)

    # Filter out non-positive quantities or prices
    valid_mask = (df_clean["quantity"] > 0) & (df_clean["unit_price"] > 0)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.info(f"Filtering {invalid_count:,} records with non-positive quantity or price.")
        df_clean = df_clean[valid_mask].reset_index(drop=True)

    # Filter test / postage / administrative items
    df_clean["stock_code"] = df_clean["stock_code"].astype(str).str.strip()
    admin_codes = {"POST", "D", "M", "BANK CHARGES", "AMAZONFEE", "DOT", "CRUK", "TEST", "TEST001", "TEST002"}
    non_product_mask = df_clean["stock_code"].str.upper().isin(admin_codes)
    admin_count = non_product_mask.sum()
    if admin_count > 0:
        logger.info(f"Filtering {admin_count:,} administrative/fee lines ({admin_codes}).")
        df_clean = df_clean[~non_product_mask].reset_index(drop=True)

    # 5. Outlier filtering (extreme quantities or prices likely due to warehouse batch errors)
    outlier_mask = (df_clean["quantity"] > 20000) | (df_clean["unit_price"] > 8000)
    outlier_count = outlier_mask.sum()
    if outlier_count > 0:
        logger.warning(f"Removing {outlier_count} extreme outlier transactions.")
        df_clean = df_clean[~outlier_mask].reset_index(drop=True)

    # 6. Parse invoice_date
    df_clean["invoice_date"] = pd.to_datetime(df_clean["invoice_date"], errors="coerce")

    # 7. Clean descriptions
    df_clean["description"] = df_clean["description"].astype(str).str.strip().str.upper()

    # 8. Compute total_amount
    df_clean["total_amount"] = (df_clean["quantity"] * df_clean["unit_price"]).round(2)

    # 9. Deduplicate
    duplicates_count = df_clean.duplicated().sum()
    if duplicates_count > 0:
        logger.info(f"Removing {duplicates_count:,} duplicate line items.")
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    final_rows = len(df_clean)
    unique_customers = df_clean["customer_id"].nunique()
    logger.info(
        f"Online Retail cleaning complete: {initial_rows:,} -> {final_rows:,} rows. "
        f"Unique customers: {unique_customers:,}."
    )
    return df_clean
