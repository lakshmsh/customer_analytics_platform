"""Data validation module for Customer Analytics Platform.

Validates schemas, missingness, value boundaries, and integrity constraints.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("data.validation")


def validate_telco_churn(df: pd.DataFrame, raise_on_error: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """Validate cleaned Telco dataset integrity.

    Args:
        df: Cleaned Telco dataframe.
        raise_on_error: Whether to raise ValueError on validation failure.

    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, report_dict)
    """
    logger.info("Validating Telco dataset integrity...")
    report = {}
    is_valid = True

    # 1. Non-empty check
    report["row_count"] = len(df)
    if len(df) == 0:
        report["empty_check"] = "FAILED: Dataset is empty"
        is_valid = False
    else:
        report["empty_check"] = "PASSED"

    # 2. Key uniqueness
    if "customer_id" in df.columns:
        dups = df["customer_id"].duplicated().sum()
        report["customer_id_unique"] = "PASSED" if dups == 0 else f"FAILED: {dups} duplicates"
        if dups > 0:
            is_valid = False
    else:
        report["customer_id_unique"] = "FAILED: customer_id missing"
        is_valid = False

    # 3. Missing values
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    report["total_missing_values"] = total_nulls
    if total_nulls > 0:
        report["missing_check"] = f"FAILED: {total_nulls} nulls found"
        is_valid = False
    else:
        report["missing_check"] = "PASSED"

    # 4. Range and boundary checks
    range_violations = []
    if (df["tenure"] < 0).any():
        range_violations.append("Negative tenure detected")
    if (df["monthly_charges"] < 0).any():
        range_violations.append("Negative monthly_charges detected")
    if (df["total_charges"] < 0).any():
        range_violations.append("Negative total_charges detected")

    if range_violations:
        report["bounds_check"] = f"FAILED: {'; '.join(range_violations)}"
        is_valid = False
    else:
        report["bounds_check"] = "PASSED"

    # 5. Target variable check
    if "churn_label" in df.columns:
        unique_targets = set(df["churn_label"].unique())
        if not unique_targets.issubset({0, 1}):
            report["target_check"] = f"FAILED: Invalid churn labels {unique_targets}"
            is_valid = False
        else:
            churn_rate = df["churn_label"].mean()
            report["target_check"] = f"PASSED (churn rate = {churn_rate * 100:.2f}%)"
    else:
        report["target_check"] = "FAILED: churn_label column missing"
        is_valid = False

    logger.info(f"Telco validation status: {'PASSED' if is_valid else 'FAILED'}")
    if not is_valid and raise_on_error:
        raise ValueError(f"Telco validation failed: {report}")

    return is_valid, report


def validate_online_retail(df: pd.DataFrame, raise_on_error: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """Validate cleaned Online Retail dataset integrity.

    Args:
        df: Cleaned Online Retail dataframe.
        raise_on_error: Whether to raise ValueError on validation failure.

    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, report_dict)
    """
    logger.info("Validating Online Retail dataset integrity...")
    report = {}
    is_valid = True

    # 1. Non-empty check
    report["row_count"] = len(df)
    if len(df) == 0:
        report["empty_check"] = "FAILED: Dataset is empty"
        is_valid = False
    else:
        report["empty_check"] = "PASSED"

    # 2. Required columns
    required_cols = {
        "invoice", "stock_code", "description", "quantity",
        "invoice_date", "unit_price", "customer_id", "country", "total_amount"
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        report["required_columns"] = f"FAILED: Missing {missing_cols}"
        is_valid = False
    else:
        report["required_columns"] = "PASSED"

    # 3. Missing customer_id or invoice_date
    null_cust = df["customer_id"].isnull().sum()
    null_date = df["invoice_date"].isnull().sum()
    if null_cust > 0 or null_date > 0:
        report["null_keys"] = f"FAILED: {null_cust} null customers, {null_date} null dates"
        is_valid = False
    else:
        report["null_keys"] = "PASSED"

    # 4. Positive quantity and price
    neg_quantity = (df["quantity"] <= 0).sum()
    neg_price = (df["unit_price"] <= 0).sum()
    if neg_quantity > 0 or neg_price > 0:
        report["values_positive"] = f"FAILED: {neg_quantity} non-positive quantity, {neg_price} non-positive price"
        is_valid = False
    else:
        report["values_positive"] = "PASSED"

    # 5. Math consistency: total_amount ≈ quantity * unit_price
    diff = np.abs(df["total_amount"] - (df["quantity"] * df["unit_price"]))
    inconsistent = (diff > 0.05).sum()
    if inconsistent > 0:
        report["math_consistency"] = f"FAILED: {inconsistent} rows where total_amount != quantity * unit_price"
        is_valid = False
    else:
        report["math_consistency"] = "PASSED"

    logger.info(f"Online Retail validation status: {'PASSED' if is_valid else 'FAILED'}")
    if not is_valid and raise_on_error:
        raise ValueError(f"Online Retail validation failed: {report}")

    return is_valid, report
