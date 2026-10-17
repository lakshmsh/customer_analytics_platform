"""Feature encoding and transformation module for tabular customer models."""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.utils.logger import get_logger

logger = get_logger("features.encoding")


def add_telco_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific composite and ratio features to Telco dataset.

    Features added:
    - security_bundle_score: Count of active security/support add-ons.
    - streaming_bundle_score: Count of active streaming add-ons.
    - charges_tenure_ratio: total_charges / (tenure + 1)
    - has_family: 1 if partner == 'Yes' or dependents == 'Yes'

    Args:
        df: Cleaned Telco dataframe.

    Returns:
        pd.DataFrame: DataFrame with additional engineered features.
    """
    df_eng = df.copy()

    # Security bundle score (0 to 4)
    sec_cols = ["online_security", "online_backup", "device_protection", "tech_support"]
    sec_score = 0
    for col in sec_cols:
        if col in df_eng.columns:
            sec_score += (df_eng[col] == "Yes").astype(int)
    df_eng["security_bundle_score"] = sec_score

    # Streaming bundle score (0 to 2)
    stream_cols = ["streaming_tv", "streaming_movies"]
    stream_score = 0
    for col in stream_cols:
        if col in df_eng.columns:
            stream_score += (df_eng[col] == "Yes").astype(int)
    df_eng["streaming_bundle_score"] = stream_score

    # Charges vs tenure ratio
    df_eng["charges_tenure_ratio"] = (df_eng["total_charges"] / (df_eng["tenure"] + 1)).round(2)

    # Family presence
    if "partner" in df_eng.columns and "dependents" in df_eng.columns:
        df_eng["has_family"] = ((df_eng["partner"] == "Yes") | (df_eng["dependents"] == "Yes")).astype(int)

    return df_eng


def encode_telco_features(
    df: pd.DataFrame,
    target_col: str = "churn_label",
    id_col: str = "customer_id"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prepare and encode Telco features for machine learning models.

    Args:
        df: Cleaned Telco dataframe.
        target_col: Column name of churn target.
        id_col: Column name of unique identifier.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Encoded feature DataFrame, metadata dictionary)
    """
    logger.info("Encoding Telco categorical and numerical features...")
    df_eng = add_telco_engineered_features(df)

    ignore_cols = {id_col, target_col, "churn"}
    feature_cols = [c for c in df_eng.columns if c not in ignore_cols]

    numeric_cols = df_eng[feature_cols].select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = df_eng[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    logger.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # One-Hot Encoding for categorical features
    df_encoded = pd.get_dummies(df_eng, columns=categorical_cols, drop_first=True, dtype=int)

    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "all_feature_cols": [c for c in df_encoded.columns if c not in ignore_cols],
        "id_col": id_col,
        "target_col": target_col
    }

    logger.info(f"Telco encoding complete. Final shape: {df_encoded.shape}")
    return df_encoded, metadata
