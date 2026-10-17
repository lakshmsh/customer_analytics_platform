"""Feature encoding module for categorical and demographic features."""

from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from src.utils.logger import get_logger

logger = get_logger("features.encoding")


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
    df_encoded = df.copy()

    # Identify numeric and categorical columns
    ignore_cols = {id_col, target_col, "churn"}
    feature_cols = [c for c in df_encoded.columns if c not in ignore_cols]

    numeric_cols = df_encoded[feature_cols].select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = df_encoded[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    logger.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Binary and multi-category One-Hot Encoding
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True, dtype=int)

    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "all_feature_cols": [c for c in df_encoded.columns if c not in ignore_cols],
        "id_col": id_col,
        "target_col": target_col
    }

    logger.info(f"Telco encoding complete. Final shape: {df_encoded.shape}")
    return df_encoded, metadata
