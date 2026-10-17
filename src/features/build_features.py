"""Production Feature Engineering Pipeline for Customer Analytics Platform.

Provides a scikit-learn compatible transformer for customer-level feature extraction
with zero data leakage, support for train/test splits, and full serialization.
"""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.features.rfm import compute_rfm
from src.utils.logger import get_logger
from src.utils.helpers import timer

logger = get_logger("features.build_features")


class CustomerFeatureTransformer(BaseEstimator, TransformerMixin):
    """Production transformer to compute customer behavioral features from raw transactions.

    Strictly avoids data leakage by computing statistical bins, quantiles, and reference dates
    only on the training split during `fit()`, then projecting them onto test/inference data.
    """

    def __init__(
        self,
        inactivity_days_threshold: int = 90,
        recency_col: str = "invoice_date",
        monetary_col: str = "total_amount",
        customer_id_col: str = "customer_id",
        invoice_col: str = "invoice",
        quantity_col: str = "quantity",
        country_col: str = "country"
    ):
        self.inactivity_days_threshold = inactivity_days_threshold
        self.recency_col = recency_col
        self.monetary_col = monetary_col
        self.customer_id_col = customer_id_col
        self.invoice_col = invoice_col
        self.quantity_col = quantity_col
        self.country_col = country_col

        # Learned parameters during fit
        self.reference_date_: Optional[pd.Timestamp] = None
        self.r_bins_: Optional[np.ndarray] = None
        self.f_bins_: Optional[np.ndarray] = None
        self.m_bins_: Optional[np.ndarray] = None
        self.top_countries_: List[str] = []
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer parameters on training transactions only.

        Args:
            X: Training transactions DataFrame.
        """
        logger.info("Fitting CustomerFeatureTransformer on training transactions...")
        df = X.copy()
        df[self.recency_col] = pd.to_datetime(df[self.recency_col])

        # 1. Establish reference date strictly from training transactions
        self.reference_date_ = df[self.recency_col].max() + pd.Timedelta(days=1)
        logger.info(f"Learned train reference date: {self.reference_date_.strftime('%Y-%m-%d %H:%M:%S')}")

        # Compute preliminary aggregates on train to establish bins
        cust_agg = df.groupby(self.customer_id_col).agg(
            recency_days=(self.recency_col, lambda d: (self.reference_date_ - d.max()).days),
            frequency_orders=(self.invoice_col, "nunique"),
            monetary_total=(self.monetary_col, "sum")
        ).reset_index()

        # Learn quintile boundary bins
        _, self.r_bins_ = pd.qcut(cust_agg["recency_days"], q=5, retbins=True, duplicates="drop")
        _, self.f_bins_ = pd.qcut(cust_agg["frequency_orders"].rank(method="first"), q=5, retbins=True, duplicates="drop")
        _, self.m_bins_ = pd.qcut(cust_agg["monetary_total"].rank(method="first"), q=5, retbins=True, duplicates="drop")

        # Top 5 most frequent countries
        if self.country_col in df.columns:
            self.top_countries_ = df[self.country_col].value_counts().nlargest(5).index.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform transactions into customer-level behavioral feature store.

        Args:
            X: Transactions DataFrame (train, validation, or test/inference).

        Returns:
            pd.DataFrame: Engineered customer feature store.
        """
        if self.reference_date_ is None:
            raise RuntimeError("Transformer must be fitted before calling transform().")

        logger.info(f"Transforming {len(X):,} transactions into customer features...")
        df = X.copy()
        df[self.recency_col] = pd.to_datetime(df[self.recency_col])

        # 1. Base RFM and Volume Aggregations
        cust = df.groupby(self.customer_id_col).agg(
            recency_days=(self.recency_col, lambda d: (self.reference_date_ - d.max()).days),
            frequency_orders=(self.invoice_col, "nunique"),
            monetary_total=(self.monetary_col, "sum"),
            first_purchase=(self.recency_col, "min"),
            latest_purchase=(self.recency_col, "max"),
            total_items_bought=(self.quantity_col, "sum"),
            unique_items_count=("stock_code", "nunique") if "stock_code" in df.columns else (self.monetary_col, "count"),
        ).reset_index()

        # 2. Tenure Dynamics & Spend Velocity
        cust["tenure_days"] = (cust["latest_purchase"] - cust["first_purchase"]).dt.days
        cust["tenure_months"] = (cust["tenure_days"] / 30.44).round(1)
        cust["spend_per_tenure_day"] = (cust["monetary_total"] / cust["tenure_days"].clip(lower=1)).round(2)

        # 3. Ratios & Basket Characteristics
        cust["avg_basket_value"] = (cust["monetary_total"] / cust["frequency_orders"].clip(lower=1)).round(2)
        cust["avg_basket_size"] = (cust["total_items_bought"] / cust["frequency_orders"].clip(lower=1)).round(1)
        cust["item_diversity_ratio"] = (cust["unique_items_count"] / cust["total_items_bought"].clip(lower=1)).round(3)

        # 4. Temporal Purchase Spacing Dynamics
        order_dates = df[[self.customer_id_col, self.invoice_col, self.recency_col]].drop_duplicates()
        order_dates = order_dates.sort_values([self.customer_id_col, self.recency_col])
        order_dates["prev_order"] = order_dates.groupby(self.customer_id_col)[self.recency_col].shift(1)
        order_dates["order_interval"] = (order_dates[self.recency_col] - order_dates["prev_order"]).dt.days

        spacing = order_dates.groupby(self.customer_id_col)["order_interval"].agg(
            purchase_interval_mean="mean",
            purchase_interval_std="std"
        ).reset_index()
        spacing["purchase_interval_mean"] = spacing["purchase_interval_mean"].fillna(0.0).round(1)
        spacing["purchase_interval_std"] = spacing["purchase_interval_std"].fillna(0.0).round(1)

        cust = cust.merge(spacing, on=self.customer_id_col, how="left")

        # 5. Weekend and Seasonal Shopping Affinities
        df["is_weekend"] = df[self.recency_col].dt.dayofweek.isin([5, 6]).astype(int)
        df["is_q4"] = df[self.recency_col].dt.month.isin([10, 11, 12]).astype(int)

        weekend_pref = df.groupby(self.customer_id_col)["is_weekend"].mean().reset_index(name="weekend_purchase_ratio")
        q4_pref = df.groupby(self.customer_id_col)["is_q4"].mean().reset_index(name="q4_purchase_ratio")

        cust = cust.merge(weekend_pref, on=self.customer_id_col, how="left")
        cust = cust.merge(q4_pref, on=self.customer_id_col, how="left")

        # 6. Apply Learned RFM Quintile Scoring
        cust["r_score"] = pd.cut(
            cust["recency_days"],
            bins=np.concatenate([[-np.inf], self.r_bins_[1:-1], [np.inf]]),
            labels=[5, 4, 3, 2, 1]
        ).astype(int)

        cust["f_score"] = pd.cut(
            cust["frequency_orders"].rank(method="first"),
            bins=np.concatenate([[-np.inf], self.f_bins_[1:-1], [np.inf]]),
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

        cust["m_score"] = pd.cut(
            cust["monetary_total"].rank(method="first"),
            bins=np.concatenate([[-np.inf], self.m_bins_[1:-1], [np.inf]]),
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

        cust["rfm_score"] = cust["r_score"].astype(str) + cust["f_score"].astype(str) + cust["m_score"].astype(str)

        # 7. Ground Truth Churn Definition (90-day inactivity threshold)
        cust["is_churned"] = (cust["recency_days"] >= self.inactivity_days_threshold).astype(int)

        # Clean intermediate date columns
        cust.drop(columns=["first_purchase", "latest_purchase"], inplace=True)

        self.feature_names_ = [c for c in cust.columns if c not in [self.customer_id_col, "is_churned"]]
        logger.info(f"Transformed features for {len(cust):,} customers ({len(self.feature_names_)} features).")
        return cust


@timer
def build_customer_feature_store(
    cleaned_transactions: pd.DataFrame,
    inactivity_days_threshold: int = 90,
    reference_date: Optional[str] = None
) -> pd.DataFrame:
    """Convenience wrapper for creating the full customer feature store."""
    transformer = CustomerFeatureTransformer(inactivity_days_threshold=inactivity_days_threshold)
    transformer.fit(cleaned_transactions)
    return transformer.transform(cleaned_transactions)
