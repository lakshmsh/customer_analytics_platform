"""Demand and Revenue Impact Forecasting Engine.

Trains multi-step time-series regression models (LightGBM) to forecast:
1. Daily Revenue (£)
2. Daily Order Volumes (Demand)
3. Revenue Impact Scenarios (Baseline vs. Unmitigated Churn vs. Retention Recovered)
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
from lightgbm import LGBMRegressor

from src.evaluation.metrics import compute_forecast_metrics
from src.utils.logger import get_logger
from src.utils.helpers import timer

logger = get_logger("models.train_forecast")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR = Path("mlruns")
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


def build_forecasting_features(daily_df: pd.DataFrame, target_col: str = "daily_revenue") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Generate lag, rolling window, and calendar features for time-series forecasting."""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 1. Lags
    for lag in [1, 2, 7, 14, 21, 28]:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # 2. Rolling window statistics
    for w in [7, 14, 30]:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()
        df[f"{target_col}_roll_std_{w}"] = df[target_col].shift(1).rolling(w).std()

    # 3. Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Drop warm-up rows containing NaN from 28-day lags
    clean_df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in clean_df.columns if c not in ["date", "daily_revenue", "daily_orders", "daily_items_sold", "unique_buyers", "avg_order_value", "day_name"]]
    X = clean_df[feature_cols]
    y = clean_df[target_col]
    return X, y, feature_cols


@timer
def train_and_evaluate_forecasting(horizon_days: int = 30) -> Dict[str, Any]:
    """Train revenue and demand forecasting models and compute impact scenarios."""
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    sqlite_db = (MLRUNS_DIR / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{sqlite_db}")
    mlflow.set_experiment("Demand_and_Revenue_Forecasting")

    logger.info("Loading daily demand and revenue time series...")
    daily_ts = pd.read_parquet("data/processed/daily_demand_revenue.parquet")

    results = {}

    # ----------------------------------------------------
    # 1. Revenue Forecast Model
    # ----------------------------------------------------
    logger.info("Training Daily Revenue Forecaster...")
    X_rev, y_rev, rev_features = build_forecasting_features(daily_ts, target_col="daily_revenue")

    # Time-based Train / Test Split (Holdout last `horizon_days` as test)
    X_train_rev, X_test_rev = X_rev.iloc[:-horizon_days], X_rev.iloc[-horizon_days:]
    y_train_rev, y_test_rev = y_rev.iloc[:-horizon_days], y_rev.iloc[-horizon_days:]

    with mlflow.start_run(run_name="Forecast_Daily_Revenue_LGBM"):
        model_rev = LGBMRegressor(
            n_estimators=150, learning_rate=0.04, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
        model_rev.fit(X_train_rev, y_train_rev)
        y_pred_rev = np.clip(model_rev.predict(X_test_rev), 0, None)
        metrics_rev = compute_forecast_metrics(y_test_rev, y_pred_rev)

        mlflow.log_params({"target": "daily_revenue", "horizon_days": horizon_days})
        mlflow.log_metrics({f"revenue_{k}": v for k, v in metrics_rev.items()})
        results["revenue"] = metrics_rev

        logger.info(f"Revenue Forecast Metrics -> MAE: £{metrics_rev['mae']:,.2f}, RMSE: £{metrics_rev['rmse']:,.2f}, MAPE: {metrics_rev['mape']*100:.2f}%")

    # ----------------------------------------------------
    # 2. Demand (Order Volume) Forecast Model
    # ----------------------------------------------------
    logger.info("Training Daily Order Volume Forecaster...")
    X_ord, y_ord, ord_features = build_forecasting_features(daily_ts, target_col="daily_orders")
    X_train_ord, X_test_ord = X_ord.iloc[:-horizon_days], X_ord.iloc[-horizon_days:]
    y_train_ord, y_test_ord = y_ord.iloc[:-horizon_days], y_ord.iloc[-horizon_days:]

    with mlflow.start_run(run_name="Forecast_Daily_Orders_LGBM"):
        model_ord = LGBMRegressor(
            n_estimators=150, learning_rate=0.04, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
        model_ord.fit(X_train_ord, y_train_ord)
        y_pred_ord = np.clip(model_ord.predict(X_test_ord), 0, None)
        metrics_ord = compute_forecast_metrics(y_test_ord, y_pred_ord)

        mlflow.log_params({"target": "daily_orders", "horizon_days": horizon_days})
        mlflow.log_metrics({f"orders_{k}": v for k, v in metrics_ord.items()})
        results["orders"] = metrics_ord

        logger.info(f"Orders Forecast Metrics -> MAE: {metrics_ord['mae']:.1f} orders, RMSE: {metrics_ord['rmse']:.1f}, MAPE: {metrics_ord['mape']*100:.2f}%")

    # ----------------------------------------------------
    # 3. Revenue Impact & Churn Simulation
    # ----------------------------------------------------
    projected_baseline_revenue = float(np.sum(y_pred_rev))
    # Unmitigated churn scenario: estimated ~12.5% revenue erosion over 30 days
    unmitigated_erosion_rate = 0.125
    revenue_at_risk = round(projected_baseline_revenue * unmitigated_erosion_rate, 2)
    # Retention program recovery: recovers ~35% of revenue at risk
    recovery_rate = 0.35
    estimated_revenue_saved = round(revenue_at_risk * recovery_rate, 2)
    net_benefit = round(estimated_revenue_saved - 2500.0, 2)  # assuming £2.5k campaign cost

    impact_scenario = {
        "horizon_days": horizon_days,
        "projected_baseline_revenue": round(projected_baseline_revenue, 2),
        "unmitigated_churn_revenue_at_risk": revenue_at_risk,
        "estimated_revenue_saved": estimated_revenue_saved,
        "estimated_campaign_cost": 2500.0,
        "estimated_net_roi": net_benefit
    }

    logger.info("Revenue Impact Simulation (30-Day Forward):")
    logger.info(f"  Projected Baseline Revenue: £{projected_baseline_revenue:,.2f}")
    logger.info(f"  Unmitigated Churn Revenue at Risk: £{revenue_at_risk:,.2f}")
    logger.info(f"  Estimated Revenue Saved with Retention Actions: £{estimated_revenue_saved:,.2f}")
    logger.info(f"  Net ROI: £{net_benefit:,.2f}")

    # ----------------------------------------------------
    # 4. Persist Forecasting Artifacts
    # ----------------------------------------------------
    joblib.dump(model_rev, MODELS_DIR / "forecast_revenue_model.joblib")
    joblib.dump(model_ord, MODELS_DIR / "forecast_orders_model.joblib")

    metadata = {
        "horizon_days": horizon_days,
        "revenue_metrics": metrics_rev,
        "orders_metrics": metrics_ord,
        "revenue_feature_cols": rev_features,
        "orders_feature_cols": ord_features,
        "impact_scenario": impact_scenario
    }
    with open(MODELS_DIR / "forecast_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Persisted forecasting models and metadata to models/")
    return metadata


if __name__ == "__main__":
    train_and_evaluate_forecasting(horizon_days=30)
