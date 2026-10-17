"""Prediction and Inference Service module for Customer Analytics Platform.

Exposes unified prediction interfaces for Churn Risk, Segmentation Personas,
Personalized Recommendations, and Demand Forecasting.
"""

from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger("models.predict")


def predict_churn(customer_features: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """Score churn risk probability and risk tier for customer record(s).

    Args:
        customer_features: Feature dict or single-row DataFrame.

    Returns:
        Dict[str, Any]: Churn probability, risk tier, and flag.
    """
    model, scaler, meta = ModelRegistry.load_churn_model()
    expected_cols = meta["feature_cols"]

    if isinstance(customer_features, dict):
        df = pd.DataFrame([customer_features])
    else:
        df = customer_features.copy()

    # Align columns
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[expected_cols].values
    if meta.get("best_model_name") == "Logistic_Regression" and scaler is not None:
        X = scaler.transform(X)

    prob = float(model.predict_proba(X)[0, 1])

    # Assign risk tier
    if prob >= 0.65:
        risk_tier = "High Risk"
    elif prob >= 0.35:
        risk_tier = "Medium Risk"
    else:
        risk_tier = "Low Risk"

    return {
        "churn_probability": round(prob, 4),
        "risk_tier": risk_tier,
        "is_high_risk": prob >= 0.65,
        "model_used": meta.get("best_model_name", "Ensemble"),
        "calibration": "Calibrated (Platt Sigmoid)"
    }


def predict_segment(customer_rfm: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """Assign cluster ID and behavioral persona for customer record."""
    model, scaler, meta = ModelRegistry.load_segmentation_model()
    features = meta["features"]

    if isinstance(customer_rfm, dict):
        df = pd.DataFrame([customer_rfm])
    else:
        df = customer_rfm.copy()

    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    X_scaled = scaler.transform(df[features].values)
    cluster_id = int(model.predict(X_scaled)[0])
    persona_mapping = {int(k): v for k, v in meta["persona_mapping"].items()}
    persona = persona_mapping.get(cluster_id, "Standard Shopper")

    return {
        "cluster_id": cluster_id,
        "persona_name": persona,
        "features_used": features
    }


def get_recommendations(
    customer_id: str,
    n_items: int = 5,
    is_churn_risk: bool = False
) -> List[Dict[str, Any]]:
    """Retrieve top-k personalized recommendations for customer."""
    recommender, meta = ModelRegistry.load_recommender()
    return recommender.recommend(str(customer_id), n_items=n_items, is_churn_risk=is_churn_risk)


def predict_forecast(horizon_days: int = 30) -> Dict[str, Any]:
    """Retrieve forward demand and revenue projections."""
    rev_model, ord_model, meta = ModelRegistry.load_forecast_models()
    return {
        "horizon_days": horizon_days,
        "impact_scenario": meta.get("impact_scenario", {}),
        "revenue_metrics": meta.get("revenue_metrics", {}),
        "orders_metrics": meta.get("orders_metrics", {})
    }
