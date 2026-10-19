"""
Pytest configuration and shared fixtures for Customer Analytics Platform tests.

All fixtures here mock out model loading and data I/O so that the full test
suite can run in CI without requiring pre-trained .joblib artifacts or
parquet datasets on disk.
"""

import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Mock model artefacts
# ---------------------------------------------------------------------------

def _make_mock_churn_model():
    """Return a mock CalibratedClassifierCV that always predicts 0.72 churn."""
    mock_estimator = MagicMock()
    mock_estimator.coef_ = np.zeros((1, 20))
    mock_estimator.intercept_ = np.array([0.0])

    mock_calibrated = MagicMock()
    mock_calibrated.estimator = mock_estimator

    model = MagicMock()
    model.calibrated_classifiers_ = [mock_calibrated]
    model.predict_proba.return_value = np.array([[0.28, 0.72]])
    return model


def _make_mock_scaler():
    """Return a StandardScaler mock that passes data through unchanged."""
    scaler = MagicMock()
    scaler.transform.side_effect = lambda X: np.array(X, dtype=float)
    return scaler


def _make_churn_metadata(n_features: int = 20) -> dict:
    """Build minimal churn metadata dict."""
    feature_cols = [
        "tenure", "monthly_charges", "total_charges", "charges_tenure_ratio",
        "senior_citizen", "has_family", "security_bundle_score",
        "streaming_bundle_score",
        "contract_Month-to-month", "contract_One year", "contract_Two year",
        "internet_service_Fiber optic", "internet_service_No",
        "payment_method_Electronic check", "payment_method_Credit card (automatic)",
        "payment_method_Mailed check", "paperless_billing_Yes",
        "partner_No", "dependents_No", "phone_service_Yes",
    ][:n_features]
    return {
        "feature_cols": feature_cols,
        "best_model_name": "LightGBM",
        "test_roc_auc": 0.851,
    }


def _make_mock_segmentation_model():
    model = MagicMock()
    model.predict.return_value = np.array([2])
    return model


def _make_segmentation_metadata() -> dict:
    return {
        "features": [
            "recency_days", "frequency_orders", "monetary_total",
            "tenure_days", "avg_basket_value", "avg_basket_size",
        ],
        "persona_mapping": {
            "0": "Budget Shopper",
            "1": "Occasional Buyer",
            "2": "High-Value Champion",
            "3": "At-Risk Customer",
        },
    }


def _make_mock_recommender():
    """Return a recommender that always produces deterministic dummy items."""
    def _recommend(customer_id, n_items=5, is_churn_risk=False):
        items = []
        for i in range(n_items):
            items.append({
                "stock_code": f"PROD-{i+1:03d}",
                "description": f"Sample Product {i+1}",
                "base_price": round(9.99 + i * 5, 2),
                "relevance_score": round(0.95 - i * 0.05, 2),
                "strategy": "collaborative" if i % 2 == 0 else "content",
                "retention_incentive": "15% OFF – Exclusive Loyalty Offer" if is_churn_risk else None,
                "discounted_price": round((9.99 + i * 5) * 0.85, 2) if is_churn_risk else None,
            })
        return items

    recommender = MagicMock()
    recommender.recommend.side_effect = _recommend
    return recommender


def _make_recommender_metadata() -> dict:
    return {"model_type": "HybridRecommender", "n_items_in_catalogue": 3896}


def _make_forecast_metadata() -> dict:
    return {
        "impact_scenario": {
            "projected_baseline_revenue": 185_000.0,
            "unmitigated_churn_revenue_at_risk": 27_750.0,
            "estimated_revenue_saved": 9_712.5,
            "estimated_campaign_cost": 1_875.0,
            "estimated_net_roi": 7_837.5,
        },
        "revenue_metrics": {"mae": 1240.5, "rmse": 1820.3, "r2": 0.87},
        "orders_metrics": {"mae": 18.4, "rmse": 26.1, "r2": 0.84},
    }


def _make_mock_forecast_model():
    model = MagicMock()
    model.predict.return_value = np.array([120_000.0])
    return model


# ---------------------------------------------------------------------------
# SHAP mock
# ---------------------------------------------------------------------------

def _make_mock_shap_explainer(model, background):
    """Return a SHAP LinearExplainer mock."""
    n_features = background.shape[1] if hasattr(background, "shape") else 20

    shap_explanation = MagicMock()
    shap_explanation.values = np.random.uniform(-0.1, 0.3, size=(1, n_features))
    shap_explanation.base_values = np.array([0.15])

    explainer = MagicMock()
    explainer.return_value = shap_explanation
    explainer.__call__ = lambda self, X: shap_explanation
    return explainer


# ---------------------------------------------------------------------------
# Background parquet mock
# ---------------------------------------------------------------------------

def _make_background_df(feature_cols):
    import pandas as pd
    data = {col: np.random.rand(200) for col in feature_cols}
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Master fixture: patch every external dependency before the app is imported
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_registry():
    """Session-scoped mock for ModelRegistry — patches all four loaders."""
    churn_model = _make_mock_churn_model()
    churn_scaler = _make_mock_scaler()
    churn_meta = _make_churn_metadata()

    seg_model = _make_mock_segmentation_model()
    seg_scaler = _make_mock_scaler()
    seg_meta = _make_segmentation_metadata()

    rec_model = _make_mock_recommender()
    rec_meta = _make_recommender_metadata()

    rev_model = _make_mock_forecast_model()
    ord_model = _make_mock_forecast_model()
    fc_meta = _make_forecast_metadata()

    patches = [
        patch(
            "src.models.registry.ModelRegistry.load_churn_model",
            return_value=(churn_model, churn_scaler, churn_meta),
        ),
        patch(
            "src.models.registry.ModelRegistry.load_segmentation_model",
            return_value=(seg_model, seg_scaler, seg_meta),
        ),
        patch(
            "src.models.registry.ModelRegistry.load_recommender",
            return_value=(rec_model, rec_meta),
        ),
        patch(
            "src.models.registry.ModelRegistry.load_forecast_models",
            return_value=(rev_model, ord_model, fc_meta),
        ),
    ]

    started = [p.start() for p in patches]
    yield started
    for p in patches:
        p.stop()


@pytest.fixture(scope="session")
def mock_parquet(mock_registry):
    """Patch pandas.read_parquet so ChurnExplainer never hits disk."""
    import pandas as pd

    churn_meta = _make_churn_metadata()
    bg_df = _make_background_df(churn_meta["feature_cols"])

    with patch("pandas.read_parquet", return_value=bg_df):
        yield


@pytest.fixture(scope="session")
def mock_shap(mock_parquet):
    """Patch shap.LinearExplainer so SHAP never runs a real computation."""
    n_features = len(_make_churn_metadata()["feature_cols"])

    shap_explanation = MagicMock()
    shap_explanation.values = np.random.uniform(-0.1, 0.3, size=(1, n_features))
    shap_explanation.base_values = np.array([0.15])

    mock_explainer_instance = MagicMock()
    mock_explainer_instance.return_value = shap_explanation
    mock_explainer_instance.__call__ = MagicMock(return_value=shap_explanation)

    with patch("shap.LinearExplainer", return_value=mock_explainer_instance):
        yield mock_explainer_instance, shap_explanation


@pytest.fixture(scope="session")
def test_client(mock_shap):
    """Build FastAPI TestClient with all dependencies mocked."""
    from src.api.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
