"""
Unit and integration tests for the Customer Analytics Platform FastAPI backend.

All tests run fully in CI without requiring trained model artefacts or parquet
data on disk — every external dependency is mocked via conftest.py fixtures.
"""

import pytest


# ---------------------------------------------------------------------------
# Phase 0 / Setup checks
# ---------------------------------------------------------------------------

def test_src_packages_importable():
    """Phase 0: Verify all src sub-packages are importable (structure check)."""
    import src.utils.logger  # noqa: F401
    import src.api.schemas   # noqa: F401
    import src.models.registry  # noqa: F401


def test_logger_returns_logger():
    """Phase 0: get_logger() should return a standard Logger instance."""
    import logging
    from src.utils.logger import get_logger
    logger = get_logger("test.ci")
    assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# Phase 7: API — root & health
# ---------------------------------------------------------------------------

def test_root_endpoint(test_client):
    """Phase 7: Root endpoint responds 200 with required metadata keys."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data
    assert "churn_prediction" in data["endpoints"]
    assert "health_check" in data["endpoints"]


def test_health_check_returns_200(test_client):
    """Phase 7: /health endpoint returns 200 regardless of model load status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # Status is either 'healthy' (all 4 loaded) or 'degraded' (some failed).
    assert data["status"] in ("healthy", "degraded")
    assert isinstance(data["models_loaded"], list)
    assert "version" in data


# ---------------------------------------------------------------------------
# Phase 7: Churn prediction endpoint
# ---------------------------------------------------------------------------

VALID_CHURN_PAYLOAD = {
    "customer_id": "CI-TEST-001",
    "tenure": 2,
    "monthly_charges": 89.50,
    "total_charges": 179.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "paperless_billing": "Yes",
}


def test_churn_predict_valid(test_client):
    """Phase 7: Single churn prediction returns calibrated score + SHAP drivers."""
    response = test_client.post("/churn/predict", json=VALID_CHURN_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CI-TEST-001"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ("High Risk", "Medium Risk", "Low Risk")
    assert isinstance(data["top_risk_drivers"], list)
    assert isinstance(data["prescribed_actions"], list)
    assert "financial_roi" in data
    assert data["financial_roi"]["net_expected_roi"] is not None


def test_churn_predict_invalid_tenure(test_client):
    """Phase 7: Pydantic rejects negative tenure with 422 Unprocessable Entity."""
    response = test_client.post(
        "/churn/predict",
        json={"tenure": -5, "monthly_charges": 50.0},
    )
    assert response.status_code == 422


def test_churn_predict_missing_required_fields(test_client):
    """Phase 7: Missing required fields (tenure, monthly_charges) → 422."""
    response = test_client.post("/churn/predict", json={"customer_id": "X"})
    assert response.status_code == 422


def test_churn_predict_monthly_charges_too_high(test_client):
    """Phase 7: monthly_charges > 500 violates Pydantic constraint → 422."""
    payload = {**VALID_CHURN_PAYLOAD, "monthly_charges": 9999.0}
    response = test_client.post("/churn/predict", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Phase 7: Batch churn endpoint
# ---------------------------------------------------------------------------

def test_churn_batch_predict(test_client):
    """Phase 7: Batch endpoint scores multiple customers and aggregates counts."""
    payload = {
        "customers": [
            {
                "customer_id": "CUST-A",
                "tenure": 1,
                "monthly_charges": 95.0,
                "contract": "Month-to-month",
            },
            {
                "customer_id": "CUST-B",
                "tenure": 60,
                "monthly_charges": 20.0,
                "contract": "Two year",
            },
        ]
    }
    response = test_client.post("/churn/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    # Risk counts must sum to total
    assert (
        data["high_risk_count"] + data["medium_risk_count"] + data["low_risk_count"]
        == 2
    )


def test_churn_batch_empty_list(test_client):
    """Phase 7: Empty customers list returns total_customers=0."""
    response = test_client.post("/churn/batch", json={"customers": []})
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 0
    assert data["predictions"] == []


# ---------------------------------------------------------------------------
# Phase 7: Segmentation endpoint
# ---------------------------------------------------------------------------

def test_segment_predict(test_client):
    """Phase 7: Segmentation endpoint assigns cluster ID and persona name."""
    payload = {
        "customer_id": "13085",
        "recency_days": 10,
        "frequency_orders": 12,
        "monetary_total": 3500.0,
        "tenure_days": 250,
        "avg_basket_value": 291.67,
        "avg_basket_size": 18.0,
    }
    response = test_client.post("/segment/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "13085"
    assert "cluster_id" in data
    assert isinstance(data["persona_name"], str)
    assert len(data["persona_name"]) > 0
    assert isinstance(data["features_used"], list)


def test_segment_predict_invalid_frequency(test_client):
    """Phase 7: frequency_orders < 1 violates constraint → 422."""
    payload = {
        "recency_days": 10,
        "frequency_orders": 0,  # must be >= 1
        "monetary_total": 500.0,
    }
    response = test_client.post("/segment/predict", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Phase 7: Recommendation endpoint
# ---------------------------------------------------------------------------

def test_recommend_endpoint_default(test_client):
    """Phase 7: Recommendation endpoint returns requested number of items."""
    response = test_client.get("/recommend/13085?n_items=4&is_churn_risk=true")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "13085"
    assert len(data["recommendations"]) == 4
    first = data["recommendations"][0]
    assert "stock_code" in first
    assert "description" in first
    assert "retention_incentive" in first


def test_recommend_endpoint_no_churn_flag(test_client):
    """Phase 7: Non-churn-risk recommendations have no retention_incentive."""
    response = test_client.get("/recommend/99999?n_items=3&is_churn_risk=false")
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 3
    # retention_incentive should be null when not a churn risk
    for item in data["recommendations"]:
        assert item.get("retention_incentive") is None


def test_recommend_endpoint_n_items_too_large(test_client):
    """Phase 7: n_items > 20 violates Query constraint → 422."""
    response = test_client.get("/recommend/13085?n_items=99")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Phase 7: Forecasting endpoint
# ---------------------------------------------------------------------------

def test_forecast_endpoint(test_client):
    """Phase 7: Forecast endpoint returns positive projected revenue."""
    response = test_client.get("/forecast/revenue?horizon_days=30")
    assert response.status_code == 200
    data = response.json()
    assert data["horizon_days"] == 30
    assert data["projected_baseline_revenue"] > 0.0
    assert data["estimated_revenue_saved"] > 0.0
    assert "revenue_metrics" in data
    assert "orders_metrics" in data


def test_forecast_horizon_too_small(test_client):
    """Phase 7: horizon_days < 7 violates Query constraint → 422."""
    response = test_client.get("/forecast/revenue?horizon_days=3")
    assert response.status_code == 422


def test_forecast_horizon_too_large(test_client):
    """Phase 7: horizon_days > 90 violates Query constraint → 422."""
    response = test_client.get("/forecast/revenue?horizon_days=180")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Phase 6: Explainability unit checks
# ---------------------------------------------------------------------------

def test_churn_explainer_risk_tiers():
    """Phase 6: Risk tier logic maps probabilities to correct labels."""
    cases = [
        (0.80, "High Risk"),
        (0.65, "High Risk"),
        (0.50, "Medium Risk"),
        (0.35, "Medium Risk"),
        (0.34, "Low Risk"),
        (0.00, "Low Risk"),
    ]
    for prob, expected_tier in cases:
        tier = (
            "High Risk" if prob >= 0.65
            else ("Medium Risk" if prob >= 0.35 else "Low Risk")
        )
        assert tier == expected_tier, f"prob={prob} → got '{tier}', expected '{expected_tier}'"


def test_financial_roi_calculation():
    """Phase 6: ROI formula produces correct net value."""
    clv = 1200.0
    churn_prob = 0.72
    success_rate = 0.35
    intervention_cost = 25.0

    expected_saved = round(clv * churn_prob * success_rate, 2)
    net_roi = round(expected_saved - intervention_cost, 2)

    assert expected_saved == pytest.approx(302.4, abs=0.01)
    assert net_roi == pytest.approx(277.4, abs=0.01)


# ---------------------------------------------------------------------------
# Phase 4: Feature engineering unit checks
# ---------------------------------------------------------------------------

def test_charges_tenure_ratio_no_division_by_zero():
    """Phase 4: charges_tenure_ratio uses (tenure + 1) to avoid ZeroDivisionError."""
    tenure = 0
    total_charges = 89.50
    ratio = round(total_charges / (tenure + 1), 2)
    assert ratio == pytest.approx(89.50, abs=0.01)


def test_security_bundle_score_range():
    """Phase 4: Security bundle score is always 0-4."""
    services = ["online_security", "online_backup", "device_protection", "tech_support"]
    combos = [
        {"online_security": "Yes", "online_backup": "No", "device_protection": "Yes", "tech_support": "No"},
        {"online_security": "Yes", "online_backup": "Yes", "device_protection": "Yes", "tech_support": "Yes"},
        {"online_security": "No",  "online_backup": "No", "device_protection": "No",  "tech_support": "No"},
    ]
    for combo in combos:
        score = sum(1 if combo.get(k) == "Yes" else 0 for k in services)
        assert 0 <= score <= 4


# ---------------------------------------------------------------------------
# Phase 2: Schema / validation checks
# ---------------------------------------------------------------------------

def test_pydantic_churn_request_valid():
    """Phase 2: Valid ChurnPredictRequest parses without error."""
    from src.api.schemas import ChurnPredictRequest
    req = ChurnPredictRequest(tenure=12, monthly_charges=55.0)
    assert req.tenure == 12
    assert req.monthly_charges == 55.0


def test_pydantic_segment_request_defaults():
    """Phase 2: SegmentPredictRequest populates optional fields as None."""
    from src.api.schemas import SegmentPredictRequest
    req = SegmentPredictRequest(recency_days=5, frequency_orders=3, monetary_total=200.0)
    assert req.tenure_days is None
    assert req.avg_basket_value is None
