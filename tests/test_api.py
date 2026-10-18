"""Unit and integration tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint responds with metadata and 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_check_endpoint():
    """Verify health check reports healthy status and all 4 models loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert len(data["models_loaded"]) == 4


def test_churn_predict_valid():
    """Verify single customer churn prediction returns calibrated score and SHAP drivers."""
    payload = {
        "customer_id": "TEST-CUST-1",
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
        "paperless_billing": "Yes"
    }
    response = client.post("/churn/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "TEST-CUST-1"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ["High Risk", "Medium Risk", "Low Risk"]
    assert "top_risk_drivers" in data
    assert "prescribed_actions" in data
    assert "financial_roi" in data
    assert data["financial_roi"]["net_expected_roi"] is not None


def test_churn_predict_invalid():
    """Verify Pydantic strictly rejects invalid inputs (e.g. negative tenure)."""
    payload = {
        "tenure": -5,  # Invalid: must be >= 0
        "monthly_charges": 50.0
    }
    response = client.post("/churn/predict", json=payload)
    assert response.status_code == 422


def test_churn_batch_predict():
    """Verify batch churn prediction endpoint scores multiple records."""
    payload = {
        "customers": [
            {
                "customer_id": "CUST-A",
                "tenure": 1,
                "monthly_charges": 95.0,
                "contract": "Month-to-month"
            },
            {
                "customer_id": "CUST-B",
                "tenure": 60,
                "monthly_charges": 20.0,
                "contract": "Two year"
            }
        ]
    }
    response = client.post("/churn/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2


def test_segment_predict():
    """Verify customer behavioral persona assignment endpoint."""
    payload = {
        "customer_id": "13085",
        "recency_days": 10,
        "frequency_orders": 12,
        "monetary_total": 3500.00,
        "tenure_days": 250,
        "avg_basket_value": 291.67,
        "avg_basket_size": 18.0
    }
    response = client.post("/segment/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "13085"
    assert "cluster_id" in data
    assert isinstance(data["persona_name"], str)


def test_recommend_endpoint():
    """Verify hybrid recommendation endpoint returns personalized product list."""
    response = client.get("/recommend/13085?n_items=4&is_churn_risk=true")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "13085"
    assert len(data["recommendations"]) == 4
    first_item = data["recommendations"][0]
    assert "stock_code" in first_item
    assert "description" in first_item
    assert "retention_incentive" in first_item


def test_forecast_endpoint():
    """Verify forward demand and revenue forecasting endpoint."""
    response = client.get("/forecast/revenue?horizon_days=30")
    assert response.status_code == 200
    data = response.json()
    assert data["horizon_days"] == 30
    assert data["projected_baseline_revenue"] > 0.0
    assert data["estimated_revenue_saved"] > 0.0
