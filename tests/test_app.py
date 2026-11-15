"""
Unit tests for Streamlit application components and analytics client (Phase 8).
"""

import pytest
from app.components.api_client import AnalyticsClient
from app.components.ui_elements import (
    plot_churn_gauge,
    plot_shap_waterfall,
    plot_rfm_radar,
    plot_revenue_forecast_chart
)


def test_analytics_client_health(mock_registry):
    """Verify client health check returns system status and models list."""
    health = AnalyticsClient.get_health()
    assert "status" in health
    assert "models_loaded" in health
    assert isinstance(health["models_loaded"], list)


def test_analytics_client_predict_churn(mock_registry, mock_shap):
    """Verify single churn prediction via AnalyticsClient."""
    payload = {
        "customer_id": "TEST-APP-01",
        "tenure": 3,
        "monthly_charges": 85.0,
        "contract": "Month-to-month"
    }
    res = AnalyticsClient.predict_churn(payload)
    assert res["customer_id"] == "TEST-APP-01"
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["risk_tier"] in ["High Risk", "Medium Risk", "Low Risk"]


def test_analytics_client_batch_churn(mock_registry, mock_shap):
    """Verify batch churn prediction via AnalyticsClient."""
    customers = [
        {"customer_id": "C1", "tenure": 2, "monthly_charges": 70.0},
        {"customer_id": "C2", "tenure": 40, "monthly_charges": 30.0}
    ]
    res = AnalyticsClient.predict_batch_churn(customers)
    assert res["total_customers"] == 2
    assert len(res["predictions"]) == 2


def test_analytics_client_segment(mock_registry):
    """Verify persona assignment via AnalyticsClient."""
    rfm = {
        "recency_days": 15,
        "frequency_orders": 6,
        "monetary_total": 1200.0
    }
    res = AnalyticsClient.get_segment(rfm)
    assert "cluster_id" in res
    assert "persona_name" in res


def test_analytics_client_recommendations(mock_registry):
    """Verify hybrid recommendation via AnalyticsClient."""
    res = AnalyticsClient.get_recommendations(customer_id="13085", n_items=3, is_churn_risk=True)
    assert res["customer_id"] == "13085"
    assert len(res["recommendations"]) == 3
    assert res["is_churn_retention"] is True


def test_analytics_client_forecast(mock_registry):
    """Verify demand and revenue forecasting via AnalyticsClient."""
    res = AnalyticsClient.get_forecast(horizon_days=30)
    assert res["horizon_days"] == 30
    assert res["projected_baseline_revenue"] > 0.0


def test_ui_elements_charts():
    """Verify Plotly figure generators execute and return valid figures."""
    fig_gauge = plot_churn_gauge(0.72)
    assert fig_gauge is not None

    drivers = [{"readable_name": "Month-to-Month Contract", "shap_impact": 0.28}]
    buffers = [{"readable_name": "Two-Year Contract", "shap_impact": -0.32}]
    fig_shap = plot_shap_waterfall(drivers, buffers)
    assert fig_shap is not None

    fig_radar = plot_rfm_radar({"recency_days": 10, "frequency_orders": 5, "monetary_total": 500.0})
    assert fig_radar is not None

    fig_forecast = plot_revenue_forecast_chart(30, 100000.0, 5000.0)
    assert fig_forecast is not None
