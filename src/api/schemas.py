"""Pydantic request and response validation schemas for Customer Analytics Platform API."""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------
# Churn Schemas
# ----------------------------------------------------
class ChurnPredictRequest(BaseModel):
    """Input payload for scoring individual customer churn risk."""
    customer_id: Optional[str] = Field("CUST-9999", description="Unique customer ID")
    tenure: int = Field(..., ge=0, le=120, description="Tenure in months")
    monthly_charges: float = Field(..., ge=0.0, le=500.0, description="Monthly recurring subscription cost ($)")
    total_charges: Optional[float] = Field(None, ge=0.0, description="Lifetime total charges ($)")
    contract: str = Field("Month-to-month", description="Contract term ('Month-to-month', 'One year', 'Two year')")
    internet_service: str = Field("Fiber optic", description="Internet service ('DSL', 'Fiber optic', 'No')")
    payment_method: str = Field("Electronic check", description="Billing method ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')")
    senior_citizen: int = Field(0, ge=0, le=1, description="1 if senior citizen, else 0")
    partner: str = Field("No", description="'Yes' or 'No'")
    dependents: str = Field("No", description="'Yes' or 'No'")
    phone_service: str = Field("Yes", description="'Yes' or 'No'")
    multiple_lines: str = Field("No", description="'Yes', 'No', or 'No phone service'")
    online_security: str = Field("No", description="'Yes', 'No', or 'No internet service'")
    online_backup: str = Field("No", description="'Yes', 'No', or 'No internet service'")
    device_protection: str = Field("No", description="'Yes', 'No', or 'No internet service'")
    tech_support: str = Field("No", description="'Yes', 'No', or 'No internet service'")
    streaming_tv: str = Field("No", description="'Yes', 'No', or 'No internet service'")
    streaming_movies: str = Field("No", description="'Yes', 'No', or 'No internet service'")
    paperless_billing: str = Field("Yes", description="'Yes' or 'No'")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "7590-VHVEG",
                "tenure": 3,
                "monthly_charges": 89.50,
                "total_charges": 268.50,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
                "senior_citizen": 0,
                "partner": "Yes",
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
        }
    }


class ChurnBatchRequest(BaseModel):
    """Batch input payload for scoring multiple customers."""
    customers: List[ChurnPredictRequest]


class ChurnPredictResponse(BaseModel):
    """Response payload with calibrated churn score, risk tier, SHAP drivers, and ROI."""
    customer_id: Optional[str]
    churn_probability: float = Field(..., description="Calibrated churn probability [0, 1]")
    risk_tier: str = Field(..., description="'High Risk', 'Medium Risk', or 'Low Risk'")
    is_high_risk: bool
    calibration: str
    top_risk_drivers: List[Dict[str, Any]]
    prescribed_actions: List[Dict[str, Any]]
    financial_roi: Dict[str, Any]


class ChurnBatchResponse(BaseModel):
    """Batch scoring response summary."""
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[ChurnPredictResponse]


# ----------------------------------------------------
# Customer Segmentation Schemas
# ----------------------------------------------------
class SegmentPredictRequest(BaseModel):
    """Input payload for customer behavioral clustering."""
    customer_id: Optional[str] = Field("13085", description="Customer ID")
    recency_days: int = Field(..., ge=0, description="Days since latest purchase")
    frequency_orders: int = Field(..., ge=1, description="Total distinct orders placed")
    monetary_total: float = Field(..., ge=0.0, description="Total cumulative spend (£)")
    tenure_days: Optional[int] = Field(None, ge=0, description="Days between first and last purchase")
    avg_basket_value: Optional[float] = Field(None, ge=0.0, description="Average spend per order")
    avg_basket_size: Optional[float] = Field(None, ge=0.0, description="Average items per order")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "13085",
                "recency_days": 14,
                "frequency_orders": 8,
                "monetary_total": 2450.00,
                "tenure_days": 180,
                "avg_basket_value": 306.25,
                "avg_basket_size": 22.5
            }
        }
    }


class SegmentPredictResponse(BaseModel):
    """Response payload with assigned cluster and behavioral persona."""
    customer_id: Optional[str]
    cluster_id: int
    persona_name: str
    features_used: List[str]


# ----------------------------------------------------
# Recommender Schemas
# ----------------------------------------------------
class RecommendRequest(BaseModel):
    """Request payload for personalized product recommendations."""
    customer_id: str = Field(..., description="Target customer ID")
    n_items: int = Field(5, ge=1, le=20, description="Number of recommendations to return")
    is_churn_risk: bool = Field(False, description="Set True to attach dynamic churn retention discounts")


class RecommendedItem(BaseModel):
    """Single recommended product item."""
    stock_code: str
    description: str
    base_price: float
    relevance_score: float
    strategy: str
    retention_incentive: Optional[str] = None
    discounted_price: Optional[float] = None


class RecommendResponse(BaseModel):
    """Response payload for smart recommendations."""
    customer_id: str
    total_recommendations: int
    is_churn_retention: bool
    recommendations: List[RecommendedItem]


# ----------------------------------------------------
# Forecasting Schemas
# ----------------------------------------------------
class ForecastResponse(BaseModel):
    """Forward demand and revenue projection response."""
    horizon_days: int
    projected_baseline_revenue: float
    unmitigated_churn_revenue_at_risk: float
    estimated_revenue_saved: float
    estimated_campaign_cost: float
    estimated_net_roi: float
    revenue_metrics: Dict[str, Any]
    orders_metrics: Dict[str, Any]


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    environment: str
    version: str
    models_loaded: List[str]
