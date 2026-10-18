"""Demand and revenue forecasting API endpoints."""

from fastapi import APIRouter, HTTPException, Query, status
from src.api.schemas import ForecastResponse
from src.models.predict import predict_forecast
from src.utils.logger import get_logger

logger = get_logger("api.routes.forecast")
router = APIRouter(prefix="/forecast", tags=["Demand & Revenue Forecasting"])


@router.get("/revenue", response_model=ForecastResponse, summary="Get Forward Demand & Revenue Forecast")
def get_revenue_forecast(
    horizon_days: int = Query(30, ge=7, le=90, description="Forward forecast window in days")
) -> ForecastResponse:
    """Returns forward 30-day demand, revenue predictions, error metrics,
    and simulated financial churn impact scenario.
    """
    try:
        data = predict_forecast(horizon_days=horizon_days)
        impact = data.get("impact_scenario", {})

        return ForecastResponse(
            horizon_days=horizon_days,
            projected_baseline_revenue=impact.get("projected_baseline_revenue", 0.0),
            unmitigated_churn_revenue_at_risk=impact.get("unmitigated_churn_revenue_at_risk", 0.0),
            estimated_revenue_saved=impact.get("estimated_revenue_saved", 0.0),
            estimated_campaign_cost=impact.get("estimated_campaign_cost", 0.0),
            estimated_net_roi=impact.get("estimated_net_roi", 0.0),
            revenue_metrics=data.get("revenue_metrics", {}),
            orders_metrics=data.get("orders_metrics", {})
        )
    except Exception as e:
        logger.error(f"Error generating forecast: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting retrieval failed: {str(e)}"
        )
