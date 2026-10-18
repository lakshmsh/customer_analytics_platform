"""Recommendation system API endpoints."""

from fastapi import APIRouter, HTTPException, Query, status
from typing import List
from src.api.schemas import RecommendRequest, RecommendResponse, RecommendedItem
from src.models.predict import get_recommendations
from src.utils.logger import get_logger

logger = get_logger("api.routes.recommend")
router = APIRouter(prefix="/recommend", tags=["Smart Product Recommendations"])


@router.get("/{customer_id}", response_model=RecommendResponse, summary="Get Product Recommendations for Customer")
def recommend_for_customer(
    customer_id: str,
    n_items: int = Query(5, ge=1, le=20, description="Number of items to return"),
    is_churn_risk: bool = Query(False, description="Whether to attach retention discount incentives")
) -> RecommendResponse:
    """Returns personalized product recommendations using the Hybrid Recommender.
    If is_churn_risk is True, attaches churn-mitigation retention discounts.
    """
    try:
        recs = get_recommendations(customer_id, n_items=n_items, is_churn_risk=is_churn_risk)
        items = [RecommendedItem(**r) for r in recs]

        return RecommendResponse(
            customer_id=customer_id,
            total_recommendations=len(items),
            is_churn_retention=is_churn_risk,
            recommendations=items
        )
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation retrieval failed: {str(e)}"
        )


@router.post("", response_model=RecommendResponse, summary="Post Recommendation Query")
def recommend_post(payload: RecommendRequest) -> RecommendResponse:
    """POST endpoint alternative for generating personalized recommendations."""
    return recommend_for_customer(
        customer_id=payload.customer_id,
        n_items=payload.n_items,
        is_churn_risk=payload.is_churn_risk
    )
