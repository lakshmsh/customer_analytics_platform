"""Customer segmentation API endpoints."""

from fastapi import APIRouter, HTTPException, status
from src.api.schemas import SegmentPredictRequest, SegmentPredictResponse
from src.models.predict import predict_segment
from src.utils.logger import get_logger

logger = get_logger("api.routes.segment")
router = APIRouter(prefix="/segment", tags=["Customer Segmentation"])


@router.post("/predict", response_model=SegmentPredictResponse, summary="Assign Behavioral Persona")
def assign_customer_segment(payload: SegmentPredictRequest) -> SegmentPredictResponse:
    """Predicts customer cluster ID and assigned behavioral persona
    from RFM and basket parameters.
    """
    try:
        data = payload.model_dump()
        rec = max(data.get("recency_days", 0), 0)
        freq = max(data.get("frequency_orders", 1), 1)
        mon = max(data.get("monetary_total", 0.0), 0.0)
        tenure = data.get("tenure_days") if data.get("tenure_days") is not None else rec * 2
        basket_val = data.get("avg_basket_value") if data.get("avg_basket_value") is not None else round(mon / freq, 2)
        basket_sz = data.get("avg_basket_size") if data.get("avg_basket_size") is not None else 10.0

        rfm_features = {
            "recency_days": rec,
            "frequency_orders": freq,
            "monetary_total": mon,
            "tenure_days": tenure,
            "avg_basket_value": basket_val,
            "avg_basket_size": basket_sz
        }

        res = predict_segment(rfm_features)

        return SegmentPredictResponse(
            customer_id=payload.customer_id,
            cluster_id=res["cluster_id"],
            persona_name=res["persona_name"],
            features_used=res["features_used"]
        )
    except Exception as e:
        logger.error(f"Error predicting segment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}"
        )
