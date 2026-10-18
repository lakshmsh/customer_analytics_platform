"""Churn prediction API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from src.api.schemas import (
    ChurnPredictRequest,
    ChurnPredictResponse,
    ChurnBatchRequest,
    ChurnBatchResponse
)
from src.api.dependencies import get_churn_explainer
from src.evaluation.explainability import ChurnExplainer
from src.utils.logger import get_logger

logger = get_logger("api.routes.churn")
router = APIRouter(prefix="/churn", tags=["Churn Risk Prediction"])


def _format_customer_features(req: ChurnPredictRequest) -> Dict[str, Any]:
    """Helper to prepare model features from request payload."""
    data = req.model_dump()
    tenure = max(data.get("tenure", 0), 0)
    monthly = max(data.get("monthly_charges", 0.0), 0.0)
    total = data.get("total_charges")
    if total is None:
        total = round(tenure * monthly, 2)

    features = {
        "tenure": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "charges_tenure_ratio": round(total / (tenure + 1), 2),
        "senior_citizen": data.get("senior_citizen", 0),
        "has_family": 1 if (data.get("partner") == "Yes" or data.get("dependents") == "Yes") else 0,
        "security_bundle_score": sum([
            1 if data.get(k) == "Yes" else 0
            for k in ["online_security", "online_backup", "device_protection", "tech_support"]
        ]),
        "streaming_bundle_score": sum([
            1 if data.get(k) == "Yes" else 0
            for k in ["streaming_tv", "streaming_movies"]
        ]),
        f"contract_{data.get('contract')}": 1,
        f"internet_service_{data.get('internet_service')}": 1,
        f"payment_method_{data.get('payment_method')}": 1,
        f"paperless_billing_{data.get('paperless_billing')}": 1,
        f"partner_{data.get('partner')}": 1,
        f"dependents_{data.get('dependents')}": 1,
        f"phone_service_{data.get('phone_service')}": 1,
        f"multiple_lines_{data.get('multiple_lines')}": 1,
    }
    return features


@router.post("/predict", response_model=ChurnPredictResponse, summary="Predict Churn Risk for Customer")
def predict_single_churn(
    payload: ChurnPredictRequest,
    explainer: ChurnExplainer = Depends(get_churn_explainer)
) -> ChurnPredictResponse:
    """Predicts calibrated churn probability, risk tier, top SHAP drivers,
    recommended business actions, and expected financial ROI for a customer.
    """
    try:
        features = _format_customer_features(payload)
        explanation = explainer.explain_customer(features, clv=1200.0)

        return ChurnPredictResponse(
            customer_id=payload.customer_id,
            churn_probability=explanation["churn_probability"],
            risk_tier=explanation["risk_tier"],
            is_high_risk=explanation["risk_tier"] == "High Risk",
            calibration="Calibrated (Platt Sigmoid)",
            top_risk_drivers=explanation["top_risk_drivers"],
            prescribed_actions=explanation["prescribed_actions"],
            financial_roi=explanation["financial_roi"]
        )
    except Exception as e:
        logger.error(f"Error scoring churn: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Churn scoring failed: {str(e)}"
        )


@router.post("/batch", response_model=ChurnBatchResponse, summary="Batch Score Churn Risk")
def predict_batch_churn(
    payload: ChurnBatchRequest,
    explainer: ChurnExplainer = Depends(get_churn_explainer)
) -> ChurnBatchResponse:
    """Score a batch of customer records and return aggregated risk breakdown."""
    try:
        predictions = []
        high_cnt, med_cnt, low_cnt = 0, 0, 0

        for cust_req in payload.customers:
            features = _format_customer_features(cust_req)
            explanation = explainer.explain_customer(features, clv=1200.0)
            tier = explanation["risk_tier"]

            if tier == "High Risk":
                high_cnt += 1
            elif tier == "Medium Risk":
                med_cnt += 1
            else:
                low_cnt += 1

            predictions.append(ChurnPredictResponse(
                customer_id=cust_req.customer_id,
                churn_probability=explanation["churn_probability"],
                risk_tier=tier,
                is_high_risk=tier == "High Risk",
                calibration="Calibrated (Platt Sigmoid)",
                top_risk_drivers=explanation["top_risk_drivers"],
                prescribed_actions=explanation["prescribed_actions"],
                financial_roi=explanation["financial_roi"]
            ))

        return ChurnBatchResponse(
            total_customers=len(payload.customers),
            high_risk_count=high_cnt,
            medium_risk_count=med_cnt,
            low_risk_count=low_cnt,
            predictions=predictions
        )
    except Exception as e:
        logger.error(f"Error in batch churn scoring: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch churn scoring failed: {str(e)}"
        )
