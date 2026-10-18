"""Model Explainability (SHAP) and Business Financial Impact Engine.

Provides:
1. Global and local SHAP feature attributions
2. Human-readable churn reason codes
3. Automated prescriptive intervention playbooks
4. Financial cost-benefit matrix & ROI threshold optimization
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger("evaluation.explainability")

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_READABLE_NAMES = {
    "tenure": "Account Longevity (Tenure)",
    "monthly_charges": "Monthly Subscription Fee ($)",
    "total_charges": "Cumulative Lifetime Billing ($)",
    "charges_tenure_ratio": "Monthly Spend Acceleration",
    "security_bundle_score": "Security & Protection Add-ons",
    "streaming_bundle_score": "Entertainment Streaming Add-ons",
    "has_family": "Partner or Dependent Presence",
    "contract_One year": "One-Year Contract Commitment",
    "contract_Two year": "Two-Year Contract Commitment",
    "contract_Month-to-month": "Month-to-Month Contract",
    "internet_service_Fiber optic": "Fiber Optic Internet Service",
    "internet_service_No": "No Internet Subscription",
    "payment_method_Credit card (automatic)": "Automatic Credit Card Billing",
    "payment_method_Electronic check": "Electronic Check Payment Method",
    "payment_method_Mailed check": "Mailed Paper Check",
    "paperless_billing_Yes": "Paperless Billing Enrolled",
    "senior_citizen": "Senior Citizen Account",
    "tech_support_Yes": "Dedicated Tech Support Active",
    "online_security_Yes": "Active Cyber Security Protection"
}

ACTION_PLAYBOOK = {
    "contract_Month-to-month": {
        "action": "Contract Migration Incentive",
        "detail": "Offer 1-year contract lock-in with a 10% monthly discount or free speed upgrade.",
        "cost": 15.0,
        "impact": "High (Reduces churn probability by ~73%)"
    },
    "internet_service_Fiber optic": {
        "action": "VIP Fiber Care & Proactive Line Check",
        "detail": "Route technical inquiries to tier-2 priority support and conduct a proactive line health audit.",
        "cost": 10.0,
        "impact": "High (Directly resolves premium service instability)"
    },
    "payment_method_Electronic check": {
        "action": "Auto-Pay Migration Incentive",
        "detail": "Offer a $10 one-time billing credit upon migrating to automated credit card or bank draft.",
        "cost": 10.0,
        "impact": "Medium (Eliminates recurring monthly payment friction)"
    },
    "monthly_charges": {
        "action": "Plan Right-Sizing & Bundle Discount",
        "detail": "Offer tailored bundle optimization to reduce monthly bill without reducing core services.",
        "cost": 25.0,
        "impact": "Medium (Restores perceived price-to-value balance)"
    },
    "charges_tenure_ratio": {
        "action": "Loyalty Price-Lock Guarantee",
        "detail": "Provide a 12-month rate freeze guarantee to mitigate sudden price increase fatigue.",
        "cost": 20.0,
        "impact": "High (Protects early-tenure high spenders)"
    },
    "security_bundle_score": {
        "action": "Complimentary Security Add-on Trial",
        "detail": "Provide 3 months complimentary access to Online Security and Cloud Backup.",
        "cost": 5.0,
        "impact": "Medium (Increases account switching costs and ecosystem stickiness)"
    }
}


class ChurnExplainer:
    """SHAP Explainer for Churn Prediction with automated prescriptive recommendations."""

    def __init__(self):
        self.model, self.scaler, self.metadata = ModelRegistry.load_churn_model()
        self.feature_names = self.metadata["feature_cols"]
        self.base_estimator = self.model.calibrated_classifiers_[0].estimator

        # Prepare background data for explainer
        df_processed = pd.read_parquet("data/processed/telco_processed.parquet")
        background_X = df_processed[self.feature_names].values[:200]
        if self.scaler is not None:
            background_X = self.scaler.transform(background_X)

        self.explainer = shap.LinearExplainer(self.base_estimator, background_X)
        logger.info("Initialized ChurnExplainer with SHAP LinearExplainer.")

    def explain_customer(
        self,
        customer_features: Dict[str, Any],
        clv: float = 1200.0,
        intervention_cost: float = 25.0,
        success_rate: float = 0.35
    ) -> Dict[str, Any]:
        """Generate local SHAP explanation, key churn drivers, and actionable recommendations.

        Args:
            customer_features: Dictionary of raw customer attributes.
            clv: Customer Lifetime Value (CLV proxy) in USD.
            intervention_cost: Cost to execute retention offer ($).
            success_rate: Probability that intervention prevents churn.

        Returns:
            Dict[str, Any]: Full local explainability report with prescriptive actions.
        """
        # Align features
        df_row = pd.DataFrame([customer_features])
        for col in self.feature_names:
            if col not in df_row.columns:
                df_row[col] = 0.0

        X = df_row[self.feature_names].values
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Predict calibrated probability
        churn_prob = float(self.model.predict_proba(X_scaled)[0, 1])

        # Compute SHAP
        shap_vals = self.explainer(X_scaled)
        values = shap_vals.values[0]
        base_val = float(shap_vals.base_values[0])

        # Separate risk drivers (+ SHAP) and retention buffers (- SHAP)
        drivers = []
        buffers = []

        for f_name, s_val in zip(self.feature_names, values):
            readable = FEATURE_READABLE_NAMES.get(f_name, f_name.replace("_", " ").title())
            entry = {
                "feature": f_name,
                "readable_name": readable,
                "shap_impact": round(float(s_val), 4),
                "feature_value": float(df_row[f_name].iloc[0])
            }
            if s_val > 0.02:
                drivers.append(entry)
            elif s_val < -0.02:
                buffers.append(entry)

        # Sort by magnitude
        drivers = sorted(drivers, key=lambda x: abs(x["shap_impact"]), reverse=True)[:5]
        buffers = sorted(buffers, key=lambda x: abs(x["shap_impact"]), reverse=True)[:5]

        # Generate Prescriptive Actions based on top drivers
        prescribed_actions = []
        for d in drivers:
            f_code = d["feature"]
            if f_code in ACTION_PLAYBOOK:
                action_info = ACTION_PLAYBOOK[f_code].copy()
                action_info["targeted_driver"] = d["readable_name"]
                action_info["driver_impact"] = d["shap_impact"]
                prescribed_actions.append(action_info)

        # Financial ROI Calculation:
        # Expected Revenue Saved = CLV * P(Churn) * SuccessRate
        # Net Benefit = Expected Revenue Saved - InterventionCost
        expected_saved = round(clv * churn_prob * success_rate, 2)
        net_roi = round(expected_saved - intervention_cost, 2)

        return {
            "churn_probability": round(churn_prob, 4),
            "risk_tier": "High Risk" if churn_prob >= 0.65 else ("Medium Risk" if churn_prob >= 0.35 else "Low Risk"),
            "base_rate_log_odds": round(base_val, 4),
            "top_risk_drivers": drivers,
            "retention_buffers": buffers,
            "prescribed_actions": prescribed_actions,
            "financial_roi": {
                "clv_assumed": clv,
                "churn_probability": round(churn_prob, 4),
                "intervention_success_rate": success_rate,
                "intervention_cost": intervention_cost,
                "expected_gross_revenue_saved": expected_saved,
                "net_expected_roi": net_roi,
                "recommendation": "EXECUTE RETENTION INTERVENTION" if net_roi > 0 and churn_prob >= 0.35 else "MONITOR ONLY"
            }
        }

    def generate_global_shap_figures(
        self,
        save_summary_path: Optional[str | Path] = FIGURES_DIR / "07_shap_summary.png",
        save_waterfall_path: Optional[str | Path] = FIGURES_DIR / "08_shap_local_waterfall.png"
    ):
        """Generate and save global beeswarm/bar and local waterfall plots."""
        logger.info("Generating global SHAP explanation plots...")
        df_processed = pd.read_parquet("data/processed/telco_processed.parquet")
        X = df_processed[self.feature_names].values[:500]
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        shap_vals = self.explainer(X_scaled)
        shap_vals.feature_names = [FEATURE_READABLE_NAMES.get(f, f) for f in self.feature_names]

        # 1. Global Bar Summary Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.bar(shap_vals, max_display=12, show=False)
        plt.title("Global SHAP Feature Importance (Churn Prediction)", fontsize=13, fontweight="bold")
        if save_summary_path:
            plt.savefig(save_summary_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Local Waterfall Plot (for high-risk sample customer)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
        plt.title("Local SHAP Attribution (Customer #0 Waterfall)", fontsize=13, fontweight="bold")
        if save_waterfall_path:
            plt.savefig(save_waterfall_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("SHAP figures generated and saved successfully.")


def optimize_business_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    clv: float = 1200.0,
    cost_intervention: float = 25.0,
    success_rate: float = 0.35
) -> Dict[str, Any]:
    """Find the optimal probability threshold maximizing net financial value."""
    thresholds = np.linspace(0.10, 0.90, 81)
    best_profit = -np.inf
    best_thresh = 0.50
    curves = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))

        # Expected Value Framework
        # TP: Customer at risk saved with success_rate -> + (CLV * success_rate - cost)
        # FP: Customer stayed anyway, intervention executed -> - cost
        # FN: Customer churned, missed -> - CLV
        # TN: Correctly left alone -> $0
        benefit_tp = tp * (clv * success_rate - cost_intervention)
        cost_fp = fp * cost_intervention
        loss_fn = fn * clv

        net_value = benefit_tp - cost_fp - loss_fn

        curves.append({
            "threshold": round(float(t), 2),
            "net_financial_value": round(float(net_value), 2),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        })

        if net_value > best_profit:
            best_profit = net_value
            best_thresh = t

    return {
        "optimal_threshold": round(float(best_thresh), 2),
        "max_net_value": round(float(best_profit), 2),
        "assumptions": {
            "clv": clv,
            "cost_intervention": cost_intervention,
            "success_rate": success_rate
        },
        "threshold_curve_sample": curves[::10]
    }
