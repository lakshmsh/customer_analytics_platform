"""Master training script for Customer Analytics Platform.

Executes all 4 machine learning pipelines sequentially:
1. Churn Prediction (Optuna + 4 Models + Probability Calibration + MLflow)
2. Customer Segmentation (K-Means + GMM + Persona Profiling + MLflow)
3. Advanced Hybrid Recommender (Collaborative + Content-Based + MLflow)
4. Demand & Revenue Forecasting (Time Series + Scenario Impact + MLflow)
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train_churn import train_and_evaluate_churn_models
from src.models.train_segmentation import train_and_evaluate_segmentation
from src.models.train_recommender import train_and_evaluate_recommender
from src.models.train_forecast import train_and_evaluate_forecasting
from src.utils.logger import get_logger

logger = get_logger("train_all")


def main():
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("   STARTING END-TO-END MODEL TRAINING SUITE (CORE + BONUSES)    ")
    logger.info("=================================================================")

    # 1. Churn Modeling
    logger.info("\n>>> [1/4] Training Churn Prediction Models...")
    churn_meta = train_and_evaluate_churn_models(tune=True)

    # 2. Customer Segmentation
    logger.info("\n>>> [2/4] Training Customer Segmentation Models...")
    seg_meta = train_and_evaluate_segmentation(n_clusters=4)

    # 3. Hybrid Recommender (Bonus 1)
    logger.info("\n>>> [3/4] Training Hybrid Recommender System...")
    rec_meta = train_and_evaluate_recommender(alpha=0.60)

    # 4. Demand & Revenue Forecasting (Bonus 2)
    logger.info("\n>>> [4/4] Training Demand & Revenue Forecasting Models...")
    forecast_meta = train_and_evaluate_forecasting(horizon_days=30)

    elapsed = time.time() - start_time
    logger.info("=================================================================")
    logger.info(f"   ALL MODELS TRAINED SUCCESSFULLY IN {elapsed:.1f} SECONDS!     ")
    logger.info("=================================================================")

    print("\n--- Model Training Summary ---")
    print(f"1. Best Churn Model: {churn_meta['best_model_name']} (Test ROC-AUC: {churn_meta['test_metrics']['roc_auc']:.4f}, PR-AUC: {churn_meta['test_metrics']['pr_auc']:.4f})")
    print(f"2. Segmentation: {seg_meta['algorithm']} (k={seg_meta['n_clusters']}, Silhouette: {seg_meta['metrics']['silhouette_score']:.4f})")
    print(f"3. Hybrid Recommender: Total Items={rec_meta['total_items']:,}, Total Users={rec_meta['total_users']:,}, Hit Rate@5={rec_meta['hit_rate_at_5']*100:.1f}%")
    print(f"4. Forecasting: 30-Day Revenue MAE=£{forecast_meta['revenue_metrics']['mae']:,.2f}, 30-Day Demand MAE={forecast_meta['orders_metrics']['mae']:.1f} orders")
    print(f"   Projected Revenue Impact Saved: £{forecast_meta['impact_scenario']['estimated_revenue_saved']:,.2f}")
    print("All artifacts saved in models/ and experiment runs logged in mlruns/.")


if __name__ == "__main__":
    main()
