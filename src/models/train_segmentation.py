"""Customer Segmentation Engine training module.

Applies unsupervised clustering (K-Means and Gaussian Mixture Models)
on scaled RFM and behavioral metrics to profile distinct customer personas.
Logs cluster quality metrics and profiles to MLflow.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from src.evaluation.metrics import compute_clustering_metrics
from src.utils.logger import get_logger
from src.utils.helpers import timer

logger = get_logger("models.train_segmentation")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR = Path("mlruns")
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

SEGMENTATION_FEATURES = [
    "recency_days",
    "frequency_orders",
    "monetary_total",
    "tenure_days",
    "avg_basket_value",
    "avg_basket_size"
]


@timer
def train_and_evaluate_segmentation(n_clusters: int = 4) -> Dict[str, Any]:
    """Train K-Means and GMM clustering models, log to MLflow, and assign personas.

    Args:
        n_clusters: Number of target clusters (default: 4).

    Returns:
        Dict[str, Any]: Segmentation metadata and persona profiles.
    """
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    sqlite_db = (MLRUNS_DIR / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{sqlite_db}")
    mlflow.set_experiment("Customer_Segmentation")

    logger.info("Loading customer feature store for segmentation...")
    df = pd.read_parquet("data/processed/customers_features.parquet")

    # Select and validate features
    available_cols = [c for c in SEGMENTATION_FEATURES if c in df.columns]
    X_raw = df[available_cols].copy()
    logger.info(f"Using {len(available_cols)} behavioral features: {available_cols}")

    # Scale using RobustScaler (resilient to extreme spending outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 1. Search Optimal K using K-Means (k=3, 4, 5)
    best_sil = -1.0
    best_k = n_clusters
    kmeans_evals = {}

    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        metrics = compute_clustering_metrics(X_scaled, labels)
        kmeans_evals[k] = metrics
        logger.info(f"K-Means (k={k}): Silhouette={metrics['silhouette_score']}, DB Index={metrics['davies_bouldin_score']}")
        if metrics["silhouette_score"] > best_sil:
            best_sil = metrics["silhouette_score"]

    # 2. Benchmark K-Means vs Gaussian Mixture Model (GMM)
    with mlflow.start_run(run_name=f"Segmentation_KMeans_k{n_clusters}"):
        best_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km_labels = best_kmeans.fit_predict(X_scaled)
        km_metrics = compute_clustering_metrics(X_scaled, km_labels)

        mlflow.log_params({"algorithm": "KMeans", "n_clusters": n_clusters})
        mlflow.log_metrics(km_metrics)

    with mlflow.start_run(run_name=f"Segmentation_GMM_k{n_clusters}"):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type="full")
        gmm_labels = gmm.fit_predict(X_scaled)
        gmm_metrics = compute_clustering_metrics(X_scaled, gmm_labels)

        mlflow.log_params({"algorithm": "GaussianMixture", "n_clusters": n_clusters})
        mlflow.log_metrics(gmm_metrics)

    logger.info(f"K-Means (k={n_clusters}) Silhouette: {km_metrics['silhouette_score']}")
    logger.info(f"GMM (k={n_clusters}) Silhouette: {gmm_metrics['silhouette_score']}")

    # Select K-Means as primary production model for distinct, spherical boundaries
    selected_model = best_kmeans
    final_labels = km_labels

    df_segmented = df.copy()
    df_segmented["cluster_id"] = final_labels

    # 3. Profile Cluster Personas
    cluster_profiles = df_segmented.groupby("cluster_id").agg(
        size=("customer_id", "count"),
        recency_median=("recency_days", "median"),
        frequency_median=("frequency_orders", "median"),
        monetary_median=("monetary_total", "median"),
        avg_basket_median=("avg_basket_value", "median"),
        churn_rate=("is_churned", "mean")
    ).reset_index()

    # Map to actionable business personas based on rank ordering
    # Highest monetary + frequency = Champions
    # High monetary + high recency = At Risk High Rollers
    # Low recency + moderate monetary = Potential Loyalists
    # High recency + low monetary = Hibernating
    def assign_persona(row: pd.Series) -> str:
        if row["monetary_median"] >= cluster_profiles["monetary_median"].quantile(0.75):
            if row["recency_median"] > cluster_profiles["recency_median"].median():
                return "At-Risk High Rollers"
            return "Champions"
        elif row["recency_median"] <= cluster_profiles["recency_median"].median():
            return "Potential Loyalists"
        else:
            return "Hibernating / Dormant"

    cluster_profiles["persona_name"] = cluster_profiles.apply(assign_persona, axis=1)
    persona_mapping = dict(zip(cluster_profiles["cluster_id"], cluster_profiles["persona_name"]))

    df_segmented["persona_name"] = df_segmented["cluster_id"].map(persona_mapping)

    logger.info("Identified Personas:")
    for _, r in cluster_profiles.iterrows():
        logger.info(
            f"  Cluster {int(r['cluster_id'])} -> '{r['persona_name']}' "
            f"(Count: {int(r['size']):,}, Median Spend: £{r['monetary_median']:.2f}, "
            f"Median Recency: {r['recency_median']:.0f}d, Churn Rate: {r['churn_rate']*100:.1f}%)"
        )

    # 4. Persist Segmented Feature Store & Artifacts
    df_segmented.to_parquet("data/processed/customers_features.parquet", index=False)
    logger.info("Updated data/processed/customers_features.parquet with cluster_id and persona_name.")

    model_path = MODELS_DIR / "segmentation_model.joblib"
    scaler_path = MODELS_DIR / "segmentation_scaler.joblib"
    meta_path = MODELS_DIR / "segmentation_metadata.json"

    joblib.dump(selected_model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "algorithm": "KMeans",
        "n_clusters": n_clusters,
        "features": available_cols,
        "metrics": km_metrics,
        "persona_mapping": {int(k): v for k, v in persona_mapping.items()},
        "cluster_profiles": cluster_profiles.to_dict(orient="records")
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Persisted segmentation model to {model_path}")
    return metadata


if __name__ == "__main__":
    train_and_evaluate_segmentation(n_clusters=4)
