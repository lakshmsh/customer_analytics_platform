"""Churn Prediction Engine training module.

Trains and benchmarks:
- Logistic Regression (Baseline)
- XGBoost
- LightGBM
- CatBoost

Features Optuna hyperparameter tuning, probability calibration, and MLflow experiment tracking.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import joblib
import numpy as np
import pandas as pd
import optuna
import mlflow
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.evaluation.metrics import compute_classification_metrics
from src.features.encoding import encode_telco_features
from src.utils.logger import get_logger
from src.utils.helpers import timer

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger("models.train_churn")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR = Path("mlruns")
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_churn_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and prepare features and target for churn modeling."""
    logger.info("Loading cleaned Telco dataset for churn modeling...")
    raw_df = pd.read_parquet("data/interim/telco_cleaned.parquet")
    df_encoded, metadata = encode_telco_features(raw_df)

    feature_cols = metadata["all_feature_cols"]
    X = df_encoded[feature_cols].copy()
    y = df_encoded["churn_label"].copy()

    logger.info(f"Loaded dataset: {X.shape[0]:,} samples, {X.shape[1]} features. Positive class rate: {y.mean() * 100:.2f}%.")
    return X, y, feature_cols


@timer
def tune_lightgbm_optuna(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 15) -> Dict[str, Any]:
    """Tune LightGBM hyperparameters using Optuna cross-validation."""
    logger.info(f"Starting Optuna hyperparameter optimization for LightGBM ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        pr_aucs = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            x_tr, x_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = LGBMClassifier(**params)
            model.fit(x_tr, y_tr)
            y_prob = model.predict_proba(x_val)[:, 1]
            metrics = compute_classification_metrics(y_val, y_prob)
            pr_aucs.append(metrics["pr_auc"])

        return np.mean(pr_aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Optuna Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best Hyperparameters: {study.best_params}")
    return study.best_params


@timer
def train_and_evaluate_churn_models(tune: bool = True) -> Dict[str, Any]:
    """Train, tune, evaluate, and serialize churn models.

    Args:
        tune: Whether to run Optuna tuning for LightGBM.

    Returns:
        Dict[str, Any]: Benchmark results and metadata.
    """
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    sqlite_db = (MLRUNS_DIR / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{sqlite_db}")
    mlflow.set_experiment("Churn_Prediction_Benchmark")

    X, y, feature_cols = load_churn_data()

    # 1. Stratified Train / Validation / Test Split (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info(f"Split sizes: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

    # Standard scale for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Class weight ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    logger.info(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

    # Hyperparameter search if requested
    best_lgb_params = {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 5,
        "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8,
        "class_weight": "balanced", "random_state": 42, "verbose": -1
    }
    if tune:
        tuned = tune_lightgbm_optuna(X_train, y_train, n_trials=10)
        best_lgb_params.update(tuned)

    # Models candidate suite
    candidate_models = {
        "Logistic_Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            X_train_scaled, X_val_scaled, X_test_scaled
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss"
            ),
            X_train, X_val, X_test
        ),
        "LightGBM": (
            LGBMClassifier(**best_lgb_params),
            X_train, X_val, X_test
        ),
        "CatBoost": (
            CatBoostClassifier(
                iterations=200, depth=5, learning_rate=0.05,
                auto_class_weights="Balanced", random_seed=42, verbose=0
            ),
            X_train, X_val, X_test
        )
    }

    results = {}
    best_model_name = None
    best_val_prauc = -1.0
    best_model_obj = None

    for name, (clf, tr_x, val_x, te_x) in candidate_models.items():
        logger.info(f"Training and evaluating: {name}...")

        with mlflow.start_run(run_name=f"Churn_{name}"):
            clf.fit(tr_x, y_train)

            # Evaluate on Validation
            val_probs = clf.predict_proba(val_x)[:, 1]
            val_metrics = compute_classification_metrics(y_val, val_probs)

            # Evaluate on Test
            test_probs = clf.predict_proba(te_x)[:, 1]
            test_metrics = compute_classification_metrics(y_test, test_probs)

            # Log to MLflow
            mlflow.log_params({"model_name": name, "class_imbalance_handled": True})
            for m_key, m_val in test_metrics.items():
                if isinstance(m_val, (int, float)):
                    mlflow.log_metric(f"test_{m_key}", m_val)
                mlflow.log_metric(f"val_{m_key}", val_metrics.get(m_key, 0))

            results[name] = {
                "validation": val_metrics,
                "test": test_metrics
            }

            logger.info(
                f"[{name}] Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
                f"Test ROC-AUC: {test_metrics['roc_auc']:.4f} | "
                f"Test PR-AUC: {test_metrics['pr_auc']:.4f}"
            )

            if val_metrics["pr_auc"] > best_val_prauc:
                best_val_prauc = val_metrics["pr_auc"]
                best_model_name = name
                best_model_obj = clf

    logger.info(f"--- BEST CHURN MODEL SELECTED: {best_model_name} (Val PR-AUC: {best_val_prauc:.4f}) ---")

    # 4. Calibrate Best Model for Accurate Probability Estimations
    logger.info(f"Calibrating probabilities for {best_model_name} via Sigmoid CalibratedClassifierCV...")
    best_tr_x = candidate_models[best_model_name][1]
    best_te_x = candidate_models[best_model_name][3]

    calibrated_model = CalibratedClassifierCV(estimator=best_model_obj, method="sigmoid", cv=3)
    calibrated_model.fit(best_tr_x, y_train)

    cal_test_probs = calibrated_model.predict_proba(best_te_x)[:, 1]
    cal_metrics = compute_classification_metrics(y_test, cal_test_probs)
    logger.info(
        f"Calibrated Test Metrics -> ROC-AUC: {cal_metrics['roc_auc']:.4f}, "
        f"PR-AUC: {cal_metrics['pr_auc']:.4f}, Brier Score: {cal_metrics['brier_score']:.4f}"
    )

    # 5. Persist Best Model and Metadata
    model_artifact_path = MODELS_DIR / "churn_model.joblib"
    scaler_artifact_path = MODELS_DIR / "churn_scaler.joblib"
    metadata_artifact_path = MODELS_DIR / "churn_metadata.json"

    joblib.dump(calibrated_model, model_artifact_path)
    joblib.dump(scaler, scaler_artifact_path)

    metadata = {
        "best_model_name": best_model_name,
        "is_calibrated": True,
        "calibration_method": "sigmoid",
        "feature_cols": feature_cols,
        "test_metrics": cal_metrics,
        "all_benchmarks": results
    }
    with open(metadata_artifact_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Persisted production churn model to: {model_artifact_path}")
    logger.info(f"Persisted model metadata to: {metadata_artifact_path}")
    return metadata


if __name__ == "__main__":
    train_and_evaluate_churn_models(tune=True)
