"""Model Registry module for loading and caching trained model artifacts."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import joblib
from src.utils.logger import get_logger

logger = get_logger("models.registry")

MODELS_DIR = Path("models")


class ModelRegistry:
    """Central registry providing cached access to trained models and metadata."""

    _models: Dict[str, Any] = {}
    _metadata: Dict[str, Any] = {}

    @classmethod
    def load_churn_model(cls) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load trained churn classifier, scaler, and metadata."""
        if "churn_model" not in cls._models:
            model_path = MODELS_DIR / "churn_model.joblib"
            scaler_path = MODELS_DIR / "churn_scaler.joblib"
            meta_path = MODELS_DIR / "churn_metadata.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Churn model not found at {model_path}. Please train models first.")

            cls._models["churn_model"] = joblib.load(model_path)
            cls._models["churn_scaler"] = joblib.load(scaler_path) if scaler_path.exists() else None
            with open(meta_path, "r", encoding="utf-8") as f:
                cls._metadata["churn"] = json.load(f)
            logger.info("Loaded Churn Model and Scaler into registry cache.")

        return cls._models["churn_model"], cls._models["churn_scaler"], cls._metadata["churn"]

    @classmethod
    def load_segmentation_model(cls) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load trained customer segmentation model, scaler, and metadata."""
        if "segmentation_model" not in cls._models:
            model_path = MODELS_DIR / "segmentation_model.joblib"
            scaler_path = MODELS_DIR / "segmentation_scaler.joblib"
            meta_path = MODELS_DIR / "segmentation_metadata.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Segmentation model not found at {model_path}. Please train models first.")

            cls._models["segmentation_model"] = joblib.load(model_path)
            cls._models["segmentation_scaler"] = joblib.load(scaler_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                cls._metadata["segmentation"] = json.load(f)
            logger.info("Loaded Segmentation Model and Scaler into registry cache.")

        return cls._models["segmentation_model"], cls._models["segmentation_scaler"], cls._metadata["segmentation"]

    @classmethod
    def load_recommender(cls) -> Tuple[Any, Dict[str, Any]]:
        """Load trained hybrid recommender engine and metadata."""
        if "recommender_model" not in cls._models:
            model_path = MODELS_DIR / "recommender_model.joblib"
            meta_path = MODELS_DIR / "recommender_metadata.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Recommender model not found at {model_path}. Please train models first.")

            cls._models["recommender_model"] = joblib.load(model_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                cls._metadata["recommender"] = json.load(f)
            logger.info("Loaded Hybrid Recommender into registry cache.")

        return cls._models["recommender_model"], cls._metadata["recommender"]

    @classmethod
    def load_forecast_models(cls) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load trained revenue & demand forecasters and metadata."""
        if "forecast_revenue_model" not in cls._models:
            rev_path = MODELS_DIR / "forecast_revenue_model.joblib"
            ord_path = MODELS_DIR / "forecast_orders_model.joblib"
            meta_path = MODELS_DIR / "forecast_metadata.json"

            if not rev_path.exists() or not ord_path.exists():
                raise FileNotFoundError("Forecasting models not found. Please train models first.")

            cls._models["forecast_revenue_model"] = joblib.load(rev_path)
            cls._models["forecast_orders_model"] = joblib.load(ord_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                cls._metadata["forecast"] = json.load(f)
            logger.info("Loaded Forecasting Models into registry cache.")

        return cls._models["forecast_revenue_model"], cls._models["forecast_orders_model"], cls._metadata["forecast"]
