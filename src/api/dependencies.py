"""FastAPI dependency injection module for model artifacts and explainers."""

from typing import Optional, Tuple, Any, Dict
from src.models.registry import ModelRegistry
from src.evaluation.explainability import ChurnExplainer
from src.utils.logger import get_logger

logger = get_logger("api.dependencies")

# Singleton explainer instance
_churn_explainer: Optional[ChurnExplainer] = None


def get_churn_explainer() -> ChurnExplainer:
    """Dependency provider for ChurnExplainer."""
    global _churn_explainer
    if _churn_explainer is None:
        logger.info("Initializing singleton ChurnExplainer for API dependency injection...")
        _churn_explainer = ChurnExplainer()
    return _churn_explainer


def get_model_registry() -> type[ModelRegistry]:
    """Dependency provider for ModelRegistry."""
    return ModelRegistry
