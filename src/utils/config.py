"""Configuration loader for Customer Analytics Platform."""

from pathlib import Path
from typing import Any, Dict
import yaml

CONFIG_DIR = Path("configs")


def load_yaml(filepath: str | Path) -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        filepath: Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config() -> Dict[str, Any]:
    """Load main project configuration."""
    return load_yaml(CONFIG_DIR / "config.yaml")


def get_paths() -> Dict[str, Any]:
    """Load project directory and artifact paths."""
    return load_yaml(CONFIG_DIR / "paths.yaml")


def get_model_params() -> Dict[str, Any]:
    """Load model parameters and hyperparameters."""
    return load_yaml(CONFIG_DIR / "model_params.yaml")
