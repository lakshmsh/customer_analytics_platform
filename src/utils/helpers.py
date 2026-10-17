"""General helper functions for data manipulation and performance tracking."""

import functools
import time
from typing import Callable, Any
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("helpers")


def timer(func: Callable) -> Callable:
    """Decorator to measure and log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        logger.info(f"Starting execution of {func.__name__}...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"Finished {func.__name__} in {elapsed:.2f} seconds.")
        return result
    return wrapper


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types to optimize memory consumption.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Memory-optimized DataFrame.
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    df_opt = df.copy()

    for col in df_opt.select_dtypes(include=["int64", "int32"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")

    for col in df_opt.select_dtypes(include=["float64"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")

    end_mem = df_opt.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.debug(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction).")
    return df_opt
