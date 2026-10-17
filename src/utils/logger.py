"""Logging utility for Customer Analytics Platform."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "customer_analytics", log_file: Optional[str] = "monitoring/app.log") -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger.
        log_file: Path to output log file. Defaults to monitoring/app.log.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
