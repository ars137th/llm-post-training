"""
Logging utilities for training and evaluation.

Provides simple logging setup for experiments.
"""

import logging
import sys
from typing import Optional
from omegaconf import DictConfig


def setup_logging(
    cfg: Optional[DictConfig] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging for training/evaluation.

    Args:
        cfg: Optional Hydra config (can contain logging settings)
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs

    Returns:
        Configured logger

    Example:
        logger = setup_logging(cfg)
        logger.info("Training started")
    """
    # Get or create logger
    logger = logging.getLogger("llm-post-training")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    # If config is provided, check for log file setting
    if cfg is not None and hasattr(cfg, 'logging'):
        log_cfg = cfg.logging
        if hasattr(log_cfg, 'log_file') and log_cfg.log_file:
            file_handler = logging.FileHandler(log_cfg.log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_cfg.log_file}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Optional logger name (default: "llm-post-training")

    Returns:
        Logger instance
    """
    if name is None:
        name = "llm-post-training"
    return logging.getLogger(name)


if __name__ == "__main__":
    # Test logging setup
    print("Testing logging setup...\n")

    logger = setup_logging()
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    print("\n✓ Logging setup working!")
