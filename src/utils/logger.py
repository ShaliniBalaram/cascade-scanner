"""Logging configuration using Loguru."""

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import get_project_root, settings


def setup_logging() -> None:
    logger.remove()

    log_dir = get_project_root() / "logs"
    log_dir.mkdir(exist_ok=True)

    # Console
    logger.add(
        sys.stderr,
        format=settings.logging.format,
        level=settings.logging.level,
        colorize=True,
    )

    # File
    logger.add(
        log_dir / "cascade_scanner.log",
        format=settings.logging.format,
        level=settings.logging.level,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression="zip",
    )

    # Errors only
    logger.add(
        log_dir / "errors.log",
        format=settings.logging.format,
        level="ERROR",
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression="zip",
    )

    logger.info(f"Logging initialized - Level: {settings.logging.level}")


def get_logger(name: str):
    return logger.bind(name=name)


setup_logging()
