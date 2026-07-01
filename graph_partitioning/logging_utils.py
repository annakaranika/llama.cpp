"""Logging utilities for graph partitioning and distributed inference simulation."""

import datetime
import logging
import os
from typing import Optional

LOGGERS = {}


def setup_output_dir(parent_dir: str) -> str:
    """Setup output directory for simulation results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{parent_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logger(
    name: str = "llm_iot_sim",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Setup logger for LLM IoT simulation.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        console: Whether to also log to console

    Returns:
        Configured logger instance
    """

    if name in LOGGERS:
        print(f"Logger '{name}' already exists. Reusing the existing logger.")
        return LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        print(f"Setting up console logging for logger: {name}")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        logger.info("Logging to file: %s", log_file)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    LOGGERS[name] = logger

    return logger


def get_logger(name: str = "llm_iot_sim") -> logging.Logger:
    """
    Retrieve a logger by name, setting it up with default parameters if not already configured.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name not in LOGGERS:
        setup_logger(name)
    return LOGGERS[name]


def make_json_safe(obj):
    """Convert an object to a JSON-serializable format."""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                k = "_".join(map(str, k))  # convert tuple to "a_b"
            else:
                k = str(k)
            new_obj[k] = make_json_safe(v)
        return new_obj
    if isinstance(obj, (list, set, tuple)):
        if all(isinstance(x, (int, float, str, bool, type(None))) for x in obj):
            return ",".join(map(str, obj))
        return [make_json_safe(v) for v in obj]
    return obj
