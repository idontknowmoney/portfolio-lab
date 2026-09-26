import logging
import os
import sys

PACKAGE_LOGGER = "portfolio_lab"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int | str = "INFO") -> None:
    """
    Configure the package-level logger. Safe to call multiple times.

    The level can be overridden with the PORTFOLIO_LAB_LOG_LEVEL environment variable.

    Parameters:
    level (int | str): Default logging level.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("PORTFOLIO_LAB_LOG_LEVEL", level))
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name, configuring logging on first use.

    Parameters:
    name (str): Usually __name__ of the calling module.

    Returns:
    logging.Logger: The configured logger.
    """
    setup_logging()
    return logging.getLogger(name)
