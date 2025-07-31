"""
Logging utilities
"""
import logging
import sys


def configure_logging(level: int = logging.INFO):
    """Configure root logger with basic settings"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
