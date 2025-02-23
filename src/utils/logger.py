from functools import lru_cache
from logging import Logger
from .logger_config import setup_logger


@lru_cache()
def get_logger(name: str) -> Logger:
    """Get or create a logger with the given name"""
    return setup_logger(name)