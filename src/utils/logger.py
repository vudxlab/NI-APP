"""
Logging configuration for the NI DAQ application.

This module provides centralized logging setup with file and console handlers,
log rotation, and configurable log levels.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ni_daq_app",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up application logger with file and console handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ~/.local/share/ni-daq-app/logs on Linux)
        log_to_file: Enable file logging
        log_to_console: Enable console logging
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler with rotation
    if log_to_file:
        if log_dir is None:
            # Default log directory
            if os.name == 'nt':  # Windows
                log_dir = os.path.join(os.getenv('APPDATA', ''), 'NIDAQApp', 'logs')
            else:  # Linux/Mac
                log_dir = os.path.join(
                    os.path.expanduser('~'),
                    '.local', 'share', 'ni-daq-app', 'logs'
                )

        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        log_file = os.path.join(log_dir, f'{name}.log')

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    logger.info(f"Logger '{name}' initialized with level {logging.getLevelName(level)}")

    return logger


def get_logger(name: str = "ni_daq_app") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


def set_log_level(level: int, logger_name: str = "ni_daq_app") -> None:
    """
    Change the logging level for an existing logger.

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Name of the logger to modify
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Update console handler level (file handler stays at DEBUG)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(level)

    logger.info(f"Log level changed to {logging.getLevelName(level)}")


# Example usage
if __name__ == "__main__":
    # Test the logger
    logger = setup_logger("test_logger", level=logging.DEBUG)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test level change
    set_log_level(logging.WARNING, "test_logger")
    logger.info("This info message should not appear in console")
    logger.warning("But this warning should appear")
