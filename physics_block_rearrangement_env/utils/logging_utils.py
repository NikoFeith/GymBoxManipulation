# physics_block_rearrangement_env/utils/logging_utils.py

import logging
import sys

# Define standard logging levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LEVEL = logging.INFO # Default level if not specified

def setup_logger(name: str, level: int = DEFAULT_LEVEL, log_file: str = None, log_format: str = DEFAULT_FORMAT):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name for the logger (e.g., __name__ for the module).
        level (int): The minimum logging level to output (e.g., logging.DEBUG, logging.INFO).
        log_file (str, optional): Path to a file to output logs. If None, only console output. Defaults to None.
        log_format (str, optional): The format string for log messages. Defaults to DEFAULT_FORMAT.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    # Prevent duplicate handlers if called multiple times for the same logger name
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level) # Set the logger's threshold

    formatter = logging.Formatter(log_format)

    # --- Console Handler ---
    # Always add a console handler
    console_handler = logging.StreamHandler(sys.stdout) # Use stdout for info/debug like print
    console_handler.setLevel(level) # Handler level can be same or higher than logger level
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler (Optional) ---
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a') # Append mode
            file_handler.setLevel(level) # Set level for file handler too
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging configured. Console level: {logging.getLevelName(level)}, File output: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file handler for {log_file}: {e}")
    else:
         logger.info(f"Logging configured. Console level: {logging.getLevelName(level)}, No file output.")


    # Prevent logs from propagating to the root logger if it has handlers
    # logger.propagate = False

    return logger

# Example of getting a level from a string (useful for config files)
def get_level_from_string(level_str: str) -> int:
    """Converts a log level string to a logging level constant."""
    return LOG_LEVELS.get(level_str.lower(), DEFAULT_LEVEL)