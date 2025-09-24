import logging
import os

def setup_logging(
    log_file: str = "app.log",
    log_level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
):
    """
    Sets up logging to file and console.

    Args:
        log_file (str): Path to the log file.
        log_level (int): Logging level.
        log_format (str): Format for log messages.
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )