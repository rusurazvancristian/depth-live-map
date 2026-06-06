import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    """Configures structured, clean logging for the pipeline execution.

    Args:
        level: Minimum log level to print.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Standard formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Avoid duplicate handlers if setup is called multiple times
    if not root.handlers:
        root.addHandler(handler)
