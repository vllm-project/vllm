import logging


def make_logger(name: str) -> logging.Logger:
    """Create a base logger"""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def log_banner(logger: logging.Logger,
               label: str,
               body: str,
               level: int = logging.INFO):
    """
    Log a message in the "banner"-style format.

    :param logger: Instance of "logging.Logger" to use
    :param label: Label for the top of the banner
    :param body: Body content inside the banner
    :param level: Logging level to use (default: INFO)
    """

    banner = f"==== {label} ====\n{body}\n===="
    logger.log(level, "\n%s", banner)
