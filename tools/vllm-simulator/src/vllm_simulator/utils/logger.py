import logging
import os


def get_logger(name: str = "vllm_simulator") -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level_str = os.getenv("VLLM_SIMULATOR_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        logger.setLevel(log_level)

        handler = logging.StreamHandler()
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger