import logging

def setup_logging(level=logging.INFO):
    logger = logging.getLogger("dllm")
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
