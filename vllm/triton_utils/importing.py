from importlib.util import find_spec

from vllm.logger import init_logger

logger = init_logger(__name__)

HAS_TRITON = find_spec("triton") is not None

if not HAS_TRITON:
    logger.info("Triton not installed; certain GPU-related functions"
                " will be not be available.")
