
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import triton
    HAS_TRITON = True
except ImportError:
    logger.info("Triton not installed; certain GPU-related functions"
                " will be not be available.")
    HAS_TRITON = False
