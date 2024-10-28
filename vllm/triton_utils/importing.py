from importlib.util import find_spec

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

HAS_TRITON = (
    find_spec("triton") is not None
    and not current_platform.is_xpu()  # Not compatible
    and not current_platform.is_neuron()  # neuron has too old torch
)

if not HAS_TRITON:
    logger.info("Triton not installed or not compatible; certain GPU-related"
                " functions will not be available.")
