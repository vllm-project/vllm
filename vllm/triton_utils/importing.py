from importlib import import_module

from vllm.logger import init_logger

logger = init_logger(__name__)

HAS_WARNED = False


def maybe_import_triton():

    global HAS_WARNED

    try:
        triton = import_module("triton")
        tl = import_module("triton.language")
        return triton, tl
    except ImportError:
        if not HAS_WARNED:
            logger.info("Triton not installed; certain GPU-related functions"
                        " will be not be available.")
            HAS_WARNED = True

        mock_triton = import_module("vllm.triton_utils.mock_triton")
        mock_tl = import_module("vllm.triton_utils.mock_tl")
        return mock_triton, mock_tl
