from vllm.logger import init_logger
from vllm.platforms import current_platform

from .punica_base import PunicaWrapperBase

logger = init_logger(__name__)


def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    if current_platform.is_cuda_alike():
        # Lazy import to avoid ImportError
        from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
        logger.info_once("Using PunicaWrapperGPU.")
        return PunicaWrapperGPU(*args, **kwargs)
    elif current_platform.is_cpu():
        # Lazy import to avoid ImportError
        from vllm.lora.punica_wrapper.punica_cpu import PunicaWrapperCPU
        logger.info_once("Using PunicaWrapperCPU.")
        return PunicaWrapperCPU(*args, **kwargs)
    elif current_platform.is_hpu():
        # Lazy import to avoid ImportError
        from vllm.lora.punica_wrapper.punica_hpu import PunicaWrapperHPU
        logger.info_once("Using PunicaWrapperHPU.")
        return PunicaWrapperHPU(*args, **kwargs)
    else:
        raise NotImplementedError
