
from vllm.platforms import current_platform
from .punica_base import PunicaWrapperBase
from functools import lru_cache


@lru_cache(maxsize=None)
def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    if current_platform.is_cuda_alike():
        from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

        return PunicaWrapperGPU(*args, **kwargs)
    else:
        raise NotImplementedError
