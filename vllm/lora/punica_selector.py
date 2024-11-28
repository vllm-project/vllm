from vllm.lora.punica_base import PunicaWrapperBase
from vllm.platforms import current_platform


def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    if current_platform.is_cuda() or current_platform.is_rocm():
        from vllm.lora.punica_gpu import PunicaWrapperGPU

        return PunicaWrapperGPU(*args, **kwargs)
    else:
        raise NotImplementedError
