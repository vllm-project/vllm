from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.platforms import current_platform


def is_quant_method_supported(quant_method: str) -> bool:
    # Currently, all quantization methods require Nvidia or AMD GPUs
    if not (current_platform.is_cuda() or current_platform.is_rocm()):
        return False

    capability = current_platform.get_device_capability()
    assert capability is not None

    min_capability = QUANTIZATION_METHODS[quant_method].get_min_capability()

    return capability.to_int() >= min_capability
