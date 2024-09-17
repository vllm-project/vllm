from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.platforms import current_platform


def is_quant_method_supported(quant_method: str) -> bool:
    # Currently, all quantization methods require Nvidia or AMD GPUs
    capability = int(current_platform.get_device_capability() or -1)
    min_capability = QUANTIZATION_METHODS[quant_method].get_min_capability()

    return capability >= min_capability
