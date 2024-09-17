from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.platforms import current_platform


def is_quant_method_supported(quant_method: str) -> bool:
    # Currently, all quantization methods require Nvidia or AMD GPUs
    capability = current_platform.get_device_capability()
    if capability is None:
        return False

    capability = capability[0] * 10 + capability[1]
    return (capability >=
            QUANTIZATION_METHODS[quant_method].get_min_capability())
