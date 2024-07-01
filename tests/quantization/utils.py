import torch

from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import get_device_capability_stateless


def is_quant_method_supported(quant_method: str) -> bool:
    # Currently, all quantization methods require Nvidia or AMD GPUs
    if not torch.cuda.is_available():
        return False

    capability = get_device_capability_stateless()
    capability = capability[0] * 10 + capability[1]
    return (capability >=
            QUANTIZATION_METHODS[quant_method].get_min_capability())
