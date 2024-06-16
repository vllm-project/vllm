import torch

from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS


def is_quant_method_supported(quant_method: str) -> bool:
    # Currently, all quantization methods require Nvidia or AMD GPUs
    if not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability()
    capability = capability[0] * 10 + capability[1]
    return (capability >=
            QUANTIZATION_METHODS[quant_method].get_min_capability())
