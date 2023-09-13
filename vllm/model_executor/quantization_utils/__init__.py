from vllm.model_executor.quantization_utils.awq import AWQConfig
from vllm.model_executor.quantization_utils.config import QuantizationConfig

_QUANTIZATION_REGISTRY = {
    "awq": AWQConfig,
}


def get_quant_config(quantization: str) -> QuantizationConfig:
    if quantization not in _QUANTIZATION_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_REGISTRY[quantization]
