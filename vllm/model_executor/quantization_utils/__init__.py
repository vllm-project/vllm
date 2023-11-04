from typing import Type

from vllm.model_executor.quantization_utils.awq import AWQConfig
from vllm.model_executor.quantization_utils.base import QuantizationConfig
from vllm.model_executor.quantization_utils.squeezellm import SqueezeLLMConfig

_QUANTIZATION_REGISTRY = {
    "awq": AWQConfig,
    "squeezellm": SqueezeLLMConfig,
}


def get_quant_class(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quant_class",
]
