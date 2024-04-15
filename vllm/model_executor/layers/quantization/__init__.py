from typing import Type

from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.marlin import MarlinConfig
from vllm.model_executor.layers.quantization.squeezellm import SqueezeLLMConfig

_QUANTIZATION_CONFIG_REGISTRY = {
    "awq": 1,
    "gptq": 2,
    "squeezellm": 3,
    "marlin": 4,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "_QUANTIZATION_CONFIG_REGISTRY",
]
