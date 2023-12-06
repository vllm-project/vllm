from typing import Type
import torch
from vllm.model_executor.layers.quantization.squeezellm import SqueezeLLMConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.utils import is_hip

_QUANTIZATION_CONFIG_REGISTRY = {
    "squeezellm": SqueezeLLMConfig,
}

if not is_hip():
    from vllm.model_executor.layers.quantization.awq import AWQConfig
    _QUANTIZATION_CONFIG_REGISTRY["awq"] = AWQConfig


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
]
