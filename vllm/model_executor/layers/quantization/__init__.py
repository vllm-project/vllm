from typing import Dict, Type

from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.deepspeedfp import (
    DeepSpeedFPConfig)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.fp8_rocm import Fp8RocmConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQMarlin24Config)
from vllm.model_executor.layers.quantization.marlin import MarlinConfig
from vllm.model_executor.layers.quantization.squeezellm import SqueezeLLMConfig
from vllm.utils import is_hip

QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "fp8": Fp8Config if not is_hip() else Fp8RocmConfig,
    "gptq": GPTQConfig,
    "squeezellm": SqueezeLLMConfig,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return QUANTIZATION_METHODS[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
