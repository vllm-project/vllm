from typing import Dict, List, Type

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

QUANTIZATION_METHODS: List[str] = [
    "aqlm",
    "awq",
    "deepspeedfp",
    "tpu_int8",
    "fp8",
    "fbgemm_fp8",
    "modelopt",
    # The order of gptq methods is important for config.py iteration over
    # override_quantization_method(..)
    "marlin",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "qqq",
    "hqq",
    "experts_int8",
    "neuron_quant",
    "ipex",
    "quark"
]


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    from .aqlm import AQLMConfig
    from .awq import AWQConfig
    from .awq_marlin import AWQMarlinConfig
    from .bitsandbytes import BitsAndBytesConfig
    from .compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig)
    from .deepspeedfp import DeepSpeedFPConfig
    from .experts_int8 import ExpertsInt8Config
    from .fbgemm_fp8 import FBGEMMFp8Config
    from .fp8 import Fp8Config
    from .gguf import GGUFConfig
    from .gptq import GPTQConfig
    from .gptq_marlin import GPTQMarlinConfig
    from .gptq_marlin_24 import GPTQMarlin24Config
    from .hqq_marlin import HQQMarlinConfig
    from .ipex_quant import IPEXConfig
    from .marlin import MarlinConfig
    from .modelopt import ModelOptFp8Config
    from .neuron_quant import NeuronQuantConfig
    from .qqq import QQQConfig
    from .tpu_int8 import Int8TpuConfig

    method_to_config: Dict[str, Type[QuantizationConfig]] = {
        "aqlm": AQLMConfig,
        "awq": AWQConfig,
        "deepspeedfp": DeepSpeedFPConfig,
        "tpu_int8": Int8TpuConfig,
        "fp8": Fp8Config,
        "fbgemm_fp8": FBGEMMFp8Config,
        "modelopt": ModelOptFp8Config,
        # The order of gptq methods is important for config.py iteration over
        # override_quantization_method(..)
        "marlin": MarlinConfig,
        "gguf": GGUFConfig,
        "gptq_marlin_24": GPTQMarlin24Config,
        "gptq_marlin": GPTQMarlinConfig,
        "awq_marlin": AWQMarlinConfig,
        "gptq": GPTQConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "qqq": QQQConfig,
        "hqq": HQQMarlinConfig,
        "experts_int8": ExpertsInt8Config,
        "neuron_quant": NeuronQuantConfig,
        "ipex": IPEXConfig,
        "quark": QuarkConfig
    }

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
