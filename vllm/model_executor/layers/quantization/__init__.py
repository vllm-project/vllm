from typing import Dict, List, Type

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

QUANTIZATION_METHODS: List[str] = [
    "aqlm",
    "awq",
    "awq_hpu",
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
    "gptq_hpu",
    "compressed-tensors",
    "bitsandbytes",
    "qqq",
    "hqq",
    "experts_int8",
    "neuron_quant",
    "ipex",
    "inc",
    "quark"
]

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import register_quantization_config
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            raise ValueError(
                f"The quantization method `{quantization}` is already exists.")
        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError("The quantization config must be a subclass of "
                             "`QuantizationConfig`.")
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        QUANTIZATION_METHODS.append(quantization)
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early
    from vllm_hpu_extension.awq_hpu import AWQHPUConfig
    from vllm_hpu_extension.gptq_hpu import GPTQHPUConfig

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
    from .inc import INCConfig
    from .ipex_quant import IPEXConfig
    from .marlin import MarlinConfig
    from .modelopt import ModelOptFp8Config
    from .neuron_quant import NeuronQuantConfig
    from .qqq import QQQConfig
    from .tpu_int8 import Int8TpuConfig

    method_to_config: Dict[str, Type[QuantizationConfig]] = {
        "aqlm": AQLMConfig,
        "awq": AWQConfig,
        "awq_hpu": AWQHPUConfig,
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
        "gptq_hpu": GPTQHPUConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "qqq": QQQConfig,
        "hqq": HQQMarlinConfig,
        "experts_int8": ExpertsInt8Config,
        "neuron_quant": NeuronQuantConfig,
        "ipex": IPEXConfig,
        "inc": INCConfig,
        "quark": QuarkConfig
    }
    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
