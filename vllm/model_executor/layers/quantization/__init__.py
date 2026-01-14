# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, get_args

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

logger = init_logger(__name__)

QuantizationMethods = Literal[
    "awq",
    "fp8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "bitblas",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
    "gptq_bitblas",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "ipex",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "mxfp8",
    "petit_nvfp4",
    "cpu_awq",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

DEPRECATED_QUANTIZATION_METHODS = [
    "tpu_int8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "fp_quant",
    "bitblas",
    "gptq_marlin_24",
    "gptq_bitblas",
    "experts_int8",
    "ipex",
    "petit_nvfp4",
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
        >>> from vllm.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
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
            logger.warning(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    from .awq import AWQConfig
    from .awq_marlin import AWQMarlinConfig
    from .bitblas import BitBLASConfig
    from .bitsandbytes import BitsAndBytesConfig
    from .compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from .cpu_wna16 import CPUAWQConfig
    from .experts_int8 import ExpertsInt8Config
    from .fbgemm_fp8 import FBGEMMFp8Config
    from .fp8 import Fp8Config
    from .fp_quant import FPQuantConfig
    from .gguf import GGUFConfig
    from .gptq import GPTQConfig
    from .gptq_bitblas import GPTQBitBLASConfig
    from .gptq_marlin import GPTQMarlinConfig
    from .gptq_marlin_24 import GPTQMarlin24Config
    from .inc import INCConfig
    from .ipex_quant import IPEXConfig
    from .modelopt import ModelOptFp8Config, ModelOptNvFp4Config
    from .moe_wna16 import MoeWNA16Config
    from .mxfp4 import Mxfp4Config
    from .mxfp8 import Mxfp8Config
    from .petit import PetitNvFp4Config
    from .ptpc_fp8 import PTPCFp8Config
    from .torchao import TorchAOConfig

    method_to_config: dict[str, type[QuantizationConfig]] = {
        "awq": AWQConfig,
        "fp8": Fp8Config,
        "fbgemm_fp8": FBGEMMFp8Config,
        "fp_quant": FPQuantConfig,
        "modelopt": ModelOptFp8Config,
        "modelopt_fp4": ModelOptNvFp4Config,
        "bitblas": BitBLASConfig,
        "gguf": GGUFConfig,
        "gptq_marlin_24": GPTQMarlin24Config,
        "gptq_marlin": GPTQMarlinConfig,
        "gptq_bitblas": GPTQBitBLASConfig,
        "awq_marlin": AWQMarlinConfig,
        "gptq": GPTQConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "ptpc_fp8": PTPCFp8Config,
        "experts_int8": ExpertsInt8Config,
        "ipex": IPEXConfig,
        "quark": QuarkConfig,
        "moe_wna16": MoeWNA16Config,
        "torchao": TorchAOConfig,
        "auto-round": INCConfig,
        "inc": INCConfig,
        "mxfp4": Mxfp4Config,
        "mxfp8": Mxfp8Config,
        "petit_nvfp4": PetitNvFp4Config,
        "cpu_awq": CPUAWQConfig,
    }
    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]
