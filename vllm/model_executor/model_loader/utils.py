"""Utilities for selecting and loading models."""
import contextlib
from typing import Tuple, Type

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def get_model_architecture(
        model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = ["fp8", "compressed-tensors"]
    # for gptq_marlin, only run fused MoE for int4
    if model_config.quantization == "gptq_marlin":
        hf_quant_config = getattr(model_config.hf_config,
                                  "quantization_config", None)
        if hf_quant_config and hf_quant_config.get("bits") == 4:
            mixtral_supported.append("gptq_marlin")

    if (model_config.quantization is not None
            and model_config.quantization not in mixtral_supported
            and "MixtralForCausalLM" in architectures):
        architectures = ["QuantMixtralForCausalLM"]

    return ModelRegistry.resolve_model_cls(architectures)


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]
