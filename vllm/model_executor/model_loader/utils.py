"""Utilities for selecting and loading models."""
import contextlib
from typing import Optional, Tuple, Type

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.adapters import as_embedding_model


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def get_model_architecture(
    model_config: ModelConfig,
    *,
    architectures: Optional[list[str]] = None,
) -> Tuple[Type[nn.Module], str]:
    if architectures is None:
        architectures = getattr(model_config.hf_config, "architectures", [])

    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = [
        "fp8", "compressed-tensors", "gptq_marlin", "awq_marlin"
    ]

    if (model_config.quantization is not None
            and model_config.quantization not in mixtral_supported
            and "MixtralForCausalLM" in architectures):
        architectures = ["QuantMixtralForCausalLM"]

    model_cls, arch = ModelRegistry.resolve_model_cls(architectures)
    if model_config.task == "embedding":
        model_cls = as_embedding_model(model_cls)

    return model_cls, arch


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]
