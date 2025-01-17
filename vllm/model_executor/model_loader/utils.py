"""Utilities for selecting and loading models."""
import contextlib
from typing import Tuple, Type

import torch
import transformers
from torch import nn

from vllm.config import ModelConfig, ModelImpl
from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.adapters import (as_classification_model,
                                                 as_embedding_model,
                                                 as_reward_model)

logger = init_logger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def is_transformers_impl_compatible(arch: str) -> bool:
    return getattr(transformers, arch)._supports_flex_attn


def get_model_architecture(
        model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
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

    vllm_supported_archs = ModelRegistry.get_supported_archs()
    for i, arch in enumerate(architectures):
        if model_config.model_impl == ModelImpl.TRANSFORMERS:
            if not is_transformers_impl_compatible:
                raise ValueError(
                    "The Transformers implementation of %s is not compatible "
                    "with vLLM.", arch)
            architectures[i] = "TransformersModel"
        if (model_config.model_impl == ModelImpl.AUTO
                and arch not in vllm_supported_archs):
            if not is_transformers_impl_compatible:
                raise ValueError(
                    "%s has no vLLM implementation and the Transformers "
                    "implementationis not compatible with vLLM.", arch)
            logger.info(
                "%s has no vLLM implementation, falling back to use "
                "Transformers implementation", arch)
            architectures[i] = "TransformersModel"

    model_cls, arch = ModelRegistry.resolve_model_cls(architectures)
    if model_config.task == "embed":
        model_cls = as_embedding_model(model_cls)
    elif model_config.task == "classify":
        model_cls = as_classification_model(model_cls)
    elif model_config.task == "reward":
        model_cls = as_reward_model(model_cls)

    return model_cls, arch


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]
