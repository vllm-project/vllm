"""Utilities for selecting and loading models."""
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.adapters import (as_classification_model,
                                                 as_embedding_model,
                                                 as_reward_model)


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
    mixtral_supported = [
        "fp8", "compressed-tensors", "gptq_marlin", "awq_marlin"
    ]

    if (model_config.quantization is not None
            and model_config.quantization not in mixtral_supported
            and "MixtralForCausalLM" in architectures):
        architectures = ["QuantMixtralForCausalLM"]

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


@dataclass
class ParamMapping:
    """
    A class to handle parameter mapping for model weight loading.
    It creates a bidirectional mapping between packed parameters and their 
    constituent parts.
    """
    packed_mapping: Dict[str, List[str]]
    inverse_packed_mapping: Dict[str, Tuple[str,
                                            int]] = field(default_factory=dict)

    def __post_init__(self):
        for packed_name, sub_params in self.packed_mapping.items():
            # Skip self-contained cases (e.g., {"W_pack": ["W_pack"]})
            if len(sub_params) == 1 and sub_params[0] == packed_name:
                continue
            for index, param_name in enumerate(sub_params):
                self.inverse_packed_mapping[param_name] = (
                    packed_name,
                    index,
                )
