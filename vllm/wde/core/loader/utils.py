"""Utilities for selecting and loading models."""
import contextlib
from typing import Tuple, Type

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.wde.core.config import ModelConfig
from vllm.wde.core.modelzoo import ModelRegistry


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

    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model_workflow(hf_config: PretrainedConfig) -> str:
    architectures = getattr(hf_config, "architectures", [])

    for arch in architectures:
        workflow = ModelRegistry.get_workflow(arch)
        if workflow is not None:
            return workflow
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]
