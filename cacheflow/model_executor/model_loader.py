"""Utilities for selecting and loading models."""
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from cacheflow.config import ModelConfig
from cacheflow.model_executor.models import (
    GPT2LMHeadModel, GPTNeoXForCausalLM, LlamaForCausalLM, OPTForCausalLM)
from cacheflow.model_executor.weight_utils import initialize_dummy_weights

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
}


def _get_model_architecture(config: PretrainedConfig) -> nn.Module:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}"
    )


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)
    torch.set_default_dtype(model_config.dtype)

    # Create a model instance.
    # The weights will be initialized as empty tensors.
    model = model_class(model_config.hf_config)
    if model_config.use_dummy_weights:
        model = model.cuda()
        # NOTE(woosuk): For accurate performance evaluation, we assign
        # random values to the weights.
        initialize_dummy_weights(model)
    else:
        # Load the weights from the cached or downloaded files.
        model.load_weights(
            model_config.model, model_config.download_dir,
            model_config.use_np_weights)
        model = model.cuda()
    return model.eval()
