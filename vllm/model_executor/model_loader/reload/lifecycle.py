# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Begin and complete a checkpoint-format weight update in the configured
reload mode (``VLLM_RELOAD_MODE``)."""

import torch

import vllm.envs as envs
from vllm.config import ModelConfig

from .direct import begin_direct_reload, finish_direct_reload
from .layerwise import finalize_layerwise_reload, initialize_layerwise_reload

__all__ = ["finish_reload", "start_reload"]

_MODE_ATTR = "_reload_mode"


def start_reload(model: torch.nn.Module) -> None:
    """Begin a weight update; `finish_reload` completes it in the same mode."""
    mode = envs.VLLM_RELOAD_MODE
    if mode == "direct":
        begin_direct_reload(model)
    else:
        initialize_layerwise_reload(model)
    setattr(model, _MODE_ATTR, mode)


def finish_reload(
    model: torch.nn.Module, model_config: ModelConfig, device: torch.device
) -> None:
    """Complete the weight update in the mode `start_reload` began."""
    if model.__dict__.pop(_MODE_ATTR, "layerwise") == "direct":
        finish_direct_reload(model, model_config, device)
    else:
        finalize_layerwise_reload(model, model_config)
