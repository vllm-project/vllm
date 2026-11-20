# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Begin and complete a checkpoint-format weight update in the reload mode
the model's cold start fixed."""

import torch

from vllm.config import ModelConfig

from .direct import begin_direct_reload, finish_direct_reload, reloads_directly
from .layerwise import finalize_layerwise_reload, initialize_layerwise_reload

__all__ = ["finish_reload", "start_reload", "weight_update_active"]


_weight_update_active = False


def weight_update_active() -> bool:
    """Whether a weight update is running, when one `load_weights` call may hold
    only part of the checkpoint. Process-wide, so a module nested in the model
    passed to `start_reload` sees it too."""
    return _weight_update_active


def start_reload(model: torch.nn.Module) -> None:
    """Begin a weight update in the configured mode."""
    global _weight_update_active
    if reloads_directly(model):
        begin_direct_reload(model)
    else:
        initialize_layerwise_reload(model)
    _weight_update_active = True


def finish_reload(
    model: torch.nn.Module, model_config: ModelConfig, device: torch.device
) -> None:
    """Complete the weight update `start_reload` began."""
    global _weight_update_active
    _weight_update_active = False
    if reloads_directly(model):
        finish_direct_reload(model, model_config, device)
    else:
        finalize_layerwise_reload(model, model_config)
