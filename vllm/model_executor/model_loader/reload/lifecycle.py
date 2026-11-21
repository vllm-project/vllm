# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Begin, complete or abort a checkpoint-format weight update in the mode
selected at model load."""

import torch

from vllm.config import ModelConfig

from .direct import (
    abort_direct_reload,
    begin_direct_reload,
    finish_direct_reload,
    reloads_directly,
)
from .layerwise import (
    abort_layerwise_reload,
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)

__all__ = ["abort_reload", "finish_reload", "start_reload", "weight_update_active"]


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


def abort_reload(model: torch.nn.Module) -> None:
    """Undo a weight update `start_reload` began that will not finish.

    Layerwise reload restores layers still loading. Direct reload keeps any
    writes already made. A full resend brings all weights to the same version.
    """
    global _weight_update_active
    _weight_update_active = False
    if reloads_directly(model):
        abort_direct_reload(model)
    else:
        abort_layerwise_reload(model)
