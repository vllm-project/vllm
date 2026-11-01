# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Begin, complete or abort a checkpoint-format weight update in the reload mode
``VLLM_RELOAD_MODE`` selects (fixed for the process)."""

import torch

from vllm.config import ModelConfig

from .direct import begin_direct_reload, direct_reload_enabled, finish_direct_reload
from .layerwise import (
    abort_layerwise_reload,
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)

__all__ = ["abort_reload", "finish_reload", "start_reload"]


def start_reload(model: torch.nn.Module) -> None:
    """Begin a weight update in the configured mode."""
    if direct_reload_enabled():
        begin_direct_reload(model)
    else:
        initialize_layerwise_reload(model)


def finish_reload(
    model: torch.nn.Module, model_config: ModelConfig, device: torch.device
) -> None:
    """Complete the weight update `start_reload` began."""
    if direct_reload_enabled():
        finish_direct_reload(model, model_config, device)
    else:
        finalize_layerwise_reload(model, model_config)


def abort_reload(model: torch.nn.Module) -> None:
    """Undo a weight update `start_reload` began that will not finish.

    Layerwise reload puts back the weights it started from. Direct reload has
    no rollback: the model keeps the partial update until a new full update
    finishes.
    """
    if not direct_reload_enabled():
        abort_layerwise_reload(model)
