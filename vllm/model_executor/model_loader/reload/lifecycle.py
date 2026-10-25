# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

import torch
from torch import nn

import vllm.envs as envs

from .direct import direct_finish, direct_start
from .layerwise import finalize_layerwise_reload, initialize_layerwise_reload

__all__ = ["ReloadMode", "finish_reload", "start_reload"]

ReloadMode = Literal["layerwise", "direct"]

_MODE_ATTR = "_reload_mode"


def start_reload(model: nn.Module, reload_mode: ReloadMode | None = None) -> None:
    """Begin a weight update; `finish_reload` completes it.

    Args:
        model: The model to update.
        reload_mode: Defaults to `VLLM_WEIGHT_RELOAD_MODE`.

    """
    reload_mode = reload_mode or envs.VLLM_WEIGHT_RELOAD_MODE
    if reload_mode == "direct":
        direct_start(model)
    elif reload_mode == "layerwise":
        initialize_layerwise_reload(model)
    else:
        raise ValueError(f"Unknown reload_mode {reload_mode!r}")
    setattr(model, _MODE_ATTR, reload_mode)


def finish_reload(
    model: nn.Module, model_config, device: torch.device | None = None
) -> None:
    """Complete the weight update in the mode `start_reload` began."""
    reload_mode = model.__dict__.pop(_MODE_ATTR, "layerwise")
    if reload_mode == "direct":
        direct_finish(model, model_config, device)
    else:
        finalize_layerwise_reload(model, model_config)
