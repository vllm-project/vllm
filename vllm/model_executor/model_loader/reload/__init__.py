# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layerwise weight reloading utilities for vLLM.

This module provides functionality to reload model weights layer-by-layer,
which is useful for weight updates without full model reconstruction.

Limitations:
1. Composition with CPU offloading has not been implemented
2. Tied parameters will only reflect processing from one of the parent layers (for
   example, only processing from embed_tokens will have an effect)
3. This design assumes that the number of weights loaded from disk is the same as the
   number of weights created at model init time. This is not true for quant methods
   which (1) pad weights or (2) load qkv weights into the same parameter. Both of these
   cases are non-issues for today's quant methods, but future quantizations may cause
   reloading to fail
"""

__all__ = [
    "record_metadata_for_reloading",
    "start_reload",
    "finish_reload",
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
    "set_torchao_reload_attrs",
    "support_quantized_model_reload_from_hp_weights",
]

import torch

from vllm.config import ModelConfig

from .direct import direct_finish, direct_start
from .layerwise import (
    finalize_layerwise_processing,
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from .torchao_decorator import (
    set_torchao_reload_attrs,
    support_quantized_model_reload_from_hp_weights,
)

# Entry points for the engines that have a `reload_mode`. Everything else names
# the mechanism it wants directly. There is no recovery step: if a reload
# raises, the model is undefined and the engine must be restarted.

_STARTED_MODE = "_reload_started_mode"


def start_reload(model: torch.nn.Module, mode: str = "layerwise") -> None:
    """Prepare ``model`` to receive checkpoint-format weights.

    ``mode`` is latched on the model, so ``finish_reload`` completes whichever
    one was started rather than re-reading a config that may have changed.
    """
    if mode == "direct":
        direct_start(model)
    elif mode == "layerwise":
        initialize_layerwise_reload(model)
    else:
        raise ValueError(f"unknown reload mode {mode!r}")
    model.__dict__[_STARTED_MODE] = mode


def finish_reload(model: torch.nn.Module, model_config: ModelConfig) -> None:
    """Complete the reload that ``start_reload`` began."""
    mode = model.__dict__.pop(_STARTED_MODE, None)
    if mode is None:
        raise RuntimeError("finish_reload called without a matching start_reload")
    if mode == "direct":
        direct_finish(model)
    else:
        finalize_layerwise_reload(model, model_config)
