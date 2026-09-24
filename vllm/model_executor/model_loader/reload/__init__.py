# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Weight reloading utilities for vLLM.

ModelReloadTracer provides opt-in, fixed-scope checkpoint reload tracing.
The existing layerwise helpers remain the worker's default reload path.

Limitations of the layerwise helpers:
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
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
    "set_torchao_reload_attrs",
    "support_quantized_model_reload_from_hp_weights",
    "ModelReloadTracer",
    "ReloadState",
    "ReloadError",
]

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
from .trace import ModelReloadTracer, ReloadError, ReloadState
