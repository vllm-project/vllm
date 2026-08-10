# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layerwise weight reloading utilities for vLLM.

This module provides functionality to reload model weights layer-by-layer,
which is useful for weight updates without full model reconstruction.

Limitations:
1. Composition with CPU offloading has not been implemented
2. Tied parameters will only reflect processing from one of the parent layers (for
   example, only processing from embed_tokens will have an effect)
3. Strict reload requires the initial checkpoint and runtime updates to use the same
   canonical loader-application schema.
"""

__all__ = [
    "record_metadata_for_reloading",
    "freeze_load_plan",
    "initialize_layerwise_reload",
    "validate_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
    "set_torchao_reload_attrs",
    "support_quantized_model_reload_from_hp_weights",
    "LoadProbeError",
    "LoadProbeReport",
    "probe_model_load",
]

from .layerwise import (
    finalize_layerwise_processing,
    finalize_layerwise_reload,
    freeze_load_plan,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
    validate_layerwise_reload,
)
from .probe import LoadProbeError, LoadProbeReport, probe_model_load
from .torchao_decorator import (
    set_torchao_reload_attrs,
    support_quantized_model_reload_from_hp_weights,
)
