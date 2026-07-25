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
3. A loader returning ``None`` or ``bool`` uses the legacy fragment-argument
   adapter. New loaders should return ``LoadReceipt`` with an explicit fragment.
"""

from vllm.model_executor.load_receipt import LoadFragment, LoadReceipt

__all__ = [
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
    "record_load_consumption",
    "finalize_load_recording",
    "set_torchao_reload_attrs",
    "support_quantized_model_reload_from_hp_weights",
    "LoadFragment",
    "LoadReceipt",
]

from .layerwise import (
    finalize_layerwise_processing,
    finalize_layerwise_reload,
    finalize_load_recording,
    initialize_layerwise_reload,
    record_load_consumption,
    record_metadata_for_reloading,
)
from .torchao_decorator import (
    set_torchao_reload_attrs,
    support_quantized_model_reload_from_hp_weights,
)
