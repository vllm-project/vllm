# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layerwise weight reloading utilities for vLLM.

This module provides functionality to reload model weights layer-by-layer,
which is useful for weight updates without full model reconstruction.

Limitations:
    - Composition with CPU offloading has not been implemented
    - Cannot handle layers where only some weight elements are loaded, but some
      weights aren't (for example, only loading up_proj, but not gate_proj)
    - Reloading Attention/MLA weights (q_scale, k_scale, v_scale) has not been
      implemented
    - Tied weights will remain tied, but will only reflect processing from one of the
      parent modules (for example, only processing from lm_head will have an effect)

TODO(@ksayers):
    - Decide on reloading interface, back-compat with reload_weights
    - Check composability with EPLB
"""

__all__ = [
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_reload",
    "support_quantized_model_reload_from_hp_weights",
]

from .layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from .torchao_decorator import support_quantized_model_reload_from_hp_weights
