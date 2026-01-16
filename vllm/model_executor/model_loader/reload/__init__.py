# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layerwise weight reloading utilities for vLLM.

This module provides functionality to reload model weights layer-by-layer,
which is useful for weight updates without full model reconstruction.

Limitations:
    - Does not compose with CPU offloading. This is because `device_loading_context`
      doesn't work in all cases (e.g., when parameter is renamed).
    - Does not handle layers where only some weight elements are loaded, but some
      weights aren't. For example, only loading q_scale, but not k_scale or v_scale
    - Unties weights during loading, but not on cuda graph

TODO:
    - Decide on reloading interface, back-compat with reload_weights
    - Do Attention/MLA processing
    - Check composability with EPLB
"""

__all__ = [
    "supports_reloading",
    "finalize_layerwise_restore_and_process",
    "layerwise_restore_and_process",
    "record_metadata_for_reloading",
]

from .decorator import supports_reloading
from .layerwise import (
    finalize_layerwise_restore_and_process,
    layerwise_restore_and_process,
    record_metadata_for_reloading,
)
