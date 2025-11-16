# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for integrating custom model implementations with vLLM."""

from vllm.model_executor.custom_models.utils import (
    convert_freqs_cis_to_real,
    create_mla_kv_cache_spec,
    load_external_weights,
    store_positions_in_context,
)

__all__ = [
    "convert_freqs_cis_to_real",
    "create_mla_kv_cache_spec",
    "load_external_weights",
    "store_positions_in_context",
]
