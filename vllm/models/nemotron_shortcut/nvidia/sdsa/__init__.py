# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SDSA: Nemotron Shortcut's sparse attention layer."""

from vllm.models.nemotron_shortcut.nvidia.sdsa.attention import (
    SdsaAttention,
    SdsaBuffers,
    allocate_sdsa_buffers,
)
from vllm.models.nemotron_shortcut.nvidia.sdsa.flashinfer_ops import check_requirements
from vllm.models.nemotron_shortcut.nvidia.sdsa.kv_cache_grouping import (
    SdsaKVCacheGrouping,
)
from vllm.models.nemotron_shortcut.nvidia.sdsa.prefill import (
    INDEXER_FORMATS,
    SdsaConfig,
    get_sdsa_config,
)

__all__ = [
    "INDEXER_FORMATS",
    "SdsaAttention",
    "SdsaBuffers",
    "SdsaConfig",
    "SdsaKVCacheGrouping",
    "allocate_sdsa_buffers",
    "check_requirements",
    "get_sdsa_config",
]
