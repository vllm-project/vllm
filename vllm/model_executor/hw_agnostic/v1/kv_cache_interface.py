# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export of KV-cache spec types from the V1 framework.

The hw-agnostic tree builds ``MLAAttentionSpec`` / ``SlidingWindowMLASpec``
/ ``AttentionSpec`` instances that the framework reads back to discover
KV-cache groups; identity must be preserved, so this shim re-exports
the same classes rather than vendoring.

The isolation lint forbids reaching into ``vllm.v1.kv_cache_interface``
directly; this shim is the single sanctioned entry point.
"""

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

__all__ = [
    "AttentionSpec",
    "KVCacheSpec",
    "MLAAttentionSpec",
    "SlidingWindowMLASpec",
]
