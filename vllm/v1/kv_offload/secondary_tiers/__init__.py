# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Secondary tier implementations for KV cache offloading.

This package contains various secondary tier storage backends that can be used
with the TieringOffloadingManager for multi-tier KV cache management.
"""

from vllm.v1.kv_offload.secondary_tiers.dummy import (
    DummyLoadStoreSpec,
    DummySecondaryTier,
)

__all__ = [
    "DummyLoadStoreSpec",
    "DummySecondaryTier",
]
