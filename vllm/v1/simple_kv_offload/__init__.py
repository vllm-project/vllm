# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple CPU offloading connector for KV cache.

This module provides a minimal, efficient CPU offloading connector that:
- Uses Triton kernels for GPU<->CPU block transfers
- Reuses BlockPool for CPU block management with LRU eviction
- Supports hybrid KV cache manager (SupportsHMA)
- Supports lazy offloading (offload only when blocks are evicted from GPU cache)
"""

from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
)

__all__ = [
    "SimpleCPUOffloadMetadata",
]
