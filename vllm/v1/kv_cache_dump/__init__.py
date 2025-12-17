# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV Cache Dumping for Decode-Only Rank Analysis.

This module provides functionality to dump KV cache tensors during decode
phase for offline rank analysis. The dumped tensors can be used to study
KV cache intrinsic dimensionality and evaluate local low-rank structure.

Usage:
    Set environment variables to enable dumping:
    - VLLM_KV_CACHE_DUMP_ENABLED=1
    - VLLM_KV_CACHE_DUMP_OUTPUT_DIR=/path/to/output
    - VLLM_KV_CACHE_DUMP_LAYERS="early,mid,late" or "0,15,31"
"""

from vllm.v1.kv_cache_dump.accumulator import (
    KVAccumulatorManager,
    get_accumulator_manager,
)
from vllm.v1.kv_cache_dump.config import (
    KVCacheDumpConfig,
    get_kv_dump_config,
    init_kv_dump_config,
)
from vllm.v1.kv_cache_dump.dumper import save_request_kv
from vllm.v1.kv_cache_dump.hook import (
    maybe_dump_kv_decode,
    set_request_context,
)

__all__ = [
    "KVCacheDumpConfig",
    "get_kv_dump_config",
    "init_kv_dump_config",
    "KVAccumulatorManager",
    "get_accumulator_manager",
    "save_request_kv",
    "maybe_dump_kv_decode",
    "set_request_context",
]
