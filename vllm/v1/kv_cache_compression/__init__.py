# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional rank-based compression for K in KV cache.

To enable (evaluation-only):
    - VLLM_KV_KEY_COMPRESS_ENABLED=1
    - Optional knobs:
        * VLLM_KV_KEY_COMPRESS_ENERGY=0.995
        * VLLM_KV_KEY_COMPRESS_MAX_RANK=0  (0/None = auto)
        * VLLM_KV_KEY_COMPRESS_MIN_TOKENS=16
        * VLLM_KV_KEY_COMPRESS_RECOMPUTE_EVERY=8
        * VLLM_KV_KEY_COMPRESS_LOG_EVERY=64
        * VLLM_KV_KEY_COMPRESS_LAYERS=early,mid,late or "0,12,23"

Requires `--enforce-eager` to avoid CUDA graph capture of Python hooks.
"""

from vllm.v1.kv_cache_compression.compressor import (
    maybe_compress_kv_decode,
    reset_kv_compression_state,
)
from vllm.v1.kv_cache_compression.config import (
    KVCacheCompressionConfig,
    get_kv_compression_config,
    init_kv_compression_config,
)

__all__ = [
    "KVCacheCompressionConfig",
    "get_kv_compression_config",
    "init_kv_compression_config",
    "maybe_compress_kv_decode",
    "reset_kv_compression_state",
]
