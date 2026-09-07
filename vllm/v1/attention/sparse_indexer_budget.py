# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte budget for one sparse-indexer logits call during prefill chunking."""

import vllm.envs as envs
from vllm.platforms import current_platform

_ENV = "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"
_MIB = 1024 * 1024

# Default when the env var is unset on an integrated (unified-memory) GPU.
INTEGRATED_GPU_MAX_LOGITS_MB = 64


def sparse_indexer_max_logits_bytes() -> int:
    """Budget for one prefill sparse-indexer logits call.

    ``VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`` (default 512) is honoured whenever it
    is set. When it is not set and the device is an integrated (unified-memory)
    GPU such as GB10 / DGX Spark, the default drops to 64 MiB: the logits tensor
    is ``(chunk queries x prefix pools)`` and changes size every chunk of a long
    prefill, so at the 512 MiB default a 200K+-token request streams hundreds of
    ~500 MB non-reusable blocks per step through the caching allocator; on
    unified memory the resulting segment requests exhaust the host and the
    driver fails before the allocator's OOM-retry can flush its cache. Smaller
    calls keep the blocks small and reusable at a modest prefill cost.

    This is the prefill sub-chunking budget only; decode-time logits and the
    profiling-run reservations are sized independently by their callers.
    """
    if envs.is_set(_ENV):
        return envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * _MIB
    if current_platform.is_integrated_gpu():
        return INTEGRATED_GPU_MAX_LOGITS_MB * _MIB
    return envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * _MIB
