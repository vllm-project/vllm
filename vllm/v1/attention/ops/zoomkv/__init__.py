# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZoomKV sparse KV retrieval operators for vLLM V1.

Supports GPU-only mode and optional K+V CPU offload
(``zoomkv_enable_offload=True``).
"""

from vllm.v1.attention.ops.zoomkv.kernels import try_load_zoomkv_c
from vllm.v1.attention.ops.zoomkv.kivi_rerank import partial_chunk_kivi_qk
from vllm.v1.attention.ops.zoomkv.offload import (
    ZoomKVCpuKeyPool,
    filter_completed_for_offload,
    get_cpu_key_pool,
    retrieval_zone_logical_range,
    set_cpu_key_pool,
)
from vllm.v1.attention.ops.zoomkv.paged import (
    gather_kv_by_logical_indices,
    gather_kv_by_logical_indices_batch,
    gather_kv_hot_and_cpu_topk,
    gather_kv_hybrid,
    sparse_decode_attention,
    sparse_decode_attention_batch,
)
from vllm.v1.attention.ops.zoomkv.quant_pack import pack_kcache_4bit
from vllm.v1.attention.ops.zoomkv.retriever import ZoomKVRetriever, ZoomKVRuntimeConfig
from vllm.v1.attention.ops.zoomkv.state import (
    ZoomKVBlockSummary,
    clear_block_summaries,
    copy_block_summaries_for_block_pairs,
    invalidate_block_summaries_for_blocks,
)

__all__ = [
    "ZoomKVBlockSummary",
    "ZoomKVCpuKeyPool",
    "ZoomKVRetriever",
    "ZoomKVRuntimeConfig",
    "clear_block_summaries",
    "copy_block_summaries_for_block_pairs",
    "filter_completed_for_offload",
    "gather_kv_by_logical_indices",
    "gather_kv_by_logical_indices_batch",
    "gather_kv_hot_and_cpu_topk",
    "gather_kv_hybrid",
    "get_cpu_key_pool",
    "retrieval_zone_logical_range",
    "invalidate_block_summaries_for_blocks",
    "pack_kcache_4bit",
    "partial_chunk_kivi_qk",
    "set_cpu_key_pool",
    "sparse_decode_attention",
    "sparse_decode_attention_batch",
    "try_load_zoomkv_c",
]
