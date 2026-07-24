# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZoomKV sparse KV retrieval operators for vLLM V1.

Supports GPU-only mode and optional K-only CPU offload
(``zoomkv_enable_offload=True``).
"""

from vllm.v1.attention.ops.zoomkv.kernels import get_quest_ops, try_load_zoomkv_c
from vllm.v1.attention.ops.zoomkv.kivi_rerank import partial_chunk_kivi_qk
from vllm.v1.attention.ops.zoomkv.offload import (
    ZoomKVCpuKeyPool,
    get_cpu_key_pool,
    set_cpu_key_pool,
)
from vllm.v1.attention.ops.zoomkv.paged import (
    gather_kv_by_logical_indices,
    gather_kv_hybrid,
    sparse_decode_attention,
)
from vllm.v1.attention.ops.zoomkv.quant_pack import pack_kcache_4bit
from vllm.v1.attention.ops.zoomkv.quest import QuestTorchOps
from vllm.v1.attention.ops.zoomkv.quest_triton import QuestTritonOps
from vllm.v1.attention.ops.zoomkv.retriever import ZoomKVRetriever, ZoomKVRuntimeConfig
from vllm.v1.attention.ops.zoomkv.state import (
    ZoomKVBlockSummary,
    clear_block_summaries,
    copy_block_summaries_for_block_pairs,
    invalidate_block_summaries_for_blocks,
)

__all__ = [
    "QuestTorchOps",
    "QuestTritonOps",
    "ZoomKVBlockSummary",
    "ZoomKVCpuKeyPool",
    "ZoomKVRetriever",
    "ZoomKVRuntimeConfig",
    "clear_block_summaries",
    "copy_block_summaries_for_block_pairs",
    "gather_kv_by_logical_indices",
    "gather_kv_hybrid",
    "get_cpu_key_pool",
    "get_quest_ops",
    "invalidate_block_summaries_for_blocks",
    "pack_kcache_4bit",
    "partial_chunk_kivi_qk",
    "set_cpu_key_pool",
    "sparse_decode_attention",
    "try_load_zoomkv_c",
]
