# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connect loaded Qwen4Exp QSA modules to their kernel-owned warmup."""

import sys
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def qwen4_exp_qsa_triton_warmup(worker: "Worker") -> None:
    """Warm every reachable QSA specialization: indexer decode-query-length
    profiles plus the sparse attention split-K/merge configs."""
    qsa_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.qsa_indexer")
    attn_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.qsa")
    if qsa_module is None or attn_module is None:
        return
    indexer = None
    owner = None
    for layer in worker.get_model().modules():
        if indexer is None and isinstance(layer, qsa_module.QSAIndexer):
            indexer = layer
        elif owner is None and isinstance(layer, attn_module.Qwen4ExpQSAAttention):
            owner = layer
    if indexer is None or owner is None:
        return

    runner = worker.model_runner

    def block_table_for(prefix: str) -> torch.Tensor:
        group_id = next(
            i
            for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            if prefix in group.layer_names
        )
        return runner.block_tables.input_block_tables[group_id]

    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        qsa_prefill_logits_width,
        use_gvr2_topk,
        warmup_gvr2_topk,
        warmup_qsa_mqa_paged_decode,
    )

    k_cache = indexer.compressed_key_cache.kv_cache
    assert k_cache.numel()
    profiles = warmup_qsa_mqa_paged_decode(
        k_cache,
        block_table_for(indexer.compressed_key_cache.prefix),
        num_heads=indexer.index_n_heads,
        head_dim=indexer.index_head_dim,
        max_decode_query_len=runner.decode_query_len,
        max_num_reqs=runner.max_num_reqs,
        max_num_batched_tokens=runner.max_num_tokens,
    )
    logger.info("Warmed up Qwen4Exp QSA decode kernels: %s.", profiles)

    # Pre-compile gvr_2 so neither CUDA-graph capture nor serving compiles it.
    block_topk = indexer.token_topk // indexer.compress_ratio
    decode_width = (
        block_table_for(indexer.compressed_key_cache.prefix).shape[1] * k_cache.shape[1]
    )
    prefill_width = qsa_prefill_logits_width(
        worker.vllm_config.model_config.max_model_len,
        indexer.compress_ratio,
        decode_width,
    )
    # prefill_width is float4-aligned, so this only checks platform and k.
    if use_gvr2_topk(indexer.topk_backend, block_topk, prefill_width):
        warmup_gvr2_topk(
            indexer.topk_backend,
            block_topk,
            decode_width=decode_width,
            max_decode_rows=runner.max_num_reqs * runner.decode_query_len,
            prefill_width=prefill_width,
        )
        logger.info(
            "Warmed up Qwen4Exp QSA gvr_2 top-k (k=%d, decode width %d, "
            "prefill width %d).",
            block_topk,
            decode_width,
            prefill_width,
        )

    if indexer.index_head_dim == 128:
        from vllm.models.qwen4_exp.nvidia.ops.qsa_prefill_deepgemm import (
            warmup_deepgemm_prefill,
        )

        warmup_deepgemm_prefill(k_cache, indexer.index_n_heads)

    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        warmup_qsa_sparse_paged_attention,
    )

    kv_cache = owner.kv_cache
    assert kv_cache.numel()
    attention_profiles = warmup_qsa_sparse_paged_attention(
        kv_cache,
        block_table_for(owner.layer_name),
        num_query_heads=owner.num_heads,
        selection_width=indexer.output_width,
    )
    logger.info(
        "Warmed up Qwen4Exp QSA sparse attention kernels: %s.",
        attention_profiles,
    )
    if owner.index_group is not None:
        hot_profiles = warmup_qsa_sparse_paged_attention(
            owner.impl.kv_view(owner.index_group.physical_kv_cache(0)),
            block_table_for(owner.layer_name),
            num_query_heads=owner.num_heads,
            selection_width=indexer.output_width,
            physical_indices=True,
        )
        logger.info("Warmed up Qwen4Exp QSA HiSparse decode kernels: %s.", hot_profiles)
