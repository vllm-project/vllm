# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connect loaded Qwen4Exp QSA modules to their kernel-owned warmup."""

import sys
from typing import TYPE_CHECKING, cast

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _block_table_for(worker: "Worker", prefix: str) -> torch.Tensor:
    runner = worker.model_runner
    group_id = next(
        i
        for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
        if prefix in group.layer_names
    )
    if worker.use_v2_model_runner:
        runner_v2 = cast("GPUModelRunnerV2", runner)
        return runner_v2.block_tables.input_block_tables[group_id]
    return runner.input_batch.block_table[group_id].get_device_tensor(
        runner.max_num_reqs
    )


def _amd_qsa_triton_warmup(worker: "Worker") -> None:
    """Compile the MFMA scorer the AMD indexer uses for prefill chunks; its
    first launch would otherwise come after warmup, in a served step."""
    qsa_module = sys.modules.get("vllm.models.qwen4_exp.amd.qsa_indexer")
    if qsa_module is None:
        return
    indexer = next(
        (
            layer
            for layer in worker.get_model().modules()
            if isinstance(layer, qsa_module.QSAIndexer)
        ),
        None,
    )
    if indexer is None:
        return

    from vllm.models.qwen4_exp.amd.ops.qsa_mfma_scorer import (
        warmup_qsa_mqa_paged_mfma,
    )

    k_cache = indexer.compressed_key_cache.kv_cache
    assert k_cache.numel()
    widths = warmup_qsa_mqa_paged_mfma(
        k_cache,
        _block_table_for(worker, indexer.compressed_key_cache.prefix),
        num_heads=indexer.index_n_heads,
        head_dim=indexer.index_head_dim,
        compress_ratio=indexer.compress_ratio,
        max_num_batched_tokens=worker.model_runner.max_num_tokens,
    )
    logger.info("Warmed up Qwen4Exp QSA MFMA scorer at score widths %s.", widths)


def _amd_qsa_flydsl_warmup(worker: "Worker") -> None:
    """Compile AITER's FlyDSL sparse prefill kernels for the QSA cache, which a
    served step would otherwise compile on its first long prefill."""
    qsa_module = sys.modules.get("vllm.models.qwen4_exp.amd.qsa")
    if qsa_module is None:
        return
    owner = next(
        (
            layer
            for layer in worker.get_model().modules()
            if isinstance(layer, qsa_module.Qwen4ExpQSAAttention)
        ),
        None,
    )
    if owner is None:
        return

    from vllm.models.qwen4_exp.amd.ops.qsa_flydsl_prefill import (
        flydsl_prefill_available,
        warmup_flydsl_qsa_prefill,
    )

    # The op reads bf16 caches only; fp8 and Octave caches are uint8 storage.
    if owner.kv_cache.dtype != torch.bfloat16 or not flydsl_prefill_available():
        return
    k_cache, v_cache = owner.impl.kv_cache_views(owner.kv_cache)
    layouts = warmup_flydsl_qsa_prefill(
        owner.num_heads, k_cache, v_cache, owner.indexer.output_width
    )
    if layouts:
        logger.info(
            "Warmed up Qwen4Exp QSA FlyDSL sparse prefill for prefill requests "
            "of %s rows.",
            layouts,
        )


def qwen4_exp_qsa_triton_warmup(worker: "Worker") -> None:
    """Warm every reachable QSA specialization: indexer decode-query-length
    profiles plus the sparse attention split-K/merge configs (NVIDIA), the
    prefill MFMA scorer (AMD)."""
    _amd_qsa_triton_warmup(worker)
    _amd_qsa_flydsl_warmup(worker)
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

    if worker.use_v2_model_runner:
        max_decode_query_len = cast("GPUModelRunnerV2", runner).decode_query_len
    else:
        max_decode_query_len = runner.uniform_decode_query_len

    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        warmup_qsa_mqa_paged_decode,
    )

    k_cache = indexer.compressed_key_cache.kv_cache
    assert k_cache.numel()
    profiles = warmup_qsa_mqa_paged_decode(
        k_cache,
        _block_table_for(worker, indexer.compressed_key_cache.prefix),
        num_heads=indexer.index_n_heads,
        head_dim=indexer.index_head_dim,
        max_decode_query_len=max_decode_query_len,
        max_num_reqs=runner.max_num_reqs,
        max_num_batched_tokens=runner.max_num_tokens,
    )
    logger.info("Warmed up Qwen4Exp QSA decode kernels: %s.", profiles)

    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        warmup_qsa_sparse_paged_attention,
    )

    kv_cache = owner.kv_cache
    assert kv_cache.numel()
    attention_profiles = warmup_qsa_sparse_paged_attention(
        kv_cache,
        _block_table_for(worker, owner.layer_name),
        num_query_heads=owner.num_heads,
        selection_width=indexer.output_width,
    )
    logger.info(
        "Warmed up Qwen4Exp QSA sparse attention kernels: %s.",
        attention_profiles,
    )
