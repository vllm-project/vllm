# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connect loaded Qwen4Exp QSA modules to their kernel-owned warmup."""

import sys
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.nvidia.indexer_qsa import QSAIndexer
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _get_qsa_indexer(worker: "Worker") -> "QSAIndexer | None":
    module = sys.modules.get("vllm.models.qwen4_exp.nvidia.indexer_qsa")
    if module is None:
        return None
    QSAIndexer = module.QSAIndexer

    return next(
        (
            module
            for module in worker.get_model().modules()
            if isinstance(module, QSAIndexer)
        ),
        None,
    )


def _get_compressed_block_table(
    worker: "Worker",
    indexer: "QSAIndexer",
) -> torch.Tensor | None:
    runner = worker.model_runner
    prefix = indexer.compressed_key_cache.prefix
    groups = runner.kv_cache_config.kv_cache_groups
    for group_id, group in enumerate(groups):
        if prefix in group.layer_names:
            if worker.use_v2_model_runner:
                block_tables = getattr(runner, "block_tables", None)
                if block_tables is None:
                    return None
                return block_tables.input_block_tables[group_id]
            return runner.input_batch.block_table[group_id].get_device_tensor(
                runner.max_num_reqs
            )
    return None


def qwen4_exp_qsa_triton_warmup(worker: "Worker") -> None:
    """Warm every reachable QSA decode-query-length specialization."""

    indexer = _get_qsa_indexer(worker)
    if indexer is None:
        return
    block_table = _get_compressed_block_table(worker, indexer)
    if block_table is None:
        logger.warning("Skipping Qwen4Exp QSA warmup: block table was not found.")
        return

    runner = worker.model_runner
    max_dql = getattr(runner, "uniform_decode_query_len", None)
    if max_dql is None:
        max_dql = getattr(runner, "decode_query_len", None)
    if max_dql is None:
        max_dql = 1 + worker.vllm_config.num_speculative_tokens
    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        warmup_qsa_mqa_paged_decode,
    )

    k_cache = indexer.compressed_key_cache.kv_cache
    if not k_cache.numel():
        logger.warning("Skipping Qwen4Exp QSA warmup: cache is not bound.")
        return

    profiles = warmup_qsa_mqa_paged_decode(
        k_cache,
        block_table,
        num_heads=indexer.index_n_heads,
        head_dim=indexer.index_head_dim,
        max_decode_query_len=int(max_dql),
        max_num_reqs=int(runner.max_num_reqs),
        max_num_batched_tokens=int(runner.max_num_tokens),
    )
    if profiles:
        logger.info("Warmed up Qwen4Exp QSA decode kernels: %s.", profiles)


__all__ = ["qwen4_exp_qsa_triton_warmup"]
