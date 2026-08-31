# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Qwen4Exp QSA Triton decode kernels."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.nvidia.indexer_qsa import QSAIndexer
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_LARGE_DECODE_REQUESTS = 33


def _qsa_decode_warmup_profiles(
    max_dql: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    profiles = []
    for dql in range(1, max_dql + 1):
        if dql <= max_num_batched_tokens:
            profiles.append((dql, 1))
        if (
            max_num_reqs >= _LARGE_DECODE_REQUESTS
            and dql * _LARGE_DECODE_REQUESTS <= max_num_batched_tokens
        ):
            profiles.append((dql, _LARGE_DECODE_REQUESTS))
    return tuple(profiles)


def _get_qsa_indexer(worker: "Worker") -> "QSAIndexer | None":
    from vllm.models.qwen4_exp.nvidia.indexer_qsa import QSAIndexer

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


@torch.inference_mode()
def qwen4_exp_qsa_triton_warmup(worker: "Worker") -> None:
    """Warm every reachable QSA decode-query-length specialization."""

    if not current_platform.is_cuda():
        return
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
    profiles = _qsa_decode_warmup_profiles(
        max_dql=int(max_dql),
        max_num_reqs=int(runner.max_num_reqs),
        max_num_batched_tokens=int(runner.max_num_tokens),
    )
    if not profiles:
        return

    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        qsa_mqa_paged_decode,
    )

    k_cache = indexer.compressed_key_cache.kv_cache
    if not k_cache.numel():
        logger.warning("Skipping Qwen4Exp QSA warmup: cache is not bound.")
        return

    logger.info("Warming up Qwen4Exp QSA decode kernels: %s.", profiles)
    for dql, num_requests in profiles:
        num_tokens = dql * num_requests
        qsa_mqa_paged_decode(
            torch.empty(
                num_tokens,
                indexer.index_n_heads,
                indexer.index_head_dim,
                dtype=torch.bfloat16,
                device=k_cache.device,
            ),
            k_cache,
            block_table[:num_requests],
            torch.zeros(num_tokens, dtype=torch.int64, device=k_cache.device),
            torch.zeros(num_requests, dtype=torch.int32, device=k_cache.device),
            indexer.compress_ratio,
            dql,
        )
    torch.accelerator.synchronize(k_cache.device)


__all__ = ["qwen4_exp_qsa_triton_warmup"]
