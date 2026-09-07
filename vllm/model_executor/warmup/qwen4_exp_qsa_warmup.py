# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connect loaded Qwen4Exp QSA modules to their kernel-owned warmup."""

import sys
from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def qwen4_exp_qsa_triton_warmup(worker: "Worker") -> None:
    """Warm every reachable QSA decode-query-length specialization."""

    qsa_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.indexer_qsa")
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

    runner = worker.model_runner
    prefix = indexer.compressed_key_cache.prefix
    group_id = next(
        i
        for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
        if prefix in group.layer_names
    )
    if worker.use_v2_model_runner:
        runner_v2 = cast("GPUModelRunnerV2", runner)
        block_table = runner_v2.block_tables.input_block_tables[group_id]
        max_decode_query_len = runner_v2.decode_query_len
    else:
        block_table = runner.input_batch.block_table[group_id].get_device_tensor(
            runner.max_num_reqs
        )
        max_decode_query_len = runner.uniform_decode_query_len

    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        warmup_qsa_mqa_paged_decode,
    )

    k_cache = indexer.compressed_key_cache.kv_cache
    assert k_cache.numel()
    profiles = warmup_qsa_mqa_paged_decode(
        k_cache,
        block_table,
        num_heads=indexer.index_n_heads,
        head_dim=indexer.index_head_dim,
        max_decode_query_len=max_decode_query_len,
        max_num_reqs=runner.max_num_reqs,
        max_num_batched_tokens=runner.max_num_tokens,
    )
    logger.info("Warmed up Qwen4Exp QSA decode kernels: %s.", profiles)
