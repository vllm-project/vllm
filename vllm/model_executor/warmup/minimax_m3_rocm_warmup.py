# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


@instrument(span_name="MiniMax M3 ROCm NVFP4 warmup")
def minimax_m3_rocm_nvfp4_warmup(worker: "Worker") -> None:
    """Compile the FlyDSL NVFP4 KV kernels at startup, not in the first requests."""
    model_module = sys.modules.get("vllm.models.minimax_m3.amd.model")
    if model_module is None:
        return
    MiniMaxM3SparseAttention = model_module.MiniMaxM3SparseAttention
    from vllm.models.minimax_m3.amd import nvfp4_warmup

    module = next(
        (
            m
            for m in worker.get_model().modules()
            if isinstance(m, MiniMaxM3SparseAttention) and m.use_nvfp4_kv
        ),
        None,
    )
    if module is None or module.kv_cache.numel() == 0:
        return

    sched = worker.scheduler_config
    spec = worker.vllm_config.speculative_config
    num_spec = spec.num_speculative_tokens if spec is not None else 0
    query_lens = range(1, num_spec + 2)
    max_model_len = worker.model_config.max_model_len
    start = time.perf_counter()
    nvfp4_warmup.warmup_insert(
        module._norm_rope_nvfp4_insert,
        (module.num_heads + 2 * module.num_kv_heads + module.num_idx_heads + 1)
        * module.head_dim,
        sched.max_num_batched_tokens,
        module.kv_cache.device,
    )
    nvfp4_warmup.warmup_attention(
        module.kv_cache,
        module.num_heads,
        module.num_kv_heads,
        module.impl.scale,
        module._k_scale,
        module._v_scale,
        query_lens,
        sched.max_num_seqs,
        sched.max_num_batched_tokens,
    )
    indexer = module.indexer.impl
    nvfp4_warmup.warmup_index(
        module.indexer.index_cache.kv_cache,
        indexer.num_index_heads,
        indexer.init_blocks,
        indexer.local_blocks,
        query_lens,
        sched.max_num_seqs,
        sched.max_num_batched_tokens,
        max_model_len,
    )
    logger.info(
        "MiniMax M3 NVFP4 FlyDSL kernels warmed up in %.1f s.",
        time.perf_counter() - start,
    )
