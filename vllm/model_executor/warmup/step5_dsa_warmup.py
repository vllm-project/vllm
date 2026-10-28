# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm Step5 DSA scored prefill before serving.

Startup dummy runs are short, so every row selects its whole history and the
indexer scoring path (summary packing, FP8 logits, top-k, page metadata and,
for multi-group caches, staged prefill pages) would otherwise JIT on the first
prefill longer than top-k regions. Under DP every rank stalls on that rank.
"""

from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _step5_dsa_impl(context):
    from vllm.models.step5.nvidia.sparse_attention import Step5DSAAttentionImpl

    for layer in context.values():
        impl = getattr(layer, "impl", None)
        if isinstance(impl, Step5DSAAttentionImpl):
            return impl
        indexer = getattr(layer, "indexer", None)
        if indexer is not None and hasattr(layer, "csa_tail_ring"):
            return indexer
    return None


def step5_dsa_warmup(worker: "Worker") -> None:
    context = worker.vllm_config.compilation_config.static_forward_context
    impl = _step5_dsa_impl(context)
    if impl is None:
        return
    runner = worker.model_runner
    adapter = getattr(impl, "adapter", None)
    topk = getattr(impl, "sparse_topk", getattr(impl, "topk", None))
    assert topk is not None
    if hasattr(impl, "sparse_topk") and adapter is None:
        return
    # One region past "select every region" (top-k x 8 tokens), plus the
    # decode token of the mixed step.
    num_tokens = topk * 8 + 17
    if num_tokens > worker.scheduler_config.max_num_batched_tokens or (
        num_tokens > runner.max_model_len
    ):
        logger.warning(
            "Skipping Step5 DSA scored-prefill warmup: needs %d tokens per step.",
            num_tokens,
        )
        return
    logger.info("Warming up Step5 DSA scored prefill with %d tokens.", num_tokens)
    if runner.max_num_reqs >= 2:
        run_mixed_prefill_decode_warmup(
            cast("GPUModelRunner", runner),
            worker.execute_model,
            worker.sample_tokens,
            num_tokens,
            req_id_prefix="_step5_dsa_warmup",
        )
    else:
        runner._dummy_run(num_tokens=num_tokens, skip_eplb=True, is_profile=True)
    if adapter is not None:
        adapter.warmup_prefill_scoring(
            impl.max_num_regions,
            topk,
            impl.indexer_heads if impl.indexer_heads * 2 == 8 else 0,
            worker.device,
        )
