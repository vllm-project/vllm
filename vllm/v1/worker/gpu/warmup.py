# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run two execute_model + sample_tokens iterations to JIT compile
    triton kernels. We must call the provided worker's execute_model for
    pipeline parallel coordination.

    The first iteration simulates a prefill with requests of
    decode_query_len + 1 prompt tokens each. The second iteration simulates
    a decode step with all requests generating decode_query_len tokens.
    """
    num_spec_steps = model_runner.num_speculative_steps
    decode_query_len = model_runner.decode_query_len
    # Use decode_query_len + 1 tokens so the prefill batch's per-request query
    # length exceeds decode_query_len, preventing it from being misclassified as
    # a uniform decode batch.
    prompt_len = decode_query_len + 1
    prompt_token_ids = list(range(prompt_len))
    # After prefill, decode generates decode_query_len tokens.
    decode_len = prompt_len + decode_query_len

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    # Compute per-request block counts for each KV cache group.
    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
    prefill_block_counts = [cdiv(prompt_len, bs) for bs in group_block_sizes]
    decode_block_counts = [cdiv(decode_len, bs) for bs in group_block_sizes]
    decode_block_deltas = [
        d - p for d, p in zip(decode_block_counts, prefill_block_counts)
    ]
    max_blocks_per_req = sum(decode_block_counts)

    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, decode_query_len),
        # Reserve block 0 (null block) and ensure we have enough blocks.
        max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
    )

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    # Assign distinct block IDs per request per group. 0 null block, start from 1.
    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    # Step 1: Prefill all requests with 1 + decode_query_len prompt tokens each.
    new_reqs = [
        NewRequestData.from_request(
            Request(req_ids[i], prompt_token_ids, sampling_params, pooling_params),
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    # Disable KV connector for warmup run.
    model_runner.kv_connector.set_disabled(True)
    worker_execute_model(prefill_output)

    if not model_runner.is_pooling_model:
        # Warm up sampler and perform a decode step for non-pooling models.

        grammar_output = None
        if model_runner.is_last_pp_rank:
            # Build a GrammarOutput to exercise the structured output bitmask
            # kernel during the prefill step.
            vocab_size = model_runner.model_config.get_vocab_size()
            bitmask_width = (vocab_size + 31) // 32
            grammar_bitmask = np.full(
                (len(req_ids), bitmask_width), fill_value=-1, dtype=np.int32
            )
            grammar_output = GrammarOutput(
                structured_output_request_ids=req_ids, grammar_bitmask=grammar_bitmask
            )

        worker_sample_tokens(grammar_output)

        # Step 2: Decode all requests with decode_query_len tokens each.
        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = list(req_ids)
        cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
        cached_req_data.num_output_tokens = [1] * num_reqs
        new_block = any(decode_block_deltas)
        cached_req_data.new_block_ids = [
            tuple(_alloc_blocks(n) for n in decode_block_deltas) if new_block else None
            for _ in range(num_reqs)
        ]

        decode_output = SchedulerOutput.make_empty()
        decode_output.scheduled_cached_reqs = cached_req_data
        decode_output.num_scheduled_tokens = {
            req_id: decode_query_len for req_id in req_ids
        }
        if num_spec_steps > 0:
            decode_output.scheduled_spec_decode_tokens = {
                req_id: [0] * num_spec_steps for req_id in req_ids
            }
        decode_output.total_num_scheduled_tokens = sum(
            decode_output.num_scheduled_tokens.values()
        )
        decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

        worker_execute_model(decode_output)
        worker_sample_tokens(None)

    # Clean up - process finish_req_ids.
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    worker_execute_model(cleanup_output)
    model_runner.kv_connector.set_disabled(False)
    torch.accelerator.synchronize()


def _get_positive_int(obj: Any, name: str, default: int) -> int:
    value = getattr(obj, name, default)
    return value if isinstance(value, int) and value > 0 else default


@torch.inference_mode()
def warmup_v1_attention_kernels(model_runner: Any) -> None:
    """Warm old V1 attention kernels through the existing dummy-run path.

    V1 profiling does not always build attention metadata, and CUDA graph
    capture locks the workspace after capture. Run small forced-attention dummy
    batches before capture so attention backends can size workspace without
    backend-specific synthetic calls.
    """
    dummy_run = getattr(model_runner, "_dummy_run", None)
    if dummy_run is None:
        return
    if getattr(model_runner, "is_pooling_model", False):
        return
    if not getattr(model_runner, "attn_groups", None):
        return

    scheduler_config = getattr(model_runner, "scheduler_config", None)
    if scheduler_config is None:
        return

    max_num_tokens = _get_positive_int(model_runner, "max_num_tokens", 0)
    max_model_len = _get_positive_int(model_runner, "max_model_len", 0)
    max_num_seqs = _get_positive_int(scheduler_config, "max_num_seqs", 1)
    max_num_batched_tokens = _get_positive_int(
        scheduler_config,
        "max_num_batched_tokens",
        max_num_tokens,
    )
    if max_num_tokens <= 0 or max_model_len <= 0 or max_num_batched_tokens <= 0:
        return

    decode_query_len = _get_positive_int(model_runner, "uniform_decode_query_len", 1)
    max_decode_tokens = min(
        max_num_tokens,
        max_num_batched_tokens,
        max_num_seqs * decode_query_len,
    )
    if max_decode_tokens >= decode_query_len:
        decode_seq_len = min(max_model_len, max(decode_query_len + 1, 8192))
        decode_kwargs = {
            "num_tokens": max_decode_tokens,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": True,
        }
        if decode_seq_len > decode_query_len:
            decode_kwargs["profile_seq_lens"] = decode_seq_len
        dummy_run(**decode_kwargs)

    # Keep this representative cached-prefill shape intentionally small. It
    # covers request-shaped attention metadata without turning warmup into a
    # long-prompt benchmark.
    prefill_tokens = min(max_num_tokens, max_num_batched_tokens, max_model_len, 64)
    prefill_seq_len = min(max_model_len, max(prefill_tokens + 1, 8192))
    if prefill_tokens > 1 and prefill_seq_len > prefill_tokens:
        dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=False,
            num_reqs_override=1,
            profile_seq_lens=prefill_seq_len,
        )

    torch.accelerator.synchronize()
