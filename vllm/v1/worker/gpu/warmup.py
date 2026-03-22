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

    The first iteration simulates a prefill with requests of 2 prompt
    tokens each. The second iteration simulates a decode step with all
    requests generating 1 token each.
    """
    prompt_token_ids = [0, 1]
    prompt_len = len(prompt_token_ids)
    decode_len = prompt_len + 1  # After prefill, one decode token is added.

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
        model_runner.scheduler_config.max_num_batched_tokens // prompt_len,
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

    # Step 1: Prefill all requests with 2 prompt tokens each.
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

        # Step 2: Decode all requests with 1 + num_spec_steps tokens each.
        spec_config = model_runner.speculative_config
        if spec_config is not None:
            num_spec_steps = spec_config.num_speculative_tokens
        else:
            num_spec_steps = 0
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
            rid: 1 + num_spec_steps for rid in req_ids
        }
        if num_spec_steps > 0:
            decode_output.scheduled_spec_decode_tokens = {
                rid: [1] * num_spec_steps for rid in req_ids
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
