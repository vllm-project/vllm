# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@torch.inference_mode()
def warmup_kernels(model_runner: GPUModelRunner) -> None:
    """Run two execute_model + sample_tokens iterations to JIT compile
    triton kernels.

    The first iteration simulates a prefill with requests of 2 prompt
    tokens each. The second iteration simulates a decode step with all
    requests generating 1 token each.
    """
    prompt_token_ids = [0, 1]
    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens // len(prompt_token_ids),
    )

    num_kv_cache_groups = len(model_runner.kv_cache_config.kv_cache_groups)
    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    # Step 1: Prefill all requests with 2 prompt tokens each.
    new_reqs = []
    num_scheduled_tokens: dict[str, int] = {}
    for i in range(num_reqs):
        # Each request uses a distinct block per KV cache group.
        block_ids = tuple([i] for _ in range(num_kv_cache_groups))
        new_reqs.append(
            NewRequestData.from_request(
                Request(req_ids[i], prompt_token_ids, sampling_params, pooling_params),
                block_ids=block_ids,
                prefill_token_ids=prompt_token_ids,
            )
        )
        num_scheduled_tokens[req_ids[i]] = len(prompt_token_ids)

    unused_scheduler_output_fields = dict[str, Any](
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    prefill_output = SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=len(prompt_token_ids) * num_reqs,
        num_common_prefix_blocks=[0] * num_kv_cache_groups,
        **unused_scheduler_output_fields,
    )

    model_runner.execute_model(prefill_output)

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

        model_runner.sample_tokens(grammar_output)

        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = list(req_ids)
        cached_req_data.new_block_ids = [None] * num_reqs
        cached_req_data.num_computed_tokens = [len(prompt_token_ids)] * num_reqs
        cached_req_data.num_output_tokens = [1] * num_reqs

        # Step 2: Decode all requests with 1 token each.
        decode_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens={rid: 1 for rid in req_ids},
            total_num_scheduled_tokens=num_reqs,
            num_common_prefix_blocks=[0] * num_kv_cache_groups,
            **unused_scheduler_output_fields,
        )
        model_runner.execute_model(decode_output)
        model_runner.sample_tokens(None)

    # Clean up - process finish_req_ids.
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    model_runner.execute_model(cleanup_output)
