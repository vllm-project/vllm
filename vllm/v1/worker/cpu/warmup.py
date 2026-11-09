# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    CrossAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.worker.cpu.standalone_model_runner import StandaloneCPUModelRunner


def _reserved_block_count(
    num_tokens: int,
    kvcache_spec: KVCacheSpec,
    *,
    num_lookahead_tokens: int,
    max_model_len: int,
    max_encoder_len: int,
) -> int:
    if isinstance(kvcache_spec, UniformTypeKVCacheSpecs):
        kvcache_spec = kvcache_spec.first_spec
    if isinstance(kvcache_spec, CircularBufferSpec):
        return 1
    if isinstance(kvcache_spec, CrossAttentionSpec):
        return cdiv(max_encoder_len, kvcache_spec.block_size)
    num_speculative_blocks = 0
    if isinstance(kvcache_spec, MambaSpec):
        num_speculative_blocks = kvcache_spec.num_speculative_blocks
        if kvcache_spec.mamba_cache_mode == "align":
            return cdiv(num_tokens, kvcache_spec.block_size) + num_speculative_blocks
    num_tokens = min(num_tokens + num_lookahead_tokens, max_model_len)
    return cdiv(num_tokens, kvcache_spec.block_size) + num_speculative_blocks


def _warmup_block_counter(
    model_runner: "StandaloneCPUModelRunner",
) -> Callable[[int, KVCacheSpec], int]:
    num_lookahead_tokens = model_runner.vllm_config.num_lookahead_tokens
    max_model_len = model_runner.max_model_len
    max_encoder_len = getattr(model_runner.model_state, "max_encoder_len", 0)

    def block_count(num_tokens: int, kvcache_spec: KVCacheSpec) -> int:
        return _reserved_block_count(
            num_tokens,
            kvcache_spec,
            num_lookahead_tokens=num_lookahead_tokens,
            max_model_len=max_model_len,
            max_encoder_len=max_encoder_len,
        )

    return block_count


@torch.inference_mode()
def warmup_kernels(
    model_runner: "StandaloneCPUModelRunner",
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    if model_runner.vllm_config.is_mm_encoder_only:
        return

    decode_query_len = model_runner.decode_query_len
    prompt_len = decode_query_len + 1
    prompt_token_ids = list(range(prompt_len))
    num_decode_steps = 1 if model_runner.is_pooling_model else 3
    decode_len = prompt_len + num_decode_steps * decode_query_len

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    max_encoder_len = getattr(model_runner.model_state, "max_encoder_len", 0)
    warmup_mm_features: list[MultiModalFeatureSpec] = []
    if model_runner.is_encoder_decoder and max_encoder_len:
        warmup_mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="",
                identifier="_warmup_encoder",
                mm_position=PlaceholderRange(offset=0, length=max_encoder_len),
            )
        ]

    block_count = _warmup_block_counter(model_runner)
    kv_cache_specs = [group.kv_cache_spec for group in kv_cache_groups]
    prefill_block_counts = [block_count(prompt_len, spec) for spec in kv_cache_specs]
    decode_block_counts = [block_count(decode_len, spec) for spec in kv_cache_specs]
    max_blocks_per_req = sum(decode_block_counts)

    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, decode_query_len),
    )
    null_blocks = model_runner.block_tables.redirect_writes_to_null_block
    if max_blocks_per_req > 0 and not null_blocks:
        num_reqs = min(
            num_reqs,
            max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
        )

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_task = model_runner.model_config.get_pooling_task(
            model_runner.get_supported_tasks()
        )
        pooling_params = PoolingParams(task=pooling_task)
        pooling_params.verify(model_runner.model_config)
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    if model_runner.kv_block_zeroer is not None:
        model_runner.kv_block_zeroer.warmup(model_runner.kv_cache_config.num_blocks)

    new_reqs = [
        NewRequestData.from_request(
            Request(
                req_ids[i],
                prompt_token_ids,
                sampling_params,
                pooling_params,
                mm_features=warmup_mm_features,
            ),
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {req_id: prompt_len for req_id in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs

    model_runner.kv_connector.set_disabled(True)
    try:
        worker_execute_model(prefill_output)

        if not model_runner.is_pooling_model:
            grammar_output = None
            if model_runner.is_last_pp_rank:
                vocab_size = model_runner.model_config.get_vocab_size()
                bitmask_width = (vocab_size + 31) // 32
                grammar_output = GrammarOutput(
                    structured_output_request_ids=req_ids,
                    grammar_bitmask=np.full(
                        (len(req_ids) * decode_query_len, bitmask_width),
                        fill_value=-1,
                        dtype=np.int32,
                    ),
                )
            worker_sample_tokens(grammar_output)

            req_computed = [prompt_len] * num_reqs
            req_blocks = [list(prefill_block_counts) for _ in range(num_reqs)]

            def _run_decode_step(indices: list[int]) -> None:
                cached_req_data = CachedRequestData.make_empty()
                cached_req_data.req_ids = [req_ids[i] for i in indices]
                cached_req_data.num_computed_tokens = [req_computed[i] for i in indices]
                cached_req_data.num_output_tokens = [1] * len(indices)
                cached_req_data.new_block_ids = []

                num_scheduled_tokens: dict[str, int] = {}
                for i in indices:
                    after = req_computed[i] + 1
                    deltas = [
                        block_count(after, spec) - held
                        for spec, held in zip(kv_cache_specs, req_blocks[i])
                    ]
                    cached_req_data.new_block_ids.append(
                        tuple(_alloc_blocks(n) for n in deltas) if any(deltas) else None
                    )
                    req_blocks[i] = [
                        held + delta for held, delta in zip(req_blocks[i], deltas)
                    ]
                    num_scheduled_tokens[req_ids[i]] = 1

                decode_output = SchedulerOutput.make_empty()
                decode_output.scheduled_cached_reqs = cached_req_data
                decode_output.num_scheduled_tokens = num_scheduled_tokens
                decode_output.total_num_scheduled_tokens = len(indices)
                worker_execute_model(decode_output)
                worker_sample_tokens(None)

                for i in indices:
                    req_computed[i] += 1

            all_indices = list(range(num_reqs))
            _run_decode_step(all_indices)
            if num_reqs > 1:
                _run_decode_step([0])

        cleanup_output = SchedulerOutput.make_empty()
        cleanup_output.finished_req_ids = set(req_ids)
        worker_execute_model(cleanup_output)
    finally:
        model_runner.kv_connector.set_disabled(False)

    if model_runner.kv_block_zeroer is not None:
        model_runner.kv_block_zeroer.zero_block_ids([0])
