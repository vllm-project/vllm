# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import CrossAttentionSpec, KVCacheSpec, MambaSpec
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


def _reserved_block_count(
    num_tokens: int,
    kvcache_spec: KVCacheSpec,
    *,
    num_lookahead_tokens: int,
    max_model_len: int,
    max_encoder_len: int,
) -> int:
    """Number of blocks the scheduler would hold for a request of `num_tokens`.

    Warmup hand-builds its `SchedulerOutput`s, so it must reserve what
    `KVCacheManager.allocate_slots` reserves: the token range plus
    `num_lookahead_tokens`, where the speculator writes the KV of its drafts.
    """
    if isinstance(kvcache_spec, CrossAttentionSpec):
        # Cross-attention blocks cover the encoder sequence only.
        return cdiv(max_encoder_len, kvcache_spec.block_size)
    num_speculative_blocks = 0
    if isinstance(kvcache_spec, MambaSpec):
        # MambaManager appends speculative running-state blocks in every cache
        # mode; align mode sizes from the uncapped, lookahead-free token range.
        num_speculative_blocks = kvcache_spec.num_speculative_blocks
        if kvcache_spec.mamba_cache_mode == "align":
            return cdiv(num_tokens, kvcache_spec.block_size) + num_speculative_blocks
    num_tokens = min(num_tokens + num_lookahead_tokens, max_model_len)
    return cdiv(num_tokens, kvcache_spec.block_size) + num_speculative_blocks


def _warmup_block_counter(
    model_runner: GPUModelRunner,
) -> Callable[[int, KVCacheSpec], int]:
    """Bind `_reserved_block_count` to `model_runner`'s reservation policy."""
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


def run_mixed_prefill_decode_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    num_tokens: int,
    *,
    mixed_step_context: AbstractContextManager[object] | None = None,
    req_id_prefix: str = "_v2_mixed_warmup",
) -> bool:
    """Run a V2 mixed prefill+decode step through normal scheduler inputs."""
    if model_runner.is_pooling_model or model_runner.max_num_reqs < 2 or num_tokens < 3:
        return False

    decode_req_id = f"{req_id_prefix}_decode_"
    prefill_req_id = f"{req_id_prefix}_prefill_"
    decode_prompt_len = 2
    decode_scheduled_tokens = 1
    prefill_len = num_tokens - decode_scheduled_tokens
    decode_token_ids = list(range(decode_prompt_len))
    prefill_token_ids = list(range(prefill_len))

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)
    block_count = _warmup_block_counter(model_runner)
    kv_cache_specs = [g.kv_cache_spec for g in kv_cache_groups]
    decode_prefill_block_counts = [
        block_count(decode_prompt_len, s) for s in kv_cache_specs
    ]
    decode_block_counts = [
        block_count(decode_prompt_len + decode_scheduled_tokens, s)
        for s in kv_cache_specs
    ]
    decode_block_deltas = [
        decode - prefill
        for decode, prefill in zip(decode_block_counts, decode_prefill_block_counts)
    ]
    prefill_block_counts = [block_count(prefill_len, s) for s in kv_cache_specs]
    required_blocks = sum(decode_block_counts) + sum(prefill_block_counts)
    if model_runner.kv_cache_config.num_blocks <= required_blocks:
        logger.warning(
            "Skipping V2 mixed prefill+decode warmup because only %d KV blocks "
            "are available for %d required warmup blocks.",
            model_runner.kv_cache_config.num_blocks,
            required_blocks,
        )
        return False

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        block_ids = list(range(next_block_id, next_block_id + num_blocks))
        next_block_id += num_blocks
        return block_ids

    sampling_params = SamplingParams(max_tokens=2, temperature=0.0)

    decode_prefill_output = SchedulerOutput.make_empty()
    decode_prefill_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=decode_req_id,
            prompt_token_ids=decode_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(_alloc_blocks(n) for n in decode_prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=decode_token_ids,
        ),
    ]
    decode_prefill_output.num_scheduled_tokens = {
        decode_req_id: decode_prompt_len,
    }
    decode_prefill_output.total_num_scheduled_tokens = decode_prompt_len
    decode_prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    decode_new_blocks = tuple(_alloc_blocks(n) for n in decode_block_deltas)
    cached_decode_req = CachedRequestData.make_empty()
    cached_decode_req.req_ids = [decode_req_id]
    cached_decode_req.num_computed_tokens = [decode_prompt_len]
    cached_decode_req.num_output_tokens = [1]
    cached_decode_req.new_block_ids = [
        decode_new_blocks if any(decode_block_deltas) else None
    ]

    mixed_output = SchedulerOutput.make_empty()
    mixed_output.scheduled_cached_reqs = cached_decode_req
    mixed_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=prefill_req_id,
            prompt_token_ids=prefill_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=prefill_token_ids,
        ),
    ]
    mixed_output.num_scheduled_tokens = {
        decode_req_id: decode_scheduled_tokens,
        prefill_req_id: prefill_len,
    }
    mixed_output.total_num_scheduled_tokens = num_tokens
    mixed_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {decode_req_id, prefill_req_id}

    context = mixed_step_context or nullcontext()
    model_runner.kv_connector.set_disabled(True)
    try:
        worker_execute_model(decode_prefill_output)
        worker_sample_tokens(None)
        with context:
            worker_execute_model(mixed_output)
            worker_sample_tokens(None)
        worker_execute_model(cleanup_output)
    finally:
        model_runner.kv_connector.set_disabled(False)
    return True


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run scheduler-realistic prefill and decode steps to JIT compile kernels.

    We must call the provided worker's execute_model for pipeline parallel
    coordination.
    """
    if model_runner.is_encoder_only:
        return

    num_spec_steps = model_runner.num_speculative_steps
    decode_query_len = model_runner.decode_query_len
    # Use decode_query_len + 1 tokens so the prefill batch's per-request query
    # length exceeds decode_query_len, preventing it from being misclassified as
    # a uniform decode batch.
    prompt_len = decode_query_len + 1
    prompt_token_ids = list(range(prompt_len))
    # Upper bound on the decode steps built in `decode_steps` below.
    num_decode_steps = 1
    if not model_runner.is_pooling_model:
        num_decode_steps = 5 if num_spec_steps > 0 else 3
    # Size the block allocation for the worst case: every request advancing
    # decode_query_len tokens on every decode step.
    decode_len = prompt_len + num_decode_steps * decode_query_len

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    # Encoder-decoder models: give each warmup request a dummy encoder input so
    # cross-attention warms up over a realistic, non-empty key sequence.
    # The dummy mm_feature is registered in the encoder cache and only its encoder
    # length is read (not the inputs themselves); the encoder itself is not scheduled.
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

    # Compute per-request block counts for each KV cache group.
    block_count = _warmup_block_counter(model_runner)
    kv_cache_specs = [g.kv_cache_spec for g in kv_cache_groups]
    prefill_block_counts = [block_count(prompt_len, s) for s in kv_cache_specs]
    decode_block_counts = [block_count(decode_len, s) for s in kv_cache_specs]
    max_blocks_per_req = sum(decode_block_counts)

    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, decode_query_len),
    )
    if max_blocks_per_req > 0:
        # Reserve block 0 (null block) and ensure we have enough blocks.
        # Encoder-only models allocate no KV blocks, so this cap doesn't apply.
        num_reqs = min(
            num_reqs,
            max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
        )

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
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

    # Assign distinct block IDs per request per group. 0 null block, start from 1.
    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    # The KV-block zeroing kernel is driven by the scheduler's
    # new_block_ids_to_zero, so none of the steps below reach it.
    if model_runner.kv_block_zeroer is not None:
        model_runner.kv_block_zeroer.warmup(model_runner.kv_cache_config.num_blocks)

    # Step 1: Prefill all requests with 1 + decode_query_len prompt tokens each.
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
                structured_output_request_ids=req_ids,
                grammar_bitmask=grammar_bitmask,
                num_spec_tokens=[0] * len(req_ids),
            )

        worker_sample_tokens(grammar_output)

        # Per-request state carried across the decode steps.
        req_computed = [prompt_len] * num_reqs
        req_blocks = [list(prefill_block_counts) for _ in range(num_reqs)]

        def _run_decode_step(indices: list[int], spec_flags: list[bool]) -> None:
            """Decode `indices`, spec-decoding the ones flagged in `spec_flags`."""
            cached_req_data = CachedRequestData.make_empty()
            cached_req_data.req_ids = [req_ids[i] for i in indices]
            cached_req_data.num_computed_tokens = [req_computed[i] for i in indices]
            cached_req_data.num_output_tokens = [1] * len(indices)
            cached_req_data.new_block_ids = []

            step_num_scheduled_tokens: dict[str, int] = {}
            step_spec_tokens: dict[str, list[int]] = {}
            for i, use_spec in zip(indices, spec_flags):
                num_tokens = decode_query_len if use_spec else 1
                after = req_computed[i] + num_tokens
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
                step_num_scheduled_tokens[req_ids[i]] = num_tokens
                if use_spec:
                    step_spec_tokens[req_ids[i]] = [0] * num_spec_steps

            decode_output = SchedulerOutput.make_empty()
            decode_output.scheduled_cached_reqs = cached_req_data
            decode_output.num_scheduled_tokens = step_num_scheduled_tokens
            decode_output.scheduled_spec_decode_tokens = step_spec_tokens
            decode_output.total_num_scheduled_tokens = sum(
                step_num_scheduled_tokens.values()
            )
            decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

            worker_execute_model(decode_output)
            worker_sample_tokens(None)

            for i, use_spec in zip(indices, spec_flags):
                req_computed[i] += decode_query_len if use_spec else 1

        all_indices = list(range(num_reqs))
        use_spec_decode = num_spec_steps > 0

        # Decode steps to warm, as (request indices, per-request spec flag).
        # Under spec decoding the scheduler drops requests the drafter proposed
        # nothing for, so warm each batch shape with and without draft tokens.
        decode_steps: list[tuple[list[int], list[bool]]] = [
            (all_indices, [use_spec_decode] * num_reqs),
        ]
        if num_reqs >= 2:
            # Mixed spec / non-spec: GDN and KDA reclassify the non-spec decode
            # as a prefill and split the batch into spec/non-spec token indices.
            decode_steps.append(([0, 1], [use_spec_decode, False]))
            if use_spec_decode:
                # Exercise the model paths that split a batch by whether each
                # request received draft tokens.
                decode_steps.append(([0, 1], [False, False]))
        if num_reqs > 1:
            decode_steps.append(([0], [use_spec_decode]))
            if use_spec_decode:
                decode_steps.append(([0], [False]))
        elif use_spec_decode:
            decode_steps.append(([0], [False]))

        for step_indices, step_spec_flags in decode_steps:
            _run_decode_step(step_indices, step_spec_flags)

    # Clean up - process finish_req_ids.
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    worker_execute_model(cleanup_output)
    model_runner.kv_connector.set_disabled(False)
    torch.accelerator.synchronize()
