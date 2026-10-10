# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Sequence
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
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    CrossAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
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
    if isinstance(kvcache_spec, UniformTypeKVCacheSpecs):
        kvcache_spec = kvcache_spec.first_spec
    if isinstance(kvcache_spec, CircularBufferSpec):
        # Circular caches keep one physical ring block for the request lifetime.
        return 1
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

    sampling_params = SamplingParams(
        max_tokens=2,
        temperature=0.0,
        watermarking=False,
    )

    decode_prefill_output = SchedulerOutput.make_empty()
    decode_prefill_output.num_spec_tokens_to_schedule = (
        model_runner.num_speculative_steps
    )
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
    mixed_output.num_spec_tokens_to_schedule = model_runner.num_speculative_steps
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
    # Adaptive costs are calibrated during capture, after this warmup. Exercise
    # fixed draft counts here, then restore the manager for capture and serving.
    adaptive_verification = model_runner.adaptive_verification
    model_runner.adaptive_verification = None
    rejection_sampler = model_runner.rejection_sampler
    adaptive_sampling = (
        rejection_sampler is not None and rejection_sampler.enable_adaptive_verification
    )
    if adaptive_sampling:
        assert rejection_sampler is not None
        rejection_sampler.enable_adaptive_verification = False
    try:
        _warmup_kernels(model_runner, worker_execute_model, worker_sample_tokens)
    finally:
        model_runner.adaptive_verification = adaptive_verification
        if adaptive_sampling:
            assert rejection_sampler is not None
            rejection_sampler.enable_adaptive_verification = True


def _warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    if model_runner.vllm_config.is_mm_encoder_only:
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
    block_tables = getattr(model_runner, "block_tables", None)
    null_blocks = block_tables is not None and (
        block_tables.redirect_writes_to_null_block
    )
    if max_blocks_per_req > 0 and not null_blocks:
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
    prefill_output.num_spec_tokens_to_schedule = num_spec_steps
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
            decode_output.num_spec_tokens_to_schedule = num_spec_steps
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
    if model_runner.kv_block_zeroer is not None:
        model_runner.kv_block_zeroer.zero_block_ids([0])
    torch.accelerator.synchronize()


# (decode requests, prefill lengths) of one batch-size warmup step.
BatchSizeWarmupStep = tuple[int, tuple[int, ...]]


def plan_batch_size_warmup(
    *,
    decode_query_len: int,
    max_num_reqs: int,
    token_budget: int,
    max_prompt_len: int,
    cudagraph_capture_sizes: Sequence[int],
) -> list[BatchSizeWarmupStep]:
    """Steps that reach every decode batch shape and logits row count once.

    With `q = decode_query_len` and prompts of `q + 1` tokens:

    1. Prefill a pool of `pool` requests.
    2. For `r = pool .. 1`: decode the first `r` pool requests (`q * r`
       logits rows, every uniform decode CUDA graph). With speculative
       decoding, also decode `r - 1` of them next to `b = 1 .. q - 1` new
       prefills (`q * (r - 1) + b` rows), so every row count a step can have
       up to `q * pool` is reached.
    3. One prefill of exactly `s` tokens for every CUDA graph capture size
       `s` (the mixed-batch graphs), except `s = q`, a one-request decode.

    The decode count never increases, so each step can retire the pool
    requests it no longer decodes. The plan depends only on the config, so
    every TP rank runs the same steps and collectives.
    """
    q = decode_query_len
    prompt_len = q + 1
    pool = min(max_num_reqs, token_budget // prompt_len)
    if pool < 1:
        return []
    steps: list[BatchSizeWarmupStep] = [(0, (prompt_len,) * pool)]
    for r in range(pool, 0, -1):
        steps.append((r, ()))
        for b in range(1, q):
            if r - 1 + b <= max_num_reqs and q * (r - 1) + b * prompt_len <= (
                token_budget
            ):
                steps.append((r - 1, (prompt_len,) * b))
    max_size = min(token_budget, max_prompt_len)
    steps += [
        (0, (size,))
        for size in sorted(set(cudagraph_capture_sizes))
        if size != q and size <= max_size
    ]
    return steps


def _batch_size_warmup_skip_reason(model_runner: GPUModelRunner) -> str | None:
    parallel_config = model_runner.parallel_config
    if model_runner.is_pooling_model:
        return "pooling model"
    if model_runner.vllm_config.is_mm_encoder_only or model_runner.is_encoder_decoder:
        return "encoder model"
    if parallel_config.pipeline_parallel_size > 1:
        return "pipeline parallel"
    if parallel_config.data_parallel_size > 1:
        # DP engines may size different KV caches and so plan different steps.
        return "data parallel"
    if model_runner.pcp_manager is not None:
        return "prefill context parallel"
    if model_runner.adaptive_verification is not None:
        return "adaptive verification"
    if model_runner.num_speculative_steps > 0 and model_runner.speculator is None:
        return "speculative decoding without a speculator"
    return None


def _default_sampling_params(model_runner: GPUModelRunner) -> SamplingParams:
    """The sampling parameters requests get by default (generation config)."""
    defaults = model_runner.model_config.get_diff_sampling_param()
    keys = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    )
    return SamplingParams(
        **{k: defaults[k] for k in keys if defaults.get(k) is not None}
    )


class _BatchSizeWarmupBuilder:
    """Turns `BatchSizeWarmupStep`s into `SchedulerOutput`s.

    Tracks each request's computed tokens and KV blocks, reserving what the
    scheduler would (lookahead included). Blocks start at 1 and are recycled
    once their request has finished.
    """

    def __init__(
        self,
        model_runner: GPUModelRunner,
        sampling_params: SamplingParams,
    ) -> None:
        self.q = model_runner.decode_query_len
        self.num_spec_steps = model_runner.num_speculative_steps
        kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
        self.specs = [g.kv_cache_spec for g in kv_cache_groups]
        self.block_count = _warmup_block_counter(model_runner)
        self.sampling_params = sampling_params
        self.vocab_size = model_runner.model_config.get_vocab_size()
        self.next_block = 1
        self.free_blocks: list[int] = []
        self.num_new_reqs = 0
        # The first step's prefills form the decode pool; later prefills
        # finish at the start of the next step.
        self.pool_filled = False
        self.pool: list[str] = []
        self.finishing: list[str] = []
        # req_id -> (computed tokens, block ids held per KV cache group)
        self.reqs: dict[str, tuple[int, list[list[int]]]] = {}
        self.max_context = 0

    def _alloc(self, n: int) -> list[int]:
        ids = self.free_blocks[:n]
        del self.free_blocks[:n]
        extra = n - len(ids)
        ids += range(self.next_block, self.next_block + extra)
        self.next_block += extra
        return ids

    def _finish(self, req_ids: list[str]) -> set[str]:
        for req_id in req_ids:
            _, held = self.reqs.pop(req_id)
            self.free_blocks += [b for ids in held for b in ids]
        return set(req_ids)

    def build(self, step: BatchSizeWarmupStep) -> SchedulerOutput:
        num_decode, prefill_lens = step
        assert num_decode <= len(self.pool)
        q = self.q
        out = SchedulerOutput.make_empty()
        out.finished_req_ids = self._finish(self.finishing + self.pool[num_decode:])
        self.finishing, self.pool = [], self.pool[:num_decode]
        out.num_common_prefix_blocks = [0] * len(self.specs)

        cached = CachedRequestData.make_empty()
        for req_id in self.pool:
            computed, held = self.reqs[req_id]
            after = computed + q
            deltas = [
                self.block_count(after, spec) - len(ids)
                for spec, ids in zip(self.specs, held)
            ]
            new_ids = tuple(self._alloc(n) for n in deltas)
            cached.req_ids.append(req_id)
            cached.num_computed_tokens.append(computed)
            cached.num_output_tokens.append(1)
            cached.new_block_ids.append(new_ids if any(deltas) else None)
            self.reqs[req_id] = (after, [a + b for a, b in zip(held, new_ids)])
            self.max_context = max(self.max_context, after)
            out.num_scheduled_tokens[req_id] = q
            if self.num_spec_steps > 0:
                out.scheduled_spec_decode_tokens[req_id] = [0] * self.num_spec_steps
        out.scheduled_cached_reqs = cached

        for prompt_len in prefill_lens:
            req_id = f"_batch_size_warmup_{self.num_new_reqs}_"
            self.num_new_reqs += 1
            token_ids = [i % self.vocab_size for i in range(prompt_len)]
            block_ids = [
                self._alloc(self.block_count(prompt_len, spec)) for spec in self.specs
            ]
            out.scheduled_new_reqs.append(
                NewRequestData(
                    req_id=req_id,
                    prompt_token_ids=token_ids,
                    mm_features=[],
                    sampling_params=self.sampling_params,
                    pooling_params=None,
                    block_ids=tuple(block_ids),
                    num_computed_tokens=0,
                    lora_request=None,
                    prefill_token_ids=token_ids,
                )
            )
            self.reqs[req_id] = (prompt_len, block_ids)
            self.max_context = max(self.max_context, prompt_len)
            out.num_scheduled_tokens[req_id] = prompt_len
            (self.finishing if self.pool_filled else self.pool).append(req_id)

        self.pool_filled = True
        out.total_num_scheduled_tokens = sum(out.num_scheduled_tokens.values())
        return out

    def build_cleanup(self) -> SchedulerOutput:
        out = SchedulerOutput.make_empty()
        out.finished_req_ids = self._finish(list(self.reqs))
        self.pool, self.finishing = [], []
        return out


@torch.inference_mode()
def warmup_batch_sizes(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run every decode batch size and logits row count once before serving.

    Runs after CUDA graph capture, so the first replay of every captured
    graph and the first eager logits GEMM, logits gather and sampler call at
    every row count happen here rather than on a live request. Kernel
    selection, lazy module loading and graph upload on that first use can
    otherwise stall serving for seconds, or stall one TP rank while its peers
    wait in a collective.
    """
    reason = _batch_size_warmup_skip_reason(model_runner)
    if reason is not None:
        logger.warning("Skipping batch-size warmup: %s.", reason)
        return

    scheduler_config = model_runner.scheduler_config
    token_budget = min(
        scheduler_config.max_num_batched_tokens,
        scheduler_config.max_num_scheduled_tokens
        or scheduler_config.max_num_batched_tokens,
        model_runner.max_num_tokens,
    )
    max_prompt_len = (
        model_runner.max_model_len - model_runner.vllm_config.num_lookahead_tokens
    )
    compilation_config = model_runner.compilation_config
    capture_sizes = (
        compilation_config.cudagraph_capture_sizes or []
        if compilation_config.cudagraph_mode
        else []
    )
    steps = plan_batch_size_warmup(
        decode_query_len=model_runner.decode_query_len,
        max_num_reqs=model_runner.max_num_reqs,
        token_budget=token_budget,
        max_prompt_len=max_prompt_len,
        cudagraph_capture_sizes=capture_sizes,
    )
    sampling_params = _default_sampling_params(model_runner)

    # Dry run on the CPU: the KV blocks and context length the steps need.
    dry_run = _BatchSizeWarmupBuilder(model_runner, sampling_params)
    for step in steps:
        dry_run.build(step)
    num_blocks = model_runner.kv_cache_config.num_blocks
    if (
        not steps
        or dry_run.next_block > num_blocks
        or dry_run.max_context > max_prompt_len
    ):
        logger.warning(
            "Skipping batch-size warmup: %d steps need %d KV blocks (%d exist) "
            "and %d tokens of context (%d allowed).",
            len(steps),
            dry_run.next_block,
            num_blocks,
            dry_run.max_context,
            max_prompt_len,
        )
        return

    builder = _BatchSizeWarmupBuilder(model_runner, sampling_params)
    start = time.perf_counter()
    model_runner.kv_connector.set_disabled(True)
    try:
        for step in steps:
            worker_execute_model(builder.build(step))
            output = worker_sample_tokens(None)
            if hasattr(output, "get_output"):
                # As the async output thread does while serving.
                output.get_output()
        worker_execute_model(builder.build_cleanup())
    finally:
        model_runner.kv_connector.set_disabled(False)
    if model_runner.kv_block_zeroer is not None:
        model_runner.kv_block_zeroer.zero_block_ids([0])
    torch.accelerator.synchronize()
    logger.info(
        "Batch-size warmup ran %d steps (decode batches of 1 to %d requests) "
        "in %.1f s.",
        len(steps),
        len(steps[0][1]),
        time.perf_counter() - start,
    )
