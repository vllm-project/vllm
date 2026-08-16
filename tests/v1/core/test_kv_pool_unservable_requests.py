# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requests the KV cache pool can never hold (#52520).

`_check_enough_kv_cache_memory` sizes the pool from each spec's
`max_memory_usage_bytes`, but `BlockPool` permanently reserves one block as the
null block, so a pool built at exactly that minimum is one block short of one
`max_model_len` request. Runtime admission used its own incremental estimate and
admitted such a request anyway: it prefilled to ~99 % of the pool, was
descheduled with that work discarded, and was never scheduled again -- a stall
whose only signal is a `waiting` gauge.

`Scheduler.max_servable_num_tokens` now derives that limit from the same
`kv_cache_utils` helper the startup check uses. It bounds a request from both
ends: a prompt already past it is failed at admission instead of being retried,
and a generate request that would grow past it stops there as a length cap
instead of being preempted at that token into a queue it can no longer leave.
"""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests

pytestmark = pytest.mark.cpu_test

# Shape of the deployment the stall was reported on (hybrid GDN + full attention,
# MTP depth 3, `--mamba-cache-mode align`), scaled down to unit-test sizes.
BLOCK_SIZE = 16
MAX_MODEL_LEN = 64 * BLOCK_SIZE  # 1024
NUM_SPEC_BLOCKS = 3
MAX_NUM_BATCHED_TOKENS = 16 * BLOCK_SIZE  # 256; forces a chunked prefill

# Blocks the startup check demands: `cdiv(max_model_len, block_size)` for the
# full-attention group plus `2 + num_speculative_blocks` for the mamba group in
# align mode (`MambaSpec.max_memory_usage_bytes`).
MAMBA_BLOCKS = 2 + NUM_SPEC_BLOCKS
STARTUP_MIN_BLOCKS = cdiv(MAX_MODEL_LEN, BLOCK_SIZE) + MAMBA_BLOCKS  # 69

# One of those blocks is the null block, so a request can use at most
# `STARTUP_MIN_BLOCKS - 1 - MAMBA_BLOCKS` blocks of attention KV.
SERVABLE_LEN = (STARTUP_MIN_BLOCKS - 1 - MAMBA_BLOCKS) * BLOCK_SIZE  # 1008


def _full_attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=NUM_SPEC_BLOCKS,
        # Hybrid pools pad every group's page up to a common size; the bound
        # helpers require that uniformity.
        page_size_padded=_full_attention_spec().page_size_bytes,
    )


def _make_scheduler(num_blocks: int) -> Scheduler:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=MAX_MODEL_LEN,
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_model_len=MAX_MODEL_LEN,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        speculative_config=SpeculativeConfig(
            model="ngram",
            method="ngram",
            num_speculative_tokens=NUM_SPEC_BLOCKS,
            prompt_lookup_max=NUM_SPEC_BLOCKS,
            prompt_lookup_min=1,
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full_attn"], _full_attention_spec()),
            KVCacheGroupSpec(["mamba"], _mamba_spec()),
        ],
    )
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        log_stats=True,
    )


def _model_output(scheduler_output, sample: bool) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[1000] if sample else [] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _run(scheduler: Scheduler, request, max_steps: int = 64) -> int:
    """Step until the request finishes; return the peak computed token count."""
    peak = 0
    for _ in range(max_steps):
        scheduler_output = scheduler.schedule()
        peak = max(peak, request.num_computed_tokens)
        scheduler.update_from_output(
            scheduler_output,
            _model_output(
                scheduler_output,
                request.num_computed_tokens >= request.num_prompt_tokens,
            ),
        )
        if request.is_finished():
            break
    return peak


def test_max_servable_num_tokens_reserves_the_null_block():
    """The bound must exclude the null block and the mamba group's own blocks."""
    scheduler = _make_scheduler(STARTUP_MIN_BLOCKS)
    assert scheduler.max_servable_num_tokens == SERVABLE_LEN
    # A pool built at the startup minimum cannot serve the whole window: that
    # asymmetry is what admission has to see.
    assert scheduler.max_servable_num_tokens < MAX_MODEL_LEN
    # One more block covers it, and then the bound must not bite at all.
    assert _make_scheduler(STARTUP_MIN_BLOCKS + 1).max_servable_num_tokens >= (
        MAX_MODEL_LEN
    )


def test_unservable_request_is_failed_not_prefilled_and_retried():
    """A request past the bound must never be scheduled, and must not linger.

    Before the fix it was admitted, chunk-prefilled to `SERVABLE_LEN` tokens,
    descheduled when the last chunk could not be allocated, and then refused by
    the same gate on every later step -- zero output tokens, forever.
    """
    scheduler = _make_scheduler(STARTUP_MIN_BLOCKS)
    [request] = create_requests(
        num_requests=1,
        num_tokens=MAX_MODEL_LEN - 1,
        max_tokens=1,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    assert not scheduler_output.num_scheduled_tokens
    assert request.num_computed_tokens == 0

    scheduler.update_from_output(
        scheduler_output, _model_output(scheduler_output, False)
    )
    assert request.status == RequestStatus.FINISHED_IGNORED
    assert request.num_preemptions == 0
    assert not scheduler.has_unfinished_requests()


def test_request_at_the_bound_is_failed_because_it_needs_an_output_slot():
    """`max_servable_num_tokens` is a sequence length, not a prompt length."""
    scheduler = _make_scheduler(STARTUP_MIN_BLOCKS)
    [request] = create_requests(
        num_requests=1,
        num_tokens=SERVABLE_LEN,
        max_tokens=1,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    assert not scheduler_output.num_scheduled_tokens
    scheduler.update_from_output(
        scheduler_output, _model_output(scheduler_output, False)
    )
    assert request.status == RequestStatus.FINISHED_IGNORED


def test_servable_request_at_the_same_pool_still_runs():
    """The bound must not reject a request the pool can actually hold."""
    scheduler = _make_scheduler(STARTUP_MIN_BLOCKS)
    [request] = create_requests(
        num_requests=1,
        num_tokens=SERVABLE_LEN - 1,
        max_tokens=1,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    peak = _run(scheduler, request)
    assert peak == SERVABLE_LEN - 1
    assert request.num_preemptions == 0
    assert request.num_output_tokens == 1
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_pool_with_headroom_serves_the_whole_window():
    """One block of headroom restores the pre-fix behaviour exactly."""
    scheduler = _make_scheduler(STARTUP_MIN_BLOCKS + 1)
    [request] = create_requests(
        num_requests=1,
        num_tokens=MAX_MODEL_LEN - 1,
        max_tokens=1,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    peak = _run(scheduler, request)
    assert peak == MAX_MODEL_LEN - 1
    assert request.num_preemptions == 0
    assert request.num_output_tokens == 1
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


# The decode case. A prompt this long clears the frontend's
# `prompt + max_tokens <= max_model_len` check and admission's
# `prompt + 1 <= max_servable_num_tokens` check, and then generation walks the
# sequence past the ceiling: there are only `SERVABLE_LEN - DECODE_PROMPT_LEN`
# output slots under it, out of `DECODE_MAX_TOKENS` asked for.
DECODE_PROMPT_LEN = SERVABLE_LEN - 8  # 1000
DECODE_MAX_TOKENS = MAX_MODEL_LEN - DECODE_PROMPT_LEN  # 24


def test_generation_stops_at_the_bound_instead_of_stalling_mid_decode():
    """A generate request must not be preempted into a queue it cannot leave.

    Both length checks pass, so the request runs. At output token 9 the
    sequence would need a block the pool does not have. Unfixed, it is
    preempted there -- and is then longer than the pool can re-prefill, so it
    is never scheduled again: output frozen mid-answer, one preemption, and
    `waiting` non-empty forever. The pool's ceiling is a length cap, so the
    request must stop on it with the tokens it produced.
    """
    scheduler = _make_scheduler(STARTUP_MIN_BLOCKS)
    [request] = create_requests(
        num_requests=1,
        num_tokens=DECODE_PROMPT_LEN,
        max_tokens=DECODE_MAX_TOKENS,
        ignore_eos=True,
        block_size=BLOCK_SIZE,
    )
    # Precisely the case the admission gate lets through.
    assert request.num_tokens + 1 <= scheduler.max_servable_num_tokens
    assert request.num_tokens + DECODE_MAX_TOKENS > scheduler.max_servable_num_tokens
    scheduler.add_request(request)

    peak = _run(scheduler, request)

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.num_tokens == SERVABLE_LEN
    assert request.num_output_tokens == SERVABLE_LEN - DECODE_PROMPT_LEN
    assert request.num_preemptions == 0
    assert peak <= SERVABLE_LEN
    assert not scheduler.has_unfinished_requests()


def test_generation_is_not_capped_when_the_pool_covers_the_window():
    """The cap must not shorten an answer the pool can actually hold.

    Same request, one block of headroom: it must produce every token it asked
    for, so the bound cannot be silently truncating generation in the normal
    case.
    """
    scheduler = _make_scheduler(STARTUP_MIN_BLOCKS + 1)
    assert scheduler.max_servable_num_tokens >= MAX_MODEL_LEN
    [request] = create_requests(
        num_requests=1,
        num_tokens=DECODE_PROMPT_LEN,
        max_tokens=DECODE_MAX_TOKENS,
        ignore_eos=True,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    _run(scheduler, request)

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.num_output_tokens == DECODE_MAX_TOKENS
    assert request.num_tokens == MAX_MODEL_LEN
    assert request.num_preemptions == 0
