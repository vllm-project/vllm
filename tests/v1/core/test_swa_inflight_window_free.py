# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-window block frees vs in-flight GPU steps.

With async scheduling / PP, `num_computed_tokens` optimistically includes
tokens of unprocessed steps whose attention windows still read the blocks
just below the optimistic boundary (and rejected spec tokens can roll it
back), so `allocate_slots` frees on the processed-token basis:
`num_computed_tokens - num_in_flight_tokens`.
"""

import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec, SlidingWindowSpec
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler, mock_kv

NUM_PROMPT_TOKENS = 100
BLOCK_SIZE = 16
SLIDING_WINDOW = 16
# Tokens 0..84 are outside the window of the next token to compute
# (100 - 16 + 1 = 85), i.e. 5 full blocks.
NUM_OUT_OF_WINDOW_BLOCKS = 85 // BLOCK_SIZE
# Chunked-local skips whole chunks left of the current one:
# (100 // 32) * 32 = 96 settled tokens -> 6 full blocks.
CHUNK_SIZE = 32
NUM_OUT_OF_CHUNK_BLOCKS = (NUM_PROMPT_TOKENS // CHUNK_SIZE) * CHUNK_SIZE // BLOCK_SIZE


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
    token_id: int = 0,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _create_swa_scheduler(async_scheduling: bool):
    return create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=async_scheduling,
        kv_cache_spec=SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=SLIDING_WINDOW,
        ),
    )


def _create_chunked_scheduler(async_scheduling: bool):
    return create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=async_scheduling,
        kv_cache_spec=ChunkedLocalAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            attention_chunk_size=CHUNK_SIZE,
        ),
    )


def _num_null_blocks(scheduler, request_id: str) -> int:
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    null_block = manager._null_block
    return sum(1 for b in manager.req_to_blocks[request_id] if b is null_block)


def test_num_in_flight_tokens_accounting():
    scheduler = create_scheduler(async_scheduling=True)
    request = create_requests(num_requests=1, num_tokens=NUM_PROMPT_TOKENS)[0]
    scheduler.add_request(request)

    out0 = scheduler.schedule()
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS
    # Async: decode scheduled before the prefill output is processed.
    out1 = scheduler.schedule()
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS + 1

    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert request.num_in_flight_tokens == 1
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert request.num_in_flight_tokens == 0


def test_swa_free_waits_for_in_flight_step():
    """Async: out-of-window blocks stay allocated until the step that still
    reads them has been processed."""
    scheduler = _create_swa_scheduler(async_scheduling=True)
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    req_id = request.request_id
    block_pool = scheduler.kv_cache_manager.block_pool

    out0 = scheduler.schedule()  # prefill, in flight from here on
    free_after_prefill = block_pool.get_num_free_blocks()

    # Decode scheduled while the prefill still reads the out-of-window blocks:
    # they must not be freed yet.
    out1 = scheduler.schedule()
    assert _num_null_blocks(scheduler, req_id) == 0
    assert block_pool.get_num_free_blocks() == free_after_prefill

    # Prefill output processed; the next allocate frees the out-of-window
    # blocks.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    scheduler.schedule()
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_WINDOW_BLOCKS
    assert (
        block_pool.get_num_free_blocks()
        == free_after_prefill + NUM_OUT_OF_WINDOW_BLOCKS
    )
    # Not double-freed on the following steps.
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    scheduler.schedule()
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_WINDOW_BLOCKS


def test_swa_free_immediate_when_sync():
    """Sync: no in-flight step at schedule time, frees happen at the first
    decode allocation as before."""
    scheduler = _create_swa_scheduler(async_scheduling=False)
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    req_id = request.request_id

    out0 = scheduler.schedule()
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert request.num_in_flight_tokens == 0

    scheduler.schedule()
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_WINDOW_BLOCKS


def test_swa_admission_cap_accounts_for_overlapping_batches():
    spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=1024,
    )
    base = spec.max_admission_blocks_per_request(
        max_num_batched_tokens=1024, max_model_len=16384
    )
    # (1024 - 1 + 1024) tokens -> 128 blocks, +1 for window misalignment.
    assert base == 129
    overlapped = spec.max_admission_blocks_per_request(
        max_num_batched_tokens=1024, max_model_len=16384, max_concurrent_batches=2
    )
    # One extra in-flight chunk is held back: (1024 - 1 + 2 * 1024) tokens.
    assert overlapped == 193


def test_chunked_local_free_waits_for_in_flight_step():
    """Chunked-local attention frees whole chunks left of the current one, and
    is exposed to the same load-WAR: with async scheduling those chunks must
    stay allocated until the in-flight step that still reads them settles."""
    scheduler = _create_chunked_scheduler(async_scheduling=True)
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    req_id = request.request_id
    block_pool = scheduler.kv_cache_manager.block_pool

    out0 = scheduler.schedule()  # prefill, in flight from here on
    free_after_prefill = block_pool.get_num_free_blocks()

    # Decode scheduled while the prefill still reads the out-of-chunk blocks.
    out1 = scheduler.schedule()
    assert _num_null_blocks(scheduler, req_id) == 0
    assert block_pool.get_num_free_blocks() == free_after_prefill

    # Prefill output processed; the next allocate frees the out-of-chunk blocks.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    scheduler.schedule()
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_CHUNK_BLOCKS
    assert (
        block_pool.get_num_free_blocks() == free_after_prefill + NUM_OUT_OF_CHUNK_BLOCKS
    )
    # Not double-freed on the following steps.
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    scheduler.schedule()
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_CHUNK_BLOCKS


def test_chunked_local_admission_cap_accounts_for_overlapping_batches():
    spec = ChunkedLocalAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=1024,
    )
    base = spec.max_admission_blocks_per_request(
        max_num_batched_tokens=1024, max_model_len=16384
    )
    # (1024 + 1024) tokens -> 128 blocks.
    assert base == 128
    overlapped = spec.max_admission_blocks_per_request(
        max_num_batched_tokens=1024, max_model_len=16384, max_concurrent_batches=2
    )
    # One extra in-flight chunk is held back: (1024 + 2 * 1024) tokens.
    assert overlapped == 192


def test_connector_finish_frees_on_settled_basis():
    """The out-of-window prune done at request finish, before the block table
    is handed to a KV connector (simple CPU offload / NIXL store), must use the
    same processed-token basis. Otherwise the connector reads/hands off a block
    the still-in-flight step is optimistically counted as done with, which is
    the load-path WAR this fix closes."""
    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
        kv_cache_spec=SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=SLIDING_WINDOW,
        ),
    )
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    req_id = request.request_id

    out0 = scheduler.schedule()  # prefill, in flight from here on
    scheduler.schedule()  # decode over-scheduled: num_computed_tokens optimistic

    # Finishing now (connector store) must NOT prune out-of-window blocks the
    # in-flight prefill still reads.
    scheduler._connector_finished(request)
    assert _num_null_blocks(scheduler, req_id) == 0

    # Once the in-flight step settles, the same prune releases them.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    scheduler._connector_finished(request)
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_WINDOW_BLOCKS
