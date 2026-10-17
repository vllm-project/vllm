# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import FinishReason, RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def test_basic_lifecycle():
    """Test lifecycle of a Remote Decode request."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        max_tokens=1,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
    )

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1): Prefill.
    # (1a): schedule()
    scheduler_output = scheduler.schedule()
    assert len(scheduler.requests) == 1
    assert len(scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1

    # (1b): execute_model()
    model_runner_output = create_model_runner_output(reqs=[request])

    # (1c): update_from_output()
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output
    )

    # Ensure the request is finished after 1 token.
    assert request.is_finished()
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    output = engine_core_outputs[0].outputs[0]
    assert output.finish_reason == FinishReason.LENGTH
    assert output.kv_transfer_params is not None

    # Request freed in Scheduler and in Persistent Batch ...
    assert request_id in scheduler.finished_req_ids
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0

    # ... but blocks should not be freed.
    assert len(scheduler.requests) == 1
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks[request_id]
    for block in blocks:
        assert block.ref_cnt == 1

    # STEP (2): Send Finished to PB.
    # (2a): schedule() - pass finished request to PB.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.requests) == 1
    assert len(scheduler.running) == 0
    assert len(scheduler_output.finished_req_ids) == 1
    assert request_id in scheduler_output.finished_req_ids
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 0
    assert len(scheduler.finished_req_ids) == 0

    # (2b): execute_model()
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT

    # (2c): update_from_output()
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP (3): Finished sending.
    # (3a): schedule() - pass finished request to PB.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.requests) == 1
    assert len(scheduler.running) == 0
    assert len(scheduler_output.finished_req_ids) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 0
    assert len(scheduler.finished_req_ids) == 0

    # (3b): execute_model()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending={request_id}
    )

    # (3c): update_from_output()
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # Confirm we do not have any memory leaks after req lifecycle.
    assert_scheduler_empty(scheduler)


def test_short_prompt_lifecycle():
    """Test lifecycle of a Remote Decode request with short prompt."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Not enough tokens for full block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = BLOCK_SIZE // 2
    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        max_tokens=1,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
    )

    scheduler.add_request(request)

    # STEP (1): Prefill.
    # (1a): schedule()
    scheduler_output = scheduler.schedule()
    assert len(scheduler.requests) == 1
    assert len(scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1

    # (1b): execute_model()
    model_runner_output = create_model_runner_output(reqs=[request])

    # (1c): update_from_output()
    # Even though tokens < block_size, there will be kv xfer for partial block.
    eco = scheduler.update_from_output(scheduler_output, model_runner_output)
    kv_transfer_params = eco[0].outputs[0].kv_transfer_params

    assert len(kv_transfer_params["remote_block_ids"]) == 1

    # Confirm we do not have any memory leaks after req lifecycle.
    # We need to mark sending finish to clear data for persistent batch.
    scheduler_output = scheduler.schedule()
    # Use create_model_runner_output to pass kv_connector_output along
    model_runner_output = create_model_runner_output(
        reqs=[request], finished_sending={request.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


def test_prefix_cache_lifecycle():
    """Test that remote decode params still work with a prefix cache hit."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Prime the KVCache.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 3
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request_normal = create_request(
        request_id=1, block_size=BLOCK_SIZE, num_tokens=NUM_TOKENS
    )

    scheduler.add_request(request_normal)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request_normal], use_eos=True
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    #####################
    # Actual Test: confirm we send all blocks.

    # Step (1): Send the KV Transfer.
    NUM_EXTERNAL_FULL_BLOCKS -= 1
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request_remote = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
    )

    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_remote])
    eco = scheduler.update_from_output(scheduler_output, model_runner_output)
    kv_transfer_params = eco[0].outputs[0].kv_transfer_params

    # Ensure we send all block ids, including the partial blocks,
    # even if there is a cache hit.
    # remote_block_ids is BlockIds (tuple of lists); sum block counts across groups.
    num_remote_blocks = sum(len(g) for g in kv_transfer_params["remote_block_ids"])
    assert num_remote_blocks == (NUM_EXTERNAL_FULL_BLOCKS + 1)

    # STEP (2): Ensure it is freed.
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending={request_remote.request_id}
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


def test_abort_during_kv_transfer():
    """Test aborting request does not release blocks for remote decode."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Prime the KVCache.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
    )

    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    # Request removed from PB but blocks should not be freed.
    assert len(scheduler.requests) == 1

    # Abort the request, and check the blocks are still not freed
    scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
    assert len(scheduler.requests) == 1

    # Simulate a finished sending notification
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_sending=[request.request_id]
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert_scheduler_empty(scheduler)


@pytest.mark.parametrize(
    "num_tokens,token_budget,num_lookahead_tokens",
    [
        (40, 24, 0),  # chunk 2 = tokens 24..39: rest of b1 plus b2
        (20, 18, 0),  # chunk 2 = tokens 18..19: fits in b1, no new block
        (40, 30, 3),  # spec decode: chunk 1 also allocates lookahead block b2
    ],
)
def test_host_buffer_save_resaves_block_straddling_chunk_boundary(
    num_tokens: int, token_budget: int, num_lookahead_tokens: int
):
    """Host-buffer mode (kv_buffer_device="cpu") copies each prefill step's
    blocks to host memory. When a chunk boundary is not block aligned, the
    next chunk writes the rest of the previous chunk's last block, so that
    block must be copied again. Blocks allocated ahead of the written tokens
    (spec-decode lookahead) must not be copied before they are written."""
    block_size = 16
    vllm_config = create_vllm_config(
        block_size=block_size,
        max_num_batched_tokens=token_budget,
        kv_role="kv_producer",
    )
    scheduler = create_scheduler(vllm_config)
    scheduler.num_lookahead_tokens = num_lookahead_tokens
    connector_scheduler = scheduler.get_kv_connector().connector_scheduler
    connector_scheduler.use_host_buffer = True
    request = create_request(
        request_id=1,
        num_tokens=num_tokens,
        block_size=block_size,
        do_remote_decode=True,
    )
    req_id = request.request_id
    scheduler.add_request(request)

    out1 = scheduler.schedule()
    (saved1,) = out1.kv_connector_metadata.reqs_to_save[req_id].local_block_ids
    (table,) = scheduler.kv_cache_manager.get_block_ids(req_id)
    # Only the blocks holding tokens written in chunk 1.
    assert saved1 == table[: -(-token_budget // block_size)]
    model_output = create_model_runner_output([request])
    model_output.sampled_token_ids = [[]]
    scheduler.update_from_output(out1, model_output)
    assert request.num_computed_tokens % block_size != 0
    first = request.num_computed_tokens // block_size
    straddling = table[first]

    out2 = scheduler.schedule()
    (saved2,) = out2.kv_connector_metadata.reqs_to_save[req_id].local_block_ids
    (table,) = scheduler.kv_cache_manager.get_block_ids(req_id)
    assert saved2[0] == straddling
    assert saved2 == table[first : -(-num_tokens // block_size)]
    assert req_id not in connector_scheduler._reqs_need_save
    assert req_id not in connector_scheduler._reqs_save_state


def test_host_buffer_save_includes_prefix_cache_hit_blocks():
    """A first chunk that starts after a local prefix-cache hit still copies
    the cached blocks: D pulls the whole prompt from the host buffer."""
    block_size = 16
    vllm_config = create_vllm_config(
        block_size=block_size,
        max_num_batched_tokens=64,
        kv_role="kv_producer",
    )
    scheduler = create_scheduler(vllm_config)
    connector_scheduler = scheduler.get_kv_connector().connector_scheduler
    connector_scheduler.use_host_buffer = True

    # Request 1 computes and caches a 32-token (2-block) prefix.
    first = create_request(
        request_id=1,
        num_tokens=40,
        common_prefix_len=32,
        block_size=block_size,
        do_remote_decode=True,
    )
    scheduler.add_request(first)
    out = scheduler.schedule()
    (cached_prefix,) = scheduler.kv_cache_manager.get_block_ids(first.request_id)
    cached_prefix = cached_prefix[:2]
    scheduler.update_from_output(out, create_model_runner_output([first]))

    # Request 2 hits the prefix and needs two chunks for the rest.
    second = create_request(
        request_id=2,
        num_tokens=110,
        common_prefix_len=32,
        block_size=block_size,
        do_remote_decode=True,
    )
    req_id = second.request_id
    scheduler.add_request(second)
    out1 = scheduler.schedule()
    (saved1,) = out1.kv_connector_metadata.reqs_to_save[req_id].local_block_ids
    (table,) = scheduler.kv_cache_manager.get_block_ids(req_id)
    assert table[:2] == cached_prefix  # the prefix came from the cache
    assert saved1 == table[: (32 + 64) // block_size]
    model_output = create_model_runner_output([second])
    model_output.sampled_token_ids = [[]]
    scheduler.update_from_output(out1, model_output)

    out2 = scheduler.schedule()
    (saved2,) = out2.kv_connector_metadata.reqs_to_save[req_id].local_block_ids
    (table,) = scheduler.kv_cache_manager.get_block_ids(req_id)
    assert saved2 == table[96 // block_size : -(-110 // block_size)]
    assert req_id not in connector_scheduler._reqs_save_state
