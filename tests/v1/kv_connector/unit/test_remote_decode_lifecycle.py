# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import FinishReason, RequestStatus

from .utils import (assert_scheduler_empty, create_model_runner_output,
                    create_request, create_scheduler, create_vllm_config)


def test_basic_lifecycle():
    """Test lifecycle of a Remote Decode request."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(request_id=1,
                             max_tokens=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_decode=True)

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1): Prefill.
    # (1a): schedule()
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1

    # (1b): execute_model()
    model_runner_output = create_model_runner_output(reqs=[request])

    # (1c): update_from_output()
    engine_core_outputs = scheduler.update_from_output(scheduler_output,
                                                       model_runner_output)

    # Ensure the request is finished after 1 tokens.
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
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0].req_to_blocks[request_id]
    for block in blocks:
        assert block.ref_cnt == 1

    # STEP (2): Send Finished to PB.
    # (2a): schedule() - pass finished request to PB.
    scheduler_output = scheduler.schedule()
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
    assert len(scheduler.running) == 0
    assert len(scheduler_output.finished_req_ids) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 0
    assert len(scheduler.finished_req_ids) == 0

    # (3b): execute_model()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.finished_sending = [request_id]

    # (3c): update_from_output()
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # Confirm we do not have any memory leaks after req lifecycle.
    assert_scheduler_empty(scheduler)


def test_short_prompt_lifecycle():
    """Test lifecycle of a Remote Decode request with short prompt."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Not enough tokens for full block.
    NUM_TOKENS = vllm_config.cache_config.block_size // 2
    request = create_request(request_id=1,
                             max_tokens=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_decode=True)

    scheduler.add_request(request)

    # STEP (1): Prefill.
    # (1a): schedule()
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1

    # (1b): execute_model()
    model_runner_output = create_model_runner_output(reqs=[request])

    # (1c): update_from_output()
    # Since tokens < block_size, there will be no kv xfer.
    # So this should be cleaned up immediately.
    _ = scheduler.update_from_output(scheduler_output, model_runner_output)

    # Confirm we do not have any memory leaks after req lifecycle.
    # We need one more call to schedule() to clear data for persistent batch.
    _ = scheduler.schedule()
    assert_scheduler_empty(scheduler)


def test_prefix_cache_lifecycle():
    """Test that remote decode params still works with a prefix cache hit."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Prime the KVCache.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 3
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request_normal = create_request(request_id=1, num_tokens=NUM_TOKENS)

    scheduler.add_request(request_normal)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal],
                                                     use_eos=True)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    #####################
    # Actual Test: confirm we send all blocks.

    # Step (1): Send the KV Transfer.
    NUM_EXTERNAL_FULL_BLOCKS -= 1
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request_remote = create_request(request_id=1,
                                    num_tokens=NUM_TOKENS,
                                    do_remote_decode=True)

    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_remote])
    eco = scheduler.update_from_output(scheduler_output, model_runner_output)
    kv_transfer_params = eco[0].outputs[0].kv_transfer_params

    # Ensure we send all block ids, even if there is a cache hit.
    assert (len(
        kv_transfer_params["remote_block_ids"]) == NUM_EXTERNAL_FULL_BLOCKS)

    # STEP (2): Ensure it is freed.
    scheduler_output = scheduler.schedule()
    scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.finished_sending = [request_remote.request_id]
    scheduler.update_from_output(scheduler_output, model_runner_output)
    _ = scheduler.schedule()
    assert_scheduler_empty(scheduler)
