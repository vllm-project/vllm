# SPDX-License-Identifier: Apache-2.0
import copy

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import FinishReason, RequestStatus

from .utils import (assert_scheduler_empty, create_model_runner_output,
                    create_request, create_scheduler, create_vllm_config)


def test_basic_lifecycle():
    """Test lifecycle of a remote prefill."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))
    START_FREE_BLOCK_QUEUE_SIZE = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks)

    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True)

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1):
    # (1a): schedule()
    scheduler_output = scheduler.schedule()

    # Nothing running and empty scheduler output.
    assert len(scheduler.running) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert len(scheduler_output.scheduled_cached_reqs) == 0
    assert len(scheduler_output.num_scheduled_tokens) == 0
    assert scheduler_output.total_num_scheduled_tokens == 0

    # Req waiting for KVs with no computed/scheduled toks ...
    assert len(scheduler.waiting) == 1
    assert request in scheduler.waiting
    assert (request.status == RequestStatus.WAITING_FOR_REMOTE_KVS)
    assert (request.num_computed_tokens == 0)

    # ... but should have (uncached) blocks allocated to it.
    block_pool = scheduler.kv_cache_manager.block_pool
    assert (block_pool.free_block_queue.num_free_blocks
            < START_FREE_BLOCK_QUEUE_SIZE)
    assert len(block_pool.cached_block_hash_to_block) == 0
    blocks = scheduler.kv_cache_manager.single_type_manager.req_to_blocks[
        request_id]
    for block in blocks:
        assert block._block_hash is None

    # (1b): forward()
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT

    # (1c): update_from_output()
    engine_core_outputs = scheduler.update_from_output(scheduler_output,
                                                       model_runner_output)
    assert len(engine_core_outputs.outputs) == 0

    # STEP (2):
    # (2a): schedule(): nothing happens!
    scheduler_output = scheduler.schedule()
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 0

    # (2b): forward(): request finishes recv.
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.finished_recving = [request_id]

    # (2c): update_from_output():
    engine_core_outputs = scheduler.update_from_output(scheduler_output,
                                                       model_runner_output)
    assert len(scheduler.waiting) == 1
    assert (request_id in scheduler.finished_recving_kv_req_ids)

    # STEP (3):
    # (3a): schedule(): this should actually schedule.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1

    # Confirm the block are actually allocated.
    num_hashed_blocks = 0
    blocks = scheduler.kv_cache_manager.single_type_manager.req_to_blocks[
        request_id]
    for block in blocks:
        assert block.ref_cnt == 1
        num_hashed_blocks += (1 if block._block_hash is not None else 0)
    assert num_hashed_blocks == NUM_EXTERNAL_FULL_BLOCKS

    # Confirm the rest of the prompt is scheduled in this step.
    scheduled_req = scheduler_output.scheduled_new_reqs[0]
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[request_id]
    num_computed_tokens = scheduled_req.num_computed_tokens
    total_prompt_tokens = len(scheduled_req.prompt_token_ids)
    assert (num_scheduled_tokens == total_prompt_tokens - num_computed_tokens)

    # (3b): execute_model()
    model_runner_output = create_model_runner_output([request])
    # (3c): update_from_output()
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # Step (4): Hit EOS.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output([request], use_eos=True)
    engine_core_outputs = scheduler.update_from_output(scheduler_output,
                                                       model_runner_output)
    scheduler.schedule()

    outputs = engine_core_outputs.outputs
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finish_reason == FinishReason.STOP
    assert_scheduler_empty(scheduler)


def test_interleaved_lifecycle():
    """Test Remote Prefills Work Well With Other Requests."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request_remote = create_request(request_id=1,
                                    num_tokens=NUM_TOKENS,
                                    do_remote_prefill=True)
    request_local_a = create_request(
        request_id=2,
        num_tokens=NUM_TOKENS,
    )
    request_local_b = create_request(
        request_id=3,
        num_tokens=NUM_TOKENS,
    )

    # STEP 1: Regular request is running.
    scheduler.add_request(request_local_a)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1

    model_runner_output = create_model_runner_output([request_local_a])
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP 2: Add a local and remote request.
    scheduler.add_request(request_local_b)
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1
    assert len(scheduler_output.scheduled_cached_reqs) == 1

    model_runner_output = create_model_runner_output(
        [request_local_a, request_local_b])
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP 3: continue running, KVs not arrived yet.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert len(scheduler_output.scheduled_cached_reqs) == 2

    model_runner_output = create_model_runner_output(
        reqs=[request_local_a, request_local_b])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert len(scheduler_output.scheduled_cached_reqs) == 2

    # STEP 4: KVs arrive.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert len(scheduler_output.scheduled_cached_reqs) == 2

    model_runner_output = create_model_runner_output(
        [request_local_a, request_local_b],
        finished_recving=[request_remote.request_id])
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP 5: RECVed KVs are sent to ModelRunner.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(scheduler.waiting) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 1
    assert len(scheduler_output.scheduled_cached_reqs) == 2

    model_runner_output = create_model_runner_output(
        [request_local_a, request_local_b, request_remote])
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # STEP 6: Hit EOS and free.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        [request_local_a, request_local_b, request_remote],
        use_eos=True,
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler.schedule()
    assert_scheduler_empty(scheduler)


def test_no_spurious_prefix_caching():
    """
    With P/D, blocks can be allocated but uncomputed for
    multiple engine steps. This test confirms that we do
    not accidentally have cache hits against uncomputed
    blocks.
    """

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 and a half full external blocks.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    # Both of these requests have prompts like [1,1,1,1,1, ...]
    request_remote = create_request(
        request_id=1,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
        use_all_1s_for_prompt_tokens=True,
    )

    request_local = create_request(
        request_id=2,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=False,
        use_all_1s_for_prompt_tokens=True,
    )

    # Schedule the remote prefill request. This should not
    # cause any blocks to be cached.
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)
    assert len(scheduler.waiting) == 1

    # Schedule the local prefill request. This should
    # cause blocks to be cached, but separately from
    scheduler.add_request(request_local)
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    local_blocks = scheduler.kv_cache_manager.single_type_manager.req_to_blocks[
        request_local.request_id]
    remote_blocks = scheduler.kv_cache_manager.single_type_manager.req_to_blocks[  # noqa: E501
        request_remote.request_id]

    # Local should have cached blocks (but not all due to preallocate).
    num_hashed_blocks = 0
    for block in local_blocks:
        assert block.ref_cnt == 1
        num_hashed_blocks += (1 if block._block_hash is not None else 0)
    assert num_hashed_blocks > 0

    # Remote blocks should not be cached.
    for block in remote_blocks:
        assert block.ref_cnt == 1
        assert block._block_hash is None


def test_full_block_prompt():
    """Test that we handle a prompt that is the full block size."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * NUM_EXTERNAL_FULL_BLOCKS)

    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True)

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1): Initialize a recv.
    scheduler_output = scheduler.schedule()
    # All blocks should be allocated.
    num_blocks = len(scheduler.kv_cache_manager.single_type_manager.
                     req_to_blocks[request_id])
    assert num_blocks == NUM_EXTERNAL_FULL_BLOCKS
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # # STEP (2): Recv.
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.finished_recving = [request_id]
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.waiting) == 1
    assert (request_id in scheduler.finished_recving_kv_req_ids)

    # # STEP (3): Run as usual.
    scheduler_output = scheduler.schedule()

    # We need to recompute the final token of the prompt to generate
    # the first new token, so we should not have a new block.
    num_blocks = len(scheduler.kv_cache_manager.single_type_manager.
                     req_to_blocks[request_id])
    assert num_blocks == NUM_EXTERNAL_FULL_BLOCKS
    assert (scheduler_output.scheduled_new_reqs[0].num_computed_tokens ==
            NUM_TOKENS - 1)
    assert (scheduler_output.num_scheduled_tokens[request_id] == 1)

    model_runner_output = create_model_runner_output([request])
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # # Step (4): Hit EOS.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output([request], use_eos=True)
    engine_core_outputs = scheduler.update_from_output(scheduler_output,
                                                       model_runner_output)
    scheduler.schedule()

    outputs = engine_core_outputs.outputs
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finish_reason == FinishReason.STOP
    assert_scheduler_empty(scheduler)


def test_cannot_schedule_after_recv():
    """
    Test that we can handle no schedule after recv due to not
    enough remaining KV blocks.
    """

    # NOTE: the KVCacheManager will use 1 null block.
    # So there are 5 total working blocks.
    TOTAL_NUM_BLOCKS = 6
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config, num_blocks=TOTAL_NUM_BLOCKS)

    # Prime the KVCache.
    NUM_PROMPT_BLOCKS = 2
    BLOCK_SIZE = vllm_config.cache_config.block_size
    # Prompt will use 2 blocks + 1 block after we schedule.
    NUM_TOKENS_LOCAL = int(BLOCK_SIZE * NUM_PROMPT_BLOCKS)
    NUM_TOKENS_REMOTE = int(BLOCK_SIZE * (NUM_PROMPT_BLOCKS + 0.5))

    request_normal = create_request(request_id=1, num_tokens=NUM_TOKENS_LOCAL)
    request_remote = create_request(request_id=2,
                                    num_tokens=NUM_TOKENS_REMOTE,
                                    do_remote_prefill=True)

    # STEP 1: 3 blocks are in use (2 for prompt, 1 for decode).
    scheduler.add_request(request_normal)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # Step 2: 5 blocks are in use (2 new for remote blocks).
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # Step 3: finish recving (5 blocks in use)
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(
        reqs=[request_normal], finished_recving=[request_remote.request_id])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # Step 4: try to schedule, not enough blocks.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # Step 5: finish the request, free it.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_normal],
                                                     use_eos=True)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1

    # Step 6: now we can schedule (with 2 blocks computed).
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_remote])
    assert (scheduler_output.scheduled_new_reqs[0].num_computed_tokens ==
            NUM_PROMPT_BLOCKS * BLOCK_SIZE)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # Step 7: free everything.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request_remote],
                                                     use_eos=True)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    _ = scheduler.schedule()
    assert_scheduler_empty(scheduler)
