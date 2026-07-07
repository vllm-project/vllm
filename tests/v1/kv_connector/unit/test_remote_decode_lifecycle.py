# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import FinishReason, RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
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


def _make_nixl_scheduler(vllm_config, **kwargs) -> NixlPullConnectorScheduler:
    """Build a standalone NIXL connector-scheduler for directly exercising
    ``request_finished``. Defaults to a single full-attention KV cache group."""
    kv_cache_config = make_kv_cache_config(
        block_size=vllm_config.cache_config.block_size, **kwargs
    )
    return NixlPullConnectorScheduler(vllm_config, "test-engine-id", kv_cache_config)


def _make_finished_request(num_computed_tokens: int, block_size: int):
    request = create_request(
        request_id=1,
        block_size=block_size,
        num_tokens=num_computed_tokens,
        do_remote_decode=True,
    )
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    request.num_computed_tokens = num_computed_tokens
    return request


@pytest.mark.parametrize("extra_lookahead_blocks", [0, 1, 2])
def test_remote_decode_drops_lookahead_blocks(extra_lookahead_blocks):
    """Regression test: request_finished must advertise exactly the blocks
    holding computed KV, not the spec-decode lookahead reservation blocks
    allocated past num_computed_tokens. Sending an extra block makes the
    decode side's suffix-trim (_apply_prefix_caching) misalign the block
    mapping and read never-written KV.
    """
    vllm_config = create_vllm_config()
    connector = _make_nixl_scheduler(vllm_config)

    block_size = vllm_config.cache_config.block_size
    # Multiple of block_size: the worst case where the lookahead slot needs a
    # brand-new block. Allocate prompt blocks + the lookahead reservation.
    num_computed_tokens = 4 * block_size
    num_prompt_blocks = num_computed_tokens // block_size  # == 4
    allocated_block_ids = list(range(1, num_prompt_blocks + extra_lookahead_blocks + 1))

    request = _make_finished_request(num_computed_tokens, block_size)
    delay_free_blocks, params = connector.request_finished(
        request, (allocated_block_ids,)
    )

    assert delay_free_blocks is True
    assert params is not None
    # Trailing lookahead blocks dropped, regardless of how many were allocated.
    assert params["remote_block_ids"] == ([1, 2, 3, 4],)
    assert params["remote_num_tokens"] == num_computed_tokens


def test_remote_decode_lookahead_clip_is_per_group():
    """Clipping is per-group with each group's own block_size: in a hybrid
    model the attention group is clipped while a Mamba/SSM state group is left
    untouched. The attention group uses a block_size != the global one, so a
    global-block_size implementation would clip it incorrectly.
    """
    vllm_config = create_vllm_config()
    global_block_size = vllm_config.cache_config.block_size  # 16
    attn_block_size = 2 * global_block_size  # 32

    kv_cache_groups = [
        KVCacheGroupSpec(
            ["mamba_layer"],
            MambaSpec(
                block_size=global_block_size,
                shapes=((16,),),
                dtypes=(torch.float16,),
            ),
        ),
        KVCacheGroupSpec(
            ["attn_layer"],
            FullAttentionSpec(
                block_size=attn_block_size,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.float16,
            ),
        ),
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=100, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups
    )
    connector = NixlPullConnectorScheduler(
        vllm_config, "test-engine-id", kv_cache_config
    )

    # 64 tokens => 2 attn blocks at block_size 32, + 1 lookahead block.
    # (cdiv(64, 16) == 4 would not clip, so this fails with the global size.)
    num_computed_tokens = 2 * attn_block_size  # 64
    request = _make_finished_request(num_computed_tokens, attn_block_size)

    # group 0: Mamba state block; group 1: 2 prompt blocks + 1 lookahead block.
    _, params = connector.request_finished(request, ([101], [1, 2, 3]))

    # Mamba group passed through; attention group clipped at its own block_size.
    assert params["remote_block_ids"] == ([101], [1, 2])


def test_remote_decode_lookahead_clip_before_sw_clip():
    """The lookahead clip must run before the sliding-window clip: the SW clip
    keeps the *last* blocks_per_sw blocks, so a trailing lookahead block would
    be absorbed into the window and a real leading window block dropped —
    unrecoverable on the decode side.
    """
    vllm_config = create_vllm_config()
    block_size = vllm_config.cache_config.block_size  # 16
    # blocks_per_sw for the SWA group = cdiv(2 * 16, 16) + 1 = 3.
    connector = _make_nixl_scheduler(
        vllm_config, swa_enabled=True, sw_size=2 * block_size
    )

    num_computed_tokens = 5 * block_size
    request = _make_finished_request(num_computed_tokens, block_size)

    # Both groups: 5 computed blocks + 1 lookahead block.
    full_ids = [1, 2, 3, 4, 5, 6]
    swa_ids = [11, 12, 13, 14, 15, 16]
    _, params = connector.request_finished(request, (full_ids, swa_ids))

    # Full-attn group: lookahead clipped. SWA group: lookahead clipped first,
    # then windowed to the last 3 computed blocks (not [14, 15, 16]).
    assert params["remote_block_ids"] == ([1, 2, 3, 4, 5], [13, 14, 15])


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
