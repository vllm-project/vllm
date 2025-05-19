# SPDX-License-Identifier: Apache-2.0

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnectorMetadata)

from .utils import create_request, create_scheduler, create_vllm_config


def test_basic_inferface():
    """Unit test for basic NixlConnector interface functionality."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True)
    request_id = request.request_id

    scheduler.add_request(request)

    # Remote Prefill, triggers NixlConnectorMetdata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, NixlConnectorMetadata)

    assert len(kv_connector_metadata.requests) == 1
    assert request_id in kv_connector_metadata.requests
    req_meta = kv_connector_metadata.requests[request_id]

    for block_id, block in zip(
            req_meta.local_block_ids, scheduler.kv_cache_manager.
            single_type_manager.req_to_blocks[request_id]):
        assert block_id == block.block_id


def test_prompt_less_than_block_size():
    """
    Test that we can handle case where prompt is < block.

    In this case, the P worker will send empty remote_block_ids.
    The D worker should not schedule an async read in this case,
    since there is nothing to pull.
    """
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # Half of a block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BLOCK_SIZE * 0.5)

    # Request will have 0 remote blocks.
    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True,
                             num_remote_blocks=0)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    # This request should not have to read async.
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, NixlConnectorMetadata)
    assert len(kv_connector_metadata.requests) == 0

    # This request should be scheduled regularly.
    assert len(scheduler_output.scheduled_new_reqs) == 1
