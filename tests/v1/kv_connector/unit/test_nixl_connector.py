# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from nixl._api import nixl_agent as NixlWrapper
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnector,
    NixlConnectorMetadata,
    KVConnectorRole)
from vllm.forward_context import ForwardContext

from .utils import create_request, create_scheduler, create_vllm_config


def test_basic_interface():
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

    # Remote Prefill, triggers NixlConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, NixlConnectorMetadata)

    assert len(kv_connector_metadata.requests) == 1
    assert request_id in kv_connector_metadata.requests
    req_meta = kv_connector_metadata.requests[request_id]

    for block_id, block in zip(
            req_meta.local_block_ids, scheduler.kv_cache_manager.coordinator.
            single_type_managers[0].req_to_blocks[request_id]):
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


class FakeNixlWrapper(NixlWrapper):
    """Mock implementation of NixlWrapper for testing."""
    
    def __init__(self, agent_name: str, *args, **kwargs):
        pass

    def get_reg_descs(self, caches_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in caches_data]
    
    def register_memory(self, descs) -> None:
        pass
        
    def get_xfer_descs(self, blocks_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in blocks_data]
    
    def prep_xfer_dlist(self, agent_name: str, descs: list) -> int:
        return uuid.uuid4().int
    
    def get_agent_metadata(self) -> bytes:
        return b"fake_agent_metadata"
    
    def add_remote_agent(self, agent_metadata: bytes) -> str:
        return str(agent_metadata)
    
    def get_new_notifs(self) -> Dict[str, List[bytes]]:
        # Used to collect done_sending, which we don't test yet.
        return {}
    
    def check_xfer_state(self, handle: int) -> str:
        return "DONE"

    def release_xfer_handle(self, handle: int) -> None:
        pass
    
    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        pass
    
    def make_prepped_xfer(self, 
                         xfer_type: str,
                         local_xfer_side_handle: int,
                         local_block_descs_ids: List[int],
                         remote_xfer_side_handle: int,
                         remote_block_descs_ids: List[int],
                         notif_msg: Optional[bytes] = None) -> int:
        return uuid.uuid4().int

    def transfer(self, handle: int) -> str:
        return "DONE"


@patch("vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper", FakeNixlWrapper)
def test_async_load_kv(dist_init):
    """Test that NixlConnector's start_load_kv should be fast."""
    
    vllm_config = create_vllm_config()
    
    # Test worker role in decode server.
    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
    metadata = NixlConnectorMetadata()
    metadata.add_new_req(request_id="id", local_block_ids=[1, 2, 3], kv_transfer_params={
        "remote_block_ids": [4, 5, 6],
        "remote_engine_id": "remote_engine",
        "remote_host": "localhost",
        "remote_port": 1234,
    })
    connector.bind_connector_metadata(metadata)

    dummy_ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        virtual_engine=0,
    )
    connector.start_load_kv(dummy_ctx)
