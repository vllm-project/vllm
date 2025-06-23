# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import uuid
from collections import defaultdict
from typing import Optional
from unittest.mock import patch

import pytest

try:
    from nixl._api import nixl_agent as NixlWrapper
except ImportError:
    NixlWrapper = None

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    KVConnectorRole, NixlAgentMetadata, NixlConnector, NixlConnectorMetadata,
    NixlConnectorWorker)
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


class FakeNixlWrapper:
    """Mock implementation of NixlWrapper for testing.
    
    We don't inherit from NixlWrapper because NixlWrapper could be None.
    """

    AGENT_METADATA = b"fake_agent_metadata"
    REMOTE_AGENT_NAME = "remote_agent"

    def __init__(self, agent_name: str, *args, **kwargs):
        self._cycles_before_xfer_done = 0
        self._check_xfer_state_cycles: defaultdict[int, int] = defaultdict(
            lambda: 0)

    def get_reg_descs(self, caches_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in caches_data]

    def register_memory(self, descs) -> None:
        pass

    def get_xfer_descs(self, blocks_data, memory_type: str) -> list:
        return [str(uuid.uuid4()) for _ in blocks_data]

    def prep_xfer_dlist(self, agent_name: str, descs: list) -> int:
        return uuid.uuid4().int

    def get_agent_metadata(self) -> bytes:
        return self.AGENT_METADATA

    def add_remote_agent(self, agent_metadata: bytes) -> str:
        return self.REMOTE_AGENT_NAME

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        # Used to collect done_sending, which we don't test yet.
        return {}

    def check_xfer_state(self, handle: int) -> str:
        if self._check_xfer_state_cycles[
                handle] >= self._cycles_before_xfer_done:
            return "DONE"
        self._check_xfer_state_cycles[handle] += 1
        return "PROC"

    def release_xfer_handle(self, handle: int) -> None:
        pass

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        pass

    def make_prepped_xfer(self,
                          xfer_type: str,
                          local_xfer_side_handle: int,
                          local_block_descs_ids: list[int],
                          remote_xfer_side_handle: int,
                          remote_block_descs_ids: list[int],
                          notif_msg: Optional[bytes] = None) -> int:
        return uuid.uuid4().int

    def transfer(self, handle: int) -> str:
        return "PROC"

    ############################################################
    # Follow are for changing the behavior during testing.
    ############################################################

    def set_cycles_before_xfer_done(self, cycles: int):
        """Set the number of cycles before a transfer is considered done."""
        self._cycles_before_xfer_done = cycles


class FakeNixlConnectorWorker(NixlConnectorWorker):

    REMOTE_ENGINE_ID = "remote_engine"

    def __init__(self, *args, hand_shake_latency: float = 1.8, **kwargs):
        super().__init__(*args, **kwargs)
        self._hand_shake_latency = hand_shake_latency

    def _nixl_handshake(self, host: str, port: int):
        # Mimic slow _nixl_handshake, as well as bypass zmq communication.
        time.sleep(self._hand_shake_latency)
        # These should've been done in register_kv_caches(), called by
        # gpu_model_runner. Here we just hardcode some dummy values.
        self.slot_size_bytes = 4096
        self.block_len = self.slot_size_bytes * self.block_size
        self.num_blocks = 1
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        self.add_remote_agent(
            NixlAgentMetadata(
                engine_id=self.REMOTE_ENGINE_ID,
                agent_metadata=FakeNixlWrapper.AGENT_METADATA,
                kv_caches_base_addr=[0],
                num_blocks=1,
                tp_size=1,
                block_len=self.block_len,
                attn_backend_name=self.backend_name,
            ))


@pytest.mark.skipif(NixlWrapper is None, reason="nixl not installed")
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper)
def test_multi_xfer_one_engine(
    # dist_init is a fixture that initializes the distributed environment.
    dist_init):
    """Test case where multiple xfers are initiated to the same engine.
    
    This test triggers the connector to load remote KV for the same
    `request_id`. The transfer is not done immediately due to
    `set_cycles_before_xfer_done`, so there is a state where there are multiple
    transfer states for the same `request_id`, and `get_finished` should handle
    it correctly (wait for all transfers to be done).
    """
    vllm_config = create_vllm_config()

    request_id = "req_id"

    # Test worker role in decode server.
    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER)
    connector.connector_worker = FakeNixlConnectorWorker(vllm_config,
                                                         connector.engine_id,
                                                         hand_shake_latency=0)
    assert isinstance(connector.connector_worker.nixl_wrapper, FakeNixlWrapper)
    connector.connector_worker.nixl_wrapper.set_cycles_before_xfer_done(3)
    for i in range(4):
        metadata = NixlConnectorMetadata()
        metadata.add_new_req(request_id=request_id,
                             local_block_ids=[i + 1, i + 2, i + 3],
                             kv_transfer_params={
                                 "remote_block_ids": [i + 4, i + 5, i + 6],
                                 "remote_engine_id":
                                 FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
                                 "remote_host": "localhost",
                                 "remote_port": 1234,
                             })
        connector.bind_connector_metadata(metadata)

        dummy_ctx = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            virtual_engine=0,
        )
        _before_load = time.perf_counter()
        connector.start_load_kv(dummy_ctx)
        _after_load = time.perf_counter()
        assert _after_load - _before_load < 0.1, "start_load_kv took " \
            f"{_after_load - _before_load} seconds"

    while True:
        _, done_recving = connector.get_finished(finished_req_ids=set())
        if len(done_recving) > 0:
            assert request_id in done_recving
            break
