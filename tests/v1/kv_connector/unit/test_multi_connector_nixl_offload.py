# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MultiConnector with NIXLConnector + OffloadingConnector.
MultiConnector supports loading a given request’s KV from a single connector
(specifically, the first connector that reports a non-zero match via
get_num_new_matched_tokens()),
while supporting save/send via multiple connectors.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnector,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
    OffloadingConnectorMetadata,
)
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from .test_nixl_connector import FakeNixlConnectorWorker, FakeNixlWrapper
from .test_offloading_connector import (
    MockOffloadingSpec,
    RequestRunner,
    generate_store_output,
)
from .utils import (
    create_vllm_config,
)


@pytest.fixture
def multi_connector_config():
    """Create a VllmConfig with MultiConnector configuration."""
    base_config = create_vllm_config(block_size=16)

    # Configure MultiConnector with both NIXL and Offloading connectors
    base_config.kv_transfer_config.kv_connector = "MultiConnector"
    base_config.kv_transfer_config.kv_connector_extra_config = {
        "connectors": [
            {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_both",
            },
            {
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "spec_name": "MockOffloadingSpec",
                    "spec_module_path": "tests.v1.kv_connector.unit.test_offloading_connector",  # noqa: E501
                    "block_size": 16,
                },
            },
        ]
    }
    with set_current_vllm_config(base_config):
        yield base_config


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_multi_connector_initialization(multi_connector_config, dist_init):
    """Test that MultiConnector properly initializes both connectors."""
    connector = MultiConnector(
        multi_connector_config, KVConnectorRole.WORKER, kv_cache_config=MagicMock()
    )

    # Verify both connectors are initialized with correct types
    assert len(connector._connectors) == 2
    assert isinstance(connector._connectors[0], NixlConnector)
    assert isinstance(connector._connectors[1], OffloadingConnector)


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_multi_connector_nixl_transfer_with_offloading_active(
    multi_connector_config, dist_init
):
    """
    Test that NIXL transfer works correctly when offloading connector is active.

    This ensures both connectors can operate simultaneously without interference.
    The NIXL connector completes a transfer while the offloading connector
    is initialized and has metadata bound.
    """
    connector = MultiConnector(
        multi_connector_config, KVConnectorRole.WORKER, kv_cache_config=MagicMock()
    )

    nixl_connector = connector._connectors[0]
    offload_connector = connector._connectors[1]

    # Replace NIXL worker with fake
    nixl_connector.connector_worker = FakeNixlConnectorWorker(
        multi_connector_config, nixl_connector.engine_id, hand_shake_latency=0.0
    )

    # Register KV caches on both connectors
    kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
        num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
    )
    kv_cache = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    connector.register_cross_layers_kv_cache(kv_cache, FlashAttentionBackend)

    # Get reference to MockOffloadingSpec to verify offloading activity
    offload_worker = offload_connector.connector_worker
    assert offload_worker is not None
    offload_spec = offload_worker.spec
    assert isinstance(offload_spec, MockOffloadingSpec)
    mock_handler = offload_spec.handler

    # Verify handler is initialized
    assert mock_handler.transfer_specs is not None

    # Create NIXL transfer request
    request_id = "nixl_with_offload_req"
    nixl_metadata = NixlConnectorMetadata()
    nixl_metadata.add_new_req_to_recv(
        request_id=request_id,
        local_block_ids=[1, 2, 3],
        kv_transfer_params={
            "remote_block_ids": [10, 11, 12],
            "remote_engine_id": FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
            "remote_request_id": f"prefill-{request_id}",
            "remote_host": "localhost",
            "remote_port": 1234,
            "remote_tp_size": 1,
        },
    )

    # Offloading metadata is empty but connector is active
    offload_metadata = OffloadingConnectorMetadata(reqs_to_load={}, reqs_to_store={})

    multi_metadata = MultiKVConnectorMetadata(
        metadata=(nixl_metadata, offload_metadata)
    )
    connector.bind_connector_metadata(multi_metadata)

    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, virtual_engine=0)
    connector.start_load_kv(ctx)
    time.sleep(0.2)

    connector.bind_connector_metadata(
        MultiKVConnectorMetadata(metadata=(NixlConnectorMetadata(), offload_metadata))
    )
    connector.start_load_kv(ctx)

    finished_sending, finished_recving = connector.get_finished(set())

    # Verify NIXL transfer completed successfully
    assert request_id in finished_recving

    # Verify both connectors are still functional (have metadata bound)
    assert connector._connectors[0].has_connector_metadata()
    assert connector._connectors[1].has_connector_metadata()

    # Verify offloading connector infrastructure is accessible
    # (The handler is ready to accept transfers when metadata triggers them)
    assert mock_handler is not None
    # Note: No transfers expected here since offload_metadata was empty,
    # but we verified the handler is properly initialized and accessible


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_extra_async_saves_tracking(multi_connector_config, dist_init):
    """
    Test _extra_async_saves tracking when BOTH connectors async-save.

    Core scenario: A request finishes and both NIXL (sending to remote)
    and Offloading (saving to CPU) want to async-save. The request
    should only report as "finished sending" after BOTH complete.
    """
    connector = MultiConnector(
        multi_connector_config, KVConnectorRole.WORKER, kv_cache_config=MagicMock()
    )

    req_id = "dual_async_save_req"

    # Simulate: request has 1 extra async save pending (2 total saves)
    connector._extra_async_saves[req_id] = 1

    # First connector reports finished, second hasn't yet
    with (
        patch.object(
            connector._connectors[0], "get_finished", return_value=({req_id}, None)
        ),
        patch.object(
            connector._connectors[1], "get_finished", return_value=(None, None)
        ),
    ):
        sending, _ = connector.get_finished(set())
        # Should NOT be finished yet - still waiting for second connector
        assert sending is None or req_id not in sending
        # Extra count should be decremented/removed
        assert req_id not in connector._extra_async_saves

    # Second connector reports finished
    with (
        patch.object(
            connector._connectors[0], "get_finished", return_value=(None, None)
        ),
        patch.object(
            connector._connectors[1], "get_finished", return_value=({req_id}, None)
        ),
    ):
        sending, _ = connector.get_finished(set())
        # NOW should be reported as finished
        assert req_id in sending


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_get_num_matched_tokens_priority(multi_connector_config, dist_init):
    """
    Test connector priority for get_num_new_matched_tokens.

    When both NIXL and Offloading could provide cached tokens,
    the FIRST connector with matches wins (based on config order).
    This is the key lookup behavior for MultiConnector.
    """
    connector = MultiConnector(
        multi_connector_config, KVConnectorRole.WORKER, kv_cache_config=MagicMock()
    )

    mock_request = MagicMock()
    mock_request.request_id = "lookup_priority_req"

    # Scenario 1: NIXL (first) has matches -> NIXL wins
    with (
        patch.object(
            connector._connectors[0],
            "get_num_new_matched_tokens",
            return_value=(100, True),
        ),
        patch.object(
            connector._connectors[1],
            "get_num_new_matched_tokens",
            return_value=(50, False),
        ),
    ):
        tokens, load_async = connector.get_num_new_matched_tokens(mock_request, 0)
        assert tokens == 100
        assert load_async is True
        # Request assigned to NIXL (index 0)
        assert connector._requests_to_connector[mock_request.request_id] == 0

    connector._requests_to_connector.clear()

    # Scenario 2: NIXL has 0, Offloading has matches -> Offloading wins
    with (
        patch.object(
            connector._connectors[0],
            "get_num_new_matched_tokens",
            return_value=(0, False),
        ),
        patch.object(
            connector._connectors[1],
            "get_num_new_matched_tokens",
            return_value=(75, False),
        ),
    ):
        tokens, load_async = connector.get_num_new_matched_tokens(mock_request, 0)
        assert tokens == 75
        # Request assigned to Offloading (index 1)
        assert connector._requests_to_connector[mock_request.request_id] == 1


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_request_finished_both_async_save(multi_connector_config, dist_init):
    """
    Test request_finished when BOTH connectors want to async-save.

    Real scenario: NIXL sends KVTransferParams to remote decode (P is done),
    Offloading saves to CPU. MultiConnector should track that 2 async saves are pending
    """
    connector = MultiConnector(
        multi_connector_config, KVConnectorRole.WORKER, kv_cache_config=MagicMock()
    )

    mock_request = MagicMock()
    mock_request.request_id = "both_save_req"

    # Both connectors return async_save=True
    with (
        patch.object(
            connector._connectors[0],
            "request_finished",
            return_value=(True, {"remote_engine_id": "engine1"}),
        ),
        patch.object(
            connector._connectors[1],
            "request_finished",
            return_value=(True, None),
        ),
    ):
        async_save, kv_params = connector.request_finished(mock_request, [1, 2, 3])

        assert async_save is True
        # 2 async saves - 1 = 1 extra tracked
        assert connector._extra_async_saves.get(mock_request.request_id) == 1
        # KV params from NIXL should be returned
        assert kv_params is not None
        assert "remote_engine_id" in kv_params


# =============================================================================
# MultiConnectorRequestRunner: Full scheduler+worker flow with real offloading
# =============================================================================


class MultiConnectorRequestRunner(RequestRunner):
    """
    Request runner for testing MultiConnector with real offloading transfers.

    Extends RequestRunner to use MultiConnector with both NIXL and Offloading
    connectors, overriding only the connector-specific methods.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        offloaded_block_size: int,
        gpu_block_size: int,
        num_gpu_blocks: int,
    ):
        self._vllm_config = vllm_config
        super().__init__(offloaded_block_size, gpu_block_size, num_gpu_blocks)

    def _create_vllm_config(self) -> VllmConfig:
        return self._vllm_config

    def _create_worker_connector(self, vllm_config: VllmConfig):
        """Create worker-side MultiConnector."""
        return MultiConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config=MagicMock()
        )

    def _extract_offloading_components(
        self,
    ) -> tuple[MagicMock, MockOffloadingSpec]:
        """Extract manager and spec from MultiConnector's sub-connectors."""
        scheduler_connector = self.scheduler_connector
        assert isinstance(scheduler_connector, MultiConnector)

        # Extract offloading components from scheduler's MultiConnector
        offload_scheduler_connector = scheduler_connector._connectors[1]
        assert isinstance(offload_scheduler_connector, OffloadingConnector)
        connector_scheduler = offload_scheduler_connector.connector_scheduler
        assert connector_scheduler is not None
        manager = connector_scheduler.manager
        assert isinstance(manager, MagicMock)

        # Extract offloading components from worker's MultiConnector
        assert isinstance(self.worker_connector, MultiConnector)
        offload_worker_connector = self.worker_connector._connectors[1]
        assert isinstance(offload_worker_connector, OffloadingConnector)
        connector_worker = offload_worker_connector.connector_worker
        assert connector_worker is not None
        offloading_spec = connector_worker.spec
        assert isinstance(offloading_spec, MockOffloadingSpec)

        return manager, offloading_spec

    def get_nixl_connector(self) -> NixlConnector:
        """Get the NIXL connector from the worker's MultiConnector."""
        assert isinstance(self.worker_connector, MultiConnector)
        nixl_connector = self.worker_connector._connectors[0]
        assert isinstance(nixl_connector, NixlConnector)
        return nixl_connector

    def get_handler_stats(self) -> dict:
        """Get stats from the mock handler."""
        handler = self.offloading_spec.handler
        return {
            "transfer_specs": len(handler.transfer_specs),
            "waiting_jobs": len(handler.waiting_jobs),
            "completed_jobs": len(handler.completed_jobs),
        }


@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.NixlWrapper",
    FakeNixlWrapper,
)
def test_multi_connector_real_offloading_transfers(dist_init, multi_connector_config):
    """
    Verify both offloading and NIXL transfers work through MultiConnector.

    This test uses MultiConnectorRequestRunner to:
    1. Create a request with enough tokens to fill blocks
    2. Run scheduler steps that trigger offloading stores
    3. Verify the MockOffloadingHandler received transfer requests
    4. Set up and verify a NIXL transfer completes alongside offloading

    This confirms both connectors work simultaneously through MultiConnector.
    """
    offloaded_block_size = 16
    gpu_block_size = 16
    num_gpu_blocks = 100

    runner = MultiConnectorRequestRunner(
        vllm_config=multi_connector_config,
        offloaded_block_size=offloaded_block_size,
        gpu_block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
    )

    # Configure manager to store middle block
    runner.manager.prepare_store.side_effect = (
        lambda block_hashes: generate_store_output(list(block_hashes)[1:2])
    )

    # Create request with 3 offloaded blocks worth of tokens
    runner.new_request(token_ids=[0] * offloaded_block_size * 3)

    # Get initial handler stats
    initial_stats = runner.get_handler_stats()

    # Run a step - this should trigger offloading
    runner.run(decoded_tokens=[0])

    # Get stats after run
    final_stats = runner.get_handler_stats()

    # Verify offloading transfers were initiated
    assert final_stats["transfer_specs"] >= initial_stats["transfer_specs"], (
        "Offloading handler should have received transfer specs"
    )

    # Verify both connectors are present and functional
    nixl_connector = runner.get_nixl_connector()
    assert isinstance(nixl_connector, NixlConnector)

    offload_connector = runner.worker_connector._connectors[1]
    assert isinstance(offload_connector, OffloadingConnector)

    # --- Now verify NIXL transfers also work alongside offloading ---

    # Replace NIXL worker with fake that completes transfers immediately
    nixl_connector.connector_worker = FakeNixlConnectorWorker(
        runner.scheduler.vllm_config,
        nixl_connector.engine_id,
        hand_shake_latency=0.0,
    )

    # Set up a NIXL transfer request via metadata
    request_id = "nixl_with_offload_req"
    nixl_metadata = NixlConnectorMetadata()
    nixl_metadata.add_new_req_to_recv(
        request_id=request_id,
        local_block_ids=[1, 2, 3],
        kv_transfer_params={
            "remote_block_ids": [10, 11, 12],
            "remote_engine_id": FakeNixlConnectorWorker.REMOTE_ENGINE_ID,
            "remote_request_id": f"prefill-{request_id}",
            "remote_host": "localhost",
            "remote_port": 1234,
            "remote_tp_size": 1,
        },
    )

    offload_metadata = OffloadingConnectorMetadata(reqs_to_load={}, reqs_to_store={})
    multi_metadata = MultiKVConnectorMetadata(
        metadata=(nixl_metadata, offload_metadata)
    )

    # Bind and execute NIXL transfer
    runner.worker_connector.bind_connector_metadata(multi_metadata)
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, virtual_engine=0)
    runner.worker_connector.start_load_kv(ctx)

    # Wait for async NIXL operations
    time.sleep(0.2)

    # Poll for results with fresh metadata
    runner.worker_connector.bind_connector_metadata(
        MultiKVConnectorMetadata(metadata=(NixlConnectorMetadata(), offload_metadata))
    )
    runner.worker_connector.start_load_kv(ctx)

    finished_sending, finished_recving = runner.worker_connector.get_finished(set())

    # Verify NIXL transfer completed
    assert request_id in finished_recving, (
        f"NIXL transfer should complete; got finished_recving={finished_recving}"
    )

    # Verify offloading handler still has the transfers from earlier
    assert runner.get_handler_stats()["transfer_specs"] > 0, (
        "Offloading handler should retain transfer specs"
    )
