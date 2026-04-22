# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MooncakeECConnector.
"""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.mooncake_connector import (
    TRANS_ERROR,
    MMHashMeta,
    MooncakeECConnector,
    MooncakeECConnectorMetadata,
    MooncakeECConnectorWorker,
    _get_encoder_cache_embed_size,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.sched.output import SchedulerOutput


# ------------------ Mock Classes ------------------ #
class MockRequest:
    def __init__(
        self,
        request_id,
        mm_hashes: list[str],
        token_counts: list[int],
        ec_transfer_params: dict | None = None,
    ):
        assert len(mm_hashes) == len(token_counts)
        self.request_id = request_id
        self._token_counts = token_counts
        self.mm_features = []
        for i, mm_hash in enumerate(mm_hashes):
            feature = MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(offset=0, length=self._token_counts[i]),
            )
            self.mm_features.append(feature)
        # Normalize None to {} so scheduler-side logic does not call .get on None.
        self.ec_transfer_params = (
            ec_transfer_params if ec_transfer_params is not None else {}
        )

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self._token_counts)
        return self._token_counts[input_id]


# ------------------ Fixtures ------------------ #
@pytest.fixture
def mock_parallel_state():
    """Mock parallel state functions to avoid initialization requirements."""
    mock_group = MagicMock()
    mock_group.rank = 0
    mock_group.local_rank = 0
    mock_group.world_size = 1

    with (
        patch.multiple(
            "vllm.distributed.parallel_state",
            get_tensor_model_parallel_rank=MagicMock(return_value=0),
            get_tensor_model_parallel_world_size=MagicMock(return_value=1),
            get_tp_group=MagicMock(return_value=mock_group),
        ),
        patch.multiple(
            "vllm.distributed.ec_transfer.ec_connector.mooncake_connector",
            get_tensor_model_parallel_rank=MagicMock(return_value=0),
            get_tensor_model_parallel_world_size=MagicMock(return_value=1),
            get_tp_group=MagicMock(return_value=mock_group),
        ),
    ):
        yield mock_group


@pytest.fixture
def mock_vllm_config_producer():
    """Fixture providing mock VllmConfig for producer role."""
    config = Mock(spec=VllmConfig)
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.is_ec_producer = True
    config.ec_transfer_config.is_ec_consumer = False
    config.ec_transfer_config.ec_connector_extra_config = {
        "device_name": "mlx5_0:1",
        "num_workers": 2,
    }
    config.parallel_config = Mock()
    config.parallel_config.tensor_model_parallel_rank = 0
    config.parallel_config.tensor_model_parallel_size = 1
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.tensor_parallel_size = 1
    config.model_config = Mock()
    config.model_config.dtype = torch.float16
    config.model_config.get_inputs_embeds_size = Mock(return_value=768)
    return config


@pytest.fixture
def mock_vllm_config_consumer():
    """Fixture providing mock VllmConfig for consumer role."""
    config = Mock(spec=VllmConfig)
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.is_ec_producer = False
    config.ec_transfer_config.is_ec_consumer = True
    config.ec_transfer_config.ec_connector_extra_config = {
        "device_name": "mlx5_0:1",
        "num_workers": 2,
    }
    config.parallel_config = Mock()
    config.parallel_config.tensor_model_parallel_rank = 0
    config.parallel_config.tensor_model_parallel_size = 1
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.tensor_parallel_size = 1
    config.model_config = Mock()
    config.model_config.dtype = torch.float16
    config.model_config.get_inputs_embeds_size = Mock(return_value=768)
    return config


@pytest.fixture
def mock_request_with_3_mm():
    """Fixture providing mock Request with 3 multimodal items."""
    request_id = "test_req_123"
    mm_hashes = ["img_hash_1", "img_hash_2", "img_hash_3"]
    token_counts = [100, 150, 200]

    # Add ec_transfer_params with routing info
    ec_transfer_params = {}
    for mm_hash in mm_hashes:
        ec_transfer_params[mm_hash] = {
            "do_remote_encode": True,
            "remote_host": "127.0.0.1",
            "remote_port": 5600,
        }

    request = MockRequest(request_id, mm_hashes, token_counts, ec_transfer_params)
    return request


# ------------------ Unit Tests ------------------ #
class TestMooncakeECConnectorBasics:
    """Test basic Mooncake EC connector functionality."""

    def test_encoder_cache_embed_size_uses_deepstack_width(self):
        model_config = Mock()
        out_hidden_size = 4096
        deepstack_visual_indexes = [7, 15, 23]
        model_config.get_inputs_embeds_size.return_value = out_hidden_size
        model_config.hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(
                out_hidden_size=out_hidden_size,
                deepstack_visual_indexes=deepstack_visual_indexes,
            )
        )

        expected_embed_size = out_hidden_size * (1 + len(deepstack_visual_indexes))
        assert _get_encoder_cache_embed_size(model_config) == expected_embed_size

    def test_encoder_cache_embed_size_falls_back_to_language_width(self):
        model_config = Mock()
        model_config.get_inputs_embeds_size.return_value = 768
        model_config.hf_config = SimpleNamespace(vision_config=SimpleNamespace())

        assert _get_encoder_cache_embed_size(model_config) == 768

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_initialization_producer(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_producer,
        mock_parallel_state,
    ):
        """Test connector initializes correctly as producer."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        assert connector.role == ECConnectorRole.SCHEDULER
        assert connector.is_producer
        assert connector.connector_scheduler is not None
        assert connector.connector_worker is None

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_initialization_consumer(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_consumer,
        mock_parallel_state,
    ):
        """Test connector initializes correctly as consumer."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        assert connector.role == ECConnectorRole.SCHEDULER
        assert not connector.is_producer
        assert connector.connector_scheduler is not None
        assert connector.connector_worker is None

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_role_assignment(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_producer,
        mock_parallel_state,
    ):
        """Test role is correctly assigned."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_engine.register_memory.return_value = 0
        mock_transfer_engine.return_value = mock_engine

        scheduler_connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )
        worker_connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.WORKER,
        )

        assert scheduler_connector.role == ECConnectorRole.SCHEDULER
        assert worker_connector.role == ECConnectorRole.WORKER


class TestCacheExistence:
    """Test cache existence checking using has_cache_item() API."""

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.make_zmq_socket"
    )
    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_tensor_model_parallel_rank"
    )
    def test_has_cache_item_producer_returns_false(
        self,
        mock_tp_rank,
        mock_zmq_socket,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_producer,
        mock_request_with_3_mm,
        mock_parallel_state,
    ):
        """Test has_cache_item returns False for producer (no remote check)."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine
        mock_tp_rank.return_value = 0

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Producer should always return False
        for mm_feature in mock_request_with_3_mm.mm_features:
            result = connector.has_cache_item(
                mm_feature.identifier, mock_request_with_3_mm
            )
            assert not result, "Producer should not check remote cache"

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.make_zmq_socket"
    )
    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_tensor_model_parallel_rank"
    )
    def test_has_cache_item_no_request(
        self,
        mock_tp_rank,
        mock_zmq_socket,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_consumer,
        mock_parallel_state,
    ):
        """Request=None hits the guard in try/except and returns False."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine
        mock_tp_rank.return_value = 0

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Without request, should return False
        result = connector.has_cache_item("test_hash", None)
        assert not result

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.make_zmq_socket"
    )
    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_tensor_model_parallel_rank"
    )
    def test_has_cache_item_no_ec_transfer_params(
        self,
        mock_tp_rank,
        mock_zmq_socket,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_consumer,
        mock_parallel_state,
    ):
        """Test has_cache_item returns False when no ec_transfer_params."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine
        mock_tp_rank.return_value = 0

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        request = MockRequest("test_req", ["hash1"], [100], ec_transfer_params=None)

        result = connector.has_cache_item("hash1", request)
        assert not result

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.make_zmq_socket"
    )
    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_tensor_model_parallel_rank"
    )
    def test_has_cache_item_consumer_optimistic_when_transfer_params_ok(
        self,
        mock_tp_rank,
        mock_zmq_socket,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_consumer,
        mock_request_with_3_mm,
        mock_parallel_state,
    ):
        """Scheduler returns True when ec_transfer_params allow remote EC.

        The return is optimistic.
        """
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine
        mock_tp_rank.return_value = 0

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        assert connector.has_cache_item("img_hash_1", mock_request_with_3_mm) is True


class TestStateManagement:
    """Test connector state management."""

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_update_state_after_alloc_3_items(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_consumer,
        mock_request_with_3_mm,
        mock_parallel_state,
    ):
        """Test state update after allocation for 3 MM items."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Initial state should be empty
        assert len(connector.connector_scheduler._mm_hashes_need_recv) == 0

        # Update state for all 3 items
        for i in range(3):
            connector.update_state_after_alloc(mock_request_with_3_mm, index=i)

        # Check state updated for all 3
        assert len(connector.connector_scheduler._mm_hashes_need_recv) == 3

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_build_connector_meta_3_items(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_consumer,
        mock_request_with_3_mm,
        mock_parallel_state,
    ):
        """Test metadata building for 3 MM items."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Setup state for all 3 items
        for i in range(3):
            connector.update_state_after_alloc(mock_request_with_3_mm, index=i)

        # Build metadata
        scheduler_output = Mock(spec=SchedulerOutput)
        metadata = connector.build_connector_meta(scheduler_output)

        # Assert
        assert isinstance(metadata, MooncakeECConnectorMetadata)
        assert len(metadata.mm_hashes_to_recv) == 3

        # State should be cleared after building
        assert len(connector.connector_scheduler._mm_hashes_need_recv) == 0

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_build_connector_meta_empty(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_consumer,
        mock_parallel_state,
    ):
        """Test metadata building with empty state."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        scheduler_output = Mock(spec=SchedulerOutput)
        metadata = connector.build_connector_meta(scheduler_output)

        assert isinstance(metadata, MooncakeECConnectorMetadata)
        assert len(metadata.mm_hashes_to_recv) == 0


class TestRequestFinished:
    """Test request_finished method."""

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_request_finished_producer_returns_params(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_producer,
        mock_request_with_3_mm,
        mock_parallel_state,
    ):
        """Test request_finished returns params for producer."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_producer,
            role=ECConnectorRole.SCHEDULER,
        )

        should_send, params = connector.request_finished(mock_request_with_3_mm)

        assert should_send
        assert params is not None
        assert len(params) == 3
        assert "img_hash_1" in params
        assert "img_hash_2" in params
        assert "img_hash_3" in params
        assert params["img_hash_1"]["do_remote_encode"] is True

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_request_finished_consumer_returns_none(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_consumer,
        mock_request_with_3_mm,
        mock_parallel_state,
    ):
        """Test request_finished returns None for consumer."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        should_send, params = connector.request_finished(mock_request_with_3_mm)

        assert not should_send
        assert params is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.make_zmq_socket"
    )
    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_wait_for_load_failure_when_producer_transfer_not_done(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_make_zmq_socket,
        mock_vllm_config_consumer,
        mock_parallel_state,
    ):
        """Remote producer does not complete pull.

        For example, the cache is missing and the hash is failed.
        """
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_engine.register_memory.return_value = 0
        mock_transfer_engine.return_value = mock_engine

        mock_sock = MagicMock()
        mock_sock.send = AsyncMock()
        mock_sock.recv = AsyncMock(return_value=TRANS_ERROR)
        mock_sock.close = Mock()
        mock_sock.setsockopt = Mock()
        mock_make_zmq_socket.return_value = mock_sock

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER,
        )

        metadata = MooncakeECConnectorMetadata()
        mm_hash = "missing_on_producer"
        metadata.add_recv_req(
            "req_1",
            mm_hash,
            MMHashMeta(num_encoder_tokens=32, mm_addr=0),
            "127.0.0.1",
            5600,
        )
        connector.bind_connector_metadata(metadata)

        encoder_cache: dict[str, torch.Tensor] = {}
        connector.start_load_caches(encoder_cache)
        failed = connector.wait_for_load()

        assert mm_hash in failed
        assert mm_hash not in encoder_cache

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_has_cache_item_incomplete_routing_params(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_consumer,
        mock_parallel_state,
    ):
        """Test has_cache_item handles incomplete routing params."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        # Request with incomplete routing params
        request = MockRequest(
            "test_req",
            ["hash1"],
            [100],
            ec_transfer_params={
                "hash1": {"remote_host": "127.0.0.1"}
            },  # Missing remote_port
        )

        result = connector.has_cache_item("hash1", request)
        assert not result

    def test_consumer_does_not_save_remote_cache_state(self):
        """Consumer-only workers must not allocate producer transfer slots."""
        connector = MooncakeECConnector.__new__(MooncakeECConnector)
        connector._is_producer = False
        connector.connector_worker = Mock()

        connector.maybe_update_remote_cache_state({"hash1": torch.randn(1)})

        connector.connector_worker.maybe_update_remote_cache_state.assert_not_called()

    @patch(
        "vllm.distributed.ec_transfer.ec_connector.mooncake_connector.TransferEngine"
    )
    @patch("vllm.distributed.ec_transfer.ec_connector.mooncake_connector.get_ip")
    def test_update_state_after_alloc_no_ec_transfer_params(
        self,
        mock_get_ip,
        mock_transfer_engine,
        mock_vllm_config_consumer,
        mock_parallel_state,
    ):
        """Test update_state_after_alloc handles missing ec_transfer_params."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_engine = Mock()
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 5000
        mock_transfer_engine.return_value = mock_engine

        connector = MooncakeECConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.SCHEDULER,
        )

        request = MockRequest("test_req", ["hash1"], [100], ec_transfer_params=None)

        # Should not raise
        connector.update_state_after_alloc(request, index=0)
        assert len(connector.connector_scheduler._mm_hashes_need_recv) == 0

    def test_repeated_save_keeps_new_addr_when_old_addr_is_freed(self):
        """Freeing an old slot must not drop the latest mm_hash mapping."""
        worker = MooncakeECConnectorWorker.__new__(MooncakeECConnectorWorker)
        worker.transfer_buffer = Mock()
        old_addr = 0x1000
        new_addr = 0x2000
        worker.transfer_buffer.store_tensor.side_effect = [old_addr, new_addr]
        worker.local_mm_addrs = {}
        worker._addr_to_mm_hash = {}
        worker._mm_lock = threading.Lock()

        mm_hash = "repeat_hash"
        encoder_cache = {mm_hash: torch.randn(2, 4)}

        worker.save_caches(encoder_cache, mm_hash)
        worker.save_caches(encoder_cache, mm_hash)

        worker._on_pool_free(old_addr)
        assert worker.local_mm_addrs[mm_hash] == new_addr

        worker._on_pool_free(new_addr)
        assert mm_hash not in worker.local_mm_addrs
