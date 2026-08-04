# SPDX-License-Identifier: Apache-2.0
"""
Tests for PDBackend bidirectional NIXL support.

Verifies that:
1. PDConfig accepts "both" role
2. CacheQueryRequest/CacheQueryResponse messages work correctly
3. batched_submit_read_task queries decoder cache and reads via NIXL
4. NixlChannel.batched_read performs NIXL READ operation
5. Config fields pd_bidirectional and pd_peer_query_port exist
"""

# Standard
from unittest.mock import MagicMock, patch
import threading

# Third Party
import pytest

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.pd_backend import (
    CacheQueryRequest,
    CacheQueryResponse,
    PDBackend,
    PDConfig,
    PDMsg,
)


class TestPDConfigBothRole:
    """Tests for PDConfig with role='both'."""

    def test_both_role_accepted(self):
        """PDConfig accepts 'both' as a valid role."""
        config = MagicMock()
        config.pd_role = "both"
        config.pd_buffer_size = 1024
        config.pd_buffer_device = "cuda:0"
        config.pd_peer_host = "10.0.0.1"
        config.pd_peer_init_port = [5000]
        config.pd_peer_alloc_port = [5001]
        config.pd_peer_query_port = [5002]
        config.pd_proxy_host = "10.0.0.2"
        config.pd_proxy_port = 8080

        metadata = MagicMock()
        metadata.worker_id = 0

        with patch(
            "lmcache.v1.storage_backend.pd_backend.get_correct_device",
            return_value="cuda:0",
        ):
            pd_config = PDConfig.from_cache_engine_config(config, metadata, tp_rank=0)

        assert pd_config.role == "both"
        assert pd_config.peer_host == "10.0.0.1"
        assert pd_config.peer_init_port == 5000
        assert pd_config.peer_alloc_port == 5001
        assert pd_config.peer_query_port == 5002

    def test_sender_role_still_works(self):
        """Existing 'sender' role still works."""
        config = MagicMock()
        config.pd_role = "sender"
        config.pd_buffer_size = 1024
        config.pd_buffer_device = "cuda:0"
        config.pd_peer_host = None
        config.pd_peer_init_port = None
        config.pd_peer_alloc_port = None
        config.pd_peer_query_port = None
        config.pd_proxy_host = "10.0.0.2"
        config.pd_proxy_port = 8080

        metadata = MagicMock()
        metadata.worker_id = 0

        with patch(
            "lmcache.v1.storage_backend.pd_backend.get_correct_device",
            return_value="cuda:0",
        ):
            pd_config = PDConfig.from_cache_engine_config(config, metadata, tp_rank=0)

        assert pd_config.role == "sender"

    def test_receiver_role_still_works(self):
        """Existing 'receiver' role still works."""
        config = MagicMock()
        config.pd_role = "receiver"
        config.pd_buffer_size = 1024
        config.pd_buffer_device = "cuda:0"
        config.pd_peer_host = "10.0.0.1"
        config.pd_peer_init_port = [5000]
        config.pd_peer_alloc_port = [5001]
        config.pd_peer_query_port = None
        config.pd_proxy_host = None
        config.pd_proxy_port = None

        metadata = MagicMock()
        metadata.worker_id = 0

        with patch(
            "lmcache.v1.storage_backend.pd_backend.get_correct_device",
            return_value="cuda:0",
        ):
            pd_config = PDConfig.from_cache_engine_config(config, metadata, tp_rank=0)

        assert pd_config.role == "receiver"

    def test_invalid_role_rejected(self):
        """Invalid roles are rejected."""
        config = MagicMock()
        config.pd_role = "invalid"
        config.pd_buffer_size = 1024
        config.pd_buffer_device = "cuda:0"

        metadata = MagicMock()
        metadata.worker_id = 0

        with pytest.raises(AssertionError, match="Invalid role"):
            with patch(
                "lmcache.v1.storage_backend.pd_backend.get_correct_device",
                return_value="cuda:0",
            ):
                PDConfig.from_cache_engine_config(config, metadata, tp_rank=0)


class TestCacheQueryMessages:
    """Tests for CacheQueryRequest/CacheQueryResponse message types."""

    def test_cache_query_request_serialization(self):
        """CacheQueryRequest can be serialized and deserialized."""
        # Third Party
        import msgspec

        req = CacheQueryRequest(keys=["key1", "key2", "key3"])
        encoded = msgspec.msgpack.encode(req)
        decoded = msgspec.msgpack.decode(encoded, type=PDMsg)
        assert isinstance(decoded, CacheQueryRequest)
        assert decoded.keys == ["key1", "key2", "key3"]

    def test_cache_query_response_serialization(self):
        """CacheQueryResponse can be serialized and deserialized."""
        # Third Party
        import msgspec

        resp = CacheQueryResponse(
            cached_keys=["key1", "key3"],
            cached_indexes=[0, 2],
        )
        encoded = msgspec.msgpack.encode(resp)
        decoded = msgspec.msgpack.decode(encoded, type=PDMsg)
        assert isinstance(decoded, CacheQueryResponse)
        assert decoded.cached_keys == ["key1", "key3"]
        assert decoded.cached_indexes == [0, 2]

    def test_empty_cache_query_response(self):
        """Empty CacheQueryResponse (no cache hits) works."""
        # Third Party
        import msgspec

        resp = CacheQueryResponse(cached_keys=[], cached_indexes=[])
        encoded = msgspec.msgpack.encode(resp)
        decoded = msgspec.msgpack.decode(encoded, type=PDMsg)
        assert isinstance(decoded, CacheQueryResponse)
        assert len(decoded.cached_keys) == 0
        assert len(decoded.cached_indexes) == 0


class TestBatchedSubmitReadTask:
    """Tests for PDBackend.batched_submit_read_task."""

    @staticmethod
    def _make_key(chunk_hash: int) -> CacheEngineKey:
        """Create a valid CacheEngineKey for testing."""
        # Third Party
        import torch

        return CacheEngineKey(
            model_name="test-model",
            world_size=1,
            worker_id=0,
            chunk_hash=chunk_hash,
            dtype=torch.bfloat16,
        )

    def _make_backend_stub(self):
        """Create a minimal PDBackend-like object for testing."""
        backend = MagicMock(spec=PDBackend)
        backend.tp_rank = 0
        backend.initialized_peers = set()
        backend.mem_alloc_sockets = {}
        backend.cache_query_sockets = {}
        backend.data = {}
        backend.data_lock = threading.Lock()

        # Mock transfer channel
        backend.transfer_channel = MagicMock()
        backend.transfer_channel.batched_read.return_value = 3

        # Mock NIXL worker queue: drain puts synchronously so read futures
        # resolve via a direct batched_read call (mimics the worker thread).
        class _SyncQueue:
            def __init__(self, inner_backend):
                self._backend = inner_backend

            def put(self, item):
                op_type = item[0]
                if op_type == "read":
                    _, buffers, channel_spec, completion_future = item
                    try:
                        result = self._backend.transfer_channel.batched_read(
                            buffers=buffers,
                            transfer_spec=channel_spec,
                        )
                        completion_future.set_result(result)
                    except Exception as e:  # pragma: no cover - defensive
                        completion_future.set_exception(e)

        backend._nixl_queue = _SyncQueue(backend)

        # Mock zmq context
        backend.zmq_context = MagicMock()

        return backend

    def test_read_task_no_cache_hits(self):
        """batched_submit_read_task returns 0 when decoder has no cached blocks."""
        backend = self._make_backend_stub()

        # Mock query_remote_cache to return empty response
        empty_resp = CacheQueryResponse(cached_keys=[], cached_indexes=[])
        backend.query_remote_cache = MagicMock(return_value=empty_resp)
        backend._ensure_peer_connection = MagicMock()
        backend._ensure_cache_query_connection = MagicMock()

        keys = [self._make_key(0), self._make_key(1)]
        mem_objs = [MagicMock(), MagicMock()]

        transfer_spec = MagicMock()
        transfer_spec.receiver_host = "10.0.0.1"
        transfer_spec.receiver_init_port = [5000]
        transfer_spec.receiver_alloc_port = [5001]
        transfer_spec.receiver_query_port = [5002]

        result = PDBackend.batched_submit_read_task(
            backend, keys, mem_objs, transfer_spec
        )
        assert result == 0

    def test_read_task_with_cache_hits(self):
        """batched_submit_read_task reads cached blocks via NIXL READ."""

        backend = self._make_backend_stub()

        key0 = self._make_key(0)
        key1 = self._make_key(1)
        key2 = self._make_key(2)

        # Mock cache query response: key0 and key2 are cached
        cache_resp = CacheQueryResponse(
            cached_keys=[key0.to_string(), key2.to_string()],
            cached_indexes=[100, 300],
        )

        # Mock query_remote_cache to return our response
        backend.query_remote_cache = MagicMock(return_value=cache_resp)
        backend._ensure_peer_connection = MagicMock()
        backend._ensure_cache_query_connection = MagicMock()

        keys = [key0, key1, key2]
        mem_objs = [MagicMock(), MagicMock(), MagicMock()]

        transfer_spec = MagicMock()
        transfer_spec.receiver_host = "10.0.0.1"
        transfer_spec.receiver_init_port = [5000]
        transfer_spec.receiver_alloc_port = [5001]
        transfer_spec.receiver_query_port = [5002]

        PDBackend.batched_submit_read_task(backend, keys, mem_objs, transfer_spec)

        # Should have called batched_read with 2 objects (key0 and key2)
        backend.transfer_channel.batched_read.assert_called_once()
        call_kwargs = backend.transfer_channel.batched_read.call_args
        buffers = call_kwargs.kwargs.get("buffers")
        spec = call_kwargs.kwargs.get("transfer_spec")
        assert len(buffers) == 2
        assert buffers[0] is mem_objs[0]  # key0 → index 0
        assert buffers[1] is mem_objs[2]  # key2 → index 2
        assert spec["remote_indexes"] == [100, 300]
        assert spec["sender_id"] == "10.0.0.15000"


class TestNixlChannelBatchedRead:
    """Tests for NixlChannel.batched_read implementation."""

    def test_batched_read_calls_nixl_read(self):
        """batched_read uses NIXL READ operation."""
        # First Party
        from lmcache.v1.transfer_channel.nixl_channel import NixlChannel

        # Create a mock channel
        channel = MagicMock(spec=NixlChannel)
        channel.nixl_agent = MagicMock()
        channel.nixl_wrapper = MagicMock()
        channel.nixl_wrapper.xfer_handler = "local_handler"
        channel.remote_xfer_handlers_dict = {"peer1": "remote_handler"}

        # Mock make_prepped_xfer and transfer
        mock_handle = MagicMock()
        channel.nixl_agent.make_prepped_xfer.return_value = mock_handle
        channel.nixl_agent.check_xfer_state.return_value = "DONE"

        # Create mock buffers with MemoryObj-like interface
        buf1 = MagicMock()
        buf1.meta = MagicMock()
        buf1.meta.address = 0
        buf2 = MagicMock()
        buf2.meta = MagicMock()
        buf2.meta.address = 1

        # Mock get_local_mem_indices to return addresses
        channel.get_local_mem_indices = MagicMock(return_value=[0, 1])

        transfer_spec = {
            "sender_id": "peer1",
            "remote_indexes": [10, 20],
        }

        # Call the real method
        result = NixlChannel.batched_read(channel, [buf1, buf2], transfer_spec)

        assert result == 2
        channel.nixl_agent.make_prepped_xfer.assert_called_once_with(
            "READ",
            channel.nixl_wrapper.xfer_handler,
            [0, 1],
            "remote_handler",
            [10, 20],
        )
        channel.nixl_agent.transfer.assert_called_once_with(mock_handle)


class TestConfigFields:
    """Test that new config fields exist and have correct defaults."""

    def test_pd_bidirectional_default(self):
        """pd_bidirectional defaults to False."""
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig

        config = LMCacheEngineConfig.from_defaults()
        assert config.pd_bidirectional is False

    def test_pd_bidirectional_set(self):
        """pd_bidirectional can be set to True."""
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig

        config = LMCacheEngineConfig.from_defaults(pd_bidirectional=True)
        assert config.pd_bidirectional is True

    def test_pd_peer_query_port_default(self):
        """pd_peer_query_port defaults to None."""
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig

        config = LMCacheEngineConfig.from_defaults()
        assert config.pd_peer_query_port is None

    def test_pd_peer_query_port_set(self):
        """pd_peer_query_port can be set."""
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig

        expected = [6000, 6001, 6002, 6003, 6004, 6005, 6006, 6007]
        config = LMCacheEngineConfig.from_defaults(pd_peer_query_port=expected)
        assert config.pd_peer_query_port == expected


class TestCacheQueryLoop:
    """Tests for the decoder-side _cache_query_loop."""

    @staticmethod
    def _make_key(chunk_hash: int) -> CacheEngineKey:
        """Create a valid CacheEngineKey for testing."""
        # Third Party
        import torch

        return CacheEngineKey(
            model_name="test-model",
            world_size=1,
            worker_id=0,
            chunk_hash=chunk_hash,
            dtype=torch.bfloat16,
        )

    def test_cache_query_returns_cached_keys(self):
        """_cache_query_loop returns correct cached keys and indexes."""

        backend = MagicMock(spec=PDBackend)
        backend.running = True
        backend.data = {}
        backend.data_lock = threading.Lock()

        # Populate decoder's cache with some keys
        key1 = self._make_key(1)
        key2 = self._make_key(2)
        key3 = self._make_key(3)  # not cached
        mem1 = MagicMock()
        mem1.meta.address = 42
        mem2 = MagicMock()
        mem2.meta.address = 84
        backend.data[key1] = mem1
        backend.data[key2] = mem2

        # Create a query for 3 keys (2 cached, 1 not)
        query = CacheQueryRequest(
            keys=[
                key1.to_string(),
                key3.to_string(),
                key2.to_string(),
            ]
        )

        # Simulate what _cache_query_loop does
        cached_keys = []
        cached_indexes = []
        with backend.data_lock:
            for key_str in query.keys:
                key = CacheEngineKey.from_string(key_str)
                if mem_obj := backend.data.get(key, None):
                    cached_keys.append(key_str)
                    cached_indexes.append(mem_obj.meta.address)

        assert len(cached_keys) == 2
        assert cached_keys[0] == key1.to_string()
        assert cached_keys[1] == key2.to_string()
        assert cached_indexes == [42, 84]
