# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures and utilities for cache controller tests."""

# Standard
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.controllers.full_sync_tracker import FullSyncTracker
from lmcache.v1.cache_controller.controllers.kv_controller import KVController
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStartRetMsg,
    FullSyncStatusMsg,
    FullSyncStatusRetMsg,
    KVOpEvent,
    RegisterMsg,
)
from lmcache.v1.cache_controller.utils import RegistryTree, WorkerInfo
from lmcache.v1.config import LMCacheEngineConfig

# Constants
LOCATION = "LocalCPUBackend"


# ============= Config & Key Creation =============


def create_test_config(
    batch_size: int = 100,
    batch_interval_ms: int = 0,
    startup_delay_s: float = 0.0,
    status_poll_interval_s: float = 0.01,
    max_retry_count: int = 3,
    retry_delay_s: float = 0.01,
):
    """Create a test configuration."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        lmcache_instance_id="test_instance",
    )
    config.extra_config = {
        "full_sync_batch_size": batch_size,
        "full_sync_batch_interval_ms": batch_interval_ms,
        "full_sync_startup_delay_s": startup_delay_s,
        "full_sync_status_poll_interval_s": status_poll_interval_s,
        "full_sync_max_retry_count": max_retry_count,
        "full_sync_retry_delay_s": retry_delay_s,
    }
    return config


def create_test_key(key_id: int) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=0,
        chunk_hash=key_id,
        dtype=torch.bfloat16,
    )


# ============= Mock Classes =============


class MockWorker:
    """Mock LMCacheWorker for testing."""

    def __init__(self):
        self.worker_id = 0
        self.messages = []
        self.req_responses = []
        self._response_index = 0

    def put_msg(self, msg):
        """Record pushed messages."""
        self.messages.append(msg)

    async def async_put_and_wait_msg(self, msg):
        """Return predefined responses for REQ-REP messages."""
        if self._response_index < len(self.req_responses):
            response = self.req_responses[self._response_index]
            self._response_index += 1
            return response
        # Default responses
        if isinstance(msg, FullSyncStartMsg):
            return FullSyncStartRetMsg(sync_id=msg.sync_id, accepted=True)
        elif isinstance(msg, FullSyncStatusMsg):
            return FullSyncStatusRetMsg(
                sync_id=msg.sync_id,
                is_complete=True,
                global_progress=1.0,
                can_exit_freeze=True,
            )
        return None

    def set_responses(self, responses):
        """Set predefined responses."""
        self.req_responses = responses
        self._response_index = 0


class MockLMCacheEngine:
    """Mock LMCacheEngine for testing."""

    def __init__(self):
        self._freeze = False
        self.freeze_calls = []

    def freeze(self, enabled: bool):
        """Record freeze mode changes."""
        self.freeze_calls.append(enabled)
        self._freeze = enabled


class MockLocalCPUBackend:
    """Mock LocalCPUBackend for testing."""

    def __init__(self, keys=None):
        self._keys = keys or []

    def get_keys(self):
        """Return predefined keys."""
        return self._keys

    def __str__(self):
        return LOCATION


# ============= Helper Class =============


class H:
    """Helper class for test utilities."""

    @staticmethod
    def create_tracker(threshold=0.8, timeout=300.0):
        """Create FullSyncTracker with registry."""
        reg = RegistryTree()
        tracker = FullSyncTracker(reg, threshold, timeout)
        return tracker, reg

    @staticmethod
    def create_controller(threshold=0.8, timeout=300.0):
        """Create KVController with test settings."""
        reg = RegistryTree()
        ctrl = KVController(reg, threshold, timeout)
        ctrl.cluster_executor = MagicMock()
        return ctrl

    @staticmethod
    def reg_worker(reg, inst, wid, ip="127.0.0.1", port=8000):
        """Register a worker."""
        reg.register_worker(inst, wid, ip, port, None, MagicMock(), time.time())

    @staticmethod
    async def reg_worker_async(ctrl, inst, wid, ip="127.0.0.1", port=8000):
        """Register worker via registration controller."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock:
            mock.return_value = MagicMock()
            mock.return_value.sent_messages = []
            return await ctrl.register(RegisterMsg(inst, wid, ip, port, None))

    @staticmethod
    def start_msg(inst, wid, sid, keys=100, batches=5):
        """Create FullSyncStartMsg."""
        return FullSyncStartMsg(inst, wid, LOCATION, sid, keys, batches)

    @staticmethod
    def batch_msg(inst, wid, sid, bid, keys):
        """Create FullSyncBatchMsg."""
        return FullSyncBatchMsg(inst, wid, LOCATION, sid, bid, keys)

    @staticmethod
    def end_msg(inst, wid, sid, total):
        """Create FullSyncEndMsg."""
        return FullSyncEndMsg(inst, wid, LOCATION, sid, total)

    @staticmethod
    def batched_op(inst, wid, op, key, seq=0):
        """Create BatchedKVOperationMsg."""
        return BatchedKVOperationMsg(inst, wid, LOCATION, [KVOpEvent(op, key, seq)])

    @staticmethod
    def start_ret(sync_id, accepted=True, error_msg=None):
        """Create FullSyncStartRetMsg."""
        return FullSyncStartRetMsg(
            sync_id=sync_id, accepted=accepted, error_msg=error_msg
        )

    @staticmethod
    def status_ret(sync_id, is_complete=True, progress=1.0, can_exit=True):
        """Create FullSyncStatusRetMsg."""
        return FullSyncStatusRetMsg(
            sync_id=sync_id,
            is_complete=is_complete,
            global_progress=progress,
            can_exit_freeze=can_exit,
        )


# ============= Pytest Fixtures =============


@pytest.fixture
def mock_reg_controller():
    """Create a mock RegistrationController."""
    controller = Mock()
    controller.get_workers = Mock(return_value=[0, 1])
    controller.get_socket = Mock()
    controller.get_peer_init_url = Mock(return_value="tcp://localhost:5000")

    # Create a mock registry with all required methods
    mock_registry = Mock()
    mock_registry.seq_tracker = {}
    mock_registry.kv_pool = {}

    def mock_get_seq_num(instance_id, worker_id, location):
        key = (instance_id, worker_id, location)
        return mock_registry.seq_tracker.get(key)

    def mock_update_seq_num(instance_id, worker_id, location, seq_num):
        key = (instance_id, worker_id, location)
        mock_registry.seq_tracker[key] = seq_num
        return True

    def mock_admit_kv(instance_id, worker_id, location, key):
        report_id = (instance_id, worker_id)
        if report_id not in mock_registry.kv_pool:
            mock_registry.kv_pool[report_id] = {}
        if location not in mock_registry.kv_pool[report_id]:
            mock_registry.kv_pool[report_id][location] = set()
        mock_registry.kv_pool[report_id][location].add(key)
        return True

    def mock_evict_kv(instance_id, worker_id, location, key):
        report_id = (instance_id, worker_id)
        if report_id in mock_registry.kv_pool:
            if location in mock_registry.kv_pool[report_id]:
                if key in mock_registry.kv_pool[report_id][location]:
                    mock_registry.kv_pool[report_id][location].remove(key)
                    if not mock_registry.kv_pool[report_id][location]:
                        del mock_registry.kv_pool[report_id][location]
                    if not mock_registry.kv_pool[report_id]:
                        del mock_registry.kv_pool[report_id]
                    return True
        return False

    def mock_find_kv(key, exclude_instance_id=None):
        for report_id, locations in mock_registry.kv_pool.items():
            instance_id, worker_id = report_id
            if exclude_instance_id and instance_id == exclude_instance_id:
                continue
            for location, keys in locations.items():
                if key in keys:
                    # Return a mock KVChunkInfo
                    # First Party
                    from lmcache.v1.cache_controller.utils import KVChunkInfo

                    return KVChunkInfo(instance_id, worker_id, location)
        return None

    def mock_get_worker_kv_keys(instance_id, worker_id, location):
        report_id = (instance_id, worker_id)
        if report_id in mock_registry.kv_pool:
            return mock_registry.kv_pool[report_id].get(location, set())
        return set()

    def mock_get_total_kv_count():
        count = 0
        for locations in mock_registry.kv_pool.values():
            for keys in locations.values():
                count += len(keys)
        return count

    def mock_get_seq_discontinuity_count():
        # Simple implementation for testing
        return getattr(mock_registry, "_seq_discontinuity_count", 0)

    def mock_deregister_worker(instance_id, worker_id):
        report_id = (instance_id, worker_id)
        if report_id in mock_registry.kv_pool:
            del mock_registry.kv_pool[report_id]
        # Also clean up seq_tracker
        keys_to_remove = [
            k
            for k in mock_registry.seq_tracker.keys()
            if k[0] == instance_id and k[1] == worker_id
        ]
        for key in keys_to_remove:
            del mock_registry.seq_tracker[key]
        return True

    def mock_find_kv_with_worker_info(key, exclude_instance_id=None):
        """Find KV and return (kv_info, peer_init_url, current_keys)"""
        for report_id, locations in mock_registry.kv_pool.items():
            instance_id, worker_id = report_id
            if exclude_instance_id and instance_id == exclude_instance_id:
                continue
            for location, keys in locations.items():
                if key in keys:
                    # First Party
                    from lmcache.v1.cache_controller.utils import KVChunkInfo

                    kv_info = KVChunkInfo(instance_id, worker_id, location)
                    # Get peer_init_url from controller
                    peer_init_url = controller.get_peer_init_url(instance_id, worker_id)
                    return (kv_info, peer_init_url, keys)
        return None

    def mock_handle_batched_kv_operations(msg):
        """Handle batched KV operations"""
        # First Party
        from lmcache.v1.cache_controller.message import OpType

        report_id = (msg.instance_id, msg.worker_id)

        for op in msg.operations:
            # Check for sequence discontinuity
            key = (msg.instance_id, msg.worker_id, msg.location)
            last_seq_num = mock_registry.seq_tracker.get(key)
            if last_seq_num is not None:
                expected_seq = last_seq_num + 1
                if op.seq_num != expected_seq:
                    # Increment discontinuity counter
                    mock_registry._seq_discontinuity_count += 1

            # Update sequence number
            mock_registry.seq_tracker[key] = op.seq_num

            # Handle operation
            if op.op_type == OpType.ADMIT:
                if report_id not in mock_registry.kv_pool:
                    mock_registry.kv_pool[report_id] = {}
                if msg.location not in mock_registry.kv_pool[report_id]:
                    mock_registry.kv_pool[report_id][msg.location] = set()
                mock_registry.kv_pool[report_id][msg.location].add(op.key)
            elif op.op_type == OpType.EVICT:
                if report_id in mock_registry.kv_pool:
                    if msg.location in mock_registry.kv_pool[report_id]:
                        mock_registry.kv_pool[report_id][msg.location].discard(op.key)
                        if not mock_registry.kv_pool[report_id][msg.location]:
                            del mock_registry.kv_pool[report_id][msg.location]
                        if not mock_registry.kv_pool[report_id]:
                            del mock_registry.kv_pool[report_id]
        return True

    mock_registry.get_seq_num = Mock(side_effect=mock_get_seq_num)
    mock_registry.update_seq_num = Mock(side_effect=mock_update_seq_num)
    mock_registry.admit_kv = Mock(side_effect=mock_admit_kv)
    mock_registry.evict_kv = Mock(side_effect=mock_evict_kv)
    mock_registry.find_kv = Mock(side_effect=mock_find_kv)
    mock_registry.get_worker_kv_keys = Mock(side_effect=mock_get_worker_kv_keys)
    mock_registry.get_total_kv_count = Mock(side_effect=mock_get_total_kv_count)
    mock_registry.get_seq_discontinuity_count = Mock(
        side_effect=mock_get_seq_discontinuity_count
    )
    mock_registry.deregister_worker = Mock(side_effect=mock_deregister_worker)
    mock_registry.find_kv_with_worker_info = Mock(
        side_effect=mock_find_kv_with_worker_info
    )
    mock_registry.handle_batched_kv_operations = Mock(
        side_effect=mock_handle_batched_kv_operations
    )
    mock_registry._seq_discontinuity_count = 0

    controller.registry = mock_registry
    return controller


@pytest.fixture
def mock_kv_controller():
    """Create a mock KVController."""
    controller = Mock()
    controller.admit = AsyncMock()
    controller.evict = AsyncMock()
    controller.lookup = AsyncMock()
    controller.clear = AsyncMock()
    controller.pin = AsyncMock()
    controller.compress = AsyncMock()
    controller.decompress = AsyncMock()
    controller.move = AsyncMock()
    controller.deregister = AsyncMock()
    controller.kv_pool = {}
    controller.seq_tracker = {}

    # Mock full_sync_tracker
    mock_tracker = Mock()
    mock_tracker.should_request_full_sync = Mock(return_value=(False, None))
    controller.full_sync_tracker = mock_tracker

    return controller


@pytest.fixture
def mock_cluster_executor():
    """Create a mock LMCacheClusterExecutor."""
    executor = Mock()
    executor.execute = AsyncMock()
    executor.clear = AsyncMock()
    executor.pin = AsyncMock()
    executor.compress = AsyncMock()
    executor.decompress = AsyncMock()
    executor.move = AsyncMock()
    executor.health = AsyncMock()
    executor.check_finish = AsyncMock()
    return executor


@pytest.fixture
def mock_zmq_socket():
    """Create a mock ZMQ socket."""
    socket = AsyncMock()
    socket.send = AsyncMock()
    socket.recv = AsyncMock()
    socket.send_multipart = AsyncMock()
    socket.recv_multipart = AsyncMock()
    socket.close = Mock()
    socket.get = Mock(return_value=0)
    return socket


@pytest.fixture
def mock_zmq_context():
    """Create a mock ZMQ context."""
    context = Mock()
    context.socket = Mock()
    context.term = Mock()
    return context


@pytest.fixture
def sample_worker_info():
    """Create sample worker info for testing."""
    return WorkerInfo(
        instance_id="test_instance",
        worker_id=0,
        ip="127.0.0.1",
        port=5000,
        peer_init_url="tcp://127.0.0.1:6000",
        registration_time=time.time(),
        last_heartbeat_time=time.time(),
    )


@pytest.fixture
def mock_lmcache_engine():
    """Create a mock LMCacheEngine."""
    engine = Mock()
    engine.move = Mock(return_value=100)
    engine.compress = Mock(return_value=100)
    engine.decompress = Mock(return_value=100)
    engine.lookup = Mock(return_value=100)
    engine.clear = Mock(return_value=100)
    engine.health = Mock(return_value=0)
    return engine
