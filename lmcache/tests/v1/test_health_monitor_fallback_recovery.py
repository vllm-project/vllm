# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for RemoteBackendHealthCheck fallback and recovery.

Tests the health check lifecycle: healthy -> failure -> LOCAL_CPU fallback -> recovery.
"""

# Standard
from typing import List, Optional
from unittest.mock import MagicMock
import asyncio
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.health_monitor.base import HealthMonitor
from lmcache.v1.health_monitor.checks.remote_backend_check import (
    RemoteBackendHealthCheck,
)
from lmcache.v1.health_monitor.constants import FallbackPolicy
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class ControllablePingConnector(RemoteConnector):
    """Mock connector with controllable ping response."""

    def __init__(self):
        self._ping_error_code = 0
        self._ping_lock = threading.Lock()

    def set_ping_error_code(self, error_code: int) -> None:
        with self._ping_lock:
            self._ping_error_code = error_code

    def support_ping(self) -> bool:
        return True

    async def ping(self) -> int:
        with self._ping_lock:
            return self._ping_error_code

    async def exists(self, key: CacheEngineKey) -> bool:
        return False

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return False

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        pass

    async def list(self) -> List[str]:
        return []

    async def close(self) -> None:
        pass


class EventLoopThread:
    """Helper to manage an event loop in a background thread."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        time.sleep(0.05)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=1.0)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def event_loop_thread():
    elt = EventLoopThread()
    elt.start()
    yield elt
    elt.stop()


@pytest.fixture
def test_config():
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=0.1,
        remote_url="blackhole://test",
        remote_serde="naive",
        lmcache_instance_id="test_health_fallback_instance",
        extra_config={
            "ping_timeout": 1.0,
            "ping_interval": 0.1,
            "fallback_policy": "local_cpu",
        },
    )


@pytest.fixture
def test_metadata():
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
        role="worker",
    )


@pytest.fixture
def controllable_connector():
    return ControllablePingConnector()


@pytest.fixture
def mock_remote_backend(event_loop_thread, test_config, controllable_connector):
    backend = MagicMock(spec=RemoteBackend)
    backend.remote_url = "controllable://test:1234"
    backend.loop = event_loop_thread.loop
    backend.connection = controllable_connector
    backend.config = test_config
    backend.init_connection = MagicMock()
    backend.get_blocking_failed_count = 0
    return backend


@pytest.fixture
def mock_local_cpu_backend():
    backend = MagicMock(spec=LocalCPUBackend)
    backend.use_hot = False
    backend.clear = MagicMock()
    return backend


@pytest.fixture
def mock_storage_manager(mock_remote_backend, mock_local_cpu_backend):
    storage_manager = MagicMock()
    storage_manager.storage_backends = {
        "RemoteBackend": mock_remote_backend,
        "LocalCPUBackend": mock_local_cpu_backend,
    }
    storage_manager.local_cpu_backend = mock_local_cpu_backend
    storage_manager.set_backend_bypass = MagicMock()
    storage_manager.is_backend_bypassed = MagicMock(return_value=False)
    return storage_manager


@pytest.fixture
def mock_manager(test_config, mock_storage_manager):
    engine = MagicMock()
    engine.storage_manager = mock_storage_manager
    manager = MagicMock()
    manager.lmcache_engine = engine
    manager._config = test_config
    return manager


# ==============================================================================
# Test Classes
# ==============================================================================


class TestRemoteBackendHealthCheckFallbackRecovery:
    """Test cases for RemoteBackendHealthCheck fallback and recovery."""

    def test_health_check_creation_and_basic_checks(
        self,
        mock_manager,
        mock_remote_backend,
        controllable_connector,
        event_loop_thread,
    ):
        """Test health check creation and basic pass/fail scenarios."""
        # Verify creation with LOCAL_CPU fallback
        checks = RemoteBackendHealthCheck.create(mock_manager)
        assert len(checks) == 1
        check = checks[0]
        assert isinstance(check, RemoteBackendHealthCheck)
        assert check.fallback_policy == FallbackPolicy.LOCAL_CPU
        assert check.get_bypass_backend_name() == "RemoteBackend"

        # Test pass when ping returns 0
        controllable_connector.set_ping_error_code(0)
        assert check.check() is True

        # Test fail when ping returns non-zero
        controllable_connector.set_ping_error_code(1)
        assert check.check() is False

    def test_fallback_and_recovery_cycle(
        self,
        mock_manager,
        mock_remote_backend,
        mock_local_cpu_backend,
        mock_storage_manager,
        controllable_connector,
        event_loop_thread,
    ):
        """Test complete fallback and recovery cycle."""
        monitor = HealthMonitor(manager=mock_manager, ping_interval=0.1)
        assert len(monitor._health_checks) == 1

        # Initially healthy
        controllable_connector.set_ping_error_code(0)
        assert monitor._run_all_checks() is True

        mock_storage_manager.set_backend_bypass.reset_mock()

        # Simulate failure - should apply LOCAL_CPU fallback
        controllable_connector.set_ping_error_code(1)
        assert monitor._run_all_checks() is True  # Still healthy due to fallback
        mock_storage_manager.set_backend_bypass.assert_called_with(
            "RemoteBackend", True
        )
        assert mock_local_cpu_backend.use_hot is True

        mock_storage_manager.set_backend_bypass.reset_mock()

        # Recovery - ping returns 0 again
        controllable_connector.set_ping_error_code(0)
        monitor._run_all_checks()
        mock_storage_manager.set_backend_bypass.assert_called_with(
            "RemoteBackend", False
        )
        assert mock_local_cpu_backend.use_hot is False
        mock_local_cpu_backend.clear.assert_called_once()

    def test_full_cycle_with_monitor_thread(
        self,
        mock_manager,
        mock_remote_backend,
        mock_local_cpu_backend,
        mock_storage_manager,
        controllable_connector,
        event_loop_thread,
    ):
        """Test fallback/recovery using the monitor thread."""
        monitor = HealthMonitor(manager=mock_manager, ping_interval=0.05)
        controllable_connector.set_ping_error_code(0)

        thread = monitor.start()
        assert thread is not None

        try:
            time.sleep(0.2)
            assert monitor.is_healthy() is True

            # Simulate failure
            controllable_connector.set_ping_error_code(1)
            time.sleep(0.2)
            assert monitor.is_healthy() is True
            assert mock_local_cpu_backend.use_hot is True

            # Simulate recovery
            controllable_connector.set_ping_error_code(0)
            time.sleep(0.2)
            assert monitor.is_healthy() is True
            assert mock_local_cpu_backend.use_hot is False
        finally:
            monitor.stop()

    def test_multiple_failures_do_not_duplicate_fallback(
        self,
        mock_manager,
        mock_local_cpu_backend,
        mock_storage_manager,
        controllable_connector,
        event_loop_thread,
    ):
        """Test that consecutive failures don't duplicate fallback actions."""
        monitor = HealthMonitor(manager=mock_manager, ping_interval=0.1)

        controllable_connector.set_ping_error_code(0)
        monitor._run_all_checks()
        mock_storage_manager.set_backend_bypass.reset_mock()

        # Multiple failures should only trigger fallback once
        controllable_connector.set_ping_error_code(1)
        monitor._run_all_checks()
        first_call_count = mock_storage_manager.set_backend_bypass.call_count
        assert first_call_count == 1

        monitor._run_all_checks()
        monitor._run_all_checks()
        assert mock_storage_manager.set_backend_bypass.call_count == first_call_count

    def test_multiple_backends_fallback_and_partial_recovery(
        self,
        mock_local_cpu_backend,
        mock_storage_manager,
        event_loop_thread,
        test_config,
    ):
        """
        Test that original_hot_cache is preserved correctly when multiple backends
        fail and only some recover.

        Scenario:
        1. Initial state: use_hot = False
        2. Backend A fails -> use_hot = True, original saved as False
        3. Backend B fails -> use_hot = True, original should still be False
        4. Backend A recovers -> use_hot should STAY True (B still down)
        5. Backend B recovers -> use_hot should be restored to False
        """
        # Create two controllable connectors for two backends
        connector_a = ControllablePingConnector()
        connector_b = ControllablePingConnector()

        # Create two mock remote backends
        backend_a = MagicMock(spec=RemoteBackend)
        backend_a.remote_url = "controllable://test_a:1234"
        backend_a.loop = event_loop_thread.loop
        backend_a.connection = connector_a
        backend_a.config = test_config
        backend_a.init_connection = MagicMock()
        backend_a.get_blocking_failed_count = 0

        backend_b = MagicMock(spec=RemoteBackend)
        backend_b.remote_url = "controllable://test_b:1234"
        backend_b.loop = event_loop_thread.loop
        backend_b.connection = connector_b
        backend_b.config = test_config
        backend_b.init_connection = MagicMock()
        backend_b.get_blocking_failed_count = 0

        # Update storage manager to have two remote backends
        mock_storage_manager.storage_backends = {
            "RemoteBackend_A": backend_a,
            "RemoteBackend_B": backend_b,
            "LocalCPUBackend": mock_local_cpu_backend,
        }

        # Create mock manager
        engine = MagicMock()
        engine.storage_manager = mock_storage_manager
        manager = MagicMock()
        manager.lmcache_engine = engine
        manager._config = test_config

        # Create two health checks manually
        check_a = RemoteBackendHealthCheck(backend=backend_a)
        check_a._backend_name = "RemoteBackend_A"

        check_b = RemoteBackendHealthCheck(backend=backend_b)
        check_b._backend_name = "RemoteBackend_B"

        # Create monitor and manually set health checks
        monitor = HealthMonitor(manager=manager, ping_interval=0.1)
        monitor._health_checks = [check_a, check_b]
        monitor._previous_check_status = {
            check_a.name(): True,
            check_b.name(): True,
        }

        # Initial state: use_hot = False
        mock_local_cpu_backend.use_hot = False

        # Step 1: Both backends healthy
        connector_a.set_ping_error_code(0)
        connector_b.set_ping_error_code(0)
        monitor._run_all_checks()
        assert mock_local_cpu_backend.use_hot is False
        assert len(monitor._bypassed_backends) == 0
        assert monitor._original_hot_cache is None

        # Step 2: Backend A fails
        connector_a.set_ping_error_code(1)
        monitor._run_all_checks()
        assert mock_local_cpu_backend.use_hot is True
        assert "RemoteBackend_A" in monitor._bypassed_backends
        assert monitor._original_hot_cache is False  # Saved original value

        # Step 3: Backend B also fails
        connector_b.set_ping_error_code(1)
        monitor._run_all_checks()
        assert mock_local_cpu_backend.use_hot is True
        assert "RemoteBackend_A" in monitor._bypassed_backends
        assert "RemoteBackend_B" in monitor._bypassed_backends
        # CRITICAL: original_hot_cache should STILL be False, not True
        assert monitor._original_hot_cache is False

        mock_local_cpu_backend.clear.reset_mock()

        # Step 4: Backend A recovers
        connector_a.set_ping_error_code(0)
        monitor._run_all_checks()
        assert "RemoteBackend_A" not in monitor._bypassed_backends
        assert "RemoteBackend_B" in monitor._bypassed_backends
        # CRITICAL: use_hot should STAY True because B is still down
        assert mock_local_cpu_backend.use_hot is True
        # clear() should NOT have been called yet
        mock_local_cpu_backend.clear.assert_not_called()
        # original_hot_cache should still be preserved
        assert monitor._original_hot_cache is False

        # Step 5: Backend B recovers
        connector_b.set_ping_error_code(0)
        monitor._run_all_checks()
        assert len(monitor._bypassed_backends) == 0
        # CRITICAL: Now use_hot should be restored to False
        assert mock_local_cpu_backend.use_hot is False
        # clear() should have been called now
        mock_local_cpu_backend.clear.assert_called_once()
        # original_hot_cache should be reset
        assert monitor._original_hot_cache is None


class TestRemoteBackendHealthCheckEdgeCases:
    """Test edge cases for RemoteBackendHealthCheck."""

    def test_health_check_with_asyncio(self, event_loop_thread, test_config):
        """Test check() properly uses asyncio for ping."""
        connector = ControllablePingConnector()
        backend = MagicMock(spec=RemoteBackend)
        backend.remote_url = "controllable://test:1234"
        backend.loop = event_loop_thread.loop
        backend.connection = connector
        backend.init_connection = MagicMock()
        backend.config = test_config
        backend.get_blocking_failed_count = 0

        check = RemoteBackendHealthCheck(backend=backend)
        check._backend_name = "RemoteBackend"

        connector.set_ping_error_code(0)
        assert check.check() is True

        connector.set_ping_error_code(42)
        assert check.check() is False

    def test_health_check_handles_ping_timeout(self, event_loop_thread):
        """Test health check handles ping timeout correctly."""

        class SlowPingConnector(ControllablePingConnector):
            async def ping(self) -> int:
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    pass
                return 0

        # Create config with short ping timeout
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.1,
            remote_url="blackhole://test",
            remote_serde="naive",
            lmcache_instance_id="test_ping_timeout",
            extra_config={
                "ping_timeout": 0.1,
                "ping_interval": 0.1,
                "fallback_policy": "local_cpu",
            },
        )

        backend = MagicMock(spec=RemoteBackend)
        backend.remote_url = "slow://test:1234"
        backend.loop = event_loop_thread.loop
        backend.connection = SlowPingConnector()
        backend.init_connection = MagicMock()
        backend.config = config
        backend.get_blocking_failed_count = 0

        check = RemoteBackendHealthCheck(backend=backend)
        check._backend_name = "RemoteBackend"

        assert check.check() is False

    def test_health_check_handles_none_connection(self, test_config):
        """Test health check handles None connection gracefully."""
        backend = MagicMock(spec=RemoteBackend)
        backend.remote_url = "none://test:1234"
        backend.connection = None
        backend.init_connection = MagicMock()
        backend.config = test_config

        check = RemoteBackendHealthCheck(backend=backend)

        assert check.check() is False


class TestFallbackWithCacheEngineOperations:
    """Test fallback behavior through CacheEngine operations (lookup/store/retrieve)."""

    def test_store_lookup_retrieve_during_fallback(
        self,
        mock_manager,
        mock_local_cpu_backend,
        mock_storage_manager,
        controllable_connector,
        event_loop_thread,
    ):
        """Test operations bypass RemoteBackend during fallback."""
        monitor = HealthMonitor(manager=mock_manager, ping_interval=0.1)

        # Start healthy
        controllable_connector.set_ping_error_code(0)
        monitor._run_all_checks()
        assert mock_local_cpu_backend.use_hot is False

        # Trigger fallback
        controllable_connector.set_ping_error_code(1)
        monitor._run_all_checks()

        # Verify fallback state
        mock_storage_manager.set_backend_bypass.assert_called_with(
            "RemoteBackend", True
        )
        assert mock_local_cpu_backend.use_hot is True
        mock_storage_manager.is_backend_bypassed.return_value = True
        assert mock_storage_manager.is_backend_bypassed("RemoteBackend") is True

        # Recovery
        controllable_connector.set_ping_error_code(0)
        monitor._run_all_checks()

        mock_storage_manager.set_backend_bypass.assert_called_with(
            "RemoteBackend", False
        )
        assert mock_local_cpu_backend.use_hot is False
        mock_local_cpu_backend.clear.assert_called_once()


class TestFallbackIntegrationWithRealStorageManager:
    """Integration tests using real StorageManager."""

    @pytest.fixture
    def real_storage_manager(self, test_metadata):
        # First Party
        from lmcache.v1.event_manager import EventManager
        from lmcache.v1.storage_backend.storage_manager import StorageManager

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=0.5,
            lmcache_instance_id="test_fallback_integration",
            extra_config={
                "ping_timeout": 1.0,
                "ping_interval": 0.1,
                "fallback_policy": "local_cpu",
            },
        )
        manager = StorageManager(
            config=config, metadata=test_metadata, event_manager=EventManager()
        )
        yield manager
        manager.close()

    def test_storage_manager_bypass_mode(self, real_storage_manager):
        """Test StorageManager bypass mode operations."""
        sm = real_storage_manager

        # Initially no backends bypassed
        assert sm.is_backend_bypassed("RemoteBackend") is False
        assert sm.is_backend_bypassed("LocalCPUBackend") is False

        # Test bypass toggle
        sm.set_backend_bypass("RemoteBackend", True)
        assert sm.is_backend_bypassed("RemoteBackend") is True

        sm.set_backend_bypass("RemoteBackend", False)
        assert sm.is_backend_bypassed("RemoteBackend") is False

    def test_get_active_backends_respects_bypass(self, real_storage_manager):
        """Test get_active_storage_backends skips bypassed backends."""
        sm = real_storage_manager
        # This test requires RemoteBackend to be configured
        assert "RemoteBackend" not in sm.storage_backends, (
            "Test requires no RemoteBackend"
        )

        # Verify bypass logic works with LocalCPUBackend
        assert "LocalCPUBackend" in sm.storage_backends
        sm.set_backend_bypass("LocalCPUBackend", True)
        backend_names = [name for name, _ in sm.get_active_storage_backends()]
        assert "LocalCPUBackend" not in backend_names
        sm.set_backend_bypass("LocalCPUBackend", False)

    def test_local_cpu_hot_cache_toggle(self, real_storage_manager):
        """Test LocalCPUBackend hot cache can be toggled."""
        assert real_storage_manager.local_cpu_backend is not None, (
            "Test requires LocalCPUBackend"
        )
        local_cpu = real_storage_manager.local_cpu_backend
        original = local_cpu.use_hot

        local_cpu.use_hot = True
        assert local_cpu.use_hot is True

        local_cpu.use_hot = False
        assert local_cpu.use_hot is False

        local_cpu.use_hot = original

    def test_contains_respects_bypass(self, real_storage_manager):
        """Test contains operation respects bypass mode."""
        sm = real_storage_manager
        test_key = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=hash("test"),
            dtype=torch.bfloat16,
        )

        # Test with LocalCPUBackend bypass
        assert "LocalCPUBackend" in sm.storage_backends, "Test requires LocalCPUBackend"
        sm.set_backend_bypass("LocalCPUBackend", True)
        result = sm.contains(test_key)
        assert result is None  # No other backend available
        sm.set_backend_bypass("LocalCPUBackend", False)
