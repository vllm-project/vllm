# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the health_monitor module."""

# Standard
from typing import List
from unittest.mock import MagicMock
import asyncio
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.health_monitor.base import HealthCheck, HealthMonitor
from lmcache.v1.health_monitor.checks.remote_backend_check import (
    RemoteBackendHealthCheck,
)

# ============================================================================
# Mock Classes
# ============================================================================


class SimpleHealthCheck(HealthCheck):
    """Configurable health check for testing."""

    def __init__(
        self, name: str = "SimpleCheck", healthy: bool = True, skip: bool = False
    ):
        self._name, self._healthy, self._skip = name, healthy, skip
        self.check_count = 0

    def name(self) -> str:
        return self._name

    def check(self) -> bool:
        self.check_count += 1
        return self._healthy

    def should_skip(self) -> bool:
        return self._skip

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    @classmethod
    def create(cls, manager) -> List[HealthCheck]:
        return [cls()]


class ExceptionHealthCheck(HealthCheck):
    def name(self) -> str:
        return "ExceptionHealthCheck"

    def check(self) -> bool:
        raise RuntimeError("Simulated failure")

    @classmethod
    def create(cls, manager) -> List[HealthCheck]:
        return [cls()]


class IrrecoverableHealthCheck(HealthCheck):
    """Health check that raises IrrecoverableException."""

    def name(self) -> str:
        return "IrrecoverableHealthCheck"

    def check(self) -> bool:
        raise IrrecoverableException("Simulated irrecoverable failure")

    @classmethod
    def create(cls, manager) -> List[HealthCheck]:
        return [cls()]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_manager():
    manager = MagicMock()
    manager.lmcache_engine = MagicMock()
    manager.lmcache_engine.config.extra_config = None
    manager.lmcache_engine.storage_manager = None
    return manager


@pytest.fixture
def monitor(mock_manager):
    return HealthMonitor(manager=mock_manager, ping_interval=0.1)


@pytest.fixture
def mock_backend():
    mock = MagicMock()
    mock.remote_url = "lm://localhost:1234"
    mock.loop = asyncio.new_event_loop()
    mock.connection = MagicMock()
    mock.connection.support_ping.return_value = True
    return mock


# ============================================================================
# Tests for HealthCheck Base Class
# ============================================================================


class TestHealthCheckBase:
    @pytest.mark.parametrize(
        "healthy,skip,expected_check,expected_skip",
        [
            (True, False, True, False),
            (False, False, False, False),
            (False, True, False, True),
        ],
    )
    def test_simple_health_check_variants(
        self, healthy, skip, expected_check, expected_skip
    ):
        check = SimpleHealthCheck(healthy=healthy, skip=skip)
        assert check.check() is expected_check
        assert check.should_skip() is expected_skip

    def test_toggle_health_state(self):
        check = SimpleHealthCheck(healthy=True)
        assert check.check() is True
        check.set_healthy(False)
        assert check.check() is False


# ============================================================================
# Tests for HealthMonitor
# ============================================================================


class TestHealthMonitor:
    def test_initialization(self, mock_manager):
        monitor = HealthMonitor(manager=mock_manager, ping_interval=10.0)
        assert monitor.is_healthy() is True
        assert monitor._ping_interval == 10.0

    @pytest.mark.parametrize(
        "checks,expected",
        [
            ([SimpleHealthCheck(healthy=True), SimpleHealthCheck(healthy=True)], True),
            (
                [SimpleHealthCheck(healthy=True), SimpleHealthCheck(healthy=False)],
                False,
            ),
            ([SimpleHealthCheck(healthy=False, skip=True)], True),  # skipped check
        ],
    )
    def test_run_all_checks(self, monitor, checks, expected):
        monitor._health_checks = checks
        assert monitor._run_all_checks() is expected

    def test_exception_treated_as_failure(self, monitor):
        monitor._health_checks = [ExceptionHealthCheck()]
        assert monitor._run_all_checks() is False

    def test_health_state_transitions(self, monitor):
        assert monitor.is_healthy() is True
        monitor._set_healthy(False)
        assert monitor.is_healthy() is False
        monitor._set_healthy(True)
        assert monitor.is_healthy() is True

    def test_start_stop_lifecycle(self, monitor):
        monitor._health_checks = [SimpleHealthCheck(healthy=True)]
        thread = monitor.start()
        assert thread is not None and thread.is_alive()
        time.sleep(0.2)
        monitor.stop()
        time.sleep(0.1)
        assert not thread.is_alive()

    @pytest.mark.parametrize(
        "checks", [[], [SimpleHealthCheck(healthy=False, skip=True)]]
    )
    def test_start_with_no_active_checks_returns_none(self, monitor, checks):
        monitor._health_checks = checks
        assert monitor.start() is None

    def test_monitor_detects_state_changes(self, monitor):
        toggle = SimpleHealthCheck(healthy=True)
        monitor._health_checks = [toggle]
        thread = monitor.start()
        assert thread is not None

        time.sleep(0.15)
        assert monitor.is_healthy() is True

        toggle.set_healthy(False)
        time.sleep(0.15)
        assert monitor.is_healthy() is False

        toggle.set_healthy(True)
        time.sleep(0.15)
        assert monitor.is_healthy() is True
        monitor.stop()

    def test_thread_safety(self, monitor):
        monitor._ping_interval = 0.05
        toggle = SimpleHealthCheck(healthy=True)
        monitor._health_checks = [toggle]
        results, stop_event = [], threading.Event()

        def reader():
            while not stop_event.is_set():
                results.append(monitor.is_healthy())
                time.sleep(0.01)

        monitor.start()
        reader_thread = threading.Thread(target=reader)
        reader_thread.start()

        for _ in range(3):
            toggle.set_healthy(False)
            time.sleep(0.03)
            toggle.set_healthy(True)
            time.sleep(0.03)

        stop_event.set()
        monitor.stop()
        reader_thread.join()
        assert len(results) > 0


# ============================================================================
# Tests for RemoteBackendHealthCheck
# ============================================================================


class TestRemoteBackendHealthCheck:
    def test_should_not_skip_when_connection_is_none(self, mock_backend):
        mock_backend.connection = None
        assert RemoteBackendHealthCheck(mock_backend).should_skip() is False

    def test_name_includes_url(self, mock_backend):
        assert "localhost:1234" in RemoteBackendHealthCheck(mock_backend).name()

    def test_create_with_remote_backend(self):
        # First Party
        from lmcache.v1.storage_backend.remote_backend import RemoteBackend

        mock_backend = MagicMock(spec=RemoteBackend)
        mock_backend.remote_url = "lm://localhost:1234"
        mock_backend.connection = MagicMock()
        mock_backend.config = MagicMock()
        mock_backend.config.extra_config = None

        mock_manager = MagicMock()
        mock_manager.lmcache_engine = MagicMock()
        mock_manager.lmcache_engine.storage_manager = MagicMock()
        mock_manager.lmcache_engine.storage_manager.storage_backends = {
            "RemoteBackend": mock_backend
        }

        checks = RemoteBackendHealthCheck.create(mock_manager)
        assert len(checks) == 1 and "localhost:1234" in checks[0].name()

    def test_create_no_remote_backend(self):
        mock_manager = MagicMock()
        mock_manager.lmcache_engine = MagicMock()
        mock_manager.lmcache_engine.storage_manager = MagicMock()
        mock_manager.lmcache_engine.storage_manager.storage_backends = {}
        assert len(RemoteBackendHealthCheck.create(mock_manager)) == 0


# ============================================================================
# Tests for IrrecoverableException Handling
# ============================================================================


class TestIrrecoverableException:
    def test_run_all_checks_propagates_irrecoverable_exception(self, monitor):
        """Test that _run_all_checks propagates IrrecoverableException."""
        monitor._health_checks = [IrrecoverableHealthCheck()]
        with pytest.raises(IrrecoverableException):
            monitor._run_all_checks()

    def test_run_all_checks_irrecoverable_before_other_checks(self, monitor):
        """Test that IrrecoverableException stops further checks."""
        irrecoverable_check = IrrecoverableHealthCheck()
        simple_check = SimpleHealthCheck(healthy=True)
        monitor._health_checks = [irrecoverable_check, simple_check]

        with pytest.raises(IrrecoverableException):
            monitor._run_all_checks()

        # simple_check should not have been called
        assert simple_check.check_count == 0

    def test_run_loop_stops_on_irrecoverable_exception(self, monitor):
        """Test that _run_loop exits when IrrecoverableException is raised."""
        monitor._health_checks = [IrrecoverableHealthCheck()]
        monitor._ping_interval = 0.05

        thread = monitor.start()
        assert thread is not None

        # Wait for the loop to encounter the exception and exit
        time.sleep(0.2)

        # The thread should have stopped due to the exception
        assert not thread.is_alive()
        # System should be marked unhealthy
        assert monitor.is_healthy() is False

    def test_run_loop_stops_immediately_on_irrecoverable(self, monitor):
        """Test that the monitor stops checking after IrrecoverableException."""
        call_count = 0

        class CountingIrrecoverableCheck(HealthCheck):
            def name(self) -> str:
                return "CountingIrrecoverableCheck"

            def check(self) -> bool:
                nonlocal call_count
                call_count += 1
                raise IrrecoverableException("Stop now")

            @classmethod
            def create(cls, manager) -> List[HealthCheck]:
                return [cls()]

        monitor._health_checks = [CountingIrrecoverableCheck()]
        monitor._ping_interval = 0.05

        thread = monitor.start()
        assert thread is not None

        # Wait enough time for multiple check cycles
        time.sleep(0.3)

        # The check should only be called once before loop exits
        assert call_count == 1
        assert not thread.is_alive()

    def test_regular_exception_does_not_stop_loop(self, monitor):
        """Test that regular exceptions don't stop the monitoring loop."""
        monitor._health_checks = [ExceptionHealthCheck()]
        monitor._ping_interval = 0.05

        thread = monitor.start()
        assert thread is not None

        # Wait for multiple check cycles
        time.sleep(0.2)

        # The thread should still be running (regular exceptions don't stop the loop)
        assert thread.is_alive()
        # System should be marked unhealthy
        assert monitor.is_healthy() is False

        monitor.stop()
