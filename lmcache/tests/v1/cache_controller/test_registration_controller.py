# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RegistrationController."""

# Standard
from unittest.mock import Mock, patch
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.controllers.registration_controller import (
    RegistrationController,
)
from lmcache.v1.cache_controller.message import (
    DeRegisterMsg,
    HealthMsg,
    HeartbeatMsg,
    QueryInstMsg,
    QueryWorkerInfoMsg,
    RegisterMsg,
)


@pytest.fixture
def reg_controller(mock_kv_controller, mock_cluster_executor):
    """Create a RegistrationController instance for testing."""
    controller = RegistrationController()
    controller.post_init(
        kv_controller=mock_kv_controller,
        cluster_executor=mock_cluster_executor,
    )
    return controller


class TestRegistrationControllerRegister:
    """Test RegistrationController register operations."""

    @pytest.mark.asyncio
    async def test_register_new_worker(self, reg_controller, mock_zmq_socket):
        """Test registering a new worker."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            msg = RegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url="tcp://127.0.0.1:6000",
            )

            await reg_controller.register(msg)

            # Verify worker is registered
            worker_ids = reg_controller.registry.get_worker_ids("test_instance")
            assert 0 in worker_ids

            # Verify socket is created
            worker_node = reg_controller.registry.get_worker("test_instance", 0)
            assert worker_node is not None
            assert worker_node.socket is not None
            assert worker_node.peer_init_url == "tcp://127.0.0.1:6000"

            # Verify instance mapping
            instance_node = reg_controller.registry.get_instance_by_ip("127.0.0.1")
            assert instance_node is not None
            assert instance_node.instance_id == "test_instance"

    @pytest.mark.asyncio
    async def test_register_without_peer_url(self, reg_controller, mock_zmq_socket):
        """Test registering a worker without peer_init_url."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            msg = RegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url=None,
            )

            await reg_controller.register(msg)

            # Verify worker is registered
            worker_node = reg_controller.registry.get_worker("test_instance", 0)
            assert worker_node is not None
            assert worker_node.socket is not None

            # Peer URL should not be registered
            assert worker_node.peer_init_url is None

    @pytest.mark.asyncio
    async def test_register_duplicate_worker(self, reg_controller, mock_zmq_socket):
        """Test duplicate registration is prevented."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            msg = RegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url="tcp://127.0.0.1:6000",
            )

            # Register once
            await reg_controller.register(msg)

            # Try to register again
            await reg_controller.register(msg)

            # Should still have only one entry
            worker_ids = reg_controller.registry.get_worker_ids("test_instance")
            assert len(worker_ids) == 1

    @pytest.mark.asyncio
    async def test_register_multiple_workers(self, reg_controller, mock_zmq_socket):
        """Test registering multiple workers for same instance."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            for worker_id in range(3):
                msg = RegisterMsg(
                    instance_id="test_instance",
                    worker_id=worker_id,
                    ip="127.0.0.1",
                    port=5000 + worker_id,
                    peer_init_url=f"tcp://127.0.0.1:{6000 + worker_id}",
                )
                await reg_controller.register(msg)

            # Verify all workers are registered
            worker_ids = reg_controller.registry.get_worker_ids("test_instance")
            assert len(worker_ids) == 3

            # Verify workers are sorted
            assert worker_ids == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_register_worker_info_timestamps(
        self, reg_controller, mock_zmq_socket
    ):
        """Test worker info contains correct timestamps."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            before_time = time.time()

            msg = RegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url="tcp://127.0.0.1:6000",
            )

            await reg_controller.register(msg)

            after_time = time.time()

            worker_node = reg_controller.registry.get_worker("test_instance", 0)
            assert worker_node is not None

            assert before_time <= worker_node.registration_time <= after_time
            assert before_time <= worker_node.last_heartbeat_time <= after_time


class TestRegistrationControllerDeregister:
    """Test RegistrationController deregister operations."""

    @pytest.mark.asyncio
    async def test_deregister_worker(self, reg_controller, mock_zmq_socket):
        """Test deregistering a worker."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            # First register
            register_msg = RegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url="tcp://127.0.0.1:6000",
            )
            await reg_controller.register(register_msg)

            # Then deregister
            deregister_msg = DeRegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
            )
            await reg_controller.deregister(deregister_msg)

            # Verify cleanup
            worker_node = reg_controller.registry.get_worker("test_instance", 0)
            assert worker_node is None

            instance_node = reg_controller.registry.get_instance("test_instance")
            assert instance_node is None

            # Note: KV cleanup is now handled directly by the registry's
            # deregister_worker method

    @pytest.mark.asyncio
    async def test_deregister_nonexistent_worker(self, reg_controller):
        """Test deregistering a non-existent worker."""
        deregister_msg = DeRegisterMsg(
            instance_id="nonexistent_instance",
            worker_id=0,
            ip="127.0.0.1",
            port=5000,
        )

        # Should not raise an error
        await reg_controller.deregister(deregister_msg)

    @pytest.mark.asyncio
    async def test_deregister_partial_workers(self, reg_controller, mock_zmq_socket):
        """Test deregistering one of multiple workers."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            # Register two workers
            for worker_id in range(2):
                msg = RegisterMsg(
                    instance_id="test_instance",
                    worker_id=worker_id,
                    ip="127.0.0.1",
                    port=5000 + worker_id,
                    peer_init_url=f"tcp://127.0.0.1:{6000 + worker_id}",
                )
                await reg_controller.register(msg)

            # Deregister only worker 0
            deregister_msg = DeRegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
            )
            await reg_controller.deregister(deregister_msg)

            # Worker 1 should still be registered
            worker_ids = reg_controller.registry.get_worker_ids("test_instance")
            assert worker_ids == [1]

            worker_node = reg_controller.registry.get_worker("test_instance", 1)
            assert worker_node is not None
            assert worker_node.socket is not None


class TestRegistrationControllerGetters:
    """Test RegistrationController getter methods."""

    @pytest.mark.asyncio
    async def test_get_socket(self, reg_controller, mock_zmq_socket):
        """Test getting socket for a worker."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            msg = RegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url="tcp://127.0.0.1:6000",
            )
            await reg_controller.register(msg)

            socket = reg_controller.get_socket("test_instance", 0)

            assert socket is not None
            assert socket == mock_zmq_socket

    def test_get_socket_nonexistent(self, reg_controller):
        """Test getting socket for non-existent worker."""
        socket = reg_controller.get_socket("nonexistent", 0)

        assert socket is None

    @pytest.mark.asyncio
    async def test_get_peer_init_url(self, reg_controller, mock_zmq_socket):
        """Test getting peer init URL for a worker."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            msg = RegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url="tcp://127.0.0.1:6000",
            )
            await reg_controller.register(msg)

            url = reg_controller.get_peer_init_url("test_instance", 0)

            assert url == "tcp://127.0.0.1:6000"

    def test_get_peer_init_url_nonexistent(self, reg_controller):
        """Test getting peer init URL for non-existent worker."""
        url = reg_controller.get_peer_init_url("nonexistent", 0)

        assert url is None

    @pytest.mark.asyncio
    async def test_get_workers(self, reg_controller, mock_zmq_socket):
        """Test getting workers for an instance."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            for worker_id in [2, 0, 1]:  # Register out of order
                msg = RegisterMsg(
                    instance_id="test_instance",
                    worker_id=worker_id,
                    ip="127.0.0.1",
                    port=5000 + worker_id,
                    peer_init_url=f"tcp://127.0.0.1:{6000 + worker_id}",
                )
                await reg_controller.register(msg)

            workers = reg_controller.get_workers("test_instance")

            # Should be sorted
            assert workers == [0, 1, 2]

    def test_get_workers_nonexistent(self, reg_controller):
        """Test getting workers for non-existent instance."""
        workers = reg_controller.get_workers("nonexistent")

        assert workers == []


class TestRegistrationControllerInstanceMapping:
    """Test RegistrationController instance ID mapping."""

    @pytest.mark.asyncio
    async def test_get_instance_id(self, reg_controller, mock_zmq_socket):
        """Test getting instance ID from IP."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            msg = RegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url="tcp://127.0.0.1:6000",
            )
            await reg_controller.register(msg)

            query_msg = QueryInstMsg(
                event_id="event_123",
                ip="127.0.0.1",
            )

            result = await reg_controller.get_instance_id(query_msg)

            assert result.event_id == "event_123"
            assert result.instance_id == "test_instance"

    @pytest.mark.asyncio
    async def test_get_instance_id_not_found(self, reg_controller):
        """Test getting instance ID for unregistered IP."""
        query_msg = QueryInstMsg(
            event_id="event_123",
            ip="192.168.1.1",
        )

        result = await reg_controller.get_instance_id(query_msg)

        assert result.event_id == "event_123"
        assert result.instance_id is None


class TestRegistrationControllerHeartbeat:
    """Test RegistrationController heartbeat operations."""

    @pytest.mark.asyncio
    async def test_heartbeat_existing_worker(self, reg_controller, mock_zmq_socket):
        """Test heartbeat updates timestamp for existing worker."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            # Register worker
            register_msg = RegisterMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url="tcp://127.0.0.1:6000",
            )
            await reg_controller.register(register_msg)

            worker_node = reg_controller.registry.get_worker("test_instance", 0)
            assert worker_node is not None
            original_time = worker_node.last_heartbeat_time

            # Wait a bit
            time.sleep(0.1)

            # Send heartbeat
            heartbeat_msg = HeartbeatMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url="tcp://127.0.0.1:6000",
            )
            await reg_controller.heartbeat(heartbeat_msg)

            # Verify timestamp was updated
            worker_node = reg_controller.registry.get_worker("test_instance", 0)
            assert worker_node is not None
            new_time = worker_node.last_heartbeat_time
            assert new_time > original_time

    @pytest.mark.asyncio
    async def test_heartbeat_unregistered_worker(self, reg_controller, mock_zmq_socket):
        """Test heartbeat triggers re-registration for unregistered worker."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            heartbeat_msg = HeartbeatMsg(
                instance_id="test_instance",
                worker_id=0,
                ip="127.0.0.1",
                port=5000,
                peer_init_url="tcp://127.0.0.1:6000",
            )

            await reg_controller.heartbeat(heartbeat_msg)

            # Worker should now be registered
            worker_node = reg_controller.registry.get_worker("test_instance", 0)
            assert worker_node is not None


class TestRegistrationControllerQueryWorkerInfo:
    """Test RegistrationController query worker info operations."""

    @pytest.mark.asyncio
    async def test_query_worker_info(self, reg_controller, mock_zmq_socket):
        """Test querying worker information."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            # Register workers
            for worker_id in range(3):
                msg = RegisterMsg(
                    instance_id="test_instance",
                    worker_id=worker_id,
                    ip="127.0.0.1",
                    port=5000 + worker_id,
                    peer_init_url=f"tcp://127.0.0.1:{6000 + worker_id}",
                )
                await reg_controller.register(msg)

            # Query all workers
            query_msg = QueryWorkerInfoMsg(
                event_id="event_123",
                instance_id="test_instance",
                worker_ids=None,
            )

            result = await reg_controller.query_worker_info(query_msg)

            assert result.event_id == "event_123"
            assert len(result.worker_infos) == 3
            assert result.worker_infos[0].worker_id == 0
            assert result.worker_infos[1].worker_id == 1
            assert result.worker_infos[2].worker_id == 2

    @pytest.mark.asyncio
    async def test_query_worker_info_specific_workers(
        self, reg_controller, mock_zmq_socket
    ):
        """Test querying specific workers."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            # Register workers
            for worker_id in range(3):
                msg = RegisterMsg(
                    instance_id="test_instance",
                    worker_id=worker_id,
                    ip="127.0.0.1",
                    port=5000 + worker_id,
                    peer_init_url=f"tcp://127.0.0.1:{6000 + worker_id}",
                )
                await reg_controller.register(msg)

            # Query specific workers
            query_msg = QueryWorkerInfoMsg(
                event_id="event_123",
                instance_id="test_instance",
                worker_ids=[0, 2],
            )

            result = await reg_controller.query_worker_info(query_msg)

            assert len(result.worker_infos) == 2
            assert result.worker_infos[0].worker_id == 0
            assert result.worker_infos[1].worker_id == 2

    @pytest.mark.asyncio
    async def test_query_worker_info_nonexistent_instance(self, reg_controller):
        """Test querying worker info for non-existent instance."""
        query_msg = QueryWorkerInfoMsg(
            event_id="event_123",
            instance_id="nonexistent",
            worker_ids=None,
        )

        result = await reg_controller.query_worker_info(query_msg)

        assert result.event_id == "event_123"
        assert len(result.worker_infos) == 0

    @pytest.mark.asyncio
    async def test_query_worker_info_empty_list(self, reg_controller, mock_zmq_socket):
        """Test querying with empty worker_ids list returns all workers."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_get_socket:
            mock_get_socket.return_value = mock_zmq_socket

            # Register workers
            for worker_id in range(2):
                msg = RegisterMsg(
                    instance_id="test_instance",
                    worker_id=worker_id,
                    ip="127.0.0.1",
                    port=5000 + worker_id,
                    peer_init_url=f"tcp://127.0.0.1:{6000 + worker_id}",
                )
                await reg_controller.register(msg)

            # Query with empty list
            query_msg = QueryWorkerInfoMsg(
                event_id="event_123",
                instance_id="test_instance",
                worker_ids=[],
            )

            result = await reg_controller.query_worker_info(query_msg)

            # Should return all workers
            assert len(result.worker_infos) == 2


class TestRegistrationControllerHealth:
    """Test RegistrationController health operations."""

    @pytest.mark.asyncio
    async def test_health_delegates_to_executor(self, reg_controller):
        """Test health check delegates to cluster executor."""
        msg = HealthMsg(
            event_id="event_123",
            instance_id="test_instance",
        )

        await reg_controller.health(msg)

        reg_controller.cluster_executor.execute.assert_called_once_with("health", msg)


class TestRegistrationControllerMetrics:
    """Test RegistrationController metrics setup."""

    @patch(
        "lmcache.v1.cache_controller.controllers.registration_controller.PrometheusLogger"
    )
    def test_metrics_setup(self, mock_prometheus_logger):
        """Test metrics are properly set up."""
        mock_logger_instance = Mock()
        mock_prometheus_logger.GetInstanceOrNone.return_value = mock_logger_instance

        _controller = RegistrationController()

        # Verify metrics function was set
        mock_logger_instance.registered_workers_count.set_function.assert_called_once()

    @patch(
        "lmcache.v1.cache_controller.controllers.registration_controller.PrometheusLogger"
    )
    def test_metrics_setup_no_logger(self, mock_prometheus_logger):
        """Test metrics setup when no logger is available."""
        mock_prometheus_logger.GetInstanceOrNone.return_value = None

        # Should not raise an error
        controller = RegistrationController()

        assert controller is not None
