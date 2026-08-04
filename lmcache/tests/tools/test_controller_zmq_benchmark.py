#!/usr/bin/env python3
"""
Tests for LMCache Controller ZMQ Benchmark Tool
"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.logging import init_logger
from lmcache.tools.controller_benchmark.benchmark import ZMQControllerBenchmark
from lmcache.tools.controller_benchmark.config import ZMQBenchmarkConfig
from lmcache.tools.controller_benchmark.handlers import OPERATION_HANDLERS
from lmcache.tools.controller_benchmark.handlers.admit import AdmitHandler
from lmcache.tools.controller_benchmark.handlers.base import (
    OperationHandler,
    SocketType,
)
from lmcache.tools.controller_benchmark.handlers.deregister import DeregisterHandler
from lmcache.tools.controller_benchmark.handlers.evict import EvictHandler
from lmcache.tools.controller_benchmark.handlers.heartbeat import HeartbeatHandler
from lmcache.tools.controller_benchmark.handlers.p2p_lookup import P2PLookupHandler
from lmcache.tools.controller_benchmark.handlers.register import RegisterHandler
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    BatchedP2PLookupMsg,
    DeRegisterMsg,
    HeartbeatMsg,
    OpType,
    RegisterMsg,
)

logger = init_logger(__name__)


class TestZMQBenchmarkConfig:
    """Test ZMQBenchmarkConfig class"""

    def test_default_operations(self):
        """Test default operation distribution"""
        config = ZMQBenchmarkConfig(
            controller_pull_url="localhost:8100",
            controller_reply_url=None,
            duration=60,
            batch_size=50,
            num_instances=10,
            num_workers=5,
            num_locations=3,
            num_keys=1000,
        )

        assert "admit" in config.operations
        assert "evict" in config.operations
        assert "heartbeat" in config.operations
        assert abs(sum(config.operations.values()) - 100.0) < 0.01

    def test_custom_operations(self):
        """Test custom operation distribution"""
        config = ZMQBenchmarkConfig(
            controller_pull_url="localhost:8100",
            controller_reply_url="localhost:8101",
            duration=60,
            batch_size=50,
            num_instances=10,
            num_workers=5,
            num_locations=3,
            num_keys=1000,
            operations={"admit": 80.0, "evict": 20.0},
        )

        assert config.operations["admit"] == 80.0
        assert config.operations["evict"] == 20.0
        assert config.controller_pull_url == "localhost:8100"
        assert config.controller_reply_url == "localhost:8101"
        assert config.duration == 60
        assert config.batch_size == 50
        assert config.num_instances == 10
        assert config.num_workers == 5
        assert config.num_locations == 3
        assert config.num_keys == 1000

    def test_invalid_operation_percentage(self):
        """Test invalid operation percentage sum"""
        with pytest.raises(ValueError, match="must sum to 100"):
            ZMQBenchmarkConfig(
                controller_pull_url="localhost:8100",
                controller_reply_url=None,
                duration=60,
                batch_size=50,
                num_instances=10,
                num_workers=5,
                num_locations=3,
                num_keys=1000,
                operations={"admit": 50, "evict": 60},  # Sum > 100
            )


class TestOperationHandlers:
    """Test operation handlers (Strategy Pattern)"""

    @pytest.fixture
    def controller_zmq_benchmark(self):
        """Create a benchmark instance for testing"""
        config = ZMQBenchmarkConfig(
            controller_pull_url="localhost:8100",
            controller_reply_url="localhost:8101",
            duration=1,
            batch_size=10,
            num_instances=2,
            num_workers=2,
            num_locations=2,
            num_keys=10,
            num_hashes=5,
            register_first=False,
        )
        return ZMQControllerBenchmark(config)

    @pytest.fixture
    def test_data(self, controller_zmq_benchmark):
        """Generate test data"""
        return controller_zmq_benchmark.generate_test_data()

    def test_admit_handler(self, controller_zmq_benchmark, test_data):
        """Test AdmitHandler functionality"""
        handler = AdmitHandler()
        msg = handler.create_message(controller_zmq_benchmark, test_data)

        assert isinstance(msg, BatchedKVOperationMsg)
        assert len(msg.operations) == controller_zmq_benchmark.config.batch_size
        assert all(op.op_type == OpType.ADMIT for op in msg.operations)
        assert (
            handler.get_message_count(controller_zmq_benchmark)
            == controller_zmq_benchmark.config.batch_size
        )
        assert handler.socket_type == SocketType.PUSH

    def test_evict_handler(self, controller_zmq_benchmark, test_data):
        """Test EvictHandler functionality"""
        handler = EvictHandler()
        msg = handler.create_message(controller_zmq_benchmark, test_data)

        assert isinstance(msg, BatchedKVOperationMsg)
        assert len(msg.operations) == controller_zmq_benchmark.config.batch_size
        assert all(op.op_type == OpType.EVICT for op in msg.operations)
        assert (
            handler.get_message_count(controller_zmq_benchmark)
            == controller_zmq_benchmark.config.batch_size
        )
        assert handler.socket_type == SocketType.PUSH

    def test_heartbeat_handler(self, controller_zmq_benchmark, test_data):
        """Test HeartbeatHandler functionality"""
        handler = HeartbeatHandler()
        msg = handler.create_message(controller_zmq_benchmark, test_data)

        assert isinstance(msg, HeartbeatMsg)
        assert handler.get_message_count(controller_zmq_benchmark) == 1
        assert handler.socket_type == SocketType.HEARTBEAT

    def test_register_handler(self, controller_zmq_benchmark, test_data):
        """Test RegisterHandler functionality"""
        handler = RegisterHandler()
        msg = handler.create_message(controller_zmq_benchmark, test_data)

        assert isinstance(msg, RegisterMsg)
        assert handler.get_message_count(controller_zmq_benchmark) == 1
        assert handler.socket_type == SocketType.DEALER

    def test_deregister_handler(self, controller_zmq_benchmark, test_data):
        """Test DeregisterHandler functionality"""
        handler = DeregisterHandler()
        msg = handler.create_message(controller_zmq_benchmark, test_data)

        assert isinstance(msg, DeRegisterMsg)
        assert handler.get_message_count(controller_zmq_benchmark) == 1
        assert handler.socket_type == SocketType.PUSH

    def test_p2p_lookup_handler(self, controller_zmq_benchmark, test_data):
        """Test P2PLookupHandler functionality"""
        handler = P2PLookupHandler()
        msg = handler.create_message(controller_zmq_benchmark, test_data)

        assert isinstance(msg, BatchedP2PLookupMsg)
        assert len(msg.hashes) == controller_zmq_benchmark.config.num_hashes
        assert (
            handler.get_message_count(controller_zmq_benchmark)
            == controller_zmq_benchmark.config.num_hashes
        )
        assert handler.socket_type == SocketType.DEALER

    def test_operation_handler_registry(self):
        """Test operation handler registry"""
        assert "admit" in OPERATION_HANDLERS
        assert "evict" in OPERATION_HANDLERS
        assert "heartbeat" in OPERATION_HANDLERS
        assert "register" in OPERATION_HANDLERS
        assert "deregister" in OPERATION_HANDLERS
        assert "p2p_lookup" in OPERATION_HANDLERS

        for handler in OPERATION_HANDLERS.values():
            assert isinstance(handler, OperationHandler)


class TestZMQControllerBenchmark:
    """Test ZMQControllerBenchmark class"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ZMQBenchmarkConfig(
            controller_pull_url="localhost:8100",
            controller_reply_url="localhost:8101",
            duration=1,
            batch_size=10,
            num_instances=2,
            num_workers=2,
            num_locations=2,
            num_keys=10,
            num_hashes=5,
            register_first=False,
        )

    @pytest.fixture
    def zmq_benchmark(self, config):
        """Create benchmark instance"""
        return ZMQControllerBenchmark(config)

    def test_generate_test_data(self, zmq_benchmark):
        """Test test data generation"""
        test_data = zmq_benchmark.generate_test_data()

        assert hasattr(test_data, "instances")
        assert hasattr(test_data, "workers")
        assert hasattr(test_data, "locations")
        assert hasattr(test_data, "keys")

        assert len(test_data.instances) == zmq_benchmark.config.num_instances
        assert len(test_data.workers) == zmq_benchmark.config.num_workers
        assert len(test_data.locations) == zmq_benchmark.config.num_locations
        assert len(test_data.keys) == zmq_benchmark.config.num_keys

    def test_sequence_number_increment(self, zmq_benchmark):
        """Test sequence number monotonic increment"""
        instance_id = "instance_0"
        worker_id = 0
        location = "location_0"

        seq1 = zmq_benchmark.get_next_sequence_number(instance_id, worker_id, location)
        seq2 = zmq_benchmark.get_next_sequence_number(instance_id, worker_id, location)
        seq3 = zmq_benchmark.get_next_sequence_number(instance_id, worker_id, location)

        assert seq1 == 0
        assert seq2 == 1
        assert seq3 == 2

    def test_sequence_number_isolation(self, zmq_benchmark):
        """Test sequence number isolation between different keys"""
        instance1, worker1, loc1 = "instance_0", 0, "location_0"
        instance2, worker2, loc2 = "instance_1", 1, "location_1"

        seq1a = zmq_benchmark.get_next_sequence_number(instance1, worker1, loc1)
        seq2a = zmq_benchmark.get_next_sequence_number(instance2, worker2, loc2)
        seq1b = zmq_benchmark.get_next_sequence_number(instance1, worker1, loc1)
        seq2b = zmq_benchmark.get_next_sequence_number(instance2, worker2, loc2)

        assert seq1a == 0
        assert seq2a == 0
        assert seq1b == 1
        assert seq2b == 1

    @pytest.mark.asyncio
    async def test_send_messages_success(self, zmq_benchmark):
        """Test successful message sending"""
        # Mock the socket
        mock_socket = AsyncMock()
        zmq_benchmark.push_socket = mock_socket

        test_data = zmq_benchmark.generate_test_data()
        handler = RegisterHandler()
        msg = handler.create_message(zmq_benchmark, test_data)

        latency = await zmq_benchmark.send_messages([msg])

        assert isinstance(latency, float)
        assert latency >= 0
        mock_socket.send_multipart.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_messages_timeout(self, zmq_benchmark):
        """Test message sending timeout"""
        # Mock the socket to raise timeout
        mock_socket = AsyncMock()
        mock_socket.send_multipart.side_effect = Exception("Send timeout")
        zmq_benchmark.push_socket = mock_socket

        test_data = zmq_benchmark.generate_test_data()
        handler = RegisterHandler()
        msg = handler.create_message(zmq_benchmark, test_data)

        with pytest.raises(Exception, match="Send timeout"):
            await zmq_benchmark.send_messages([msg])

    @pytest.mark.asyncio
    async def test_send_request_success(self, zmq_benchmark):
        """Test successful request sending via DEALER socket"""
        # Mock the socket
        mock_socket = AsyncMock()
        mock_socket.send_multipart = AsyncMock()
        # DEALER receives: [empty_frame, payload]
        mock_socket.recv_multipart = AsyncMock(return_value=[b"", b"response"])
        zmq_benchmark.req_socket = mock_socket

        test_data = zmq_benchmark.generate_test_data()
        handler = P2PLookupHandler()
        msg = handler.create_message(zmq_benchmark, test_data)

        latency, response = await zmq_benchmark.send_request(msg)

        assert isinstance(latency, float)
        assert latency >= 0
        assert response == b"response"
        mock_socket.send_multipart.assert_called_once()
        mock_socket.recv_multipart.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_request_timeout(self, zmq_benchmark):
        """Test request sending timeout via DEALER socket"""
        # Mock the socket to raise timeout
        mock_socket = AsyncMock()
        mock_socket.send_multipart = AsyncMock()
        mock_socket.recv_multipart = AsyncMock(side_effect=Exception("Request timeout"))
        zmq_benchmark.req_socket = mock_socket

        test_data = zmq_benchmark.generate_test_data()
        handler = P2PLookupHandler()
        msg = handler.create_message(zmq_benchmark, test_data)

        with pytest.raises(Exception, match="Request timeout"):
            await zmq_benchmark.send_request(msg)

    @pytest.mark.asyncio
    async def test_execute_operation_success(self, zmq_benchmark):
        """Test successful operation execution"""
        # Mock send_messages to avoid actual socket operations
        with patch.object(
            zmq_benchmark, "send_messages", AsyncMock(return_value=0.001)
        ):
            test_data = zmq_benchmark.generate_test_data()

            (
                msg_count,
                req_count,
                latency,
                error,
            ) = await zmq_benchmark._execute_operation("admit", test_data)

            assert msg_count == zmq_benchmark.config.batch_size
            assert req_count == 1
            assert latency > 0
            assert error is None

    @pytest.mark.asyncio
    async def test_execute_operation_unknown(self, zmq_benchmark):
        """Test unknown operation execution"""
        test_data = zmq_benchmark.generate_test_data()

        msg_count, req_count, latency, error = await zmq_benchmark._execute_operation(
            "unknown_op", test_data
        )

        assert msg_count == 0
        assert req_count == 0
        assert latency == 0.0
        assert isinstance(error, ValueError)

    def test_build_operation_distribution(self, zmq_benchmark):
        """Test operation distribution building"""
        operations = zmq_benchmark._build_operation_distribution()

        assert len(operations) > 0
        assert all(op in zmq_benchmark.config.operations for op in operations)

        # Verify distribution roughly matches percentages
        op_counts = {}
        for op in operations:
            op_counts[op] = op_counts.get(op, 0) + 1

        total_ops = len(operations)
        for op_name, percentage in zmq_benchmark.config.operations.items():
            expected_count = int(total_ops * percentage / 100)
            actual_count = op_counts.get(op_name, 0)
            # Allow some tolerance for random distribution
            assert abs(actual_count - expected_count) <= 2


class TestIntegration:
    """Integration tests for the ZMQ benchmark tool"""

    def test_operation_distribution(self):
        """Test operation distribution calculation"""
        config = ZMQBenchmarkConfig(
            controller_pull_url="localhost:8100",
            controller_reply_url=None,
            duration=60,
            batch_size=50,
            num_instances=10,
            num_workers=5,
            num_locations=3,
            num_keys=1000,
            operations={"admit": 50, "evict": 30, "heartbeat": 20},
        )

        zmq_benchmark = ZMQControllerBenchmark(config)
        # Verify benchmark was created with correct config
        assert zmq_benchmark.config.batch_size == 50

        # Build operation distribution
        operations = []
        total_ops = 1000
        for op_name, percentage in config.operations.items():
            count = int(total_ops * percentage / 100)
            operations.extend([op_name] * count)

        # Verify distribution
        assert len(operations) == total_ops

        op_counts: Dict[str, int] = {}
        for op in operations:
            op_counts[op] = op_counts.get(op, 0) + 1

        assert op_counts["admit"] == 500
        assert op_counts["evict"] == 300
        assert op_counts["heartbeat"] == 200

    @pytest.mark.asyncio
    async def test_benchmark_setup_and_cleanup(self):
        """Test benchmark setup and cleanup"""
        config = ZMQBenchmarkConfig(
            controller_pull_url="localhost:8100",
            controller_reply_url="localhost:8101",
            duration=1,
            batch_size=10,
            num_instances=2,
            num_workers=2,
            num_locations=2,
            num_keys=10,
            register_first=False,
        )

        zmq_benchmark = ZMQControllerBenchmark(config)

        # Mock ZMQ context and sockets
        with patch(
            "lmcache.tools.controller_benchmark.benchmark.get_zmq_context"
        ) as mock_context:
            mock_ctx = MagicMock()
            mock_context.return_value = mock_ctx

            mock_socket = MagicMock()
            mock_ctx.socket.return_value = mock_socket

            await zmq_benchmark.setup()

            # Verify setup was called
            mock_context.assert_called_once()

            # Test cleanup
            zmq_benchmark.cleanup()
            # Verify sockets were closed
            assert mock_socket.close.called

    def test_results_structure(self):
        """Test results structure initialization"""
        config = ZMQBenchmarkConfig(
            controller_pull_url="localhost:8100",
            controller_reply_url="localhost:8101",
            duration=1,
            batch_size=10,
            num_instances=2,
            num_workers=2,
            num_locations=2,
            num_keys=10,
            register_first=False,
        )

        zmq_benchmark = ZMQControllerBenchmark(config)
        results = zmq_benchmark.results

        assert hasattr(results, "operations")
        assert hasattr(results, "memory_usage")
        assert hasattr(results, "total_messages")
        assert hasattr(results, "total_requests")
        assert hasattr(results, "overall_qps")
        assert hasattr(results, "overall_rps")

        assert isinstance(results.operations, dict)
        assert isinstance(results.memory_usage, list)
        assert results.total_messages == 0
        assert results.total_requests == 0

    def test_normalize_heartbeat_url_localhost_connection(self):
        """Test heartbeat URL normalization when connecting to localhost"""
        config = ZMQBenchmarkConfig(
            controller_pull_url="127.0.0.1:8100",
            controller_reply_url="127.0.0.1:8101",
            duration=1,
            batch_size=10,
            num_instances=2,
            num_workers=2,
            num_locations=2,
            num_keys=10,
            register_first=False,
        )

        zmq_benchmark = ZMQControllerBenchmark(config)

        # When connecting locally, remote IPs should be replaced with 127.0.0.1
        assert (
            zmq_benchmark._normalize_heartbeat_url("10.0.0.5:7557") == "127.0.0.1:7557"
        )
        assert (
            zmq_benchmark._normalize_heartbeat_url("192.168.1.100:7557")
            == "127.0.0.1:7557"
        )
        # localhost should remain unchanged
        assert (
            zmq_benchmark._normalize_heartbeat_url("127.0.0.1:7557") == "127.0.0.1:7557"
        )
        assert (
            zmq_benchmark._normalize_heartbeat_url("localhost:7557") == "localhost:7557"
        )

    def test_normalize_heartbeat_url_remote_connection(self):
        """Test heartbeat URL normalization when connecting to remote host"""
        config = ZMQBenchmarkConfig(
            controller_pull_url="10.0.0.5:8100",
            controller_reply_url="10.0.0.5:8101",
            duration=1,
            batch_size=10,
            num_instances=2,
            num_workers=2,
            num_locations=2,
            num_keys=10,
            register_first=False,
        )

        zmq_benchmark = ZMQControllerBenchmark(config)

        # When connecting to remote, should keep the original IP
        assert (
            zmq_benchmark._normalize_heartbeat_url("10.0.0.5:7557") == "10.0.0.5:7557"
        )
        assert (
            zmq_benchmark._normalize_heartbeat_url("192.168.1.100:7557")
            == "192.168.1.100:7557"
        )
