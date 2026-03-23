# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import AsyncMock, Mock, patch

import msgspec.msgpack
import pytest
import zmq

from vllm.config import FaultToleranceConfig, VllmConfig
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.client_sentinel import ClientSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def mock_vllm_config():
    """Create mock VllmConfig object"""
    config = Mock(spec=VllmConfig)
    config.parallel_config = Mock(
        data_parallel_index=0,
        data_parallel_size=2,
        data_parallel_size_local=2,
        local_engines_only=False,
    )
    config.fault_tolerance_config = FaultToleranceConfig(engine_recovery_timeout_sec=10)
    return config


@pytest.fixture
def mock_ft_addresses():
    """Create mock FaultToleranceZmqAddresses object"""
    addresses = Mock()
    addresses.engine_fault_socket_addr = "tcp://127.0.0.1:5557"
    addresses.fault_state_pub_socket_addr = "tcp://127.0.0.1:5558"
    return addresses


@pytest.fixture
def client_sentinel(mock_vllm_config, mock_ft_addresses):
    """Create ClientSentinel fixture with mocked sockets."""
    fault_receiver_socket = AsyncMock()
    fault_state_pub_socket = AsyncMock()

    with (
        patch(
            "vllm.v1.fault_tolerance.client_sentinel.make_zmq_socket",
            side_effect=[fault_receiver_socket, fault_state_pub_socket],
        ),
        patch(
            "vllm.v1.fault_tolerance.client_sentinel.asyncio.create_task"
        ) as mock_create_task,
    ):

        def _capture_task(coro):
            # ClientSentinel starts run() in __init__; close it in tests to avoid
            # "coroutine was never awaited" warnings when create_task is mocked.
            coro.close()
            return Mock()

        mock_create_task.side_effect = _capture_task
        shutdown_callback = AsyncMock()
        sentinel = ClientSentinel(
            vllm_config=mock_vllm_config,
            fault_tolerance_addresses=mock_ft_addresses,
            shutdown_callback=shutdown_callback,
        )

    sentinel.instance_shutdown_callback = shutdown_callback
    return sentinel


# -------------------------- Test Cases --------------------------
@pytest.mark.asyncio
async def test_client_sentinel_initialization(client_sentinel: ClientSentinel):
    """Test ClientSentinel initialization logic."""
    assert client_sentinel.engine_status_dict == {
        0: {"status": "healthy"},
        1: {"status": "healthy"},
    }
    assert client_sentinel.start_rank == 0
    assert client_sentinel.fault_receiver_socket is not None
    assert client_sentinel.fault_state_pub_socket is not None
    assert client_sentinel._shutdown_task is None


@pytest.mark.asyncio
async def test_monitor_and_report_on_fault(client_sentinel: ClientSentinel):
    """Fault should update status and publish fault-state report."""
    fault_info = FaultInfo(
        engine_id="0",
        type="EngineDeadError",
        message="dead",
        engine_status=EngineStatusType.DEAD,
    )
    client_sentinel.fault_receiver_socket.recv_multipart = AsyncMock(
        side_effect=[
            [b"", b"", msgspec.msgpack.encode(fault_info)],
            zmq.ZMQError(),
        ]
    )

    await client_sentinel.run()

    assert client_sentinel.engine_status_dict[0]["status"] == "dead"
    client_sentinel.fault_state_pub_socket.send_multipart.assert_awaited_once()

    sent_topic, sent_payload = (
        client_sentinel.fault_state_pub_socket.send_multipart.await_args.args[0]
    )
    assert sent_topic == b"vllm_fault"
    assert msgspec.msgpack.decode(sent_payload) == {
        "total_engines": 2,
        "engines": [{"id": 0, "status": "dead"}, {"id": 1, "status": "healthy"}],
    }


@pytest.mark.asyncio
async def test_shutdown(client_sentinel: ClientSentinel):
    """Test shutdown method."""
    with patch("vllm.v1.fault_tolerance.client_sentinel.close_sockets") as mock_close:
        client_sentinel.shutdown()

    mock_close.assert_called_once_with(
        [
            client_sentinel.fault_receiver_socket,
            client_sentinel.fault_state_pub_socket,
        ]
    )
    assert client_sentinel.sentinel_dead is True
