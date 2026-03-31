# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import msgspec.msgpack
import pytest
import zmq

from vllm.config import FaultToleranceConfig, VllmConfig
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.client_sentinel import ClientSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo, FaultToleranceRequest

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
    addresses.ft_request_addresses = ["tcp://127.0.0.1:5555"]
    addresses.ft_result_addresses = ["tcp://127.0.0.1:5556"]
    addresses.engine_fault_socket_addr = "tcp://127.0.0.1:5557"
    addresses.fault_state_pub_socket_addr = "tcp://127.0.0.1:5558"
    return addresses


@pytest.fixture
def mock_call_utility_async():
    """Create mock call_utility_async function"""
    return AsyncMock(
        return_value={"request_id": "request_id", "success": True, "reason": None}
    )


@pytest.fixture
def client_sentinel(mock_vllm_config, mock_ft_addresses, mock_call_utility_async):
    """Fixed ClientSentinel fixture (mock Poller)"""
    # 1. Mock Poller class and return mock Poller object
    mock_poller = Mock()
    mock_poller.register = Mock()
    mock_poller.poll = AsyncMock(return_value=[])  # Return empty events by default
    mock_poller_class = Mock(return_value=mock_poller)

    # 2. Mock make_zmq_socket to return mock Socket
    mock_socket = AsyncMock()
    # Add necessary attributes to mock socket (avoid errors in other places)
    mock_socket.fd = Mock(return_value=1)  # Mock file descriptor
    mock_socket.getsockopt = Mock(return_value=0)

    # 3. Batch mock related dependencies
    with (
        patch(
            "vllm.v1.fault_tolerance.client_sentinel.make_zmq_socket",
            return_value=mock_socket,
        ),
        patch("zmq.asyncio.Poller", mock_poller_class),
        patch(
            "vllm.v1.fault_tolerance.client_sentinel.asyncio.create_task"
        ) as mock_create_task,
    ):
        # 4. Disable real async tasks (avoid run/_monitor_and_pause_on_fault execution)
        mock_create_task.return_value = Mock()

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
            call_utility_async=mock_call_utility_async,
            core_engines=[b"engine_0", b"engine_1"],
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
    # Verify ZMQ sockets are created
    assert len(client_sentinel.ft_request_sockets) == 1
    assert len(client_sentinel.ft_result_sockets) == 1


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
async def test_pause_operation(
    client_sentinel: ClientSentinel, mock_call_utility_async
):
    """Test pause method"""
    # Mock all engines to pause successfully
    mock_call_utility_async.return_value = {
        "request_id": "request_id",
        "success": True,
        "reason": None,
    }

    request = FaultToleranceRequest.builder(
        request_id="request_id",
        instruction="pause",
        params={"timeout": 3},
    )

    # Execute pause
    result = await client_sentinel.pause(request)

    # Verify result
    assert result.request_id == "request_id"
    assert result.success is True
    assert result.reason is None

    # Verify call parameters
    assert mock_call_utility_async.call_count == 2
    for call in mock_call_utility_async.call_args_list:
        assert call.args[0] == "handle_fault"
        assert call.args[1] == request
        assert call.kwargs["engine"] in {b"engine_0", b"engine_1"}


@pytest.mark.asyncio
async def test_pause_operation_timeout(
    client_sentinel: ClientSentinel, mock_call_utility_async
):
    """Pause should fail if engine responses exceed request timeout."""

    async def slow_response(*args, **kwargs):
        await asyncio.sleep(0.1)
        return {"request_id": "request_id", "success": True, "reason": None}

    mock_call_utility_async.side_effect = slow_response

    request = FaultToleranceRequest.builder(
        request_id="request_id",
        instruction="pause",
        params={"timeout": 0.01},
    )

    result = await client_sentinel.pause(request)

    assert result.request_id == "request_id"
    assert result.success is False
    assert result.reason == "Timed out after 0.01s waiting for engine responses."


@pytest.mark.asyncio
async def test_shutdown(client_sentinel: ClientSentinel):
    """Test shutdown method."""
    with patch("vllm.v1.fault_tolerance.client_sentinel.close_sockets") as mock_close:
        client_sentinel.shutdown()

    assert mock_close.call_count == 2
    first_call_args = mock_close.call_args_list[0].args[0]
    second_call_args = mock_close.call_args_list[1].args[0]
    assert first_call_args == [
        client_sentinel.fault_receiver_socket,
        client_sentinel.fault_state_pub_socket,
    ]
    assert second_call_args == (
        client_sentinel.ft_request_sockets + client_sentinel.ft_result_sockets
    )
    assert client_sentinel.sentinel_dead is True
