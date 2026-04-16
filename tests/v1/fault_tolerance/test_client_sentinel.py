# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import msgspec.msgpack
import pytest
import zmq

from vllm.config import FaultToleranceConfig, ParallelConfig
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.client_sentinel import ClientSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo, FaultToleranceRequest

pytestmark = pytest.mark.skip_global_cleanup


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
def mock_parallel_config():
    """Create mock ParallelConfig object"""
    config = Mock(spec=ParallelConfig)

    config.data_parallel_index = 0
    config.data_parallel_size = 2
    config.data_parallel_size_local = 2
    config.local_engines_only = False
    config.gloo_timeout_seconds = 5
    config.fault_tolerance_config = FaultToleranceConfig()
    return config


@pytest.fixture
def client_sentinel(mock_parallel_config, mock_ft_addresses, mock_call_utility_async):
    """Create ClientSentinel fixture with mocked sockets."""
    fault_receiver_socket = AsyncMock()

    with (
        patch(
            "vllm.v1.fault_tolerance.client_sentinel.make_zmq_socket",
            return_value=fault_receiver_socket,
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
        sentinel = ClientSentinel(
            parallel_config=mock_parallel_config,
            fault_tolerance_addresses=mock_ft_addresses,
            call_utility_async=mock_call_utility_async,
            core_engines=[b"engine_0", b"engine_1"],
        )

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

    assert mock_close.call_count == 2
    assert client_sentinel.sentinel_dead is True
