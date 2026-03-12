# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import uuid
from unittest.mock import AsyncMock, Mock, patch

import msgspec.msgpack
import pytest
import zmq.asyncio

from vllm.config import FaultToleranceConfig, VllmConfig
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.client_sentinel import ClientSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
    FaultToleranceResult,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def mock_vllm_config():
    """Create mock VllmConfig object"""
    config = Mock(spec=VllmConfig)
    config.parallel_config = Mock(
        data_parallel_index=0,
        data_parallel_size=2,
        data_parallel_size_local=2,
    )
    config.fault_tolerance_config = FaultToleranceConfig(gloo_comm_timeout=10)
    return config


@pytest.fixture
def mock_ft_addresses():
    """Create mock FaultToleranceZmqAddresses object"""
    addresses = Mock()
    addresses.all_client_input_addresses = ["tcp://127.0.0.1:5555"]
    addresses.all_client_output_addresses = ["tcp://127.0.0.1:5556"]
    addresses.engine_fault_socket_addr = "tcp://127.0.0.1:5557"
    addresses.fault_state_pub_socket_addr = "tcp://127.0.0.1:5558"
    addresses.ft_config = Mock(fault_state_pub_topic="fault_state")
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

        # 5. Create ClientSentinel instance
        sentinel = ClientSentinel(
            vllm_config=mock_vllm_config,
            fault_tolerance_addresses=mock_ft_addresses,
            call_utility_async=mock_call_utility_async,
            core_engines=[b"engine_0", b"engine_1"],
        )

        return sentinel


# -------------------------- Test Cases --------------------------
@pytest.mark.asyncio
async def test_client_sentinel_initialization(client_sentinel: ClientSentinel):
    """Test ClientSentinel initialization logic"""
    # Verify engine status dictionary initialization is correct
    assert client_sentinel.engine_status_dict == {
        0: {"status": EngineStatusType.HEALTHY},
        1: {"status": EngineStatusType.HEALTHY},
    }

    # Verify key attribute initialization
    assert client_sentinel.start_rank == 0

    assert not client_sentinel.is_faulted.is_set()

    # Verify ZMQ sockets are created
    assert len(client_sentinel.input_sockets) == 1
    assert len(client_sentinel.output_sockets) == 1
    assert client_sentinel.fault_receiver_socket is not None
    assert client_sentinel.fault_state_pub_socket is not None


@pytest.mark.asyncio
async def test_retry_success(client_sentinel: ClientSentinel, mock_call_utility_async):
    """Test retry method (success scenario)"""
    # Mock engine to return successful result
    mock_call_utility_async.return_value = {
        "request_id": "request_id",
        "success": True,
        "reason": "success",
    }

    # Execute retry
    result = await client_sentinel.retry(timeout=5)

    # Verify result
    assert result is True
    assert not client_sentinel.is_faulted.is_set()

    # Verify call parameters
    mock_call_utility_async.assert_awaited()
    call_args = mock_call_utility_async.call_args[0]
    assert call_args[0] == "handle_fault"
    assert isinstance(call_args[1], FaultToleranceRequest)
    assert call_args[1].instruction == "retry"
    assert call_args[1].params["timeout"] == 5


@pytest.mark.asyncio
async def test_retry_failure(client_sentinel: ClientSentinel, mock_call_utility_async):
    """Test retry method (failure scenario)"""
    # Mock one engine to return failure
    mock_call_utility_async.side_effect = [
        {"success": True},
        {"success": False, "reason": "engine dead"},
    ]

    # Mark one engine as DEAD first
    client_sentinel.engine_status_dict[1] = {"status": EngineStatusType.DEAD}

    # Execute retry
    result = await client_sentinel.retry()

    # Verify result
    assert result is False


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

    # Execute pause
    result = await client_sentinel.pause(timeout=3, soft_pause=True)

    # Verify result
    assert result is True
    assert client_sentinel.engine_status_dict[0]["status"] == EngineStatusType.PAUSED
    assert client_sentinel.engine_status_dict[1]["status"] == EngineStatusType.PAUSED

    # Verify call parameters
    call_args = mock_call_utility_async.call_args[0][1]
    assert call_args.instruction == "pause"
    assert call_args.params["timeout"] == 3
    assert call_args.params["soft_pause"] is True


@pytest.mark.asyncio
async def test_monitor_and_pause_on_fault(client_sentinel: ClientSentinel):
    """Test fault monitoring and automatic pause logic (fix parameter conflict)"""

    fault_info = FaultInfo(engine_id="0", type="dead", message="dead")
    client_sentinel.fault_receiver_socket.recv_multipart = AsyncMock(
        return_value=[b"", b"", msgspec.msgpack.encode(fault_info)]
    )

    async def mock_poll(self, timeout):
        if not hasattr(mock_poll, "called"):
            mock_poll.called = True  # type: ignore[attr-defined]
            return [(client_sentinel.fault_receiver_socket, zmq.POLLIN)]
        else:
            client_sentinel.sentinel_dead = True
            return []

    with (
        patch.object(asyncio, "sleep", return_value=None),
        patch("zmq.asyncio.Poller.poll", mock_poll),
    ):  # Replace Poller.poll
        await client_sentinel._monitor_and_pause_on_fault()

    assert client_sentinel.engine_status_dict[0]["status"] == EngineStatusType.DEAD
    assert client_sentinel.is_faulted.is_set()
    client_sentinel.fault_state_pub_socket.send_multipart.assert_awaited()


@pytest.mark.asyncio
async def test_handle_command_from_socket(client_sentinel: ClientSentinel):
    """Test handling external commands from ZMQ socket (fix ft_args type error)"""
    # 1. Construct valid FaultToleranceRequest parameter dictionary
    test_request_id = str(uuid.uuid4())
    ft_request_dict = {
        "request_id": test_request_id,
        "instruction": "retry",
        "params": {"timeout": 2},
    }

    # 2. Construct correctly formatted encoded data (core fix)
    encode_data = [
        0,  # client_index (int)
        123,  # call_id (int)
        b"",  # empty byte placeholder
        [ft_request_dict],  # ft_args → dictionary wrapped in list (key!)
    ]

    # 3. Mock data received by socket (DEALER mode: first frame is empty)
    mock_data = [
        b"",  # empty frame for DEALER mode
        msgspec.msgpack.encode(encode_data),  # encode complete data
    ]

    # 4. Mock recv_multipart method of input socket
    client_sentinel.input_sockets[0].recv_multipart = AsyncMock(return_value=mock_data)

    # 5. Mock poller.poll as AsyncMock (avoid await list error)
    async def mock_poll(self, timeout):
        if not hasattr(mock_poll, "called"):
            mock_poll.called = True  # type: ignore[attr-defined]
            return [(client_sentinel.input_sockets[0], zmq.POLLIN)]
        else:
            client_sentinel.sentinel_dead = True
            return []

    # 6. Mock _send_utility_result for assertion
    client_sentinel._send_utility_result = AsyncMock()

    # 7. Execute run method
    with patch("zmq.asyncio.Poller.poll", mock_poll):
        await client_sentinel.run()

    # 8. Verify command handling result
    client_sentinel._send_utility_result.assert_awaited_with(
        0,  # client_index
        123,  # call_id
        FaultToleranceResult(
            success=True,
            request_id=test_request_id,
            reason=None,  # add required fields
        ),
    )


@pytest.mark.asyncio
async def test_shutdown(client_sentinel: ClientSentinel):
    """Test shutdown method"""
    # Mock close_sockets function
    with patch("vllm.v1.fault_tolerance.client_sentinel.close_sockets") as mock_close:
        client_sentinel.shutdown()
        from unittest.mock import call

        mock_close.assert_has_calls(
            [
                call(
                    [
                        client_sentinel.fault_receiver_socket,
                        client_sentinel.fault_state_pub_socket,
                    ]
                ),
                call(client_sentinel.input_sockets + client_sentinel.output_sockets),
            ],
            any_order=True,
        )
        assert client_sentinel.sentinel_dead is True
