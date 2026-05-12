# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import socket
import time
from unittest.mock import Mock, patch

import pytest
import zmq
from msgspec import msgpack

from vllm.config import FaultToleranceConfig, ParallelConfig
from vllm.v1.engine import EngineStatusType
from vllm.v1.engine.exceptions import EngineLoopPausedError
from vllm.v1.fault_tolerance import EngineCoreSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
)
from vllm.v1.utils import get_engine_client_zmq_addr

pytestmark = pytest.mark.skip_global_cleanup


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def addr_dict():
    ports = [_find_free_port() for _ in range(3)]
    return {
        "client_cmd_addr": f"tcp://127.0.0.1:{ports[0]}",
        "worker_cmd_addr": f"tcp://127.0.0.1:{ports[1]}",
        "engine_fault_socket_addr": f"tcp://127.0.0.1:{ports[2]}",
    }


@pytest.fixture
def mock_parallel_config():
    """Create mock ParallelConfig object"""
    config = Mock(spec=ParallelConfig)

    config.data_parallel_index = 0
    config.data_parallel_size = 2
    config.data_parallel_size_local = 2
    config.tensor_parallel_size = 1
    config.pipeline_parallel_size = 1
    config.local_engines_only = False

    config.fault_tolerance_config = FaultToleranceConfig(engine_recovery_timeout_sec=10)
    return config


def create_engine_core_sentinel(
    parallel_config: ParallelConfig,
    addr_dict: dict,
    sentinel_identity: bytes = b"engine_sentinel_0",
):
    engine = Mock()
    engine.engine_index = 0
    engine.input_queue = queue.Queue()
    engine.vllm_config = Mock()
    engine.vllm_config.parallel_config = parallel_config
    engine.dp_group = "old_dp_group"
    engine.step_counter = 123
    worker_cmd_addr = get_engine_client_zmq_addr(True, "0.0.0.0")
    sentinel = EngineCoreSentinel(
        parallel_config,
        engine_fault_socket_addr=addr_dict["engine_fault_socket_addr"],
        sentinel_identity=sentinel_identity,
        engine=engine,
        worker_cmd_addr=worker_cmd_addr,
    )
    return sentinel


@pytest.mark.parametrize(
    "engine_exception, expected_type, expected_status, expected_message",
    [
        (
            RuntimeError("test exception"),
            "RuntimeError",
            EngineStatusType.UNHEALTHY,
            "test exception",
        ),
        (
            EngineLoopPausedError("paused"),
            "EngineLoopPausedError",
            EngineStatusType.PAUSED,
            "[EnginePaused] paused",
        ),
    ],
)
def test_busy_loop_exception_forwarded_to_client(
    addr_dict,
    mock_parallel_config,
    engine_exception,
    expected_type,
    expected_status,
    expected_message,
):
    """
    Verify that an engine exception reported to EngineCoreSentinel
    is forwarded as a FaultInfo message to the client-facing
    engine fault socket.
    """
    sentinel_identity = b"engine_sentinel_0"
    sentinel = create_engine_core_sentinel(
        mock_parallel_config, addr_dict, sentinel_identity=sentinel_identity
    )

    # Bind a ROUTER to the engine_fault_socket_addr to receive the fault report.
    ctx = zmq.Context()
    engine_fault_receiver = ctx.socket(zmq.ROUTER)
    engine_fault_receiver.bind(addr_dict["engine_fault_socket_addr"])

    try:
        time.sleep(0.1)
        sentinel.report_fault_events(engine_exception, expected_status)
        # Wait for the sentinel to forward the fault to the engine_fault socket.
        if not engine_fault_receiver.poll(timeout=5000):
            pytest.fail("Timeout waiting for engine fault message from sentinel")

        parts = engine_fault_receiver.recv_multipart()
        assert len(parts) >= 2
        fault_info = msgpack.decode(parts[-1], type=FaultInfo)
        assert fault_info.type == expected_type
        assert fault_info.engine_id == 0
        assert fault_info.engine_status == expected_status
        assert fault_info.message == expected_message
    finally:
        engine_fault_receiver.close(linger=0)
        sentinel.shutdown()
        ctx.term()


def test_engine_status_wire_format_is_pinned():
    expected = {
        EngineStatusType.HEALTHY: (0, "healthy"),
        EngineStatusType.DEAD: (1, "dead"),
        EngineStatusType.UNHEALTHY: (2, "unhealthy"),
        EngineStatusType.PAUSED: (3, "paused"),
        EngineStatusType.HUNG: (4, "hung"),
    }

    missing = set(EngineStatusType) - set(expected)
    assert not missing, (
        "Add new EngineStatusType values to this pin table. "
        f"Missing: {sorted(m.name for m in missing)}"
    )

    for member, (expected_int, expected_str) in expected.items():
        assert int(member) == expected_int
        assert member.name.lower() == expected_str


@pytest.mark.parametrize("dp_size", [1, 2])
def test_retry(mock_parallel_config, addr_dict, dp_size):
    mock_parallel_config.data_parallel_size = dp_size
    mock_parallel_config.stateless_init_dp_group.return_value = "new_dp_group"
    sentinel_identity = b"engine_sentinel_0"
    engine_core_sentinel = create_engine_core_sentinel(
        mock_parallel_config, addr_dict, sentinel_identity=sentinel_identity
    )
    engine_core_sentinel.busy_loop_paused.set()
    ft_req = FaultToleranceRequest(
        "1", "retry", {"timeout": 2, "coord_store_port": 54321}
    )

    try:
        with (
            patch.object(
                engine_core_sentinel,
                "_execute_command_on_workers",
            ) as execute_on_workers,
            patch.object(
                engine_core_sentinel, "clean_engine_state"
            ) as clean_engine_state,
            patch(
                "vllm.v1.fault_tolerance.engine_core_sentinel."
                "stateless_destroy_torch_distributed_process_group"
            ) as destroy_dp_group,
        ):
            result = engine_core_sentinel.retry(ft_req)

        assert result.success
        assert mock_parallel_config._coord_store_port == 54321
        execute_on_workers.assert_called_once()
        worker_req, target_workers = execute_on_workers.call_args.args
        assert worker_req.instruction == "retry"
        assert target_workers == engine_core_sentinel.worker_identities
        assert execute_on_workers.call_args.kwargs["timeout"] == 2
        clean_engine_state.assert_called_once_with()

        if dp_size > 1:
            destroy_dp_group.assert_called_once_with("old_dp_group")
            mock_parallel_config.stateless_init_dp_group.assert_called_once_with()
            assert engine_core_sentinel.host.dp_group == "new_dp_group"
            assert engine_core_sentinel.host.step_counter == 0
        else:
            destroy_dp_group.assert_not_called()
            mock_parallel_config.stateless_init_dp_group.assert_not_called()
            assert engine_core_sentinel.host.dp_group == "old_dp_group"
            assert engine_core_sentinel.host.step_counter == 123

        assert engine_core_sentinel.run_busy_loop.is_set()
    finally:
        engine_core_sentinel.shutdown()
