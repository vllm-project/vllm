# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import socket
import threading
import time
from unittest.mock import patch

import pytest
import zmq

from vllm.utils.network_utils import make_zmq_socket, split_zmq_path
from vllm.v1.utils import APIServerProcessManager, wait_for_completion_or_failure

# Global variables to control worker behavior
WORKER_RUNTIME_SECONDS = 0.5


# Mock implementation of run_api_server_worker
def mock_run_api_server_worker(listen_address, sock, args, client_config=None):
    """Mock run_api_server_worker that runs for a specific time."""
    print(f"Mock worker started with client_config: {client_config}")
    time.sleep(WORKER_RUNTIME_SECONDS)
    print("Mock worker completed successfully")


# Module-level stand-in for `run_api_server_worker_proc` exercising the
# wildcard-bind / pipe-back path. Mirrors what `MPClient` does for the
# `client_addresses` branch: bind ROUTER + PULL with the placeholder URI,
# read the kernel-assigned endpoint via `LAST_ENDPOINT`, and ship a
# `(in_endpoint, out_endpoint)` tuple over `client_config["zmq_addr_pipe"]`.
# Stays alive after reporting so that the parent's `connection.wait`
# does not see the process sentinel fire concurrently with the pipe.
def pipe_back_stub_worker(listen_address, sock, args, client_config):
    ctx = zmq.Context()
    try:
        in_sock = make_zmq_socket(
            ctx, client_config["input_address"], zmq.ROUTER, bind=True
        )
        out_sock = make_zmq_socket(
            ctx, client_config["output_address"], zmq.PULL, bind=True
        )
        try:
            pipe = client_config["zmq_addr_pipe"]
            try:
                pipe.send(
                    (
                        in_sock.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        out_sock.getsockopt(zmq.LAST_ENDPOINT).decode(),
                    )
                )
            finally:
                pipe.close()
            time.sleep(60)
        finally:
            in_sock.close(linger=0)
            out_sock.close(linger=0)
    finally:
        ctx.term()


@pytest.fixture
def api_server_args():
    """Fixture to provide arguments for APIServerProcessManager."""
    sock = socket.socket()
    return {
        "target_server_fn": mock_run_api_server_worker,
        "listen_address": "localhost:8000",
        "sock": sock,
        "args": "test_args",  # Simple string to avoid pickling issues
        "num_servers": 3,
        "input_addresses": [
            "tcp://127.0.0.1:5001",
            "tcp://127.0.0.1:5002",
            "tcp://127.0.0.1:5003",
        ],
        "output_addresses": [
            "tcp://127.0.0.1:6001",
            "tcp://127.0.0.1:6002",
            "tcp://127.0.0.1:6003",
        ],
        "stats_update_address": "tcp://127.0.0.1:7000",
    }


@pytest.mark.parametrize("with_stats_update", [True, False])
def test_api_server_process_manager_init(api_server_args, with_stats_update):
    """Test initializing the APIServerProcessManager."""
    # Set the worker runtime to ensure tests complete in reasonable time
    global WORKER_RUNTIME_SECONDS
    WORKER_RUNTIME_SECONDS = 0.5

    # Copy the args to avoid mutating them
    args = api_server_args.copy()

    if not with_stats_update:
        args.pop("stats_update_address")
    manager = APIServerProcessManager(**args)

    try:
        # Verify the manager was initialized correctly
        assert len(manager.processes) == 3

        # Verify all processes are running
        for proc in manager.processes:
            assert proc.is_alive()

        print("Waiting for processes to run...")
        time.sleep(WORKER_RUNTIME_SECONDS / 2)

        # They should still be alive at this point
        for proc in manager.processes:
            assert proc.is_alive()

    finally:
        # Always clean up the processes
        print("Cleaning up processes...")
        manager.shutdown()

        # Give processes time to terminate
        time.sleep(0.2)

        # Verify all processes were terminated
        for proc in manager.processes:
            assert not proc.is_alive()


@patch("vllm.v1.utils.run_api_server_worker_proc", mock_run_api_server_worker)
def test_wait_for_completion_or_failure(api_server_args):
    """Test that wait_for_completion_or_failure works with failures."""
    global WORKER_RUNTIME_SECONDS
    WORKER_RUNTIME_SECONDS = 1.0

    # Create the manager
    manager = APIServerProcessManager(**api_server_args)

    try:
        assert len(manager.processes) == 3

        # Create a result capture for the thread
        result: dict[str, Exception | None] = {"exception": None}

        def run_with_exception_capture():
            try:
                wait_for_completion_or_failure(api_server_manager=manager)
            except Exception as e:
                result["exception"] = e
            finally:
                manager.shutdown()

        # Start a thread to run wait_for_completion_or_failure
        wait_thread = threading.Thread(target=run_with_exception_capture, daemon=True)
        wait_thread.start()

        # Let all processes run for a short time
        time.sleep(0.2)

        # All processes should still be running
        assert all(proc.is_alive() for proc in manager.processes)

        # Now simulate a process failure
        print("Simulating process failure...")
        manager.processes[0].terminate()

        # Wait for the wait_for_completion_or_failure
        # to detect and handle the failure
        # This should trigger it to terminate all other processes
        wait_thread.join(timeout=1.0)

        # The wait thread should have exited
        assert not wait_thread.is_alive()

        # Verify that an exception was raised with appropriate error message
        assert result["exception"] is not None
        assert "died with exit code" in str(result["exception"])

        # All processes should now be terminated
        for i, proc in enumerate(manager.processes):
            assert not proc.is_alive(), f"Process {i} should not be alive"

    finally:
        manager.shutdown()
        time.sleep(0.2)


@pytest.mark.timeout(30)
def test_normal_completion(api_server_args):
    """Test that wait_for_completion_or_failure works in normal completion."""
    global WORKER_RUNTIME_SECONDS
    WORKER_RUNTIME_SECONDS = 0.1

    # Create the manager
    manager = APIServerProcessManager(**api_server_args)

    try:
        # Give processes time to terminate
        # wait for processes to complete
        remaining_processes = manager.processes.copy()
        while remaining_processes:
            for proc in remaining_processes:
                if not proc.is_alive():
                    remaining_processes.remove(proc)
            time.sleep(0.1)

        # Verify all processes have terminated
        for i, proc in enumerate(manager.processes):
            assert not proc.is_alive(), f"Process {i} still alive after terminate()"

        # Now call wait_for_completion_or_failure
        # since all processes have already
        # terminated, it should return immediately
        # with no error
        try:
            wait_for_completion_or_failure(api_server_manager=manager)
        finally:
            manager.shutdown()

    finally:
        # Clean up just in case
        manager.shutdown()
        time.sleep(0.2)


@pytest.mark.timeout(30)
def test_external_process_monitoring(api_server_args):
    """Test that wait_for_completion_or_failure handles additional processes."""
    global WORKER_RUNTIME_SECONDS
    WORKER_RUNTIME_SECONDS = 100

    # Create and start the external process
    # (simulates local_engine_manager or coordinator)
    spawn_context = multiprocessing.get_context("spawn")
    external_proc = spawn_context.Process(
        target=mock_run_api_server_worker, name="MockExternalProcess"
    )
    external_proc.start()

    # Create the class to simulate a coordinator
    class MockCoordinator:
        def __init__(self, proc):
            self.proc = proc

        def shutdown(self):
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=0.5)

    # Create a mock coordinator with the external process
    mock_coordinator = MockCoordinator(external_proc)

    # Create the API server manager
    manager = APIServerProcessManager(**api_server_args)

    try:
        # Verify manager initialization
        assert len(manager.processes) == 3

        # Create a result capture for the thread
        result: dict[str, Exception | None] = {"exception": None}

        def run_with_exception_capture():
            try:
                wait_for_completion_or_failure(
                    api_server_manager=manager, coordinator=mock_coordinator
                )
            except Exception as e:
                result["exception"] = e
            finally:
                manager.shutdown()
                mock_coordinator.shutdown()

        # Start a thread to run wait_for_completion_or_failure
        wait_thread = threading.Thread(target=run_with_exception_capture, daemon=True)
        wait_thread.start()

        # Terminate the external process to trigger a failure
        time.sleep(0.2)
        external_proc.terminate()

        # Wait for the thread to detect the failure
        wait_thread.join(timeout=1.0)

        # The wait thread should have completed
        assert not wait_thread.is_alive(), (
            "wait_for_completion_or_failure thread still running"
        )

        # Verify that an exception was raised with appropriate error message
        assert result["exception"] is not None, "No exception was raised"
        error_message = str(result["exception"])
        assert "died with exit code" in error_message, (
            f"Unexpected error message: {error_message}"
        )
        assert "MockExternalProcess" in error_message, (
            f"Error doesn't mention external process: {error_message}"
        )

        # Verify that all API server processes were terminated as a result
        for i, proc in enumerate(manager.processes):
            assert not proc.is_alive(), f"API server process {i} was not terminated"

    finally:
        # Clean up
        manager.shutdown()
        mock_coordinator.shutdown()
        time.sleep(0.2)


@pytest.mark.timeout(60)
def test_zmq_pipe_back_end_to_end():
    """Wildcard `tcp://host:0` placeholders go in; each child binds with a
    kernel-assigned port and reports the actual endpoint over its pipe;
    APIServerProcessManager mutates ``input_addresses``/``output_addresses``
    in place to reflect the real ports.

    Runs purely on the local host; the parent⇄child pipe-back loop is the
    same code path used in multi-node deployments (the cross-node aspect
    is just remote engines later DEALER-connecting to the reported
    endpoints, which is outside this component's scope).
    """
    host = "127.0.0.1"
    num_servers = 4
    placeholder_inputs = [f"tcp://{host}:0"] * num_servers
    placeholder_outputs = [f"tcp://{host}:0"] * num_servers

    sock = socket.socket()
    manager = APIServerProcessManager(
        target_server_fn=pipe_back_stub_worker,
        listen_address=f"tcp://{host}:0",
        sock=sock,
        args="test_args",
        num_servers=num_servers,
        input_addresses=placeholder_inputs,
        output_addresses=placeholder_outputs,
    )

    try:
        assert len(manager.processes) == num_servers

        # `__init__` mutates the input lists in place once each child has
        # reported. After return, no entry should still be a port-0
        # placeholder.
        for addr in placeholder_inputs + placeholder_outputs:
            scheme, parsed_host, port = split_zmq_path(addr)
            assert scheme == "tcp", addr
            assert parsed_host == host, addr
            assert port and int(port) > 0, addr

        # All kernel-picked ports are distinct across the 2*N sockets.
        all_addrs = placeholder_inputs + placeholder_outputs
        assert len(set(all_addrs)) == len(all_addrs), all_addrs
    finally:
        manager.shutdown()
        time.sleep(0.2)
        sock.close()


@pytest.mark.timeout(30)
def test_zmq_pipe_back_passes_through_explicit_addresses():
    """When no address ends in ``:0`` the deferred-bind path must NOT
    activate: no per-child pipe is created, no gather is performed, and
    the input/output lists are untouched. Guards against
    ``defer_bind`` accidentally matching real-port TCP URIs."""
    explicit_inputs = [
        "tcp://127.0.0.1:5001",
        "tcp://127.0.0.1:5002",
        "tcp://127.0.0.1:5003",
    ]
    explicit_outputs = [
        "tcp://127.0.0.1:6001",
        "tcp://127.0.0.1:6002",
        "tcp://127.0.0.1:6003",
    ]
    inputs_snapshot = list(explicit_inputs)
    outputs_snapshot = list(explicit_outputs)

    sock = socket.socket()
    manager = APIServerProcessManager(
        target_server_fn=mock_run_api_server_worker,
        listen_address="localhost:8000",
        sock=sock,
        args="test_args",
        num_servers=3,
        input_addresses=explicit_inputs,
        output_addresses=explicit_outputs,
    )
    try:
        # Lists must be byte-identical to what was passed in.
        assert explicit_inputs == inputs_snapshot
        assert explicit_outputs == outputs_snapshot
    finally:
        manager.shutdown()
        time.sleep(0.2)
        sock.close()


@pytest.mark.timeout(30)
def test_zmq_pipe_back_child_crash_before_report():
    """If a deferred-bind child exits before sending its endpoints,
    ``APIServerProcessManager.__init__`` must surface an exception (no
    silent hang up to ``_ZMQ_ADDR_REPORT_TIMEOUT_S``).

    The exact exception type depends on whether ``connection.wait``
    observes the pipe-EOF or the process sentinel first:
      * sentinel first  → ``RuntimeError`` from the explicit guard
      * pipe-EOF first  → ``EOFError`` from ``parent_conn.recv()``

    Both indicate the same root cause and are equally informative for
    the user; tightening this to a single type would be a small
    follow-up (wrap the ``recv()`` in ``try/except EOFError`` and
    re-raise as ``RuntimeError``).
    """
    host = "127.0.0.1"
    num_servers = 2

    sock = socket.socket()
    with pytest.raises((RuntimeError, EOFError)):
        # `mock_run_api_server_worker` exits after a short sleep without
        # touching `zmq_addr_pipe` — simulates a child that dies before
        # the bind/report step.
        APIServerProcessManager(
            target_server_fn=mock_run_api_server_worker,
            listen_address=f"tcp://{host}:0",
            sock=sock,
            args="test_args",
            num_servers=num_servers,
            input_addresses=[f"tcp://{host}:0"] * num_servers,
            output_addresses=[f"tcp://{host}:0"] * num_servers,
        )
    sock.close()
