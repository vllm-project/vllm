# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import socket
import threading
import time
from unittest.mock import patch

import pytest
import zmq

from vllm.utils.network_utils import make_zmq_listener, make_zmq_socket
from vllm.v1.utils import (
    APIServerProcessManager,
    wait_for_completion_or_failure,
)

# Global variables to control worker behavior
WORKER_RUNTIME_SECONDS = 0.5


# Mock implementation of run_api_server_worker
def mock_run_api_server_worker(listen_address, sock, args, client_config=None):
    """Mock run_api_server_worker that runs for a specific time."""
    print(f"Mock worker started with client_config: {client_config}")
    time.sleep(WORKER_RUNTIME_SECONDS)
    print("Mock worker completed successfully")


def _adopt_zmq_listener_worker(listen_address, sock, args, client_config):
    ctx = zmq.Context()
    router = make_zmq_socket(
        ctx,
        client_config["input_address"],
        zmq.ROUTER,
        bind=True,
        listener=client_config["input_listener"],
    )
    pull = make_zmq_socket(
        ctx,
        client_config["output_address"],
        zmq.PULL,
        bind=True,
        listener=client_config["output_listener"],
    )
    try:
        identity, payload = router.recv_multipart()
        assert payload == b"input"
        assert pull.recv() == b"output"
        router.send_multipart([identity, b"ok"])
    finally:
        router.close(linger=0)
        pull.close(linger=0)
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
        "input_listeners": [
            make_zmq_listener("tcp://127.0.0.1:0", zmq.ROUTER) for _ in range(3)
        ],
        "output_listeners": [
            make_zmq_listener("tcp://127.0.0.1:0", zmq.PULL) for _ in range(3)
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
        assert all(
            listener.socket.fileno() == -1
            for listener in (
                *args["input_listeners"],
                *args["output_listeners"],
            )
        )

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


def test_api_server_child_adopts_inherited_zmq_listeners():
    input_listener = make_zmq_listener("tcp://127.0.0.1:0", zmq.ROUTER)
    output_listener = make_zmq_listener("tcp://127.0.0.1:0", zmq.PULL)
    input_address = input_listener.address
    output_address = output_listener.address
    http_socket = socket.socket()
    manager = APIServerProcessManager(
        target_server_fn=_adopt_zmq_listener_worker,
        listen_address="localhost:8000",
        sock=http_socket,
        args="test_args",
        num_servers=1,
        input_listeners=[input_listener],
        output_listeners=[output_listener],
    )

    ctx = zmq.Context()
    dealer = make_zmq_socket(
        ctx, input_address, zmq.DEALER, bind=False, identity=b"test"
    )
    push = make_zmq_socket(ctx, output_address, zmq.PUSH, bind=False)
    dealer.setsockopt(zmq.RCVTIMEO, 30_000)
    try:
        assert input_listener.socket.fileno() == -1
        assert output_listener.socket.fileno() == -1
        dealer.send(b"input")
        push.send(b"output")
        assert dealer.recv() == b"ok"
        manager.processes[0].join(timeout=10)
        assert manager.processes[0].exitcode == 0
    finally:
        manager.shutdown()
        dealer.close(linger=0)
        push.close(linger=0)
        ctx.term()
        http_socket.close()


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
    prev_worker_runtime = WORKER_RUNTIME_SECONDS
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
        WORKER_RUNTIME_SECONDS = prev_worker_runtime


def test_rust_frontend_launch_log_redacts_credentials(monkeypatch, caplog):
    """The Rust frontend command carries every non-default arg as JSON.
    Credentials must not reach the log line."""
    import subprocess as subprocess_mod

    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.utils import RustFrontendProcessManager

    hf_token = "hf_TESTTOKEN123"
    api_key = "sk-TESTKEY456"
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(
        ["--model", "org/model", "--hf-token", hf_token, "--api-key", api_key]
    )

    class _FakeProc:
        pid = 4321
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    input_listener = make_zmq_listener("tcp://127.0.0.1:0", zmq.ROUTER)
    output_listener = make_zmq_listener("tcp://127.0.0.1:0", zmq.PULL)
    try:
        monkeypatch.setattr(subprocess_mod, "Popen", lambda *a, **kw: _FakeProc())
        with caplog.at_level("INFO", logger="vllm.v1.utils"):
            RustFrontendProcessManager(
                binary_path="/nonexistent/vllm-rs",
                sock=sock,
                args=args,
                input_listener=input_listener,
                output_listener=output_listener,
                engine_start_index=0,
                engine_count=1,
                data_parallel_size=1,
            )
    finally:
        sock.close()

    message = caplog.text
    assert "Launching Rust frontend:" in message
    assert hf_token not in message
    assert api_key not in message
    assert '"hf_token": "***"' in message
