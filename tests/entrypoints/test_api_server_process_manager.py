# SPDX-License-Identifier: Apache-2.0

import socket
import threading
import time
from unittest.mock import patch

import pytest

from vllm.entrypoints.cli.serve import APIServerProcessManager

# Global variables to control worker behavior
WORKER_RUNTIME_SECONDS = 0.5


# Mock implementation of run_api_server_worker
def mock_run_api_server_worker(listen_address, sock, args, client_config=None):
    """Mock run_api_server_worker that runs for a specific time."""
    print(f"Mock worker started with client_config: {client_config}")
    time.sleep(WORKER_RUNTIME_SECONDS)
    print("Mock worker completed successfully")


@pytest.fixture
def api_server_args():
    """Fixture to provide arguments for APIServerProcessManager."""
    sock = socket.socket()
    return {
        "listen_address":
        "localhost:8000",
        "sock":
        sock,
        "args":
        "test_args",  # Simple string to avoid pickling issues
        "num_servers":
        3,
        "input_addresses": [
            "tcp://127.0.0.1:5001", "tcp://127.0.0.1:5002",
            "tcp://127.0.0.1:5003"
        ],
        "output_addresses": [
            "tcp://127.0.0.1:6001", "tcp://127.0.0.1:6002",
            "tcp://127.0.0.1:6003"
        ],
        "stats_update_address":
        "tcp://127.0.0.1:7000",
    }


@patch("vllm.entrypoints.cli.serve.run_api_server_worker",
       mock_run_api_server_worker)
@pytest.mark.parametrize("with_stats_update", [True, False])
def test_api_server_process_manager_init(api_server_args, with_stats_update):
    """Test initializing the APIServerProcessManager."""
    # Set the worker runtime to ensure tests complete in reasonable time
    global WORKER_RUNTIME_SECONDS
    WORKER_RUNTIME_SECONDS = 0.5

    # Copy the args to avoid mutating the
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
        manager.close()

        # Give processes time to terminate
        time.sleep(0.2)

        # Verify all processes were terminated
        for proc in manager.processes:
            assert not proc.is_alive()


@patch("vllm.entrypoints.cli.serve.run_api_server_worker",
       mock_run_api_server_worker)
def test_wait_for_completion_or_failure(api_server_args):
    """Test that wait_for_completion_or_failure works with failures."""
    global WORKER_RUNTIME_SECONDS
    WORKER_RUNTIME_SECONDS = 1.0

    # Create the manager
    manager = APIServerProcessManager(**api_server_args)

    try:
        assert len(manager.processes) == 3

        # Create a result capture for the thread
        result = {"exception": None}

        def run_with_exception_capture():
            try:
                manager.wait_for_completion_or_failure()
            except Exception as e:
                result["exception"] = e

        # Start a thread to run wait_for_completion_or_failure
        wait_thread = threading.Thread(target=run_with_exception_capture,
                                       daemon=True)
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
        manager.close()
        time.sleep(0.2)


@patch("vllm.entrypoints.cli.serve.run_api_server_worker",
       mock_run_api_server_worker)
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
            assert not proc.is_alive(
            ), f"Process {i} still alive after terminate()"

        # Now call wait_for_completion_or_failure
        # since all processes have already
        # terminated, it should return immediately
        # with no error
        manager.wait_for_completion_or_failure()

    finally:
        # Clean up just in case
        manager.close()
        time.sleep(0.2)
