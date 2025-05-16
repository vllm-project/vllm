# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import socket
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from vllm.entrypoints.cli.serve import APIServerProcessManager

# Global variables to control worker behavior
WORKER_SHOULD_FAIL = False
WORKER_RUNTIME_SECONDS = 0.5


# Mock implementation of run_api_server_worker
def mock_run_api_server_worker(listen_address, sock, args, client_config=None):
    """Mock implementation of run_api_server_worker that can be configured to fail or run for a specific time."""
    print(f"Mock worker started with client_config: {client_config}")
    start_time = time.time()

    # Run until time expires or we're told to fail
    while time.time() - start_time < WORKER_RUNTIME_SECONDS:
        if WORKER_SHOULD_FAIL:
            print("Worker failed as requested")
            return 1  # Non-zero exit code
        time.sleep(0.1)


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
    """Test initializing the APIServerProcessManager with and without stats configurations."""
    # Set the worker runtime to ensure tests complete in reasonable time
    global WORKER_RUNTIME_SECONDS, WORKER_SHOULD_FAIL
    WORKER_RUNTIME_SECONDS = 0.5
    WORKER_SHOULD_FAIL = False

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
    """Test that wait_for_completion_or_failure properly handles process failures."""
    global WORKER_RUNTIME_SECONDS, WORKER_SHOULD_FAIL
    WORKER_RUNTIME_SECONDS = 1.0
    WORKER_SHOULD_FAIL = False

    # Create the manager
    manager = APIServerProcessManager(**api_server_args)

    try:
        assert len(manager.processes) == 3

        # Start a thread to run wait_for_completion_or_failure
        wait_thread = threading.Thread(
            target=manager.wait_for_completion_or_failure, daemon=True)
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
        time.sleep(0.5)

        # The wait thread should have exited
        assert not wait_thread.is_alive()

        # All processes should now be terminated
        for i, proc in enumerate(manager.processes):
            assert not proc.is_alive(), f"Process {i} should not be alive"

    finally:
        manager.close()
        time.sleep(0.2)


@patch("vllm.entrypoints.cli.serve.run_api_server_worker",
       mock_run_api_server_worker)
def test_normal_completion(api_server_args):
    """Test that when all processes complete normally, wait_for_completion_or_failure exits normally."""
    global WORKER_RUNTIME_SECONDS, WORKER_SHOULD_FAIL
    # Set short runtime so processes complete quickly
    WORKER_RUNTIME_SECONDS = 0.3
    WORKER_SHOULD_FAIL = False

    # Create the manager
    manager = APIServerProcessManager(**api_server_args)

    # Create mocks to verify the behavior
    manager.close = MagicMock()

    try:
        # Start a thread to run wait_for_completion_or_failure
        wait_thread = threading.Thread(
            target=manager.wait_for_completion_or_failure, daemon=True)
        wait_thread.start()

        # Wait for processes to complete normally
        time.sleep(WORKER_RUNTIME_SECONDS + 0.2)

        # The wait thread should have exited
        assert not wait_thread.is_alive()

        # The close method should have been called
        manager.close.assert_called_once()

    finally:
        # Restore the original close method and ensure cleanup
        manager.close = APIServerProcessManager.close.__get__(manager)
        manager.close()
        time.sleep(0.2)


@patch("vllm.entrypoints.cli.serve.run_api_server_worker",
       mock_run_api_server_worker)
def test_extra_process_failure_terminates_api_servers(api_server_args):
    """Test that API server processes are terminated if a monitored process fails."""
    global WORKER_RUNTIME_SECONDS, WORKER_SHOULD_FAIL
    WORKER_RUNTIME_SECONDS = 1.0
    WORKER_SHOULD_FAIL = False

    # Create a dummy extra process that will be monitored
    def dummy_process_fn():
        time.sleep(0.3)  # Run for a short time
        return 1  # Exit with error

    # Create the monitored process
    spawn_context = multiprocessing.get_context("spawn")
    monitored_proc = spawn_context.Process(target=dummy_process_fn,
                                           name="MonitoredProcess")
    monitored_proc.start()

    # Create the manager with the extra process to monitor
    args = api_server_args.copy()
    manager = APIServerProcessManager(**args,
                                      extra_processes_to_healthcheck=[
                                          monitored_proc
                                      ])

    try:
        # Verify the manager was initialized correctly
        assert len(manager.processes) == 3
        assert len(manager.extra_processes_to_healthcheck) == 1

        # Verify all processes are running initially
        for proc in manager.processes:
            assert proc.is_alive()

        # Start a thread to run wait_for_completion_or_failure
        wait_thread = threading.Thread(
            target=manager.wait_for_completion_or_failure, daemon=True)
        wait_thread.start()

        # Wait for the monitored process to fail
        time.sleep(0.5)

        # The monitored process should have exited with an error code
        assert not monitored_proc.is_alive()
        assert monitored_proc.exitcode != 0

        # Wait for the wait_for_completion_or_failure to detect the failure
        time.sleep(0.5)

        # The wait thread should have raised an exception and exited
        assert not wait_thread.is_alive()

        # All API server processes should have been terminated
        for i, proc in enumerate(manager.processes):
            assert not proc.is_alive(
            ), f"API server process {i} should not be alive"

    finally:
        # Clean up
        manager.close()
        if monitored_proc.is_alive():
            monitored_proc.terminate()
        time.sleep(0.2)


@patch("vllm.entrypoints.cli.serve.run_api_server_worker",
       mock_run_api_server_worker)
@patch("multiprocessing.connection.wait")
def test_error_handling_in_wait_for_completion(mock_wait, api_server_args):
    """Test that any exception in wait_for_completion_or_failure terminates all processes."""
    global WORKER_RUNTIME_SECONDS, WORKER_SHOULD_FAIL
    WORKER_RUNTIME_SECONDS = 1.0
    WORKER_SHOULD_FAIL = False

    # Mock the wait function to raise an exception
    mock_wait.side_effect = Exception("Simulated error in connection.wait")

    # Create the manager
    manager = APIServerProcessManager(**api_server_args)
    manager.close = MagicMock()  # Mock the close method to verify it's called

    try:
        # Start a thread to run wait_for_completion_or_failure
        error_event = threading.Event()

        def run_with_error_capture():
            try:
                manager.wait_for_completion_or_failure()
            except Exception:
                error_event.set()

        wait_thread = threading.Thread(target=run_with_error_capture,
                                       daemon=True)
        wait_thread.start()

        # Wait for the exception to be caught
        time.sleep(0.3)

        # Verify that the error was caught
        assert error_event.is_set(), "Expected exception was not raised"

        # Verify the close method was called to terminate all processes
        manager.close.assert_called_once()

    finally:
        # Restore the original close method and ensure cleanup
        manager.close = APIServerProcessManager.close.__get__(manager)
        manager.close()
        time.sleep(0.2)


@patch("vllm.entrypoints.cli.serve.run_api_server_worker",
       mock_run_api_server_worker)
def test_run_servers_with_failure(api_server_args):
    """Test the behavior when a process fails during the run_servers flow."""
    global WORKER_RUNTIME_SECONDS, WORKER_SHOULD_FAIL
    WORKER_RUNTIME_SECONDS = 0.5
    WORKER_SHOULD_FAIL = False

    # Create the manager with a mock process that will fail
    manager = APIServerProcessManager(**api_server_args)

    # Replace one of the real processes with a mock that will fail
    real_process = manager.processes[0]
    mock_process = Mock()
    mock_process.is_alive.return_value = True
    mock_process.exitcode = None
    mock_process.name = "MockFailingProcess"
    mock_process.pid = 12345

    # Set up the mock to "fail" after a short time
    def side_effect_alive():
        # Return True first time, then False to simulate failure
        mock_process.is_alive.return_value = False
        mock_process.exitcode = 1
        return True

    mock_process.is_alive.side_effect = side_effect_alive

    # Replace the real process with our mock
    manager.processes[0] = mock_process

    try:
        # Create a result capture for the thread
        result = {"exception": None}

        def run_with_exception_capture():
            try:
                manager.wait_for_completion_or_failure()
            except Exception as e:
                result["exception"] = e

        # Start the thread to run wait_for_completion_or_failure
        wait_thread = threading.Thread(target=run_with_exception_capture,
                                       daemon=True)
        wait_thread.start()

        # Give time for the mock process to "fail"
        time.sleep(0.3)

        # Wait for the thread to exit
        wait_thread.join(timeout=0.5)

        # Verify an exception was raised
        assert result["exception"] is not None
        assert "died with exit code" in str(result["exception"])

    finally:
        # Put back the real process and clean up
        manager.processes[0] = real_process
        manager.close()
        time.sleep(0.2)
