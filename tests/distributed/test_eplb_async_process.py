# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest

from vllm.distributed.eplb import EPLBProcess


def dummy_target_function(arg1, arg2):
    """Simulate target function for testing"""
    return arg1 + arg2, arg1 * arg2


def test_eplb_process_initialization():
    """Test EPLBProcess initialization"""
    process = EPLBProcess(target_func=dummy_target_function,
                          num_wait_worker_iterations=5)

    assert process.target_func == dummy_target_function
    assert process._num_wait_worker_iterations == 5
    assert process._is_running is True
    assert process._has_pending_task is False
    assert process._input_queue is not None
    assert process._result_queue is not None
    assert process._exception_queue is not None
    assert process._process is not None
    assert process._process.is_alive() is True

    # Clean up
    process.cleanup()


def test_eplb_process_submit_task():
    """Test task submission functionality"""
    process = EPLBProcess(target_func=dummy_target_function,
                          num_wait_worker_iterations=3)

    # Submit task
    success = process.submit_task(args=(10, 5),
                                  post_process_args={"test": "data"})

    assert success is True
    assert process._has_pending_task is True
    assert process._args == (10, 5)
    assert process._post_process_args == {"test": "data"}

    # Clean up
    process.cleanup()


def test_eplb_process_submit_task_when_busy():
    """Test submitting new task when existing task is pending"""
    process = EPLBProcess(target_func=dummy_target_function,
                          num_wait_worker_iterations=3)

    # Submit first task
    success1 = process.submit_task(args=(10, 5),
                                   post_process_args={"test": "data1"})

    # Attempt to submit second task
    success2 = process.submit_task(args=(20, 10),
                                   post_process_args={"test": "data2"})

    assert success1 is True
    assert success2 is False  # Should fail because task is pending

    # Clean up
    process.cleanup()


def test_eplb_process_step_without_pending_task():
    """Test step method without pending task"""
    process = EPLBProcess(target_func=dummy_target_function,
                          num_wait_worker_iterations=3)

    # Call step without submitting task
    result = process.step()

    assert result is False  # Should return False because no pending task

    # Clean up
    process.cleanup()


def test_eplb_process_step_with_pending_task():
    """Test step method with pending task"""
    process = EPLBProcess(
        target_func=dummy_target_function,
        num_wait_worker_iterations=1  # Only wait for one step
    )

    # Submit task
    process.submit_task(args=(10, 5), post_process_args={"test": "data"})

    # Wait for task to complete
    time.sleep(0.5)  # Give process time to execute

    # Call step to check result
    result = process.step()

    assert result is True  # Should return True, indicating result processed
    assert process.result == (15, 50)  # 10+5=15, 10 * 5=50
    assert process._has_pending_task is False  # Task completed

    # Clean up
    process.cleanup()


def test_eplb_process_step_before_completion():
    """Test calling step method before task completion"""
    process = EPLBProcess(
        target_func=lambda x: time.sleep(0.2),  # Function that takes time
        num_wait_worker_iterations=10  # Requires multiple wait steps
    )

    # Submit task
    process.submit_task(
        args=(0.2, ),  # Sleep for 0.2 seconds
        post_process_args={"test": "data"})

    # Call step immediately (task should not be completed yet)
    result = process.step()

    assert result is False  # Should return False because task not completed
    assert process._has_pending_task is True  # Task still in progress

    # Clean up
    process.cleanup()


def test_eplb_process_exception_handling():
    """Test exception handling"""

    def failing_function():
        raise ValueError("Test exception")

    process = EPLBProcess(target_func=failing_function,
                          num_wait_worker_iterations=3)

    # Submit task that will fail
    process.submit_task(args=(), post_process_args={"test": "data"})

    # Wait for task to complete
    time.sleep(0.5)

    # Calling step should raise exception
    with pytest.raises(RuntimeError, match="Asynchronous process failed"):
        process.step()

    # Clean up
    process.cleanup()


def test_eplb_process_cleanup():
    """Test cleanup functionality"""
    process = EPLBProcess(target_func=dummy_target_function,
                          num_wait_worker_iterations=3)

    # Ensure process is running
    assert process._is_running is True
    assert process._process.is_alive() is True

    # Perform cleanup
    process.cleanup()

    # Check state
    assert process._is_running is False
    assert process._process is None
    assert process._input_queue is None
    assert process._result_queue is None
    assert process._exception_queue is None


if __name__ == "__main__":
    pytest.main([__file__])
