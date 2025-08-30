# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from queue import Queue
from unittest.mock import patch

import pytest

from vllm.distributed.eplb.eplb_state import EPLBThread


# Simple target function for testing
def mock_target_func(*args):
    """Mock target function that returns the sum of input arguments"""
    time.sleep(0.1)
    return sum(args)


# Target function that raises an exception for testing error handling
def mock_target_func_with_exception(*args):
    """Mock target function that raises an exception"""
    time.sleep(0.1)
    raise ValueError("Test exception")


class TestEPLBThread:
    """Test class for EPLBThread"""

    def test_init(self):
        """Test initialization of EPLBThread"""
        target_func = mock_target_func
        num_wait_iterations = 5

        thread = EPLBThread(target_func, num_wait_iterations)

        assert thread.target_func == target_func
        assert thread._num_wait_worker_iterations == num_wait_iterations
        assert thread._thread is None
        assert thread._input_queue is None
        assert thread._result_queue is None
        assert thread._exception_queue is None
        assert thread._step_counter == 0
        assert thread._result is None
        assert thread._args is None
        assert not thread._is_running
        assert thread._post_process_args is None

    def test_start_success(self):
        """Test successful startup of the asynchronous thread"""
        thread = EPLBThread(mock_target_func, 3)
        args = (1, 2, 3)
        post_process_args = {"param1": "value1"}

        result = thread.start(args, post_process_args)

        assert result is True
        assert thread._is_running is True
        assert thread._args == args
        assert thread._post_process_args == post_process_args
        assert thread._thread is not None
        assert thread._thread.is_alive() is True

        # Clean up resources
        thread.cleanup()

    def test_start_failure(self):
        """Test failure scenario when starting the thread"""
        thread = EPLBThread(mock_target_func, 3)

        # Mock Thread to raise exception during initialization
        with patch('threading.Thread') as mock_thread:
            mock_thread.side_effect = Exception("Thread creation failed")

            result = thread.start((1, 2, 3), {})

            assert result is False
            assert thread._is_running is False

    def test_worker_normal_execution(self):
        """Test normal execution flow of the worker thread"""
        input_queue: Queue = Queue()
        result_queue: Queue = Queue()
        exception_queue: Queue = Queue()

        # Put test arguments into the input queue
        test_args = (1, 2, 3)
        input_queue.put(test_args)

        # Create EPLBThread instance and call worker method directly
        thread = EPLBThread(mock_target_func, 3)
        thread._worker(input_queue, result_queue, exception_queue)

        # Verify results - use get() with timeout to ensure we wait for the item
        try:
            result = result_queue.get(timeout=1.0)  # Wait up to 1 second
            assert result == 6  # 1+2+3=6
        except Exception as e:
            pytest.fail(f"Failed to get result from queue: {str(e)}")

        # Verify exception queue is empty
        assert exception_queue.empty() is True

    def test_worker_exception_handling(self):
        """Test worker thread handles exceptions correctly"""
        input_queue: Queue = Queue()
        result_queue: Queue = Queue()
        exception_queue: Queue = Queue()

        # Put test arguments into the input queue
        test_args = (1, 2, 3)
        input_queue.put(test_args)

        # Create EPLBThread instance with exception-raising target
        # and call worker
        thread = EPLBThread(mock_target_func_with_exception, 3)
        thread._worker(input_queue, result_queue, exception_queue)

        # Verify exception handling results
        assert result_queue.empty() is False
        assert result_queue.get() is None
        assert exception_queue.empty() is False
        assert isinstance(exception_queue.get(), ValueError)

    def test_step_normal_processing(self):
        """Test normal processing flow of the step() method"""
        thread = EPLBThread(mock_target_func, 2)
        thread.start((1, 2, 3), {})

        # Wait for the child thread to complete execution
        time.sleep(0.2)

        # First step should trigger result processing
        result = thread.step()
        assert result is True
        assert thread._result == 6  # 1+2+3=6

        # Clean up resources
        thread.cleanup()

    def test_step_with_exception(self):
        """Test step() method handles exceptions from child thread"""
        thread = EPLBThread(mock_target_func_with_exception, 2)
        thread.start((1, 2, 3), {})

        # Wait for the child thread to execute and raise exception
        time.sleep(0.2)

        # step() should raise RuntimeError when handling the exception
        with pytest.raises(RuntimeError, match="Asynchronous thread failed:"):
            thread.step()

        # Clean up resources
        thread.cleanup()

    def test_should_process_conditions(self):
        """Test all conditions in the _should_process() method"""
        thread = EPLBThread(mock_target_func, 3)

        # Should return True when no thread or queue exists
        assert thread._should_process() is True

        # Start the thread to initialize resources
        thread.start((1, 2, 3), {})

        # Should return False when step count < threshold, thread is alive,
        # and queue is empty
        assert thread._should_process() is False

        # Increment step count to meet the threshold
        thread._step_counter = 3
        assert thread._should_process() is True

        # Simulate thread termination
        thread._thread.join(timeout=1.0)
        assert thread._should_process() is True

        # Simulate non-empty result queue
        thread._result_queue.put("test_result")
        assert thread._should_process() is True

        # Clean up resources
        thread.cleanup()

    def test_cleanup(self):
        """Test the cleanup() method releases resources properly"""
        thread = EPLBThread(mock_target_func, 3)
        thread.start((1, 2, 3), {})

        # Verify thread is running before cleanup
        assert thread._is_running is True
        assert thread._thread is not None

        # Execute cleanup
        thread.cleanup()

        # Verify all resources are released
        assert thread._is_running is False
        assert thread._thread is None
        assert thread._input_queue is None
        assert thread._result_queue is None
        assert thread._exception_queue is None

    def test_properties(self):
        """Test property accessors of EPLBThread"""
        thread = EPLBThread(mock_target_func, 3)
        post_process_args = {"param1": "value1"}

        thread.start((1, 2, 3), post_process_args)

        assert thread.is_running is True
        assert thread.result is None  # Result not fetched yet

        # Simulate setting a result
        thread._result = "test_result"
        assert thread.result == "test_result"
        assert thread.post_process_args == post_process_args

        # Clean up resources
        thread.cleanup()
