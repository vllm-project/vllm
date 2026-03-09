
import pytest
from unittest.mock import MagicMock
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from collections import deque
from contextlib import nullcontext

from vllm.v1.engine.core import EngineCore
from vllm import envs

def test_step_timeout(monkeypatch):
    """
    Test that step_with_batch_queue raises TimeoutError when the future does not complete
    within VLLM_ENGINE_STEP_TIMEOUT_MS.
    """
    # Set short timeout: 100ms
    monkeypatch.setattr(envs, "VLLM_ENGINE_STEP_TIMEOUT_MS", 100)
    
    # Mock EngineCore instance
    core = MagicMock(spec=EngineCore)
    core.batch_queue_size = 1
    
    # Bind the method under test to the mock instance
    # This allows us to call core.step_with_batch_queue() and have it use our mock 'self'
    core.step_with_batch_queue = EngineCore.step_with_batch_queue.__get__(core, EngineCore)
    
    # Setup batch_queue with a Future that will hang (never set result)
    future = Future() 
    scheduler_output = MagicMock()
    exec_future = Future()
    
    # batch_queue.pop() returns (future, scheduler_output, exec_future)
    # The method uses batch_queue.pop(), so we need to mock batch_queue
    # But batch_queue is accessed via self.batch_queue
    core.batch_queue = deque([(future, scheduler_output, exec_future)])
    
    # Mock context managers for logging
    core.log_error_detail.return_value = nullcontext()
    core.log_iteration_details.return_value = nullcontext()
    
    # Also need to mock _process_aborts_queue and scheduler.update_from_output
    # But those are called AFTER the blocking call.
    # So if timeout works, we won't reach them.
    
    # Run the method and expect TimeoutError
    with pytest.raises(FutureTimeoutError):
        core.step_with_batch_queue()

def test_step_success(monkeypatch):
    """
    Test that step_with_batch_queue works when future completes on time.
    """
    # Set timeout: 1000ms
    monkeypatch.setattr(envs, "VLLM_ENGINE_STEP_TIMEOUT_MS", 1000)
    
    core = MagicMock(spec=EngineCore)
    core.batch_queue_size = 1
    core.step_with_batch_queue = EngineCore.step_with_batch_queue.__get__(core, EngineCore)
    
    # Setup future with result
    future = Future()
    expected_output = MagicMock()
    future.set_result(expected_output)
    
    scheduler_output = MagicMock()
    exec_future = Future()
    
    core.batch_queue = deque([(future, scheduler_output, exec_future)])
    
    core.log_error_detail.return_value = nullcontext()
    core.log_iteration_details.return_value = nullcontext()
    
    # Mock update_from_output
    core.scheduler.update_from_output.return_value = {}
    
    # Mock deferred_scheduler_output (None)
    # The method checks `if deferred_scheduler_output:` at the end.
    # We need to make sure `scheduler_output` is what it expects.
    # In the method:
    # deferred_scheduler_output = scheduler_output if pending_structured_output_tokens else None
    # Wait, deferred_scheduler_output is a local variable in the method?
    # No, it's from the PREVIOUS step, stored in `deferred_scheduler_output`?
    # Let's look at the code.
    
    # The method logic:
    # 1. Schedule new work (if scheduler has requests)
    # 2. Check if batch_queue is empty (if so return)
    # 3. Pop from batch_queue and wait
    
    # The scheduling part is at the beginning.
    # If we want to skip scheduling part and just test popping, we need to handle the first part.
    
    # In the mock, we can make scheduler.schedule() return None or raise?
    # No, the code calls self.scheduler.schedule().
    
    # Let's mock scheduler.schedule
    mock_sched_out = MagicMock()
    mock_sched_out.pending_structured_output_tokens = False
    core.scheduler.schedule.return_value = mock_sched_out
    
    core.model_executor.execute_model.return_value = Future()
    core.is_ec_consumer = False
    core.is_pooling_model = False
    
    # Run
    outputs, executed = core.step_with_batch_queue()
    
    assert outputs == {} # From update_from_output
