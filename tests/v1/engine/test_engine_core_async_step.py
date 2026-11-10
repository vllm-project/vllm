# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Result handling of step_with_batch_queue() for steps without sampling.

For any step with scheduled tokens, step_with_batch_queue() submits both
execute_model() and sample_tokens(). The worker contract allows
execute_model() to return a complete ModelRunnerOutput when no sampling
is required; sample_tokens() then has no pending state and resolves to
None. These tests pin the engine's resolution of the relevant
combinations of the two futures' results with mocked executors, so they
run on any host.
"""

from collections import deque
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import ModelRunnerOutput

# Mock-only tests: they run on any host and hold no distributed state to
# clean up, so they opt out of the global post-test cleanup.
pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _make_batch_queue_core() -> MagicMock:
    core = MagicMock()
    core.batch_queue = deque()
    core.batch_queue_size = 2
    core.is_mm_encoder_only = False
    core.is_pooling_model = False
    core.scheduler.has_requests.side_effect = [True, False, False]
    scheduled = core.scheduler.schedule.return_value
    scheduled.total_num_scheduled_tokens = 16
    scheduled.pending_structured_output_tokens = False
    scheduled.has_structured_output_spec_decode = False
    # The real EngineCore routes future results through _wait_for_output.
    core._wait_for_output.side_effect = lambda future: future.result()
    return core


def _drive_batch_queue_core(core) -> None:
    EngineCore.step_with_batch_queue(core)
    if core.batch_queue:
        EngineCore.step_with_batch_queue(core)


@pytest.mark.parametrize(
    "exec_result,sample_result,expected",
    [
        # Sampling deferred to sample_tokens().
        (None, "output", "output"),
        # Sampleless step: execute_model() already returned the output.
        ("output", None, "output"),
        # No output and no exception: unexpected.
        (None, None, "error"),
    ],
)
def test_async_step_sampleless_output(exec_result, sample_result, expected):
    """None from sample_tokens() must fall back to the execute_model()
    output instead of failing a step that requires no sampling; a None
    result from both futures is still an unexpected error."""
    core = _make_batch_queue_core()
    scheduled = core.scheduler.schedule.return_value

    output = MagicMock()
    executed: Future[ModelRunnerOutput | None] = Future()
    executed.set_result(output if exec_result == "output" else None)
    sampled: Future[ModelRunnerOutput | None] = Future()
    sampled.set_result(output if sample_result == "output" else None)
    core.model_executor.execute_model.return_value = executed
    core.model_executor.sample_tokens.return_value = sampled

    if expected == "error":
        with pytest.raises(RuntimeError, match="unexpected error"):
            _drive_batch_queue_core(core)
        return
    _drive_batch_queue_core(core)
    core.scheduler.update_from_output.assert_called_once_with(scheduled, output)


def test_async_step_execute_failure_surfaces():
    """A failed execute_model() must surface its own exception."""
    core = _make_batch_queue_core()
    executed: Future[ModelRunnerOutput | None] = Future()
    executed.set_exception(RuntimeError("worker boom"))
    sampled: Future[ModelRunnerOutput | None] = Future()
    sampled.set_result(None)
    core.model_executor.execute_model.return_value = executed
    core.model_executor.sample_tokens.return_value = sampled

    with pytest.raises(RuntimeError, match="worker boom"):
        _drive_batch_queue_core(core)


def test_async_step_consecutive_sampleless_steps():
    """Consecutive sampleless steps must each deliver their own output."""
    outputs = [MagicMock() for _ in range(2)]
    scheduled_steps = []
    for _ in outputs:
        scheduled = MagicMock()
        scheduled.total_num_scheduled_tokens = 16
        scheduled.pending_structured_output_tokens = False
        scheduled.has_structured_output_spec_decode = False
        scheduled_steps.append(scheduled)
    core = _make_batch_queue_core()
    core.scheduler.has_requests.side_effect = [True, True, False, False]
    core.scheduler.schedule.side_effect = scheduled_steps
    executed: list[Future[ModelRunnerOutput | None]] = [Future() for _ in outputs]
    sampled: list[Future[ModelRunnerOutput | None]] = [Future() for _ in outputs]
    for i, output in enumerate(outputs):
        executed[i].set_result(output)
        sampled[i].set_result(None)
    core.model_executor.execute_model.side_effect = executed
    core.model_executor.sample_tokens.side_effect = sampled

    EngineCore.step_with_batch_queue(core)
    EngineCore.step_with_batch_queue(core)
    EngineCore.step_with_batch_queue(core)
    calls = core.scheduler.update_from_output.call_args_list
    assert len(calls) == 2
    assert calls[0].args == (scheduled_steps[0], outputs[0])
    assert calls[1].args == (scheduled_steps[1], outputs[1])
