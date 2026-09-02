# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import DPEngineCoreProc

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize(
    ("enabled", "scheduler_will_call_worker", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_scheduler_target_generation_ownership(
    enabled: bool,
    scheduler_will_call_worker: bool,
    expected: bool,
) -> None:
    core = object.__new__(DPEngineCoreProc)
    core.dp_execution_contract_enabled = enabled

    assert (
        core._scheduler_owns_target_generation(scheduler_will_call_worker) is expected
    )


def _make_async_batch_queue_core() -> DPEngineCoreProc:
    core = object.__new__(DPEngineCoreProc)
    core.model_executor = Mock()
    core.scheduler = Mock()
    core.batch_queue_size = 2
    core.batch_queue = deque(maxlen=core.batch_queue_size)
    core.prefill_schedule_interval = 1
    core.step_counter = 0
    core.is_ec_consumer = True
    core.is_pooling_model = False
    core.check_for_draft_tokens = False
    core.log_error_detail = Mock(return_value=nullcontext())
    core.capture_iteration_details = Mock(return_value=nullcontext(None))
    core._process_aborts_queue = Mock()
    core._attach_iteration_details = Mock()
    core.output_queue = Mock()
    core.post_step = Mock()
    core.step_fn = core.step_with_batch_queue
    return core


def test_async_engine_step_records_zero_token_worker_launch() -> None:
    core = _make_async_batch_queue_core()
    scheduler_output = SchedulerOutput.make_empty()
    core.scheduler.has_requests.side_effect = [True, True, False]
    core.scheduler.schedule.return_value = scheduler_output

    model_executed, worker_call_issued = core._process_engine_step()

    assert not model_executed
    assert worker_call_issued
    assert len(core.batch_queue) == 1
    core.model_executor.execute_model.assert_called_once_with(
        scheduler_output, non_block=True
    )


def test_async_engine_step_does_not_count_result_retirement_as_worker_launch() -> None:
    core = _make_async_batch_queue_core()
    core._worker_call_issued_in_step = True
    core.scheduler.has_requests.return_value = False
    scheduler_output = SchedulerOutput.make_empty()
    output_future = Mock()
    output_future.result.return_value = object()
    core.batch_queue.append((output_future, scheduler_output, Mock()))
    core.scheduler.update_from_output.return_value = {}

    model_executed, worker_call_issued = core._process_engine_step()

    assert not model_executed
    assert not worker_call_issued
    assert not core.batch_queue
    core.model_executor.execute_model.assert_not_called()
