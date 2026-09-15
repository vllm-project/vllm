# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


def test_batch_queue_propagates_execute_model_error() -> None:
    execution_error = RuntimeError("model execution failed")
    execute_future: Future[ModelRunnerOutput | None] = Future()
    execute_future.set_exception(execution_error)
    sample_future: Future[ModelRunnerOutput] = Future()
    sample_future.set_result(EMPTY_MODEL_RUNNER_OUTPUT)

    scheduler_output = MagicMock()
    engine_core = MagicMock()
    engine_core.batch_queue_size = 2
    engine_core.batch_queue = deque([(sample_future, scheduler_output, execute_future)])
    engine_core.scheduler.has_requests.return_value = False
    engine_core.capture_iteration_details.return_value = nullcontext(None)
    engine_core.log_error_detail.return_value = nullcontext()

    with pytest.raises(RuntimeError, match="model execution failed") as exc_info:
        EngineCore.step_with_batch_queue(engine_core)

    assert exc_info.value is execution_error


def test_batch_queue_resolves_shared_execution_future_once() -> None:
    shared_future = MagicMock()
    shared_future.result.return_value = EMPTY_MODEL_RUNNER_OUTPUT

    scheduler_output = MagicMock()
    engine_core = MagicMock()
    engine_core.batch_queue_size = 2
    engine_core.batch_queue = deque([(shared_future, scheduler_output, shared_future)])
    engine_core.scheduler.has_requests.return_value = False
    engine_core.capture_iteration_details.return_value = nullcontext(None)
    engine_core.log_error_detail.return_value = nullcontext()

    EngineCore.step_with_batch_queue(engine_core)

    shared_future.result.assert_called_once_with()
