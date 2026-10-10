# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch queue (async scheduling): an execute_model() failure reaches the engine.

After a failed execute_model() the output rank's sample_tokens() returns an empty
ModelRunnerOutput (not None), so the engine must raise execute_model()'s exception
instead of passing the empty output to the scheduler (KeyError on the request ids).
"""

import contextlib
from collections import deque
from types import SimpleNamespace as NS

import pytest

from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.multiproc_executor import FutureWrapper
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT


def _engine(exec_error: Exception | None):
    futures: deque = deque()
    updates: list = []

    def exec_response():
        if exec_error is not None:
            raise exec_error

    executor = NS(
        execute_model=lambda so, non_block: FutureWrapper(futures, exec_response),
        sample_tokens=lambda go, non_block: FutureWrapper(
            futures, lambda: EMPTY_MODEL_RUNNER_OUTPUT
        ),
    )
    pending = [True]

    def has_requests():
        return pending.pop() if pending else False

    def update_from_output(so, out):
        updates.append(out)
        return {}

    scheduler = NS(
        has_requests=has_requests,
        schedule=lambda throttle: NS(
            total_num_scheduled_tokens=1, pending_structured_output_tokens=False
        ),
        get_grammar_bitmask=lambda so: None,
        update_from_output=update_from_output,
    )
    core = EngineCore.__new__(EngineCore)
    core.__dict__.update(
        batch_queue=deque(),
        batch_queue_size=2,
        scheduler=scheduler,
        model_executor=executor,
        is_mm_encoder_only=False,
        is_pooling_model=False,
        _should_throttle_prefills=lambda: False,
        log_error_detail=lambda so: contextlib.nullcontext(),
        capture_iteration_details=lambda so: contextlib.nullcontext(),
        _process_aborts_queue=lambda: None,
        _attach_iteration_details=lambda outs, details: None,
    )
    return core, updates


def test_execute_model_error_is_raised():
    core, updates = _engine(RuntimeError("Worker failed with error 'boom'"))
    assert core.step_with_batch_queue() == (None, True)  # queued, not awaited
    with pytest.raises(RuntimeError, match="boom"):
        core.step_with_batch_queue()
    assert updates == []


def test_execute_model_success_unchanged():
    core, updates = _engine(None)
    assert core.step_with_batch_queue() == (None, True)
    core.step_with_batch_queue()
    assert updates == [EMPTY_MODEL_RUNNER_OUTPUT]
