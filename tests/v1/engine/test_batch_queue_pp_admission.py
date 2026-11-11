# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.core import EngineCore


@pytest.mark.parametrize(
    "pp_size,num_tokens,is_ec_consumer,idle",
    [
        (4, 0, True, True),  # PP: empty step -> stop scheduling, wait for arrivals
        (1, 0, True, False),  # no PP: upstream loop
        (4, 100, False, False),  # step with tokens is never idle (EC producer)
    ],
)
def test_idle_step_admission(pp_size, num_tokens, is_ec_consumer, idle):
    core = object.__new__(EngineCore)
    core.batch_queue_size = pp_size + 1
    core.batch_queue = deque(maxlen=core.batch_queue_size)
    core._idle_step_in_flight = False
    core._pp_admission = pp_size > 1
    core.is_ec_consumer = is_ec_consumer
    core.is_mm_encoder_only = False
    core.is_pooling_model = False
    core.num_executed_steps = 0
    core._should_throttle_prefills = lambda: False
    core.deferred_step = None
    core.log_error_detail = lambda _: contextlib.nullcontext()
    core.scheduler = MagicMock()
    core.scheduler.has_decode_awaiting_cadence.return_value = False
    core.scheduler.schedule.return_value = SimpleNamespace(
        total_num_scheduled_tokens=num_tokens,
        pending_structured_output_tokens=False,
        has_structured_output_spec_decode=False,
    )
    core.model_executor = MagicMock()
    core.model_executor.execute_model.return_value = Future()
    core.model_executor.sample_tokens.return_value = Future()
    # A request arrives while waiting on the oldest batch.
    core._wait_for_new_request = MagicMock(return_value=True)

    assert core.step_with_batch_queue() == (None, num_tokens > 0)
    assert core._idle_step_in_flight == idle
    assert core._wait_for_new_request.called == idle
    assert len(core.batch_queue) == 1
    if idle:
        # While the idle step is in flight, no second empty step is scheduled.
        assert core.step_with_batch_queue() == (None, False)
        assert core.scheduler.schedule.call_count == 1
