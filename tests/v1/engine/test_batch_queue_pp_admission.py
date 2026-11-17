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
    "pp_size,num_tokens,awaiting_cadence,waits",
    [
        (4, 0, False, True),  # PP: empty step nobody needs -> wait for arrivals
        (4, 0, True, False),  # PP: decode waiting on the cadence -> keep stepping
        (1, 0, False, False),  # no PP: upstream loop
        (4, 100, False, False),  # step with tokens: keep filling the queue
    ],
)
def test_empty_step_admission(pp_size, num_tokens, awaiting_cadence, waits):
    core = object.__new__(EngineCore)
    core.batch_queue_size = pp_size + 1
    core.batch_queue = deque(maxlen=core.batch_queue_size)
    core._pp_admission = pp_size > 1
    core.is_mm_encoder_only = False
    core.is_pooling_model = False
    core.num_executed_steps = 0
    core._should_throttle_prefills = lambda: False
    core.deferred_step = None
    core.log_error_detail = lambda _: contextlib.nullcontext()
    core.scheduler = MagicMock()
    core.scheduler.has_decode_awaiting_cadence.return_value = awaiting_cadence
    core.scheduler.schedule.return_value = SimpleNamespace(
        total_num_scheduled_tokens=num_tokens,
        pending_structured_output_tokens=False,
        has_structured_output_spec_decode=False,
    )
    core.model_executor = MagicMock()
    core.model_executor.execute_model.return_value = Future()
    core.model_executor.sample_tokens.return_value = Future()
    # A request arrives while waiting on the oldest batch.
    core._handle_inputs_while_waiting = MagicMock(return_value=True)

    assert core.step_with_batch_queue() == (None, num_tokens > 0)
    assert len(core.batch_queue) == 1
    if waits:
        core._handle_inputs_while_waiting.assert_called_once_with(
            core.batch_queue[-1][0], stop_on_add=True
        )
    else:
        core._handle_inputs_while_waiting.assert_not_called()
