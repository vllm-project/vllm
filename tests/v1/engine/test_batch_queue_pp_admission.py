# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from concurrent.futures import Future
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.core import EngineCore


@pytest.mark.parametrize(
    "pp_admission,num_tokens,should_schedule,waits",
    [
        (True, 0, False, True),  # PP: scheduler idle -> wait for arrivals
        (True, 0, True, False),  # PP: scheduler wants more steps -> keep stepping
        (False, 0, False, False),  # no PP: upstream loop
        (True, 100, False, False),  # step with tokens: keep filling the queue
    ],
)
def test_free_slot_admission(pp_admission, num_tokens, should_schedule, waits):
    core = MagicMock()
    core.batch_queue_size = 5
    core.batch_queue = deque(maxlen=core.batch_queue_size)
    core._pp_admission = pp_admission
    core.is_mm_encoder_only = False
    core.is_pooling_model = False
    core.num_executed_steps = 0
    core._should_throttle_prefills.return_value = False
    core.deferred_step = None
    core._wait_for_new_request = partial(EngineCore._wait_for_new_request, core)
    core.scheduler.should_schedule.side_effect = [True, should_schedule]
    core.scheduler.schedule.return_value = SimpleNamespace(
        total_num_scheduled_tokens=num_tokens,
        pending_structured_output_tokens=False,
        has_structured_output_spec_decode=False,
    )
    core.model_executor.execute_model.return_value = Future()
    core.model_executor.sample_tokens.return_value = Future()
    # A request arrives while waiting on the oldest batch.
    core._handle_inputs_while_waiting.return_value = True
    core._wait_for_output.side_effect = RuntimeError("blocked")

    if not (waits or should_schedule or num_tokens):
        # Without PP admission, block on the oldest batch.
        with pytest.raises(RuntimeError, match="blocked"):
            EngineCore.step_with_batch_queue(core)
        return
    assert EngineCore.step_with_batch_queue(core) == (None, num_tokens > 0)
    assert len(core.batch_queue) == 1
    if waits:
        core._handle_inputs_while_waiting.assert_called_once_with(
            core.batch_queue[-1][0], stop_on_add=True
        )
    else:
        core._handle_inputs_while_waiting.assert_not_called()
