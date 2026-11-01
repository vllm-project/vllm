# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.metrics.iteration_timing import (
    IterationTimer,
    make_iteration_timer,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_ready_events_are_reused_without_waiting_for_pending_or_contended_drains():
    events = [Mock() for _ in range(4)]
    events[0].elapsed_time.return_value = 7.0
    events[1].query.return_value = False
    events[2].elapsed_time.return_value = 8.0
    events[3].query.return_value = True
    timer = IterationTimer(2, event_factory=iter(events).__next__)
    timer.start(4)
    timer.finish()

    assert timer.drain_samples() == ()
    events[0].elapsed_time.assert_not_called()
    events[1].query.return_value = True
    with timer._drain_lock:
        assert timer.drain_samples() == ()

    def append_during_query():
        events[1].query.side_effect = None
        timer.start(5)
        timer.finish()
        return True

    events[1].query.side_effect = append_during_query
    assert timer.drain_samples() == ((4, 7.0), (5, 8.0))

    # The factory is exhausted; the next intervals must reuse the ready pairs.
    timer.start(6)
    timer.finish()
    timer.start(7)
    timer.finish()
    assert timer.drain_samples() == ((6, 7.0), (7, 8.0))
    for event in events:
        assert event.record.call_count == 2
        event.synchronize.assert_not_called()


def test_empty_step_is_not_timed_and_shared_output_is_not_mutated():
    events = [Mock(query=Mock(return_value=True)) for _ in range(2)]
    events[0].elapsed_time.return_value = 1.0
    timer = IterationTimer(1, event_factory=iter(events).__next__)
    timer.start(None)
    timer.finish()
    assert timer.drain_into(EMPTY_MODEL_RUNNER_OUTPUT) is EMPTY_MODEL_RUNNER_OUTPUT
    for event in events:
        event.record.assert_not_called()
    timer.start(9)
    timer.finish()
    output = timer.drain_into(EMPTY_MODEL_RUNNER_OUTPUT)
    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert output.iteration_timing_samples == ((9, 1.0),)
    assert EMPTY_MODEL_RUNNER_OUTPUT.iteration_timing_samples == ()


def test_disabled_timing_allocates_no_events(monkeypatch):
    factory = Mock(side_effect=AssertionError("unexpected timing event allocation"))
    monkeypatch.setattr(IterationTimer, "_make_event", factory)
    config = SimpleNamespace(
        observability_config=SimpleNamespace(iteration_timing_enabled=False)
    )
    assert make_iteration_timer(config, rank=0) is None
    factory.assert_not_called()


def test_only_one_worker_per_replica_allocates_timing_events(monkeypatch):
    factory = Mock(return_value=Mock())
    monkeypatch.setattr(IterationTimer, "_make_event", factory)
    config = SimpleNamespace(
        observability_config=SimpleNamespace(iteration_timing_enabled=True),
        max_concurrent_batches=2,
        parallel_config=SimpleNamespace(
            world_size=8,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
        ),
    )
    timers = [make_iteration_timer(config, rank) for rank in range(8)]
    assert sum(timer is not None for timer in timers) == 1
    assert timers[4] is not None
    assert factory.call_count == 6
    factory.return_value.record.assert_not_called()
