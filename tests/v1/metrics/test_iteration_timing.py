# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.metrics.iteration_timing import (
    IterationTimer,
    is_iteration_output_rank,
    make_iteration_timer,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_ready_events_are_reused_without_waiting_for_pending_or_contended_drains():
    events = [Mock(), Mock()]
    events[0].elapsed_time.return_value = 7.0
    events[1].query.return_value = False
    timer = IterationTimer(1, event_factory=iter(events).__next__)
    timer.start(4)
    timer.finish()

    assert timer.drain_samples() == ()
    events[0].elapsed_time.assert_not_called()
    events[1].query.return_value = True
    with timer._drain_lock:
        assert timer.drain_samples() == ()
    with timer._pending_lock:
        assert timer.drain_samples() == ()
    assert timer.drain_samples() == ((4, 0.007),)

    # The factory is exhausted; the next interval must reuse the ready pair.
    timer.start(5)
    timer.finish()
    assert timer.drain_samples() == ((5, 0.007),)
    for event in events:
        assert event.record.call_count == 2
        event.synchronize.assert_not_called()


def test_later_ready_event_does_not_overtake_pending_iteration():
    events = [Mock() for _ in range(4)]
    for start in events[::2]:
        start.elapsed_time.return_value = 1.0
    events[1].query.return_value = False
    events[3].query.return_value = True
    timer = IterationTimer(2, event_factory=iter(events).__next__)
    for iteration_id in (7, 8):
        timer.start(iteration_id)
        timer.finish()
    assert timer.drain_samples() == ()
    events[3].query.assert_not_called()
    events[1].query.return_value = True
    assert timer.drain_samples() == ((7, 0.001), (8, 0.001))
    assert timer.drain_samples() == ()


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
    assert output.iteration_timing_samples == ((9, 0.001),)
    assert EMPTY_MODEL_RUNNER_OUTPUT.iteration_timing_samples == ()


def test_cancel_discards_inflight_pair_instead_of_reusing_it():
    events = [Mock(query=Mock(return_value=True)) for _ in range(4)]
    events[2].elapsed_time.return_value = 2.0
    timer = IterationTimer(1, event_factory=iter(events).__next__)
    timer.start(1)
    timer.cancel()
    assert timer.drain_samples() == ()
    timer.start(2)
    timer.finish()
    assert timer.drain_samples() == ((2, 0.002),)
    events[0].record.assert_called_once()
    events[1].record.assert_not_called()


@pytest.mark.parametrize("enabled,output_rank", [(False, True), (True, False)])
def test_disabled_and_non_output_rank_allocate_no_events(
    monkeypatch, enabled, output_rank
):
    factory = Mock(side_effect=AssertionError("unexpected timing event allocation"))
    monkeypatch.setattr(IterationTimer, "_make_event", factory)
    config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_iteration_timing=enabled)
    )
    assert make_iteration_timer(config, is_output_rank=output_rank) is None
    factory.assert_not_called()


@pytest.mark.parametrize("tp,pcp", [(1, 1), (4, 1), (2, 2)])
def test_only_one_worker_per_replica_allocates_timing_events(monkeypatch, tp, pcp):
    factory = Mock(return_value=Mock())
    monkeypatch.setattr(IterationTimer, "_make_event", factory)
    config = SimpleNamespace(
        observability_config=SimpleNamespace(enable_iteration_timing=True),
        max_concurrent_batches=2,
        parallel_config=SimpleNamespace(
            world_size=tp * pcp,
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
        ),
    )
    timers = [
        make_iteration_timer(config, is_iteration_output_rank(config, rank))
        for rank in range(tp * pcp)
    ]
    assert sum(timer is not None for timer in timers) == 1
    assert timers[0] is not None
    assert factory.call_count == 6
    factory.return_value.record.assert_not_called()
