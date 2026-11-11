# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import time
from collections import deque
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats


class FakeEngineCore:
    def _make_iteration_details_stats(
        self, iteration_details: SchedulerIterationDetails
    ) -> SchedulerStats:
        return SchedulerStats(iteration_details=iteration_details)


def make_iteration_details() -> SchedulerIterationDetails:
    return SchedulerIterationDetails(
        iteration_index=1,
        num_ctx_requests=2,
        num_ctx_tokens=3,
        num_generation_requests=4,
        num_generation_tokens=5,
        elapsed_ms=6.7,
    )


def make_fake_engine(
    log_stats: bool = True, gpu_iteration_timing: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        log_stats=log_stats,
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(
                enable_logging_iteration_details=True,
            ),
            gpu_iteration_timing=gpu_iteration_timing,
        ),
    )


def test_capture_iteration_details_disabled_without_log_stats():
    engine = make_fake_engine(log_stats=False)

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is None

    assert not hasattr(engine, "_iteration_index")


def test_capture_iteration_details_fills_elapsed_time():
    engine = make_fake_engine()

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is not None
        assert iteration_details.elapsed_ms == 0.0
        assert iteration_details.is_dummy
        time.sleep(0.001)

    assert iteration_details is not None
    assert iteration_details.elapsed_ms > 0.0
    assert engine._iteration_index == 1


def test_attach_iteration_details_uses_existing_output():
    iteration_details = make_iteration_details()
    outputs = {
        2: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
        1: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
    }

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert 0 not in outputs
    assert outputs[2].scheduler_stats is not None
    assert outputs[2].scheduler_stats.iteration_details == iteration_details
    assert outputs[1].scheduler_stats is not None
    assert outputs[1].scheduler_stats.iteration_details is None


def test_attach_iteration_details_falls_back_to_client_zero_without_outputs():
    iteration_details = make_iteration_details()
    outputs: dict[int, EngineCoreOutputs] = {}

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert set(outputs) == {0}
    assert outputs[0].scheduler_stats is not None
    assert outputs[0].scheduler_stats.iteration_details == iteration_details


def test_capture_iteration_details_defers_to_gpu_timing():
    engine = make_fake_engine(gpu_iteration_timing=True)

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is None

    assert engine._iteration_timings_pending


def make_idle_engine(timings_pending: bool) -> SimpleNamespace:
    """An EngineCoreProc that has run out of work, as far as going idle needs."""
    engine = SimpleNamespace(
        _iteration_timings_pending=timings_pending,
        _deferred_inputs=deque(),
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        aborts_queue=queue.Queue(),
        process_input_queue_block=True,
        model_executor=Mock(),
        scheduler=SimpleNamespace(make_stats=SchedulerStats),
        handled=[],
        is_running=lambda: True,
        _notify_idle_state_callbacks=lambda: None,
    )
    engine.has_work = lambda: ("add", "request") in engine.handled
    engine._handle_client_request = lambda *request: engine.handled.append(request)
    for method in ("_flush_iteration_timings", "_publish_iteration_records"):
        setattr(engine, method, partial(getattr(EngineCoreProc, method), engine))
    return engine


def test_going_idle_collects_the_last_gpu_timings_then_takes_new_work():
    record = make_iteration_details()
    engine = make_idle_engine(timings_pending=True)
    # Deferred during the last step; it could put the executor to sleep.
    engine._deferred_inputs.append(("utility", "sleep"))

    def flush():
        assert not engine.handled  # the flush precedes every other input
        # A request arrives while the engine waits for the GPU.
        engine.input_queue.put(("add", "request"))
        return [record]

    engine.model_executor.flush_iteration_timings.side_effect = flush

    EngineCoreProc._process_input_queue(engine)

    client_index, outputs = engine.output_queue.get_nowait()
    assert client_index == 0
    assert outputs.scheduler_stats.iteration_records == [record]
    assert engine.handled == [("utility", "sleep"), ("add", "request")]
    assert not engine._iteration_timings_pending


def test_going_idle_without_pending_gpu_timings_sends_no_flush():
    engine = make_idle_engine(timings_pending=False)
    engine.input_queue.put(("add", "request"))

    EngineCoreProc._process_input_queue(engine)

    engine.model_executor.flush_iteration_timings.assert_not_called()
    assert engine.output_queue.empty()
