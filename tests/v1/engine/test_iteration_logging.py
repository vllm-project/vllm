# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import time
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.engine import core as engine_core
from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
from vllm.v1.executor.multiproc_executor import FutureWrapper
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput


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


class InputQueue(queue.Queue):
    """An input queue the engine cannot block on: left with nothing to do, it
    returns to the test instead of hanging."""

    def get(self, block=True, timeout=None):
        return super().get(block=False)


def make_idle_engine(timings_pending: bool) -> Any:
    """An EngineCoreProc that has run out of work, as far as going idle needs."""
    engine: Any = object.__new__(EngineCoreProc)
    engine._iteration_timings_pending = timings_pending
    engine._iteration_flush = None
    engine._idle_state_callbacks = []
    engine._deferred_inputs = deque()
    engine.input_queue = InputQueue()
    engine.output_queue = queue.Queue()
    engine.aborts_queue = queue.Queue()
    engine.process_input_queue_block = True
    engine.shutdown_state = EngineShutdownState.RUNNING
    engine.model_executor = Mock()
    engine.scheduler = SimpleNamespace(make_stats=SchedulerStats)
    engine.handled = []
    new_work = (EngineCoreRequestType.ADD, EngineCoreRequestType.START_DP_WAVE)
    engine.has_work = lambda: any(request[0] in new_work for request in engine.handled)
    engine._handle_client_request = lambda *request: engine.handled.append(request)
    return engine


def flush_output(record: SchedulerIterationDetails) -> ModelRunnerOutput:
    return ModelRunnerOutput(req_ids=[], iteration_records=[record])


class FakeWorker:
    """The worker of a MultiprocExecutor, whose replies the test supplies."""

    def __init__(self):
        self.futures: deque[FutureWrapper] = deque()
        self.replies: deque = deque()
        # What happens during each of the engine's brief waits, in order.
        self.while_waiting: list = []

    def rpc(self) -> FutureWrapper:
        return FutureWrapper(
            self.futures,
            get_response=self.replies.popleft,
            response_ready=lambda: bool(self.replies),
            wait_for_response=lambda pending, deadline: self.while_waiting.pop(0)(),
        )


def test_going_idle_collects_the_last_gpu_timings_then_takes_new_work():
    record = make_iteration_details()
    engine = make_idle_engine(timings_pending=True)
    # Deferred during the last step; it could put the executor to sleep.
    engine._deferred_inputs.append((EngineCoreRequestType.UTILITY, "sleep"))
    request = (EngineCoreRequestType.ADD, "request")

    def flush():
        assert not engine.handled  # the flush precedes every other input
        # A request arrives while the executor waits for the GPU in place.
        engine.input_queue.put(request)
        future = Future[ModelRunnerOutput]()
        future.set_result(flush_output(record))
        return future

    engine.model_executor.flush_iteration_timings.side_effect = flush

    engine._process_input_queue()

    client_index, outputs = engine.output_queue.get_nowait()
    assert client_index == 0
    assert outputs.scheduler_stats.iteration_records == [record]
    assert engine.handled == [(EngineCoreRequestType.UTILITY, "sleep"), request]
    assert not engine._iteration_timings_pending


def test_idle_engine_publishes_the_last_gpu_timings_once_the_gpu_finishes():
    record = make_iteration_details()
    engine = make_idle_engine(timings_pending=True)
    worker = FakeWorker()
    engine.model_executor.flush_iteration_timings.side_effect = worker.rpc

    def gpu_finishes():
        assert engine.output_queue.empty()
        worker.replies.append(flush_output(record))
        engine.input_queue.put((EngineCoreRequestType.ADD, "request"))

    worker.while_waiting = [gpu_finishes]

    engine._process_input_queue()

    _, outputs = engine.output_queue.get_nowait()
    assert outputs.scheduler_stats.iteration_records == [record]
    assert engine.output_queue.empty()
    assert engine.handled == [(EngineCoreRequestType.ADD, "request")]


@pytest.mark.parametrize(
    "new_work", [EngineCoreRequestType.ADD, EngineCoreRequestType.START_DP_WAVE]
)
def test_new_work_arriving_during_the_flush_does_not_wait_for_the_gpu(new_work):
    record = make_iteration_details()
    engine = make_idle_engine(timings_pending=True)
    worker = FakeWorker()
    engine.model_executor.flush_iteration_timings.side_effect = worker.rpc
    worker.while_waiting = [lambda: engine.input_queue.put((new_work, "request"))]

    engine._process_input_queue()

    assert engine.handled == [(new_work, "request")]
    assert engine.output_queue.empty()

    # The flush resolves, in order, while the engine waits for the next step.
    step = worker.rpc()
    worker.replies.extend([flush_output(record), "step output"])
    assert engine._wait_for_output(step) == "step output"
    _, outputs = engine.output_queue.get_nowait()
    assert outputs.scheduler_stats.iteration_records == [record]
    assert engine.output_queue.empty()


def test_input_that_starts_no_work_waits_for_the_flush():
    record = make_iteration_details()
    engine = make_idle_engine(timings_pending=True)
    engine.model_executor.flush_iteration_timings.side_effect = lambda: FutureWrapper(
        deque(),
        get_response=lambda: flush_output(record),  # the blocking read
        response_ready=lambda: False,
        wait_for_response=Mock(side_effect=AssertionError("still spinning")),
    )
    engine.input_queue.put((EngineCoreRequestType.UTILITY, "sleep"))
    flushed_first = []
    engine._handle_client_request = lambda *request: flushed_first.append(
        not engine.output_queue.empty()
    )

    engine._process_input_queue()

    assert flushed_first == [True]
    _, outputs = engine.output_queue.get_nowait()
    assert outputs.scheduler_stats.iteration_records == [record]


@pytest.mark.parametrize(
    ("waiting", "readable"),
    [
        # Nothing arrives within the spin budget: the engine blocks for the flush.
        ([], False),
        # poll() reads the failure; the worker monitor's own message stays queued.
        ([(EngineCoreRequestType.EXECUTOR_FAILED, b"")], True),
    ],
)
def test_flush_failing_while_waited_for_fails_the_engine_first(
    monkeypatch, waiting, readable
):
    monkeypatch.setattr(engine_core, "_INPUT_SPIN_DURATION_S", 0.0)
    engine = make_idle_engine(timings_pending=True)

    def blocking_read():
        raise TimeoutError("RPC call to flush_iteration_timings timed out.")

    engine.model_executor.flush_iteration_timings.side_effect = lambda: FutureWrapper(
        deque(),
        get_response=blocking_read,
        response_ready=lambda: readable,
        wait_for_response=Mock(side_effect=AssertionError("still spinning")),
    )
    for request in waiting:
        engine.input_queue.put(request)

    # The failure stops the engine before anything else runs.
    with pytest.raises(TimeoutError):
        engine._process_input_queue()

    assert not engine.handled
    assert list(engine.input_queue.queue) == waiting  # and is reported once


def test_new_work_does_not_end_the_wait_while_a_pause_waits_for_idle():
    record = make_iteration_details()
    engine = make_idle_engine(timings_pending=True)
    engine.model_executor.flush_iteration_timings.side_effect = lambda: FutureWrapper(
        deque(),
        get_response=lambda: flush_output(record),  # the blocking read
        response_ready=lambda: False,
    )
    flushed_first = []
    engine._idle_state_callbacks.append(
        lambda engine: flushed_first.append(not engine.output_queue.empty())
    )
    engine.input_queue.put((EngineCoreRequestType.ADD, "request"))

    engine._process_input_queue()

    # The pause's own RPC has no timeout: it must not queue behind the flush.
    assert flushed_first == [True]


def test_flush_failing_during_later_work_fails_the_engine_like_a_failed_executor():
    engine = make_idle_engine(timings_pending=False)
    failed = Future[ModelRunnerOutput]()
    failed.set_exception(RuntimeError("Worker failed"))

    engine._publish_iteration_flush(failed)

    assert engine.input_queue.get_nowait()[0] == EngineCoreRequestType.EXECUTOR_FAILED
    assert engine.output_queue.empty()


def test_dummy_batch_publishes_the_gpu_timings_it_returns():
    record = make_iteration_details()
    engine = make_idle_engine(timings_pending=False)
    engine.model_executor.execute_dummy_batch.return_value = [record]

    engine.execute_dummy_batch()

    _, outputs = engine.output_queue.get_nowait()
    assert outputs.scheduler_stats.iteration_records == [record]


def test_going_idle_without_pending_gpu_timings_sends_no_flush():
    engine = make_idle_engine(timings_pending=False)
    engine.input_queue.put((EngineCoreRequestType.ADD, "request"))

    engine._process_input_queue()

    engine.model_executor.flush_iteration_timings.assert_not_called()
    assert engine.output_queue.empty()
