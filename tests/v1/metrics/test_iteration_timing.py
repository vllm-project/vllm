# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.v1.metrics.iteration_timing as iteration_timing
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import IterationDetails
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.metrics import loggers
from vllm.v1.metrics.iteration_timing import (
    IterationTimer,
    finish_iteration_timing,
    make_iteration_timer,
    start_iteration_timing,
    time_dummy_batch,
)
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


class FakeGpu:
    """A CUDA stream whose progress the test controls."""

    def __init__(self):
        self.now_ms = 0.0
        self.events: list[FakeEvent] = []
        self.waits = 0

    def run(self, ms: float = 1.0):
        """Complete everything recorded so far, one event every `ms`."""
        for event in self.events:
            self.now_ms += ms
            event.completed_ms = self.now_ms
        self.events.clear()


class FakeEvent:
    gpu: FakeGpu

    def __init__(self, enable_timing=False, blocking=False):
        self.completed_ms = None

    def record(self):
        self.gpu.events.append(self)

    def query(self):
        return self.completed_ms is not None

    def synchronize(self):
        self.gpu.waits += 1
        self.gpu.run()

    def elapsed_time(self, end):
        if self.completed_ms is None or end.completed_ms is None:
            raise RuntimeError("Both events must be completed")
        return end.completed_ms - self.completed_ms


class FakeAsyncOutput(AsyncModelRunnerOutput):
    def get_output(self):
        return ModelRunnerOutput(req_ids=[])

    def get_kv_connector_output(self):
        return "kv output"


class FakeWorker:
    def __init__(self):
        self.iteration_timer = IterationTimer()
        self.fail_next_step = False

    @start_iteration_timing
    def execute_model(self, scheduler_output):
        return None

    @finish_iteration_timing
    def sample_tokens(self):
        if self.fail_next_step:
            self.fail_next_step = False
            return None  # as when execute_model failed
        return FakeAsyncOutput()

    @time_dummy_batch
    def execute_dummy_batch(self):
        pass

    def step(self, num_tokens=1, timed=True):
        """Run one iteration and return the records its output carries."""
        details = IterationDetails(num_decode_tokens=num_tokens) if timed else None
        self.execute_model(SimpleNamespace(iteration_details=details))
        output = self.sample_tokens()
        return output and output.get_output().iteration_records

    def flush(self):
        return self.iteration_timer.flush().get_output().iteration_records


@pytest.fixture
def gpu(monkeypatch):
    gpu = FakeEvent.gpu = FakeGpu()
    fake_torch = SimpleNamespace(Event=FakeEvent, cuda=SimpleNamespace(Event=FakeEvent))
    monkeypatch.setattr(iteration_timing, "torch", fake_torch)
    return gpu


def test_record_rides_a_later_output_once_its_gpu_work_completes(gpu):
    worker = FakeWorker()

    # The step's own end event follows its output, so it is still pending.
    assert worker.step(num_tokens=3) == []
    assert worker.step(num_tokens=5) == []
    gpu.run(ms=4.0)

    first, second = worker.step()
    assert (first.iteration_index, first.num_generation_tokens) == (0, 3)
    assert (second.iteration_index, second.num_generation_tokens) == (1, 5)
    assert first.elapsed_ms == 4.0 and not first.is_dummy
    assert gpu.waits == 0


def test_flush_waits_for_the_last_iteration_when_the_engine_goes_idle(gpu):
    worker = FakeWorker()
    assert worker.step(num_tokens=7) == []

    (record,) = worker.flush()

    assert record.num_generation_tokens == 7
    assert gpu.waits == 1
    assert worker.flush() == []


def test_dummy_batches_return_earlier_records_and_flush_collects_the_tail(gpu):
    worker = FakeWorker()

    assert worker.execute_dummy_batch() == []
    gpu.run()
    assert [record.is_dummy for record in worker.execute_dummy_batch()] == [True]
    assert worker.execute_dummy_batch() == []

    assert [record.iteration_index for record in worker.flush()] == [1, 2]


def test_only_iterations_that_finish_with_a_snapshot_are_timed(gpu):
    worker = FakeWorker()

    worker.fail_next_step = True
    assert worker.step() is None  # execute_model failed
    assert worker.step(timed=False) == []  # no scheduling snapshot
    # A step that schedules no tokens has no sample_tokens call.
    worker.execute_model(SimpleNamespace(iteration_details=IterationDetails()))
    assert worker.flush() == []

    # None of these stops later iterations from being timed.
    worker.step(num_tokens=2)
    assert [record.num_generation_tokens for record in worker.flush()] == [2]


def test_failed_timing_read_drops_the_record_not_the_step(gpu, monkeypatch):
    worker = FakeWorker()
    worker.step()
    gpu.run()

    def broken_read(self, end):
        raise RuntimeError("CUDA error")

    with monkeypatch.context() as patch:
        patch.setattr(FakeEvent, "elapsed_time", broken_read)
        assert worker.step() == []

    # Later records are unaffected.
    assert [record.iteration_index for record in worker.flush()] == [1]


def broken_event_call(self):
    raise RuntimeError("CUDA error")


def test_failed_event_recording_drops_the_record_not_the_step(gpu, monkeypatch):
    worker = FakeWorker()
    warning = Mock()
    monkeypatch.setattr(iteration_timing.logger, "warning_once", warning)
    scheduler_output = SimpleNamespace(iteration_details=IterationDetails())

    with monkeypatch.context() as patch:
        patch.setattr(FakeEvent, "record", broken_event_call)
        worker.execute_model(scheduler_output)  # no start event
    assert isinstance(worker.sample_tokens(), FakeAsyncOutput)

    worker.execute_model(scheduler_output)
    with monkeypatch.context() as patch:
        patch.setattr(FakeEvent, "record", broken_event_call)
        assert isinstance(worker.sample_tokens(), FakeAsyncOutput)  # no end event
        assert worker.execute_dummy_batch() == []  # neither event
        assert worker.flush() == []

    # Later iterations are timed again; the three dropped ones keep their index.
    assert warning.called
    worker.step(num_tokens=2)
    (record,) = worker.flush()
    assert (record.iteration_index, record.num_generation_tokens) == (3, 2)


def test_failed_flush_wait_drops_the_tail_not_the_flush(gpu, monkeypatch):
    worker = FakeWorker()
    worker.step()

    monkeypatch.setattr(FakeEvent, "record", broken_event_call)
    monkeypatch.setattr(FakeEvent, "synchronize", broken_event_call)

    assert worker.flush() == []


def test_untimed_worker_is_untouched():
    worker = FakeWorker()
    worker.iteration_timer = None

    assert isinstance(worker.sample_tokens(), FakeAsyncOutput)
    assert worker.execute_dummy_batch() is None


def test_output_carrying_records_keeps_its_kv_connector_output(gpu):
    worker = FakeWorker()
    worker.step()
    gpu.run()
    worker.execute_model(SimpleNamespace(iteration_details=IterationDetails()))

    assert worker.sample_tokens().get_kv_connector_output() == "kv output"


def test_executor_resolves_the_flush_into_records(gpu):
    worker = FakeWorker()
    worker.step(num_tokens=3)
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.driver_worker = SimpleNamespace(
        flush_iteration_timings=worker.iteration_timer.flush
    )

    (record,) = executor.flush_iteration_timings().result(timeout=0).iteration_records

    assert record.num_generation_tokens == 3


def test_only_the_output_rank_times_iterations():
    parallel_config = SimpleNamespace(
        world_size=8, tensor_parallel_size=2, prefill_context_parallel_size=1
    )
    config = SimpleNamespace(gpu_iteration_timing=True, parallel_config=parallel_config)
    executor = SimpleNamespace(world_size=8, parallel_config=parallel_config)
    output_rank = MultiprocExecutor._get_output_rank(executor)

    timed = [rank for rank in range(8) if make_iteration_timer(config, rank)]
    assert timed == [output_rank]

    config.gpu_iteration_timing = False
    assert make_iteration_timer(config, output_rank) is None


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({}, True),
        ({"flag": False}, False),
        ({"device_type": "cpu"}, False),
        ({"runner_type": "pooling"}, False),
        ({"is_mm_encoder_only": True}, False),
        ({"backend": "uni"}, True),
        ({"backend": "ray"}, False),
        ({"api_servers": 2}, False),  # the frontend then logs no iteration details
        ({"multiprocessing": False}, False),  # the in-process engine never idles
    ],
)
def test_gpu_iteration_timing_conditions(change, expected, monkeypatch):
    case = {
        "flag": True,
        "device_type": "cuda",
        "runner_type": "generate",
        "is_mm_encoder_only": False,
        "backend": "mp",
        "api_servers": 1,
        "multiprocessing": True,
    } | change
    monkeypatch.setenv(
        "VLLM_ENABLE_V1_MULTIPROCESSING", str(int(case["multiprocessing"]))
    )
    config = SimpleNamespace(
        observability_config=SimpleNamespace(
            enable_logging_iteration_details=case["flag"]
        ),
        device_config=SimpleNamespace(device_type=case["device_type"]),
        model_config=SimpleNamespace(runner_type=case["runner_type"]),
        is_mm_encoder_only=case["is_mm_encoder_only"],
        parallel_config=SimpleNamespace(
            distributed_executor_backend=case["backend"],
            _api_process_count=case["api_servers"],
        ),
    )

    assert VllmConfig.gpu_iteration_timing.fget(config) is expected  # type: ignore[attr-defined]  # property


def test_logger_logs_every_iteration_record(gpu, monkeypatch):
    worker = FakeWorker()
    worker.step(num_tokens=3)
    worker.step(num_tokens=5)
    stats = SchedulerStats(iteration_records=worker.flush())
    info = Mock()
    monkeypatch.setattr(loggers.logger, "info", info)
    stat_logger = SimpleNamespace(
        kv_cache_device="GPU", _log_prefix_for_engine=lambda engine_idx: ""
    )
    stat_logger._log_iteration = lambda *args: loggers.LoggingStatLogger._log_iteration(
        stat_logger, *args
    )

    loggers.LoggingStatLogger._log_iteration_details(stat_logger, stats, 0)

    assert [call.args[2:7] for call in info.call_args_list] == [
        (0, 0, 0, 0, 3),
        (1, 0, 0, 0, 5),
    ]
