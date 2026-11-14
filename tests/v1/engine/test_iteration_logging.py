# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import time
from types import SimpleNamespace

import pytest

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


def test_dummy_batch_publishes_the_gpu_timings_it_returns():
    record = make_iteration_details()
    engine = SimpleNamespace(
        model_executor=SimpleNamespace(execute_dummy_batch=lambda: [record]),
        scheduler=SimpleNamespace(make_stats=SchedulerStats),
        output_queue=queue.Queue(),
    )

    EngineCoreProc.execute_dummy_batch(engine)

    client_index, outputs = engine.output_queue.get_nowait()
    assert client_index == 0
    assert outputs.scheduler_stats.iteration_records == [record]


@pytest.mark.parametrize(
    ("records", "stats"),
    [
        ([], SchedulerStats()),  # nothing has completed
        (None, SchedulerStats()),  # the worker does not time iterations
        ([make_iteration_details()], None),  # stats are disabled
    ],
)
def test_dummy_batch_publishes_nothing_without_gpu_timings_or_stats(records, stats):
    engine = SimpleNamespace(
        model_executor=SimpleNamespace(execute_dummy_batch=lambda: records),
        scheduler=SimpleNamespace(make_stats=lambda: stats),
        output_queue=queue.Queue(),
    )

    EngineCoreProc.execute_dummy_batch(engine)

    assert engine.output_queue.empty()
