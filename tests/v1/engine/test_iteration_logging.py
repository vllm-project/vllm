# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import IterationDetails
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.metrics.iteration_workload import ScheduledWorkload
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats


class FakeEngineCore:
    scheduler = SimpleNamespace(make_stats=SchedulerStats)
    _make_iteration_details_stats = EngineCore._make_iteration_details_stats


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
    collect_stats: bool = True, gpu_iteration_timing: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        collect_stats=collect_stats,
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(
                enable_logging_iteration_details=True,
            ),
            gpu_iteration_timing=gpu_iteration_timing,
        ),
    )


def test_untimed_iteration_details_disabled_without_log_stats():
    engine = make_fake_engine(collect_stats=False)

    assert EngineCore._untimed_iteration_details(engine, None) is None
    assert not hasattr(engine, "_iteration_index")


def test_untimed_iteration_details_have_no_elapsed_time():
    engine = make_fake_engine()

    iteration_details = EngineCore._untimed_iteration_details(engine, None)

    assert iteration_details is not None
    assert iteration_details.elapsed_ms is None
    assert iteration_details.is_dummy
    assert engine._iteration_index == 1


def test_untimed_iteration_details_copy_the_scheduled_counts():
    engine = make_fake_engine()
    workload = ScheduledWorkload(sum_decode_kv_tokens=7)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=11,
        iteration_details=IterationDetails(1, 8, 2, 3, 4, 5, workload),
    )

    iteration_details = EngineCore._untimed_iteration_details(engine, scheduler_output)

    assert iteration_details == SchedulerIterationDetails(
        0,
        1,
        8,
        2,
        3,
        num_encoder_inputs=4,
        num_encoder_output_tokens=5,
        scheduled_workload=workload,
    )


def test_attach_iteration_details_uses_existing_output():
    iteration_details = make_iteration_details()
    # The scheduler hands the stats the worker output's list, which is shared.
    worker_records: list[SchedulerIterationDetails] = []
    outputs = {
        2: EngineCoreOutputs(
            scheduler_stats=SchedulerStats(iteration_records=worker_records)
        ),
        1: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
    }

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert 0 not in outputs
    assert outputs[2].scheduler_stats is not None
    assert outputs[2].scheduler_stats.iteration_records == [iteration_details]
    assert worker_records == []
    assert outputs[1].scheduler_stats is not None
    assert outputs[1].scheduler_stats.iteration_records == []


def test_attach_iteration_details_falls_back_to_client_zero_without_outputs():
    iteration_details = make_iteration_details()
    outputs: dict[int, EngineCoreOutputs] = {}

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert set(outputs) == {0}
    assert outputs[0].scheduler_stats is not None
    assert outputs[0].scheduler_stats.iteration_records == [iteration_details]


def test_untimed_iteration_details_defer_to_gpu_timing():
    engine = make_fake_engine(gpu_iteration_timing=True)

    assert EngineCore._untimed_iteration_details(engine, None) is None


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
