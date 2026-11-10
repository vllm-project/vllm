# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import ObservabilityConfig, VllmConfig
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
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
    log_stats: bool = True, iteration_timing_enabled: bool = False
) -> SimpleNamespace:
    observability = ObservabilityConfig(enable_logging_iteration_details=True)
    observability.iteration_timing_enabled = iteration_timing_enabled
    return SimpleNamespace(
        log_stats=log_stats,
        vllm_config=SimpleNamespace(observability_config=observability),
    )


@pytest.mark.parametrize(
    "overrides,expected",
    [
        pytest.param({}, True, id="supported-uni"),
        pytest.param({"logging": False, "executor": "mp"}, False, id="logging-off"),
        pytest.param({"stats": False, "executor": "mp"}, False, id="stats-off"),
        pytest.param({"multiprocessing": False}, False, id="in-process"),
        pytest.param({"device": "cpu"}, False, id="non-cuda"),
        pytest.param({"runner": "pooling"}, False, id="pooling"),
        pytest.param(
            {"pipeline_parallel_size": 2, "executor": "mp"},
            True,
            id="supported-pp2",
        ),
        pytest.param({"executor": "ray"}, False, id="ray"),
        pytest.param(
            {"stats": False, "executor": "mp", "previously_enabled": True},
            False,
            id="clear-stale-state",
        ),
    ],
)
def test_iteration_logging_enables_supported_timing_only(overrides, expected, caplog):
    logging = overrides.get("logging", True)
    stats = overrides.get("stats", True)
    config = VllmConfig()
    config.model_config = SimpleNamespace(
        runner_type=overrides.get("runner", "generate")
    )
    config.device_config.device_type = overrides.get("device", "cuda")
    config.parallel_config.pipeline_parallel_size = overrides.get(
        "pipeline_parallel_size", 1
    )
    config.parallel_config.distributed_executor_backend = overrides.get(
        "executor", "uni"
    )
    config.observability_config = ObservabilityConfig(
        enable_logging_iteration_details=logging
    )
    config.observability_config.iteration_timing_enabled = overrides.get(
        "previously_enabled", False
    )

    config.resolve_iteration_timing(stats, overrides.get("multiprocessing", True))

    assert config.observability_config.iteration_timing_enabled is expected
    if logging and stats and not expected:
        assert "Logging iteration counts without GPU timing" in caplog.text


@pytest.mark.parametrize("log_stats, gpu_timing", [(False, False), (True, True)])
def test_legacy_iteration_details_skip_disabled_logging_and_gpu_timing(
    log_stats, gpu_timing
):
    engine = make_fake_engine(log_stats, iteration_timing_enabled=gpu_timing)

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is None

    assert not hasattr(engine, "_iteration_index")


def test_capture_iteration_details_without_gpu_timing_omits_elapsed_time():
    engine = make_fake_engine()

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is not None
        assert iteration_details.elapsed_ms is None
        assert iteration_details.is_dummy

    assert iteration_details is not None
    assert iteration_details.elapsed_ms is None
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
