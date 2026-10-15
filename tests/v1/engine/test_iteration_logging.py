# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import DeviceConfig, VllmConfig
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.metrics import loggers
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


def make_fake_engine(log_stats: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        log_stats=log_stats,
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(
                enable_logging_iteration_details=True,
            )
        ),
    )


def test_capture_iteration_details_disabled_without_log_stats():
    engine = make_fake_engine(log_stats=False)

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is None

    assert not hasattr(engine, "_iteration_index")


def test_capture_iteration_details_without_model_output_omits_duration():
    engine = make_fake_engine()

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is not None
        assert iteration_details.elapsed_ms is None
        assert iteration_details.is_dummy

    assert iteration_details is not None
    assert iteration_details.elapsed_ms is None
    assert engine._iteration_index == 1


@pytest.mark.parametrize("elapsed_ms", [None, 12.5])
def test_attach_iteration_details_uses_model_duration_and_existing_output(elapsed_ms):
    iteration_details = make_iteration_details()
    outputs = {
        2: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
        1: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
    }

    model_output = ModelRunnerOutput([], {}, iteration_elapsed_ms=elapsed_ms)
    EngineCore._attach_iteration_details(
        FakeEngineCore(), outputs, iteration_details, model_output
    )

    assert 0 not in outputs
    assert outputs[2].scheduler_stats is not None
    assert outputs[2].scheduler_stats.iteration_details == iteration_details
    assert iteration_details.elapsed_ms == elapsed_ms
    assert outputs[1].scheduler_stats is not None
    assert outputs[1].scheduler_stats.iteration_details is None


def test_attach_iteration_details_falls_back_to_client_zero_without_outputs():
    iteration_details = make_iteration_details()
    outputs: dict[int, EngineCoreOutputs] = {}

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert set(outputs) == {0}
    assert outputs[0].scheduler_stats is not None
    assert outputs[0].scheduler_stats.iteration_details == iteration_details


@pytest.mark.parametrize("elapsed_ms", [None, 12.5])
def test_iteration_logging_omits_unavailable_duration(monkeypatch, elapsed_ms):
    stat_logger = loggers.LoggingStatLogger(
        VllmConfig(device_config=DeviceConfig("cpu"))
    )
    messages = []
    monkeypatch.setattr(
        loggers.logger, "info", lambda fmt, *args: messages.append(fmt % args)
    )
    details = make_iteration_details()
    details.elapsed_ms = elapsed_ms

    stat_logger.record(SchedulerStats(iteration_details=details), None)

    assert len(messages) == 1
    assert ("iteration elapsed time" in messages[0]) == (elapsed_ms is not None)
