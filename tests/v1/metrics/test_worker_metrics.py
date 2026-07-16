# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.metrics.stats import SchedulerStats, WorkerTimingStats
from vllm.v1.metrics.worker import WorkerTimingProm
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test


class FakeHistogram:
    instances: dict[str, "FakeHistogram"] = {}

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.labelnames = kwargs["labelnames"]
        self.observations: list[tuple[tuple[object, ...], float]] = []
        self._labels: tuple[object, ...] = ()
        self.instances[name] = self

    def labels(self, *labels: object) -> "FakeHistogram":
        child = FakeHistogram.__new__(FakeHistogram)
        child.name = self.name
        child.observations = self.observations
        child._labels = labels
        return child

    def observe(self, value: float) -> None:
        self.observations.append((self._labels, value))


class FakeWorkerTimingProm(WorkerTimingProm):
    _histogram_cls = FakeHistogram


def make_stat(proposer_time: float | None) -> WorkerTimingStats:
    model_time = 0.008
    return WorkerTimingStats(
        iteration_index=3,
        phase="prefill",
        num_model_tokens=16,
        num_requests=4,
        num_prefill_requests=1,
        num_prefill_tokens=13,
        num_decode_requests=3,
        num_decode_tokens=3,
        model_time_seconds=model_time,
        proposer_time_seconds=proposer_time,
        total_time_seconds=model_time + (proposer_time or 0.0),
    )


def test_prometheus_histograms_preserve_worker_dimensions() -> None:
    FakeHistogram.instances.clear()
    metrics = FakeWorkerTimingProm(
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )

    metrics.observe([make_stat(0.002), make_stat(None)], engine_idx=0)

    model = FakeHistogram.instances["vllm:worker_model_execute_time_seconds"]
    proposer = FakeHistogram.instances["vllm:worker_spec_decode_proposer_time_seconds"]
    total = FakeHistogram.instances["vllm:worker_step_time_seconds"]
    assert model.labelnames[-2:] == ["phase", "num_model_tokens"]
    assert proposer.labelnames[-2:] == ["phase", "num_requests"]
    assert total.labelnames[-2:] == ["phase", "num_model_tokens"]
    assert model.observations == [
        (("model", "0", "prefill", "16"), 0.008),
        (("model", "0", "prefill", "16"), 0.008),
    ]
    assert proposer.observations == [(("model", "0", "prefill", "4"), 0.002)]
    assert total.observations == [
        (("model", "0", "prefill", "16"), 0.01),
        (("model", "0", "prefill", "16"), 0.008),
    ]


def test_worker_timing_samples_reach_the_frontend() -> None:
    outputs = EngineCoreOutputs(
        scheduler_stats=SchedulerStats(worker_timing_samples=[make_stat(0.002)])
    )

    encoded = MsgpackEncoder().encode(outputs)
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoded)

    assert decoded.scheduler_stats is not None
    assert decoded.scheduler_stats.worker_timing_samples == [make_stat(0.002)]


def test_iteration_logging_uses_worker_device_timing() -> None:
    engine_core = EngineCore.__new__(EngineCore)
    engine_core.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        device_config=SimpleNamespace(device_type="cuda"),
        observability_config=SimpleNamespace(enable_logging_iteration_details=True),
    )
    output = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, worker_timing_samples=[make_stat(0.002)]
    )

    with patch("vllm.v1.engine.core.logger.info") as log_info:
        engine_core.log_worker_iteration_details(output)

    assert log_info.call_count == 1
    assert log_info.call_args.args[1] == 3
    assert log_info.call_args.args[-1] == pytest.approx(10.0)
