# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.metrics.loggers import (
    WAITING_REASON_CAPACITY,
    WAITING_REASON_DEFERRED,
    PrometheusStatLogger,
)

pytestmark = pytest.mark.cpu_test


class _RecordingGauge:
    """Replacement for prometheus_client.Gauge that records set() calls.

    The real multiprocess value class is chosen when prometheus_client is
    imported (before any test can set PROMETHEUS_MULTIPROC_DIR), so a
    multiprocess-mode test is not reliable in-process. What matters for
    #59988 is the seeding behaviour: every scheduler-state gauge must be set
    to 0 as soon as the logger is constructed.
    """

    instances: list["_RecordingGauge"] = []

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.children: list[_RecordingGaugeChild] = []
        _RecordingGauge.instances.append(self)

    def labels(self, *labelvalues, **kwlabelvalues):
        child = _RecordingGaugeChild(list(labelvalues) + list(kwlabelvalues.values()))
        self.children.append(child)
        return child


class _RecordingGaugeChild:
    def __init__(self, labelvalues: list[object]):
        self.labelvalues = labelvalues
        self.set_values: list[float] = []

    def set(self, value: float) -> None:
        self.set_values.append(value)


def _make_vllm_config():
    """A minimal VllmConfig-like object for the PrometheusStatLogger ctor.

    The logger only reads the attributes below, so a SimpleNamespace keeps the
    test free of model/tokenizer resolution and compiled-op imports.
    """
    return SimpleNamespace(
        observability_config=SimpleNamespace(
            show_hidden_metrics=False,
            kv_cache_metrics=False,
            custom_histogram_buckets=None,
        ),
        model_config=SimpleNamespace(
            served_model_name="test-model",
            max_model_len=1024,
            is_diffusion=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        lora_config=None,
    )


@pytest.mark.parametrize("engine_indexes", [[0], [0, 1], [0, 1, 2]])
def test_scheduler_state_gauges_seed_zero_on_startup(monkeypatch, engine_indexes):
    """#59988: multiprocess_mode="mostrecent" drops never-set samples, so an
    idle multi-API-server deployment exported no series for the scheduler-state
    gauges until the first request. The logger must seed them with 0 at
    construction time, like record_sleep_state() does for the sleep state."""
    _RecordingGauge.instances = []
    monkeypatch.setattr(PrometheusStatLogger, "_gauge_cls", _RecordingGauge)

    PrometheusStatLogger(_make_vllm_config(), engine_indexes=engine_indexes)

    gauges = {g.name: g for g in _RecordingGauge.instances}
    for name in (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
    ):
        children = gauges[name].children
        assert len(children) == len(engine_indexes)
        assert all(child.set_values == [0] for child in children)

    by_reason = gauges["vllm:num_requests_waiting_by_reason"].children
    assert len(by_reason) == 2 * len(engine_indexes)
    assert all(child.set_values == [0] for child in by_reason)
    reasons = {child.labelvalues[-1] for child in by_reason}
    assert reasons == {WAITING_REASON_CAPACITY, WAITING_REASON_DEFERRED}
