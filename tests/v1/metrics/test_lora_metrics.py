# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrometheusStatLogger's LoRA load-event consumer, on a private registry so
the test does not fight the process-global one."""

import pytest
from prometheus_client import CollectorRegistry, Gauge

from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.notifications import CustomNotification, LoRALoadEvent

LABELNAMES = ["model_name", "engine"]


def _lora_logger(registry: CollectorRegistry, engine_indexes=(0,)):
    """A PrometheusStatLogger with only the LoRA gauges, built the way
    __init__ builds them."""
    logger = PrometheusStatLogger.__new__(PrometheusStatLogger)
    logger.per_engine_labelvalues = {
        idx: ["test-model", str(idx)] for idx in engine_indexes
    }
    gpu = Gauge("vllm:num_gpu_loaded_lora_adapters", "", LABELNAMES, registry=registry)
    cpu = Gauge("vllm:num_cpu_loaded_lora_adapters", "", LABELNAMES, registry=registry)
    logger.gauge_lora_gpu_adapters = {
        idx: gpu.labels(*labels)
        for idx, labels in logger.per_engine_labelvalues.items()
    }
    logger.gauge_lora_cpu_adapters = {
        idx: cpu.labels(*labels)
        for idx, labels in logger.per_engine_labelvalues.items()
    }
    logger.gauge_lora_adapter_loaded = Gauge(
        "vllm:lora_adapter_loaded",
        "",
        LABELNAMES + ["adapter_name", "level", "pinned"],
        registry=registry,
    )
    logger._lora_loaded_series = {}
    return logger


def _loaded_series(registry: CollectorRegistry) -> dict[tuple[str, str, str], float]:
    """(adapter_name, level, pinned) -> value for every live series."""
    return {
        (s.labels["adapter_name"], s.labels["level"], s.labels["pinned"]): s.value
        for metric in registry.collect()
        if metric.name == "vllm:lora_adapter_loaded"
        for s in metric.samples
    }


def _count(registry: CollectorRegistry, name: str, engine: str = "0") -> float:
    for metric in registry.collect():
        if metric.name == name:
            for s in metric.samples:
                if s.labels["engine"] == engine:
                    return s.value
    raise AssertionError(f"{name} has no sample for engine {engine}")


def test_load_event_publishes_one_series_per_resident_adapter():
    registry = CollectorRegistry()
    logger = _lora_logger(registry)

    logger.record_engine_notifications(
        [
            LoRALoadEvent(
                gpu_adapters=["alpha", "beta"],
                cpu_adapters=["alpha", "beta", "gamma"],
                pinned_adapters=["alpha"],
            )
        ]
    )

    assert _loaded_series(registry) == {
        ("alpha", "gpu", "true"): 1.0,
        ("beta", "gpu", "false"): 1.0,
        ("gamma", "cpu", "false"): 1.0,
    }
    assert _count(registry, "vllm:num_gpu_loaded_lora_adapters") == 2
    assert _count(registry, "vllm:num_cpu_loaded_lora_adapters") == 3


def test_eviction_removes_the_series():
    """A snapshot replaces the previous one: evicted adapters must not keep
    reading as resident, and a tier change must not leave the old series."""
    registry = CollectorRegistry()
    logger = _lora_logger(registry)
    logger.record_engine_notifications(
        [LoRALoadEvent(gpu_adapters=["alpha", "beta"], cpu_adapters=["alpha", "beta"])]
    )

    logger.record_engine_notifications(
        [LoRALoadEvent(gpu_adapters=["gamma"], cpu_adapters=["beta", "gamma"])]
    )

    assert _loaded_series(registry) == {
        ("beta", "cpu", "false"): 1.0,
        ("gamma", "gpu", "false"): 1.0,
    }
    assert _count(registry, "vllm:num_gpu_loaded_lora_adapters") == 1
    assert _count(registry, "vllm:num_cpu_loaded_lora_adapters") == 2


def test_empty_snapshot_clears_everything():
    registry = CollectorRegistry()
    logger = _lora_logger(registry)
    logger.record_engine_notifications(
        [LoRALoadEvent(gpu_adapters=["alpha"], cpu_adapters=["alpha"])]
    )

    logger.record_engine_notifications([LoRALoadEvent()])

    assert _loaded_series(registry) == {}
    assert _count(registry, "vllm:num_gpu_loaded_lora_adapters") == 0
    assert _count(registry, "vllm:num_cpu_loaded_lora_adapters") == 0


def test_events_apply_in_order_within_one_batch():
    """Additive delivery: the last snapshot in a batch wins."""
    registry = CollectorRegistry()
    logger = _lora_logger(registry)

    logger.record_engine_notifications(
        [
            LoRALoadEvent(gpu_adapters=["alpha"], cpu_adapters=["alpha"]),
            LoRALoadEvent(gpu_adapters=["beta"], cpu_adapters=["beta"]),
        ]
    )

    assert _loaded_series(registry) == {("beta", "gpu", "false"): 1.0}


def test_engines_are_tracked_independently():
    registry = CollectorRegistry()
    logger = _lora_logger(registry, engine_indexes=(0, 1))

    logger.record_engine_notifications(
        [LoRALoadEvent(gpu_adapters=["alpha"], cpu_adapters=["alpha"])], engine_idx=0
    )
    logger.record_engine_notifications(
        [LoRALoadEvent(gpu_adapters=["beta"], cpu_adapters=["beta"])], engine_idx=1
    )
    logger.record_engine_notifications([LoRALoadEvent()], engine_idx=0)

    assert _loaded_series(registry) == {("beta", "gpu", "false"): 1.0}
    assert _count(registry, "vllm:num_gpu_loaded_lora_adapters", engine="0") == 0
    assert _count(registry, "vllm:num_gpu_loaded_lora_adapters", engine="1") == 1


def test_custom_notifications_are_ignored():
    registry = CollectorRegistry()
    logger = _lora_logger(registry)

    logger.record_engine_notifications([CustomNotification(key="my_plugin")])

    assert _loaded_series(registry) == {}


def test_load_event_without_lora_config_is_a_noop():
    """Engines started without --enable-lora register no LoRA gauges."""
    logger = PrometheusStatLogger.__new__(PrometheusStatLogger)
    logger.gauge_lora_adapter_loaded = None

    logger.record_engine_notifications([LoRALoadEvent(cpu_adapters=["alpha"])])


@pytest.mark.parametrize("pinned", [True, False])
def test_pinned_label_tracks_the_event(pinned):
    registry = CollectorRegistry()
    logger = _lora_logger(registry)

    logger.record_engine_notifications(
        [
            LoRALoadEvent(
                gpu_adapters=["alpha"],
                cpu_adapters=["alpha"],
                pinned_adapters=["alpha"] if pinned else [],
            )
        ]
    )

    assert _loaded_series(registry) == {("alpha", "gpu", str(pinned).lower()): 1.0}
