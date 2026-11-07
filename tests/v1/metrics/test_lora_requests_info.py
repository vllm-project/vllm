# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the vllm:lora_requests_info series lifecycle."""

import pytest
from prometheus_client import REGISTRY

from vllm.config import DeviceConfig, LoRAConfig, ModelConfig, VllmConfig
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import SchedulerStats

pytestmark = pytest.mark.cpu_test

TEST_MODEL = "Qwen/Qwen2.5-0.5B"
METRIC_NAME = "vllm:lora_requests_info"


def build_logger_config(lora_config: LoRAConfig | None = None) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(model=TEST_MODEL),
        device_config=DeviceConfig(device="cuda"),
        lora_config=lora_config,
    )


def make_lora_config() -> LoRAConfig:
    return LoRAConfig(max_loras=8)


def lora_info_series() -> list[tuple[str, str]]:
    """Collect the exposed (running, waiting) label pairs of the gauge."""
    found = []
    for metric in REGISTRY.collect():
        if metric.name != METRIC_NAME:
            continue
        for sample in metric.samples:
            found.append(
                (
                    sample.labels["running_lora_adapters"],
                    sample.labels["waiting_lora_adapters"],
                )
            )
    return found


def record_states(logger: PrometheusStatLogger, states: list[dict[str, int]]):
    for state in states:
        logger.record(SchedulerStats(running_lora_adapters=state), None)


def test_lora_requests_info_does_not_accumulate_series():
    config = build_logger_config(make_lora_config())
    try:
        logger = PrometheusStatLogger(config)
        states = [
            {},
            {"adapter-a": 1},
            {"adapter-a": 1, "adapter-b": 2},
            {"adapter-b": 1},
            {},
        ]
        record_states(logger, states)
        # Only the last state may remain exposed.
        assert lora_info_series() == [("", "")]
    finally:
        unregister_vllm_metrics()


def test_lora_requests_info_keeps_current_state():
    config = build_logger_config(make_lora_config())
    try:
        logger = PrometheusStatLogger(config)
        record_states(logger, [{"adapter-a": 1, "adapter-b": 1}])
        assert lora_info_series() == [("adapter-a,adapter-b", "")]
    finally:
        unregister_vllm_metrics()


def test_lora_requests_info_per_engine_children():
    config = build_logger_config(make_lora_config())
    try:
        logger = PrometheusStatLogger(config, engine_indexes=[0, 1])
        logger.record(
            SchedulerStats(running_lora_adapters={"adapter-a": 1}), None, engine_idx=0
        )
        logger.record(
            SchedulerStats(running_lora_adapters={"adapter-b": 1}), None, engine_idx=1
        )
        logger.record(
            SchedulerStats(running_lora_adapters={"adapter-c": 1}), None, engine_idx=0
        )
        assert sorted(lora_info_series()) == [("adapter-b", ""), ("adapter-c", "")]
    finally:
        unregister_vllm_metrics()


def test_lora_requests_info_shared_label_between_engines():
    """A shared state is only removed once the last engine holding it
    moves on."""
    config = build_logger_config(make_lora_config())
    try:
        logger = PrometheusStatLogger(config, engine_indexes=[0, 1])
        shared = {"adapter-a": 1}
        logger.record(SchedulerStats(running_lora_adapters=shared), None, engine_idx=0)
        logger.record(SchedulerStats(running_lora_adapters=shared), None, engine_idx=1)
        assert lora_info_series() == [("adapter-a", "")]

        # engine 0 moves on — engine 1 still holds the shared state
        logger.record(
            SchedulerStats(running_lora_adapters={"adapter-b": 1}), None, engine_idx=0
        )
        assert sorted(lora_info_series()) == [("adapter-a", ""), ("adapter-b", "")]

        # engine 1 moves on too — now the shared state is removed
        logger.record(
            SchedulerStats(running_lora_adapters={"adapter-c": 1}), None, engine_idx=1
        )
        assert sorted(lora_info_series()) == [("adapter-b", ""), ("adapter-c", "")]
    finally:
        unregister_vllm_metrics()


def test_lora_requests_info_absent_without_lora_config():
    config = build_logger_config()
    try:
        logger = PrometheusStatLogger(config)
        assert logger.gauge_lora_info is None
        record_states(logger, [{"adapter-a": 1}])
        assert lora_info_series() == []
    finally:
        unregister_vllm_metrics()


if __name__ == "__main__":
    pytest.main([__file__])
