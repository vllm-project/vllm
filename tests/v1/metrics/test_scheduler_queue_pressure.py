# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.stats import SchedulerStats


def _make_vllm_config(max_num_seqs: int):
    return SimpleNamespace(
        observability_config=SimpleNamespace(
            show_hidden_metrics=False,
            kv_cache_metrics=False,
        ),
        model_config=SimpleNamespace(
            served_model_name="test-model",
            max_model_len=128,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
    )


def _get_scheduler_queue_pressure() -> dict[str, float]:
    values: dict[str, float] = {}
    metrics_text = generate_latest().decode()
    for family in text_string_to_metric_families(metrics_text):
        if family.name != "vllm:scheduler_queue_pressure":
            continue
        for sample in family.samples:
            if sample.name == "vllm:scheduler_queue_pressure":
                values[sample.labels["reason"]] = sample.value
    return values


def test_scheduler_queue_pressure_normalizes_waiting_reqs_by_capacity():
    logger = PrometheusStatLogger(_make_vllm_config(max_num_seqs=128))

    logger.record(
        SchedulerStats(
            num_waiting_reqs=32,
            num_skipped_waiting_reqs=16,
        ),
        iteration_stats=None,
    )

    assert _get_scheduler_queue_pressure() == {
        "capacity": 0.25,
        "deferred": 0.125,
    }
