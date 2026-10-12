# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import subprocess
import sys

import pytest

from vllm.v1.metrics.loggers import WAITING_REASON_CAPACITY, WAITING_REASON_DEFERRED
from vllm.v1.metrics.prometheus import get_prometheus_registry
from vllm.v1.metrics.stats import KV_FETCH_STAGES

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

MODEL = "distilbert/distilgpt2"

# Builds the logger the way an API server process does at startup, then records
# scheduler stats for engine 1 only, so engine 0 is still before its first stats.
# A fresh interpreter is needed because prometheus_client chooses multiprocess
# storage when it is imported.
CREATE_LOGGER = f"""
import sys

from vllm.config import KVTransferConfig, ModelConfig, VllmConfig
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.stats import KV_FETCH_STAGES, SchedulerStats

kv_transfer_config = (
    KVTransferConfig(kv_connector="ExampleConnector", kv_role="kv_consumer")
    if sys.argv[1] == "True"
    else None
)
logger = PrometheusStatLogger(
    VllmConfig(
        model_config=ModelConfig(model={MODEL!r}),
        kv_transfer_config=kv_transfer_config,
    ),
    engine_indexes=[0, 1],
)
logger.record(
    SchedulerStats(
        num_running_reqs=7,
        num_waiting_reqs=3,
        num_skipped_waiting_reqs=1,
        kv_cache_usage=0.5,
        num_kv_fetch_reqs_by_stage=dict.fromkeys(KV_FETCH_STAGES, 2),
    ),
    None,
    engine_idx=1,
)
"""


@pytest.mark.parametrize("kv_consumer", [False, True])
def test_scheduler_gauges_exported_before_first_stats(
    tmp_path, monkeypatch, kv_consumer
):
    """With --api-server-count > 1, an idle server reports these gauges as 0,
    and the first scheduler stats replace the 0."""
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    subprocess.run(
        [sys.executable, "-c", CREATE_LOGGER, str(kv_consumer)],
        check=True,
        env=os.environ.copy(),
    )

    registry = get_prometheus_registry()
    expected_by_engine: dict[str, dict[str, float]] = {
        "0": dict(running=0, waiting=0, capacity=0, deferred=0, kv=0, fetch=0),
        "1": dict(running=7, waiting=4, capacity=3, deferred=1, kv=0.5, fetch=2),
    }
    for engine, expected in expected_by_engine.items():
        labels = {"engine": engine, "model_name": MODEL}
        for name, key in (
            ("vllm:num_requests_running", "running"),
            ("vllm:num_requests_waiting", "waiting"),
            ("vllm:kv_cache_usage_perc", "kv"),
        ):
            value = registry.get_sample_value(name, labels)
            assert value == expected[key], (name, engine)
        for reason, key in (
            (WAITING_REASON_CAPACITY, "capacity"),
            (WAITING_REASON_DEFERRED, "deferred"),
        ):
            value = registry.get_sample_value(
                "vllm:num_requests_waiting_by_reason", {**labels, "reason": reason}
            )
            assert value == expected[key], (reason, engine)
        for stage in KV_FETCH_STAGES:
            value = registry.get_sample_value(
                "vllm:num_requests_kv_fetch_by_stage", {**labels, "stage": stage}
            )
            assert value == (expected["fetch"] if kv_consumer else None), (
                stage,
                engine,
            )
