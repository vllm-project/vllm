# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import subprocess
import sys

import pytest

from vllm.v1.metrics.prometheus import get_prometheus_registry
from vllm.v1.metrics.stats import KV_FETCH_STAGES

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

MODEL = "distilbert/distilgpt2"

# Builds the logger the way an API server process does at startup, before any
# scheduler stats arrive. A fresh interpreter is needed because prometheus_client
# chooses multiprocess storage when it is imported.
CREATE_LOGGER = f"""
import sys

from vllm.config import KVTransferConfig, ModelConfig, VllmConfig
from vllm.v1.metrics.loggers import PrometheusStatLogger

kv_transfer_config = (
    KVTransferConfig(kv_connector="ExampleConnector", kv_role="kv_consumer")
    if sys.argv[1] == "True"
    else None
)
PrometheusStatLogger(
    VllmConfig(
        model_config=ModelConfig(model={MODEL!r}),
        kv_transfer_config=kv_transfer_config,
    ),
    engine_indexes=[0, 1],
)
"""


@pytest.mark.parametrize("kv_consumer", [False, True])
def test_scheduler_gauges_exported_before_first_stats(
    tmp_path, monkeypatch, kv_consumer
):
    """With --api-server-count > 1, an idle server reports these gauges as 0."""
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    subprocess.run(
        [sys.executable, "-c", CREATE_LOGGER, str(kv_consumer)],
        check=True,
        env=os.environ.copy(),
    )

    registry = get_prometheus_registry()
    for engine in ("0", "1"):
        labels = {"engine": engine, "model_name": MODEL}
        for name in (
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:kv_cache_usage_perc",
        ):
            assert registry.get_sample_value(name, labels) == 0, (name, engine)
        for reason in ("capacity", "deferred"):
            value = registry.get_sample_value(
                "vllm:num_requests_waiting_by_reason", {**labels, "reason": reason}
            )
            assert value == 0, (reason, engine)
        for stage in KV_FETCH_STAGES:
            value = registry.get_sample_value(
                "vllm:num_requests_kv_fetch_by_stage", {**labels, "stage": stage}
            )
            assert value == (0 if kv_consumer else None), (stage, engine)
