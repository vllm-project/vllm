# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV offloading usage gauges are exported before the first request."""

import json
import time

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.simple_kv_offload.metrics import MetricName

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

MODEL_NAME = "vllm-project/tiny-random-LlamaForCausalLM"


@pytest.mark.parametrize(
    "connector,expected",
    [
        (
            "OffloadingConnector",
            {
                CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: 0,
                CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: 0,
                CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: 0,
            },
        ),
        (
            "SimpleCPUOffloadConnector",
            # The CPU block pool's null block counts as used.
            {MetricName.USED_BLOCKS: 1, MetricName.PENDING_STORE_BLOCKS: 0},
        ),
    ],
)
def test_usage_gauges_before_first_request(
    connector: str, expected: dict[str, float]
) -> None:
    kv_transfer_config = {
        "kv_connector": connector,
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": 64 << 20},
    }
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "512",
        "--enforce-eager",
        "--kv-cache-memory-bytes",
        str(64 << 20),
        "--api-server-count",
        "2",
        "--kv-transfer-config",
        json.dumps(kv_transfer_config),
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as server:
        # API server 0 records the engine's initial stats shortly after startup.
        deadline = time.monotonic() + 30
        while True:
            text = requests.get(server.url_for("metrics"), timeout=10).text
            samples = {
                sample.name: sample.value
                for family in text_string_to_metric_families(text)
                for sample in family.samples
                if sample.name in expected
            }
            if samples == expected or time.monotonic() > deadline:
                break
            time.sleep(1)
    assert samples == expected
