# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake admission and P/D accuracy with same-host HIP transport on ROCm."""

import json
import os

import pytest
import requests

from tests.utils import RemoteOpenAIServer, spawn_new_process_for_each_test
from tests.v1.kv_connector.rocm_pd_accuracy_utils import (
    QWEN_MODEL,
    run_rocm_pd_accuracy,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm P/D equivalents"),
]


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.parametrize(
    "model,prefill_tp,decode_tp",
    [
        pytest.param(QWEN_MODEL, 1, 1, id="mooncake-hip-qwen-p1-d1"),
        pytest.param(QWEN_MODEL, 2, 2, id="mooncake-hip-qwen-p2-d2"),
    ],
)
@spawn_new_process_for_each_test
def test_rocm_pd_accuracy(model, prefill_tp, decode_tp, monkeypatch):
    run_rocm_pd_accuracy("mooncake", model, prefill_tp, decode_tp, monkeypatch)


@pytest.fixture(scope="module", params=["kv_producer", "kv_consumer"])
def validation_server(request):
    config = {
        "kv_connector": "MooncakeConnector",
        "kv_role": request.param,
        "kv_connector_extra_config": {"mooncake_protocol": "hip", "device_name": ""},
    }
    with RemoteOpenAIServer(
        QWEN_MODEL,
        [
            "--revision",
            "c1899de289a04d12100db370d81485cdf75e47ca",
            "--dtype",
            "bfloat16",
            "--enforce-eager",
            "--max-model-len",
            "256",
            "--gpu-memory-utilization",
            "0.2",
            "--kv-transfer-config",
            json.dumps(config),
        ],
        env_dict={
            "VLLM_HOST_IP": "127.0.0.1",
            "VLLM_PORT": str(get_open_port()),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(get_open_port()),
            "PATH": os.environ["PATH"],
        },
        max_wait_seconds=900,
    ) as server:
        yield request.param, server


@pytest.mark.distributed(num_gpus=1)
def test_invalid_transfer_params_preserve_serving(validation_server):
    """Reject role and metadata errors before they can occupy the engine."""
    role, server = validation_server
    prefill = {
        "do_remote_prefill": True,
        "transfer_id": "invalid-admission",
        "remote_engine_id": "prefill",
        "remote_bootstrap_addr": "http://127.0.0.1:1",
    }
    cases = (
        [prefill]
        if role == "kv_producer"
        else [
            {"do_remote_decode": True, "transfer_id": "invalid-admission"},
            *[
                {key: value for key, value in prefill.items() if key != missing}
                for missing in (
                    "transfer_id",
                    "remote_engine_id",
                    "remote_bootstrap_addr",
                )
            ],
        ]
    )
    payload = {
        "model": QWEN_MODEL,
        "prompt": "Hello",
        "max_tokens": 1,
        "temperature": 0,
    }
    for params in cases:
        response = requests.post(
            server.url_for("v1/completions"),
            json={**payload, "kv_transfer_params": params},
            timeout=60,
        )
        assert response.status_code == 400, response.text
        assert "kv_transfer_params" in response.json()["error"]["message"]
        healthy = requests.post(
            server.url_for("v1/completions"), json=payload, timeout=60
        )
        assert healthy.status_code == 200, healthy.text
        assert healthy.json()["usage"]["completion_tokens"] == 1
        assert requests.get(server.url_for("health"), timeout=10).status_code == 200
