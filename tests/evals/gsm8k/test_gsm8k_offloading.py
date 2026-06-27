# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GSM8K correctness test for hybrid SSM models with OffloadingConnector.

Uses nvidia/Nemotron-H-8B-Base-8K (NemotronH architecture — hybrid
attention + Mamba layers) to guard against stride computation bugs in the
offloading worker (e.g. https://github.com/vllm-project/vllm/pull/46888)
and silent KV cache data corruption during CPU offloading.

Usage:
    pytest -s -v tests/evals/gsm8k/test_gsm8k_offloading.py
"""

import json

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

from .gsm8k_eval import evaluate_gsm8k

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

MODEL = "nvidia/Nemotron-H-8B-Base-8K"
NUM_QUESTIONS = 200
NUM_FEWSHOT = 5
# Baseline accuracy is ~0.49 on 200 questions (measured on GB200).
ACCURACY_THRESHOLD = 0.45
TOLERANCE = 0.05

KV_TRANSFER_CONFIG = json.dumps({
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "spec_name": "CPUOffloadingSpec",
        "cpu_bytes_to_use": 4 << 30,
        "eviction_policy": "lru",
    },
})

SERVER_ARGS = [
    "--enforce-eager",
    "--max-model-len", "4096",
    "--enable-prefix-caching",
    "--no-disable-hybrid-kv-cache-manager",
    "--kv-transfer-config", KV_TRANSFER_CONFIG,
    "--trust-remote-code",
    "--disable-uvicorn-access-log",
]


def test_gsm8k_offloading_correctness():
    with RemoteOpenAIServer(MODEL, SERVER_ARGS) as server:
        url = server.url_for("v1")
        host_port = url.split("://")[1].split("/")[0]
        host, port = host_port.rsplit(":", 1)

        results = evaluate_gsm8k(
            num_questions=NUM_QUESTIONS,
            num_shots=NUM_FEWSHOT,
            host=f"http://{host}",
            port=int(port),
        )

        print(f"GSM8K + Offloading (Nemotron-H-8B): "
              f"accuracy={results['accuracy']:.4f}, "
              f"invalid_rate={results['invalid_rate']:.3f}, "
              f"latency={results['latency']:.1f}s")

        assert results["accuracy"] >= ACCURACY_THRESHOLD - TOLERANCE, (
            f"GSM8K accuracy {results['accuracy']:.4f} below "
            f"{ACCURACY_THRESHOLD - TOLERANCE:.4f}"
        )
