# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GSM8K correctness test with OffloadingConnector (CPU KV offloading).

Regression guard for stride computation bugs in the offloading worker
(e.g. https://github.com/vllm-project/vllm/pull/46888) and silent KV
cache data corruption during CPU offloading.

Covers three architecture families:
  - Hybrid SSM (NemotronH: attention + Mamba layers)
  - Dense transformer (Gemma 4)
  - MoE (DeepSeek-V4-Flash)

Usage:
    pytest -s -v tests/evals/gsm8k/test_gsm8k_offloading.py
"""

import json
from dataclasses import dataclass, field

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

from .gsm8k_eval import evaluate_gsm8k

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

NUM_QUESTIONS = 200
NUM_FEWSHOT = 5


def _kv_transfer_config(cpu_gib: int = 4) -> str:
    return json.dumps(
        {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "CPUOffloadingSpec",
                "cpu_bytes_to_use": cpu_gib << 30,
                "eviction_policy": "lru",
            },
        }
    )


@dataclass
class OffloadingModelConfig:
    id: str
    model: str
    accuracy_threshold: float
    tolerance: float = 0.05
    extra_server_args: list[str] = field(default_factory=list)
    cpu_offload_gib: int = 4
    startup_timeout: int = 600


MODELS = [
    OffloadingModelConfig(
        id="nemotron-h-8b",
        model="nvidia/Nemotron-H-8B-Base-8K",
        # Baseline ~0.49 on 200 questions (measured on GB200).
        accuracy_threshold=0.45,
    ),
    OffloadingModelConfig(
        id="gemma-4-e4b-it",
        model="google/gemma-4-E4B-it",
        # Baseline ~0.64 on 200 questions (measured on GB200).
        accuracy_threshold=0.55,
    ),
    OffloadingModelConfig(
        id="deepseek-v4-flash",
        model="deepseek-ai/DeepSeek-V4-Flash",
        # Baseline ~0.97 on 200 questions (measured on GB200).
        accuracy_threshold=0.90,
        extra_server_args=[
            "--tensor-parallel-size",
            "2",
            "--enable-expert-parallel",
            "--kv-cache-dtype",
            "fp8",
            "--block-size",
            "256",
        ],
        cpu_offload_gib=16,
        startup_timeout=1200,
    ),
]


@pytest.mark.parametrize("cfg", MODELS, ids=lambda c: c.id)
def test_gsm8k_offloading_correctness(cfg: OffloadingModelConfig):
    server_args = [
        "--enforce-eager",
        "--max-model-len",
        "4096",
        "--enable-prefix-caching",
        "--no-disable-hybrid-kv-cache-manager",
        "--kv-transfer-config",
        _kv_transfer_config(cfg.cpu_offload_gib),
        "--trust-remote-code",
        "--disable-uvicorn-access-log",
        *cfg.extra_server_args,
    ]

    with RemoteOpenAIServer(
        cfg.model,
        server_args,
        max_wait_seconds=cfg.startup_timeout,
    ) as server:
        url = server.url_for("v1")
        host_port = url.split("://")[1].split("/")[0]
        host, port = host_port.rsplit(":", 1)

        results = evaluate_gsm8k(
            num_questions=NUM_QUESTIONS,
            num_shots=NUM_FEWSHOT,
            host=f"http://{host}",
            port=int(port),
        )

        print(
            f"GSM8K + Offloading ({cfg.id}): "
            f"accuracy={results['accuracy']:.4f}, "
            f"invalid_rate={results['invalid_rate']:.3f}, "
            f"latency={results['latency']:.1f}s"
        )

        assert results["accuracy"] >= cfg.accuracy_threshold - cfg.tolerance, (
            f"GSM8K accuracy {results['accuracy']:.4f} below "
            f"{cfg.accuracy_threshold - cfg.tolerance:.4f}"
        )
