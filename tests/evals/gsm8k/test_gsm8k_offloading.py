# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GSM8K correctness test for CPU KV offloading connectors.

Regression guard for stride computation bugs in the offloading worker
(e.g. https://github.com/vllm-project/vllm/pull/46888) and silent KV
cache data corruption during CPU offloading.

Covers both KV offloading connectors (OffloadingConnector and
SimpleCPUOffloadConnector) across four architecture families:
  - Hybrid Mamba (NemotronH: attention + Mamba)
  - Heterogeneous head dim (Gemma 4)
  - Hybrid GDN (Qwen 3.5: attention + GatedDeltaNet)
  - Compressed attention (DeepSeek-V4-Flash: CSA)

Usage:
    pytest -s -v evals/gsm8k/test_gsm8k_offloading.py
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


def _kv_transfer_config(connector: str, cpu_gib: int = 4) -> str:
    if connector == "OffloadingConnector":
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
    elif connector == "SimpleCPUOffloadConnector":
        return json.dumps(
            {
                "kv_connector": "SimpleCPUOffloadConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": cpu_gib << 30,
                },
            }
        )
    else:
        raise ValueError(f"Unknown connector: {connector}")


@dataclass
class OffloadingModelConfig:
    id: str
    model: str
    connector: str
    accuracy_threshold: float
    tolerance: float = 0.05
    extra_server_args: list[str] = field(default_factory=list)
    cpu_offload_gib: int = 4
    startup_timeout: int = 600


MODELS = [
    # ── OffloadingConnector ──────────────────────────────────────────
    OffloadingModelConfig(
        id="offloading-nemotron-h-8b",
        model="nvidia/Nemotron-H-8B-Base-8K",
        connector="OffloadingConnector",
        # Baseline ~0.49 on 200 questions (measured on GB200).
        accuracy_threshold=0.45,
    ),
    OffloadingModelConfig(
        id="offloading-gemma-4-e4b-it",
        model="google/gemma-4-E4B-it",
        connector="OffloadingConnector",
        # Baseline ~0.64 on 200 questions (measured on GB200).
        accuracy_threshold=0.55,
    ),
    OffloadingModelConfig(
        id="offloading-qwen3.5-35b",
        model="Qwen/Qwen3.5-35B-A3B",
        connector="OffloadingConnector",
        accuracy_threshold=0.75,
        extra_server_args=[
            "--enable-expert-parallel",
        ],
    ),
    OffloadingModelConfig(
        id="offloading-deepseek-v4-flash",
        model="deepseek-ai/DeepSeek-V4-Flash",
        connector="OffloadingConnector",
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
    # ── SimpleCPUOffloadConnector ────────────────────────────────────
    OffloadingModelConfig(
        id="simple-nemotron-h-8b",
        model="nvidia/Nemotron-H-8B-Base-8K",
        connector="SimpleCPUOffloadConnector",
        accuracy_threshold=0.45,
    ),
    OffloadingModelConfig(
        id="simple-gemma-4-e4b-it",
        model="google/gemma-4-E4B-it",
        connector="SimpleCPUOffloadConnector",
        accuracy_threshold=0.55,
    ),
    OffloadingModelConfig(
        id="simple-qwen3.5-35b",
        model="Qwen/Qwen3.5-35B-A3B",
        connector="SimpleCPUOffloadConnector",
        accuracy_threshold=0.75,
        extra_server_args=[
            "--enable-expert-parallel",
        ],
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
        _kv_transfer_config(cfg.connector, cfg.cpu_offload_gib),
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
            f"GSM8K + {cfg.connector} ({cfg.id}): "
            f"accuracy={results['accuracy']:.4f}, "
            f"invalid_rate={results['invalid_rate']:.3f}, "
            f"latency={results['latency']:.1f}s"
        )

        assert results["accuracy"] >= cfg.accuracy_threshold - cfg.tolerance, (
            f"GSM8K accuracy {results['accuracy']:.4f} below "
            f"{cfg.accuracy_threshold - cfg.tolerance:.4f}"
        )
