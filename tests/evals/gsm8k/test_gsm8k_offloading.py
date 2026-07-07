# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GSM8K correctness test for CPU KV offloading connectors.

Regression guard for stride computation bugs in the offloading worker
(e.g. https://github.com/vllm-project/vllm/pull/46888) and silent KV
cache data corruption during CPU offloading.  Runs GSM8K twice, dropping
the GPU prefix cache (but not the CPU cache) between runs so the second
run reloads offloaded KV data from CPU.  The reload is best effort: the
reset only succeeds once in-flight offload transfers have released their
GPU blocks, so it is retried with a bounded timeout.

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
import time
from dataclasses import dataclass, field

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

from .gsm8k_eval import evaluate_gsm8k

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

NUM_QUESTIONS = 200
NUM_FEWSHOT = 5

_OFFLOAD_SYNC_TIMEOUT = 60


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


def _force_engine_step(base_url: str) -> None:
    """Force an engine step so completed offload transfers get processed."""
    requests.post(
        f"{base_url}/v1/completions",
        json={"prompt": "0", "max_tokens": 1, "temperature": 0},
        timeout=60,
    ).raise_for_status()


def _reset_gpu_prefix_cache(base_url: str) -> None:
    """Best-effort drop of the GPU prefix cache, keeping the CPU (connector)
    cache, so the next run reloads KV data through the connector.

    The reset fails while asynchronous offload transfers still hold GPU
    blocks; retry briefly rather than failing the test.  Requires
    VLLM_SERVER_DEV_MODE=1.
    """
    deadline = time.monotonic() + _OFFLOAD_SYNC_TIMEOUT
    while time.monotonic() < deadline:
        resp = requests.post(
            f"{base_url}/reset_prefix_cache",
            params={"reset_external": "false"},
            timeout=30,
        )
        resp.raise_for_status()
        if resp.json().get("success"):
            return
        _force_engine_step(base_url)


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
            "--tensor-parallel-size",
            "2",
            "--enable-expert-parallel",
        ],
        startup_timeout=1200,
    ),
    OffloadingModelConfig(
        id="offloading-deepseek-v4-flash",
        model="deepseek-ai/DeepSeek-V4-Flash",
        connector="OffloadingConnector",
        # Baseline ~0.97 on 200 questions (measured on GB200).
        accuracy_threshold=0.90,
        extra_server_args=[
            "--tensor-parallel-size",
            "4",
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
            "--tensor-parallel-size",
            "2",
            "--enable-expert-parallel",
        ],
        startup_timeout=1200,
    ),
    OffloadingModelConfig(
        id="simple-deepseek-v4-flash",
        model="deepseek-ai/DeepSeek-V4-Flash",
        connector="SimpleCPUOffloadConnector",
        accuracy_threshold=0.90,
        extra_server_args=[
            "--tensor-parallel-size",
            "4",
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
    if "--tensor-parallel-size" in cfg.extra_server_args:
        tp_size = int(
            cfg.extra_server_args[
                cfg.extra_server_args.index("--tensor-parallel-size") + 1
            ]
        )
        if current_platform.device_count() < tp_size:
            pytest.skip(f"Requires {tp_size} GPUs")

    # Prefix caching must be explicitly enabled: SimpleCPUOffloadConnector requires it.
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
        # /reset_prefix_cache requires dev mode.
        env_dict={"VLLM_SERVER_DEV_MODE": "1"},
        max_wait_seconds=cfg.startup_timeout,
    ) as server:
        base_url = f"http://{server.host}:{server.port}"

        for run_idx in range(1, 3):
            results = evaluate_gsm8k(
                num_questions=NUM_QUESTIONS,
                num_shots=NUM_FEWSHOT,
                host=f"http://{server.host}",
                port=server.port,
            )

            print(
                f"GSM8K run {run_idx}/2 + {cfg.connector} ({cfg.id}): "
                f"accuracy={results['accuracy']:.4f}, "
                f"invalid_rate={results['invalid_rate']:.3f}, "
                f"latency={results['latency']:.1f}s"
            )

            assert results["accuracy"] >= (cfg.accuracy_threshold - cfg.tolerance), (
                f"GSM8K run {run_idx}/2 accuracy "
                f"{results['accuracy']:.4f} below "
                f"{cfg.accuracy_threshold - cfg.tolerance:.4f}"
            )

            if run_idx == 1:
                # Give the async offload a moment to finish, then drop the
                # GPU prefix cache so the second run reloads from CPU.
                time.sleep(1)
                _reset_gpu_prefix_cache(base_url)
