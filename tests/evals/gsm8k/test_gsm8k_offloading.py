# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GSM8K correctness test for CPU KV offloading connectors.

Regression guard for stride computation bugs in the offloading worker
(e.g. https://github.com/vllm-project/vllm/pull/46888) and silent KV
cache data corruption during CPU offloading.  Runs GSM8K twice, dropping
the GPU prefix cache (but not the CPU cache) between passes so the second
pass must reload offloaded KV data from CPU, and asserts on
``vllm:external_prefix_cache_hits`` to verify loading occurred.  For
OffloadingConnector, KV events are used to wait for the asynchronous
offload to complete between passes.

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
import socket
import time
from dataclasses import dataclass, field

import msgspec
import pytest
import requests
import zmq

from tests.utils import RemoteOpenAIServer
from vllm.distributed.kv_events import BlockStored, KVEventBatch
from vllm.platforms import current_platform

from .gsm8k_eval import evaluate_gsm8k

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

NUM_QUESTIONS = 200
NUM_FEWSHOT = 5

_KV_EVENTS_TOPIC = "kv-events"
_EVENT_POLL_MS = 1000
_OFFLOAD_SYNC_TIMEOUT = 180
_EXTERNAL_CACHE_COUNTERS = (
    "vllm:external_prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
)


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


class _CPUStoreSubscriber:
    """Collects BlockStored(medium="CPU") events, which OffloadingConnector
    emits once a block is committed to the CPU cache."""

    def __init__(self, endpoint: str, topic: str):
        self.sub = zmq.Context.instance().socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self.sub.connect(endpoint)
        self.decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

    def get_new_cpu_stored_blocks(self) -> int:
        num_blocks = 0
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        while dict(poller.poll(_EVENT_POLL_MS)).get(self.sub) == zmq.POLLIN:
            _, _, payload = self.sub.recv_multipart()
            for event in self.decoder.decode(payload).events:
                if isinstance(event, BlockStored) and event.medium == "CPU":
                    num_blocks += len(event.block_hashes)
        return num_blocks

    def close(self):
        self.sub.close()


def _force_engine_step(base_url: str) -> None:
    """Force an engine step so completed offload transfers get processed."""
    requests.post(
        f"{base_url}/v1/completions",
        json={"prompt": "0", "max_tokens": 1, "temperature": 0},
        timeout=60,
    ).raise_for_status()


def _wait_for_cpu_stores(subscriber: _CPUStoreSubscriber, base_url: str) -> None:
    """Wait until the asynchronous offload commits blocks to the CPU cache."""
    total_blocks = 0
    deadline = time.monotonic() + _OFFLOAD_SYNC_TIMEOUT
    while time.monotonic() < deadline:
        new_blocks = subscriber.get_new_cpu_stored_blocks()
        total_blocks += new_blocks
        if new_blocks == 0:
            if total_blocks > 0:
                return
            _force_engine_step(base_url)
    assert total_blocks > 0, f"no CPU store events within {_OFFLOAD_SYNC_TIMEOUT}s"


def _scrape_counter(base_url: str, name: str) -> float:
    text = requests.get(f"{base_url}/metrics", timeout=30).text
    values = [
        float(line.rpartition(" ")[2])
        for line in text.splitlines()
        if line.split("{", 1)[0].split(" ", 1)[0] in (name, f"{name}_total")
    ]
    assert values, f"{name} not found in /metrics"
    return sum(values)


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

    # Only OffloadingConnector emits CPU BlockStored events.
    use_kv_events = cfg.connector == "OffloadingConnector"
    if use_kv_events:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            events_port = s.getsockname()[1]
        server_args += [
            "--kv-events-config",
            json.dumps(
                {
                    "enable_kv_cache_events": True,
                    "publisher": "zmq",
                    "endpoint": f"tcp://*:{events_port}",
                    "topic": _KV_EVENTS_TOPIC,
                }
            ),
        ]

    with RemoteOpenAIServer(
        cfg.model,
        server_args,
        # /reset_prefix_cache requires dev mode.
        env_dict={"VLLM_SERVER_DEV_MODE": "1"},
        max_wait_seconds=cfg.startup_timeout,
    ) as server:
        base_url = f"http://{server.host}:{server.port}"
        subscriber = (
            _CPUStoreSubscriber(f"tcp://127.0.0.1:{events_port}", _KV_EVENTS_TOPIC)
            if use_kv_events
            else None
        )
        try:
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

                assert results["accuracy"] >= (
                    cfg.accuracy_threshold - cfg.tolerance
                ), (
                    f"GSM8K run {run_idx}/2 accuracy "
                    f"{results['accuracy']:.4f} below "
                    f"{cfg.accuracy_threshold - cfg.tolerance:.4f}"
                )

                if run_idx == 1:
                    # Wait for async offloading to finish before dropping the
                    # GPU prefix cache: in-flight transfers hold GPU blocks
                    # (failing the reset), and the second pass must not miss
                    # on the CPU cache.
                    if subscriber is not None:
                        _wait_for_cpu_stores(subscriber, base_url)
                    else:
                        for _ in range(3):
                            _force_engine_step(base_url)

                    before = [
                        _scrape_counter(base_url, name)
                        for name in _EXTERNAL_CACHE_COUNTERS
                    ]
                    requests.post(
                        f"{base_url}/reset_prefix_cache",
                        params={"reset_external": "false"},
                        timeout=30,
                    ).raise_for_status()

            queries, hits = (
                _scrape_counter(base_url, name) - b
                for name, b in zip(_EXTERNAL_CACHE_COUNTERS, before)
            )
            print(
                f"{cfg.id}: run 2/2 loaded {hits:.0f}/{queries:.0f} "
                "externally queried tokens from the CPU cache"
            )
            assert queries > 0 and hits > 0, "no KV data was loaded from CPU"
        finally:
            if subscriber is not None:
                subscriber.close()
