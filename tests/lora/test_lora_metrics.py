# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loaded-adapter metrics reach the frontend's Prometheus registry, from
adapter RPCs on an idle engine and from adapters a step loads, once per load
under TP, PP, DP and several API servers, without stale series."""

import time
from collections.abc import Callable
from typing import Any

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families

import vllm.envs as envs
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.metrics.reader import Gauge

from ..utils import RemoteOpenAIServer, multi_gpu_marks

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_MODULE_PATH = "charent/self_cognition_Alice"
LORA_RANK = 8

pytestmark = pytest.mark.timeout(300)


@pytest.fixture(autouse=True)
def reset_envs_cache():
    """An in-process engine core caches environment variables."""
    yield
    envs.disable_envs_cache()


def make_lora_request(lora_id: int):
    return LoRARequest(
        lora_name=f"adapter-{lora_id}", lora_int_id=lora_id, lora_path=LORA_MODULE_PATH
    )


def _loaded_series(llm: LLMEngine) -> dict[tuple[str, str, str], float]:
    """(adapter_name, level, pinned) -> value, for the live series."""
    return {
        (m.labels["adapter_name"], m.labels["level"], m.labels["pinned"]): m.value
        for m in llm.get_metrics()
        if isinstance(m, Gauge) and m.name == "vllm:lora_adapter_loaded"
    }


def _gauge(llm: LLMEngine, name: str) -> float:
    for m in llm.get_metrics():
        if isinstance(m, Gauge) and m.name == name:
            return m.value
    raise AssertionError(f"missing gauge {name}")


def _series_after(
    llm: LLMEngine, expected: Callable[[dict], bool], timeout_s: float = 60.0
) -> dict[tuple[str, str, str], float]:
    """Step the engine until the loaded series satisfy `expected`."""
    deadline = time.monotonic() + timeout_s
    series = _loaded_series(llm)
    while not expected(series):
        assert time.monotonic() < deadline, f"metrics never converged: {series}"
        llm.step()
        series = _loaded_series(llm)
    return series


def _make_engine(**kwargs: Any) -> LLMEngine:
    engine_args = EngineArgs(
        **kwargs,
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=3,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        disable_log_stats=False,
    )
    return LLMEngine.from_engine_args(engine_args)


LoRASamples = dict[tuple[str, ...], float]


def _lora_samples(server: RemoteOpenAIServer) -> LoRASamples:
    """From /metrics, keyed by engine: load counts by transition, the
    adapter counts per tier, and the adapters whose loaded series is 1."""
    response = requests.get(server.url_for("metrics"), timeout=10)
    response.raise_for_status()
    samples: LoRASamples = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            engine = sample.labels.get("engine", "")
            if sample.name == "vllm:lora_adapter_load_seconds_count":
                samples["count", engine, sample.labels["transition"]] = sample.value
            elif sample.name == "vllm:num_gpu_loaded_lora_adapters":
                samples["num_gpu", engine] = sample.value
            elif sample.name == "vllm:num_cpu_loaded_lora_adapters":
                samples["num_cpu", engine] = sample.value
            elif sample.name == "vllm:lora_adapter_loaded":
                key = ("loaded", engine, sample.labels["adapter_name"])
                samples[*key, sample.labels["level"]] = sample.value
    return samples


def _loaded(samples: LoRASamples, engine: str) -> set[str]:
    """Adapters whose loaded series reads 1."""
    return {
        key[2]
        for key, value in samples.items()
        if key[:2] == ("loaded", engine) and value == 1
    }


def _samples_after(
    server: RemoteOpenAIServer,
    expected: Callable[[LoRASamples], bool],
    timeout_s: float = 60.0,
) -> LoRASamples:
    """Poll /metrics until `expected` holds, then once more after a pause so
    that late duplicate reports would show up."""
    deadline = time.monotonic() + timeout_s
    while not expected(samples := _lora_samples(server)):
        assert time.monotonic() < deadline, samples
        time.sleep(0.5)
    time.sleep(2)
    return _lora_samples(server)


@pytest.mark.parametrize(
    ("parallel_args", "num_engines"),
    [
        pytest.param(["--api-server-count", "2"], 1, id="api_servers2"),
        pytest.param(
            ["--tensor-parallel-size", "2"],
            1,
            id="tp2",
            marks=multi_gpu_marks(num_gpus=2),
        ),
        pytest.param(
            ["--pipeline-parallel-size", "2"],
            1,
            id="pp2",
            marks=multi_gpu_marks(num_gpus=2),
        ),
        pytest.param(
            ["--data-parallel-size", "2"],
            2,
            id="dp2",
            marks=multi_gpu_marks(num_gpus=2),
        ),
    ],
)
def test_lora_load_metrics_count_once(parallel_args: list[str], num_engines: int):
    """Each load is counted once per engine, whichever rank reports it and
    however many API servers share the registry."""
    adapters = ["a1", "a2", "a3"]
    args = [
        "--enable-lora",
        "--max-loras", "2",
        "--max-cpu-loras", "3",
        "--max-lora-rank", str(LORA_RANK),
        "--max-model-len", "128",
        "--gpu-memory-utilization", "0.8",
        "--enforce-eager",
        *parallel_args,
        "--lora-modules", *[f"{name}={LORA_MODULE_PATH}" for name in adapters],
    ]  # fmt: skip
    engines = [str(i) for i in range(num_engines)]
    with RemoteOpenAIServer(MODEL_PATH, args) as server:
        client = server.get_client()
        for name in [*adapters, adapters[0]]:
            client.completions.create(model=name, prompt="Who are you?", max_tokens=4)

        # Each adapter is read from disk once per engine: the CPU tier holds
        # all three.
        samples = _samples_after(
            server,
            lambda s: all(
                s.get(("count", e, "load"), 0) >= len(adapters) for e in engines
            ),
        )
        for engine in engines:
            assert samples["count", engine, "load"] == len(adapters), samples
            assert _loaded(samples, engine) == set(adapters), samples
            assert samples["num_gpu", engine] == 2, samples
            assert samples["num_cpu", engine] == 3, samples


@pytest.mark.parametrize("api_server_count", [1, 2])
def test_lora_loaded_series_follow_eviction(api_server_count: int):
    """Four adapters share three CPU slots and two GPU slots, so adapters are
    evicted and change tier. Every series left must read 1: a multiprocess
    registry shared by several API servers cannot remove one."""
    adapters = ["a1", "a2", "a3", "a4"]
    args = [
        "--enable-lora",
        "--max-loras", "2",
        "--max-cpu-loras", "3",
        "--max-lora-rank", str(LORA_RANK),
        "--max-model-len", "128",
        "--gpu-memory-utilization", "0.8",
        "--enforce-eager",
        "--api-server-count", str(api_server_count),
        "--lora-modules", *[f"{name}={LORA_MODULE_PATH}" for name in adapters],
    ]  # fmt: skip
    with RemoteOpenAIServer(MODEL_PATH, args) as server:
        client = server.get_client()
        for name in [*adapters, adapters[0]]:
            client.completions.create(model=name, prompt="Who are you?", max_tokens=4)

        samples = _samples_after(
            server,
            lambda s: len(_loaded(s, "0")) == 3 and s.get(("num_gpu", "0")) == 2,
        )
        series = {key: value for key, value in samples.items() if key[0] == "loaded"}
        assert set(series.values()) == {1}, series
        assert len(series) == samples["num_cpu", "0"] == 3, series
        assert sum(key[3] == "gpu" for key in series) == 2, series
        assert "a1" in _loaded(samples, "0"), series


@pytest.mark.parametrize("multiprocessing", ["1", "0"])
def test_lora_load_metrics_track_adapter_changes(monkeypatch, multiprocessing):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", multiprocessing)
    llm = _make_engine()

    llm.add_lora(make_lora_request(1))
    llm.add_lora(make_lora_request(2))
    assert _series_after(llm, lambda s: len(s) == 2) == {
        ("adapter-1", "gpu", "false"): 1.0,
        ("adapter-2", "gpu", "false"): 1.0,
    }

    # GPU slots are full: the third add evicts adapter-1 to the CPU tier.
    llm.add_lora(make_lora_request(3))
    assert _series_after(llm, lambda s: len(s) == 3) == {
        ("adapter-1", "cpu", "false"): 1.0,
        ("adapter-2", "gpu", "false"): 1.0,
        ("adapter-3", "gpu", "false"): 1.0,
    }
    assert _gauge(llm, "vllm:num_gpu_loaded_lora_adapters") == 2
    assert _gauge(llm, "vllm:max_gpu_lora_adapters") == 2
    assert _gauge(llm, "vllm:max_cpu_lora_adapters") == 3
    assert _gauge(llm, "vllm:num_cpu_loaded_lora_adapters") == 3

    llm.pin_lora(1)
    series = _series_after(llm, lambda s: ("adapter-1", "gpu", "true") in s)
    assert ("adapter-1", "cpu", "false") not in series

    llm.remove_lora(2)
    series = _series_after(llm, lambda s: not any(n == "adapter-2" for n, _, _ in s))
    assert len(series) == 2
    assert _gauge(llm, "vllm:num_cpu_loaded_lora_adapters") == 2

    # Not added up front: the step loads it, and the snapshot comes back on that
    # step's ModelRunnerOutput.
    llm.add_request(
        "req-4",
        "Who are you?",
        SamplingParams(max_tokens=4),
        lora_request=make_lora_request(4),
    )
    while llm.has_unfinished_requests():
        llm.step()
    series = _series_after(llm, lambda s: ("adapter-4", "gpu", "false") in s)
    assert ("adapter-1", "gpu", "true") in series
    assert _gauge(llm, "vllm:num_gpu_loaded_lora_adapters") == 2


def test_lora_rpc_while_a_batch_is_in_flight(monkeypatch):
    """With async scheduling the engine core runs adapter RPCs while a step's
    output is still pending. The RPC's snapshot is newer than that step's, so
    the reported adapters must match the worker's caches once both arrive."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = _make_engine(async_scheduling=True)

    llm.add_lora(make_lora_request(1))
    llm.add_lora(make_lora_request(2))
    _series_after(llm, lambda s: len(s) == 2)

    # The step loads adapter-3, so its output carries a snapshot.
    llm.add_request(
        "req-3",
        "Who are you?",
        SamplingParams(max_tokens=8),
        lora_request=make_lora_request(3),
    )
    assert llm.step() == []
    assert isinstance(llm.engine_core, InprocClient)
    assert llm.engine_core.engine_core.batch_queue, "the step is not in flight"

    llm.remove_lora(1)
    while llm.has_unfinished_requests():
        llm.step()

    loaded = {f"adapter-{i}" for i in llm.list_loras()}
    assert loaded == {"adapter-2", "adapter-3"}
    assert {name for name, _, _ in _loaded_series(llm)} == loaded
