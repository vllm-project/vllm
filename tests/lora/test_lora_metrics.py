# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loaded-adapter metrics reach the frontend's Prometheus registry, from
adapter RPCs on an idle engine, from adapters a step loads, and from dummy
batches."""

import time
from collections.abc import Callable

import pytest

import vllm.envs as envs
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.metrics.reader import Gauge

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


def _make_engine() -> LLMEngine:
    engine_args = EngineArgs(
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


@pytest.mark.parametrize("use_v2_model_runner", ["0", "1"])
def test_lora_load_metrics_follow_dummy_batches(monkeypatch, use_v2_model_runner):
    """After a dummy batch the reported adapters match the worker's."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", use_v2_model_runner)
    llm = _make_engine()

    llm.add_lora(make_lora_request(1))
    llm.add_lora(make_lora_request(3))
    _series_after(llm, lambda s: len(s) == 2)

    llm.engine_core.execute_dummy_batch()
    loaded = {f"adapter-{i}" for i in llm.list_loras()}
    _series_after(llm, lambda s: {name for name, _, _ in s} == loaded)
    assert _gauge(llm, "vllm:num_cpu_loaded_lora_adapters") == len(loaded)
