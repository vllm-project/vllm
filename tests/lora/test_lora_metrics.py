# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loaded-adapter metrics reach the frontend's Prometheus registry through
the engine notification channel, with no request traffic."""

import time
from collections.abc import Callable

from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.metrics.reader import Gauge

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_MODULE_PATH = "charent/self_cognition_Alice"
LORA_RANK = 8


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


def test_lora_load_metrics_track_adapter_changes():
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
    llm = LLMEngine.from_engine_args(engine_args)

    def series_after(expected: Callable[[dict], bool], timeout_s: float = 60.0):
        """Step the idle engine, one notification-only output per step, until
        the loaded series satisfy `expected`."""
        deadline = time.monotonic() + timeout_s
        series = _loaded_series(llm)
        while not expected(series):
            assert time.monotonic() < deadline, f"metrics never converged: {series}"
            llm.step()
            series = _loaded_series(llm)
        return series

    llm.add_lora(make_lora_request(1))
    llm.add_lora(make_lora_request(2))
    assert series_after(lambda s: len(s) == 2) == {
        ("adapter-1", "gpu", "false"): 1.0,
        ("adapter-2", "gpu", "false"): 1.0,
    }

    # GPU slots are full: the third add evicts adapter-1 to the CPU tier.
    llm.add_lora(make_lora_request(3))
    assert series_after(lambda s: len(s) == 3) == {
        ("adapter-1", "cpu", "false"): 1.0,
        ("adapter-2", "gpu", "false"): 1.0,
        ("adapter-3", "gpu", "false"): 1.0,
    }
    assert _gauge(llm, "vllm:num_gpu_loaded_lora_adapters") == 2
    assert _gauge(llm, "vllm:num_cpu_loaded_lora_adapters") == 3

    llm.pin_lora(1)
    series = series_after(lambda s: ("adapter-1", "gpu", "true") in s)
    assert ("adapter-1", "cpu", "false") not in series

    llm.remove_lora(2)
    series = series_after(lambda s: not any(n == "adapter-2" for n, _, _ in s))
    assert len(series) == 2
    assert _gauge(llm, "vllm:num_cpu_loaded_lora_adapters") == 2
