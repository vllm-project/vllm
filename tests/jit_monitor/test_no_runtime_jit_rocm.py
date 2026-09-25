# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the MRV2 sampler warmup registry populated on ROCm."""

import pytest

from vllm import LLM
from vllm.platforms import current_platform

from ..models.utils import dummy_hf_overrides
from ..utils import create_new_process_for_each_test
from .test_no_runtime_jit import _run_shape_battery

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="Requires ROCm")


def _worker_monitor_state(worker):
    from vllm.utils import jit_monitor

    return (
        jit_monitor.is_active(),
        worker.observability_config.jit_monitor_mode,
    )


@create_new_process_for_each_test("spawn")
def _run_rocm_shape_battery() -> None:
    llm = LLM(
        "Qwen/Qwen3-0.6B",
        max_model_len=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.03,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        load_format="dummy",
        hf_overrides=dummy_hf_overrides,
        enforce_eager=False,
        jit_monitor_mode="error",
    )
    try:
        states = llm.collective_rpc(_worker_monitor_state, timeout=30)
        assert states and all(state == (True, "error") for state in states)
        _run_shape_battery(llm)
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=30)


def test_v2_sampler_warmup_rocm(monkeypatch, tmp_path):
    """Warmup must cover the existing prefill, decode, and sampler battery.

    A fresh cache and child process prevent previous model executions from
    hiding a missing warmup specialization.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    _run_rocm_shape_battery()
