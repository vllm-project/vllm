# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the MRV2 JIT warmup coverage populated on ROCm.

Covers both the sampler warmup registry (dense models) and the Qwen3.5 GDN
(Gated DeltaNet) linear-attention path.
"""

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


def _worker_gdn_layer_count(worker):
    from vllm.model_executor.warmup.qwen_triton_warmup import (
        _iter_qwen_gdn_layers,
    )

    forward_context = worker.model_runner.compilation_config.static_forward_context
    return sum(1 for _ in _iter_qwen_gdn_layers(forward_context))


def _gdn_hf_overrides(hf_config, **kwargs):
    """Truncate to a two-layer hybrid: one GDN + one full-attention layer.

    Qwen3.5 alternates GDN with a full-attention layer every 4th layer
    (3:1 linear:full). ``dummy_hf_overrides`` reduces ``num_hidden_layers``
    to one, which leaves a GDN layer but drops every full-attention layer;
    keeping one of each matches the hybrid layout the model is served with.
    """
    config = dummy_hf_overrides(hf_config, **kwargs)
    text_config = hf_config.get_text_config()
    text_config.num_hidden_layers = 2
    text_config.layer_types = ["linear_attention", "full_attention"]
    return config


@create_new_process_for_each_test("spawn")
def _run_rocm_shape_battery(model: str, hf_overrides, expect_gdn=False) -> None:
    llm = LLM(
        model,
        max_model_len=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.03,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        load_format="dummy",
        hf_overrides=hf_overrides,
        enforce_eager=False,
        jit_monitor_mode="error",
    )
    try:
        states = llm.collective_rpc(_worker_monitor_state, timeout=30)
        assert states and all(state == (True, "error") for state in states)
        if expect_gdn:
            counts = llm.collective_rpc(_worker_gdn_layer_count, timeout=30)
            assert counts and all(count > 0 for count in counts)
        _run_shape_battery(llm)
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=30)


def _isolate_rocm_jit_caches(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Fresh caches and a spawned engine core per test.

    A previous run's cache must not hide a missing warmup specialization.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))


def test_v2_sampler_warmup_rocm(monkeypatch, tmp_path):
    """Warmup must cover the existing prefill, decode, and sampler battery.

    A fresh cache and child process prevent previous model executions from
    hiding a missing warmup specialization.
    """
    _isolate_rocm_jit_caches(monkeypatch, tmp_path)
    _run_rocm_shape_battery("Qwen/Qwen3-0.6B", dummy_hf_overrides)


def test_qwen_gdn_no_runtime_jit_rocm(monkeypatch, tmp_path):
    """The Qwen3.5 GDN path must not JIT-compile during the standard
    inference battery on ROCm.

    Qwen3.5-0.8B exercises the QwenGatedDeltaNet linear-attention kernels
    (causal-conv update, gated-delta recurrent decode, FLA chunk prefill)
    that dense models never touch. Under the default graph configuration,
    JIT warmup and cudagraph capture must cover every compile key the
    battery needs; a miss fails the test via jit_monitor_mode="error".

    Runs the in-tree generic Triton GDN path (VLLM_ROCM_USE_AITER=0), so
    the regression does not depend on optional AITER availability; the
    AITER GDN path is out of scope for this test.
    """
    _isolate_rocm_jit_caches(monkeypatch, tmp_path)
    _run_rocm_shape_battery("Qwen/Qwen3.5-0.8B", _gdn_hf_overrides, expect_gdn=True)
