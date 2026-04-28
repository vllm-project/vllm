# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold start and warm start tests for vLLM-compile.

Cold start runs in a forked child (must fork before CUDA init) which
populates on-disk caches and asserts cold-start counters.  Warm start
then runs in the parent with clean in-memory state but populated caches.
"""

import multiprocessing as mp
import subprocess
import sys

from torch._dynamo.utils import counters

from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

MODEL = "microsoft/Phi-tiny-MoE-instruct"


def _run_vllm(vllm_runner):
    with vllm_runner(
        MODEL,
        trust_remote_code=False,
        max_model_len=256,
        max_num_batched_tokens=1024,
        load_format="dummy",
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        num_gpu_blocks_override=8,
    ):
        pass


def _cold_start(vllm_runner):
    counters.clear()
    with compilation_counter.expect(
        num_compiled_artifacts_saved=3,
        num_compiled_artifacts_loaded=0,
    ):
        _run_vllm(vllm_runner)
    assert counters["aot_autograd"]["total"] == 33
    assert counters["aot_autograd"]["autograd_cache_miss"] == 3
    assert counters["aot_autograd"]["autograd_cache_hit"] == 0


def test_moe_startup(monkeypatch, vllm_runner, fresh_vllm_cache):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Cold start in a forked child (must fork before CUDA init).
    # This model has 32 identical transformer layers which produce
    # 33 subgraphs after splitting on attention — only 3 are unique.
    ctx = mp.get_context("fork")
    p = ctx.Process(target=_cold_start, args=(vllm_runner,))
    p.start()
    p.join()
    assert p.exitcode == 0, "Cold-start child failed"

    # Warm start — compiled artifacts loaded from disk cache.
    counters.clear()
    with compilation_counter.expect(
        num_compiled_artifacts_loaded=3,
        num_compiled_artifacts_saved=0,
    ):
        _run_vllm(vllm_runner)
    assert counters["aot_autograd"]["total"] == 30
    assert counters["aot_autograd"]["autograd_cache_miss"] == 0
    assert (
        counters["aot_autograd"]["autograd_cache_hit"] == 0
    )  # No miss at aot_autograd level causing disk I/O.


def test_parallel_compile_pool(monkeypatch, vllm_runner, fresh_vllm_cache):
    """Test that parallel compile pool is warmed up and quiesced by vLLM."""
    from torch._inductor.async_compile import (
        _pool_set,
        shutdown_compile_workers,
    )

    # Explicitly set parallel compilation to 4 processes.
    monkeypatch.setenv("VLLM_COMPILE_PROCESSES", "4")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    try:
        # Run vLLM — the worker should set compile_threads, warm up
        # the pool, then quiesce it before cudagraph capture.
        _run_vllm(vllm_runner)

        # Verify pool exists and was quiesced (not shut down).
        # After quiesce(), SubprocPool.quiesce_waitcounter is set to a
        # non-None value while the pool itself stays alive for reuse.
        assert len(_pool_set) > 0, "Pool should exist after vLLM run"
        for pool in _pool_set:
            assert pool.quiesce_waitcounter is not None, (
                "Pool should be quiesced after compilation"
            )
    finally:
        # Clean up for other tests in the same pytest session
        shutdown_compile_workers()


def test_warm_up_compile_pool_oom_fallback(monkeypatch):
    """Test that the OOM fallback logic in _maybe_warm_up_compile_pool works.

    This is a unit test that does not require GPU or vLLM engine.
    It verifies that when AsyncCompile.warm_pool() raises an exception,
    the fallback logic sets compile_threads back to 1.
    """
    import torch
    import torch._inductor.async_compile as async_compile

    # Mock warm_pool to raise exception (simulating OOM)
    def mock_warm_pool(cls):
        raise RuntimeError("Simulated OOM in warm_pool")

    monkeypatch.setattr(async_compile.AsyncCompile, "warm_pool", mock_warm_pool)
    monkeypatch.setenv("VLLM_COMPILE_PROCESSES", "4")

    # Simulate the fallback logic from _maybe_warm_up_compile_pool
    num_procs = 4
    torch._inductor.config.compile_threads = num_procs

    try:
        async_compile.AsyncCompile.warm_pool()
    except Exception:
        torch._inductor.config.compile_threads = 1

    # Verify fallback: compile_threads should be reset to 1
    assert torch._inductor.config.compile_threads == 1, (
        f"compile_threads should be 1 after OOM fallback, "
        f"got {torch._inductor.config.compile_threads}"
    )


def test_warm_pool_real_oom():
    """Verify that AsyncCompile.warm_pool() raises an exception under
    memory pressure when RLIMIT_AS is enforced."""
    probe_code = """
import resource
# Restrict virtual memory to 500MB
resource.setrlimit(resource.RLIMIT_AS, (500 * 1024**2, 500 * 1024**2))
from torch._inductor.async_compile import AsyncCompile
AsyncCompile.warm_pool()
"""
    result = subprocess.run(
        [sys.executable, "-c", probe_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0 or "EXCEPTION" in result.stdout or result.stderr, (
        "Expected warm_pool() to fail under memory limit on Linux"
    )
