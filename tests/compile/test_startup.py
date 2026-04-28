# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold start and warm start tests for vLLM-compile.

Cold start runs in a forked child (must fork before CUDA init) which
populates on-disk caches and asserts cold-start counters.  Warm start
then runs in the parent with clean in-memory state but populated caches.
"""

import multiprocessing as mp

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


def test_parallel_compile_pool_oom_fallback(
    monkeypatch, vllm_runner, fresh_vllm_cache
):
    """Two-part test:
    1. Verify PyTorch AsyncCompile.warm_pool() raises an exception under memory
       pressure, proving it can fail with OOM-like errors.
    2. Verify vLLM catches the exception and gracefully falls back to 1 compile
       process when warm_pool() fails.
    """
    import subprocess
    import sys

    # === Step 1: Prove warm_pool() can raise an exception under memory limit ===
    # Run in a subprocess with restricted memory to observe real OOM behavior.
    probe_code = """
import resource
# Restrict virtual memory to 500MB
resource.setrlimit(resource.RLIMIT_AS, (500 * 1024**2, 500 * 1024**2))
try:
    from torch._inductor.async_compile import AsyncCompile
    AsyncCompile.warm_pool()
    print("NO_EXCEPTION")
except Exception as e:
    print(f"EXCEPTION:{type(e).__name__}")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe_code], capture_output=True, text=True, timeout=60
    )
    stdout = result.stdout.strip()
    # Verify that warm_pool() fails with some exception under memory pressure
    assert stdout.startswith("EXCEPTION:"), (
        f"Expected warm_pool() to raise an exception under memory limit, "
        f"got: {stdout}"
    )
    exception_type = stdout.split("EXCEPTION:")[1]

    # === Step 2: Mock the same exception type to test vLLM's fallback logic ===
    import torch
    import torch._inductor.async_compile as async_compile

    monkeypatch.setenv("VLLM_COMPILE_PROCESSES", "4")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Mock warm_pool to raise the same exception type observed in Step 1
    exc_class = getattr(__builtins__, exception_type, Exception)
    if not isinstance(exc_class, type):
        exc_class = Exception

    def mock_warm_pool_fail(cls):
        raise exc_class("Simulated OOM in warm_pool")

    monkeypatch.setattr(async_compile.AsyncCompile, "warm_pool", mock_warm_pool_fail)

    # vLLM should catch the exception and fallback gracefully without crashing
    _run_vllm(vllm_runner)

    # Verify fallback: compile_threads should be reset to 1
    assert torch._inductor.config.compile_threads == 1, (
        f"compile_threads should be 1 after OOM fallback, "
        f"got {torch._inductor.config.compile_threads}"
    )
