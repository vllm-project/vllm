# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from benchmarks.kernels import benchmark_moe


def test_clear_triton_jit_cache_uses_runtime_cache(monkeypatch):
    clear_calls = []
    runtime = SimpleNamespace(
        cache=SimpleNamespace(clear=lambda: clear_calls.append(True))
    )
    monkeypatch.setattr(benchmark_moe, "triton", SimpleNamespace(runtime=runtime))

    benchmark_moe._clear_triton_jit_cache()

    assert clear_calls == [True]


def test_clear_triton_jit_cache_uses_jit_registry(monkeypatch):
    first_device_cache = {"cuda:0": object()}
    second_device_cache = {"cuda:0": object(), "cuda:1": object()}
    registry = {
        "first": SimpleNamespace(device_caches=first_device_cache),
        "second": SimpleNamespace(device_caches=second_device_cache),
    }
    runtime = SimpleNamespace(
        jit=SimpleNamespace(_triton_jit_function_registry=registry)
    )
    monkeypatch.setattr(benchmark_moe, "triton", SimpleNamespace(runtime=runtime))

    benchmark_moe._clear_triton_jit_cache()

    assert first_device_cache == {}
    assert second_device_cache == {}


def test_clear_triton_cache_releases_references_before_allocator(monkeypatch):
    clear_order = []
    monkeypatch.setattr(
        benchmark_moe,
        "_clear_triton_jit_cache",
        lambda: clear_order.append("triton"),
    )
    monkeypatch.setattr(benchmark_moe.gc, "collect", lambda: clear_order.append("gc"))
    monkeypatch.setattr(benchmark_moe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        benchmark_moe.torch.accelerator,
        "empty_cache",
        lambda: clear_order.append("allocator"),
    )

    benchmark_moe.clear_triton_cache()

    assert clear_order == ["triton", "gc", "allocator"]


def test_run_with_oom_recovery_clears_and_retries_once(monkeypatch):
    events = []

    def run_config():
        events.append("run")
        if events == ["run"]:
            raise benchmark_moe.torch.OutOfMemoryError("synthetic OOM")
        return 1.25

    monkeypatch.setattr(
        benchmark_moe,
        "clear_triton_cache",
        lambda: events.append("clear"),
    )

    result = benchmark_moe._run_with_oom_recovery(run_config)

    assert result == 1.25
    assert events == ["run", "clear", "run"]


def test_run_with_oom_recovery_propagates_second_oom(monkeypatch):
    events = []

    def run_config():
        events.append("run")
        raise benchmark_moe.torch.OutOfMemoryError("persistent OOM")

    monkeypatch.setattr(
        benchmark_moe,
        "clear_triton_cache",
        lambda: events.append("clear"),
    )

    with pytest.raises(benchmark_moe.torch.OutOfMemoryError):
        benchmark_moe._run_with_oom_recovery(run_config)

    assert events == ["run", "clear", "run"]
