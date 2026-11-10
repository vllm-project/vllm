# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Rust scheduler (`RustScheduler`, `RustAsyncScheduler`) must behave
exactly like `Scheduler` / `AsyncScheduler`: differential runs of the fuzzer
in `rust_core_diff.py`, plus how it and its native collaborators are
selected."""

import pytest

from tests.v1.core.rust_core_diff import FuzzConfig, build_scheduler, run_one
from tests.v1.core.utils import create_scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

pytest.importorskip("vllm._rust_core")

from vllm.v1.core.sched import rust_scheduler  # noqa: E402
from vllm.v1.core.sched.rust_scheduler import (  # noqa: E402
    RustAsyncScheduler,
    RustScheduler,
    rust_scheduler_cls,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "sample", ["base", "extended", "nixl", "mooncake", "structured", "streaming"]
)
@pytest.mark.parametrize("seed", range(4))
def test_matches_python_scheduler(seed: int, sample: str, use_async: bool):
    """Every SchedulerOutput, EngineCoreOutputs, KV event, stat and grammar
    bitmask matches, with the native NIXL and Mooncake store scheduler halves
    (`run_one` asserts they are used)."""
    cfg = getattr(FuzzConfig, f"sample_{sample}" if sample != "base" else "sample")(
        seed
    )
    if cfg.layout == "dsv41" and sample == "mooncake":
        pytest.skip("the DSV4.1 layout is covered by the fuzzer")
    ref_cls, test_cls = (
        (AsyncScheduler, RustAsyncScheduler)
        if use_async
        else (Scheduler, RustScheduler)
    )
    ok, msg = run_one(
        cfg,
        lambda cfg: build_scheduler(cfg, ref_cls),
        lambda cfg: build_scheduler(cfg, test_cls),
        use_async=use_async,
    )
    assert ok, msg


def test_native_hooks_can_be_disabled(monkeypatch):
    """`VLLM_RUST_CORE_NATIVE_HOOKS=0` keeps the Python connector hooks, with
    the same behavior."""
    monkeypatch.setenv("VLLM_RUST_CORE_NATIVE_HOOKS", "0")
    cfg = FuzzConfig.sample_nixl(2)
    scheduler = build_scheduler(cfg, RustScheduler)
    assert scheduler._core.connector_impl == "python"
    ok, msg = run_one(
        cfg,
        lambda cfg: build_scheduler(cfg, Scheduler),
        lambda cfg: build_scheduler(cfg, RustScheduler),
    )
    assert ok, msg


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_env_selects_rust_scheduler_with_python_fallback(
    async_scheduling: bool, monkeypatch
):
    """`VLLM_USE_RUST_CORE` picks the Rust scheduler, and the Python one for
    configurations the Rust one rejects."""
    cls = rust_scheduler_cls(async_scheduling)
    scheduler = create_scheduler(async_scheduling=async_scheduling, scheduler_cls=cls)
    assert isinstance(
        scheduler, RustAsyncScheduler if async_scheduling else RustScheduler
    )

    base = create_scheduler(async_scheduling=async_scheduling)
    args = dict(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        structured_output_manager=base.structured_output_manager,
        block_size=base.block_size,
    )
    monkeypatch.setattr(
        rust_scheduler, "unsupported_reason", lambda *args: "a test configuration"
    )
    fallback = cls(**args)
    assert type(fallback) is (AsyncScheduler if async_scheduling else Scheduler)
    with pytest.raises(NotImplementedError, match="a test configuration"):
        RustScheduler(**args)
