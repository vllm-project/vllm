# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton-cache PTX-fallback guard in vllm.platforms.cuda.

Covers issue #41871 — stale entries in ~/.triton/cache silently producing
wrong code when the running GPU's compute capability is not in
torch.cuda.get_arch_list() (e.g. sm_121 device falling back to sm_120 PTX).
"""

from __future__ import annotations

import logging
import shutil

import pytest
import torch

from vllm.platforms import cuda as cuda_platform


@pytest.fixture(autouse=True)
def _reset_guard_cache():
    """Reset the @cache on _maybe_warn_arch_ptx_fallback between tests so each
    test exercises the guard from a clean state."""
    cuda_platform._maybe_warn_arch_ptx_fallback.cache_clear()
    yield
    cuda_platform._maybe_warn_arch_ptx_fallback.cache_clear()


@pytest.fixture(autouse=True)
def _local_rank_zero(monkeypatch: pytest.MonkeyPatch):
    """Default tests to LOCAL_RANK == 0 so the guard's body runs. Individual
    tests that exercise the non-master short-circuit override this."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "LOCAL_RANK", 0)


class _ListHandler(logging.Handler):
    """Capture records from a non-propagating vLLM child logger.

    vLLM's `vllm` logger sets propagate=False, so the stock pytest `caplog`
    fixture (which hooks the root logger) misses records emitted by
    `vllm.platforms.cuda`. We attach our own handler to that specific
    logger for the duration of the test.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def cuda_log() -> _ListHandler:
    handler = _ListHandler()
    logger = logging.getLogger("vllm.platforms.cuda")
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def test_no_warning_when_arch_is_in_arch_list(
    monkeypatch: pytest.MonkeyPatch,
    cuda_log: _ListHandler,
):
    """sm_90 device on a torch built with sm_90 → no warning, no wipe."""
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])
    monkeypatch.delenv("VLLM_FORCE_TRITON_CACHE_INVALIDATE", raising=False)

    cuda_platform._maybe_warn_arch_ptx_fallback(9, 0)

    assert not any("PTX fallback" in r.getMessage() for r in cuda_log.records)


def test_warning_emitted_on_ptx_fallback(
    monkeypatch: pytest.MonkeyPatch,
    cuda_log: _ListHandler,
):
    """sm_121 device on a torch built with sm_120 only → warning emitted,
    no wipe unless the env var is set."""
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_120"])
    monkeypatch.delenv("VLLM_FORCE_TRITON_CACHE_INVALIDATE", raising=False)

    cuda_platform._maybe_warn_arch_ptx_fallback(12, 1)

    warnings = [
        r.getMessage() for r in cuda_log.records if r.levelno == logging.WARNING
    ]
    assert any("PTX fallback" in m for m in warnings), warnings
    assert any("41871" in m for m in warnings), warnings


def test_warning_is_emitted_only_once(
    monkeypatch: pytest.MonkeyPatch,
    cuda_log: _ListHandler,
):
    """Per-process cap-key cache means repeated calls for the same arch only
    warn once, so workers don't spam the log."""
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_120"])
    monkeypatch.delenv("VLLM_FORCE_TRITON_CACHE_INVALIDATE", raising=False)

    cuda_platform._maybe_warn_arch_ptx_fallback(12, 1)
    cuda_platform._maybe_warn_arch_ptx_fallback(12, 1)
    cuda_platform._maybe_warn_arch_ptx_fallback(12, 1)

    fallback_warnings = [
        r for r in cuda_log.records if "PTX fallback" in r.getMessage()
    ]
    assert len(fallback_warnings) == 1, [r.getMessage() for r in cuda_log.records]


def test_env_set_wipes_existing_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    cuda_log: _ListHandler,
):
    """With VLLM_FORCE_TRITON_CACHE_INVALIDATE=1, a present cache directory
    is removed."""
    cache_dir = tmp_path / "triton_cache"
    cache_dir.mkdir()
    (cache_dir / "stale.cubin").write_bytes(b"\x00\x01\x02")
    assert (cache_dir / "stale.cubin").exists()

    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_120"])
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("VLLM_FORCE_TRITON_CACHE_INVALIDATE", "1")
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_FORCE_TRITON_CACHE_INVALIDATE", True)

    cuda_platform._maybe_warn_arch_ptx_fallback(12, 1)

    assert not cache_dir.exists(), "Cache dir should have been wiped"
    assert any("wiped Triton cache" in r.getMessage() for r in cuda_log.records), [
        r.getMessage() for r in cuda_log.records
    ]


def test_env_set_but_no_cache_directory_logs_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    cuda_log: _ListHandler,
):
    """If the cache dir doesn't exist, do not raise — log INFO and continue."""
    cache_dir = tmp_path / "does_not_exist"
    assert not cache_dir.exists()

    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_120"])
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("VLLM_FORCE_TRITON_CACHE_INVALIDATE", "1")
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_FORCE_TRITON_CACHE_INVALIDATE", True)

    cuda_platform._maybe_warn_arch_ptx_fallback(12, 1)

    assert any("does not exist" in r.getMessage() for r in cuda_log.records), [
        r.getMessage() for r in cuda_log.records
    ]


def test_get_arch_list_raising_is_handled(
    monkeypatch: pytest.MonkeyPatch,
    cuda_log: _ListHandler,
):
    """If torch.cuda.get_arch_list() raises (uncommon backends), the guard
    must stay silent rather than tripping startup."""

    def _boom():
        raise RuntimeError("backend has no get_arch_list")

    monkeypatch.setattr(torch.cuda, "get_arch_list", _boom)
    monkeypatch.delenv("VLLM_FORCE_TRITON_CACHE_INVALIDATE", raising=False)

    # Should not raise
    cuda_platform._maybe_warn_arch_ptx_fallback(12, 1)

    assert not any("PTX fallback" in r.getMessage() for r in cuda_log.records)


def test_non_master_local_rank_is_silent_and_no_wipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    cuda_log: _ListHandler,
):
    """LOCAL_RANK != 0 short-circuits BEFORE the arch check, the warning,
    and the cache wipe — so worker processes don't race on rmtree or
    spam the log on multi-GPU nodes."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "LOCAL_RANK", 3)

    cache_dir = tmp_path / "triton_cache"
    cache_dir.mkdir()
    (cache_dir / "stale.cubin").write_bytes(b"\x00")

    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_120"])
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("VLLM_FORCE_TRITON_CACHE_INVALIDATE", "1")
    monkeypatch.setattr(envs, "VLLM_FORCE_TRITON_CACHE_INVALIDATE", True)

    cuda_platform._maybe_warn_arch_ptx_fallback(12, 1)

    # No warning, no wipe.
    assert not any("PTX fallback" in r.getMessage() for r in cuda_log.records)
    assert cache_dir.exists() and (cache_dir / "stale.cubin").exists(), (
        "Non-master worker must not touch the shared cache directory"
    )


def test_rmtree_filenotfound_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    cuda_log: _ListHandler,
):
    """If the cache dir disappears between the isdir check and the rmtree
    (e.g. an external cleaner or another local-master process raced us),
    we treat that as success — no spurious WARNING."""
    cache_dir = tmp_path / "triton_cache"
    cache_dir.mkdir()
    real_rmtree = shutil.rmtree

    def _vanishes_then_rmtree(path, *a, **kw):
        # Simulate the race: directory exists at isdir check, gone by rmtree.
        real_rmtree(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_120"])
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("VLLM_FORCE_TRITON_CACHE_INVALIDATE", "1")
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_FORCE_TRITON_CACHE_INVALIDATE", True)
    monkeypatch.setattr(cuda_platform.shutil, "rmtree", _vanishes_then_rmtree)

    cuda_platform._maybe_warn_arch_ptx_fallback(12, 1)

    # The PTX-fallback notice still fires, but no "failed to wipe" WARNING.
    failure_warnings = [
        r for r in cuda_log.records if "failed to wipe" in r.getMessage()
    ]
    assert failure_warnings == []
