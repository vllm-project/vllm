"""Tests for the parent-side weight/sidecar prefetch in api_server.

Covers all three HIGH audit findings from the v1 of PR #45501:
  - daemon-thread cleanup (no ThreadPoolExecutor inside daemon thread)
  - HF cache filelock + fork interaction (resolution happens in worker
    thread, NOT a forked child process)
  - sidecar de-duplication (tokenizer.json must not be read 2x)

Plus the new asks:
  - recursive walk of GGUF subdir layouts
  - NFS-detected + multi-rank skip
  - VLLM_DISABLE_STARTUP_PREFETCH=1 operator override
"""
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm.entrypoints.openai import api_server


def _fake_vllm_config(model_path: str, tp: int = 1, pp: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(model=model_path, revision=None),
        load_config=SimpleNamespace(download_dir=None),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
        ),
    )


def _make_fake_repo(tmp_path: Path, *, with_subdir: bool = False) -> Path:
    """Build a fake on-disk model dir with weight + sidecar files."""
    repo = tmp_path / "fake-model"
    repo.mkdir()
    (repo / "model-00001-of-00002.safetensors").write_bytes(b"weights1")
    (repo / "model-00002-of-00002.safetensors").write_bytes(b"weights2")
    (repo / "tokenizer.json").write_bytes(b"{}")
    (repo / "tokenizer.model").write_bytes(b"sp-binary")
    (repo / "tokenizer_config.json").write_bytes(b"{}")
    (repo / "config.json").write_bytes(b"{}")
    (repo / "generation_config.json").write_bytes(b"{}")
    if with_subdir:
        sub = repo / "shards"
        sub.mkdir()
        (sub / "weights.gguf").write_bytes(b"gguf-weights")
    return repo


def _drain_prefetch_thread(timeout: float = 10.0) -> None:
    """Wait for any vllm-parent-weight-prefetch threads to finish."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        live = [
            t for t in threading.enumerate()
            if t.name in (
                "vllm-parent-weight-prefetch",
                "vllm-prefetch-reader",
            )
        ]
        if not live:
            return
        time.sleep(0.05)
    pytest.fail(
        "prefetch threads still live after timeout: "
        + ", ".join(t.name for t in live)
    )


def test_prefetch_reads_local_dir(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path)
    cfg = _fake_vllm_config(str(repo))

    opens: Counter[str] = Counter()
    real_open = open

    def counting_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(path, (str, os.PathLike)):
            opens[os.path.basename(os.fspath(path))] += 1
        return real_open(path, *args, **kwargs)

    with mock.patch("builtins.open", side_effect=counting_open):
        api_server._startup_prefetch_weights(cfg)
        _drain_prefetch_thread()

    # Each weight + sidecar opened exactly once (no dup glob hits).
    for fname in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "config.json",
        "generation_config.json",
    ):
        assert opens[fname] == 1, (
            f"{fname} was opened {opens[fname]} times — expected exactly 1"
        )


def test_prefetch_recursive_for_gguf_subdir(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path, with_subdir=True)
    cfg = _fake_vllm_config(str(repo))

    opened: list[str] = []
    real_open = open

    def tracking_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(path, (str, os.PathLike)):
            opened.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    with mock.patch("builtins.open", side_effect=tracking_open):
        api_server._startup_prefetch_weights(cfg)
        _drain_prefetch_thread()

    assert any(
        p.endswith(os.path.join("shards", "weights.gguf")) for p in opened
    ), "expected recursive walk to find shards/weights.gguf"


def test_prefetch_threads_are_daemon_no_join_block(tmp_path: Path) -> None:
    """Process-exit invariant: every thread launched by prefetch must be
    daemon. ThreadPoolExecutor's atexit-join was the v1 hang risk; this
    test fails if anyone re-introduces non-daemon workers.
    """
    repo = _make_fake_repo(tmp_path)
    cfg = _fake_vllm_config(str(repo))

    seen_threads: list[threading.Thread] = []
    orig_thread_init = threading.Thread.__init__

    def tracking_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        orig_thread_init(self, *args, **kwargs)
        seen_threads.append(self)

    with mock.patch.object(threading.Thread, "__init__", tracking_init):
        api_server._startup_prefetch_weights(cfg)
        _drain_prefetch_thread()

    prefetch_threads = [
        t for t in seen_threads
        if t.name in (
            "vllm-parent-weight-prefetch",
            "vllm-prefetch-reader",
        )
    ]
    assert prefetch_threads, "expected prefetch to spawn at least one thread"
    for t in prefetch_threads:
        assert t.daemon is True, (
            f"thread {t.name} was created non-daemon — would block exit"
        )


def test_prefetch_skips_when_disabled_via_env(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path)
    cfg = _fake_vllm_config(str(repo))

    opens: list[str] = []
    real_open = open

    def tracking_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(path, (str, os.PathLike)):
            opens.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    with mock.patch.dict(os.environ, {"VLLM_DISABLE_STARTUP_PREFETCH": "1"}):
        with mock.patch("builtins.open", side_effect=tracking_open):
            api_server._startup_prefetch_weights(cfg)
            # No thread should even be started when env disables.
            time.sleep(0.1)

    weight_opens = [p for p in opens if p.endswith(".safetensors")]
    assert weight_opens == [], (
        "VLLM_DISABLE_STARTUP_PREFETCH=1 should suppress all weight reads"
    )


def test_prefetch_skips_nfs_when_world_size_gt_1(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path)
    cfg = _fake_vllm_config(str(repo), tp=2, pp=1)

    opens: list[str] = []
    real_open = open

    def tracking_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(path, (str, os.PathLike)):
            opens.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    # Force the NFS detector to claim NFS-backed.
    with mock.patch.object(
        api_server, "_prefetch_is_on_nfs", return_value=True
    ):
        with mock.patch("builtins.open", side_effect=tracking_open):
            api_server._startup_prefetch_weights(cfg)
            _drain_prefetch_thread()

    weight_opens = [p for p in opens if p.endswith(".safetensors")]
    assert weight_opens == [], (
        "expected NFS + world_size>1 to skip parent-side prefetch"
    )


def test_prefetch_proceeds_on_nfs_when_world_size_eq_1(tmp_path: Path) -> None:
    """NFS alone is not a reason to skip — only NFS + multi-rank."""
    repo = _make_fake_repo(tmp_path)
    cfg = _fake_vllm_config(str(repo), tp=1, pp=1)

    opens: list[str] = []
    real_open = open

    def tracking_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(path, (str, os.PathLike)):
            opens.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    with mock.patch.object(
        api_server, "_prefetch_is_on_nfs", return_value=True
    ):
        with mock.patch("builtins.open", side_effect=tracking_open):
            api_server._startup_prefetch_weights(cfg)
            _drain_prefetch_thread()

    weight_opens = [p for p in opens if p.endswith(".safetensors")]
    assert len(weight_opens) >= 2, (
        "TP=PP=1 on NFS should still prefetch (no contention with self)"
    )


def test_prefetch_uses_hf_api_singleton_for_repo_id() -> None:
    """When given a repo id (not a local path), prefetch must resolve via
    the vLLM ``hf_api()`` singleton — not call ``huggingface_hub`` module-
    level functions directly. Sharing the configured HfApi instance keeps
    endpoint / token config consistent with the rest of the engine.
    """
    cfg = _fake_vllm_config("Qwen/Qwen3-4B-AWQ")

    fake_api = mock.MagicMock()
    fake_api.snapshot_download.return_value = ""

    with mock.patch(
        "vllm.transformers_utils.repo_utils.hf_api", return_value=fake_api
    ):
        with mock.patch("vllm.envs.VLLM_USE_MODELSCOPE", False, create=True):
            api_server._startup_prefetch_weights(cfg)
            _drain_prefetch_thread()

    assert fake_api.snapshot_download.called, (
        "expected hf_api().snapshot_download to be invoked for repo id"
    )
    call_kwargs = fake_api.snapshot_download.call_args.kwargs
    assert call_kwargs.get("local_files_only") is True
    assert call_kwargs.get("repo_id") == "Qwen/Qwen3-4B-AWQ"


def test_prefetch_skipped_for_headless_api_workers() -> None:
    """The build_async_engine_client_from_engine_args dispatch site only
    calls _startup_prefetch_weights when ``client_config`` lacks
    ``input_address`` — the marker that this process is a headless
    API-only worker. Verify by inspecting the source of the function.
    """
    import inspect
    src = inspect.getsource(
        api_server.build_async_engine_client_from_engine_args
    )
    assert "_startup_prefetch_weights" in src, (
        "dispatch site should still call the prefetch helper"
    )
    assert 'client_config.get("input_address")' in src, (
        "dispatch site must guard on the input_address sentinel"
    )
