# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: custom AR graph capture under expandable_segments.

Covers https://github.com/vllm-project/vllm/issues/42609 — pre-capture VMM
activations are not legacy-IPC exportable; capture must not crash and must
still produce a correct all-reduce.

Also unit-tests nesting-safe expandable_segments disable (no GPU required).
"""

from __future__ import annotations

import os
import queue
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators import custom_all_reduce as car
from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    _disable_expandable_segments_for_cuda_ipc,
    _should_use_registered_buffer_for_capture,
    _tensor_is_legacy_ipc_capable,
)
from vllm.platforms import current_platform


# ---------------------------------------------------------------------------
# Unit tests (CPU / no multi-GPU required)
# ---------------------------------------------------------------------------


def test_should_use_registered_buffer_rocm_preserves_zero_copy(monkeypatch):
    """Non-CUDA platforms must keep historical registered=True capture path."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    # Fake tensor; should never call the CUDA probe.
    t = torch.zeros(8)
    with patch.object(car, "_tensor_is_legacy_ipc_capable") as probe:
        assert _should_use_registered_buffer_for_capture(t) is True
        probe.assert_not_called()


def test_should_use_registered_buffer_cuda_uses_probe(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    t = torch.zeros(8)
    with patch.object(car, "_tensor_is_legacy_ipc_capable", return_value=False) as probe:
        assert _should_use_registered_buffer_for_capture(t) is False
        probe.assert_called_once()
    with patch.object(car, "_tensor_is_legacy_ipc_capable", return_value=True) as probe:
        assert _should_use_registered_buffer_for_capture(t) is True
        probe.assert_called_once()


def test_expandable_segments_disable_is_nested_safe(monkeypatch):
    """Nested disable must only set False once and restore True once."""
    calls: list[bool] = []

    def fake_set(enabled: bool) -> None:
        calls.append(enabled)

    monkeypatch.setattr(car, "_set_expandable_segments", fake_set)
    monkeypatch.setattr(car, "_env_expandable_segments_enabled", lambda: True)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    # Reset depth in case a previous test left it non-zero.
    car._expandable_segments_disable_depth = 0

    with _disable_expandable_segments_for_cuda_ipc(True):
        with _disable_expandable_segments_for_cuda_ipc(True):
            assert car._expandable_segments_disable_depth == 2
        assert car._expandable_segments_disable_depth == 1
    assert car._expandable_segments_disable_depth == 0
    assert calls == [False, True]


def test_expandable_segments_disable_noop_when_inactive(monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(
        car, "_set_expandable_segments", lambda enabled: calls.append(enabled)
    )
    monkeypatch.setattr(car, "_env_expandable_segments_enabled", lambda: True)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    car._expandable_segments_disable_depth = 0

    with _disable_expandable_segments_for_cuda_ipc(False):
        pass
    assert calls == []
    assert car._expandable_segments_disable_depth == 0


# ---------------------------------------------------------------------------
# Multi-GPU integration (pytest + mp.spawn)
# ---------------------------------------------------------------------------


def _expandable_segments_graph_worker(
    local_rank: int, world_size: int, result_q: mp.Queue
) -> None:
    # Must run before first CUDA alloc in this process.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("VLLM_SKIP_P2P_CHECK", "1")
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29527")

    try:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        n = 4096
        # Pre-capture allocation under expandable_segments (the residual hole).
        x = torch.full((n,), float(rank + 1), device=device, dtype=torch.float32)
        legacy = _tensor_is_legacy_ipc_capable(x)
        # On expandable_segments we expect VMM → not legacy-IPC-capable.
        # If the platform still reports capable, the fix is a no-op path but
        # correctness must still hold — fail the test so CI notices.
        if current_platform.is_cuda() and legacy:
            result_q.put(
                (
                    rank,
                    "SKIP_UNEXPECTED_LEGACY_IPC",
                    "pre-capture tensor reported legacy_ipc=True under "
                    "expandable_segments:True; cannot prove fallback path",
                )
            )
            dist.destroy_process_group()
            return

        assert not _should_use_registered_buffer_for_capture(x), (
            "CUDA capture should choose registered=False for non-legacy-IPC "
            f"tensor (rank={rank})"
        )

        ca = CustomAllreduce(group=dist.group.WORLD, device=device, max_size=1 << 20)
        if ca.disabled:
            result_q.put((rank, "SKIP_CUSTOM_AR_DISABLED", None))
            dist.destroy_process_group()
            return

        expected = float(sum(range(1, world_size + 1)))
        out_e = ca.custom_all_reduce(x)
        assert out_e is not None
        torch.cuda.synchronize()
        assert (out_e - expected).abs().max().item() < 1e-5

        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        static_in = x.clone()
        # Clone under expandable_segments should also be non-legacy-IPC.
        assert not _tensor_is_legacy_ipc_capable(static_in)
        static_out = torch.empty_like(static_in)
        with torch.cuda.stream(s):
            with ca.capture():
                g.capture_begin()
                y = ca.custom_all_reduce(static_in)
                assert y is not None
                static_out.copy_(y)
                g.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        static_in.fill_(float(rank + 1))
        g.replay()
        torch.cuda.synchronize()
        err_g = (static_out - expected).abs().max().item()
        assert err_g < 1e-5, f"graph replay err={err_g}"

        ca.close()
        dist.barrier()
        dist.destroy_process_group()
        result_q.put((rank, "OK", err_g))
    except Exception as e:
        result_q.put((local_rank, "FAIL", repr(e)))


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="expandable_segments legacy-IPC regression is CUDA-specific",
)
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Needs 2 GPUs with P2P for custom AR graph IPC",
)
def test_custom_all_reduce_graph_pre_capture_vmm_expandable_segments():
    """TP=2: pre-capture VMM input + expandable_segments must not crash."""
    world_size = 2
    ctx = mp.get_context("spawn")
    result_q: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_expandable_segments_graph_worker,
            args=(rank, world_size, result_q),
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    results = []
    try:
        for _ in range(world_size):
            results.append(result_q.get(timeout=180))
    except queue.Empty as e:
        for p in procs:
            p.kill()
        raise TimeoutError("workers did not report results") from e
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
                p.join(5)

    statuses = {r[1] for r in results}
    if "FAIL" in statuses:
        pytest.fail(f"worker failure: {results}")
    if statuses == {"SKIP_CUSTOM_AR_DISABLED"}:
        pytest.skip("Custom all-reduce disabled on this platform/config")
    if "SKIP_UNEXPECTED_LEGACY_IPC" in statuses:
        pytest.skip(
            "Allocator did not produce non-legacy-IPC tensors under "
            "expandable_segments:True; cannot prove fallback path"
        )
    assert statuses == {"OK"}, results
    assert len(results) == world_size
