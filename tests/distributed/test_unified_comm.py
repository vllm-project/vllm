# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for the unified_comm abstraction layer.

These tests exercise :class:`UnifiedCommAdapter` end-to-end on top of a
real ``torch.distributed`` process group:
  * ``test_unified_comm_disabled_returns_none`` runs in a single process
    and verifies that the adapter is a no-op when the feature flag is
    not set, so the default ``GroupCoordinator`` code path is unaffected.
  * ``test_unified_comm_all_reduce_2gpu`` spawns two worker processes,
    creates one adapter per rank, and checks numerical equivalence
    between ``adapter.all_reduce`` and ``torch.distributed.all_reduce``.
  * ``test_unified_comm_all_gather_2gpu`` and
    ``test_unified_comm_broadcast_2gpu`` cover the other primitives
    that the adapter routes for ``GroupCoordinator``.

The tests follow the same pattern as ``tests/distributed/test_pynccl.py``
and skip cleanly when fewer than two accelerators are available.
"""

import os

import multiprocess as mp
import pytest
import torch
import torch.distributed

from vllm.distributed.parallel_state import init_distributed_environment
from vllm.distributed.unified_comm import (
    UnifiedCommAdapter,
    is_unified_comm_enabled,
)
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


# ---------------------------------------------------------------------------
# Single-process tests: feature flag plumbing.
# ---------------------------------------------------------------------------


def test_unified_comm_flag_default_off(monkeypatch):
    monkeypatch.delenv("UNIFIED_COMM_ENABLED", raising=False)
    assert is_unified_comm_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes"])
def test_unified_comm_flag_truthy_values(monkeypatch, val):
    monkeypatch.setenv("UNIFIED_COMM_ENABLED", val)
    assert is_unified_comm_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", ""])
def test_unified_comm_flag_falsy_values(monkeypatch, val):
    monkeypatch.setenv("UNIFIED_COMM_ENABLED", val)
    assert is_unified_comm_enabled() is False


def test_unified_comm_disabled_returns_none(monkeypatch):
    """When the env var is not set, ``try_create`` must return ``None``
    so the default code path is selected.
    """
    monkeypatch.delenv("UNIFIED_COMM_ENABLED", raising=False)
    adapter = UnifiedCommAdapter.try_create(
        ranks=[0, 1],
        local_rank=0,
        device=torch.device("cpu"),
    )
    assert adapter is None


# ---------------------------------------------------------------------------
# Multi-process helpers (mirrors tests/distributed/test_pynccl.py).
# ---------------------------------------------------------------------------


def _distributed_run(fn, world_size: int):
    procs: list[mp.Process] = []
    for i in range(world_size):
        env: dict[str, str] = {
            "RANK": str(i),
            "LOCAL_RANK": str(i),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12347",
            # Always exercise the unified path in worker processes.
            "UNIFIED_COMM_ENABLED": "1",
        }
        p = mp.Process(target=fn, args=(env,))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    for p in procs:
        assert p.exitcode == 0, f"worker {p.pid} exited with {p.exitcode}"


def _worker_wrapper(fn):
    def wrapped(env):
        update_environment_variables(env)
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        init_distributed_environment()
        fn(device)

    return wrapped


def _build_adapter(device: torch.device) -> UnifiedCommAdapter:
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    adapter = UnifiedCommAdapter.try_create(
        ranks=list(range(world_size)),
        local_rank=rank,
        device=device,
    )
    assert adapter is not None, (
        "UnifiedCommAdapter.try_create returned None even though "
        "UNIFIED_COMM_ENABLED=1; check backend registration."
    )
    return adapter


# ---------------------------------------------------------------------------
# Multi-process collectives: numerical parity with torch.distributed.
# ---------------------------------------------------------------------------


@_worker_wrapper
def _all_reduce_worker(device: torch.device):
    adapter = _build_adapter(device)
    world_size = torch.distributed.get_world_size()
    # Sum-of-ones over `world_size` ranks must equal `world_size` per element.
    x = torch.ones(1024, dtype=torch.float32, device=device)
    out = adapter.all_reduce(x)
    assert out is not None
    torch.accelerator.synchronize()
    assert torch.all(out == world_size).cpu().item()
    adapter.destroy()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 accelerators to run the test.",
)
def test_unified_comm_all_reduce_2gpu():
    _distributed_run(_all_reduce_worker, 2)


@_worker_wrapper
def _all_gather_worker(device: torch.device):
    adapter = _build_adapter(device)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    # rank-th slot filled with rank value, gathered shape == world_size * 8.
    x = torch.full((8,), float(rank), dtype=torch.float32, device=device)
    out = adapter.all_gather(x, dim=0)
    assert out is not None
    torch.accelerator.synchronize()
    assert out.shape[0] == world_size * 8
    expected = torch.cat([torch.full((8,), float(r)) for r in range(world_size)]).to(
        device
    )
    assert torch.allclose(out, expected)
    adapter.destroy()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 accelerators to run the test.",
)
def test_unified_comm_all_gather_2gpu():
    _distributed_run(_all_gather_worker, 2)


@_worker_wrapper
def _broadcast_worker(device: torch.device):
    adapter = _build_adapter(device)
    rank = torch.distributed.get_rank()
    # rank 0 broadcasts the value 7; every rank should receive it.
    payload = 7.0 if rank == 0 else 0.0
    x = torch.full((4,), float(payload), dtype=torch.float32, device=device)
    out = adapter.broadcast(x, src=0)
    assert out is not None
    torch.accelerator.synchronize()
    assert torch.all(out == 7.0).cpu().item()
    adapter.destroy()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 accelerators to run the test.",
)
def test_unified_comm_broadcast_2gpu():
    _distributed_run(_broadcast_worker, 2)
