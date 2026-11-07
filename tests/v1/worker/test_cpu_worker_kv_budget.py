# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU worker's automatic KV cache sizing must report the right cause.

A worker whose own footprint exceeds its reservation and a node that is short
of free memory need opposite remedies, so they get separate errors.
"""

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.utils.cpu_resource_utils import MemoryNodeInfo

# Importing the CPU worker patches torch and the GPU buffer utilities for the
# whole process (vllm/v1/worker/cpu/shm.py), so the import is deferred into
# the tests and they run only on the CPU platform.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu(), reason="CPU worker test; needs the CPU platform"
)

GiB = 1 << 30


def _worker(monkeypatch, *, reserved, rss, total=8 * GiB, available=4 * GiB):
    import vllm.v1.worker.cpu_worker as cpu_worker_mod
    from vllm.v1.worker.cpu_worker import CPUWorker

    worker = CPUWorker.__new__(CPUWorker)
    worker.requested_cpu_memory = reserved
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
    monkeypatch.setattr(CPUWorker, "_should_warm_up_model", lambda self: False)
    monkeypatch.setattr(
        cpu_worker_mod,
        "get_allowed_cpu_list",
        lambda: [SimpleNamespace(numa_node=0)],
    )
    monkeypatch.setattr(
        cpu_worker_mod,
        "get_memory_node_info",
        lambda node: MemoryNodeInfo(total_memory=total, available_memory=available),
    )
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=rss))
    monkeypatch.setattr(cpu_worker_mod.psutil, "Process", lambda pid: process)
    return worker


def test_footprint_above_reservation_asks_for_a_larger_reservation(monkeypatch):
    # 0.76 GiB already in use against a 0.16 GiB reservation (2% of 8 GiB).
    worker = _worker(monkeypatch, reserved=int(0.16 * GiB), rss=int(0.76 * GiB))
    with pytest.raises(ValueError) as exc:
        worker.determine_available_memory()
    msg = str(exc.value)
    assert "increase `--gpu-memory-utilization` to more than 0.10" in msg
    assert "--kv-cache-memory-bytes" in msg
    # Neither the misleading remedy nor a negative KV size.
    assert "other processes" not in msg
    assert "-0." not in msg


def test_short_node_memory_suggests_a_reservation_that_fits(monkeypatch):
    # The default reservation on a busy machine: 1 GiB in use plus 2 GiB free,
    # less 1 GiB of headroom, backs 2/8 of the node.
    worker = _worker(monkeypatch, reserved=6 * GiB, rss=1 * GiB, available=2 * GiB)
    with pytest.raises(ValueError) as exc:
        worker.determine_available_memory()
    msg = str(exc.value)
    assert "decrease `--gpu-memory-utilization` to 0.25 or lower" in msg
    assert "--kv-cache-memory-bytes" in msg
    assert "reduce CPU memory used by other processes" in msg


def test_suggestion_leaves_headroom_below_the_boundary(monkeypatch):
    # Reported on a 7.76 GiB node: the exact boundary (0.63) passed this check
    # and the worker was then OOM-killed, while 0.50 ran.
    total = int(7.76 * GiB)
    available = int(3.79 * GiB)
    worker = _worker(
        monkeypatch,
        reserved=int(0.92 * total),
        rss=int(0.63 * total) - available,
        total=total,
        available=available,
    )
    with pytest.raises(ValueError) as exc:
        worker.determine_available_memory()
    assert "decrease `--gpu-memory-utilization` to 0.50 or lower" in str(exc.value)


@pytest.mark.parametrize(
    "available", [int(0.01 * GiB), int(0.9 * GiB)], ids=["0.01GiB", "0.9GiB"]
)
def test_nearly_full_node_only_asks_to_free_memory(monkeypatch, available):
    # Once the headroom is set aside, no smaller reservation leaves room for KV.
    worker = _worker(monkeypatch, reserved=6 * GiB, rss=1 * GiB, available=available)
    with pytest.raises(ValueError) as exc:
        worker.determine_available_memory()
    msg = str(exc.value)
    assert "decrease `--gpu-memory-utilization`" not in msg
    assert "To resolve: reduce CPU memory used by other processes" in msg


def test_budget_that_fits_sizes_the_kv_cache(monkeypatch):
    worker = _worker(monkeypatch, reserved=3 * GiB, rss=1 * GiB)
    assert worker.determine_available_memory() == 2 * GiB
