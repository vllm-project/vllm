# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU worker's automatic KV cache sizing must report the right cause.

A worker whose own footprint exceeds its reservation and a node that is short
of free memory need opposite remedies, so they get separate errors.
"""

import math
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
    worker.memory_node = 0
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
    monkeypatch.setattr(CPUWorker, "_should_warm_up_model", lambda self: False)
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


def test_short_node_memory_still_asks_to_free_memory(monkeypatch):
    worker = _worker(monkeypatch, reserved=6 * GiB, rss=1 * GiB, available=2 * GiB)
    with pytest.raises(ValueError, match="Reduce CPU memory used by other processes"):
        worker.determine_available_memory()


def test_budget_that_fits_sizes_the_kv_cache(monkeypatch):
    worker = _worker(monkeypatch, reserved=3 * GiB, rss=1 * GiB)
    assert worker.determine_available_memory() == 2 * GiB


def _construct(
    monkeypatch,
    *,
    rank,
    cpu_node,
    memory_nodes,
    local_world_size,
    node_info,
    cgroup_limit=None,
    kv_bytes=None,
):
    """Run CPUWorker.__init__ without a real engine."""
    import vllm.v1.worker.cpu_worker as cpu_worker_mod
    import vllm.v1.worker.gpu_worker as gpu_worker_mod
    from vllm.v1.worker.cpu_worker import CPUWorker

    def fake_base_init(self, vllm_config, *args, **kwargs):
        self.cache_config = vllm_config.cache_config
        self.parallel_config = SimpleNamespace()

    monkeypatch.setattr(gpu_worker_mod.Worker, "__init__", fake_base_init)
    # __init__ only uses torch for torch.ops._C.init_cpu_memory_env.
    fake_c = SimpleNamespace(init_cpu_memory_env=lambda nodes: None)
    monkeypatch.setattr(
        cpu_worker_mod, "torch", SimpleNamespace(ops=SimpleNamespace(_C=fake_c))
    )
    monkeypatch.setattr(
        cpu_worker_mod, "get_worker_memory_nodes", lambda ids=None: memory_nodes
    )
    monkeypatch.setattr(
        cpu_worker_mod,
        "get_allowed_cpu_list",
        lambda: [SimpleNamespace(id=rank, physical_core=rank, numa_node=cpu_node)],
    )
    monkeypatch.setattr(cpu_worker_mod, "get_memory_node_info", node_info)
    # raising=False: before this fix cpu_worker didn't import it.
    monkeypatch.setattr(
        cpu_worker_mod,
        "get_cgroup_memory_limit",
        lambda: cgroup_limit,
        raising=False,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            assigned_physical_gpu_ids=None, local_world_size=local_world_size
        ),
        cache_config=SimpleNamespace(
            resolved_gpu_memory_utilization=0.9, kv_cache_memory_bytes=kv_bytes
        ),
    )
    return CPUWorker(
        config, local_rank=rank, rank=rank, distributed_init_method="tcp://127.0.0.1:1"
    )


def test_workers_share_the_cgroup_limit(monkeypatch):
    """TP=2 on two 96 GiB NUMA nodes inside a 64 GiB container: together the
    workers must not reserve more than the container may use."""
    cgroup_limit = 64 * GiB

    def node_info(node):
        # get_memory_node_info() clamps each node's total to the cgroup limit.
        return MemoryNodeInfo(total_memory=cgroup_limit, available_memory=cgroup_limit)

    reserved = [
        _construct(
            monkeypatch,
            rank=rank,
            cpu_node=rank,
            memory_nodes=[0, 1],
            local_world_size=2,
            node_info=node_info,
            cgroup_limit=cgroup_limit,
        ).requested_cpu_memory
        for rank in (0, 1)
    ]
    assert sum(reserved) <= cgroup_limit
    assert reserved[0] == reserved[1] == math.ceil(32 * GiB * 0.9)


def test_single_worker_reservation_is_unchanged(monkeypatch):
    worker = _construct(
        monkeypatch,
        rank=0,
        cpu_node=0,
        memory_nodes=[0],
        local_world_size=1,
        node_info=lambda node: MemoryNodeInfo(
            total_memory=24 * GiB, available_memory=20 * GiB
        ),
        cgroup_limit=24 * GiB,
    )
    assert worker.requested_cpu_memory == math.ceil(24 * GiB * 0.9)


def test_kv_budget_uses_the_fallback_memory_node(monkeypatch):
    """The CPU reports group id 7 (e.g. an s390x book id) but only node 0 has
    memory: __init__ falls back to node 0, and the KV budget must too."""
    queried = []

    def node_info(node):
        queried.append(node)
        return MemoryNodeInfo(total_memory=64 * GiB, available_memory=60 * GiB)

    worker = _construct(
        monkeypatch,
        rank=0,
        cpu_node=7,
        memory_nodes=[0],
        local_world_size=1,
        node_info=node_info,
        kv_bytes=8 * GiB,
    )
    monkeypatch.setattr(type(worker), "_should_warm_up_model", lambda self: False)

    assert worker.determine_available_memory() == 8 * GiB
    assert queried == [0, 0]
