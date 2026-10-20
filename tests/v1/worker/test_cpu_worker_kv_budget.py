# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU worker must start and size its KV cache correctly.

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


def test_short_node_memory_still_asks_to_free_memory(monkeypatch):
    worker = _worker(monkeypatch, reserved=6 * GiB, rss=1 * GiB, available=2 * GiB)
    with pytest.raises(ValueError, match="Reduce CPU memory used by other processes"):
        worker.determine_available_memory()


def test_budget_that_fits_sizes_the_kv_cache(monkeypatch):
    worker = _worker(monkeypatch, reserved=3 * GiB, rss=1 * GiB)
    assert worker.determine_available_memory() == 2 * GiB


def test_cpu_worker_defaults_to_v1_without_triton(monkeypatch):
    import vllm.config.vllm as config_module
    from vllm.config import VllmConfig
    from vllm.v1.worker import cpu_worker, gpu_worker

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(config_module, "HAS_TRITON", False)
    monkeypatch.setattr(gpu_worker, "HAS_TRITON", False)
    monkeypatch.setattr(cpu_worker, "get_visible_memory_node", lambda: [0])
    monkeypatch.setattr(
        cpu_worker, "get_allowed_cpu_list", lambda: [SimpleNamespace(numa_node=0)]
    )
    monkeypatch.setattr(
        cpu_worker,
        "get_memory_node_info",
        lambda node: MemoryNodeInfo(total_memory=8 * GiB, available_memory=8 * GiB),
    )
    monkeypatch.setattr(
        cpu_worker.torch.ops._C,
        "init_cpu_memory_env",
        lambda nodes: None,
        raising=False,
    )

    config = VllmConfig()
    worker = cpu_worker.CPUWorker(
        config, local_rank=0, rank=0, distributed_init_method="env://"
    )
    assert not config.use_v2_model_runner
    assert not worker.use_v2_model_runner
