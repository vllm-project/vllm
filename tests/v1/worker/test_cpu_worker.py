# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.utils.cpu_resource_utils import LogicalCPUInfo, MemoryNodeInfo
from vllm.v1.worker.cpu_worker import CPUWorker
from vllm.v1.worker.gpu_worker import Worker


def test_worker_reuses_visible_fallback_node_for_kv_sizing(monkeypatch):
    """KV sizing must reuse the visible node selected during initialization."""
    cache_config = SimpleNamespace(
        gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=128,
    )
    parallel_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        parallel_config=parallel_config,
    )

    monkeypatch.setattr(
        "vllm.v1.worker.cpu_worker.get_allowed_cpu_list",
        lambda: [LogicalCPUInfo(id=0, physical_core=0, numa_node=7)],
    )
    monkeypatch.setattr(
        "vllm.v1.worker.cpu_worker.get_visible_memory_node",
        lambda: [0],
    )

    queried_nodes = []

    def get_memory_node_info(node_id):
        queried_nodes.append(node_id)
        return MemoryNodeInfo(total_memory=1024, available_memory=512)

    monkeypatch.setattr(
        "vllm.v1.worker.cpu_worker.get_memory_node_info",
        get_memory_node_info,
    )

    initialized_nodes = []
    monkeypatch.setattr(
        torch.ops._C,
        "init_cpu_memory_env",
        lambda nodes: initialized_nodes.append(nodes),
    )

    def init_worker(self, vllm_config, *args, **kwargs):
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.model_runner = SimpleNamespace()

    monkeypatch.setattr(Worker, "__init__", init_worker)
    monkeypatch.setattr(CPUWorker, "_should_warm_up_model", lambda self: False)

    worker = CPUWorker(vllm_config, 0, 0, "")

    assert worker.determine_available_memory() == 128
    assert initialized_nodes == [[0]]
    assert queried_nodes == [0, 0]
