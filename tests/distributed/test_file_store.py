# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import torch

import vllm.config
import vllm.distributed.parallel_state as parallel_state
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
)


def _file_store_worker(rank, init_method, result_queue):
    try:
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            backend="gloo",
            distributed_init_method=init_method,
        )
        result = torch.tensor(rank)
        torch.distributed.all_reduce(result)
        result_queue.put(result.item())
    finally:
        destroy_distributed_environment()


def test_file_store_preserves_raw_path(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0")
    store_path = tmp_path / "store#?"

    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method=f"file://{store_path}",
        )
        assert store_path.exists()
    finally:
        destroy_distributed_environment()


def test_file_store_rendezvouses_local_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0")
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.SimpleQueue()

    torch.multiprocessing.spawn(
        _file_store_worker,
        args=(f"file://{tmp_path / 'store#?'}", result_queue),
        nprocs=2,
        join=True,
    )

    assert sorted(result_queue.get() for _ in range(2)) == [1, 1]


def test_multinode_overrides_file_store_rendezvous(monkeypatch):
    monkeypatch.setenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0")
    parallel_config = SimpleNamespace(
        distributed_executor_backend="mp",
        nnodes=2,
        data_parallel_size=1,
        enable_elastic_ep=False,
        data_parallel_rank=0,
        world_size_across_dp=2,
        master_addr="192.0.2.1",
        master_port=29500,
        nnodes_within_dp=2,
    )
    config = SimpleNamespace(parallel_config=parallel_config)
    calls = []
    original_world = parallel_state._WORLD
    original_node_count = parallel_state._NODE_COUNT

    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_backend_available", lambda _: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed, "init_process_group", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        parallel_state,
        "init_world_group",
        lambda *_: SimpleNamespace(world_size=2, cpu_group=None),
    )
    monkeypatch.setattr(parallel_state, "_node_count", lambda _: 1)

    try:
        init_distributed_environment(
            world_size=2,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method="file:///tmp/vllm_dist_test",
        )
    finally:
        parallel_state._WORLD = original_world
        parallel_state._NODE_COUNT = original_node_count

    assert calls == [
        {
            "backend": "gloo",
            "init_method": "tcp://192.0.2.1:29500",
            "store": None,
            "world_size": 2,
            "rank": 0,
            "timeout": None,
        }
    ]


def test_gloo_socket_ifname_defaults_to_lo_for_single_node_mp(monkeypatch):
    monkeypatch.setenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0")
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    parallel_config = SimpleNamespace(
        distributed_executor_backend="mp",
        nnodes=1,
        data_parallel_size=1,
        enable_elastic_ep=False,
        data_parallel_rank=0,
        world_size_across_dp=2,
        nnodes_within_dp=1,
        pipeline_parallel_size=2,
    )
    config = SimpleNamespace(parallel_config=parallel_config)
    original_world = parallel_state._WORLD
    original_node_count = parallel_state._NODE_COUNT

    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_backend_available", lambda _: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: None)
    monkeypatch.setattr(
        parallel_state,
        "init_world_group",
        lambda *_: SimpleNamespace(world_size=2, cpu_group=None),
    )
    monkeypatch.setattr(parallel_state, "_node_count", lambda _: 1)

    try:
        init_distributed_environment(
            world_size=2,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method="tcp://127.0.0.1:29500",
        )
        assert os.environ.get("GLOO_SOCKET_IFNAME") == "lo"
    finally:
        parallel_state._WORLD = original_world
        parallel_state._NODE_COUNT = original_node_count


def test_gloo_socket_ifname_preserved_if_already_set(monkeypatch):
    monkeypatch.setenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth0")
    parallel_config = SimpleNamespace(
        distributed_executor_backend="mp",
        nnodes=1,
        data_parallel_size=1,
        enable_elastic_ep=False,
        data_parallel_rank=0,
        world_size_across_dp=2,
        nnodes_within_dp=1,
        pipeline_parallel_size=2,
    )
    config = SimpleNamespace(parallel_config=parallel_config)
    original_world = parallel_state._WORLD
    original_node_count = parallel_state._NODE_COUNT

    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_backend_available", lambda _: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: None)
    monkeypatch.setattr(
        parallel_state,
        "init_world_group",
        lambda *_: SimpleNamespace(world_size=2, cpu_group=None),
    )
    monkeypatch.setattr(parallel_state, "_node_count", lambda _: 1)

    try:
        init_distributed_environment(
            world_size=2,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method="tcp://127.0.0.1:29500",
        )
        assert os.environ.get("GLOO_SOCKET_IFNAME") == "eth0"
    finally:
        parallel_state._WORLD = original_world
        parallel_state._NODE_COUNT = original_node_count


def test_gloo_socket_ifname_not_defaulted_for_multi_node(monkeypatch):
    monkeypatch.setenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0")
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    parallel_config = SimpleNamespace(
        distributed_executor_backend="mp",
        nnodes=2,
        data_parallel_size=1,
        enable_elastic_ep=False,
        data_parallel_rank=0,
        world_size_across_dp=2,
        master_addr="192.0.2.1",
        master_port=29500,
        nnodes_within_dp=2,
        pipeline_parallel_size=2,
    )
    config = SimpleNamespace(parallel_config=parallel_config)
    original_world = parallel_state._WORLD
    original_node_count = parallel_state._NODE_COUNT

    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_backend_available", lambda _: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: None)
    monkeypatch.setattr(
        parallel_state,
        "init_world_group",
        lambda *_: SimpleNamespace(world_size=2, cpu_group=None),
    )
    monkeypatch.setattr(parallel_state, "_node_count", lambda _: 1)

    try:
        init_distributed_environment(
            world_size=2,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method="tcp://192.0.2.1:29500",
        )
        assert "GLOO_SOCKET_IFNAME" not in os.environ
    finally:
        parallel_state._WORLD = original_world
        parallel_state._NODE_COUNT = original_node_count
