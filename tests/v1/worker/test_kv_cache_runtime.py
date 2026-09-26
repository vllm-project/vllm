# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed data-lifetime tests for layer-sharded cache materialization."""

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.utils.network_utils import get_open_port
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.kv_cache_placement import (
    KVCacheBundle,
    KVCachePlacement,
    build_kv_cache_storage,
)
from vllm.v1.worker.kv_cache_runtime import KVCacheRuntime
from vllm.v1.worker.utils import allocate_kv_cache


def _runtime_worker(rank: int, port: int, world_size: int):
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
    try:
        # Unequal component sizes and >2 nonowner layers exercise both scratch
        # slots repeatedly. The auxiliary component is acquired before main KV.
        num_layers = 2 * world_size + 3
        owners = [
            owner
            for owner in range(world_size)
            for _ in range(num_layers // world_size + (owner < num_layers % world_size))
        ]
        names = [
            f"layer{i}.{part}" for i in range(num_layers) for part in ("mla", "index")
        ]
        names.append("draft")
        specs = {
            name: MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=128 if name.endswith("mla") else 32,
                dtype=torch.bfloat16,
            )
            for name in names
        }
        config = KVCacheConfig(
            7,
            [],
            [KVCacheGroupSpec(names, UniformTypeKVCacheSpecs(16, specs))],
        )
        placement = KVCachePlacement(
            rank,
            world_size,
            tuple(
                KVCacheBundle((f"layer{i}.mla", f"layer{i}.index"), owners[i])
                for i in range(num_layers)
            )
            + (KVCacheBundle(("draft",), None),),
            alignment=2 * 1024 * 1024,
        )
        config = build_kv_cache_storage(config, placement, KVCacheLayout.LBNHC)
        caches = allocate_kv_cache(
            config, torch.device("cuda", rank), KVCacheLayout.LBNHC
        )
        runtime = KVCacheRuntime(config, caches)
        caches["draft"].fill_(111 + rank)
        observations = []
        for step in range(5):
            with set_forward_context(None, vllm_config), runtime.forward(step > 0):
                for index, bundle in enumerate(placement.bundles[:-1]):
                    # Delay compute to expose premature scratch reuse.
                    runtime.acquire(bundle.layers[1])
                    torch.cuda._sleep(100_000)
                    runtime.acquire(bundle.layers[0])
                    for component, name in enumerate(bundle.layers):
                        if step:
                            observations.append(
                                (
                                    caches[name].clone(),
                                    10 * index + component + step - 1,
                                )
                            )
                        caches[name].fill_(10 * index + component + step)
                    runtime.release(bundle.layers[0])
            # Do not synchronize between steps: next-step broadcasts must also
            # wait for the previous owner's writes and receiver scratch use.
        torch.cuda.synchronize()
        for observed, expected in observations:
            assert torch.all(observed == expected), (rank, expected)
        assert torch.all(caches["draft"] == 111 + rank)
        for i, owner in enumerate(owners):
            if owner == rank:
                assert torch.all(caches[f"layer{i}.mla"] == 10 * i + 4)
    finally:
        cleanup_dist_env_and_memory()


def test_materialization_preserves_owner_and_scratch_lifetimes():
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    mp.spawn(
        _runtime_worker,
        args=(get_open_port(), world_size),
        nprocs=world_size,
        join=True,
    )
