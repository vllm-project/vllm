# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed data-lifetime tests for layer-sharded cache materialization."""

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    destroy_model_parallel,
    get_kvpp_group,
    get_pcp_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.utils import warmup_process_group
from vllm.forward_context import acquire_kv_cache, release_kv_cache, set_forward_context
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
from vllm.v1.worker.kvpp_runtime import KVPPRuntime, maybe_prepare_kvpp
from vllm.v1.worker.utils import allocate_kv_cache


def _runtime_worker(rank: int, port: int, tp_size: int, pcp_size: int, pp_size: int):
    replica_size = pcp_size * tp_size
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=replica_size * pp_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    vllm_config = VllmConfig()
    vllm_config.cache_config.enable_kvpp = True
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            prefill_context_model_parallel_size=pcp_size,
            pipeline_model_parallel_size=pp_size,
        )
    try:
        group = get_kvpp_group()
        stage_start = rank // replica_size * replica_size
        assert group.ranks == list(range(stage_start, stage_start + replica_size))
        assert group.rank_in_group == (
            get_pcp_group().rank_in_group * tp_size + get_tp_group().rank_in_group
        )
        assert group.device_group is not get_pcp_group().device_group
        assert group.device_group is not get_tp_group().device_group
        warmup_process_group(group, ["broadcast"])
        local_rank = group.rank_in_group
        stage_value = 128 * (rank // replica_size)
        # Unequal component sizes and >2 nonowner layers exercise both scratch
        # slots repeatedly. The auxiliary component is acquired before main KV.
        num_layers = 2 * replica_size + 3
        owners = [
            owner
            for owner in range(replica_size)
            for _ in range(
                num_layers // replica_size + (owner < num_layers % replica_size)
            )
        ]
        names = [
            f"layer{i}.{part}" for i in range(num_layers) for part in ("mla", "index")
        ]
        names.append("draft")
        specs = {
            name: MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=31 if name.endswith("mla") else 17,
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
            local_rank,
            replica_size,
            tuple(
                KVCacheBundle((f"layer{i}.mla", f"layer{i}.index"), owners[i])
                for i in range(num_layers)
            )
            + (KVCacheBundle(("draft",), None),),
        )
        config = build_kv_cache_storage(config, placement, KVCacheLayout.LBNHC)
        caches = allocate_kv_cache(
            config, torch.device("cuda", rank), KVCacheLayout.LBNHC
        )
        kvpp_runtime = KVPPRuntime(config, caches)
        caches["draft"].fill_(111 + rank)
        observations = []
        for step in range(5):
            with set_forward_context(None, vllm_config, kvpp_runtime=kvpp_runtime):
                maybe_prepare_kvpp(kvpp_runtime, np.array([step], dtype=np.int32))
                for index, bundle in enumerate(placement.bundles[:-1]):
                    # Delay compute to expose premature scratch reuse.
                    acquire_kv_cache(bundle.layers[1])
                    if step == 1 and index == 0:
                        # Reject reuse without discarding the pending prefetch.
                        with pytest.raises(AssertionError, match="Previous KVPP"):
                            maybe_prepare_kvpp(
                                kvpp_runtime, np.array([step], dtype=np.int32)
                            )
                    torch.cuda._sleep(100_000)
                    acquire_kv_cache(bundle.layers[0])
                    for component, name in enumerate(bundle.layers):
                        if step:
                            observations.append(
                                (
                                    caches[name].clone(),
                                    stage_value + 10 * index + component + step - 1,
                                )
                            )
                        caches[name].fill_(stage_value + 10 * index + component + step)
                    release_kv_cache(bundle.layers[0])
            # Do not synchronize between steps: next-step broadcasts must also
            # wait for the previous owner's writes and receiver scratch use.
        torch.cuda.synchronize()
        for observed, expected in observations:
            assert torch.all(observed == expected), (rank, expected)
        assert torch.all(caches["draft"] == 111 + rank)
        for i, owner in enumerate(owners):
            if owner == local_rank:
                assert torch.all(caches[f"layer{i}.mla"] == stage_value + 10 * i + 4)

        del kvpp_runtime, group
        # Reinitializing model parallelism must not retain a stale KVPP group.
        for enabled in (False, True):
            destroy_model_parallel()
            with pytest.raises(AssertionError, match="KVPP group is not initialized"):
                get_kvpp_group()
            vllm_config.cache_config.enable_kvpp = enabled
            with set_current_vllm_config(vllm_config):
                initialize_model_parallel(
                    tensor_model_parallel_size=tp_size,
                    prefill_context_model_parallel_size=pcp_size,
                    pipeline_model_parallel_size=pp_size,
                )
            if enabled:
                warmup_process_group(get_kvpp_group(), ["broadcast", "all_reduce"])
                probe = torch.tensor([rank], device="cuda")
                get_kvpp_group().broadcast(probe)
                assert probe.item() == stage_start
            else:
                with pytest.raises(
                    AssertionError, match="KVPP group is not initialized"
                ):
                    get_kvpp_group()
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "tp_size,pcp_size,pp_size", [(2, 1, 1), (2, 1, 2), (1, 2, 2), (2, 2, 1)]
)
def test_materialization_preserves_owner_and_scratch_lifetimes(
    tp_size, pcp_size, pp_size
):
    world_size = tp_size * pcp_size * pp_size
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    mp.spawn(
        _runtime_worker,
        args=(get_open_port(), tp_size, pcp_size, pp_size),
        nprocs=world_size,
        join=True,
    )
