# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import msgspec
import pytest
import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.worker_base import WorkerBase

pytestmark = pytest.mark.cpu_test

BASE_BLOCK_SIZE = 1536
DCP_WORLD_SIZE = 8


@pytest.mark.parametrize("interleave,expected", [(None, 16), (8, 8)])
def test_global_layout_finalizes_worker_interleave_before_profiling(
    interleave, expected
):
    """Workers with different local blocks must use the global interleave."""
    config = VllmConfig(
        cache_config=CacheConfig(block_size=64),
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            distributed_executor_backend="mp",
            cp_kv_cache_interleave_size=interleave or 1,
            _allow_auto_resolve_cp_interleave_size=interleave is None,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="NixlConnector", kv_role="kv_both"
        ),
    )
    config.model_config = SimpleNamespace(supports_kv_cache_grouping=False)
    config.cache_config.kv_cache_layout = "BLHNC"
    workers = []
    specs = []
    for i, block_size in enumerate((16, 64)):
        worker = object.__new__(WorkerBase)
        worker.vllm_config = deepcopy(config)
        worker.vllm_config.cache_config.kv_cache_layout = None
        workers.append(worker)
        specs.append(
            {
                f"layer{i}": FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.float16,
                )
            }
        )
    parallel_configs = [config.parallel_config] + [
        worker.vllm_config.parallel_config for worker in workers
    ]

    class ProfilingStarted(Exception):
        pass

    class LocalExecutor:
        set_kv_cache_layout = Executor.set_kv_cache_layout

        def get_kv_cache_specs(self):
            return specs

        def get_supported_kv_cache_layouts(self):
            return [["BLHNC"] for _ in workers]

        def collective_rpc(self, method, args):
            for worker in workers:
                getattr(worker, method)(*args)

        def determine_available_memory(self):
            assert all(
                parallel.cp_kv_cache_interleave_size == expected
                for parallel in parallel_configs
            )
            assert all(
                worker.vllm_config.cache_config.kv_cache_layout == "BLHNC"
                for worker in workers
            )
            raise ProfilingStarted

    engine = object.__new__(EngineCore)
    engine.model_executor = LocalExecutor()
    with pytest.raises(ProfilingStarted):
        engine._initialize_kv_caches(config)


def _make_engine_core_with_dcp_manager() -> EngineCore:
    mla_spec = MLAAttentionSpec(
        block_size=BASE_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        block_size=BASE_BLOCK_SIZE,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["mla"], mla_spec),
            KVCacheGroupSpec(["mamba"], mamba_spec),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=BASE_BLOCK_SIZE * DCP_WORLD_SIZE * 2,
        scheduler_block_size=BASE_BLOCK_SIZE * DCP_WORLD_SIZE,
        hash_block_size=BASE_BLOCK_SIZE,
        dcp_world_size=DCP_WORLD_SIZE,
    )
    scheduler = SimpleNamespace(kv_cache_config=config, kv_cache_manager=manager)
    engine_core = object.__new__(EngineCore)
    engine_core.scheduler = scheduler
    return engine_core


def test_kv_cache_group_metadata_uses_effective_dcp_block_sizes() -> None:
    engine_core = _make_engine_core_with_dcp_manager()

    metadata = engine_core.get_kv_cache_group_metadata()

    # DCP scales attention blocks but does not scale replicated Mamba state.
    assert [item["block_size"] for item in metadata] == [12288, 1536]
    assert [item["kind"] for item in metadata] == ["mla_attention", "mamba"]
    assert [item["prefix_cacheable"] for item in metadata] == [True, True]
    assert [item["prefix_replay_tokens"] for item in metadata] == [0, 0]

    managers = engine_core.scheduler.kv_cache_manager.coordinator.single_type_managers
    assert [item["block_size"] for item in metadata] == [
        manager.block_size for manager in managers
    ]


@pytest.mark.parametrize(
    "spec_type,bounded_replay,wrapped,cacheable,replay_tokens",
    [
        pytest.param(CircularBufferSpec, False, False, False, 0, id="circular-buffer"),
        pytest.param(SlidingWindowMLASpec, False, False, True, 0, id="sliding-mla"),
        pytest.param(SlidingWindowMLASpec, True, False, False, 128, id="replay"),
        pytest.param(SlidingWindowMLASpec, True, True, False, 128, id="wrapped-replay"),
    ],
)
def test_kv_cache_group_metadata_reports_prefix_policy(
    spec_type, bounded_replay, wrapped, cacheable, replay_tokens
) -> None:
    """The same attention kind can require a different prefix reuse policy."""
    kwargs = dict(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )
    if spec_type is SlidingWindowMLASpec:
        kwargs["sliding_window"] = 128
        kwargs["bounded_replay"] = bounded_replay
    spec = spec_type(**kwargs)
    if wrapped:
        spec = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"layer": spec})
    engine_core = object.__new__(EngineCore)
    engine_core.scheduler = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)]
        ),
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(
                single_type_managers=[SimpleNamespace(block_size=16)]
            )
        ),
    )

    metadata = msgspec.msgpack.decode(
        msgspec.msgpack.encode(engine_core.get_kv_cache_group_metadata())
    )

    assert metadata[0]["prefix_cacheable"] is cacheable
    assert metadata[0]["prefix_replay_tokens"] == replay_tokens
