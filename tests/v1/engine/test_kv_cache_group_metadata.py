# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.utils import new_kv_cache_group
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
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
        cache_config=CacheConfig(block_size=64, enable_extensible_kv_cache=False),
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
        spec = AttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=64, dtype=torch.float16
        )
        specs.append({f"layer{i}": FullAttentionManager.layer_config(spec)})
    parallel_configs = [config.parallel_config] + [
        worker.vllm_config.parallel_config for worker in workers
    ]

    class ProfilingStarted(Exception):
        pass

    class LocalExecutor:
        set_kv_cache_layout = Executor.set_kv_cache_layout

        def get_kv_cache_layer_configs(self):
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


def _make_dcp_manager() -> KVCacheManager:
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
    )
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            new_kv_cache_group(["mla"], mla_spec),
            new_kv_cache_group(
                ["mamba"], mamba_spec, MambaManager.config(mamba_cache_mode="align")
            ),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=BASE_BLOCK_SIZE * DCP_WORLD_SIZE * 2,
        scheduler_block_size=BASE_BLOCK_SIZE * DCP_WORLD_SIZE,
        hash_block_size=BASE_BLOCK_SIZE,
        dcp_world_size=DCP_WORLD_SIZE,
        enable_kv_cache_events=True,
    )
    return manager


def _make_engine_core_with_dcp_manager() -> EngineCore:
    manager = _make_dcp_manager()
    scheduler = SimpleNamespace(
        kv_cache_config=manager.kv_cache_config, kv_cache_manager=manager
    )
    engine_core = object.__new__(EngineCore)
    engine_core.scheduler = scheduler
    return engine_core


def test_kv_cache_group_metadata_uses_effective_dcp_block_sizes() -> None:
    engine_core = _make_engine_core_with_dcp_manager()

    metadata = engine_core.get_kv_cache_group_metadata()

    # DCP scales attention blocks but does not scale replicated Mamba state.
    assert [item["block_size"] for item in metadata] == [12288, 1536]
    assert [item["kind"] for item in metadata] == ["mla_attention", "mamba"]

    managers = engine_core.scheduler.kv_cache_manager.coordinator.single_type_managers
    assert [item["block_size"] for item in metadata] == [
        manager.block_size for manager in managers
    ]


@pytest.mark.parametrize("lookahead", [0, 1])
@pytest.mark.parametrize("salt", [None, "tenant-a"])
@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("multimodal", [False, True])
def test_hash_requests_transport_preserves_kv_event_hashes(
    lookahead, salt, wire, multimodal
):
    import msgspec

    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.engine import (
        EngineCoreHashRequest,
        EngineCoreRequest,
    )
    from vllm.v1.request import Request

    manager = _make_dcp_manager()
    init_none_hash(sha256)
    block_hasher = get_request_block_hasher(BASE_BLOCK_SIZE, sha256, lookahead)
    core_request = EngineCoreRequest(
        request_id="hash-only",
        prompt_token_ids=list(range(BASE_BLOCK_SIZE * DCP_WORLD_SIZE + 1)),
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="adapter:image-a",
                mm_position=PlaceholderRange(offset=100, length=BASE_BLOCK_SIZE * 2),
            )
        ]
        if multimodal
        else None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0,
        lora_request=LoRARequest("adapter", 1, "/adapter") if multimodal else None,
        cache_salt=salt,
        data_parallel_rank=None,
    )
    expected = Request.from_engine_core_request(core_request, block_hasher).block_hashes
    transport = EngineCoreHashRequest.from_request(core_request)
    if wire:
        transport = msgspec.convert(
            msgspec.msgpack.decode(msgspec.msgpack.encode(transport)),
            type=EngineCoreHashRequest,
        )
    hashed = Request(
        request_id="hash",
        prompt_token_ids=transport.prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=transport.mm_features,
        lora_request=transport.lora_request,
        cache_salt=transport.cache_salt,
        block_hasher=block_hasher,
    )
    assert hashed.block_hashes == expected

    from vllm.distributed.kv_events import BlockStored
    from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

    generation = Request.from_engine_core_request(core_request, block_hasher)
    assert (
        manager.allocate_slots(generation, BASE_BLOCK_SIZE * DCP_WORLD_SIZE) is not None
    )
    stores = [
        event for event in manager.take_events() if isinstance(event, BlockStored)
    ]
    assert stores
    for event in stores:
        assert all(
            h in [maybe_convert_block_hash(value) for value in expected]
            for h in event.block_hashes
        )
