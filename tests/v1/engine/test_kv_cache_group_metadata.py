# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import torch
import zmq

from tests.v1.core.utils import new_kv_cache_group
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.kv_events import KVEventsConfig
from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.engine import EngineCoreReadyResponse
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import MPClient
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

    managers = engine_core.scheduler.kv_cache_manager.coordinator.single_type_managers
    assert [item["block_size"] for item in metadata] == [
        manager.block_size for manager in managers
    ]


@pytest.fixture
def ready_response() -> EngineCoreReadyResponse:
    return EngineCoreReadyResponse(
        max_model_len=256,
        num_gpu_blocks=100,
        block_size=1056,
        dp_stats_address=None,
        dtype="bfloat16",
        vllm_version="test",
        world_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_rank=0,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        instance_id="test-instance",
        supports_lora=False,
        max_loras=0,
    )


@contextmanager
def ready_client(*payloads: bytes) -> Iterator[MPClient]:
    """Construct a real client with ready messages from external ZMQ peers."""
    config = VllmConfig(
        model_config=ModelConfig(
            model="vllm-project/tiny-random-LlamaForCausalLM",
            max_model_len=512,
            enforce_eager=True,
        ),
        cache_config=CacheConfig(block_size=16),
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            data_parallel_size=len(payloads), data_parallel_size_local=len(payloads)
        ),
    )
    addresses = {
        "input_address": get_open_zmq_ipc_path(),
        "output_address": get_open_zmq_ipc_path(),
    }
    with ExitStack() as stack:
        context = stack.enter_context(zmq.Context())
        for rank, payload in enumerate(payloads):
            socket = stack.enter_context(context.socket(zmq.DEALER))
            socket.linger = 0
            socket.setsockopt(zmq.IDENTITY, rank.to_bytes(2, "little"))
            socket.connect(addresses["input_address"])
            socket.send(payload)

        client = MPClient(False, config, Executor, False, client_addresses=addresses)
        stack.callback(client.shutdown)
        yield client


@pytest.mark.parametrize(
    ("effective_size", "other_size"),
    [(None, None), (4224, 4224), (4224, 1056), (4224, None)],
)
def test_apply_ready_response_syncs_engine_metadata(
    ready_response, effective_size, other_size
):
    response = replace(
        ready_response,
        coord_store_port=1234,
        coordinator_input_address="tcp://127.0.0.1:1235",
        coordinator_output_address="tcp://127.0.0.1:1236",
    )
    fields = msgspec.msgpack.decode(msgspec.msgpack.encode(response))
    if effective_size is None:
        del fields["effective_attention_block_size"]
    else:
        fields["effective_attention_block_size"] = effective_size

    with ready_client(msgspec.msgpack.encode(fields)) as client:
        cache_config = client.vllm_config.cache_config
        assert cache_config.block_size == 1056
        assert cache_config.effective_attention_block_size == effective_size
        assert cache_config.num_gpu_blocks == 100
        assert client.vllm_config.model_config.max_model_len == 256

        fields["effective_attention_block_size"] = other_size
        client._apply_ready_response(msgspec.msgpack.encode(fields))
        assert cache_config.effective_attention_block_size == (
            effective_size if effective_size == other_size else None
        )

        client._apply_ready_response(b"")
        assert cache_config.effective_attention_block_size is None
        assert client.vllm_config.parallel_config._coord_store_port == 1234
        assert client.coordinator_input_address == "tcp://127.0.0.1:1235"
        assert client.coordinator_output_address == "tcp://127.0.0.1:1236"


def test_apply_ready_response_syncs_mamba_block_size(ready_response):
    response = replace(ready_response, mamba_block_size=1056)
    with ready_client(msgspec.msgpack.encode(response)) as client:
        assert client.vllm_config.cache_config.mamba_block_size == 1056


def test_apply_ready_response_retains_kv_event_sources(ready_response):
    """Retain publishers by DP rank and replace the port when a rank returns."""
    rank1 = KVEventsConfig(
        enable_kv_cache_events=True, publisher="zmq", endpoint="tcp://10.0.0.2:41233"
    )
    rank0_response = replace(ready_response, world_size=2, data_parallel_size=2)
    rank1_response = replace(
        rank0_response, data_parallel_rank=1, kv_events_config=rank1
    )
    with ready_client(
        msgspec.msgpack.encode(rank0_response), msgspec.msgpack.encode(rank1_response)
    ) as client:
        assert client.get_kv_event_sources() == {1: rank1}

        rank1_again = replace(rank1, endpoint="tcp://10.0.0.2:47001")
        client._apply_ready_response(
            msgspec.msgpack.encode(
                replace(rank1_response, kv_events_config=rank1_again)
            )
        )
        assert client.get_kv_event_sources() == {1: rank1_again}
