# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import msgspec
import openai  # use the official client for correctness check
import psutil
import pytest
import pytest_asyncio
import requests
import zmq
from prometheus_client.parser import text_string_to_metric_families
from transformers import Qwen3MoeConfig

from tests.utils import RemoteOpenAIServer
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed.elastic_ep.bootstrap import (
    ExternalElasticEPBootstrapServer,
    prepare_external_elastic_ep_launch,
)
from vllm.distributed.elastic_ep.external_elastic_ep import (
    ExternalElasticEPScaleCoordinator,
    ExternalElasticEPScaleUpHandshakeMetadata,
    ExternalElasticEPScaleUpHandshakeServer,
    _PreparedExternalElasticEPScale,
)
from vllm.distributed.utils import create_tcp_store
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.network_utils import get_open_port, make_zmq_socket
from vllm.v1.engine import (
    EngineCoreRequest,
    EngineCoreRequestType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
)
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.engine.utils import EngineZmqAddresses
from vllm.v1.executor import UniProcExecutor

if TYPE_CHECKING:
    from vllm.v1.engine.core_client import DPAsyncMPClient

MODEL_NAME = os.getenv("MODEL_NAME", "ibm-research/PowerMoE-3b")
ELASTIC_EP_MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"

# Number of data parallel ranks for external LB testing
DP_SIZE = int(os.getenv("DP_SIZE", "2"))
# Default tensor parallel size to use
TP_SIZE = int(os.getenv("TP_SIZE", "1"))


@pytest.mark.parametrize(
    "options",
    [
        {"enable_elastic_ep": False},
        {"data_parallel_rank": None},
        {"data_parallel_rank": -1},
        {"data_parallel_rank": 4},
        {"elastic_ep_initial_dp_size": 0},
        {"elastic_ep_initial_dp_size": 1},
        {"elastic_ep_initial_dp_size": 5},
        {"data_parallel_hybrid_lb": True},
        {"data_parallel_size_local": 0},
        {"data_parallel_size_local": -1},
        {"data_parallel_size_local": 3},
        {"data_parallel_backend": "ray"},
        {"data_parallel_rpc_port": 0},
        {"data_parallel_rpc_port": 65536},
        {"elastic_ep_allocation_index": 0},
        {"elastic_ep_allocation_index": -1, "data_parallel_rank": None},
        {"elastic_ep_allocation_index": 4, "data_parallel_rank": None},
        {
            "elastic_ep_allocation_index": 0,
            "data_parallel_rank": None,
            "data_parallel_external_lb": True,
        },
        {
            "elastic_ep_allocation_index": 0,
            "data_parallel_rank": None,
            "elastic_ep_auto_bootstrap": False,
        },
        {
            "elastic_ep_allocation_index": 0,
            "data_parallel_rank": None,
            "data_parallel_size_local": 2,
            "elastic_ep_initial_dp_size": 3,
        },
        {
            "elastic_ep_allocation_index": 0,
            "data_parallel_rank": None,
            "data_parallel_size_local": 2,
            "elastic_ep_max_dp_size": 3,
        },
    ],
)
def test_external_elastic_ep_auto_bootstrap_rejects_unsupported_launch(options):
    arguments: dict[str, Any] = dict(
        elastic_ep_auto_bootstrap=True,
        elastic_ep_initial_dp_size=2,
        data_parallel_rank=0,
        elastic_ep_max_dp_size=4,
        enable_elastic_ep=True,
        enable_expert_parallel=True,
        enable_eplb=True,
    )
    arguments.update(options)
    with pytest.raises(ValueError):
        prepare_external_elastic_ep_launch(EngineArgs(**arguments))


@pytest.mark.parametrize("scale_up", [False, True])
def test_external_elastic_ep_bootstrap_binds_one_launch_to_one_handshake(scale_up):
    first = ExternalElasticEPBootstrapServer(4, [2, 3], scale_up=scale_up)
    launch = first.discover(2, "process-a")
    assert launch.identity is not None
    assert launch.scale_up == scale_up
    assert first.discover(2, "process-a") == launch
    first.validate(2, list(launch.identity))
    with pytest.raises(ValueError, match="another launch"):
        first.discover(2, "process-b")
    with pytest.raises(ValueError, match="expected bootstrap"):
        first.discover(1, "process-c")
    with pytest.raises(ValueError, match="does not match"):
        first.validate(2, None)

    later = ExternalElasticEPBootstrapServer(4, [2, 3], scale_up=True)
    with pytest.raises(ValueError, match="does not match"):
        later.validate(2, list(launch.identity))
    later.validate(3, None)  # Explicit legacy launches remain compatible.


@pytest.mark.parametrize("enable_fault_tolerance", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize(
    "rank,target_size,initial_size,local_size",
    [(2, 3, 2, 1), (2, 4, 2, 1), (3, 4, 2, 1), (2, 3, 4, 1), (8, 12, 8, 4)],
)
def test_external_elastic_ep_auto_bootstrap_uses_native_scale_handshake(
    monkeypatch,
    rank,
    target_size,
    initial_size,
    local_size,
    enable_fault_tolerance,
    indexed,
):
    """The active handshake resolves append startup, including retired initial ranks."""
    port = get_open_port()
    request = ReconfigureDistributedRequest(
        new_data_parallel_size=target_size,
        new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
        new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
        new_data_parallel_master_ip="127.0.0.1",
        new_data_parallel_master_port=12345,
        new_data_parallel_master_port_list=[12346],
        coord_store_port=12347,
    )
    server = ExternalElasticEPScaleUpHandshakeServer(
        handshake_address=f"tcp://127.0.0.1:{port}",
        expected_new_ranks=list(range(2, target_size)),
        addresses=EngineZmqAddresses(inputs=["tcp://127.0.0.1:12348"], outputs=[]),
        bootstrap=request,
        num_redundant_experts=0,
        elastic_ep_max_dp_size=max(4, target_size),
    )
    monkeypatch.delenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", raising=False)
    server.start()
    try:
        args = EngineArgs(
            elastic_ep_auto_bootstrap=True,
            elastic_ep_initial_dp_size=initial_size,
            **(
                {"elastic_ep_allocation_index": rank // local_size}
                if indexed
                else {"data_parallel_rank": rank}
            ),
            data_parallel_size_local=local_size,
            data_parallel_address="127.0.0.1",
            data_parallel_rpc_port=port,
            elastic_ep_max_dp_size=max(4, target_size),
            enable_elastic_ep=True,
            enable_expert_parallel=True,
            enable_eplb=True,
            enable_fault_tolerance=enable_fault_tolerance,
            all2all_backend=(
                "nixl_ep" if enable_fault_tolerance else "allgather_reducescatter"
            ),
        )
        launch = prepare_external_elastic_ep_launch(args)
        assert args.elastic_ep_initial_dp_size == initial_size
        assert launch.data_parallel_size == target_size
        assert launch.first_rank == rank
        assert launch.scale_up
        assert args.enable_fault_tolerance is enable_fault_tolerance
        for engine_rank in range(rank, rank + local_size):
            with (
                zmq.Context() as ctx,
                make_zmq_socket(
                    ctx,
                    server.handshake_address,
                    zmq.DEALER,
                    bind=False,
                    identity=engine_rank.to_bytes(2, "little"),
                    linger=0,
                ) as socket,
            ):
                assert launch.identity is not None
                socket.send(
                    msgspec.msgpack.encode(
                        {
                            "status": "HELLO",
                            "bootstrap": ("stale", launch.identity[1]),
                        }
                    )
                )
                assert socket.poll(5000)
                assert (
                    "does not match" in msgspec.msgpack.decode(socket.recv())["error"]
                )
                server.raise_if_failed()
                socket.send(
                    msgspec.msgpack.encode(
                        {"status": "HELLO", "bootstrap": launch.identity}
                    )
                )
                assert socket.poll(5000)
                metadata = msgspec.msgpack.decode(
                    socket.recv(), type=ExternalElasticEPScaleUpHandshakeMetadata
                )
                assert metadata.engine_metadata.addresses == server.addresses
                assert metadata.engine_metadata.parallel_config == {
                    "data_parallel_master_ip": request.new_data_parallel_master_ip,
                    "data_parallel_master_port": request.new_data_parallel_master_port,
                    "_data_parallel_master_port_list": (
                        request.new_data_parallel_master_port_list
                    ),
                    "data_parallel_size": target_size,
                    "_coord_store_port": request.coord_store_port,
                }
                socket.send(msgspec.msgpack.encode({"status": "READY"}))
    finally:
        server.stop()


def test_external_elastic_ep_bootstrap_reserves_a_range_atomically():
    server = ExternalElasticEPBootstrapServer(12, range(8, 12), scale_up=True)
    first = server.discover(10, "existing")
    with pytest.raises(ValueError, match="another launch"):
        server.discover(8, "overlap", 4)
    # The failed range must not claim its free prefix or suffix.
    prefix = server.discover(8, "prefix", 2)
    suffix = server.discover(11, "suffix")
    for rank, launch in [(8, prefix), (9, prefix), (10, first), (11, suffix)]:
        assert launch.identity is not None
        server.validate(rank, list(launch.identity))
    assert server.discover(8, "prefix", 2) == prefix
    with pytest.raises(ValueError, match="different rank range"):
        server.discover(8, "prefix")


@pytest.mark.parametrize("rank,count", [(7, 2), (11, 2), (8, 0), (8, -1), (8, True)])
def test_external_elastic_ep_bootstrap_rejects_an_invalid_range(rank, count):
    server = ExternalElasticEPBootstrapServer(12, range(8, 12), scale_up=True)
    with pytest.raises(ValueError):
        server.discover(rank, "invalid", count)
    launch = server.discover(8, "valid", 4)
    assert launch.identity is not None
    for engine_rank in range(8, 12):
        server.validate(engine_rank, list(launch.identity))


def test_external_elastic_ep_bootstrap_requires_the_whole_range_to_await_startup():
    server = ExternalElasticEPBootstrapServer(8, range(4, 8), scale_up=False)
    with pytest.raises(ValueError, match="expected bootstrap"):
        server.discover(4, "late", 4, available_ranks={4, 5, 7})
    launch = server.discover(4, "initial", 2, available_ranks={4, 5, 7})
    assert launch.identity is not None
    server.validate(4, list(launch.identity))
    server.validate(5, list(launch.identity))


@pytest.mark.parametrize("local_size", [None, 1, 4])
def test_external_elastic_ep_allocation_index_zero_initial_world(
    local_size, monkeypatch
):
    monkeypatch.delenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", raising=False)
    args = EngineArgs(
        elastic_ep_auto_bootstrap=True,
        elastic_ep_initial_dp_size=8,
        elastic_ep_max_dp_size=12,
        elastic_ep_allocation_index=0,
        data_parallel_size_local=local_size,
        enable_elastic_ep=True,
        enable_expert_parallel=True,
        enable_eplb=True,
    )
    launch = prepare_external_elastic_ep_launch(args)
    assert launch.first_rank == 0
    assert launch.data_parallel_size == 8
    assert not launch.scale_up


@pytest.mark.parametrize(
    "local_size,nnodes", [(None, 1), (None, 2), (1, 1), (4, 1), (8, 1)]
)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Real EngineArgs with EPLB requires a CUDA-like platform",
)
def test_external_lb_elastic_ep_local_range_config(
    local_size, nnodes, indexed, tmp_path, monkeypatch
):
    Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=16,
        num_experts_per_tok=2,
        vocab_size=128,
    ).save_pretrained(tmp_path)
    monkeypatch.setenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "0")
    config = (
        EngineArgs(
            model=str(tmp_path),
            elastic_ep_auto_bootstrap=True,
            elastic_ep_initial_dp_size=8,
            elastic_ep_max_dp_size=8 + (local_size or 1),
            **(
                {"elastic_ep_allocation_index": 0}
                if indexed
                else {"data_parallel_rank": 0}
            ),
            data_parallel_size_local=local_size,
            nnodes=nnodes,
            enable_elastic_ep=True,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config={"use_async": False},
        )
        .create_engine_config()
        .parallel_config
    )
    assert config.data_parallel_size == 8
    assert config.data_parallel_rank == 0
    resolved_local_size = local_size or 1
    assert config.data_parallel_size_local == resolved_local_size
    assert config.data_parallel_external_lb is (resolved_local_size == 1)
    assert config.data_parallel_hybrid_lb is (1 < resolved_local_size < 8)
    assert config.elastic_ep_auto_bootstrap
    assert config.uses_external_elastic_ep


@pytest.mark.parametrize("enable_fault_tolerance", [False, True])
def test_external_elastic_ep_auto_bootstrap_preserves_executor_launch_arguments(
    enable_fault_tolerance,
):
    """Bootstrap selects DP startup, not the model executor's node geometry."""
    args = EngineArgs(
        elastic_ep_auto_bootstrap=True,
        elastic_ep_initial_dp_size=2,
        data_parallel_size_local=1,
        data_parallel_rank=0,
        tensor_parallel_size=2,
        nnodes=4,
        node_rank=0,
        distributed_executor_backend="mp",
        elastic_ep_max_dp_size=4,
        enable_elastic_ep=True,
        enable_expert_parallel=True,
        enable_eplb=True,
        enable_fault_tolerance=enable_fault_tolerance,
        all2all_backend=(
            "nixl_ep" if enable_fault_tolerance else "allgather_reducescatter"
        ),
    )
    launch = prepare_external_elastic_ep_launch(args)
    assert not launch.scale_up
    assert launch.data_parallel_size == 2
    assert (args.nnodes, args.node_rank, args.tensor_parallel_size) == (4, 0, 2)
    assert args.distributed_executor_backend == "mp"
    assert args.enable_fault_tolerance is enable_fault_tolerance


@pytest.mark.asyncio
async def test_external_elastic_ep_local_range_routes_after_old_size_stats():
    """Incomplete global snapshots cannot replace a joiner's local load state."""
    config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=4,
            data_parallel_rank=2,
            data_parallel_size_local=2,
            data_parallel_hybrid_lb=True,
            distributed_executor_backend="mp",
        )
    )
    input_address = f"tcp://127.0.0.1:{get_open_port()}"
    output_address = f"tcp://127.0.0.1:{get_open_port()}"
    with (
        zmq.Context() as engine_ctx,
        zmq.asyncio.Context() as stats_ctx,
        engine_ctx.socket(zmq.DEALER) as rank2,
        engine_ctx.socket(zmq.DEALER) as rank3,
        make_zmq_socket(
            stats_ctx, "tcp://127.0.0.1:0", zmq.XPUB, bind=True, linger=0
        ) as coordinator,
    ):
        for rank, engine in enumerate((rank2, rank3), start=2):
            engine.setsockopt(zmq.IDENTITY, rank.to_bytes(2, "little"))
            engine.setsockopt(zmq.LINGER, 0)
            engine.connect(input_address)
            engine.send(b"")
        client = DPLBAsyncMPClient(
            config,
            UniProcExecutor,
            log_stats=False,
            client_addresses={
                "input_address": input_address,
                "output_address": output_address,
                "stats_update_address": coordinator.getsockopt_string(
                    zmq.LAST_ENDPOINT
                ),
            },
        )
        try:
            assert await asyncio.wait_for(coordinator.recv(), timeout=5) == b"\x01"
            remote_counts = [[9, 9, 0.8], [9, 9, 0.8]]
            snapshots = [
                (remote_counts, rank2),
                (remote_counts + [[4, 4, 0.8]], rank3),
                (remote_counts + [[7, 4, 0.5], [0, 0, 0.0]], rank3),
                (remote_counts, rank3),
            ]

            async def wait_for_snapshot(wave):
                while client.current_wave != wave:
                    await asyncio.sleep(0)

            for wave, (counts, expected_engine) in enumerate(snapshots, start=1):
                expected_counts = [
                    row.copy()
                    for row in (counts[2:4] if len(counts) == 4 else client.lb_engines)
                ]
                running = wave % 2 == 1
                await coordinator.send(msgspec.msgpack.encode((counts, wave, running)))
                await asyncio.wait_for(wait_for_snapshot(wave), timeout=5)
                assert client.engines_running is running
                request = EngineCoreRequest(
                    request_id=f"joiner-{wave}",
                    prompt_token_ids=[1],
                    mm_features=None,
                    sampling_params=SamplingParams(max_tokens=1),
                    pooling_params=None,
                    arrival_time=time.time(),
                    lora_request=None,
                    cache_salt=None,
                    data_parallel_rank=None,
                )
                await client.add_request_async(request)
                assert await asyncio.to_thread(expected_engine.poll, 5000)
                assert (
                    expected_engine.recv_multipart()[0]
                    == EngineCoreRequestType.ADD.value
                )
                expected_counts[0 if expected_engine is rank2 else 1][0] += 1
                assert client.lb_engines == expected_counts
                assert client.engine_ranks_managed == [2, 3]
        finally:
            client.shutdown()
            await asyncio.gather(
                client.resources.stats_update_task,
                client.resources.output_queue_task,
                return_exceptions=True,
            )
            client.resources.ctx.term()


def test_external_elastic_ep_auto_bootstrap_preserves_initial_rank_connection(
    monkeypatch,
):
    """Discovery must not compete with an EngineCore's ZMQ routing identity."""
    port = get_open_port()
    address = f"tcp://127.0.0.1:{port}"
    rank_identity = (1).to_bytes(2, "little")
    responder = ExternalElasticEPBootstrapServer(2, [0, 1], scale_up=False)
    args = EngineArgs(
        elastic_ep_auto_bootstrap=True,
        elastic_ep_initial_dp_size=2,
        data_parallel_rank=1,
        data_parallel_address="127.0.0.1",
        data_parallel_rpc_port=port,
        elastic_ep_max_dp_size=4,
        enable_elastic_ep=True,
        enable_expert_parallel=True,
        enable_eplb=True,
    )
    monkeypatch.delenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", raising=False)
    monkeypatch.setenv("VLLM_ENGINE_READY_TIMEOUT_S", "5")
    with (
        zmq.Context() as ctx,
        make_zmq_socket(ctx, address, zmq.ROUTER, bind=True, linger=0) as listener,
        make_zmq_socket(
            ctx, address, zmq.DEALER, bind=False, identity=rank_identity, linger=0
        ) as engine,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        engine.send(b"engine-connected")
        assert listener.poll(5000)
        assert listener.recv_multipart() == [rank_identity, b"engine-connected"]
        launch_request = executor.submit(prepare_external_elastic_ep_launch, args)
        assert listener.poll(5000)
        identity, payload = listener.recv_multipart()
        assert identity != rank_identity
        responder.reply(listener, identity, msgspec.msgpack.decode(payload))
        launch = launch_request.result(timeout=5)
        assert launch.data_parallel_size == 2
        assert not launch.scale_up
        assert launch.identity is not None
        engine.send(msgspec.msgpack.encode({"bootstrap": launch.identity}))
        assert listener.poll(5000)
        identity, payload = listener.recv_multipart()
        assert identity == rank_identity
        responder.validate(1, msgspec.msgpack.decode(payload)["bootstrap"])


@pytest.fixture
def external_elastic_ep_local_range():
    control, reconfig = (
        create_tcp_store(
            "127.0.0.1", 0, is_master=True, world_size=-1, wait_for_workers=False
        )
        for _ in range(2)
    )
    return _PreparedExternalElasticEPScale(
        control_store=control,
        reconfig_store=reconfig,
        epoch="range-resize",
        bootstrap=ReconfigureDistributedRequest(
            new_data_parallel_size=12,
            new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_master_ip="127.0.0.1",
            new_data_parallel_master_port=12345,
            new_data_parallel_master_port_list=[12346],
            coord_store_port=reconfig.port,
        ),
        dp_rank=0,
        cur_data_parallel_size=8,
        num_redundant_experts=32,
        scale_up=True,
        handshake_server=None,
        local_size=4,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", ["prepare_ready", "commit_ready"])
@pytest.mark.parametrize("peer_fails", [False, True])
async def test_external_elastic_ep_local_range_barrier(
    external_elastic_ep_local_range, barrier, peer_fails
):
    """An owner publishes all four ranks, then waits for every other owner."""
    prepared = external_elastic_ep_local_range
    coordinator = ExternalElasticEPScaleCoordinator
    task = asyncio.create_task(
        coordinator._wait_for_client_barrier(prepared, barrier, timeout_s=5)
    )
    observer = create_tcp_store(
        "127.0.0.1",
        prepared.reconfig_store.port,
        is_master=False,
        world_size=-1,
        wait_for_workers=False,
    )
    try:
        await asyncio.to_thread(
            observer.wait,
            [coordinator.key(prepared.epoch, barrier, rank) for rank in range(4)],
            timedelta(seconds=5),
        )
        assert not task.done()
        if peer_fails:
            prepared.control_store.set(
                coordinator.key(prepared.epoch, "error"), b"peer range failed"
            )
            with pytest.raises(RuntimeError, match="peer range failed"):
                await asyncio.wait_for(task, timeout=5)
        else:
            peer = replace(prepared, dp_rank=4)
            await asyncio.wait_for(
                asyncio.gather(
                    task,
                    coordinator._wait_for_client_barrier(peer, barrier, timeout_s=5),
                ),
                timeout=5,
            )
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_rank,scale_up", [(0, True), (4, False), (8, False)])
@pytest.mark.parametrize("peer_fails", [False, True])
async def test_external_elastic_ep_local_range_completion(
    external_elastic_ep_local_range, first_rank, scale_up, peer_fails
):
    """Finishing only the first three local engines never completes the owner."""
    prepared = external_elastic_ep_local_range
    coordinator = ExternalElasticEPScaleCoordinator
    bootstrap = msgspec.structs.replace(
        prepared.bootstrap, new_data_parallel_size=12 if scale_up else 8
    )
    notification = (
        "shutdown_complete" if not scale_up and first_rank >= 8 else "old_rank_finished"
    )
    for rank in range(first_rank, first_rank + 3):
        prepared.reconfig_store.set(
            coordinator.key(prepared.epoch, notification, rank), b"1"
        )
    task = asyncio.create_task(
        coordinator._wait_for_local_reconfig_finished(
            prepared.control_store,
            prepared.reconfig_store,
            prepared.epoch,
            bootstrap,
            first_rank,
            scale_up,
            timeout_s=5,
            local_size=4,
        )
    )
    try:
        await asyncio.sleep(0)  # Let the waiter inspect the incomplete barrier.
        assert not task.done()
        if peer_fails:
            prepared.control_store.set(
                coordinator.key(prepared.epoch, "error"), b"last rank failed"
            )
            with pytest.raises(RuntimeError, match="last rank failed"):
                await asyncio.wait_for(task, timeout=5)
        else:
            prepared.reconfig_store.set(
                coordinator.key(prepared.epoch, notification, first_rank + 3), b"1"
            )
            await asyncio.wait_for(task, timeout=5)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class DictStore:
    def __init__(self, values: dict[str, bytes]):
        self.values = values

    def check(self, keys: list[str]) -> bool:
        return all(key in self.values for key in keys)

    def get(self, key: str) -> bytes:
        return self.values[key]

    def set(self, key: str, value: bytes) -> None:
        self.values[key] = value


def test_external_elastic_ep_visible_epoch_has_bootstrap_and_phase(monkeypatch):
    client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_master_port=12345,
                _data_parallel_master_port_list=[12346, 12347],
            )
        )
    )
    coordinator = ExternalElasticEPScaleCoordinator(cast("DPAsyncMPClient", client))
    control_store, reconfig_store = DictStore({}), DictStore({})
    coordinator.reconfig_store_ref = reconfig_store
    monkeypatch.setattr(
        coordinator, "_setup_reconfig_bootstrap", lambda: ("127.0.0.1", 12348)
    )
    epoch, bootstrap = coordinator._prepare_reconfig_bootstrap(control_store, 2, 3)

    for store in (control_store, reconfig_store):
        # DictStore preserves insertion order: metadata must precede the epoch.
        assert list(store.values)[-1] == coordinator.key("current_epoch")
        assert store.get(coordinator.key(epoch, "bootstrap")) == msgspec.msgpack.encode(
            bootstrap
        )
        assert store.get(coordinator.key(epoch, "phase")) == b"preparing"
        assert store.get(coordinator.key("current_epoch")) == epoch.encode()
        monkeypatch.setattr(
            coordinator, "_get_reconfig_store", lambda store=store: store
        )
        assert coordinator.get_status()["requested_data_parallel_size"] == 3


def test_external_elastic_ep_calculates_target_expert_redundancy():
    client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(get_num_experts=lambda: 128),
            parallel_config=SimpleNamespace(
                eplb_config=SimpleNamespace(num_redundant_experts=0)
            ),
        )
    )
    coordinator = ExternalElasticEPScaleCoordinator(cast("DPAsyncMPClient", client))

    assert coordinator._calculate_num_redundant_experts(2, 3) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("dp_rank", range(4))
async def test_validate_expert_capacity_on_every_rank(dp_rank, monkeypatch):
    """Validate expert capacity on every rank before waiting for an epoch."""
    client = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(get_num_experts=lambda: 128),
            parallel_config=SimpleNamespace(
                data_parallel_rank=dp_rank,
                elastic_ep_max_dp_size=4,
                data_parallel_master_ip="127.0.0.1",
                _coord_store_port=12345,
                eplb_config=SimpleNamespace(num_redundant_experts=0),
            ),
        )
    )
    coordinator = ExternalElasticEPScaleCoordinator(cast("DPAsyncMPClient", client))
    store = DictStore({})
    monkeypatch.setattr(
        "vllm.distributed.utils.get_cached_tcp_store_client", lambda *args: store
    )

    with pytest.raises(ValueError, match="cannot retain all model experts"):
        await asyncio.wait_for(coordinator.prepare(4, 2), timeout=1)

    assert not store.values
    assert coordinator.prepared_scale is None


@pytest.mark.asyncio
async def test_external_elastic_ep_late_rank_observes_epoch_error():
    coordinator = ExternalElasticEPScaleCoordinator(
        cast("DPAsyncMPClient", SimpleNamespace())
    )
    epoch = "failed-epoch"
    store = DictStore(
        {
            coordinator.key("current_epoch"): epoch.encode(),
            coordinator.key(epoch, "error"): b"scale failed",
        }
    )

    with pytest.raises(RuntimeError, match="scale failed"):
        await coordinator._wait_for_bootstrap(store, requested_new_dp_size=3)


class ExternalLBServerManager:
    """Manages data parallel vLLM server instances for external
    load balancer testing."""

    def __init__(
        self,
        model_name: str,
        dp_size: int,
        api_server_count: int,
        base_server_args: list,
        tp_size: int = TP_SIZE,
        auto_bootstrap: bool = False,
        local_size: int = 1,
    ):
        self.model_name = model_name
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.api_server_count = api_server_count
        self.base_server_args = base_server_args
        self.auto_bootstrap = auto_bootstrap
        self.local_size = local_size
        self.servers: list[tuple[RemoteOpenAIServer, list[str]]] = []
        self.server_threads: list[threading.Thread] = []

    def start_rank(
        self,
        rank: int,
        dp_size: int | None = None,
        elastic_ep_scale_up_launch: bool = False,
    ) -> tuple[RemoteOpenAIServer, list[str]]:
        """Start a client owning one contiguous local DP range."""
        dp_size = self.dp_size if dp_size is None else dp_size
        server_args = self.base_server_args.copy()
        server_args.extend(
            [
                "--data-parallel-rank",
                str(rank),
                "--data-parallel-size-local",
                str(self.local_size),
                "--tensor-parallel-size",
                str(self.tp_size),
                "--port",
                str(8000 + rank),
                "--api-server-count",
                str(self.api_server_count),
            ]
        )

        if self.auto_bootstrap:
            server_args.extend(
                [
                    "--elastic-ep-auto-bootstrap",
                    "--elastic-ep-initial-dp-size",
                    str(self.dp_size),
                ]
            )
        else:
            server_args.extend(["--data-parallel-size", str(dp_size)])

        env_dict = {
            "VLLM_SERVER_DEV_MODE": "1",
            current_platform.device_control_env_var: ",".join(
                str(current_platform.device_id_to_physical_device_id(i))
                for i in range(
                    rank * self.tp_size, (rank + self.local_size) * self.tp_size
                )
            ),
        }
        if elastic_ep_scale_up_launch:
            env_dict["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "1"

        server = RemoteOpenAIServer(
            self.model_name,
            server_args,
            auto_port=False,
            env_dict=env_dict,
        )
        print(
            f"Server rank {rank} started successfully with "
            f"{self.api_server_count} API servers"
        )
        server_and_args = (server, server_args)
        self.servers.append(server_and_args)
        return server_and_args

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        """Start all initial server instances for external LB mode."""
        start_errors: list[Exception] = []
        for rank in range(0, self.dp_size, self.local_size):
            # Use a thread to start each server to allow parallel initialization
            def start_server(r: int):
                try:
                    self.start_rank(r)
                except Exception as e:
                    print(f"Failed to start server rank {r}: {e}")
                    start_errors.append(e)

            thread = threading.Thread(target=start_server, args=(rank,))
            thread.start()

            self.server_threads.append(thread)

        # Wait for all servers to start
        for thread in self.server_threads:
            thread.join()

        if start_errors:
            start_error = start_errors[0]
            raise RuntimeError("Failed to start external LB servers") from start_error

        if len(self.servers) * self.local_size != self.dp_size:
            raise Exception("Servers failed to start")

        return self.servers

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop all server instances."""
        servers = [s for s, _ in self.servers]
        self.servers.clear()
        try:
            RemoteOpenAIServer.shutdown_many(servers)
        except Exception as e:
            print(f"Error stopping servers: {e}")


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
    ]


@pytest.fixture(
    scope="module",
    params=[(1, False), (4, False), (1, True)],
    ids=["api-1-standard", "api-4-standard", "api-1-ep-enabled"],
)
def server_manager(request, default_server_args):
    api_server_count, enable_expert_parallel = request.param
    server_args = default_server_args.copy()
    if enable_expert_parallel:
        server_args.append("--enable-expert-parallel")

    server_manager = ExternalLBServerManager(
        MODEL_NAME, DP_SIZE, api_server_count, server_args
    )

    with server_manager:
        yield server_manager


@pytest.fixture
def servers(server_manager):
    return server_manager.servers


@pytest_asyncio.fixture
async def clients(servers: list[tuple[RemoteOpenAIServer, list[str]]]):
    # Create a client for each server
    async with AsyncExitStack() as stack:
        yield [
            await stack.enter_async_context(server.get_async_client())
            for server, _ in servers
        ]


def _get_parallel_config(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("server_info?config_format=json"))
    response.raise_for_status()

    vllm_config = response.json()["vllm_config"]
    return vllm_config["parallel_config"]


def _successful_requests_by_engine(
    server: RemoteOpenAIServer, ranks: range
) -> dict[int, float]:
    response = requests.get(server.url_for("metrics"), timeout=10)
    response.raise_for_status()
    counts = dict.fromkeys(ranks, 0.0)
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name == "vllm:request_success_total":
                counts[int(sample.labels["engine"])] += sample.value
    return counts


def _assert_local_range_serving(
    server: RemoteOpenAIServer, first_rank: int, local_size: int
) -> None:
    ranks = range(first_rank, first_rank + local_size)
    with server.get_client() as client:
        for local_rank in reversed(range(local_size)):
            expected = _successful_requests_by_engine(server, ranks)
            expected[first_rank + local_rank] += 1
            completion = client.completions.create(
                model=ELASTIC_EP_MODEL_NAME,
                prompt="Hello, my name is",
                max_tokens=5,
                temperature=0.0,
                extra_headers={"X-data-parallel-rank": str(local_rank)},
            )
            assert completion.choices[0].finish_reason in ("length", "stop")
            deadline = time.monotonic() + 10
            while (actual := _successful_requests_by_engine(server, ranks)) != expected:
                assert time.monotonic() < deadline, (actual, expected)
                time.sleep(0.1)
            print(f"Client {first_rank}, local rank {local_rank}: {actual}")

        # The unhinted balancer must not adopt remotely owned engines on resize.
        completion = client.completions.create(
            model=ELASTIC_EP_MODEL_NAME,
            prompt="Hello, my name is",
            max_tokens=5,
            temperature=0.0,
        )
        assert completion.choices[0].finish_reason in ("length", "stop")


def _assert_expert_parallel_config(
    servers: list[tuple[RemoteOpenAIServer, list[str]]],
):
    for server, server_args in servers:
        expected = "--enable-expert-parallel" in server_args
        parallel_config = _get_parallel_config(server)
        assert parallel_config["enable_expert_parallel"] is expected


def _send_scale_command(
    server: RemoteOpenAIServer, new_dp_size: int
) -> requests.Response:
    return requests.post(
        server.url_for("scale_elastic_ep"),
        json={"new_data_parallel_size": new_dp_size},
        timeout=600,
    )


def _wait_for_scale_requests_to_start(
    scale_requests: list[Future[requests.Response]],
    servers: list[RemoteOpenAIServer],
    timeout: float = 30,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for request in scale_requests:
            if request.done():
                response = request.result()
                pytest.fail(
                    "Scale request finished before the new rank was launched: "
                    f"{response.status_code} {response.text}"
                )

        statuses = [
            requests.post(server.url_for("is_scaling_elastic_ep"), timeout=10)
            for server in servers
        ]
        for status in statuses:
            status.raise_for_status()
        if all(status.json()["phase"] == "preparing" for status in statuses):
            return
        time.sleep(0.1)

    pytest.fail("Timed out waiting for all scale requests to start")


@pytest.mark.distributed(num_gpus=3)
@pytest.mark.skipif(
    TP_SIZE != 1 or current_platform.device_count() < 3,
    reason="External Elastic EP scale-up requires three GPUs with TP_SIZE=1",
)
@pytest.mark.skipif(
    os.getenv("VLLM_USE_RUST_FRONTEND") == "1",
    reason="Elastic EP scaling is not supported by the Rust frontend",
)
@pytest.mark.parametrize(
    "auto_bootstrap,enable_fault_tolerance",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_external_lb_elastic_ep_scale_up(
    default_server_args, auto_bootstrap, enable_fault_tolerance
) -> None:
    server_args = [
        *default_server_args,
        "--enable-expert-parallel",
        "--all2all-backend",
        "nixl_ep" if enable_fault_tolerance else "allgather_reducescatter",
        "--attention-backend",
        "TRITON_MLA",
        "--enable-elastic-ep",
        "--elastic-ep-max-dp-size",
        "3",
        "--enable-eplb",
        "--eplb-config",
        '{"communicator":"pynccl","num_redundant_experts":0,"use_async":false}',
    ]
    if enable_fault_tolerance:
        server_args.append("--enable-fault-tolerance")
    server_manager = ExternalLBServerManager(
        ELASTIC_EP_MODEL_NAME,
        dp_size=2,
        api_server_count=1,
        base_server_args=server_args,
        auto_bootstrap=auto_bootstrap,
    )

    with server_manager:
        existing_servers = [server for server, _ in server_manager.servers]
        with ThreadPoolExecutor(max_workers=2) as executor:
            scale_requests = [
                executor.submit(_send_scale_command, server, 3)
                for server in existing_servers
            ]
            _wait_for_scale_requests_to_start(scale_requests, existing_servers)
            server_manager.start_rank(
                rank=2,
                dp_size=None if auto_bootstrap else 3,
                elastic_ep_scale_up_launch=not auto_bootstrap,
            )

            for request in scale_requests:
                response = request.result(timeout=600)
                assert response.status_code == 200, response.text

        for server, launch_args in server_manager.servers:
            if auto_bootstrap:
                assert "--data-parallel-size" not in launch_args
                initial_size_index = launch_args.index("--elastic-ep-initial-dp-size")
                assert launch_args[initial_size_index + 1] == "2"
            parallel_config = _get_parallel_config(server)
            assert parallel_config["data_parallel_size"] == 3
            assert parallel_config["enable_fault_tolerance"] is enable_fault_tolerance
            assert parallel_config["eplb_config"]["num_redundant_experts"] == 32
            with server.get_client() as client:
                completion = client.completions.create(
                    model=ELASTIC_EP_MODEL_NAME,
                    prompt="Hello, my name is",
                    max_tokens=5,
                    temperature=0.0,
                )
            assert completion.choices[0].finish_reason in ("length", "stop")


@pytest.mark.parametrize(
    "first_ranks,local_size",
    [((0, 2), 2), ((0,), 4)],
    ids=["hybrid-local-ranges", "internal-local-range"],
)
@pytest.mark.parametrize("indexed", [False, True], ids=["explicit", "indexed"])
@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    TP_SIZE != 1 or current_platform.device_count() < 4,
    reason="External Elastic EP range bootstrap requires four GPUs with TP_SIZE=1",
)
@pytest.mark.skipif(
    os.getenv("VLLM_USE_RUST_FRONTEND") == "1",
    reason="Automatic Elastic EP bootstrap requires the Python frontend",
)
def test_external_lb_elastic_ep_bootstrap_local_range(
    default_server_args, first_ranks, local_size, indexed
):
    """A client independently routes inference to every engine it bootstraps."""
    server_args = [
        *default_server_args,
        "--enable-expert-parallel",
        "--enable-elastic-ep",
        "--elastic-ep-auto-bootstrap",
        "--elastic-ep-initial-dp-size",
        "4",
        "--elastic-ep-max-dp-size",
        "4",
        "--data-parallel-size-local",
        str(local_size),
        "--data-parallel-address",
        "127.0.0.1",
        "--data-parallel-rpc-port",
        str(get_open_port()),
        "--tensor-parallel-size",
        "1",
        "--api-server-count",
        "1",
        "--attention-backend",
        "TRITON_MLA",
        "--enable-eplb",
        "--eplb-config",
        '{"communicator":"pynccl","use_async":false}',
    ]
    servers: list[RemoteOpenAIServer] = []

    def start_client(first_rank: int):
        identity_args = (
            ["--elastic-ep-allocation-index", str(first_rank // local_size)]
            if indexed
            else ["--data-parallel-rank", str(first_rank)]
        )
        server = RemoteOpenAIServer(
            ELASTIC_EP_MODEL_NAME,
            [*server_args, *identity_args],
            env_dict={
                "VLLM_SERVER_DEV_MODE": "1",
                current_platform.device_control_env_var: ",".join(
                    str(current_platform.device_id_to_physical_device_id(i))
                    for i in range(first_rank, first_rank + local_size)
                ),
            },
        )
        servers.append(server)
        return server

    try:
        with ThreadPoolExecutor(max_workers=len(first_ranks)) as executor:
            clients = list(executor.map(start_client, first_ranks))
        for first_rank, server in zip(first_ranks, clients):
            config = _get_parallel_config(server)
            assert config["data_parallel_size"] == 4
            assert config["data_parallel_rank"] == first_rank
            assert config["data_parallel_size_local"] == local_size
            assert config["_api_process_count"] == 1
            _assert_local_range_serving(server, first_rank, local_size)
            assert not config["data_parallel_external_lb"]
            assert config["data_parallel_hybrid_lb"] is (local_size < 4)
    finally:
        RemoteOpenAIServer.shutdown_many(servers)


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    TP_SIZE != 1 or current_platform.device_count() < 4,
    reason="External Elastic EP local-range resize requires four GPUs with TP_SIZE=1",
)
@pytest.mark.skipif(
    os.getenv("VLLM_USE_RUST_FRONTEND") == "1",
    reason="Native Elastic EP resizing requires the Python frontend",
)
def test_external_lb_elastic_ep_local_range_resize(default_server_args, monkeypatch):
    """One request per owner grows DP2→DP4→DP2 without changing local ownership."""
    monkeypatch.setenv("VLLM_ELASTIC_EP_DRAIN_REQUESTS", "1")
    manager = ExternalLBServerManager(
        ELASTIC_EP_MODEL_NAME,
        dp_size=2,
        api_server_count=1,
        base_server_args=[
            *default_server_args,
            "--enable-expert-parallel",
            "--enable-elastic-ep",
            "--elastic-ep-max-dp-size",
            "4",
            "--enable-eplb",
            "--eplb-config",
            '{"communicator":"pynccl","use_async":false}',
            "--attention-backend",
            "TRITON_MLA",
            "--data-parallel-address",
            "127.0.0.1",
            "--data-parallel-rpc-port",
            str(get_open_port()),
        ],
        auto_bootstrap=True,
        local_size=2,
    )
    with ThreadPoolExecutor(max_workers=2) as executor, manager:
        survivor = manager.servers[0][0]
        _assert_local_range_serving(survivor, 0, 2)
        growth = executor.submit(_send_scale_command, survivor, 4)
        _wait_for_scale_requests_to_start([growth], [survivor])
        joiner, joiner_args = manager.start_rank(2)
        assert "--data-parallel-size" not in joiner_args
        response = growth.result(timeout=600)
        assert response.status_code == 200, response.text

        for first_rank, server in [(0, survivor), (2, joiner)]:
            config = _get_parallel_config(server)
            assert config["data_parallel_size"] == 4
            assert config["data_parallel_rank"] == first_rank
            assert config["data_parallel_size_local"] == 2
            assert config["data_parallel_hybrid_lb"]
            assert not config["data_parallel_external_lb"]
            _assert_local_range_serving(server, first_rank, 2)

        retiring_engines = [
            proc
            for proc in psutil.Process(joiner.proc.pid).children(recursive=True)
            if proc.cmdline() and proc.cmdline()[0].startswith("VLLM::EngineCore_DP")
        ]
        assert len(retiring_engines) == 2
        shrink = [
            executor.submit(_send_scale_command, server, 2)
            for server in (survivor, joiner)
        ]
        for request in shrink:
            response = request.result(timeout=600)
            assert response.status_code == 200, response.text

        _, alive = psutil.wait_procs(retiring_engines, timeout=60)
        assert not alive, [proc.pid for proc in alive]
        assert joiner.proc.poll() is None
        config = _get_parallel_config(survivor)
        assert config["data_parallel_size"] == 2
        assert config["data_parallel_rank"] == 0
        assert config["data_parallel_size_local"] == 2
        assert not config["data_parallel_hybrid_lb"]
        _assert_local_range_serving(survivor, 0, 2)
        response = requests.post(
            joiner.url_for("v1/completions"),
            json={"model": ELASTIC_EP_MODEL_NAME, "prompt": "Hello", "max_tokens": 5},
            timeout=10,
        )
        assert response.status_code == 503
        for server in (survivor, joiner):
            response = requests.post(
                server.url_for("is_scaling_elastic_ep"), timeout=10
            )
            response.raise_for_status()
            assert response.json()["phase"] == "completed"


def test_external_lb_server_info(server_manager):
    servers = server_manager.servers
    api_server_count = server_manager.api_server_count

    for i, (server, _) in enumerate(servers):
        print(f"Testing {i=}")

        # Each request will hit one of the API servers
        # `n_reqs` is set so that there is a good chance each server
        # receives at least one request
        n_reqs = 2 * api_server_count * api_server_count
        parallel_configs = [_get_parallel_config(server) for _ in range(n_reqs)]
        api_process_counts = [c["_api_process_count"] for c in parallel_configs]
        api_process_ranks = [c["_api_process_rank"] for c in parallel_configs]

        assert all(c == api_server_count for c in api_process_counts), (
            api_process_counts
        )
        assert all(0 <= r < api_server_count for r in api_process_ranks), (
            api_process_ranks
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_external_lb_single_completion(
    clients: list[openai.AsyncOpenAI],
    servers: list[tuple[RemoteOpenAIServer, list[str]]],
    model_name: str,
) -> None:
    async def make_request(client: openai.AsyncOpenAI):
        completion = await client.completions.create(
            model=model_name, prompt="Hello, my name is", max_tokens=10, temperature=1.0
        )

        assert completion.id is not None
        assert completion.choices is not None and len(completion.choices) == 1

        choice = completion.choices[0]
        # The exact number of tokens can vary slightly with temperature=1.0,
        # so we check for a reasonable minimum length.
        assert len(choice.text) >= 1
        # Finish reason might not always be 'length' if the model finishes early
        # or due to other reasons, especially with high temperature.
        # So, we'll accept 'length' or 'stop'.
        assert choice.finish_reason in ("length", "stop")

        # Token counts can also vary, so we check they are positive.
        assert completion.usage.completion_tokens > 0
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.total_tokens > 0
        return completion

    # Test single request to each server
    for i, client in enumerate(clients):
        result = await make_request(client)
        assert result is not None
        print(f"Server {i} handled single completion request successfully")

    await asyncio.sleep(0.5)

    # Send requests to all servers in round-robin fashion
    num_requests_per_server = 25  # Total 50 requests across 2 servers
    all_tasks = []

    for i, client in enumerate(clients):
        tasks = [make_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(completion is not None for completion in results)

    await asyncio.sleep(0.5)

    # Second burst of requests
    all_tasks = []
    for i, client in enumerate(clients):
        tasks = [make_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(completion is not None for completion in results)

    _, server_args = servers[0]
    api_server_count = (
        server_args.count("--api-server-count")
        and server_args[server_args.index("--api-server-count") + 1]
        or 1
    )
    print(
        f"Successfully completed external LB test with {len(clients)} servers "
        f"(API server count: {api_server_count})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_external_lb_completion_streaming(
    clients: list[openai.AsyncOpenAI],
    servers: list[tuple[RemoteOpenAIServer, list[str]]],
    model_name: str,
) -> None:
    prompt = "What is an LLM?"

    async def make_streaming_request(client: openai.AsyncOpenAI):
        # Perform a non-streaming request to get the expected full output
        single_completion = await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
        )
        single_output = single_completion.choices[0].text

        # Perform the streaming request
        stream = await client.completions.create(
            model=model_name, prompt=prompt, max_tokens=5, temperature=0.0, stream=True
        )
        chunks: list[str] = []
        finish_reason_count = 0
        last_chunk = None
        async for chunk in stream:
            chunks.append(chunk.choices[0].text)
            if chunk.choices[0].finish_reason is not None:
                finish_reason_count += 1
            last_chunk = chunk  # Keep track of the last chunk

        # finish reason should only return in the last block for OpenAI API
        assert finish_reason_count == 1, "Finish reason should appear exactly once."
        assert last_chunk is not None, "Stream should have yielded at least one chunk."
        assert last_chunk.choices[0].finish_reason == "length", (
            "Finish reason should be 'length'."
        )
        # Check that the combined text matches the non-streamed version.
        assert "".join(chunks) == single_output, (
            "Streamed output should match non-streamed output."
        )
        return True  # Indicate success for this request

    # Test single request to each server
    for i, client in enumerate(clients):
        result = await make_streaming_request(client)
        assert result is not None
        print(f"Server {i} handled single streaming request successfully")

    await asyncio.sleep(0.5)

    # Send streaming requests to all servers in round-robin fashion
    num_requests_per_server = 25  # Total 50 requests across 2 servers
    all_tasks = []

    for i, client in enumerate(clients):
        tasks = [make_streaming_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(results), "Not all streaming requests completed successfully."

    await asyncio.sleep(0.5)

    # Second burst of streaming requests
    all_tasks = []
    for i, client in enumerate(clients):
        tasks = [make_streaming_request(client) for _ in range(num_requests_per_server)]
        all_tasks.extend(tasks)

    results = await asyncio.gather(*all_tasks)
    assert len(results) == num_requests_per_server * len(clients)
    assert all(results), "Not all streaming requests completed successfully."

    _, server_args = servers[0]
    api_server_count = (
        server_args.count("--api-server-count")
        and server_args[server_args.index("--api-server-count") + 1]
        or 1
    )
    print(
        f"Successfully completed external LB streaming test with "
        f"{len(clients)} servers (API server count: {api_server_count})"
    )
