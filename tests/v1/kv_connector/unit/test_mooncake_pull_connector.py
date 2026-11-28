# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Mooncake pull connector.

The transfer engine is the only thing simulated: `FakeCopyEngine` copies bytes
between the very descriptor arrays the connector hands it and keeps one memory
registration table per engine, so a descriptor that points at memory the reader
or its peer never registered fails the test instead of moving wrong bytes.
"""

import asyncio
import contextlib
import ctypes
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    SendBlockMeta,
    TransferRegion,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_pull_connector import (  # noqa: E501
    MooncakePullConnector,
    MooncakePullConnectorWorker,
    MooncakePullRequest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.read_leases import (
    ReadLeases,
)
from vllm.utils.network_utils import get_open_port, make_zmq_path
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec

from .utils import create_vllm_config, make_kv_cache_group

BASE_WORKER = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"
LAYER = "model.layers.0.self_attn"
MAMBA_LAYER = "model.layers.1.linear_attn"
BLOCK_SIZE = 16
HEAD_SIZE = 64
KV_HEADS = 4


class FakeCopyEngine:
    """Mooncake TransferEngine stand-in that copies between registered buffers.

    Every instance owns the memory it registered, and instances find each other
    by the session string a read targets, so a reader can only touch memory its
    peer really registered.
    """

    _engines: dict[str, "FakeCopyEngine"] = {}
    _next_port = 20000

    def __init__(self, *args, **kwargs):
        self.registered: list[tuple[int, int]] = []
        self.reads: list[tuple[str, list[int], list[int], list[int]]] = []
        self.fail_reads = 0
        # A gate blocks one read until the test releases it, and `read_started`
        # says when the engine is inside that read.
        self.read_gates: list[threading.Event | None] = []
        self.read_started: list[threading.Event] = []
        self.port = 0

    @classmethod
    def reset(cls) -> None:
        cls._engines.clear()
        cls._next_port = 20000

    def initialize(self, local_hostname, metadata_server, protocol, device_name) -> int:
        self.port = FakeCopyEngine._next_port
        FakeCopyEngine._next_port += 1
        FakeCopyEngine._engines[f"{local_hostname}:{self.port}"] = self
        return 0

    def get_rpc_port(self) -> int:
        return self.port

    def batch_register_memory(self, buffer_addresses, capacities) -> int:
        self.registered.extend(
            (base, base + capacity)
            for base, capacity in zip(buffer_addresses, capacities, strict=True)
        )
        return 0

    def batch_unregister_memory(self, buffer_addresses) -> int:
        removed = set(buffer_addresses)
        self.registered = [
            (base, limit) for base, limit in self.registered if base not in removed
        ]
        return 0

    @staticmethod
    def _check_registered(
        table: list[tuple[int, int]], addr: int, length: int, owner: str
    ) -> None:
        end = addr + length
        assert any(base <= addr and end <= limit for base, limit in table), (
            f"descriptor [{addr}, {end}) is not inside {owner} registered "
            f"memory {table}"
        )

    def batch_transfer_sync_read(
        self, target_hostname, buffers, peer_buffer_addresses, lengths
    ) -> int:
        index = len(self.reads)
        self.reads.append(
            (target_hostname, list(buffers), list(peer_buffer_addresses), list(lengths))
        )
        peer = FakeCopyEngine._engines.get(target_hostname)
        assert peer is not None, f"no fake engine is listening at {target_hostname}"
        started = threading.Event()
        started.set()
        self.read_started.append(started)
        if index < len(self.read_gates) and self.read_gates[index] is not None:
            self.read_gates[index].wait()
        if self.fail_reads:
            self.fail_reads -= 1
            return -1
        for local, remote, length in zip(
            buffers, peer_buffer_addresses, lengths, strict=True
        ):
            self._check_registered(peer.registered, remote, length, "the peer's")
            self._check_registered(self.registered, local, length, "the reader's")
            ctypes.memmove(local, remote, length)
        return 0


@pytest.fixture(autouse=True)
def _reset_fake_engines():
    FakeCopyEngine.reset()
    yield
    FakeCopyEngine.reset()


@pytest.fixture(autouse=True)
def _cpu_device():
    """Keep device selection valid for worker threads on CPU-only hosts."""
    with (
        patch("torch.accelerator.current_device_index", return_value=0),
        patch(f"{BASE_WORKER}.current_platform.set_device"),
    ):
        yield


@contextlib.contextmanager
def patch_pull_worker_dependencies(
    tp_size: int = 1, tp_ranks: list[int] | None = None
) -> Iterator[None]:
    """Patch everything a pull worker needs from the distributed runtime."""
    rank_patch = (
        patch(f"{BASE_WORKER}.get_tensor_model_parallel_rank", side_effect=tp_ranks)
        if tp_ranks is not None
        else patch(f"{BASE_WORKER}.get_tensor_model_parallel_rank", return_value=0)
    )
    with (
        patch(f"{BASE_WORKER}.TransferEngine", FakeCopyEngine),
        patch(f"{BASE_WORKER}.get_ip", return_value="127.0.0.1"),
        patch(f"{BASE_WORKER}.get_current_attn_backends", return_value=[]),
        patch("torch.accelerator.current_device_index", return_value=0),
        patch(f"{BASE_WORKER}.current_platform.set_device"),
        patch(
            "vllm.v1.hisparse.runtime.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        rank_patch,
        patch(
            f"{BASE_WORKER}.get_tensor_model_parallel_world_size",
            return_value=tp_size,
        ),
        patch(f"{BASE_WORKER}.get_pp_group") as mock_pp,
        patch(f"{BASE_WORKER}.should_launch_bootstrap_server", return_value=False),
    ):
        mock_pp.return_value.rank_in_group = 0
        yield


@pytest.fixture
def bootstrap_server():
    """Bootstrap server the prefiller registers with and the decoder queries."""
    port = get_open_port()
    server = MooncakeBootstrapServer("127.0.0.1", port)
    server.start()
    yield server
    server.shutdown()


def make_kv_cache_config(
    block_size: int = BLOCK_SIZE,
    num_kv_heads: int = KV_HEADS,
    page_size_padded: int | None = None,
    host_resident: bool = False,
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            make_kv_cache_group(
                [LAYER],
                AttentionSpec(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=HEAD_SIZE,
                    dtype=torch.float16,
                    page_size_padded=page_size_padded,
                ),
                host_resident=host_resident,
            )
        ],
    )


def page_bytes(num_kv_heads: int = KV_HEADS, block_size: int = BLOCK_SIZE) -> int:
    """Bytes of one KV page: K and V for every KV head this rank holds."""
    return block_size * num_kv_heads * HEAD_SIZE * torch.float16.itemsize * 2


def make_worker(
    vllm_config,
    kv_cache_config: KVCacheConfig,
) -> MooncakePullConnectorWorker:
    with set_current_vllm_config(vllm_config):
        connector = MooncakePullConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )
    worker = connector.connector_worker
    assert worker is not None
    return worker


def make_kv_tensor(num_blocks: int, block_bytes: int) -> torch.Tensor:
    """A KV cache whose block stride is exactly one page."""
    return torch.zeros((num_blocks, block_bytes), dtype=torch.int8)


def run_on(loop: asyncio.AbstractEventLoop, coro) -> asyncio.Future:
    return asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))


async def wait_for(predicate, timeout: float = 10.0):
    """Poll an async predicate until it is true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = await predicate()
        if result:
            return result
        await asyncio.sleep(0.02)
    raise AssertionError("timed out waiting for the transfer to finish")


async def fetch_recved(worker: MooncakePullConnectorWorker) -> set[str]:
    return await run_on(worker.receiver_loop, worker.fetch_finished_recving_reqs())


async def fetch_sent(worker: MooncakePullConnectorWorker) -> set[str]:
    return await run_on(worker.sender_loop, worker.fetch_finished_sending_reqs())


async def wait_read_start(engine: FakeCopyEngine, index: int) -> None:
    await wait_for(lambda: _read_started(engine, index))


async def _read_started(engine: FakeCopyEngine, index: int) -> bool:
    return len(engine.read_started) > index


def register_ready_transfer(
    worker: MooncakePullConnectorWorker,
    transfer_id: str,
    p_req_id: str,
    block_ids: list[list[int]],
    ready: bool = True,
) -> SendBlockMeta:
    """Stand in for the prefiller's scheduler having finished a request."""
    send_meta = SendBlockMeta(
        p_req_id=p_req_id,
        transfer_id=transfer_id,
        local_block_ids=block_ids,
        ready=asyncio.Event(),
    )
    if ready:
        send_meta.ready.set()
    worker.reqs_need_send[transfer_id] = send_meta
    return send_meta


def send_metadata(
    p_req_id: str, transfer_id: str, block_ids: list[list[int]] | None = None
) -> MooncakeConnectorMetadata:
    """Scheduler metadata for a prefiller request that just finished."""
    metadata = MooncakeConnectorMetadata()
    metadata.reqs_to_send[p_req_id] = (transfer_id, block_ids or [])
    return metadata


async def mark_producer_ready(send_meta: SendBlockMeta) -> None:
    """Let the prefiller's request finish, from the loop that owns the event."""
    send_meta.expire_time = (
        time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
    )
    send_meta.ready.set()


def pull_meta(
    d_req_id: str,
    transfer_id: str,
    block_ids: list[list[int]],
    remote_engine_id: str,
    port: int,
) -> PullReqMeta:
    return PullReqMeta(
        d_req_id=d_req_id,
        transfer_id=transfer_id,
        local_block_ids=block_ids,
        remote_engine_id=remote_engine_id,
        remote_bootstrap_addr=f"http://127.0.0.1:{port}",
    )


def producer_layout(block_len: int = 4096, base_addr: int = 0x1000):
    return MooncakeXferMetadata(
        remote_hostname="producer-host",
        remote_port=12345,
        remote_tp_size=1,
        remote_tp_rank=0,
        remote_pp_size=1,
        physical_blocks_per_logical=1,
        producer_cache_replicated=False,
        layout_id=1,
        req_blocks={},
        kv_caches_base_addr=[base_addr],
        block_lens=[block_len],
        kv_block_lens=[block_len],
        registered_layer_names=[LAYER],
        registered_layer_indices=[0],
        registered_group_indices=[0],
    )


def region(
    layer_name: str,
    group_index: int,
    base_addr: int,
    block_len: int,
    layer_index: int = 0,
) -> TransferRegion:
    return TransferRegion(
        layer_name=layer_name,
        layer_index=layer_index,
        base_addr=base_addr,
        block_len=block_len,
        kv_block_len=block_len,
        group_index=group_index,
    )


@contextlib.contextmanager
def decoder_worker(
    kv_cache_config: KVCacheConfig | None = None,
) -> Iterator[MooncakePullConnectorWorker]:
    vllm_config = create_vllm_config(
        kv_connector="MooncakePullConnector", kv_role="kv_consumer"
    )
    with patch_pull_worker_dependencies():
        worker = make_worker(vllm_config, kv_cache_config or make_kv_cache_config())
        worker.kv_caches_base_addr = [0x2000]
        worker.block_len_per_layer = [4096]
        worker.kv_block_len_per_layer = [4096]
        worker.registered_layer_names = [LAYER]
        worker.registered_layer_indices = [0]
        worker.registered_group_indices = [0]
        try:
            yield worker
        finally:
            worker.shutdown()


@dataclass
class Pair:
    """Producers and a decoder over fake engines, caches registered."""

    producers: list[MooncakePullConnectorWorker]
    decoder: MooncakePullConnectorWorker
    producer_kvs: list[torch.Tensor]
    decoder_kv: torch.Tensor
    producer_block_bytes: int
    decoder_block_bytes: int
    port: int

    @property
    def producer(self) -> MooncakePullConnectorWorker:
        assert len(self.producers) == 1
        return self.producers[0]

    @property
    def engine(self) -> FakeCopyEngine:
        return self.decoder.transfer_backend.engine

    def fill(self, start: int = 0, stop: int = 8) -> None:
        for block_id in range(start, stop):
            self.producer_kvs[0][block_id].fill_(block_id + 1)

    def ready(
        self,
        transfer_id: str,
        p_req_id: str,
        block_ids: list[list[int]],
        *,
        ready: bool = True,
        producer_index: int = 0,
    ) -> SendBlockMeta:
        return register_ready_transfer(
            self.producers[producer_index],
            transfer_id,
            p_req_id,
            block_ids,
            ready=ready,
        )

    def read(
        self, requests: dict[str, tuple[str, list[list[int]]]]
    ) -> dict[str, PullReqMeta]:
        """Start a read of the given local block ids per request."""
        metas = {
            d_req_id: pull_meta(
                d_req_id,
                transfer_id,
                block_ids,
                self.producers[0].engine_id,
                self.port,
            )
            for d_req_id, (transfer_id, block_ids) in requests.items()
        }
        asyncio.run_coroutine_threadsafe(
            self.decoder._start_load_kv({self.producers[0].engine_id: metas}),
            self.decoder.receiver_loop,
        )
        return metas


@contextlib.contextmanager
def pair_factory(
    *,
    bootstrap_port: int,
    producer_block_size: int = BLOCK_SIZE,
    decoder_block_size: int = BLOCK_SIZE,
    producer_kv_heads: int = KV_HEADS,
    decoder_kv_heads: int = KV_HEADS,
    producer_page_size_padded: int | None = None,
    producer_tp_size: int = 1,
    producer_tp_ranks: list[int] | None = None,
    num_producers: int = 1,
    num_producer_blocks: int = 8,
    num_decoder_blocks: int = 32,
) -> Iterator[Pair]:
    """Build a registered producer/decoder pair and tear every worker down."""
    producer_page = page_bytes(producer_kv_heads, producer_block_size)
    decoder_page = page_bytes(decoder_kv_heads, decoder_block_size)
    producer_storage = producer_page
    if producer_page_size_padded is not None:
        producer_storage = producer_page_size_padded

    producer_config = create_vllm_config(
        kv_connector="MooncakePullConnector",
        kv_role="kv_producer",
        block_size=producer_block_size,
    )
    decoder_config = create_vllm_config(
        kv_connector="MooncakePullConnector",
        kv_role="kv_consumer",
        block_size=decoder_block_size,
    )
    if num_producers > 1:
        producer_config.kv_transfer_config.engine_id = "p-engine"

    producer_cache_config = make_kv_cache_config(
        block_size=producer_block_size,
        num_kv_heads=producer_kv_heads,
        page_size_padded=producer_page_size_padded,
    )
    decoder_cache_config = make_kv_cache_config(
        block_size=decoder_block_size,
        num_kv_heads=decoder_kv_heads,
    )

    with patch_pull_worker_dependencies(producer_tp_size, producer_tp_ranks):
        producers = [
            make_worker(producer_config, producer_cache_config)
            for _ in range(num_producers)
        ]
        producer_kvs = [
            make_kv_tensor(num_producer_blocks, producer_storage)
            for _ in range(num_producers)
        ]
        for producer, producer_kv in zip(producers, producer_kvs, strict=True):
            producer.register_kv_caches({LAYER: producer_kv})

    with patch_pull_worker_dependencies():
        decoder = make_worker(decoder_config, decoder_cache_config)
        decoder_kv = make_kv_tensor(num_decoder_blocks, decoder_page)
        decoder.register_kv_caches({LAYER: decoder_kv})

    try:
        yield Pair(
            producers,
            decoder,
            producer_kvs,
            decoder_kv,
            producer_page,
            decoder_page,
            bootstrap_port,
        )
    finally:
        # A failing test may leave a read parked on its gate; release it so the
        # reader thread can exit instead of keeping the process alive.
        for gate in decoder.transfer_backend.engine.read_gates:
            if gate is not None:
                gate.set()
        with patch_pull_worker_dependencies():
            decoder.shutdown()
        with patch_pull_worker_dependencies(producer_tp_size, producer_tp_ranks):
            for producer in producers:
                producer.shutdown()


@pytest.fixture
def pair(monkeypatch, bootstrap_server, request):
    """A registered producer/decoder pair, torn down with the test."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv(
        "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", str(getattr(request, "param", 5))
    )
    with pair_factory(bootstrap_port=bootstrap_server.port) as workers:
        yield workers


@pytest.mark.parametrize(
    "resolved_ids, expected_remote, expected_lengths",
    [
        ([[2, 3]], [0x1000 + 2 * 4096], [2 * 4096]),
        ([[1, 2, 3, 4]], [0x1000 + 3 * 4096], [2 * 4096]),
    ],
    ids=["coalesced", "trimmed-tail"],
)
def test_plan_reads_addresses_producer_and_local_blocks(
    resolved_ids, expected_remote, expected_lengths
):
    block_len = 4096
    pull_metas = {"d-req": pull_meta("d-req", "xfer", [[10, 11]], "p-engine", 1)}
    consumer = [region(LAYER, 0, 0x2000, block_len)]
    producer = [region(LAYER, 0, 0x1000, block_len)]

    with decoder_worker() as worker:
        local_ptrs, remote_ptrs, lengths, err_reqs, err_msg = worker._plan_reads(
            producer_layout(block_len),
            pull_metas,
            {"d-req": ("xfer", resolved_ids)},
            consumer,
            producer,
        )

    assert (err_reqs, err_msg) == ([], None)
    np.testing.assert_array_equal(remote_ptrs, expected_remote)
    np.testing.assert_array_equal(local_ptrs, [0x2000 + 10 * block_len])
    np.testing.assert_array_equal(lengths, expected_lengths)


@pytest.mark.parametrize(
    "local_ids, producer_ids, expected_msg",
    [
        ([[10, 11, 12]], [[2, 3]], "P num blocks less than D"),
        ([[10]], [[2], [3]], "KV group count mismatch"),
    ],
    ids=["too-few", "group-count"],
)
def test_plan_reads_reports_bad_block_counts(local_ids, producer_ids, expected_msg):
    pull_metas = {"d-req": pull_meta("d-req", "xfer", local_ids, "p-engine", 1)}
    consumer = [region(LAYER, 0, 0x2000, 4096)]
    producer = [region(LAYER, 0, 0x1000, 4096)]

    with decoder_worker() as worker:
        _, _, _, err_reqs, err_msg = worker._plan_reads(
            producer_layout(),
            pull_metas,
            {"d-req": ("xfer", producer_ids)},
            consumer,
            producer,
        )

    assert err_reqs == ["d-req"]
    assert err_msg == expected_msg


def test_plan_reads_drops_mamba_placeholders_in_multiple_groups():
    """A Mamba null placeholder is skipped without shifting the other group."""
    attention_bytes = page_bytes(1, BLOCK_SIZE)
    mamba_bytes = 512
    consumer = [
        region(LAYER, 0, 0x2000, attention_bytes),
        region(MAMBA_LAYER, 1, 0x4000, mamba_bytes, layer_index=1),
    ]
    producer = [
        region(LAYER, 0, 0x1000, attention_bytes),
        region(MAMBA_LAYER, 1, 0x3000, mamba_bytes, layer_index=1),
    ]
    pull_metas = {"d-req": pull_meta("d-req", "xfer", [[10, 11], [6]], "p-engine", 1)}
    resolved = {
        "d-req": ("xfer", [[2, 3], [NULL_BLOCK_ID, 5]]),
    }

    with decoder_worker(make_hybrid_gdn_kv_cache_config(BLOCK_SIZE)) as worker:
        local_ptrs, remote_ptrs, lengths, err_reqs, err_msg = worker._plan_reads(
            producer_layout(), pull_metas, resolved, consumer, producer
        )

    assert (err_reqs, err_msg) == ([], None)
    np.testing.assert_array_equal(
        remote_ptrs,
        np.array(
            [0x1000 + 2 * attention_bytes, 0x3000 + 5 * mamba_bytes],
            dtype=np.uint64,
        ),
    )
    np.testing.assert_array_equal(
        local_ptrs,
        np.array(
            [0x2000 + 10 * attention_bytes, 0x4000 + 6 * mamba_bytes],
            dtype=np.uint64,
        ),
    )
    np.testing.assert_array_equal(
        lengths, np.array([2 * attention_bytes, mamba_bytes], dtype=np.uint64)
    )


def make_hybrid_gdn_kv_cache_config(block_size: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            make_kv_cache_group(
                [LAYER],
                AttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            make_kv_cache_group(
                [MAMBA_LAYER],
                MambaSpec(
                    block_size=block_size,
                    shapes=((6, 3), (1, 2, 2)),
                    dtypes=(torch.float16, torch.float16),
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                ),
            ),
        ],
    )


@pytest.mark.parametrize("tp_rank", [0, 1], ids=["leader", "non-leader"])
def test_hisparse_pull_reads_only_the_host_groups_this_rank_owns(tp_rank):
    """A non-leader rank never hands the shared host group to a read."""
    host_group = make_kv_cache_group(
        [LAYER],
        AttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=torch.float16,
        ),
        host_resident=True,
    )
    device_group = make_kv_cache_group(
        [MAMBA_LAYER],
        AttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=torch.float16,
        ),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[host_group, device_group],
        hisparse_shared_host_pool=True,
    )
    vllm_config = create_vllm_config(
        kv_connector="MooncakePullConnector", kv_role="kv_consumer"
    )
    with (
        patch_pull_worker_dependencies(tp_size=2, tp_ranks=[tp_rank]),
        patch(f"{BASE_WORKER}.get_hisparse_host_leader_rank", return_value=0),
        set_current_vllm_config(vllm_config),
    ):
        worker = make_worker(vllm_config, kv_cache_config)
        worker.tp_rank = tp_rank
        try:
            block_ids = [[1, 2], [3, 4]]
            filtered = worker._filter_host_block_ids(block_ids)
            if tp_rank == 0:
                assert filtered == block_ids
            else:
                assert filtered == [[], [3, 4]]

            req_meta = {"d-req": pull_meta("d-req", "xfer", block_ids, "p-engine", 1)}
            meta = worker._build_pull_metadata(req_meta, include_layout=False)
            assert meta.req_blocks["d-req"][1] == filtered
        finally:
            worker.shutdown()


@pytest.mark.asyncio
async def test_pull_reads_and_reuses_a_validated_layout(pair):
    """The decoder reads bytes, caches the layout, and drops it on a restart."""
    producer = pair.producer
    decoder = pair.decoder
    seen: list[MooncakePullRequest] = []
    original = producer._handle_pull_request

    async def spy(identity, sock, request):
        seen.append(request)
        await original(identity, sock, request)

    producer._handle_pull_request = spy  # type: ignore[method-assign]

    pair.fill()
    pair.ready("xfer-1", "p-req-1", [[2, 3]])
    pair.read({"d-req-1": ("xfer-1", [[10, 11]])})
    assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-1"}
    assert pair.decoder_kv[10].tolist() == [3] * pair.decoder_block_bytes
    assert pair.decoder_kv[11].tolist() == [4] * pair.decoder_block_bytes

    worker_addr = make_zmq_path("tcp", producer.hostname, producer.side_channel_port)
    layout = decoder._peer_layouts[worker_addr]
    assert layout.layout_id

    pair.ready("xfer-2", "p-req-2", [[4]])
    pair.read({"d-req-2": ("xfer-2", [[12]])})
    assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-2"}
    assert pair.decoder_kv[12].tolist() == [5] * pair.decoder_block_bytes

    # The prefiller comes back with another layout, so the cached one is stale.
    producer._kv_layout_id += 1
    pair.ready("xfer-3", "p-req-3", [[5]])
    pair.read({"d-req-3": ("xfer-3", [[13]])})
    assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-3"}
    assert decoder.get_block_ids_with_load_errors() == {13}
    assert worker_addr not in decoder._peer_layouts

    pair.ready("xfer-4", "p-req-4", [[6]])
    pair.read({"d-req-4": ("xfer-4", [[14]])})
    assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-4"}
    assert pair.decoder_kv[14].tolist() == [7] * pair.decoder_block_bytes
    assert [request.want_regions for request in seen] == [True, False, False, True]


@pytest.mark.asyncio
async def test_pull_releases_the_producer_on_a_local_prefix_hit(pair):
    """A full local hit releases the producer without a duplicate done."""
    producer = pair.producer
    decoder = pair.decoder

    # The producer's scheduler has not registered the transfer yet, so the
    # reader's completion has to be accounted for once it does.
    pair.read({"d-req-hit": ("xfer-hit", [])})
    assert await wait_for(
        lambda: run_on(producer.sender_loop, has_read_done(producer, "xfer-hit"))
    )
    assert await fetch_recved(decoder) == set()
    assert pair.engine.reads == []

    await run_on(
        producer.sender_loop,
        producer.record_send_reqs(send_metadata("p-req-hit", "xfer-hit", [[2]])),
    )
    assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-hit"}
    assert "xfer-hit" not in producer.reqs_need_send
    assert pair.decoder_kv[2].sum().item() == 0


@pytest.mark.parametrize("pair", [1], indirect=True)
@pytest.mark.asyncio
async def test_a_slow_second_wave_is_kept_alive_by_lease_renewal(pair):
    """A later wave is not expired while its read is still running."""
    producer = pair.producer
    pair.producer_kvs[0][2].fill_(7)
    pair.producer_kvs[0][3].fill_(9)
    gates = [threading.Event(), threading.Event()]
    pair.engine.read_gates = gates
    send_metas = [
        pair.ready("xfer-1", "p-req-1", [[2]], ready=False),
        pair.ready("xfer-2", "p-req-2", [[3]], ready=False),
    ]
    pair.read({"d-req-1": ("xfer-1", [[10]]), "d-req-2": ("xfer-2", [[11]])})
    await run_on(producer.sender_loop, mark_producer_ready(send_metas[0]))
    await wait_read_start(pair.engine, 0)
    # The second response arrives while the first read is blocked.
    await run_on(producer.sender_loop, mark_producer_ready(send_metas[1]))
    gates[0].set()
    await wait_read_start(pair.engine, 1)

    # Past the abort timeout, only a renewal keeps wave 2's blocks alive.
    await asyncio.sleep(1.2)
    assert "p-req-2" not in await fetch_sent(producer)
    assert "xfer-2" in producer.reqs_need_send

    gates[1].set()
    assert await wait_for(lambda: fetch_recved(pair.decoder)) == {
        "d-req-1",
        "d-req-2",
    }
    assert pair.decoder_kv[10].tolist() == [7] * pair.decoder_block_bytes
    assert pair.decoder_kv[11].tolist() == [9] * pair.decoder_block_bytes
    assert "p-req-2" in await wait_for(lambda: fetch_sent(producer))


@pytest.mark.asyncio
async def test_cancelled_read_reports_only_after_the_engine_returns(
    monkeypatch, bootstrap_server
):
    """A cancelled receiver cannot free the blocks the engine is still writing."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "10")
    with pair_factory(bootstrap_port=bootstrap_server.port) as pair:
        producer = pair.producer
        decoder = pair.decoder
        gate = threading.Event()
        decoder.transfer_backend.engine.read_gates = [gate]
        producer_kv = pair.producer_kvs[0]
        producer_kv[4].fill_(5)
        register_ready_transfer(producer, "xfer-cancel", "p-req-cancel", [[4]])
        metas = {
            "d-req-cancel": pull_meta(
                "d-req-cancel",
                "xfer-cancel",
                [[10]],
                producer.engine_id,
                bootstrap_server.port,
            )
        }
        worker_addr = make_zmq_path(
            "tcp", producer.hostname, producer.side_channel_port
        )

        async def read_then_cancel():
            task = asyncio.create_task(
                decoder.receive_kv_from_single_worker(worker_addr, metas)
            )
            await wait_read_start(decoder.transfer_backend.engine, 0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        await run_on(decoder.receiver_loop, read_then_cancel())

        # The engine thread runs on after the cancellation, so reporting the
        # request now would let the scheduler hand block 10 to another request.
        assert await fetch_recved(decoder) == set()
        assert producer._leases.holders("xfer-cancel")
        gate.set()

        assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-cancel"}
        assert decoder.get_block_ids_with_load_errors() == {10}
        assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-cancel"}


@pytest.mark.parametrize("pair", [30], indirect=True)
@pytest.mark.asyncio
async def test_abort_before_pull_fails_at_once_without_a_placeholder(pair):
    """An aborted transfer has a tombstone, so a later pull does not wait."""
    producer = pair.producer
    metadata = MooncakeConnectorMetadata()
    metadata.reqs_not_processed = {"xfer-gone"}
    await run_on(producer.sender_loop, producer.record_send_reqs(metadata))
    assert "xfer-gone" in producer._expired_transfer_ids

    pair.read({"d-req-gone": ("xfer-gone", [[10]])})
    # The abort timeout is 30 s, so only the tombstone can answer this fast.
    assert await wait_for(lambda: fetch_recved(pair.decoder), timeout=3.0) == {
        "d-req-gone"
    }
    assert pair.decoder.get_block_ids_with_load_errors() == {10}
    assert "xfer-gone" not in producer.reqs_need_send


async def has_read_done(producer: MooncakePullConnectorWorker, transfer_id: str):
    return producer._leases.reported(transfer_id) > 0


@pytest.mark.parametrize(
    "factory_kwargs,hands_out_blocks",
    [
        pytest.param(
            {"producer_block_size": BLOCK_SIZE, "decoder_block_size": BLOCK_SIZE * 2},
            True,
            id="block-size",
        ),
        pytest.param(
            {
                "producer_kv_heads": 2,
                "decoder_kv_heads": 2,
                "producer_page_size_padded": 2 * 16 * 64 * 2 * 2 + 64,
                "producer_tp_size": 2,
                "producer_tp_ranks": [0],
            },
            False,
            id="padded-heads",
        ),
    ],
)
@pytest.mark.asyncio
async def test_decoder_rejects_an_unaddressable_layout(
    monkeypatch, bootstrap_server, factory_kwargs, hands_out_blocks
):
    """A layout whose regions cannot describe this cache fails the request."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    with pair_factory(bootstrap_port=bootstrap_server.port, **factory_kwargs) as pair:
        producer = pair.producer
        decoder = pair.decoder
        register_ready_transfer(producer, "xfer-size", "p-req-size", [[2]])
        pair.read({"d-req-size": ("xfer-size", [[10]])})

        assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-size"}
        assert decoder.get_block_ids_with_load_errors() == {10}
        assert pair.engine.reads == []
        assert not decoder._peer_layouts
        if hands_out_blocks:
            # Rejecting the layout has to release the blocks it handed out.
            assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-size"}
            assert "xfer-size" not in producer.reqs_need_send
            assert not producer._leases.holders("xfer-size")


def test_read_leases_do_not_reopen_reported_readers():
    leases = ReadLeases()
    reader = ("d-engine", 0, 0)
    assert leases.hand_out("xfer", reader)
    leases.report("xfer", reader)
    assert not leases.hand_out("xfer", reader)
    # A transfer known only from its report still needs forgetting.
    assert "xfer" in set(leases)
    leases.forget("xfer")
    assert "xfer" not in set(leases)


@pytest.mark.asyncio
async def test_finish_without_resolving_a_request_fails_it(pair):
    producer = pair.producer
    decoder = pair.decoder

    async def finish_early(identity, sock, request):
        response = MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH,
            layout_id=producer._kv_layout_id,
            metadata=producer._build_xfer_metadata({}, include_regions=True),
        )
        await sock.send_multipart((identity, producer._encoder.encode(response)))

    producer._handle_pull_request = finish_early  # type: ignore[method-assign]
    pair.ready("xfer-missing", "p-req-missing", [[2]])
    pair.read({"d-req-missing": ("xfer-missing", [[10]])})

    assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-missing"}
    assert decoder.get_block_ids_with_load_errors() == {10}
