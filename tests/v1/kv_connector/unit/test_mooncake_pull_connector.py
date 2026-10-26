# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Mooncake pull connector.

The transfer engine is the only thing simulated: `FakeCopyEngine` copies bytes
between the very descriptor lists the connector hands it and keeps the memory
registration table, so a descriptor that points outside registered memory fails
the test instead of silently moving the wrong bytes.
"""

import asyncio
import contextlib
import ctypes
import time
from dataclasses import dataclass
from functools import partial
from unittest.mock import patch

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    MooncakeXferMetadata,
    PullReqMeta,
    SendBlockMeta,
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
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

BASE_WORKER = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"
LAYER = "model.layers.0.self_attn"
BLOCK_SIZE = 16
HEAD_SIZE = 64
KV_HEADS = 4

# Memory registered with any engine in this test module, which is what the RDMA
# fabric knows about, in contrast to one engine's own registration.
REGISTERED_MEMORY: list[tuple[int, int]] = []


class FakeCopyEngine:
    """Mooncake TransferEngine stand-in that copies between registered buffers."""

    def __init__(self, *args, **kwargs):
        self.reads: list[tuple[str, list[int], list[int], list[int]]] = []
        self.fail_reads = 0
        # Seconds one read takes, so a test can outlive the abort timeout.
        self.read_delay = 0.0

    def initialize(self, local_hostname, metadata_server, protocol, device_name) -> int:
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def batch_register_memory(self, buffer_addresses, capacities) -> int:
        REGISTERED_MEMORY.extend(
            (base, base + capacity)
            for base, capacity in zip(buffer_addresses, capacities, strict=True)
        )
        return 0

    def _check_registered(self, addr: int, length: int) -> None:
        end = addr + length
        assert any(
            base <= addr and end <= limit for base, limit in REGISTERED_MEMORY
        ), (
            f"descriptor [{addr}, {end}) is not inside the registered memory "
            f"{REGISTERED_MEMORY}"
        )

    def batch_transfer_sync_read(
        self, target_hostname, buffers, peer_buffer_addresses, lengths
    ) -> int:
        self.reads.append(
            (target_hostname, list(buffers), list(peer_buffer_addresses), list(lengths))
        )
        if self.read_delay:
            time.sleep(self.read_delay)
        if self.fail_reads:
            self.fail_reads -= 1
            return -1
        for local, remote, length in zip(
            buffers, peer_buffer_addresses, lengths, strict=True
        ):
            self._check_registered(remote, length)
            self._check_registered(local, length)
            ctypes.memmove(local, remote, length)
        return 0


@pytest.fixture(autouse=True)
def _clear_registered_memory():
    REGISTERED_MEMORY.clear()
    yield
    REGISTERED_MEMORY.clear()


@contextlib.contextmanager
def patch_pull_worker_dependencies(tp_size: int = 1, tp_ranks: list[int] | None = None):
    """Patch everything a pull worker needs from the distributed runtime."""
    rank_patch = (
        patch(f"{BASE_WORKER}.get_tensor_model_parallel_rank", side_effect=tp_ranks)
        if tp_ranks is not None
        else patch(f"{BASE_WORKER}.get_tensor_model_parallel_rank", return_value=0)
    )
    with (
        patch(f"{BASE_WORKER}.TransferEngine", FakeCopyEngine),
        patch(f"{BASE_WORKER}.get_ip", return_value="127.0.0.1"),
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
):
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [LAYER],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=HEAD_SIZE,
                    dtype=torch.float16,
                    page_size_padded=page_size_padded,
                ),
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


def make_decoder_worker(
    base_addr: int = 0x2000, block_len: int = 4096
) -> MooncakePullConnectorWorker:
    """A decoder worker with hand-written region metadata, for planning tests."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakePullConnector", kv_role="kv_consumer"
    )
    with patch_pull_worker_dependencies():
        worker = make_worker(vllm_config, make_kv_cache_config())
    worker.kv_caches_base_addr = [base_addr]
    worker.block_len_per_layer = [block_len]
    worker.kv_block_len_per_layer = [block_len]
    worker.registered_layer_names = [LAYER]
    worker.registered_layer_indices = [0]
    worker.registered_group_indices = [0]
    return worker


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


@dataclass
class Pair:
    """A producer and a decoder over one fake engine, caches registered."""

    producer: MooncakePullConnectorWorker
    decoder: MooncakePullConnectorWorker
    producer_kv: torch.Tensor
    decoder_kv: torch.Tensor
    block_bytes: int
    port: int

    def fill(self, start: int = 0, stop: int = 8) -> None:
        for block_id in range(start, stop):
            self.producer_kv[block_id].fill_(block_id + 1)

    def ready(
        self,
        transfer_id: str,
        p_req_id: str,
        block_ids: list[list[int]],
        *,
        ready: bool = True,
    ) -> SendBlockMeta:
        return register_ready_transfer(
            self.producer, transfer_id, p_req_id, block_ids, ready=ready
        )

    def read(self, requests: dict[str, tuple[str, list[list[int]]]]) -> None:
        """Start a read of the given local block ids per request."""
        metas = {
            d_req_id: pull_meta(
                d_req_id, transfer_id, block_ids, self.producer.engine_id, self.port
            )
            for d_req_id, (transfer_id, block_ids) in requests.items()
        }
        asyncio.run_coroutine_threadsafe(
            self.decoder._start_load_kv({self.producer.engine_id: metas}),
            self.decoder.receiver_loop,
        )


@pytest.fixture
def pair(monkeypatch, bootstrap_server, request):
    """A registered producer/decoder pair, torn down with the test.

    Parametrize indirectly with the abort timeout in seconds when a test needs
    a different one.
    """
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv(
        "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", str(getattr(request, "param", 5))
    )
    block_bytes = page_bytes()
    with patch_pull_worker_dependencies():
        producer = make_worker(
            create_vllm_config(
                kv_connector="MooncakePullConnector", kv_role="kv_producer"
            ),
            make_kv_cache_config(),
        )
        decoder = make_worker(
            create_vllm_config(
                kv_connector="MooncakePullConnector", kv_role="kv_consumer"
            ),
            make_kv_cache_config(),
        )
        try:
            producer_kv = make_kv_tensor(8, block_bytes)
            producer.register_kv_caches({LAYER: producer_kv})
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})
            yield Pair(
                producer,
                decoder,
                producer_kv,
                decoder_kv,
                block_bytes,
                bootstrap_server.port,
            )
        finally:
            decoder.shutdown()
            producer.shutdown()


# ---------------------------------------------------------------------------
#  Read planning
# ---------------------------------------------------------------------------
def test_plan_reads_points_at_producer_blocks_and_local_blocks():
    """A read takes the producer's bytes into this worker's own blocks."""
    worker = make_decoder_worker()
    block_len = 4096
    pull_metas = {"d-req": pull_meta("d-req", "xfer", [[10, 11]], "p-engine", 1)}
    resolved = {"d-req": ("xfer", [[2, 3]])}

    local_ptrs, remote_ptrs, lengths, err_reqs, err_msg = worker._plan_reads(
        producer_layout(block_len), pull_metas, resolved
    )

    assert (err_reqs, err_msg) == ([], None)
    # Homogeneous TP: the contiguous pair is coalesced into one descriptor.
    assert remote_ptrs == [0x1000 + 2 * block_len]
    assert local_ptrs == [0x2000 + 10 * block_len]
    assert lengths == [2 * block_len]


def test_plan_reads_trims_the_producer_tail():
    """A partial local prefix hit reads only the newest producer blocks."""
    worker = make_decoder_worker()
    block_len = 4096
    pull_metas = {"d-req": pull_meta("d-req", "xfer", [[10, 11]], "p-engine", 1)}
    resolved = {"d-req": ("xfer", [[1, 2, 3, 4]])}

    _, remote_ptrs, lengths, err_reqs, err_msg = worker._plan_reads(
        producer_layout(block_len), pull_metas, resolved
    )

    assert (err_reqs, err_msg) == ([], None)
    assert remote_ptrs == [0x1000 + 3 * block_len]
    assert lengths == [2 * block_len]


def test_plan_reads_reports_a_producer_with_too_few_blocks():
    """The decoder cannot read more blocks than the producer holds."""
    worker = make_decoder_worker()
    pull_metas = {"d-req": pull_meta("d-req", "xfer", [[10, 11, 12]], "p-engine", 1)}
    resolved = {"d-req": ("xfer", [[2, 3]])}

    local_ptrs, remote_ptrs, lengths, err_reqs, err_msg = worker._plan_reads(
        producer_layout(), pull_metas, resolved
    )

    assert (local_ptrs, remote_ptrs, lengths) == ([], [], [])
    assert err_reqs == ["d-req"]
    assert err_msg == "P num blocks less than D"


def test_plan_reads_rejects_mismatched_group_counts():
    worker = make_decoder_worker()
    pull_metas = {"d-req": pull_meta("d-req", "xfer", [[10]], "p-engine", 1)}
    resolved = {"d-req": ("xfer", [[2], [3]])}

    _, _, _, err_reqs, err_msg = worker._plan_reads(
        producer_layout(), pull_metas, resolved
    )

    assert err_reqs == ["d-req"]
    assert err_msg == "KV group count mismatch"


def test_plan_reads_rejects_mismatched_region_lengths():
    """A layout whose region sizes cannot describe this cache is refused."""
    worker = make_decoder_worker()
    pull_metas = {"d-req": pull_meta("d-req", "xfer", [[10]], "p-engine", 1)}
    resolved = {"d-req": ("xfer", [[2]])}

    _, _, _, err_reqs, err_msg = worker._plan_reads(
        producer_layout(block_len=1234), pull_metas, resolved
    )

    assert err_reqs == ["d-req"]
    assert "region length mismatch" in err_msg


# ---------------------------------------------------------------------------
#  End to end: the decoder reads what the prefiller holds
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pull_reads_producer_blocks_byte_for_byte(pair):
    """The decoder's blocks hold the producer's bytes afterwards."""
    pair.fill()
    pair.ready("xfer-1", "p-req-1", [[2, 3]])
    pair.read({"d-req-1": ("xfer-1", [[10, 11]])})

    assert await wait_for(lambda: fetch_recved(pair.decoder)) == {"d-req-1"}
    assert await wait_for(lambda: fetch_sent(pair.producer)) == {"p-req-1"}

    assert pair.decoder_kv[10].tolist() == [3] * pair.block_bytes
    assert pair.decoder_kv[11].tolist() == [4] * pair.block_bytes
    # Untouched blocks stay empty.
    assert pair.decoder_kv[12].sum().item() == 0
    # The reader asked for the address book once.
    layout = pair.decoder._peer_layouts[
        f"tcp://127.0.0.1:{pair.producer.side_channel_port}"
    ]
    assert layout.layout_id
    assert layout.kv_caches_base_addr == pair.producer.kv_caches_base_addr
    assert "xfer-1" not in pair.producer.reqs_need_send


@pytest.mark.asyncio
async def test_pull_waits_for_a_later_ready_producer(pair):
    """A reader waits while the prefiller is still computing the request."""
    pair.producer_kv[5].fill_(42)
    send_meta = pair.ready("xfer-late", "p-req-late", [[5]], ready=False)
    pair.read({"d-req-late": ("xfer-late", [[20]])})

    # The reader cannot read before the producer's blocks are ready.
    await asyncio.sleep(0.3)
    assert pair.decoder_kv[20].sum().item() == 0

    await run_on(pair.producer.sender_loop, mark_producer_ready(send_meta))

    assert await wait_for(lambda: fetch_recved(pair.decoder)) == {"d-req-late"}
    assert pair.decoder_kv[20].tolist() == [42] * pair.block_bytes
    assert await wait_for(lambda: fetch_sent(pair.producer)) == {"p-req-late"}


@pytest.mark.asyncio
async def test_piecemeal_ready_requests_keep_one_waiter_each(pair):
    """Requests that become ready one by one do not multiply prefiller waiters."""
    producer = pair.producer

    active_waiters = 0
    peak_waiters = 0
    original = producer._wait_for_blocks

    async def counting_wait(d_req_id, send_meta):
        nonlocal active_waiters, peak_waiters
        active_waiters += 1
        peak_waiters = max(peak_waiters, active_waiters)
        try:
            return await original(d_req_id, send_meta)
        finally:
            active_waiters -= 1

    producer._wait_for_blocks = counting_wait  # type: ignore[method-assign]

    send_metas = [
        pair.ready(f"xfer-{index}", f"p-req-{index}", [[index]], ready=False)
        for index in (1, 2, 3)
    ]
    pair.read(
        {f"d-req-{index}": (f"xfer-{index}", [[10 + index]]) for index in (1, 2, 3)}
    )

    for send_meta in send_metas:
        await asyncio.sleep(0.1)
        await run_on(producer.sender_loop, mark_producer_ready(send_meta))

    received: set[str] = set()

    async def all_read() -> bool:
        received.update(await fetch_recved(pair.decoder))
        return len(received) == 3

    assert await wait_for(all_read)
    assert received == {"d-req-1", "d-req-2", "d-req-3"}
    assert peak_waiters == 3
    assert active_waiters == 0


@pytest.mark.asyncio
async def test_second_wave_landing_during_a_read_still_completes(pair):
    """A second wave can answer while the decoder is inside the first read."""
    producer = pair.producer
    pair.fill()
    # Long enough that the prefiller's next response lands mid-read.
    pair.decoder.engine.read_delay = 0.5

    send_metas = [
        pair.ready(f"xfer-{index}", f"p-req-{index}", [[index]], ready=False)
        for index in (1, 2)
    ]
    pair.read({f"d-req-{index}": (f"xfer-{index}", [[10 + index]]) for index in (1, 2)})

    await run_on(producer.sender_loop, mark_producer_ready(send_metas[0]))
    # The reader is now inside the 0.5 s read of the first wave.
    await asyncio.sleep(0.1)
    await run_on(producer.sender_loop, mark_producer_ready(send_metas[1]))

    assert await wait_for(lambda: fetch_recved(pair.decoder)) == {
        "d-req-1",
        "d-req-2",
    }
    assert pair.decoder_kv[11].tolist() == [2] * pair.block_bytes
    assert pair.decoder_kv[12].tolist() == [3] * pair.block_bytes


@pytest.mark.asyncio
async def test_pull_caches_the_address_book_between_reads(pair):
    """Only the first request to a peer carries the address book."""
    producer = pair.producer
    seen: list[MooncakePullRequest] = []
    original = producer._handle_pull_request

    async def spy(identity, sock, request):
        seen.append(request)
        await original(identity, sock, request)

    producer._handle_pull_request = spy  # type: ignore[method-assign]

    pair.fill()
    for index in (1, 2):
        pair.ready(f"xfer-{index}", f"p-req-{index}", [[index]])
        pair.read({f"d-req-{index}": (f"xfer-{index}", [[index + 10]])})
        assert await wait_for(lambda: fetch_recved(pair.decoder)) == {f"d-req-{index}"}

    assert [request.want_regions for request in seen] == [True, False]
    assert pair.decoder_kv[11].tolist() == [2] * pair.block_bytes
    assert pair.decoder_kv[12].tolist() == [3] * pair.block_bytes


@pytest.mark.asyncio
async def test_pull_recovers_from_a_restarted_producer(pair):
    """A new producer address book fails that request, then reads work again."""
    producer = pair.producer
    decoder = pair.decoder
    seen: list[MooncakePullRequest] = []
    original = producer._handle_pull_request

    async def spy(identity, sock, request):
        seen.append(request)
        await original(identity, sock, request)

    producer._handle_pull_request = spy  # type: ignore[method-assign]

    async def read(index: int, p_block: int, d_block: int) -> None:
        pair.ready(f"xfer-{index}", f"p-req-{index}", [[p_block]])
        pair.read({f"d-req-{index}": (f"xfer-{index}", [[d_block]])})
        assert await wait_for(lambda: fetch_recved(decoder)) == {f"d-req-{index}"}

    pair.producer_kv[2].fill_(7)
    pair.producer_kv[3].fill_(9)
    worker_addr = f"tcp://127.0.0.1:{producer.side_channel_port}"

    await read(1, 2, 10)
    assert pair.decoder_kv[10].tolist() == [7] * pair.block_bytes

    # The prefiller comes back with a different address book.
    producer._kv_layout_id += 1

    await read(2, 3, 11)
    assert pair.decoder_kv[11].sum().item() == 0
    assert decoder.get_block_ids_with_load_errors() == {11}
    assert worker_addr not in decoder._peer_layouts

    await read(3, 3, 12)
    assert pair.decoder_kv[12].tolist() == [9] * pair.block_bytes
    assert [request.want_regions for request in seen] == [True, False, True]


@pytest.mark.asyncio
async def test_pull_heterogeneous_tp_reads_both_producer_ranks(
    monkeypatch, bootstrap_server
):
    """One decoder TP rank reads the slices two producer ranks hold."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    producer_block_bytes = page_bytes(num_kv_heads=2)
    decoder_block_bytes = page_bytes(num_kv_heads=4)

    with patch_pull_worker_dependencies(tp_size=2, tp_ranks=[0, 1]):
        p_config = create_vllm_config(
            kv_connector="MooncakePullConnector", kv_role="kv_producer"
        )
        # Both producer ranks belong to one engine.
        p_config.kv_transfer_config.engine_id = "p-engine"
        producers = [
            make_worker(p_config, make_kv_cache_config(num_kv_heads=2))
            for _ in range(2)
        ]

    with patch_pull_worker_dependencies():
        d_config = create_vllm_config(
            kv_connector="MooncakePullConnector", kv_role="kv_consumer"
        )
        decoder = make_worker(d_config, make_kv_cache_config(num_kv_heads=4))

    try:
        for rank, producer in enumerate(producers):
            producer_kv = make_kv_tensor(8, producer_block_bytes)
            producer_kv[3].fill_(rank + 1)
            producer.register_kv_caches({LAYER: producer_kv})
            register_ready_transfer(producer, "xfer-tp", f"p-req-{rank}", [[3]])

        decoder_kv = make_kv_tensor(32, decoder_block_bytes)
        decoder.register_kv_caches({LAYER: decoder_kv})

        metas = {
            "d-req-tp": pull_meta(
                "d-req-tp",
                "xfer-tp",
                [[7]],
                producers[0].engine_id,
                bootstrap_server.port,
            )
        }
        asyncio.run_coroutine_threadsafe(
            decoder._start_load_kv({producers[0].engine_id: metas}),
            decoder.receiver_loop,
        )

        assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-tp"}
        # Producer rank 0 fills the first slice, rank 1 the second.
        assert decoder_kv[7][:producer_block_bytes].tolist() == [1] * (
            producer_block_bytes
        )
        assert decoder_kv[7][producer_block_bytes:].tolist() == [2] * (
            producer_block_bytes
        )
        sent = [await wait_for(partial(fetch_sent, producer)) for producer in producers]
        assert sent == [{"p-req-0"}, {"p-req-1"}]
    finally:
        with patch_pull_worker_dependencies():
            decoder.shutdown()
            for producer in producers:
                producer.shutdown()


@pytest.mark.asyncio
async def test_pull_releases_the_producer_on_a_local_prefix_hit(pair):
    """A full local hit releases the producer without moving any bytes."""
    producer = pair.producer
    decoder = pair.decoder

    # The producer's scheduler has not registered the transfer yet, so the
    # reader's completion has to be accounted for once it does.
    pair.read({"d-req-hit": ("xfer-hit", [])})

    # An empty read moves no bytes and the decoder never enters
    # WAITING_FOR_REMOTE_KVS, so the reader reports the transfer as done for
    # the prefiller without reporting a local completion.
    assert await wait_for(
        lambda: run_on(producer.sender_loop, has_read_done(producer, "xfer-hit"))
    )
    assert await fetch_recved(decoder) == set()
    assert decoder.engine.reads == []

    await run_on(
        producer.sender_loop,
        producer.record_send_reqs(send_metadata("p-req-hit", "xfer-hit", [[2]])),
    )
    assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-hit"}
    assert "xfer-hit" not in producer.reqs_need_send
    assert pair.decoder_kv[2].sum().item() == 0


# ---------------------------------------------------------------------------
#  Failure paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pair", [1], indirect=True)
@pytest.mark.asyncio
async def test_slow_read_keeps_the_producer_blocks(pair):
    """A read that outlives the abort timeout is not mistaken for a dead reader."""
    producer = pair.producer
    pair.producer_kv[2].fill_(7)

    send_meta = pair.ready("xfer-slow", "p-req-slow", [[2]])
    # The prefiller's own request is already over, so its expiry is due.
    send_meta.expire_time = time.perf_counter() - 100
    # The decoder performs the read, so its engine is the slow one.
    pair.decoder.engine.read_delay = 2.5

    pair.read({"d-req-slow": ("xfer-slow", [[10]])})

    # Well past the abort timeout the read is still running, so the prefiller
    # must keep holding the blocks it handed out.
    await asyncio.sleep(1.8)
    assert await fetch_sent(producer) == set()
    assert "xfer-slow" in producer.reqs_need_send
    assert producer._leases.holders("xfer-slow")

    assert await wait_for(lambda: fetch_recved(pair.decoder)) == {"d-req-slow"}
    assert pair.decoder_kv[10].tolist() == [7] * pair.block_bytes
    assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-slow"}


@pytest.mark.asyncio
async def test_cancelled_read_reports_only_after_the_engine_returns(
    monkeypatch, bootstrap_server
):
    """A cancelled receiver cannot free the blocks the engine is still writing."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "10")
    block_bytes = page_bytes()

    with patch_pull_worker_dependencies():
        p_config = create_vllm_config(
            kv_connector="MooncakePullConnector", kv_role="kv_producer"
        )
        d_config = create_vllm_config(
            kv_connector="MooncakePullConnector", kv_role="kv_consumer"
        )
        producer = make_worker(p_config, make_kv_cache_config())
        decoder = make_worker(d_config, make_kv_cache_config())

        try:
            producer_kv = make_kv_tensor(8, block_bytes)
            producer_kv[4].fill_(5)
            producer.register_kv_caches({LAYER: producer_kv})
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})

            register_ready_transfer(producer, "xfer-cancel", "p-req-cancel", [[4]])
            decoder.engine.read_delay = 1.5

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
                await asyncio.sleep(0.5)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            await run_on(decoder.receiver_loop, read_then_cancel())

            # The engine thread runs on after the cancellation, so reporting the
            # request now would let the scheduler hand block 10 to another
            # request while this read still writes into it.
            assert await fetch_recved(decoder) == set()
            assert producer._leases.holders("xfer-cancel")

            assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-cancel"}
            assert decoder.get_block_ids_with_load_errors() == {10}
            assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-cancel"}
        finally:
            decoder.shutdown()
            producer.shutdown()


@pytest.mark.asyncio
async def test_failed_peer_waits_for_the_other_peer_read(monkeypatch, bootstrap_server):
    """A failed peer cannot report the request while another peer still writes."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "10")
    producer_block_bytes = page_bytes(num_kv_heads=2)
    decoder_block_bytes = page_bytes(num_kv_heads=4)

    with patch_pull_worker_dependencies(tp_size=2, tp_ranks=[0, 1]):
        p_config = create_vllm_config(
            kv_connector="MooncakePullConnector", kv_role="kv_producer"
        )
        p_config.kv_transfer_config.engine_id = "p-engine"
        producers = [
            make_worker(p_config, make_kv_cache_config(num_kv_heads=2))
            for _ in range(2)
        ]

    with patch_pull_worker_dependencies():
        d_config = create_vllm_config(
            kv_connector="MooncakePullConnector", kv_role="kv_consumer"
        )
        decoder = make_worker(d_config, make_kv_cache_config(num_kv_heads=4))

    try:
        decoder_kv = make_kv_tensor(32, decoder_block_bytes)
        decoder.register_kv_caches({LAYER: decoder_kv})
        producer_kv = make_kv_tensor(8, producer_block_bytes)
        for rank, producer in enumerate(producers):
            producer.register_kv_caches({LAYER: producer_kv})
        # The first peer refuses the handshake; the second is still computing
        # the request it was asked for.
        producers[0]._validate_head_resharding_layout = (
            lambda *args, **kwargs: "refused for this test"
        )
        late = register_ready_transfer(
            producers[1], "xfer-mix", "p-req-1", [[3]], ready=False
        )

        metas = {
            "d-req-mix": pull_meta(
                "d-req-mix",
                "xfer-mix",
                [[7]],
                producers[0].engine_id,
                bootstrap_server.port,
            )
        }
        asyncio.run_coroutine_threadsafe(
            decoder._start_load_kv({producers[0].engine_id: metas}),
            decoder.receiver_loop,
        )

        # One peer has already failed; the other has not started writing yet, so
        # nothing may be reported while its read can still land.
        await asyncio.sleep(0.6)
        assert await fetch_recved(decoder) == set()

        await run_on(producers[1].sender_loop, mark_producer_ready(late))

        # Once both peers are done the request is reported as failed.
        assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-mix"}
        assert decoder.get_block_ids_with_load_errors() == {7}
    finally:
        with patch_pull_worker_dependencies():
            decoder.shutdown()
            for producer in producers:
                producer.shutdown()


@pytest.mark.asyncio
async def test_reader_failure_releases_the_producer(pair):
    """A failed read fails the request and still frees the producer's blocks."""
    pair.decoder.engine.fail_reads = 1

    pair.ready("xfer-fail", "p-req-fail", [[2]])
    pair.read({"d-req-fail": ("xfer-fail", [[10]])})

    assert await wait_for(lambda: fetch_recved(pair.decoder)) == {"d-req-fail"}
    assert pair.decoder.get_block_ids_with_load_errors() == {10}
    assert await wait_for(lambda: fetch_sent(pair.producer)) == {"p-req-fail"}


@pytest.mark.parametrize("pair", [60], indirect=True)
@pytest.mark.asyncio
async def test_producer_abort_fails_the_reader(pair):
    """A dropped producer request fails instead of waiting for the timeout."""
    producer = pair.producer
    pair.ready("xfer-abort", "p-req-abort", [], ready=False)
    pair.read({"d-req-abort": ("xfer-abort", [[10]])})

    # Wait until the producer's reader is registered, then drop the request the
    # way an abort does.
    await wait_for(lambda: reader_registered(producer, "xfer-abort"), timeout=5.0)
    metadata = send_metadata("p-req-abort", "xfer-abort")
    metadata.reqs_to_send.clear()
    metadata.reqs_not_processed = {"xfer-abort"}
    await run_on(producer.sender_loop, producer.record_send_reqs(metadata))

    assert await wait_for(lambda: fetch_recved(pair.decoder)) == {"d-req-abort"}
    assert pair.decoder.get_block_ids_with_load_errors() == {10}


async def reader_registered(producer: MooncakePullConnectorWorker, transfer_id: str):
    return await run_on(
        producer.sender_loop, has_expected_readers(producer, transfer_id)
    )


async def has_expected_readers(producer: MooncakePullConnectorWorker, transfer_id: str):
    send_meta = producer.reqs_need_send.get(transfer_id)
    return send_meta is not None and send_meta.need_send > 0


async def has_read_done(producer: MooncakePullConnectorWorker, transfer_id: str):
    return producer._leases.reported(transfer_id) > 0


@pytest.mark.parametrize("pair", [1], indirect=True)
@pytest.mark.asyncio
async def test_producer_timeout_fails_the_reader(pair):
    """A request the prefiller never finishes fails the reader."""
    pair.ready("xfer-stuck", "p-req-stuck", [], ready=False)
    pair.read({"d-req-stuck": ("xfer-stuck", [[10]])})

    assert await wait_for(lambda: fetch_recved(pair.decoder), timeout=15.0) == {
        "d-req-stuck"
    }
    assert pair.decoder.get_block_ids_with_load_errors() == {10}


@pytest.mark.asyncio
async def test_decoder_rejects_an_incompatible_block_size(
    monkeypatch, bootstrap_server
):
    """A producer whose pages do not line up fails the request with a reason."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")

    with patch_pull_worker_dependencies():
        p_config = create_vllm_config(
            kv_connector="MooncakePullConnector",
            kv_role="kv_producer",
            block_size=BLOCK_SIZE,
        )
        d_config = create_vllm_config(
            kv_connector="MooncakePullConnector",
            kv_role="kv_consumer",
            block_size=BLOCK_SIZE * 2,
        )
        producer = make_worker(p_config, make_kv_cache_config())
        decoder = make_worker(d_config, make_kv_cache_config(block_size=BLOCK_SIZE * 2))

        try:
            producer_kv = make_kv_tensor(8, page_bytes())
            producer.register_kv_caches({LAYER: producer_kv})
            decoder_kv = make_kv_tensor(32, page_bytes(block_size=BLOCK_SIZE * 2))
            decoder.register_kv_caches({LAYER: decoder_kv})

            register_ready_transfer(producer, "xfer-size", "p-req-size", [[2]])
            metas = {
                "d-req-size": pull_meta(
                    "d-req-size",
                    "xfer-size",
                    [[10]],
                    producer.engine_id,
                    bootstrap_server.port,
                )
            }
            asyncio.run_coroutine_threadsafe(
                decoder._start_load_kv({producer.engine_id: metas}),
                decoder.receiver_loop,
            )

            assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-size"}
            assert decoder.get_block_ids_with_load_errors() == {10}
            assert decoder.engine.reads == []
            assert not decoder._peer_layouts
            # Rejecting the layout has to release the blocks already handed out.
            assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-size"}
            assert "xfer-size" not in producer.reqs_need_send
            assert not producer._leases.holders("xfer-size")
        finally:
            decoder.shutdown()
            producer.shutdown()


@pytest.mark.asyncio
async def test_producer_rejects_head_resharding_a_padded_cache(
    monkeypatch, bootstrap_server
):
    """A reader with another TP degree cannot slice a padded KV page."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    producer_block_bytes = page_bytes(num_kv_heads=2)
    padded_block_bytes = producer_block_bytes + 64

    with patch_pull_worker_dependencies(tp_size=2, tp_ranks=[0, 1]):
        p_config = create_vllm_config(
            kv_connector="MooncakePullConnector", kv_role="kv_producer"
        )
        producer = make_worker(
            p_config,
            make_kv_cache_config(num_kv_heads=2, page_size_padded=padded_block_bytes),
        )
    with patch_pull_worker_dependencies():
        d_config = create_vllm_config(
            kv_connector="MooncakePullConnector", kv_role="kv_consumer"
        )
        decoder = make_worker(d_config, make_kv_cache_config(num_kv_heads=2))

    try:
        producer_kv = make_kv_tensor(8, padded_block_bytes)
        producer.register_kv_caches({LAYER: producer_kv})
        decoder_kv = make_kv_tensor(32, producer_block_bytes)
        decoder.register_kv_caches({LAYER: decoder_kv})

        register_ready_transfer(producer, "xfer-pad", "p-req-pad", [[2]])
        metas = {
            "d-req-pad": pull_meta(
                "d-req-pad",
                "xfer-pad",
                [[10]],
                producer.engine_id,
                bootstrap_server.port,
            )
        }
        asyncio.run_coroutine_threadsafe(
            decoder._start_load_kv({producer.engine_id: metas}),
            decoder.receiver_loop,
        )

        assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-pad"}
        assert decoder.get_block_ids_with_load_errors() == {10}
        assert decoder.engine.reads == []
        # The prefiller refuses before handing out block ids, so nothing is held.
        assert not producer._leases.holders("xfer-pad")
    finally:
        with patch_pull_worker_dependencies():
            decoder.shutdown()
            producer.shutdown()


# ---------------------------------------------------------------------------
#  Prefiller bookkeeping
# ---------------------------------------------------------------------------
def test_read_leases_hold_blocks_until_reported_or_quiet():
    """The release rule: a lease ends when its reader reports or goes quiet."""
    leases = ReadLeases()
    reader, other = ("d-engine", 0, 0), ("d-engine", 1, 0)
    timeout = 10.0
    start = 1000.0

    assert leases.may_release("xfer", timeout, now=start)
    assert not leases.renew("xfer", reader, now=start)

    leases.hand_out("xfer", reader, now=start)
    leases.hand_out("xfer", other, now=start)
    assert leases.holders("xfer") == [reader, other]
    assert not leases.may_release("xfer", timeout, now=start + 5)

    leases.report("xfer", reader)
    assert leases.reported("xfer") == 1
    assert not leases.may_release("xfer", timeout, now=start + 5)

    # A reader that keeps renewing holds its blocks past the timeout.
    assert leases.renew("xfer", other, now=start + 9)
    assert not leases.may_release("xfer", timeout, now=start + 15)

    # Neglect ends the lease, and the transfer is free to be released.
    assert leases.may_release("xfer", timeout, now=start + 25)
    assert not leases.holders("xfer")

    leases.forget("xfer")
    assert leases.reported("xfer") == 0


def test_expiry_skips_transfers_a_reader_is_still_reading():
    """Blocks handed out to a reader are never expired underneath it."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakePullConnector", kv_role="kv_producer"
    )
    with patch_pull_worker_dependencies():
        producer = make_worker(vllm_config, make_kv_cache_config())

    reader = ("d-engine", 0, 0)
    fresh = SendBlockMeta(
        p_req_id="p-req-fresh",
        transfer_id="xfer-fresh",
        local_block_ids=[[1]],
        ready=asyncio.Event(),
        expire_time=time.perf_counter() - 100,
    )
    producer.reqs_need_send["xfer-fresh"] = fresh
    producer._leases.hand_out("xfer-fresh", reader)

    expired = SendBlockMeta(
        p_req_id="p-req-expired",
        transfer_id="xfer-expired",
        local_block_ids=[[2]],
        ready=asyncio.Event(),
        expire_time=time.perf_counter() - 100,
    )
    producer.reqs_need_send["xfer-expired"] = expired

    finished = asyncio.run(producer.fetch_finished_sending_reqs())

    assert finished == {"p-req-expired"}
    assert "xfer-fresh" in producer.reqs_need_send
    assert "xfer-expired" not in producer.reqs_need_send


def test_a_reader_that_never_reports_stops_holding_blocks(monkeypatch):
    """A reader that dies after taking block ids cannot pin them forever."""
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakePullConnector", kv_role="kv_producer"
    )
    with patch_pull_worker_dependencies():
        producer = make_worker(vllm_config, make_kv_cache_config())
        try:
            producer.reqs_need_send["xfer-silent"] = SendBlockMeta(
                p_req_id="p-req-silent",
                transfer_id="xfer-silent",
                local_block_ids=[[1]],
                ready=asyncio.Event(),
                expire_time=time.perf_counter() - 100,
            )
            reader = ("d-engine", 0, 0)
            producer._leases.hand_out("xfer-silent", reader)
            assert producer._send_idle("xfer-silent") is False

            # Backdate the lease: the reader never reported and never renewed.
            producer._leases.hand_out(
                "xfer-silent", reader, now=time.perf_counter() - 6
            )
            assert asyncio.run(producer.fetch_finished_sending_reqs()) == {
                "p-req-silent"
            }
            assert "xfer-silent" not in producer.reqs_need_send
        finally:
            producer.shutdown()
