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
from functools import partial
from unittest.mock import patch

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compat import (
    compute_mooncake_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    PullReqMeta,
    SendBlockMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_pull_connector import (  # noqa: E501
    MooncakePullConnector,
    MooncakePullConnectorWorker,
    MooncakePullLayout,
    MooncakePullRequest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    EngineEntry,
    MooncakeBootstrapServer,
)
from vllm.utils.network_utils import get_open_port
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


def make_kv_cache_config(block_size: int = BLOCK_SIZE, num_kv_heads: int = KV_HEADS):
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
    return MooncakePullLayout(
        engine_id="p-engine",
        hostname="producer-host",
        rpc_port=12345,
        tp_rank=0,
        pp_rank=0,
        tp_size=1,
        pp_size=1,
        block_size=BLOCK_SIZE,
        physical_blocks_per_logical=1,
        use_mla=False,
        has_mamba=False,
        producer_cache_replicated=False,
        kv_caches_base_addr=[base_addr],
        block_lens=[block_len],
        kv_block_lens=[block_len],
        layer_names=[LAYER],
        layer_indices=[0],
        group_indices=[0],
        layout_id="layout",
    )


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
#  Layout handshake
# ---------------------------------------------------------------------------
def test_layout_id_tracks_the_address_book():
    worker = make_decoder_worker()

    layout = worker._get_layout()
    assert layout.layout_id
    assert worker._get_layout() is layout

    worker._layout = None
    worker.kv_caches_base_addr = [0x9000]
    assert worker._get_layout().layout_id != layout.layout_id


def test_layout_mismatch_reason_names_the_difference():
    worker = make_decoder_worker()
    assert worker._layout_mismatch_reason(producer_layout()) is None

    wrong_block_size = producer_layout()
    wrong_block_size.block_size = BLOCK_SIZE * 2
    assert "block size" in worker._layout_mismatch_reason(wrong_block_size)

    wrong_ratio = producer_layout()
    wrong_ratio.physical_blocks_per_logical = 17
    assert "physical blocks per logical" in worker._layout_mismatch_reason(wrong_ratio)

    wrong_mla = producer_layout()
    wrong_mla.use_mla = True
    assert "use_mla" in worker._layout_mismatch_reason(wrong_mla)


# ---------------------------------------------------------------------------
#  End to end: the decoder reads what the prefiller holds
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pull_reads_producer_blocks_byte_for_byte(monkeypatch, bootstrap_server):
    """The decoder's blocks hold the producer's bytes afterwards."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
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
            for block_id in range(8):
                producer_kv[block_id].fill_(block_id + 1)
            producer.register_kv_caches({LAYER: producer_kv})
            assert producer.side_channel_port > 0

            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})

            register_ready_transfer(producer, "xfer-1", "p-req-1", [[2, 3]])
            metas = {
                "d-req-1": pull_meta(
                    "d-req-1",
                    "xfer-1",
                    [[10, 11]],
                    producer.engine_id,
                    bootstrap_server.port,
                )
            }

            asyncio.run_coroutine_threadsafe(
                decoder._start_load_kv({producer.engine_id: metas}),
                decoder.receiver_loop,
            )

            assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-1"}
            assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-1"}

            assert decoder_kv[10].tolist() == [3] * block_bytes
            assert decoder_kv[11].tolist() == [4] * block_bytes
            # Untouched blocks stay empty.
            assert decoder_kv[12].sum().item() == 0
            # The reader asked for the address book once.
            layout = decoder._peer_layouts[
                f"tcp://127.0.0.1:{producer.side_channel_port}"
            ]
            assert layout.layout_id
            assert layout.kv_caches_base_addr == producer.kv_caches_base_addr
            assert "xfer-1" not in producer.reqs_need_send
        finally:
            decoder.shutdown()
            producer.shutdown()


@pytest.mark.asyncio
async def test_pull_waits_for_a_later_ready_producer(monkeypatch, bootstrap_server):
    """A reader waits while the prefiller is still computing the request."""
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
            producer_kv[5].fill_(42)
            producer.register_kv_caches({LAYER: producer_kv})
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})

            send_meta = register_ready_transfer(
                producer, "xfer-late", "p-req-late", [[5]], ready=False
            )
            metas = {
                "d-req-late": pull_meta(
                    "d-req-late",
                    "xfer-late",
                    [[20]],
                    producer.engine_id,
                    bootstrap_server.port,
                )
            }
            asyncio.run_coroutine_threadsafe(
                decoder._start_load_kv({producer.engine_id: metas}),
                decoder.receiver_loop,
            )

            # The reader cannot read before the producer's blocks are ready.
            await asyncio.sleep(0.3)
            assert decoder_kv[20].sum().item() == 0

            await run_on(producer.sender_loop, mark_producer_ready(send_meta))

            assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-late"}
            assert decoder_kv[20].tolist() == [42] * block_bytes
            assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-late"}
        finally:
            decoder.shutdown()
            producer.shutdown()


@pytest.mark.asyncio
async def test_pull_caches_the_address_book_between_reads(
    monkeypatch, bootstrap_server
):
    """Only the first request to a peer carries the address book."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
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

        seen: list[MooncakePullRequest] = []
        original = producer._handle_pull_request

        async def spy(identity, sock, request):
            seen.append(request)
            await original(identity, sock, request)

        producer._handle_pull_request = spy  # type: ignore[method-assign]

        try:
            producer_kv = make_kv_tensor(8, block_bytes)
            for block_id in range(8):
                producer_kv[block_id].fill_(block_id + 1)
            producer.register_kv_caches({LAYER: producer_kv})
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})

            for index in (1, 2):
                register_ready_transfer(
                    producer, f"xfer-{index}", f"p-req-{index}", [[index]]
                )
                metas = {
                    f"d-req-{index}": pull_meta(
                        f"d-req-{index}",
                        f"xfer-{index}",
                        [[index + 10]],
                        producer.engine_id,
                        bootstrap_server.port,
                    )
                }
                asyncio.run_coroutine_threadsafe(
                    decoder._start_load_kv({producer.engine_id: metas}),
                    decoder.receiver_loop,
                )
                assert await wait_for(lambda: fetch_recved(decoder)) == {
                    f"d-req-{index}"
                }

            assert [request.want_layout for request in seen] == [True, False]
            assert decoder_kv[11].tolist() == [2] * block_bytes
            assert decoder_kv[12].tolist() == [3] * block_bytes
        finally:
            decoder.shutdown()
            producer.shutdown()


@pytest.mark.asyncio
async def test_pull_recovers_from_a_restarted_producer(monkeypatch, bootstrap_server):
    """A new producer address book fails that request, then reads work again."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
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

        seen: list[MooncakePullRequest] = []
        original = producer._handle_pull_request

        async def spy(identity, sock, request):
            seen.append(request)
            await original(identity, sock, request)

        producer._handle_pull_request = spy  # type: ignore[method-assign]

        async def read(index: int, p_block: int, d_block: int) -> None:
            register_ready_transfer(
                producer, f"xfer-{index}", f"p-req-{index}", [[p_block]]
            )
            metas = {
                f"d-req-{index}": pull_meta(
                    f"d-req-{index}",
                    f"xfer-{index}",
                    [[d_block]],
                    producer.engine_id,
                    bootstrap_server.port,
                )
            }
            asyncio.run_coroutine_threadsafe(
                decoder._start_load_kv({producer.engine_id: metas}),
                decoder.receiver_loop,
            )
            assert await wait_for(lambda: fetch_recved(decoder)) == {f"d-req-{index}"}

        try:
            producer_kv = make_kv_tensor(8, block_bytes)
            producer_kv[2].fill_(7)
            producer_kv[3].fill_(9)
            producer.register_kv_caches({LAYER: producer_kv})
            worker_addr = f"tcp://127.0.0.1:{producer.side_channel_port}"
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})

            await read(1, 2, 10)
            assert decoder_kv[10].tolist() == [7] * block_bytes

            # The prefiller comes back with a different address book.
            producer._layout.layout_id = "restarted"

            await read(2, 3, 11)
            assert decoder_kv[11].sum().item() == 0
            assert decoder.get_block_ids_with_load_errors() == {11}
            assert worker_addr not in decoder._peer_layouts

            await read(3, 3, 12)
            assert decoder_kv[12].tolist() == [9] * block_bytes
            assert [request.want_layout for request in seen] == [True, False, True]
        finally:
            decoder.shutdown()
            producer.shutdown()


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
async def test_pull_releases_the_producer_on_a_local_prefix_hit(
    monkeypatch, bootstrap_server
):
    """A full local hit releases the producer without moving any bytes."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
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
            producer_kv[2].fill_(7)
            producer.register_kv_caches({LAYER: producer_kv})
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})

            # The producer's scheduler has not registered the transfer yet, so
            # the reader's completion has to be accounted for once it does.
            metas = {
                "d-req-hit": pull_meta(
                    "d-req-hit",
                    "xfer-hit",
                    [],
                    producer.engine_id,
                    bootstrap_server.port,
                )
            }
            asyncio.run_coroutine_threadsafe(
                decoder._start_load_kv({producer.engine_id: metas}),
                decoder.receiver_loop,
            )

            # An empty read moves no bytes and the decoder never enters
            # WAITING_FOR_REMOTE_KVS, so the reader reports the transfer as
            # done for the prefiller without reporting a local completion.
            assert await wait_for(
                lambda: run_on(
                    producer.sender_loop, has_read_done(producer, "xfer-hit")
                )
            )
            assert await fetch_recved(decoder) == set()
            assert decoder.engine.reads == []

            await run_on(
                producer.sender_loop,
                producer.record_send_reqs(
                    send_metadata("p-req-hit", "xfer-hit", [[2]])
                ),
            )
            assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-hit"}
            assert "xfer-hit" not in producer.reqs_need_send
            assert decoder_kv[2].sum().item() == 0
        finally:
            decoder.shutdown()
            producer.shutdown()


# ---------------------------------------------------------------------------
#  Failure paths
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reader_failure_releases_the_producer(monkeypatch, bootstrap_server):
    """A failed read fails the request and still frees the producer's blocks."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
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
            producer.register_kv_caches({LAYER: producer_kv})
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})
            decoder.engine.fail_reads = 1

            register_ready_transfer(producer, "xfer-fail", "p-req-fail", [[2]])
            metas = {
                "d-req-fail": pull_meta(
                    "d-req-fail",
                    "xfer-fail",
                    [[10]],
                    producer.engine_id,
                    bootstrap_server.port,
                )
            }
            asyncio.run_coroutine_threadsafe(
                decoder._start_load_kv({producer.engine_id: metas}),
                decoder.receiver_loop,
            )

            assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-fail"}
            assert decoder.get_block_ids_with_load_errors() == {10}
            assert await wait_for(lambda: fetch_sent(producer)) == {"p-req-fail"}
        finally:
            decoder.shutdown()
            producer.shutdown()


@pytest.mark.asyncio
async def test_producer_abort_fails_the_reader(monkeypatch, bootstrap_server):
    """A dropped producer request fails instead of waiting for the timeout."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "60")
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
            producer.register_kv_caches({LAYER: producer_kv})
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})

            register_ready_transfer(
                producer, "xfer-abort", "p-req-abort", [], ready=False
            )
            metas = {
                "d-req-abort": pull_meta(
                    "d-req-abort",
                    "xfer-abort",
                    [[10]],
                    producer.engine_id,
                    bootstrap_server.port,
                )
            }
            asyncio.run_coroutine_threadsafe(
                decoder._start_load_kv({producer.engine_id: metas}),
                decoder.receiver_loop,
            )

            # Wait until the producer's reader is registered, then drop the
            # request the way an abort does.
            await wait_for(
                lambda: reader_registered(producer, "xfer-abort"), timeout=5.0
            )
            metadata = send_metadata("p-req-abort", "xfer-abort")
            metadata.reqs_to_send.clear()
            metadata.reqs_not_processed = {"xfer-abort"}
            await run_on(producer.sender_loop, producer.record_send_reqs(metadata))

            assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-abort"}
            assert decoder.get_block_ids_with_load_errors() == {10}
        finally:
            decoder.shutdown()
            producer.shutdown()


async def reader_registered(producer: MooncakePullConnectorWorker, transfer_id: str):
    return await run_on(
        producer.sender_loop, has_expected_readers(producer, transfer_id)
    )


async def has_expected_readers(producer: MooncakePullConnectorWorker, transfer_id: str):
    send_meta = producer.reqs_need_send.get(transfer_id)
    return send_meta is not None and send_meta.need_send > 0


async def has_read_done(producer: MooncakePullConnectorWorker, transfer_id: str):
    return bool(producer._done_readers.get(transfer_id))


@pytest.mark.asyncio
async def test_producer_timeout_fails_the_reader(monkeypatch, bootstrap_server):
    """A request the prefiller never finishes fails the reader."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "1")
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
            producer.register_kv_caches({LAYER: producer_kv})
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})

            register_ready_transfer(
                producer, "xfer-stuck", "p-req-stuck", [], ready=False
            )
            metas = {
                "d-req-stuck": pull_meta(
                    "d-req-stuck",
                    "xfer-stuck",
                    [[10]],
                    producer.engine_id,
                    bootstrap_server.port,
                )
            }
            asyncio.run_coroutine_threadsafe(
                decoder._start_load_kv({producer.engine_id: metas}),
                decoder.receiver_loop,
            )

            assert await wait_for(lambda: fetch_recved(decoder), timeout=15.0) == {
                "d-req-stuck"
            }
            assert decoder.get_block_ids_with_load_errors() == {10}
        finally:
            decoder.shutdown()
            producer.shutdown()


@pytest.mark.asyncio
async def test_decoder_rejects_a_push_peer(monkeypatch, bootstrap_server):
    """A prefiller running the push connector is refused before any transfer."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    block_bytes = page_bytes()

    with patch_pull_worker_dependencies():
        d_config = create_vllm_config(
            kv_connector="MooncakePullConnector", kv_role="kv_consumer"
        )
        decoder = make_worker(d_config, make_kv_cache_config())

        try:
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})

            # A prefiller that registered itself as the push connector.
            bootstrap_server.workers[0] = EngineEntry(
                engine_id="p-engine",
                connector="MooncakeConnector",
                worker_addr={0: {0: "tcp://127.0.0.1:1"}},
            )
            metas = {
                "d-req-mix": pull_meta(
                    "d-req-mix",
                    "xfer-mix",
                    [[10]],
                    "p-engine",
                    bootstrap_server.port,
                )
            }
            asyncio.run_coroutine_threadsafe(
                decoder._start_load_kv({"p-engine": metas}), decoder.receiver_loop
            )

            assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-mix"}
            assert decoder.get_block_ids_with_load_errors() == {10}
            assert decoder.engine.reads == []
        finally:
            decoder.shutdown()


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
        finally:
            decoder.shutdown()
            producer.shutdown()


# ---------------------------------------------------------------------------
#  Prefiller bookkeeping
# ---------------------------------------------------------------------------
def test_reader_topology_must_be_consistent():
    """Two readers claiming different topologies for one transfer is a bug."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakePullConnector", kv_role="kv_producer"
    )
    with patch_pull_worker_dependencies():
        producer = make_worker(vllm_config, make_kv_cache_config())

    send_meta = SendBlockMeta(
        p_req_id="p-req",
        transfer_id="xfer",
        local_block_ids=[[1]],
        ready=asyncio.Event(),
    )
    producer._resolve_expected_readers(send_meta, reader_tp_size=1, reader_pp_size=1)
    assert send_meta.need_send == 1

    with pytest.raises(RuntimeError, match="reader topology"):
        producer._resolve_expected_readers(
            send_meta, reader_tp_size=2, reader_pp_size=1
        )


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
    producer._handed_out_readers["xfer-fresh"] = {reader}

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


# ---------------------------------------------------------------------------
#  Compatibility hash
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({}, True),
        ({"block_size": BLOCK_SIZE * 2}, True),
        ({"cache_dtype": "fp8"}, False),
        ({"model": "facebook/opt-350m"}, False),
    ],
)
def test_compat_hash_tracks_the_transferred_cache(overrides, match):
    """The hash changes with the KV cache format, not with the local mapping."""
    base = create_vllm_config()
    other = create_vllm_config(**overrides)
    if "model" in overrides:
        other.model_config.model = overrides["model"]

    assert (
        compute_mooncake_compatibility_hash(base, "FLASH_ATTN")
        == compute_mooncake_compatibility_hash(other, "FLASH_ATTN")
    ) is match


def test_compat_hash_separates_the_two_transfer_modes():
    """A push peer and a pull peer must never agree on a handshake."""
    config = create_vllm_config()

    pull_hash = compute_mooncake_compatibility_hash(
        config, "FLASH_ATTN", transfer_mode="pull"
    )
    assert pull_hash != compute_mooncake_compatibility_hash(
        config, "FLASH_ATTN", transfer_mode="push"
    )
    assert pull_hash != compute_mooncake_compatibility_hash(config, "FLASHINFER")
    assert compute_mooncake_compatibility_hash(config, "FLASH_ATTN") == pull_hash


@pytest.mark.asyncio
async def test_incompatible_peer_fails_before_any_read(monkeypatch, bootstrap_server):
    """A peer whose configuration differs is refused, not partially read."""
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(bootstrap_server.port))
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
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
            producer.register_kv_caches({LAYER: producer_kv})
            decoder_kv = make_kv_tensor(32, block_bytes)
            decoder.register_kv_caches({LAYER: decoder_kv})

            # The peer pretends to run a different configuration.
            producer._compat_hash = "not-this-decoder"
            register_ready_transfer(producer, "xfer-mix", "p-req-mix", [[2]])
            metas = {
                "d-req-mix": pull_meta(
                    "d-req-mix",
                    "xfer-mix",
                    [[10]],
                    producer.engine_id,
                    bootstrap_server.port,
                )
            }
            asyncio.run_coroutine_threadsafe(
                decoder._start_load_kv({producer.engine_id: metas}),
                decoder.receiver_loop,
            )

            assert await wait_for(lambda: fetch_recved(decoder)) == {"d-req-mix"}
            assert decoder.get_block_ids_with_load_errors() == {10}
            assert decoder.engine.reads == []
        finally:
            decoder.shutdown()
            producer.shutdown()


def test_pull_connector_drives_the_read_worker():
    """The registered pull name runs the read-mode worker, not the push one."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakePullConnector", kv_role="kv_consumer"
    )
    with patch_pull_worker_dependencies():
        worker = make_worker(vllm_config, make_kv_cache_config())
        try:
            assert isinstance(worker, MooncakePullConnectorWorker)
            assert worker.connector_name == "MooncakePullConnector"
        finally:
            worker.shutdown()
