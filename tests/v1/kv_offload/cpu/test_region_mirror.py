# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot mirroring between a remote worker's offload region and the head's."""

import itertools
import json
import sys
import threading
import time
import uuid
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import zmq

from vllm.v1.kv_offload.base import OffloadingWorker, TransferResult
from vllm.v1.kv_offload.cpu import region_mirror
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.node_topology import MirrorTopology
from vllm.v1.kv_offload.cpu.region_mirror import (
    RegionMirrorClient,
    RegionMirrorServer,
    RegionMirrorWorker,
    SlotLayout,
    attach_region_mirror,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

NUM_CHUNKS = 4
SLOT = 4096
WORLD_SIZE = 3
REMOTE_RANK = 2
TOKEN = "secret-token"
REMOTES = frozenset({REMOTE_RANK})
ALLOWED = frozenset({"127.0.0.1"})
HEAD = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=SLOT * WORLD_SIZE, size=SLOT)
LOCAL = SlotLayout(num_chunks=NUM_CHUNKS, row_stride=SLOT, size=SLOT)


def _region(layout: SlotLayout) -> SharedOffloadRegion:
    return SharedOffloadRegion(
        engine_id=f"test-mirror-{uuid.uuid4().hex}",
        num_chunks=NUM_CHUNKS,
        rank=None,
        kv_bytes_per_chunk=layout.row_stride,
        cpu_page_size=SLOT,
    )


def _head_slot(rows: np.ndarray, chunk: int, rank: int) -> np.ndarray:
    return rows[chunk, rank * SLOT : (rank + 1) * SLOT]


def _wait_for(predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


@pytest.fixture
def regions():
    head, remote = _region(HEAD), _region(LOCAL)
    yield head, remote
    head.cleanup()
    remote.cleanup()


@pytest.fixture
def mirror(regions):
    head, remote = regions
    server = RegionMirrorServer(
        head, HEAD, "127.0.0.1", TOKEN, REMOTES, allowed_hosts=ALLOWED
    )
    client = RegionMirrorClient(remote, LOCAL, REMOTE_RANK, server.address, TOKEN)
    yield HEAD.rows(head), LOCAL.rows(remote), client, server
    client.shutdown()
    server.shutdown()


def _raw_request(
    address: str, frames: list[bytes], *, source: str | None = None, wait_ms=5000
) -> list[bytes] | None:
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(
        address
        if source is None
        else f"tcp://{source}:0;{address.removeprefix('tcp://')}"
    )
    try:
        socket.send_multipart(frames)
        if not socket.poll(wait_ms):
            return None
        return socket.recv_multipart()
    finally:
        socket.close()
        context.term()


def _header(**fields) -> bytes:
    request = {
        "id": "r1",
        "token": TOKEN,
        "seq": 10**9,
        "epoch": "e",
        "op": "put",
        "rank": REMOTE_RANK,
        "chunks": [0],
    }
    request.update(fields)
    return json.dumps(request).encode()


def test_push_writes_only_the_remote_slot(mirror):
    head_rows, local_rows, client, _ = mirror
    local_rows[:] = 7
    client.push(np.array([1, 3]))

    for chunk in range(NUM_CHUNKS):
        expected = 7 if chunk in (1, 3) else 0
        assert (_head_slot(head_rows, chunk, REMOTE_RANK) == expected).all()
        assert (_head_slot(head_rows, chunk, 0) == 0).all()


def test_pull_reads_the_remote_slot(mirror):
    head_rows, local_rows, client, _ = mirror
    _head_slot(head_rows, 2, REMOTE_RANK)[:] = 5
    client.pull(np.array([2]))

    assert (local_rows[2] == 5).all()
    assert (local_rows[1] == 0).all()


@pytest.mark.parametrize(
    "frames",
    [
        [b"not json"],
        [_header(token="wrong"), b"\0" * SLOT],
        [_header(op="delete"), b"\0" * SLOT],
        [_header(rank=0), b"\0" * SLOT],
        [_header(rank=True), b"\0" * SLOT],
        [_header(rank=WORLD_SIZE), b"\0" * SLOT],
        [_header(seq="1"), b"\0" * SLOT],
        [_header(chunks=[-1]), b"\0" * SLOT],
        [_header(chunks=[1.0]), b"\0" * SLOT],
        [_header(chunks=[NUM_CHUNKS]), b"\0" * SLOT],
        [_header(chunks=[]), b""],
        [_header(), b"\0" * (SLOT - 1)],
        [_header(op="get", chunks=list(range(NUM_CHUNKS)) * 3000)],
        # Head-node ranks write the head region themselves.
        [_header(rank=1), b"\1" * SLOT],
        [_header(), b"\1" * SLOT, b"\2", b"\3"],
    ],
)
def test_server_rejects_invalid_requests_and_keeps_serving(mirror, frames):
    head_rows, local_rows, client, server = mirror

    reply = _raw_request(server.address, frames)

    assert reply[-1] == b"error"
    assert not head_rows.any()
    local_rows[:] = 3
    client.push(np.array([0]))
    assert (_head_slot(head_rows, 0, REMOTE_RANK) == 3).all()


def test_large_copies_are_split_into_bounded_requests(regions):
    head, remote = regions
    server = RegionMirrorServer(
        head, HEAD, "127.0.0.1", TOKEN, REMOTES, allowed_hosts=ALLOWED
    )
    client = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, server.address, TOKEN, max_request_bytes=SLOT
    )
    sent: list[int] = []
    request = client._request
    client._request = lambda op, chunk_ids, *payload: (
        sent.append(len(chunk_ids)) or request(op, chunk_ids, *payload)
    )
    head_rows, local_rows = HEAD.rows(head), LOCAL.rows(remote)
    try:
        local_rows[:] = 6
        client.push(np.arange(NUM_CHUNKS))
        head_rows[:, REMOTE_RANK * SLOT :] = 8
        client.pull(np.arange(NUM_CHUNKS))
    finally:
        client.shutdown()
        server.shutdown()

    assert sent == [1] * (2 * NUM_CHUNKS)
    assert (local_rows == 8).all()
    del head_rows, local_rows


class _ScriptedHead:
    """ROUTER that answers requests as scripted, on a background thread."""

    def __init__(self, script):
        self.context = zmq.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        port = self.router.bind_to_random_port("tcp://127.0.0.1")
        self.address = f"tcp://127.0.0.1:{port}"
        self.thread = threading.Thread(target=script, args=(self.router,))
        self.thread.start()

    def close(self):
        self.thread.join()
        self.router.close()
        self.context.term()


def test_client_drops_replies_to_other_requests(regions):
    _, remote = regions

    def script(router):
        identity, header, _ = router.recv_multipart()
        request_id = json.loads(header)["id"].encode()
        router.send_multipart([identity, b"stale-id", b"error"])
        router.send_multipart([identity, request_id, b"ok"])

    head = _ScriptedHead(script)
    client = RegionMirrorClient(remote, LOCAL, REMOTE_RANK, head.address, TOKEN)
    try:
        client.push(np.array([0]))
    finally:
        head.close()
        client.shutdown()


def test_client_retries_on_a_fresh_socket(regions):
    _, remote = regions
    seen: list[tuple[bytes, int]] = []

    def script(router):
        for answer in (False, True):
            identity, header, *_ = router.recv_multipart()
            request = json.loads(header)
            seen.append((identity, request["seq"]))
            if answer:
                router.send_multipart([identity, request["id"].encode(), b"ok"])

    head = _ScriptedHead(script)
    client = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, head.address, TOKEN, timeout_ms=200
    )
    try:
        client.push(np.array([0]))
    finally:
        head.close()
        client.shutdown()

    (first_socket, first_seq), (second_socket, second_seq) = seen
    assert first_socket != second_socket
    assert second_seq > first_seq


def test_client_gives_up_after_its_attempts(regions):
    _, remote = regions
    head = _ScriptedHead(lambda router: None)
    client = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, head.address, TOKEN, timeout_ms=100, attempts=2
    )
    try:
        with pytest.raises(RuntimeError, match="after 2 attempts"):
            client.push(np.array([0]))
    finally:
        head.close()
        client.shutdown()


def _inner() -> MagicMock:
    # Spec'd on OffloadingWorker: the real CPU worker needs a GPU.
    inner = create_autospec(OffloadingWorker, instance=True)
    inner.submit_store.return_value = True
    inner.submit_load.return_value = True
    inner.get_finished.return_value = []
    return inner


def test_store_is_reported_only_after_its_push(mirror):
    head_rows, local_rows, client, _ = mirror
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)

    assert worker.submit_store(1, MagicMock(), CPULoadStoreSpec([0]))
    local_rows[0] = 9
    inner.get_finished.return_value = [TransferResult(job_id=1, success=True)]
    first = worker.get_finished()
    inner.get_finished.return_value = []

    results = first + _wait_for(worker.get_finished)
    assert results == [TransferResult(job_id=1, success=True)]
    assert (_head_slot(head_rows, 0, REMOTE_RANK) == 9).all()
    worker.shutdown()


def test_load_dma_starts_only_after_its_pull(mirror):
    head_rows, local_rows, client, _ = mirror
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    _head_slot(head_rows, 3, REMOTE_RANK)[:] = 4
    src, dst = CPULoadStoreSpec([3]), MagicMock()

    assert worker.submit_load(2, src, dst)

    def dma_submitted() -> bool:
        # get_finished() submits the DMA of a load whose pull completed.
        worker.get_finished()
        return inner.submit_load.called

    _wait_for(dma_submitted)

    inner.submit_load.assert_called_once_with(2, src, dst)
    assert (local_rows[3] == 4).all()
    worker.shutdown()


def test_wait_submits_a_pending_load_before_waiting(mirror):
    _, _, client, _ = mirror
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    src, dst = CPULoadStoreSpec([1]), MagicMock()

    worker.submit_load(5, src, dst)
    worker.wait({5})

    inner.submit_load.assert_called_once_with(5, src, dst)
    inner.wait.assert_called_once_with({5})
    worker.shutdown()


def test_a_failed_push_is_fenced_before_it_is_reported():
    inner = _inner()
    client = MagicMock(chunks_per_request=64, stopped=False)
    client.push.side_effect = RuntimeError("head slow")
    worker = RegionMirrorWorker(inner, client=client)

    worker.submit_store(1, MagicMock(), CPULoadStoreSpec([0]))
    inner.get_finished.return_value = [TransferResult(job_id=1, success=True)]
    assert worker.get_finished() == []
    inner.get_finished.return_value = []
    results = _wait_for(worker.get_finished)
    worker.shutdown()

    assert [(r.job_id, r.success) for r in results] == [(1, False)]
    client.fence.assert_called_once()
    # Only the shutdown stops the client.
    assert client.stop.call_args_list == [()]


def test_an_unfenced_failed_push_is_never_reported():
    # The head may still apply a queued attempt of the push, so its chunk
    # must not be released for reuse: the worker fails instead.
    inner = _inner()
    client = MagicMock(chunks_per_request=64, stopped=False)
    client.push.side_effect = RuntimeError("head unreachable")
    client.fence.side_effect = RuntimeError("head unreachable")
    worker = RegionMirrorWorker(inner, client=client)

    worker.submit_store(1, MagicMock(), CPULoadStoreSpec([0]))
    # The inner worker reports the store's DMA once.
    inner.get_finished.side_effect = itertools.chain(
        [[TransferResult(job_id=1, success=True)]], itertools.repeat([])
    )
    reported: list[TransferResult] = []

    def finished():
        try:
            reported.extend(worker.get_finished())
        except RuntimeError as e:
            return e

    error = _wait_for(finished)
    worker.shutdown()

    assert "could not fence" in str(error)
    assert reported == []
    assert "fenced" in client.stop.call_args_list[0].args[0]


def test_a_fence_rejects_older_puts_of_its_rank(mirror):
    head_rows, _, _, server = mirror
    fence = [_header(op="fence", chunks=[], seq=10, epoch="e1")]
    assert _raw_request(server.address, fence)[-1] == b"ok"

    late = [_header(seq=9, epoch="e1"), b"\1" * SLOT]

    assert _raw_request(server.address, late)[-1] == b"error"
    assert not head_rows.any()


def test_failed_pull_reports_a_failed_load_without_its_dma():
    inner = _inner()
    inner.get_finished.return_value = []
    client = MagicMock(chunks_per_request=64, stopped=False)
    client.pull.side_effect = RuntimeError("head unreachable")
    worker = RegionMirrorWorker(inner, client=client)

    worker.submit_load(2, CPULoadStoreSpec([0]), MagicMock())
    results = _wait_for(worker.get_finished)
    worker.shutdown()

    assert [(r.job_id, r.success) for r in results] == [(2, False)]
    inner.submit_load.assert_not_called()


def test_shutdown_is_prompt_when_the_head_is_gone(regions):
    _, remote = regions
    head = _ScriptedHead(lambda router: None)
    client = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, head.address, TOKEN, timeout_ms=30_000
    )
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    for job_id in range(5):
        worker.submit_store(job_id, MagicMock(), CPULoadStoreSpec([0]))
    inner.get_finished.return_value = [
        TransferResult(job_id=j, success=True) for j in range(5)
    ]
    worker.get_finished()

    start = time.monotonic()
    worker.shutdown()

    assert time.monotonic() - start < 5
    inner.shutdown.assert_called_once()
    head.close()


def test_shutdown_releases_mirror_views_before_the_region_closes(monkeypatch):
    from vllm.v1.kv_offload.cpu import shared_offload_region

    warnings: list[str] = []
    monkeypatch.setattr(
        shared_offload_region.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(msg),
    )
    head, remote = _region(HEAD), _region(LOCAL)
    server = RegionMirrorServer(
        head, HEAD, "127.0.0.1", TOKEN, REMOTES, allowed_hosts=ALLOWED
    )
    client = RegionMirrorClient(remote, LOCAL, REMOTE_RANK, server.address, TOKEN)
    head_inner, remote_inner = _inner(), _inner()
    head_inner.shutdown.side_effect = head.cleanup
    remote_inner.shutdown.side_effect = remote.cleanup

    RegionMirrorWorker(remote_inner, client=client).shutdown()
    RegionMirrorWorker(head_inner, server=server).shutdown()

    # A live mirror view would make mmap.close() fail with BufferError.
    assert warnings == []
    assert head.mmap_obj is None and remote.mmap_obj is None


class _Exchange:
    """In-process stand-in for the group all-gather between two ranks."""

    def __init__(self, world_size: int):
        self.barrier = threading.Barrier(world_size)
        self.slots: list = [None] * world_size

    def all_gather(self, topology, value):
        self.slots[topology.rank] = value
        self.barrier.wait(timeout=10)
        values = list(self.slots)
        self.barrier.wait(timeout=10)
        return values


def _run_ranks(
    monkeypatch,
    regions,
    *,
    server_error: Exception | None = None,
    host_ip: str | None = "127.0.0.1",
):
    if host_ip is None:
        monkeypatch.delenv("VLLM_HOST_IP", raising=False)
    else:
        monkeypatch.setenv("VLLM_HOST_IP", host_ip)
    exchange = _Exchange(2)
    monkeypatch.setattr(region_mirror, "_all_gather", exchange.all_gather)
    monkeypatch.setattr(
        "vllm.utils.network_utils.get_ip",
        lambda: host_ip or "127.0.0.1",
        raising=True,
    )
    if server_error is not None:
        monkeypatch.setattr(
            region_mirror,
            "RegionMirrorServer",
            MagicMock(side_effect=server_error),
        )
    head, remote = regions
    outcomes: dict[int, object] = {}

    def rank(index: int, region: SharedOffloadRegion, layout: SlotLayout):
        topology = MirrorTopology(
            group=None, rank=index, on_head=index == 0, remote_ranks=frozenset({1})
        )
        try:
            outcomes[index] = attach_region_mirror(
                _inner(), region, HEAD, layout, topology
            )
        except Exception as e:
            outcomes[index] = e

    threads = [
        threading.Thread(target=rank, args=(0, head, HEAD)),
        threading.Thread(target=rank, args=(1, remote, LOCAL)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return outcomes


def test_attach_connects_remote_ranks_through_the_head(monkeypatch, regions):
    outcomes = _run_ranks(monkeypatch, regions)

    head_worker, remote_worker = outcomes[0], outcomes[1]
    assert isinstance(head_worker, RegionMirrorWorker)
    assert isinstance(remote_worker, RegionMirrorWorker)
    assert remote_worker._client is not None
    remote_worker.shutdown()
    head_worker.shutdown()


def test_attach_fails_on_every_rank_when_the_head_cannot_serve(monkeypatch, regions):
    outcomes = _run_ranks(monkeypatch, regions, server_error=OSError("bind failed"))

    assert isinstance(outcomes[0], OSError)
    assert isinstance(outcomes[1], RuntimeError)
    assert "mirror server" in str(outcomes[1])


@pytest.mark.parametrize("rank, failing_exchange", [(0, 1), (1, 2)])
def test_attach_releases_mirror_endpoints_when_an_exchange_fails(
    monkeypatch, rank, failing_exchange
):
    """A peer that fails during a later exchange must not leak the server's
    port and region view, or the client's sockets."""
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setattr("vllm.utils.network_utils.get_ip", lambda: "127.0.0.1")
    server, client = MagicMock(address="tcp://127.0.0.1:1"), MagicMock()
    monkeypatch.setattr(region_mirror, "RegionMirrorServer", lambda *a, **k: server)
    monkeypatch.setattr(region_mirror, "RegionMirrorClient", lambda *a, **k: client)
    replies = [
        [(None, "127.0.0.1")] * 2,
        [(None, (server.address, TOKEN)), (None, None)],
    ]

    def all_gather(topology, value):
        if len(replies) == 2 - failing_exchange:
            raise RuntimeError("peer lost")
        return replies.pop(0)

    monkeypatch.setattr(region_mirror, "_all_gather", all_gather)
    topology = MirrorTopology(
        group=None, rank=rank, on_head=rank == 0, remote_ranks=frozenset({1})
    )

    with pytest.raises(RuntimeError, match="peer lost"):
        attach_region_mirror(_inner(), MagicMock(), HEAD, LOCAL, topology)
    (server if rank == 0 else client).shutdown.assert_called_once()


def test_wait_does_not_wait_for_pushes():
    gate = threading.Event()
    client = MagicMock(chunks_per_request=64, stopped=False)
    client.push.side_effect = lambda chunk_ids: gate.wait(10)
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    worker.submit_store(1, MagicMock(), CPULoadStoreSpec([2]))
    inner.get_finished.return_value = [TransferResult(job_id=1, success=True)]
    assert worker.get_finished() == []  # the push is now in flight

    start = time.monotonic()
    worker.wait({1})

    assert time.monotonic() - start < 5
    inner.wait.assert_called_once_with({1})
    gate.set()
    worker.shutdown()


def test_loads_overtake_a_store_between_its_requests():
    order: list[str] = []
    started = threading.Event()
    gate = threading.Event()

    def push(chunk_ids):
        order.append("store")
        started.set()
        gate.wait(10)

    client = MagicMock(chunks_per_request=1, stopped=False)
    client.push.side_effect = push
    client.pull.side_effect = lambda chunk_ids: order.append("load")
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    worker.submit_store(0, MagicMock(), CPULoadStoreSpec([0, 1, 2]))
    inner.get_finished.return_value = [TransferResult(job_id=0, success=True)]
    worker.get_finished()
    inner.get_finished.return_value = []
    assert started.wait(10)
    worker.submit_load(1, CPULoadStoreSpec([3]), MagicMock())

    gate.set()
    _wait_for(lambda: len(order) == 4)
    worker.shutdown()

    assert order == ["store", "load", "store", "store"]


def test_waiting_stores_run_between_bursts_of_loads():
    order: list[str] = []
    gate = threading.Event()

    def pull(chunk_ids):
        gate.wait(10)
        order.append("load")

    client = MagicMock(chunks_per_request=64, stopped=False)
    client.pull.side_effect = pull
    client.push.side_effect = lambda chunk_ids: order.append("store")
    inner = _inner()
    worker = RegionMirrorWorker(inner, client=client)
    worker.submit_store(0, MagicMock(), CPULoadStoreSpec([0]))
    for job_id in range(1, 8):
        worker.submit_load(job_id, CPULoadStoreSpec([job_id % NUM_CHUNKS]), MagicMock())
    inner.get_finished.return_value = [TransferResult(job_id=0, success=True)]
    worker.get_finished()
    inner.get_finished.return_value = []

    gate.set()
    _wait_for(lambda: len(order) == 8)
    worker.shutdown()

    # The first load was already running when the store was queued.
    assert order == ["load"] * 4 + ["store"] + ["load"] * 3


def test_slots_larger_than_the_request_limit_still_move(regions):
    head, remote = regions
    server = RegionMirrorServer(
        head,
        HEAD,
        "127.0.0.1",
        TOKEN,
        REMOTES,
        allowed_hosts=ALLOWED,
        max_request_bytes=SLOT // 4,
    )
    client = RegionMirrorClient(
        remote, LOCAL, REMOTE_RANK, server.address, TOKEN, max_request_bytes=SLOT // 4
    )
    head_rows, local_rows = HEAD.rows(head), LOCAL.rows(remote)
    try:
        local_rows[:] = 12
        client.push(np.arange(NUM_CHUNKS))
    finally:
        client.shutdown()
        server.shutdown()

    assert (head_rows[:, REMOTE_RANK * SLOT :] == 12).all()
    del head_rows, local_rows


def test_server_rejects_stale_puts_until_a_new_epoch(mirror):
    # Within an epoch an older put is a delayed retry; a restarted client
    # starts a new epoch and its own sequence.
    head_rows, _, _, server = mirror
    put = [_header(seq=5, epoch="old"), b"\1" * SLOT]
    assert _raw_request(server.address, put)[-1] == b"ok"
    stale = [_header(seq=4, epoch="old"), b"\2" * SLOT]
    assert _raw_request(server.address, stale)[-1] == b"error"

    restarted = [_header(seq=0, epoch="new"), b"\3" * SLOT]

    assert _raw_request(server.address, restarted)[-1] == b"ok"
    assert (head_rows[0, REMOTE_RANK * SLOT :] == 3).all()


@pytest.mark.parametrize("old_client_put_first", [True, False])
def test_server_rejects_late_puts_of_a_replaced_client(mirror, old_client_put_first):
    # A put of the replaced client that arrives after the new client's must
    # not overwrite a chunk the new client now owns.
    head_rows, _, _, server = mirror
    if old_client_put_first:
        first = [_header(seq=5, epoch="old"), b"\1" * SLOT]
    else:
        # The replaced client only completed its handshake.
        first = [_header(op="ping", epoch="old", chunks=[])]
    assert _raw_request(server.address, first)[-1] == b"ok"
    new = [_header(op="ping", epoch="new", chunks=[])]
    assert _raw_request(server.address, new)[-1] == b"ok"
    put = [_header(seq=0, epoch="new"), b"\3" * SLOT]
    assert _raw_request(server.address, put)[-1] == b"ok"

    late = [_header(seq=6, epoch="old"), b"\2" * SLOT]

    assert _raw_request(server.address, late)[-1] == b"error"
    assert (head_rows[0, REMOTE_RANK * SLOT :] == 3).all()
    # The new client's sequence is unaffected.
    put = [_header(seq=1, epoch="new"), b"\4" * SLOT]
    assert _raw_request(server.address, put)[-1] == b"ok"


@pytest.mark.skipif(sys.platform != "linux", reason="binds a second loopback address")
def test_server_admits_only_the_remote_workers_hosts(mirror):
    # A host outside the allowlist cannot even queue a valid request.
    _, _, _, server = mirror
    ping = [_header(op="ping", chunks=[])]

    assert _raw_request(server.address, ping, source="127.0.0.2", wait_ms=1000) is None
    reply = _raw_request(server.address, ping, source="127.0.0.1")
    assert reply is not None and reply[-1] == b"ok"


@pytest.mark.parametrize(
    "host_ip, match", [("0.0.0.0", "not a host address"), (None, "VLLM_HOST_IP")]
)
def test_attach_requires_a_host_address(monkeypatch, regions, host_ip, match):
    outcomes = _run_ranks(monkeypatch, regions, host_ip=host_ip)

    for outcome in outcomes.values():
        assert isinstance(outcome, RuntimeError)
    assert match in str(outcomes[0])
