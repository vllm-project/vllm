# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PDBackendAsync (async PD sender/receiver).

No NIXL, CUDA, or real ZMQ peers required — all I/O is mocked with
asyncio.sleep stubs so tests run fast (< 1 s total) in CI.  Assertions
focus on timing and call ordering; data integrity is covered separately
by the NIXL integration tests.
"""

# Standard
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import itertools
import threading
import time

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.pd_backend import AllocRequest as SyncAllocRequest
from lmcache.v1.storage_backend.pd_backend import AllocResponse as SyncAllocResponse
from lmcache.v1.storage_backend.pd_backend import PDMsg as SyncPDMsg
from lmcache.v1.storage_backend.pd_backend_async import (
    AllocRequest as AsyncAllocRequest,
)
from lmcache.v1.storage_backend.pd_backend_async import (
    AllocResponse as AsyncAllocResponse,
)
from lmcache.v1.storage_backend.pd_backend_async import (
    CancelNotif,
    PDBackendAsync,
)
from lmcache.v1.storage_backend.pd_backend_async import PDMsg as AsyncPDMsg
from lmcache.v1.storage_backend.pd_backend_async import (
    ProxyNotif,
)

TRANSFER_DELAY = 0.15
NONBLOCKING_THRESHOLD_RATIO = 0.25
CI_SERIAL_TIMEOUT_MARGIN = 3
_DEFAULT_SHAPE = [4, 2, 16, 8, 128]


def _make_key(i: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test",
        world_size=1,
        worker_id=0,
        chunk_hash=i,
        dtype=torch.bfloat16,
    )


def _make_mem_obj(idx: int = 0) -> MemoryObj:
    obj = MagicMock(spec=MemoryObj)
    obj.meta = SimpleNamespace(
        address=idx,
        fmt=MemoryFormat.KV_2LTD,
        shape=torch.Size(_DEFAULT_SHAPE),
        dtype=torch.bfloat16,
    )
    obj._ref_count = 1

    def _ref_up():
        obj._ref_count += 1

    def _ref_down():
        obj._ref_count -= 1

    def _get_ref_count():
        return obj._ref_count

    obj.ref_count_up.side_effect = _ref_up
    obj.ref_count_down.side_effect = _ref_down
    obj.get_ref_count.side_effect = _get_ref_count
    return obj


def _make_transfer_spec(
    receiver_host="127.0.0.1",
    init_port=9100,
    alloc_port=9101,
    req_id="req-0",
    is_last_prefill=True,
    num_transferred_tokens=0,
    total_chunks=0,
):
    return SimpleNamespace(
        receiver_host=receiver_host,
        receiver_init_port=[init_port],
        receiver_alloc_port=[alloc_port],
        req_id=req_id,
        is_last_prefill=is_last_prefill,
        num_transferred_tokens=num_transferred_tokens,
        total_chunks=total_chunks,
    )


def _make_alloc_req(
    keys,
    last_chunk_toks=16,
    req_id="",
    is_last_batch=False,
    shape=None,
    total_chunks=0,
):
    return AsyncAllocRequest(
        keys=[k.to_string() for k in keys],
        fmt=MemoryFormat.KV_2LTD.value,
        shape=list(shape or _DEFAULT_SHAPE),
        dtype="bfloat16",
        last_chunk_toks=last_chunk_toks,
        req_id=req_id,
        is_last_batch=is_last_batch,
        total_chunks=total_chunks,
    )


def _auto_alloc():
    """Allocator that returns a fresh MemoryObj stub on every call."""
    c = itertools.count()

    def alloc(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        return _make_mem_obj(idx=next(c))

    return alloc


def _pd_backend_patches():
    return (
        patch(
            "lmcache.v1.storage_backend.pd_backend_async.get_zmq_context",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend_async.CreateTransferChannel",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.storage_backend.pd_backend_async.get_correct_device",
            return_value="cpu",
        ),
    )


@contextmanager
def _patched_pd():
    p1, p2, p3 = _pd_backend_patches()
    with p1, p2, p3:
        yield


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def async_sender():
    p1, p2, p3 = _pd_backend_patches()

    with p1, p2 as mock_create_tc, p3:
        alloc_socket = MagicMock()
        alloc_response = AsyncAllocResponse(remote_indexes=[0])
        alloc_socket.recv_multipart = AsyncMock(
            return_value=[b"", msgspec.msgpack.encode(alloc_response)]
        )
        alloc_socket.send_multipart = AsyncMock()

        tc = MagicMock()

        async def _slow_write(*a, **kw):
            await asyncio.sleep(TRANSFER_DELAY)
            return 1

        tc.async_batched_write = _slow_write
        mock_create_tc.return_value = tc

        # First Party
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.metadata import LMCacheMetadata

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="sender",
            pd_proxy_host="127.0.0.1",
            pd_proxy_port=5555,
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
        )
        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 16, 8, 128),
        )
        backend = PDBackendAsync(config, metadata)
        # Override async proxy socket with a mock so tests don't need a real ZMQ
        # connection.  _init_sender() already ran on the sender loop; we replace
        # the socket here before any tests send ProxyNotifs.
        backend._async_proxy_socket = AsyncMock()

        receiver_id = "127.0.0.1" + str(9100)
        backend.initialized_peers.add(receiver_id)
        backend._async_alloc_sockets[receiver_id] = alloc_socket

        yield backend
        backend.close()


@pytest.fixture
def async_receiver():
    p1, p2, p3 = _pd_backend_patches()

    with p1, p2, p3:
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.metadata import LMCacheMetadata

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="receiver",
            pd_peer_host="127.0.0.1",
            pd_peer_init_port=[9200],
            pd_peer_alloc_port=[9201],
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
        )
        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 16, 8, 128),
        )
        backend = PDBackendAsync(config, metadata)
        yield backend
        for mem_obj in backend.data.values():
            try:
                mem_obj.ref_count_down()
            except Exception:
                pass
        backend.close()


# ── sender tests ──────────────────────────────────────────────────────────


def test_sender_nonblocking_fifo_transfers(async_sender):
    """batched_submit_put_task returns immediately; all requests complete."""
    N = 4
    done_events = [threading.Event() for _ in range(N)]

    def make_cb(i):
        def cb(key):
            done_events[i].set()

        return cb

    t0 = time.monotonic()
    for i in range(N):
        async_sender.batched_submit_put_task(
            [_make_key(i)],
            [_make_mem_obj(i)],
            transfer_spec=_make_transfer_spec(req_id=f"req-{i}"),
            on_complete_callback=make_cb(i),
        )
    enqueue_elapsed = time.monotonic() - t0

    assert enqueue_elapsed < TRANSFER_DELAY * NONBLOCKING_THRESHOLD_RATIO

    timeout = TRANSFER_DELAY * N * CI_SERIAL_TIMEOUT_MARGIN
    for i, ev in enumerate(done_events):
        assert ev.wait(timeout=timeout), f"req-{i} did not complete"


def test_sender_flow_control_backpressure(async_sender):
    """allocate() blocks when staging buffer is physically full, unblocks on notify."""
    sentinel = _make_mem_obj(idx=77)
    should_block = [True]

    def alloc_fn(*args, **kw):
        if should_block[0]:
            return None
        return sentinel

    async_sender.memory_allocator.allocate = alloc_fn

    result = []
    blocked = threading.Event()
    unblocked = threading.Event()

    def worker():
        blocked.set()
        result.append(async_sender.allocate(torch.Size(_DEFAULT_SHAPE), torch.bfloat16))
        unblocked.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert blocked.wait(timeout=2.0)
    time.sleep(0.1)
    assert not unblocked.is_set(), "allocate() should be blocked"

    # Make allocator return sentinel and wake the condition.
    should_block[0] = False
    async_sender._notify_staging_freed()
    assert unblocked.wait(timeout=2.0)
    assert result[0] is sentinel
    t.join(timeout=1.0)


def test_sender_chunk_ordering(async_sender):
    """Last prefill chunk waits for prior slow chunk before sending ProxyNotif."""
    SLOW, FAST = 0.30, 0.05
    REQ_ID = "req-chunked"

    # total_chunks=2 is passed via transfer_spec so the sender loop initializes
    # per-request tracking internally (try_admit is not called from test code
    # because it belongs to the sender loop, not caller threads).

    call_count = 0
    call_lock = threading.Lock()

    async def controlled_write(*a, **kw):
        nonlocal call_count
        with call_lock:
            call_count += 1
            idx = call_count
        await asyncio.sleep(SLOW if idx == 1 else FAST)
        return 1

    async_sender.transfer_channel.async_batched_write = controlled_write

    notify_times = []
    sent_data = []

    async def record_send(data):
        notify_times.append(time.monotonic())
        sent_data.append(data)

    async_sender._async_proxy_socket.send = record_send

    async_sender.batched_submit_put_task(
        [_make_key(0)],
        [_make_mem_obj(0)],
        transfer_spec=_make_transfer_spec(
            req_id=REQ_ID, is_last_prefill=False, total_chunks=2
        ),
    )
    time.sleep(0.01)

    done = threading.Event()
    async_sender.batched_submit_put_task(
        [_make_key(1)],
        [_make_mem_obj(1)],
        transfer_spec=_make_transfer_spec(
            req_id=REQ_ID, is_last_prefill=True, total_chunks=2
        ),
        on_complete_callback=lambda k: done.set(),
    )
    t_submit = time.monotonic()

    assert done.wait(timeout=SLOW * 3)

    # Wait for ProxyNotif to be sent (fires after the slow batch 1 completes).
    timeout_end = time.monotonic() + SLOW * 2
    while not notify_times and time.monotonic() < timeout_end:
        time.sleep(0.01)

    assert len(notify_times) == 1

    notif = msgspec.msgpack.decode(sent_data[0], type=AsyncPDMsg)
    assert isinstance(notif, ProxyNotif) and notif.req_id == REQ_ID

    elapsed = notify_times[0] - t_submit
    assert elapsed >= SLOW * 0.8, (
        f"ProxyNotif too early ({elapsed:.3f}s) — fast chunk didn't wait for slow"
    )


def test_sender_per_receiver_concurrency(async_sender):
    """Different-receiver requests run concurrently."""
    SLOW, FAST = 0.25, 0.05

    recv1_id = "127.0.0.1" + str(9100)
    recv2_id = "127.0.0.1" + str(9200)

    sock2 = MagicMock()
    sock2.recv_multipart = AsyncMock(
        return_value=[
            b"",
            msgspec.msgpack.encode(AsyncAllocResponse(remote_indexes=[0])),
        ]
    )
    sock2.send_multipart = AsyncMock()
    async_sender.initialized_peers.add(recv2_id)
    async_sender._async_alloc_sockets[recv2_id] = sock2

    delays = {recv1_id: SLOW, recv2_id: FAST}
    orig_transfer = async_sender._async_transfer_task

    async def patched_transfer(**kw):
        rid = kw.get("receiver_id", "")
        await asyncio.sleep(delays.get(rid, FAST))
        cb = kw.get("on_complete_callback")
        for key in kw.get("keys", []):
            if cb:
                try:
                    cb(key)
                except Exception:
                    pass
        async_sender._notify_staging_freed()

    async_sender._async_transfer_task = patched_transfer

    events = {"A": threading.Event(), "B": threading.Event(), "C": threading.Event()}
    times = {}
    lock = threading.Lock()

    def cb(name):
        def f(key):
            with lock:
                times[name] = time.monotonic()
            events[name].set()

        return f

    async_sender.batched_submit_put_task(
        [_make_key(10)],
        [_make_mem_obj(10)],
        transfer_spec=_make_transfer_spec(
            init_port=9100, alloc_port=9101, req_id="A", is_last_prefill=True
        ),
        on_complete_callback=cb("A"),
    )
    async_sender.batched_submit_put_task(
        [_make_key(20)],
        [_make_mem_obj(20)],
        transfer_spec=_make_transfer_spec(
            init_port=9200, alloc_port=9201, req_id="B", is_last_prefill=True
        ),
        on_complete_callback=cb("B"),
    )
    async_sender.batched_submit_put_task(
        [_make_key(30)],
        [_make_mem_obj(30)],
        transfer_spec=_make_transfer_spec(
            init_port=9100, alloc_port=9101, req_id="C", is_last_prefill=True
        ),
        on_complete_callback=cb("C"),
    )

    for name, ev in events.items():
        assert ev.wait(timeout=SLOW * 6), f"{name} timed out"

    assert times["B"] < times["A"], "B (fast recv2) should finish before A (slow recv1)"

    async_sender._async_transfer_task = orig_transfer


# ── receiver tests ────────────────────────────────────────────────────────


def test_receiver_nonblocking_async_sleep(async_receiver):
    """Busy-wait retries yield via asyncio.sleep, not time.sleep."""
    RETRY_COUNT = 5
    TOKS_A, TOKS_B = 16, 8

    key_a, key_b = _make_key(100), _make_key(200)
    obj_a, obj_b = _make_mem_obj(idx=10), _make_mem_obj(idx=20)

    finish_order = []
    orig_put = async_receiver.put

    def tracked_put(key, mem_obj):
        if key == key_a:
            finish_order.append("a")
        elif key == key_b:
            finish_order.append("b")
        return orig_put(key, mem_obj)

    async_receiver.put = tracked_put

    calls = {}

    def patched_alloc(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        tok_dim = MemoryFormat.KV_2LTD.token_dim()
        toks = shapes[tok_dim] if isinstance(shapes, torch.Size) else shapes[tok_dim]
        calls[toks] = calls.get(toks, 0) + 1
        if toks == TOKS_A and calls[toks] <= RETRY_COUNT:
            return None
        return obj_a if toks == TOKS_A else obj_b

    async_receiver.allocate = patched_alloc

    req_a = _make_alloc_req(
        [key_a], last_chunk_toks=TOKS_A, shape=[4, 2, TOKS_A, 8, 128]
    )
    req_b = _make_alloc_req(
        [key_b], last_chunk_toks=TOKS_B, shape=[4, 2, TOKS_B, 8, 128]
    )

    async def _run():
        await asyncio.gather(
            async_receiver._async_allocate_and_put(req_a),
            async_receiver._async_allocate_and_put(req_b),
        )

    asyncio.run(_run())

    assert finish_order == ["b", "a"], (
        f"Got {finish_order}, busy-wait likely uses time.sleep instead of asyncio.sleep"
    )


def test_receiver_flow_control_inflight(async_receiver):
    """Allocation blocks when memory is full, unblocks when remove() frees a chunk."""
    mem_obj = _make_mem_obj(idx=60)
    freed = [False]  # Controls when allocate succeeds

    def patched_alloc(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        return mem_obj if freed[0] else None

    async_receiver.allocate = patched_alloc

    alloc_req = _make_alloc_req([_make_key(600)])

    # Wait longer than several poll intervals so condition-timeout retries
    # still return None (freed[0] is still False).
    block_duration = async_receiver._condition_poll_interval * 5

    async def run():
        completed = asyncio.Event()
        holder = []

        async def do_alloc():
            holder.append(await async_receiver._async_allocate_and_put(alloc_req))
            completed.set()

        async def free_later():
            await asyncio.sleep(block_duration)
            assert not completed.is_set(), "should still be blocked"
            freed[0] = True
            # Notify the alloc freed condition to wake the blocked coroutine.
            async with async_receiver._alloc_freed_condition:
                async_receiver._alloc_freed_condition.notify_all()

        await asyncio.gather(do_alloc(), free_later())
        assert holder[0].remote_indexes == [mem_obj.meta.address]

    asyncio.run(run())


def test_receiver_last_chunk_shape_override(async_receiver):
    """Last chunk's token dim is overridden to last_chunk_toks."""
    mem_obj = _make_mem_obj(idx=30)
    LAST_TOKS = 7

    shapes_seen = []

    def tracking_alloc(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        shapes_seen.append(shapes)
        return mem_obj

    async_receiver.allocate = tracking_alloc

    keys = [_make_key(300), _make_key(301), _make_key(302)]
    asyncio.run(
        async_receiver._async_allocate_and_put(
            _make_alloc_req(keys, last_chunk_toks=LAST_TOKS)
        )
    )

    assert len(shapes_seen) == 3
    tok_dim = MemoryFormat.KV_2LTD.token_dim()
    assert shapes_seen[-1][tok_dim] == LAST_TOKS


@pytest.mark.parametrize(
    "total_declared, b1_n, b2_n",
    [(5, 5, 1), (4, 3, 2)],
    ids=["exact-then-overflow", "partial-then-overflow"],
)
def test_receiver_fail_fast_overflow(async_receiver, total_declared, b1_n, b2_n):
    """
    Cumulative chunks > declared total_chunks → RuntimeError;
    all keys rolled back.
    """
    async_receiver.allocate = _auto_alloc()
    req_id = "req-failfast"

    b1_keys = [_make_key(i) for i in range(b1_n)]

    async def run():
        r1 = await async_receiver._async_allocate_and_put(
            _make_alloc_req(b1_keys, req_id=req_id, total_chunks=total_declared)
        )
        assert -1 not in r1.remote_indexes and len(r1.remote_indexes) == b1_n

        with pytest.raises(RuntimeError, match="total_chunks"):
            await async_receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_key(5000 + i) for i in range(b2_n)],
                    req_id=req_id,
                    total_chunks=total_declared,
                )
            )

    asyncio.run(run())

    # Fail-fast path rolls back prior batch keys too.
    for k in b1_keys:
        assert not async_receiver.contains(k, pin=False)

    # unrelated req_id should work fine
    async def check_other():
        r = await async_receiver._async_allocate_and_put(
            _make_alloc_req([_make_key(20000)], req_id="req-other", total_chunks=1)
        )
        assert -1 not in r.remote_indexes

    asyncio.run(check_other())


def test_receiver_alloc_timeout(async_receiver):
    """
    allocate() returning None past deadline → RuntimeError;
    prior batches rolled back.
    """
    req_id = "req-timeout"
    async_receiver.allocate = _auto_alloc()

    b1_keys = [_make_key(1000 + i) for i in range(3)]
    r1 = asyncio.run(
        async_receiver._async_allocate_and_put(
            _make_alloc_req(b1_keys, req_id=req_id, total_chunks=6)
        )
    )
    assert -1 not in r1.remote_indexes

    # second batch: first key ok, rest always None
    n = [0]

    def fail_after_first(shapes, dtype, fmt=MemoryFormat.KV_2LTD, **kw):
        n[0] += 1
        return _make_mem_obj(idx=999) if n[0] == 1 else None

    async_receiver.allocate = fail_after_first
    async_receiver._allocation_timeout = 0.05

    async def run():
        with pytest.raises(RuntimeError, match="timeout"):
            await async_receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_key(2000 + i) for i in range(3)],
                    req_id=req_id,
                    total_chunks=6,
                )
            )

    asyncio.run(run())

    # New design rolls back prior batches on failure.
    for k in b1_keys:
        assert not async_receiver.contains(k, pin=False)


def test_receiver_is_last_batch_cleanup(async_receiver):
    """is_last_batch=True removes req_id from _req_allocated_keys."""
    req_id = "req-lifecycle"
    async_receiver.allocate = _auto_alloc()

    asyncio.run(
        async_receiver._async_allocate_and_put(
            _make_alloc_req(
                [_make_key(3000 + i) for i in range(3)],
                req_id=req_id,
                is_last_batch=False,
                total_chunks=5,
            )
        )
    )
    assert req_id in async_receiver._req_allocated_keys
    assert len(async_receiver._req_allocated_keys[req_id]) == 3

    asyncio.run(
        async_receiver._async_allocate_and_put(
            _make_alloc_req(
                [_make_key(4000 + i) for i in range(2)],
                req_id=req_id,
                is_last_batch=True,
                total_chunks=5,
            )
        )
    )
    assert req_id not in async_receiver._req_allocated_keys


def test_receiver_admission_control(async_receiver):
    """Reservation-based admission: concurrent requests can proceed."""
    async_receiver.allocate = _auto_alloc()

    log = []

    async def run():
        log.append("A1-start")
        await async_receiver._async_allocate_and_put(
            _make_alloc_req(
                [_make_key(7000 + i) for i in range(2)],
                req_id="req-A",
                is_last_batch=False,
                total_chunks=3,
            )
        )
        log.append("A1-done")

        async def do_b():
            log.append("B-start")
            await async_receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_key(8000)],
                    req_id="req-B",
                    is_last_batch=True,
                    total_chunks=1,
                )
            )
            log.append("B-done")

        async def do_a2():
            await asyncio.sleep(0.02)
            log.append("A2-start")
            await async_receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_key(9000)],
                    req_id="req-A",
                    is_last_batch=True,
                    total_chunks=3,
                )
            )
            log.append("A2-done")

        await asyncio.gather(do_b(), do_a2())

    asyncio.run(run())

    # With reservation-based admission, A2 and B can proceed concurrently.
    # All should complete successfully.
    assert "A1-done" in log
    assert "A2-done" in log
    assert "B-done" in log


def test_receiver_error_response(async_receiver):
    """_handle_alloc_request sends error AllocResponse when allocation fails."""
    payload = msgspec.msgpack.encode(_make_alloc_req([_make_key(0)], req_id="req-err"))
    identity = b"fake-sender"

    frames_sent = []
    sock = MagicMock()
    sock.send_multipart = AsyncMock(side_effect=lambda f: frames_sent.append(f))

    orig = async_receiver._async_allocate_and_put

    async def failing(req):
        raise RuntimeError("boom")

    async_receiver._async_allocate_and_put = failing
    asyncio.run(async_receiver._handle_alloc_request(sock, identity, payload))
    async_receiver._async_allocate_and_put = orig

    assert len(frames_sent) == 1
    f = frames_sent[0]
    assert f[0] == identity and f[1] == b""
    resp = msgspec.msgpack.decode(f[2], type=AsyncPDMsg)
    assert isinstance(resp, AsyncAllocResponse) and resp.remote_indexes == [-1]


# ── close() ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("role", ["sender", "receiver"])
def test_close_stops_thread(role, async_sender, async_receiver):
    """close() stops the background event-loop thread."""
    backend = async_sender if role == "sender" else async_receiver
    attr = "_sender_thread" if role == "sender" else "_recv_thread"
    assert getattr(backend, attr).is_alive()
    backend.close()
    assert not getattr(backend, attr).is_alive()
    backend.running = False


# ── pd_max_prefill_len init check ────────────────────────────────────────


def test_pd_max_prefill_len_check():
    """pd_max_prefill_len > buffer capacity → ValueError on init."""
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata

    def recv_cfg(max_len):
        return LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="receiver",
            pd_peer_host="127.0.0.1",
            pd_peer_init_port=[9200],
            pd_peer_alloc_port=[9201],
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
            pd_max_prefill_len=max_len,
        )

    def send_cfg(max_len):
        return LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="sender",
            pd_proxy_host="127.0.0.1",
            pd_proxy_port=5555,
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
            pd_max_prefill_len=max_len,
        )

    meta = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 16, 8, 128),
    )

    with _patched_pd():
        with pytest.raises(ValueError, match="pd_max_prefill_len"):
            PDBackendAsync(recv_cfg(5000), meta)

    with _patched_pd():
        with pytest.raises(ValueError, match="pd_max_prefill_len"):
            PDBackendAsync(send_cfg(5000), meta)

    with _patched_pd():
        PDBackendAsync(recv_cfg(4096), meta).close()  # boundary ok

    with _patched_pd():
        PDBackendAsync(recv_cfg(0), meta).close()  # 0 skips check


# ── wire-format compatibility (sync ↔ async) ─────────────────────────────


def test_sync_request_decoded_as_async():
    req = SyncAllocRequest(
        keys=["k0", "k1"],
        fmt=0,
        shape=[4, 2, 16, 8, 128],
        dtype="bfloat16",
        last_chunk_toks=7,
    )
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=AsyncPDMsg)
    assert isinstance(decoded, AsyncAllocRequest)
    assert (
        decoded.keys == ["k0", "k1"]
        and decoded.req_id == ""
        and not decoded.is_last_batch
    )


def test_async_response_decoded_as_sync():
    resp = AsyncAllocResponse(remote_indexes=[100, 200])
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(resp), type=SyncPDMsg)
    assert isinstance(decoded, SyncAllocResponse)
    assert decoded.already_sent_indexes == [] and decoded.remote_indexes == [100, 200]


def test_sync_response_decoded_as_async():
    resp = SyncAllocResponse(already_sent_indexes=[0], remote_indexes=[100])
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(resp), type=AsyncPDMsg)
    assert isinstance(decoded, AsyncAllocResponse)
    assert decoded.already_sent_indexes == [0] and decoded.remote_indexes == [100]


# ── new coverage tests ────────────────────────────────────────────────────


def test_receiver_reject_legacy_sender_zero_total_chunks(async_receiver):
    """req_id + total_chunks=0 → RuntimeError (legacy sender rejected).

    Verifies that a request with a req_id but total_chunks=0 is rejected
    immediately, since the new reservation-based design requires senders to
    declare total_chunks upfront.
    """
    async_receiver.allocate = _auto_alloc()
    req = _make_alloc_req([_make_key(0)], req_id="req-legacy", total_chunks=0)

    async def run():
        with pytest.raises(RuntimeError, match="total_chunks"):
            await async_receiver._async_allocate_and_put(req)

    asyncio.run(run())


def test_receiver_reservation_admission_timeout(async_receiver):
    """Admission times out when buffer is fully reserved by another request.

    Verifies that when the entire buffer is reserved by an in-progress request,
    a new request's admission attempt times out and raises RuntimeError.
    """
    async_receiver.allocate = _auto_alloc()
    # Set the reservation manager's own admission timeout to keep the test fast.
    async_receiver._recv_reservation_mgr._allocation_timeout = 0.05
    total = async_receiver._recv_reservation_mgr._total_chunks

    async def run():
        # Admit a request that reserves the entire buffer (not yet released).
        await async_receiver._async_allocate_and_put(
            _make_alloc_req(
                [_make_key(i) for i in range(total)],
                req_id="req-hog",
                total_chunks=total,
                is_last_batch=False,
            )
        )

        # Second request cannot be admitted → timeout.
        with pytest.raises(RuntimeError, match="timed out"):
            await async_receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_key(9000)],
                    req_id="req-blocked",
                    total_chunks=1,
                )
            )

    asyncio.run(run())


def test_receiver_admission_fails_fast_when_request_exceeds_buffer(async_receiver):
    """A request larger than the whole buffer is rejected immediately.

    When a prefill's declared total_chunks exceeds the receiver buffer's total
    capacity it can never be admitted, no matter how much the buffer drains.
    The receiver must reject it right away with an actionable error instead of
    blocking for the full pd_allocation_timeout_sec and then reporting a
    misleading "over-subscribed" timeout that tears down the transfer.
    """
    async_receiver.allocate = _auto_alloc()
    total = async_receiver._recv_reservation_mgr.get_total_chunks()
    oversized = total + 1

    # A large admission timeout would be burned if the fail-fast were missing;
    # the guard must return well before it could elapse.
    async_receiver._recv_reservation_mgr._allocation_timeout = 30.0

    async def run():
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="buffer too small"):
            await async_receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_key(9999)],
                    req_id="req-too-big",
                    total_chunks=oversized,
                )
            )
        elapsed = time.monotonic() - t0
        # Fail fast: nowhere near the 30s admission timeout.
        assert elapsed < 1.0
        # The rejected request must not leave a reservation behind.
        mgr = async_receiver._recv_reservation_mgr
        assert "req-too-big" not in mgr._reservations
        assert mgr._total_reserved == 0

    asyncio.run(run())


def test_receiver_cancel_notif_releases_keys_and_reservation(async_receiver):
    """CancelNotif removes keys, releases reservation, and cleans up tracking.

    Verifies that when a CancelNotif arrives via _handle_alloc_request, all
    previously allocated keys are removed, the per-request tracking entry is
    deleted, and the reservation is fully released.
    """
    async_receiver.allocate = _auto_alloc()
    req_id = "req-cancel"
    keys = [_make_key(5000 + i) for i in range(3)]

    async def run():
        # Allocate first batch (not last).
        await async_receiver._async_allocate_and_put(
            _make_alloc_req(keys, req_id=req_id, total_chunks=5, is_last_batch=False)
        )
        assert req_id in async_receiver._req_allocated_keys
        assert async_receiver._recv_reservation_mgr._total_reserved > 0

        # Send CancelNotif via _handle_alloc_request.
        cancel = CancelNotif(req_id=req_id, keys=[k.to_string() for k in keys])
        payload = msgspec.msgpack.encode(cancel)
        sock = MagicMock()
        sock.send_multipart = AsyncMock()
        await async_receiver._handle_alloc_request(sock, b"sender-1", payload)

        # Verify cleanup.
        for k in keys:
            assert not async_receiver.contains(k, pin=False)
        assert req_id not in async_receiver._req_allocated_keys
        assert async_receiver._recv_reservation_mgr._total_reserved == 0

    asyncio.run(run())


def test_receiver_no_req_id_skips_admission(async_receiver):
    """Empty req_id bypasses reservation admission and chunk accounting.

    Verifies that anonymous requests (req_id="") are allocated without
    touching the reservation manager, leaving total_reserved unchanged.
    """
    async_receiver.allocate = _auto_alloc()

    async def run():
        r = await async_receiver._async_allocate_and_put(
            _make_alloc_req([_make_key(0), _make_key(1)], req_id="", total_chunks=0)
        )
        assert len(r.remote_indexes) == 2
        assert -1 not in r.remote_indexes
        # No reservation should be held for anonymous requests.
        assert async_receiver._recv_reservation_mgr._total_reserved == 0

    asyncio.run(run())


def test_receiver_batch_failure_rolls_back_prior_batches(async_receiver):
    """Second batch allocation timeout → both batches rolled back.

    Verifies that when a batch fails mid-allocation, all previously allocated
    keys from earlier batches for the same request are also removed and the
    reservation is released, since the decoder needs all chunks to proceed.
    """
    async_receiver.allocate = _auto_alloc()
    req_id = "req-rollback"
    b1_keys = [_make_key(1000 + i) for i in range(3)]

    async def run():
        # First batch succeeds.
        r1 = await async_receiver._async_allocate_and_put(
            _make_alloc_req(b1_keys, req_id=req_id, total_chunks=6, is_last_batch=False)
        )
        assert -1 not in r1.remote_indexes
        for k in b1_keys:
            assert async_receiver.contains(k, pin=False)

        # Second batch: allocator always returns None → timeout.
        async_receiver.allocate = lambda *a, **kw: None
        async_receiver._allocation_timeout = 0.05

        with pytest.raises(RuntimeError, match="timeout"):
            await async_receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_key(2000 + i) for i in range(3)],
                    req_id=req_id,
                    total_chunks=6,
                    is_last_batch=True,
                )
            )

        # Prior batch keys must be rolled back too.
        for k in b1_keys:
            assert not async_receiver.contains(k, pin=False)
        assert req_id not in async_receiver._req_allocated_keys
        assert async_receiver._recv_reservation_mgr._total_reserved == 0

    asyncio.run(run())


def test_receiver_reservation_released_unblocks_waiting_request(async_receiver):
    """Releasing a reservation unblocks a concurrently waiting admission.

    Covers the notify_all() call in async_release_reservation.
    """
    async_receiver.allocate = _auto_alloc()
    total = async_receiver._recv_reservation_mgr._total_chunks

    async def run():
        # Fill all reservation with req-fill (not last batch, so not released yet).
        await async_receiver._async_allocate_and_put(
            _make_alloc_req(
                [_make_key(i) for i in range(total)],
                req_id="req-fill",
                total_chunks=total,
                is_last_batch=False,
            )
        )

        result: list = []

        async def blocked_req():
            r = await async_receiver._async_allocate_and_put(
                _make_alloc_req(
                    [_make_key(9000)],
                    req_id="req-wait",
                    total_chunks=1,
                )
            )
            result.append(r)

        async def release_later():
            await asyncio.sleep(0.05)
            # Release req-fill's reservation by sending the last batch.
            await async_receiver._async_allocate_and_put(
                _make_alloc_req(
                    [],
                    req_id="req-fill",
                    total_chunks=total,
                    is_last_batch=True,
                )
            )

        await asyncio.gather(blocked_req(), release_later())
        assert len(result) == 1
        assert -1 not in result[0].remote_indexes

    asyncio.run(run())


# ── ref-count call verification for already_sent_indexes (dedup path) ─────


def test_receiver_dedup_pins_existing_key_via_ref_count_up(async_receiver):
    """
    _async_allocate_and_put calls ref_count_up
    on pre-existing key (via contains(pin=True)).
    """
    existing_key = _make_key(11000)
    existing_obj = _make_mem_obj(idx=42)
    async_receiver.data[existing_key] = existing_obj

    new_key = _make_key(11001)
    new_obj = _make_mem_obj(idx=99)
    async_receiver.allocate = lambda *a, **kw: new_obj

    req = _make_alloc_req(
        [existing_key, new_key],
        req_id="req-dedup-pin",
        total_chunks=2,
        is_last_batch=True,
    )
    resp = asyncio.run(async_receiver._async_allocate_and_put(req))

    assert resp.already_sent_indexes == [0]
    assert len(resp.remote_indexes) == 1
    # The critical assertion: production code called ref_count_up exactly once
    # on the existing object (via contains(pin=True)).
    existing_obj.ref_count_up.assert_called_once()
    # New object should NOT have ref_count_up called by dedup logic.
    new_obj.ref_count_up.assert_not_called()


def test_receiver_dedup_multiple_existing_keys_each_pinned_once(async_receiver):
    """Each pre-existing key gets exactly one ref_count_up call."""
    keys = [_make_key(12000 + i) for i in range(4)]
    objs = [_make_mem_obj(idx=i) for i in range(4)]

    # Index 0, 2 pre-exist.
    async_receiver.data[keys[0]] = objs[0]
    async_receiver.data[keys[2]] = objs[2]

    alloc_counter = itertools.count(start=500)

    def counting_alloc(*a, **kw):
        return _make_mem_obj(idx=next(alloc_counter))

    async_receiver.allocate = counting_alloc

    req = _make_alloc_req(
        keys,
        req_id="req-multi-dedup",
        total_chunks=4,
        is_last_batch=True,
    )
    resp = asyncio.run(async_receiver._async_allocate_and_put(req))

    assert sorted(resp.already_sent_indexes) == [0, 2]
    assert len(resp.remote_indexes) == 2
    objs[0].ref_count_up.assert_called_once()
    objs[2].ref_count_up.assert_called_once()


def test_remove_calls_ref_count_down_and_conditional_delete(async_receiver):
    """remove() calls ref_count_down; deletes from data only when get_ref_count()==0."""
    key = _make_key(13000)
    obj = _make_mem_obj(idx=50)
    obj._ref_count = 3
    async_receiver.data[key] = obj

    # First remove: ref_count_down called, but get_ref_count() returns 2 → not deleted
    async_receiver.remove(key)
    assert obj.ref_count_down.call_count == 1
    assert key in async_receiver.data

    # Second remove
    async_receiver.remove(key)
    assert obj.ref_count_down.call_count == 2
    assert key in async_receiver.data

    # Third remove: get_ref_count() will return 0 → deleted
    async_receiver.remove(key)
    assert obj.ref_count_down.call_count == 3
    assert key not in async_receiver.data


def test_put_duplicate_calls_ref_count_down_on_new_obj(async_receiver):
    """Duplicate put() calls ref_count_down on the *new* (rejected) object."""
    key = _make_key(14000)
    first_obj = _make_mem_obj(idx=60)
    second_obj = _make_mem_obj(idx=61)

    async_receiver.put(key, first_obj)
    first_obj.ref_count_down.assert_not_called()

    async_receiver.put(key, second_obj)
    # Production code drops the new obj:
    second_obj.ref_count_down.assert_called_once()
    # Original untouched:
    first_obj.ref_count_down.assert_not_called()
    assert async_receiver.data[key] is first_obj


def test_sender_dedup_calls_ref_count_down_on_skipped_chunks():
    """Sender calls ref_count_down on dedup'd chunks without RDMA write."""
    p1, p2, p3 = _pd_backend_patches()
    with p1, p2 as mock_create_tc, p3:
        alloc_response = AsyncAllocResponse(
            remote_indexes=[100],
            already_sent_indexes=[0],
        )
        alloc_socket = MagicMock()
        alloc_socket.recv_multipart = AsyncMock(
            return_value=[b"", msgspec.msgpack.encode(alloc_response)]
        )
        alloc_socket.send_multipart = AsyncMock()

        tc = MagicMock()
        tc.async_batched_write = AsyncMock(return_value=1)
        mock_create_tc.return_value = tc

        # First Party
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.metadata import LMCacheMetadata

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=16,
            pd_role="sender",
            pd_proxy_host="127.0.0.1",
            pd_proxy_port=5555,
            pd_buffer_size=64 * 1024 * 1024,
            pd_buffer_device="cpu",
        )
        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(4, 2, 16, 8, 128),
        )
        sender = PDBackendAsync(config, metadata)
        sender._async_proxy_socket = AsyncMock()

        receiver_id = "127.0.0.1" + str(9100)
        sender.initialized_peers.add(receiver_id)
        sender._async_alloc_sockets[receiver_id] = alloc_socket

        mem_obj_0 = _make_mem_obj(idx=0)
        mem_obj_0._ref_count = 2  # post ref_count_up
        mem_obj_1 = _make_mem_obj(idx=1)
        mem_obj_1._ref_count = 2

        keys = [_make_key(15000), _make_key(15001)]

        async def run():
            await sender._async_transfer_task(
                keys=keys,
                memory_objs=[mem_obj_0, mem_obj_1],
                receiver_id=receiver_id,
                on_complete_callback=None,
                transfer_spec=_make_transfer_spec(
                    req_id="req-sender-dedup",
                    total_chunks=2,
                ),
            )

        asyncio.run(run())

        # mem_obj_0 (dedup'd): ref_count_down called exactly once (skip RDMA path)
        mem_obj_0.ref_count_down.assert_called_once()
        # mem_obj_1 (sent): ref_count_down called exactly once (after RDMA write)
        mem_obj_1.ref_count_down.assert_called_once()
        # Only mem_obj_1 was sent via RDMA
        tc.async_batched_write.assert_called_once()
        call_kw = tc.async_batched_write.call_args[1]
        assert call_kw["objects"] == [mem_obj_1]

        sender.close()


def test_dedup_pin_then_remove_verifies_call_sequence(async_receiver):
    """Full lifecycle: verify ref_count_up/down call counts through dedup + removes."""
    key = _make_key(16000)
    obj = _make_mem_obj(idx=70)
    async_receiver.data[key] = obj

    # Dedup pin via contains(pin=True)
    async_receiver.contains(key, pin=True)
    obj.ref_count_up.assert_called_once()

    # First remove
    async_receiver.remove(key)
    assert obj.ref_count_down.call_count == 1
    assert key in async_receiver.data  # ref_count=1 after mock side_effect

    # Second remove
    async_receiver.remove(key)
    assert obj.ref_count_down.call_count == 2
    assert key not in async_receiver.data  # ref_count=0 → deleted
