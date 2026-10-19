# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session completion, credit and cancellation contracts."""

import asyncio
import contextlib
import json
import os
import queue
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgspec
import pytest
import zmq.asyncio
from zmq.constants import SocketType

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import session_ipc as ipc
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import session_process
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.protocol import (
    CompressionAbort,
    CompressionAbortAck,
    CompressionChunkPull,
    CompressionChunkReady,
    CompressionSessionOpen,
    CompressionSessionOpened,
    MooncakeXferMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.session_process import (
    SessionProcess,
    _ProducerSession,
    _SendState,
    _SessionService,
)

_ENC = msgspec.msgpack.Encoder()
_REQ_DEC = msgspec.msgpack.Decoder(
    CompressionSessionOpen | CompressionChunkPull | CompressionAbort
)


def _make_init(needs_sender, needs_receiver, slots=2):
    return ipc.InitMessage(
        needs_sender=needs_sender,
        needs_receiver=needs_receiver,
        register_with_bootstrap=False,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        engine_id="A",
        dp_rank=0,
        hostname="127.0.0.1",
        rpc_port=1234,
        bootstrap_host="127.0.0.1",
        bootstrap_port=1,
        slot_addresses=[
            ipc.SlotAddresses(1000 + i * 100, 2000 + i * 100, 3000 + i * 100)
            for i in range(slots)
        ],
        kv_caches_base_addr=[4096],
        block_lens=[16],
        kv_block_lens=[16],
        registered_layer_names=["l0"],
        registered_layer_indices=[0],
        registered_group_indices=[0],
        fa_group_block_sizes={0: 16},
    )


def _open_message(profile="", **topology):
    fields = {"remote_tp_size": 1, "remote_tp_rank": 0, "remote_pp_size": 1, **topology}
    metadata = MooncakeXferMetadata(
        "peer",
        1,
        req_blocks={"d": ("t", [[1]])},
        kv_caches_base_addr=[1],
        block_lens=[16],
        kv_block_lens=[16],
        **fields,
    )
    return CompressionSessionOpen(metadata, "n", profile)


@pytest.mark.parametrize("layout_id", [0, 7])
def test_session_open_preserves_layout_id(layout_id):
    request = _open_message(layout_id=layout_id)
    restored = _REQ_DEC.decode(_ENC.encode(request))
    assert restored.metadata == request.metadata


@pytest.fixture
def make_service():
    def factory(sender=True, slots=2):
        service = _SessionService(None, _make_init(sender, not sender, slots))
        service._to_engine = lambda message: None
        service._sender_slots, service._receiver_slots = (
            asyncio.Queue(),
            asyncio.Queue(),
        )
        for slot in range(slots):
            service._sender_slots.put_nowait(slot)
            service._receiver_slots.put_nowait(slot)
        service._slot_payload, service._slot_aux, service._slot_tail = (
            [1, 2][:slots],
            [3, 4][:slots],
            [5, 6][:slots],
        )
        return service

    return factory


@pytest.fixture
def producer_session(make_service):
    def factory():
        service = make_service(slots=1)
        service._loop = asyncio.get_running_loop()
        service._abort_timeout = 0.1
        state = _SendState("t", p_req_id="p", ready=True)
        state.sending = state.need_send = 1
        service._send_meta["t"] = state
        session = _ProducerSession(
            identity=b"peer",
            transfer_id="t",
            d_req_id="d",
            nonce="n",
            remote_session="peer:1",
            local_block_ids=[[1]],
            valid_tokens_by_group={0: 16},
            logical_block_count=1,
            total_chunks=1,
            send_state=state,
        )
        service._producer_sessions[(b"peer", "t", "d")] = session
        return service, state, CompressionChunkPull("t", "d", 0, 1, 2, 3, "n")

    return factory


@pytest.fixture
def receiver_peer(make_service):
    context = zmq.asyncio.Context()
    router = context.socket(SocketType.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    service = make_service(sender=False)
    service._zmq = context
    yield service, router, f"tcp://127.0.0.1:{port}"
    router.close(linger=0)
    context.term()


def test_paired_helpers_complete_requests_and_release_processes():
    """Real spawn/Pipe/ZMQ/bootstrap exchange in both directions, without GPUs."""
    jobs = []
    lock = threading.Lock()

    def handler(job):
        with lock:
            jobs.append(job)
        if isinstance(job, ipc.OpenProducerJob):
            assert job.local_block_ids == [[1, 2, 3, 4]]
            assert job.metadata.req_blocks[job.d_req_id][1] == [[11, 12]]
            return ipc.JobDone(
                job.job_id,
                logical_block_count=2,
                total_chunks=4,
                local_block_ids=[[3, 4]],
                valid_tokens_by_group={0: 32},
            )
        if isinstance(job, (ipc.EncodeChunkJob, ipc.DecodeChunkJob)):
            first = 3 if isinstance(job, ipc.EncodeChunkJob) else 11
            assert job.local_block_ids == [[first + job.chunk_id // 2]]
            assert job.valid_tokens_by_group == {0: 16}
        return ipc.JobDone(job.job_id, u8_bytes=16, codec_bytes=16)

    processes = [
        SessionProcess(
            handler,
            _make_init(sender, not sender),
            queue.SimpleQueue(),
            queue.SimpleQueue(),
            queue.SimpleQueue(),
            _Stats(),
            -1,
        )
        for sender in (True, False)
    ]
    producer, consumer = processes
    started = []
    httpd = None
    thread = None
    try:
        for process in processes:
            started.append(process)
            process.start()

        class QueryHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(
                    {
                        "0": {
                            "engine_id": "A",
                            "worker_addr": {
                                "0": {
                                    "0": f"tcp://127.0.0.1:{producer.side_channel_port}"
                                }
                            },
                        }
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), QueryHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        producer.publish(
            ipc.ProducerMetadata(
                announced=[
                    ipc.ProducerAnnouncement(f"p{i}", f"t{i}", [[1, 2, 3, 4]], 64)
                    for i in range(2)
                ]
            )
        )
        consumer.publish(
            ipc.ConsumerMetadata(
                requests=[
                    ipc.ConsumerRequest(
                        f"d{i}",
                        f"t{i}",
                        [[11, 12]],
                        64,
                        "A",
                        f"http://127.0.0.1:{httpd.server_address[1]}",
                    )
                    for i in range(2)
                ]
            )
        )
        assert {_wait_queue(consumer.finished_recving) for _ in range(2)} == {
            "d0",
            "d1",
        }
        assert {_wait_queue(producer.finished_sending) for _ in range(2)} == {
            "p0",
            "p1",
        }
        assert consumer.invalid_block_ids.empty()
        with lock:
            assert sum(isinstance(job, ipc.EncodeChunkJob) for job in jobs) == 8
            assert sum(isinstance(job, ipc.DecodeChunkJob) for job in jobs) == 8
    finally:
        for process in reversed(started):
            process.shutdown()
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=5)


class _Stats:
    def __init__(self):
        self.expired = 0
        self.failed = 0

    def record_kv_expired_req(self):
        self.expired += 1

    def record_failed_recv(self):
        self.failed += 1


def _wait_queue(target: queue.SimpleQueue, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return target.get_nowait()
        except queue.Empty:
            time.sleep(0.02)
    return None


@pytest.mark.asyncio
async def test_failed_slot_is_quarantined_when_drain_fails(producer_session):
    service, _, request = producer_session()

    async def submit(job):
        return ipc.JobDone(job.job_id, error="encode failed")

    async def drain(slots):
        raise RuntimeError("drain failed")

    service._submit, service._drain_slots = submit, drain
    response = await service._serve_producer_chunk(b"peer", request)
    assert response.error == "encode failed"
    assert service._sender_slots.empty()


@pytest.mark.asyncio
async def test_pre_submit_abort_returns_unused_slot(producer_session):
    service, _, request = producer_session()
    service._closed[("t", "d", "n")] = time.monotonic() + 10
    jobs = []

    async def submit(job):
        jobs.append(job)
        return ipc.JobDone(job.job_id)

    service._submit = submit
    response = await service._serve_producer_chunk(b"peer", request)
    assert response.error is not None
    assert not any(isinstance(job, ipc.EncodeChunkJob) for job in jobs)
    assert service._sender_slots.qsize() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "topology,profile,error",
    [
        ({"remote_tp_size": 2, "remote_tp_rank": 1}, "P", "not paired"),
        ({"remote_tp_size": 4}, "P", "matching TP/PP"),
        ({"remote_tp_size": 2, "remote_pp_size": 2}, "P", "matching TP/PP"),
        ({"remote_tp_size": 2}, "D", "profile mismatch"),
    ],
)
async def test_incompatible_peer_is_rejected_before_engine_job(
    make_service, topology, profile, error
):
    service = make_service()
    service._init.tp_size = 2
    service._init.compression_profile = "P"
    state = _SendState("t", p_req_id="p", ready=True)
    service._send_meta["t"] = state

    async def unexpected_job(job):
        pytest.fail("Invalid peer submitted an engine job")

    service._submit = unexpected_job
    response = await service._open_producer_session(
        b"peer", _open_message(profile, **topology)
    )
    assert error in response.error
    assert not service._producer_sessions and not service._pending_jobs
    assert state.sending == state.sent == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_chunk", [0, 1])
async def test_decode_failure_propagates_before_or_after_last_chunk(
    receiver_peer, failed_chunk
):
    """A real local ZMQ exchange must report both early and final decode errors."""
    service, router, address = receiver_peer
    decoded_first = asyncio.Event()

    async def submit(job):
        if isinstance(job, ipc.DecodeChunkJob):
            if job.chunk_id == 0:
                decoded_first.set()
            return ipc.JobDone(
                job_id=job.job_id,
                error="injected" if job.chunk_id == failed_chunk else None,
            )
        return ipc.JobDone(job_id=job.job_id)

    service._submit = submit

    async def producer():
        identity, data = await router.recv_multipart()
        await router.send_multipart(
            [identity, _ENC.encode(CompressionSessionOpened(1, 2))]
        )
        while True:
            identity, data = await router.recv_multipart()
            request = _REQ_DEC.decode(data)
            if isinstance(request, CompressionAbort):
                await router.send_multipart(
                    [
                        identity,
                        _ENC.encode(
                            CompressionAbortAck(
                                request.transfer_id, request.d_req_id, request.nonce
                            )
                        ),
                    ]
                )
                return
            if request.chunk_id == 1:
                await decoded_first.wait()
                if failed_chunk == 0:
                    continue
            await router.send_multipart(
                [
                    identity,
                    _ENC.encode(CompressionChunkReady(request.chunk_id, 16, 16)),
                ]
            )

    task = asyncio.create_task(producer())
    try:
        request = ipc.ConsumerRequest("d", "t", [[3]], 16, "p", "unused")
        failed = await asyncio.wait_for(
            service._run_consumer_session(address, request), 5
        )
        await asyncio.wait_for(task, 5)
        assert failed == {3}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_helper_failure_does_not_finish_while_decode_is_writing():
    """Scheduler must not reuse destination blocks while an engine job is active."""
    from concurrent.futures import ThreadPoolExecutor

    process = object.__new__(SessionProcess)
    process._failed = threading.Event()
    process._failure_reason = None
    process._ready = threading.Event()
    process._state_lock = threading.RLock()
    process.finished_recving = queue.SimpleQueue()
    process.invalid_block_ids = queue.SimpleQueue()
    process._stats = _Stats()
    process._out = queue.Queue()
    entered, release = threading.Event(), threading.Event()

    def handler(job):
        entered.set()
        assert release.wait(5), "test failed to release engine job"
        return ipc.JobDone(job_id=job.job_id)

    process._handler = handler
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            process._run_job,
            ipc.DecodeChunkJob(
                job_id=1,
                transfer_id="t",
                d_req_id="d",
                nonce="n",
                chunk_id=0,
                slot_id=0,
                local_block_ids=[[3]],
                valid_tokens_by_group={0: 16},
                codec_bytes=16,
                u8_bytes=16,
            ),
        )
        try:
            assert entered.wait(5)
            process._mark_failed("injected EOF during decode")
            assert process.finished_recving.empty(), "decode still owns destination KV"
            assert process.invalid_block_ids.empty()
            with pytest.raises(RuntimeError, match="worker must stop"):
                process.check_health()
            with pytest.raises(RuntimeError, match="worker must stop"):
                process.publish(ipc.ConsumerMetadata())
        finally:
            release.set()
            future.result(timeout=5)


@pytest.mark.parametrize("hma", [False, True])
def test_helper_load_failure_uses_upstream_result_scope(hma):
    process = object.__new__(SessionProcess)
    process._failed = threading.Event()
    process._state_lock = threading.RLock()
    process.finished_recving = queue.SimpleQueue()
    process.invalid_block_ids = queue.SimpleQueue()
    process.failed_recving = queue.SimpleQueue() if hma else None
    process._stats = _Stats()

    process._handle_recv_done(ipc.RecvDone(d_req_id="d", failed_blocks=[3]))

    assert process.finished_recving.get_nowait() == "d"
    if hma:
        assert process.failed_recving.get_nowait() == "d"
        assert process.invalid_block_ids.empty()
    else:
        assert process.invalid_block_ids.get_nowait() == {3}


@pytest.mark.asyncio
@pytest.mark.parametrize("blocks", [[], [[]], [[3]]])
async def test_helper_empty_pull_only_releases_producer(
    make_service, monkeypatch, blocks
):
    from unittest.mock import AsyncMock

    service = make_service(sender=False)
    monkeypatch.setattr(service, "_ensure_remote_agent", AsyncMock())
    monkeypatch.setattr(service, "_handshake_worker_addrs", lambda engine: ["peer"])
    transfer = AsyncMock(return_value=set())
    monkeypatch.setattr(service, "_run_consumer_session", transfer)
    completions: list[ipc.RecvDone] = []
    service._to_engine = completions.append
    request = ipc.ConsumerRequest("d", "t", blocks, 0, "p", "http://bootstrap")

    await service._recv_request(request)

    transfer.assert_awaited_once_with("peer", request)
    assert completions == ([ipc.RecvDone("d")] if any(blocks) else [])


@pytest.mark.asyncio
async def test_late_job_completion_precedes_cuda_drain(make_service):
    service = make_service()
    service._loop = asyncio.get_running_loop()
    service._abort_timeout = 0.02
    sent = []
    drain_sent = asyncio.Event()

    def send(job):
        sent.append(job.job_id)
        if job.job_id != 100:
            drain_sent.set()
            service._on_engine_message(ipc.JobDone(job_id=job.job_id))

    service._to_engine = send
    with pytest.raises(asyncio.TimeoutError):
        await service._submit(
            ipc.EncodeChunkJob(
                100, "t", "d", "n", 0, 0, [[0]], {0: 16}, "peer", 1, 2, 3
            )
        )
    task = asyncio.create_task(service._drain_slots([0]))
    await asyncio.sleep(0)
    assert sent == [100]
    assert not drain_sent.is_set()
    service._on_engine_message(ipc.JobDone(job_id=100))
    await task
    assert drain_sent.is_set()
    assert not service._pending_jobs


@pytest.mark.asyncio
async def test_slot_drain_does_not_wait_for_other_request(make_service):
    service = make_service()
    service._loop = asyncio.get_running_loop()
    other = service._loop.create_future()
    job = ipc.EncodeChunkJob(100, "other", "d", "n", 0, 1, [[0]], {}, "peer", 1, 2, 3)
    service._pending_jobs[100] = job, other
    service._to_engine = lambda drain: service._on_engine_message(
        ipc.JobDone(drain.job_id)
    )
    await service._drain_slots([0])
    assert not other.done()
    assert service._pending_jobs[100] == (job, other)


@pytest.mark.parametrize("failure", ["helper_freeze", "stuck_job"])
def test_helper_fault_has_bounded_shutdown_without_finishing_blocks(
    monkeypatch, failure
):
    monkeypatch.setattr(session_process, "_HEARTBEAT_TIMEOUT_S", 0.2)
    monkeypatch.setattr(session_process, "_SHUTDOWN_TIMEOUT_S", 0.2)
    process = SessionProcess(
        lambda job: ipc.JobDone(job.job_id),
        _make_init(False, True),
        queue.SimpleQueue(),
        queue.SimpleQueue(),
        queue.SimpleQueue(),
        _Stats(),
        -1,
    )
    release = threading.Event()
    entered = threading.Event()

    def stuck_job():
        entered.set()
        release.wait()

    try:
        process.start()
        if failure == "helper_freeze":
            os.kill(process._process.pid, signal.SIGSTOP)
            deadline = time.monotonic() + 3
            while not process._failed.is_set() and time.monotonic() < deadline:
                try:
                    process.check_health()
                except RuntimeError:
                    break
                time.sleep(0.01)
            assert process._failed.is_set()
        else:
            process._jobs.add(process._pool.submit(stuck_job))
            assert entered.wait(2)
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="worker must stop"):
            process.shutdown()
        assert time.monotonic() - start < 3
        assert not process._process.is_alive()
        assert process.finished_sending.empty()
        assert process.finished_recving.empty()
        assert process.invalid_block_ids.empty()
    finally:
        release.set()
        if process._process.is_alive():
            os.kill(process._process.pid, signal.SIGCONT)
        with contextlib.suppress(RuntimeError):
            process.shutdown()


@pytest.mark.asyncio
async def test_abort_ack_waits_for_timed_out_engine_write(producer_session):
    """A timed-out encode waiter cannot authorize reuse of the remote slot."""
    service, state, request = producer_session()
    jobs, completions = [], []
    draining = asyncio.Event()
    original_drain = service._drain_slots

    async def drain(slots):
        draining.set()
        await original_drain(slots)

    def send(message):
        if isinstance(message, ipc.EncodeChunkJob):
            jobs.append(message)
        elif isinstance(message, ipc.DrainSlotsJob):
            service._on_engine_message(ipc.JobDone(job_id=message.job_id))
        else:
            completions.append(message)

    service._drain_slots = drain
    service._to_engine = send
    transfer = asyncio.create_task(service._serve_producer_chunk(b"peer", request))
    service._operations[("t", "d", "n")] = {transfer}
    await asyncio.wait_for(draining.wait(), 2)
    abort = asyncio.create_task(
        service._abort_producer(b"peer", CompressionAbort("t", "d", "cancel", "n"))
    )
    await asyncio.sleep(0)
    assert not transfer.done()
    assert not abort.done()
    assert not completions
    service._on_engine_message(ipc.JobDone(job_id=jobs[0].job_id))
    await transfer
    ack = await abort
    assert ack.error is None
    assert service._sender_slots.qsize() == 1
    assert len(completions) == 1
    assert isinstance(completions[0], ipc.FinalEvent)


@pytest.mark.asyncio
@pytest.mark.parametrize("during_open", [False, True])
@pytest.mark.parametrize("acknowledged", [True, False])
async def test_consumer_cancellation_aborts_before_reusing_slot(
    receiver_peer, during_open, acknowledged
):
    service, router, address = receiver_peer
    service._receiver_slots = asyncio.Queue()
    service._receiver_slots.put_nowait(0)
    notifications: list[ipc.HelperToEngine] = []
    service._to_engine = notifications.append
    service._abort_timeout = 0.1
    granted, abort_seen, allow_ack = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def submit(job):
        return ipc.JobDone(job_id=job.job_id)

    service._submit = submit

    async def producer():
        identity, _ = await router.recv_multipart()
        if not during_open:
            await router.send_multipart(
                [identity, _ENC.encode(CompressionSessionOpened(1, 1))]
            )
            _, data = await router.recv_multipart()
            assert isinstance(_REQ_DEC.decode(data), CompressionChunkPull)
        granted.set()
        identity, data = await router.recv_multipart()
        abort = _REQ_DEC.decode(data)
        assert isinstance(abort, CompressionAbort)
        abort_seen.set()
        await allow_ack.wait()
        queued = (
            CompressionSessionOpened(1, 1)
            if during_open
            else CompressionChunkReady(0, 0, 0)
        )
        await router.send_multipart([identity, _ENC.encode(queued)])
        if not acknowledged:
            return
        await router.send_multipart(
            [
                identity,
                _ENC.encode(
                    CompressionAbortAck(abort.transfer_id, abort.d_req_id, abort.nonce)
                ),
            ]
        )

    peer = asyncio.create_task(producer())
    consumer = asyncio.create_task(
        service._run_consumer_session(
            address,
            ipc.ConsumerRequest("d", "t", [[3]], 16, "p", "unused"),
        )
    )
    try:
        await asyncio.wait_for(granted.wait(), 2)
        consumer.cancel()
        await asyncio.wait_for(abort_seen.wait(), 2)
        assert not consumer.done()
        assert service._receiver_slots.qsize() == int(during_open)
        allow_ack.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, 2)
        await peer
        assert service._receiver_slots.qsize() == int(acknowledged or during_open)
        if acknowledged:
            assert service._fatal_error is None
        else:
            assert service._fatal_error is not None
            assert any(
                isinstance(message, ipc.HelperFailed) for message in notifications
            )
            assert not any(
                isinstance(message, ipc.RecvDone) for message in notifications
            )
    finally:
        allow_ack.set()
        for task in (peer, consumer):
            task.cancel()
        await asyncio.gather(peer, consumer, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_open_retains_source_until_engine_write_finishes(make_service):
    """SessionOpen may already be writing raw hybrid state before its reply."""
    service = make_service()
    service._loop = asyncio.get_running_loop()
    state = _SendState("t", p_req_id="p", ready=True)
    service._send_meta["t"] = state
    submitted = asyncio.Event()
    jobs, completions = [], []

    def send(message):
        if isinstance(message, ipc.OpenProducerJob):
            jobs.append(message)
            submitted.set()
        else:
            completions.append(message)

    service._to_engine = send
    task = asyncio.create_task(service._open_producer_session(b"peer", _open_message()))
    await asyncio.wait_for(submitted.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert state.sending == 1
    assert not completions
    service._on_engine_message(ipc.JobDone(job_id=jobs[0].job_id))
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert state.sending == 0
    assert len(completions) == 1
    assert service._fatal_error is None


@pytest.mark.asyncio
async def test_duplicate_open_cannot_overwrite_pending_session(make_service):
    service = make_service()
    waiting = asyncio.Event()

    async def wait_ready(state, operation_key):
        waiting.set()
        await asyncio.Event().wait()

    service._wait_ready = wait_ready
    request = _open_message()
    first = asyncio.create_task(service._open_producer_session(b"peer", request))
    try:
        await asyncio.wait_for(waiting.wait(), 2)
        second = await service._open_producer_session(b"peer", request)
        assert second.error == "Compression session already open"
        assert not service._pending_jobs
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
    assert not service._opening_sessions
    assert not service._send_meta


@pytest.mark.parametrize("chunk_id", range(6))
def test_chunk_ipc_metadata_is_bounded_and_preserves_partial_suffix(chunk_id):
    service = object.__new__(_SessionService)
    service._init = _make_init(False, False)
    service._init.logical_blocks_per_chunk = 2
    service._init.fa_group_block_sizes = {1: 16}
    start = chunk_id // 2 * 2
    blocks, tokens = service._chunk_metadata(
        [[999], list(range(100, 105))], {1: 73}, chunk_id, 5
    )
    assert blocks == [[], list(range(100 + start, min(102 + start, 105)))]
    assert tokens == {1: min(32, 73 - start * 16)}
