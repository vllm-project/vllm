# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-process compression session helper.

The helper owns everything that used to fight the engine's main thread: the
ZMQ transport, the SessionOpen/ChunkPull/ChunkReady/Abort state machine, slot
scheduling, timeout sweeps and completion accounting.  The engine process
keeps the KV cache, the compression arena and the CUDA encode/decode path and
executes short jobs forwarded over a duplex ``multiprocessing`` pipe.

This module contains both sides:

* ``SessionProcess`` -- engine-side handle (spawns the process, forwards jobs
  to a job pool, publishes completion events into the connector queues).
* ``_SessionService`` -- helper-side asyncio service.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

import httpx
import msgspec
import zmq
import zmq.asyncio

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.protocol import (
    CompressionAbort,
    CompressionAbortAck,
    CompressionChunkPull,
    CompressionChunkReady,
    CompressionRequest,
    CompressionResponse,
    CompressionSessionOpen,
    CompressionSessionOpened,
    MooncakeXferMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.tokens import (
    chunk_token_counts,
    chunk_window,
    suffix_token_counts,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.session_ipc import (
    ConsumerMetadata,
    ConsumerRequest,
    DecodeChunkJob,
    DrainSlotsJob,
    EncodeChunkJob,
    EngineStop,
    EngineToHelper,
    FinalEvent,
    HelperFailed,
    HelperHeartbeat,
    HelperJob,
    HelperReady,
    HelperToEngine,
    InitMessage,
    JobDone,
    OpenProducerJob,
    ProducerMetadata,
    RecvDone,
    Stopped,
    handshake_target_ranks,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_ABORT_TIMEOUT_S = 480.0
_STARTUP_TIMEOUT_S = 300.0
_SWEEP_INTERVAL_S = 0.1
_POLL_INTERVAL_S = 0.05
_PENDING_REQ_EXPIRY_S = _ABORT_TIMEOUT_S + 60.0
_HEARTBEAT_INTERVAL_S = 1.0
_HEARTBEAT_TIMEOUT_S = 30.0
_SHUTDOWN_TIMEOUT_S = 5.0


@dataclass(slots=True)
class _SendState:
    transfer_id: str
    p_req_id: str = ""
    local_block_ids: list[list[int]] | None = None
    num_tokens: int | None = None
    ready: bool = False
    event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0

    def __post_init__(self):
        self.local_block_ids = self.local_block_ids or []
        if self.ready:
            self.event.set()


@dataclass(slots=True, kw_only=True)
class _ProducerSession:
    identity: bytes
    transfer_id: str
    d_req_id: str
    nonce: str
    remote_session: str
    local_block_ids: list[list[int]]
    valid_tokens_by_group: dict[int, int]
    logical_block_count: int
    total_chunks: int
    send_state: _SendState
    completed_chunks: int = 0
    inflight_chunks: int = 0
    state: str = "ACTIVE"
    last_activity: float = field(default_factory=time.monotonic)
    seen_chunks: set[int] = field(default_factory=set)


async def _cancel_tasks(*tasks):
    pending = [task for task in tasks if task is not None]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


class _SessionService:
    """Helper-side asyncio service."""

    def __init__(self, conn, init: InitMessage):
        self._conn = conn
        self._init = init
        self._abort_timeout = float(init.abort_timeout_s)
        self._encoder = msgspec.msgpack.Encoder()
        self._request_decoder = msgspec.msgpack.Decoder(CompressionRequest)
        self._response_decoder = msgspec.msgpack.Decoder(
            CompressionSessionOpened | CompressionChunkReady | CompressionAbortAck
        )
        self._engine_decoder = msgspec.msgpack.Decoder(EngineToHelper)
        self._write_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._fatal_error: str | None = None
        self._stopping = False
        self._next_job_id = 0
        self._pending_jobs: dict[int, tuple[HelperJob, asyncio.Future]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._side_channel_port = 0

        self._zmq: zmq.asyncio.Context | None = None
        self._router: zmq.asyncio.Socket | None = None
        self._listener_task: asyncio.Task | None = None

        # Producer state
        self._send_meta: dict[str, _SendState] = {}
        self._sender_slots: asyncio.Queue[int] | None = None
        self._closed: dict[tuple[str, str, str], float] = {}
        self._operations: dict[tuple[str, str, str], set[asyncio.Task]] = {}
        self._producer_sessions: dict[tuple[bytes, str, str], _ProducerSession] = {}
        self._opening_sessions: set[tuple[bytes, str, str]] = set()

        # Consumer state
        self._receiver_slots: asyncio.Queue[int] | None = None
        self._slot_payload: list[int] = []
        self._slot_aux: list[int] = []
        self._slot_tail: list[int] = []
        self._agents: dict[str, dict[int, dict[int, str]]] = {}
        self._remote_tp_sizes: dict[str, int] = {}
        self._bootstrap_pending: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _to_engine(self, message: HelperToEngine) -> None:
        data = self._encoder.encode(message)
        with self._write_lock:
            self._conn.send_bytes(data)

    def _start_reader(self) -> None:
        assert self._loop is not None
        loop = self._loop

        def run():
            while True:
                try:
                    raw = self._conn.recv_bytes()
                except (EOFError, OSError):
                    loop.call_soon_threadsafe(self._on_pipe_closed)
                    return
                try:
                    message = self._engine_decoder.decode(raw)
                except Exception:
                    logger.exception("Session helper: undecodable engine message")
                    continue
                loop.call_soon_threadsafe(self._on_engine_message, message)

        threading.Thread(
            target=run, name="kvs-session-pipe-reader", daemon=True
        ).start()

    def _on_pipe_closed(self) -> None:
        self._stopping = True
        if self._stop_event is not None:
            self._stop_event.set()

    def _on_engine_message(self, message) -> None:
        if isinstance(message, EngineStop):
            self._stopping = True
            if self._stop_event is not None:
                self._stop_event.set()
        elif isinstance(message, ProducerMetadata):
            if not self._stopping:
                self._apply_producer_metadata(message)
        elif isinstance(message, ConsumerMetadata):
            if self._stopping:
                return
            for request in message.requests:
                task = asyncio.create_task(self._recv_request(request))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        elif isinstance(message, JobDone):
            pending = self._pending_jobs.pop(message.job_id, None)
            if pending is None:
                logger.warning(
                    "Session helper: completion for unknown job %s", message.job_id
                )
            elif not pending[1].done():
                pending[1].set_result(message)
        else:
            logger.warning("Session helper: unexpected message %r", message)

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def _alloc_job(self) -> int:
        self._next_job_id += 1
        return self._next_job_id

    def _chunk_metadata(self, block_ids, tokens, chunk_id, logical_block_count):
        start, count, _ = chunk_window(
            chunk_id, logical_block_count, self._init.logical_blocks_per_chunk
        )
        block_sizes = self._init.fa_group_block_sizes
        return (
            [
                group[start : start + count] if index in block_sizes else []
                for index, group in enumerate(block_ids)
            ],
            chunk_token_counts(tokens, block_sizes, start, count),
        )

    async def _submit(self, job: HelperJob) -> JobDone:
        assert self._loop is not None
        future: asyncio.Future = self._loop.create_future()
        self._pending_jobs[job.job_id] = job, future
        self._to_engine(job)
        # Only JobDone retires an engine job. Cancelling its waiter cannot
        # cancel a queued CPU job, a CUDA kernel or a Mooncake write.
        return await asyncio.wait_for(asyncio.shield(future), self._abort_timeout)

    def _fail_worker(self, reason: str) -> None:
        if self._fatal_error is None:
            self._fatal_error = reason
            self._stopping = True
            self._to_engine(HelperFailed(reason=reason))

    async def _drain_jobs(self, operation_key=None, slot_ids=None) -> None:
        pending = tuple(
            future
            for job, future in self._pending_jobs.values()
            if (
                operation_key is None
                or (
                    getattr(job, "transfer_id", None),
                    getattr(job, "d_req_id", None),
                    getattr(job, "nonce", None),
                )
                == operation_key
            )
            and (slot_ids is None or getattr(job, "slot_id", None) in slot_ids)
        )
        if pending:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*pending)), self._abort_timeout
            )

    async def _drain_slots(self, slot_ids: list[int]) -> None:
        if not slot_ids:
            return
        # First join engine jobs, including their synchronous network writes.
        # A stream event alone does not cover a queued job or remote DMA.
        await self._drain_jobs(slot_ids=slot_ids)
        job = DrainSlotsJob(job_id=self._alloc_job(), slot_ids=list(slot_ids))
        result = await self._submit(job)
        if result.error is not None:
            raise RuntimeError(f"Session helper: slot drain failed: {result.error}")

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    def _apply_producer_metadata(self, message: ProducerMetadata) -> None:
        for announcement in message.announced:
            state = self._send_meta.get(announcement.transfer_id)
            if state is None:
                state = _SendState(announcement.transfer_id)
                self._send_meta[announcement.transfer_id] = state
            state.p_req_id = announcement.p_req_id
            state.local_block_ids = announcement.local_block_ids
            state.num_tokens = announcement.num_tokens
            state.expire_time = time.monotonic() + self._abort_timeout
            state.ready = True
            state.event.set()
        for published in message.published:
            state = self._send_meta.get(published.transfer_id)
            if state is None:
                self._send_meta[published.transfer_id] = _SendState(
                    published.transfer_id, p_req_id=published.p_req_id
                )
            elif not state.p_req_id:
                state.p_req_id = published.p_req_id
        for transfer_id in message.dropped:
            self._send_meta.pop(transfer_id, None)

    async def _start_producer(self) -> None:
        assert self._zmq is not None
        self._sender_slots = asyncio.Queue()
        for slot_id in range(len(self._init.slot_addresses)):
            self._sender_slots.put_nowait(slot_id)
        self._router = self._zmq.socket(zmq.ROUTER)
        self._side_channel_port = self._router.bind_to_random_port(
            f"tcp://{self._init.hostname}"
        )
        logger.info(
            "Session helper producer listening on tcp://%s:%d",
            self._init.hostname,
            self._side_channel_port,
        )
        await self._register_with_bootstrap()
        self._listener_task = asyncio.create_task(self._listen())
        sweep = asyncio.create_task(self._sweep_sends())
        self._tasks.add(sweep)
        sweep.add_done_callback(self._tasks.discard)

    async def _register_with_bootstrap(self) -> None:
        if not self._init.register_with_bootstrap:
            return
        url = f"http://{self._init.bootstrap_host}:{self._init.bootstrap_port}/register"
        payload = {
            "engine_id": self._init.engine_id,
            "dp_rank": self._init.dp_rank,
            "tp_rank": self._init.tp_rank,
            "pp_rank": self._init.pp_rank,
            "addr": f"tcp://{self._init.hostname}:{self._side_channel_port}",
        }
        while not self._stopping:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                logger.info("Session helper registered with bootstrap %s", url)
                return
            except httpx.ConnectError:
                await asyncio.sleep(1)
            except Exception as exc:
                logger.error(
                    "Session helper bootstrap registration failed: %s. Retrying in 1s.",
                    exc,
                )
                await asyncio.sleep(1)

    async def _listen(self) -> None:
        assert self._router is not None
        while not self._stopping:
            try:
                identity, raw = await self._router.recv_multipart()
            except (zmq.ContextTerminated, asyncio.CancelledError):
                return
            try:
                request = self._request_decoder.decode(raw)
            except Exception:
                logger.exception("Session helper: undecodable request")
                continue
            self._dispatch_producer(identity, request)

    def _dispatch_producer(self, identity: bytes, request: CompressionRequest) -> None:
        if self._stopping and not isinstance(request, CompressionAbort):
            return
        if isinstance(request, CompressionSessionOpen):
            if len(request.metadata.req_blocks) == 1:
                req_id, (transfer_id, _) = next(
                    iter(request.metadata.req_blocks.items())
                )
            else:
                req_id, transfer_id = "", ""
        else:
            req_id, transfer_id = request.d_req_id, request.transfer_id
        operation_key = (transfer_id, req_id, request.nonce)
        now = time.monotonic()
        self._closed = {
            key: expiry for key, expiry in self._closed.items() if expiry > now
        }
        task = asyncio.create_task(self._handle_producer_request(identity, request))
        if not isinstance(request, CompressionAbort):
            self._operations.setdefault(operation_key, set()).add(task)

            def finished(done, key=operation_key):
                pending = self._operations.get(key)
                if pending is not None:
                    pending.discard(done)
                    if not pending:
                        self._operations.pop(key, None)

            task.add_done_callback(finished)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_producer_request(
        self, identity: bytes, request: CompressionRequest
    ) -> None:
        response: CompressionResponse
        if isinstance(request, CompressionSessionOpen):
            response = await self._open_producer_session(identity, request)
        elif isinstance(request, CompressionChunkPull):
            response = await self._serve_producer_chunk(identity, request)
        else:
            response = await self._abort_producer(identity, request)
        assert self._router is not None
        await self._router.send_multipart((identity, self._encoder.encode(response)))

    async def _wait_ready(
        self, state: _SendState, operation_key: tuple[str, str, str]
    ) -> None:
        deadline = time.monotonic() + self._abort_timeout
        while not state.ready:
            if operation_key in self._closed:
                raise ValueError("Session aborted before producer became ready.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Producer block table was not announced in time.")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(state.event.wait(), _POLL_INTERVAL_S)

    async def _open_producer_session(
        self, identity: bytes, request: CompressionSessionOpen
    ) -> CompressionSessionOpened:
        if len(request.metadata.req_blocks) != 1:
            return CompressionSessionOpened(
                0, 0, error="Expected one request per session"
            )
        d_req_id, (transfer_id, _) = next(iter(request.metadata.req_blocks.items()))
        key = (identity, transfer_id, d_req_id)
        if key in self._opening_sessions or key in self._producer_sessions:
            return CompressionSessionOpened(
                0, 0, error="Compression session already open"
            )
        self._opening_sessions.add(key)
        try:
            return await self._open_producer_session_once(identity, request)
        finally:
            self._opening_sessions.discard(key)

    async def _open_producer_session_once(
        self, identity: bytes, request: CompressionSessionOpen
    ) -> CompressionSessionOpened:
        if request.profile != self._init.compression_profile:
            return CompressionSessionOpened(
                0, 0, error="Mooncake compression profile mismatch"
            )
        metadata = request.metadata
        d_req_id, (transfer_id, remote_block_ids) = next(
            iter(metadata.req_blocks.items())
        )
        operation_key = (transfer_id, d_req_id, request.nonce)
        state = self._send_meta.get(transfer_id)
        created = False
        if state is None:
            state = _SendState(transfer_id)
            self._send_meta[transfer_id] = state
            created = True
        send_target_claimed = False
        try:
            if (
                metadata.remote_tp_size != self._init.tp_size
                or metadata.remote_pp_size != self._init.pp_size
            ):
                raise ValueError(
                    "Compressed transport requires matching TP/PP topology."
                )
            remote_tp_ranks = handshake_target_ranks(
                self._init.tp_rank, self._init.tp_size, metadata.remote_tp_size
            )
            if metadata.remote_tp_rank not in remote_tp_ranks:
                raise ValueError("Consumer TP rank is not paired with this producer.")
            await self._wait_ready(state, operation_key)
            state.sending += 1
            send_target_claimed = True
            if not state.need_send:
                state.need_send = len(remote_tp_ranks)
            if operation_key in self._closed:
                raise ValueError("Session was aborted before KV transfer.")
            job = OpenProducerJob(
                job_id=self._alloc_job(),
                transfer_id=transfer_id,
                d_req_id=d_req_id,
                nonce=request.nonce,
                metadata=metadata,
                local_block_ids=state.local_block_ids,
                num_tokens=state.num_tokens,
            )
            result = await self._submit(job)
            if result.error is not None:
                raise RuntimeError(result.error)
            session = _ProducerSession(
                identity=identity,
                transfer_id=transfer_id,
                d_req_id=d_req_id,
                nonce=request.nonce,
                remote_session=f"{metadata.remote_hostname}:{metadata.remote_port}",
                local_block_ids=result.local_block_ids,
                valid_tokens_by_group=result.valid_tokens_by_group,
                logical_block_count=result.logical_block_count,
                total_chunks=result.total_chunks,
                send_state=state,
            )
            self._producer_sessions[(identity, transfer_id, d_req_id)] = session
            send_target_claimed = False
            if session.total_chunks == 0:
                self._complete_producer_session(identity, session)
            return CompressionSessionOpened(
                logical_block_count=result.logical_block_count,
                total_chunks=result.total_chunks,
                profile=self._init.compression_profile,
            )
        except (Exception, asyncio.CancelledError) as exc:
            if send_target_claimed:
                try:
                    await self._drain_jobs(operation_key)
                except Exception:
                    self._fail_worker("Could not drain producer open job")
                if self._fatal_error is None:
                    self._finish_send_meta(state)
            if (
                created
                and self._send_meta.get(transfer_id) is state
                and not state.ready
                and not state.p_req_id
                and state.sending == 0
            ):
                self._send_meta.pop(transfer_id, None)
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning(
                "Session helper SessionOpen failed: d_req_id=%s transfer_id=%s: %s",
                d_req_id,
                transfer_id,
                exc,
            )
            return CompressionSessionOpened(
                logical_block_count=0, total_chunks=0, error=str(exc)
            )

    async def _serve_producer_chunk(
        self, identity: bytes, request: CompressionChunkPull
    ) -> CompressionChunkReady:
        key = (identity, request.transfer_id, request.d_req_id)
        session = self._producer_sessions.get(key)
        if session is None:
            return CompressionChunkReady(
                chunk_id=request.chunk_id,
                data_bytes=0,
                payload_bytes=0,
                error="Unknown or completed compression session.",
            )
        if (
            request.nonce != session.nonce
            or not 0 <= request.chunk_id < session.total_chunks
            or request.chunk_id in session.seen_chunks
        ):
            return CompressionChunkReady(
                request.chunk_id,
                0,
                0,
                error="Invalid or duplicate compression chunk",
            )
        session.seen_chunks.add(request.chunk_id)
        session.last_activity = time.monotonic()
        assert self._sender_slots is not None
        slot_id = -1
        counted = False
        slot_synced = False
        job_submitted = False
        try:
            if session.state != "ACTIVE":
                raise RuntimeError(f"Compression session is {session.state}.")
            session.inflight_chunks += 1
            counted = True
            slot_id = await self._sender_slots.get()
            operation_key = (
                request.transfer_id,
                request.d_req_id,
                request.nonce,
            )
            if session.state != "ACTIVE" or operation_key in self._closed:
                raise ValueError("Session aborted while waiting for a slot.")
            block_ids, tokens = self._chunk_metadata(
                session.local_block_ids,
                session.valid_tokens_by_group,
                request.chunk_id,
                session.logical_block_count,
            )
            job = EncodeChunkJob(
                job_id=self._alloc_job(),
                transfer_id=request.transfer_id,
                d_req_id=request.d_req_id,
                nonce=request.nonce,
                chunk_id=request.chunk_id,
                slot_id=slot_id,
                local_block_ids=block_ids,
                valid_tokens_by_group=tokens,
                remote_session=session.remote_session,
                payload_addr=request.payload_addr,
                aux_addr=request.aux_addr,
                tail_addr=request.tail_addr,
            )
            job_submitted = True
            result = await self._submit(job)
            if result.error is not None:
                raise RuntimeError(result.error)
            slot_synced = True
            session.completed_chunks += 1
            if (
                session.state == "ACTIVE"
                and session.completed_chunks == session.total_chunks
            ):
                self._complete_producer_session(identity, session)
            return CompressionChunkReady(
                chunk_id=request.chunk_id,
                data_bytes=result.u8_bytes,
                payload_bytes=result.codec_bytes,
            )
        except Exception as exc:
            session.state = "FAILED"
            logger.exception(
                "Session helper send failed: d_req_id=%s transfer_id=%s chunk=%s",
                request.d_req_id,
                request.transfer_id,
                request.chunk_id,
            )
            return CompressionChunkReady(
                chunk_id=request.chunk_id,
                data_bytes=0,
                payload_bytes=0,
                error=str(exc),
            )
        finally:
            if slot_id >= 0:
                reusable = slot_synced or not job_submitted
                if not reusable:
                    try:
                        await self._drain_slots([slot_id])
                        reusable = True
                    except Exception:
                        reusable = False
                        self._fail_worker("Could not drain producer slot")
                        logger.exception(
                            "Session helper could not drain sender slot %d", slot_id
                        )
                if reusable:
                    self._sender_slots.put_nowait(slot_id)
                else:
                    logger.error(
                        "Session helper quarantined sender slot %d after failed "
                        "encode cleanup.",
                        slot_id,
                    )
            if counted:
                session.inflight_chunks -= 1
                session.last_activity = time.monotonic()
            if session.inflight_chunks == 0 and session.state in (
                "CANCELLED",
                "FAILED",
            ):
                self._discard_producer_session(key, session)

    def _complete_producer_session(
        self, identity: bytes, session: _ProducerSession
    ) -> None:
        key = (identity, session.transfer_id, session.d_req_id)
        if self._producer_sessions.pop(key, None) is not session:
            return
        session.state = "COMPLETE"
        self._closed[(session.transfer_id, session.d_req_id, session.nonce)] = (
            time.monotonic() + _PENDING_REQ_EXPIRY_S
        )
        self._finish_send_meta(session.send_state)

    def _finish_send_meta(self, state: _SendState) -> None:
        state.sending -= 1
        state.sent += 1
        if (
            state.sent == state.need_send
            and self._send_meta.pop(state.transfer_id, None) is not None
        ):
            self._to_engine(FinalEvent(kind="sending", req_id=state.p_req_id))

    def _discard_producer_session(
        self, key: tuple[bytes, str, str], session: _ProducerSession
    ) -> None:
        if self._fatal_error is not None:
            return
        if self._producer_sessions.pop(key, None) is not session:
            return
        if session.state in ("CANCELLED", "FAILED"):
            self._finish_send_meta(session.send_state)
        else:
            session.send_state.sending -= 1

    async def _abort_producer(
        self, identity: bytes, request: CompressionAbort
    ) -> CompressionAbortAck:
        operation_key = (request.transfer_id, request.d_req_id, request.nonce)
        self._closed[operation_key] = time.monotonic() + _PENDING_REQ_EXPIRY_S
        key = (identity, request.transfer_id, request.d_req_id)
        session = self._producer_sessions.get(key)
        if session is None:
            session = next(
                (
                    candidate
                    for candidate in self._producer_sessions.values()
                    if candidate.transfer_id == request.transfer_id
                    and candidate.d_req_id == request.d_req_id
                    and candidate.nonce == request.nonce
                ),
                None,
            )
        if session is not None and session.nonce == request.nonce:
            session.state = "CANCELLED"
            if session.inflight_chunks == 0:
                self._discard_producer_session(
                    (session.identity, session.transfer_id, session.d_req_id),
                    session,
                )
        pending = tuple(self._operations.get(operation_key, ()))
        error = self._fatal_error
        try:

            async def drain():
                if pending:
                    await asyncio.shield(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                while (
                    session is not None
                    and session.nonce == request.nonce
                    and session.inflight_chunks
                ):
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(drain(), self._abort_timeout)
            await self._drain_jobs(operation_key)
            error = self._fatal_error
        except TimeoutError:
            error = "Producer could not drain aborted session."
            self._fail_worker(error)
        for candidate_key, candidate in tuple(self._producer_sessions.items()):
            if (
                candidate.transfer_id,
                candidate.d_req_id,
                candidate.nonce,
            ) == operation_key:
                candidate.state = "CANCELLED"
                if candidate.inflight_chunks == 0:
                    self._discard_producer_session(candidate_key, candidate)
        return CompressionAbortAck(
            transfer_id=request.transfer_id,
            d_req_id=request.d_req_id,
            nonce=request.nonce,
            error=error,
        )

    async def _sweep_sends(self) -> None:
        while not self._stopping:
            await asyncio.sleep(_SWEEP_INTERVAL_S)
            now = time.monotonic()
            for key, session in tuple(self._producer_sessions.items()):
                if (
                    session.inflight_chunks == 0
                    and now - session.last_activity > self._abort_timeout
                ):
                    self._closed[
                        (session.transfer_id, session.d_req_id, session.nonce)
                    ] = now + _PENDING_REQ_EXPIRY_S
                    session.state = "CANCELLED"
                    self._discard_producer_session(key, session)
            for transfer_id, state in list(self._send_meta.items()):
                if state.p_req_id and state.expire_time < now and state.sending == 0:
                    logger.warning(
                        "Session helper: request %s timed out without being sent; "
                        "freeing producer blocks.",
                        state.p_req_id,
                    )
                    self._to_engine(
                        FinalEvent(kind="sending_expired", req_id=state.p_req_id)
                    )
                    self._send_meta.pop(transfer_id, None)

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    async def _recv_request(self, request: ConsumerRequest) -> None:
        failed: set[int] = set()
        try:
            await self._ensure_remote_agent(
                request.remote_engine_id, request.remote_bootstrap_addr
            )
            worker_addrs = self._handshake_worker_addrs(request.remote_engine_id)
            if not worker_addrs:
                raise RuntimeError(
                    f"No Mooncake worker addresses for engine "
                    f"{request.remote_engine_id}"
                )
            results = await asyncio.gather(
                *(
                    self._run_consumer_session(worker_addr, request)
                    for worker_addr in worker_addrs
                ),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    failed.update(self._all_blocks(request.local_block_ids))
                else:
                    failed.update(result)
        except (Exception, asyncio.CancelledError):
            logger.exception(
                "Session helper receive failed: d_req_id=%s transfer_id=%s",
                request.d_req_id,
                request.transfer_id,
            )
            failed.update(self._all_blocks(request.local_block_ids))
        finally:
            if (
                self._fatal_error is None
                and not self._stopping
                and any(request.local_block_ids)
            ):
                self._to_engine(
                    RecvDone(d_req_id=request.d_req_id, failed_blocks=sorted(failed))
                )

    @staticmethod
    def _all_blocks(block_ids: list[list[int]]) -> set[int]:
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

        return {
            block for group in block_ids for block in group if block != NULL_BLOCK_ID
        }

    async def _ensure_remote_agent(self, engine_id: str, bootstrap_addr: str) -> None:
        if engine_id in self._agents:
            return
        event = self._bootstrap_pending.get(bootstrap_addr)
        if event is None:
            event = asyncio.Event()
            self._bootstrap_pending[bootstrap_addr] = event
            task = asyncio.create_task(self._query_bootstrap(bootstrap_addr, event))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        await event.wait()
        if engine_id not in self._agents:
            raise RuntimeError(
                f"Remote engine {engine_id} not found at bootstrap {bootstrap_addr}"
            )

    async def _query_bootstrap(self, bootstrap_addr: str, event: asyncio.Event) -> None:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(bootstrap_addr + "/query")
                response.raise_for_status()
                data: dict = response.json()
            for _, dp_entry in data.items():
                engine_id = dp_entry["engine_id"]
                self._agents[engine_id] = {
                    int(tp_rank): {
                        int(pp_rank): worker_addr
                        for pp_rank, worker_addr in tp_entry.items()
                    }
                    for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                }
                self._remote_tp_sizes[engine_id] = len(dp_entry["worker_addr"])
        except Exception:
            logger.exception(
                "Session helper failed to query bootstrap server %s",
                bootstrap_addr,
            )
        finally:
            event.set()
            self._bootstrap_pending.pop(bootstrap_addr, None)

    def _handshake_worker_addrs(self, engine_id: str) -> list[str]:
        remote_tp_ranks = handshake_target_ranks(
            self._init.tp_rank,
            self._init.tp_size,
            self._remote_tp_sizes[engine_id],
        )
        worker_addrs: list[str] = []
        for remote_tp_rank in remote_tp_ranks:
            pp_to_addr = self._agents[engine_id][remote_tp_rank]
            if (
                self._init.pp_size == len(pp_to_addr)
                and self._init.pp_rank in pp_to_addr
            ):
                pp_ranks = [self._init.pp_rank]
            else:
                pp_ranks = sorted(pp_to_addr)
            worker_addrs.extend(pp_to_addr[pp_rank] for pp_rank in pp_ranks)
        return worker_addrs

    async def _run_consumer_session(
        self, worker_addr: str, request: ConsumerRequest
    ) -> set[int]:
        assert self._zmq is not None
        assert self._receiver_slots is not None
        nonce = f"{self._init.tp_rank}-{time.monotonic_ns()}"
        metadata = MooncakeXferMetadata(
            remote_hostname=self._init.hostname,
            remote_port=self._init.rpc_port,
            remote_tp_size=self._init.tp_size,
            remote_tp_rank=self._init.tp_rank,
            req_blocks={
                request.d_req_id: (
                    request.transfer_id,
                    request.local_block_ids,
                )
            },
            kv_caches_base_addr=self._init.kv_caches_base_addr,
            block_lens=self._init.block_lens,
            kv_block_lens=self._init.kv_block_lens,
            registered_layer_names=self._init.registered_layer_names,
            registered_layer_indices=self._init.registered_layer_indices,
            registered_group_indices=self._init.registered_group_indices,
            registered_shared_group_ids=self._init.registered_shared_group_ids,
            registered_row_offsets=self._init.registered_row_offsets,
            remote_pp_size=self._init.pp_size,
        )
        open_request = CompressionSessionOpen(
            metadata=metadata, nonce=nonce, profile=self._init.compression_profile
        )
        sock = self._zmq.socket(zmq.DEALER)
        sock.setsockopt(
            zmq.IDENTITY,
            f"vllm-mooncake-{self._init.tp_rank}-{time.monotonic_ns()}".encode(),
        )
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int((self._abort_timeout + 60) * 1000))

        owned: set[int] = set()
        inflight: dict[int, int] = {}
        decode_tasks: set[asyncio.Task] = set()
        issue_task: asyncio.Task | None = None
        recv_task: asyncio.Task | None = None
        session_error: BaseException | None = None
        safe_to_reuse = True
        open_attempted = False
        try:
            sock.connect(worker_addr)
            open_attempted = True
            await sock.send(self._encoder.encode(open_request))
            opened = self._response_decoder.decode(await sock.recv())
            if not isinstance(opened, CompressionSessionOpened):
                raise RuntimeError("Unexpected compression session response.")
            if opened.error:
                raise RuntimeError(opened.error)
            if opened.profile != self._init.compression_profile:
                raise RuntimeError("Mooncake compression profile mismatch")
            logical_block_count = opened.logical_block_count
            total_chunks = opened.total_chunks

            if total_chunks:
                assert request.num_tokens is not None
                session_tokens = suffix_token_counts(
                    request.num_tokens,
                    request.local_block_ids,
                    self._init.fa_group_block_sizes,
                )

                async def issue_all():
                    assert self._receiver_slots is not None
                    for chunk_id in range(total_chunks):
                        slot_id = await self._receiver_slots.get()
                        owned.add(slot_id)
                        inflight[chunk_id] = slot_id
                        pull = CompressionChunkPull(
                            transfer_id=request.transfer_id,
                            d_req_id=request.d_req_id,
                            chunk_id=chunk_id,
                            payload_addr=self._slot_payload[slot_id],
                            aux_addr=self._slot_aux[slot_id],
                            tail_addr=self._slot_tail[slot_id],
                            nonce=nonce,
                        )
                        await sock.send(self._encoder.encode(pull))

                async def recv_one():
                    return await sock.recv()

                issue_task = asyncio.create_task(issue_all())
                recv_task = asyncio.create_task(recv_one())
                received_chunks: set[int] = set()
                decoded_count = 0
                while decoded_count < total_chunks:
                    wait_set = set(decode_tasks)
                    wait_set.add(recv_task)
                    if issue_task is not None:
                        wait_set.add(issue_task)
                    done, _ = await asyncio.wait(
                        wait_set, return_when=asyncio.FIRST_COMPLETED
                    )
                    old_recv = recv_task
                    finished_issue = issue_task
                    if old_recv in done:
                        ready = self._response_decoder.decode(old_recv.result())
                        recv_task = asyncio.create_task(recv_one())
                        if not isinstance(ready, CompressionChunkReady):
                            raise RuntimeError("Unexpected compression chunk response.")
                        if ready.error:
                            raise RuntimeError(ready.error)
                        if ready.chunk_id in received_chunks:
                            raise RuntimeError(
                                f"Duplicate compression chunk {ready.chunk_id}."
                            )
                        if not 0 <= ready.chunk_id < total_chunks:
                            raise RuntimeError(
                                f"Compression chunk {ready.chunk_id} is outside "
                                f"the session range [0, {total_chunks})."
                            )
                        received_chunks.add(ready.chunk_id)
                        slot_id = inflight.pop(ready.chunk_id)
                        block_ids, tokens = self._chunk_metadata(
                            request.local_block_ids,
                            session_tokens,
                            ready.chunk_id,
                            logical_block_count,
                        )
                        job = DecodeChunkJob(
                            job_id=self._alloc_job(),
                            transfer_id=request.transfer_id,
                            d_req_id=request.d_req_id,
                            nonce=nonce,
                            chunk_id=ready.chunk_id,
                            slot_id=slot_id,
                            local_block_ids=block_ids,
                            valid_tokens_by_group=tokens,
                            codec_bytes=ready.codec_bytes,
                            u8_bytes=ready.u8_bytes,
                        )
                        task = asyncio.create_task(
                            self._decode_and_release(job, slot_id, owned)
                        )
                        decode_tasks.add(task)
                    if finished_issue is not None and finished_issue in done:
                        finished_issue.result()
                        issue_task = None
                    for task in done:
                        if task is old_recv or task is finished_issue:
                            continue
                        if task in decode_tasks:
                            decode_tasks.discard(task)
                            task.result()
                            decoded_count += 1
                await _cancel_tasks(recv_task)
                recv_task = None
                if decode_tasks:
                    await asyncio.gather(*tuple(decode_tasks))
            return set()
        except (Exception, asyncio.CancelledError) as exc:
            session_error = exc
            logger.exception(
                "Session helper compressed receive failed: d_req_id=%s transfer_id=%s",
                request.d_req_id,
                request.transfer_id,
            )
            try:
                await _cancel_tasks(issue_task, recv_task)
                issue_task = recv_task = None
                if open_attempted:
                    await self._send_abort(sock, request, nonce, str(exc))
            except Exception:
                safe_to_reuse = False
                self._fail_worker("Consumer cancellation was not acknowledged")
                logger.error(
                    "Session helper abort was not acknowledged: d_req_id=%s",
                    request.d_req_id,
                )
            if isinstance(exc, asyncio.CancelledError):
                raise
            return self._all_blocks(request.local_block_ids)
        finally:
            await _cancel_tasks(issue_task, recv_task)
            if decode_tasks:
                results = await asyncio.gather(
                    *tuple(decode_tasks), return_exceptions=True
                )
                for result in results:
                    if isinstance(result, BaseException) and session_error is None:
                        session_error = result
            if owned:
                drained = True
                try:
                    await self._drain_slots(sorted(owned))
                except Exception:
                    drained = False
                    safe_to_reuse = False
                    self._fail_worker("Could not drain consumer jobs and slots")
                    logger.exception(
                        "Session helper could not drain receiver slots %s",
                        sorted(owned),
                    )
                if safe_to_reuse and drained:
                    for slot_id in owned:
                        self._receiver_slots.put_nowait(slot_id)
                elif owned:
                    logger.error(
                        "Session helper quarantined receiver slots %s after "
                        "cancellation could not be drained.",
                        sorted(owned),
                    )
            sock.close(linger=0)

    async def _decode_and_release(
        self, job: DecodeChunkJob, slot_id: int, owned: set[int]
    ) -> None:
        assert self._receiver_slots is not None
        result = await self._submit(job)
        if result.error is not None:
            raise RuntimeError(result.error)
        owned.discard(slot_id)
        self._receiver_slots.put_nowait(slot_id)

    async def _send_abort(
        self,
        sock: zmq.asyncio.Socket,
        request: ConsumerRequest,
        nonce: str,
        reason: str,
    ) -> None:
        abort = CompressionAbort(
            transfer_id=request.transfer_id,
            d_req_id=request.d_req_id,
            reason=reason,
            nonce=nonce,
        )
        await sock.send(self._encoder.encode(abort))
        deadline = time.monotonic() + self._abort_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Producer did not acknowledge cancellation.")
            payload = await asyncio.wait_for(sock.recv(), remaining)
            ack = self._response_decoder.decode(payload)
            # Replies issued before Abort may already be queued on this socket.
            if isinstance(ack, (CompressionSessionOpened, CompressionChunkReady)):
                continue
            if (
                isinstance(ack, CompressionAbortAck)
                and ack.nonce == nonce
                and ack.transfer_id == request.transfer_id
                and ack.d_req_id == request.d_req_id
                and not ack.error
            ):
                return
            raise RuntimeError(
                "Producer did not acknowledge safe session cancellation."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _heartbeat(self) -> None:
        while not self._stopping:
            self._to_engine(HelperHeartbeat())
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._zmq = zmq.asyncio.Context()
        self._start_reader()
        if self._init.needs_sender:
            await self._start_producer()
        if self._init.needs_receiver:
            self._receiver_slots = asyncio.Queue()
            for slot_id in range(len(self._init.slot_addresses)):
                self._receiver_slots.put_nowait(slot_id)
            for address in self._init.slot_addresses:
                self._slot_payload.append(address.payload)
                self._slot_aux.append(address.aux)
                self._slot_tail.append(address.tail)
        self._to_engine(HelperReady(side_channel_port=self._side_channel_port))
        heartbeat = asyncio.create_task(self._heartbeat())
        self._tasks.add(heartbeat)
        heartbeat.add_done_callback(self._tasks.discard)
        await self._stop_event.wait()
        await self._shutdown()

    async def _shutdown(self) -> None:
        self._stopping = True
        if self._router is not None:
            self._router.close(linger=0)
        tasks = [task for task in self._tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await self._drain_jobs()
        except Exception:
            self._fail_worker("Shutdown could not drain engine jobs")
        if self._zmq is not None:
            self._zmq.term()
        if self._fatal_error is None:
            self._to_engine(Stopped())


def _session_process_main(conn, init_bytes: bytes) -> None:
    try:
        init = msgspec.msgpack.decode(init_bytes, type=InitMessage)
        service = _SessionService(conn, init)
        asyncio.run(service.run())
    except BaseException:
        logger.exception("Session helper process crashed")
    finally:
        conn.close()


class SessionProcess:
    """Engine-side handle to one out-of-process compression session helper."""

    def __init__(
        self,
        handler,
        init: InitMessage,
        finished_sending: queue.SimpleQueue,
        finished_recving: queue.SimpleQueue,
        invalid_block_ids: queue.SimpleQueue | queue.Queue,
        stats,
        device_id: int,
        failed_recving: queue.SimpleQueue | queue.Queue | None = None,
    ):
        self._handler = handler
        self._init = init
        self.finished_sending = finished_sending
        self.finished_recving = finished_recving
        self.invalid_block_ids = invalid_block_ids
        self.failed_recving = failed_recving
        self._stats = stats
        self._device_id = device_id
        self.side_channel_port = 0

        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(HelperToEngine)
        self._out: queue.SimpleQueue = queue.SimpleQueue()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._shutdown = False
        self._failed = threading.Event()
        self._failure_reason: str | None = None
        self._state_lock = threading.RLock()
        self._last_heartbeat = float("inf")
        self._jobs: set[Future] = set()

        context = multiprocessing.get_context("spawn")
        self._conn, child_conn = context.Pipe(duplex=True)
        self._child_conn = child_conn
        self._process = context.Process(
            target=_session_process_main,
            args=(child_conn, msgspec.msgpack.encode(init)),
            name=f"kvs-session-{init.tp_rank}",
            daemon=True,
        )
        self._reader_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None

        initializer = None
        initargs: tuple = ()
        if device_id >= 0:
            from vllm.platforms import current_platform

            initializer = current_platform.set_device
            initargs = (device_id,)
        self._pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="kvs-session-job",
            initializer=initializer,
            initargs=initargs,
        )

    def start(self) -> None:
        current = multiprocessing.current_process()
        was_daemon = current.daemon
        try:
            current.daemon = False
            self._process.start()
        finally:
            current.daemon = was_daemon
            self._child_conn.close()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="kvs-session-completion", daemon=True
        )
        self._reader_thread.start()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="kvs-session-out", daemon=True
        )
        self._writer_thread.start()
        if not self._ready.wait(_STARTUP_TIMEOUT_S):
            raise RuntimeError("Compression session helper did not become ready.")
        self.check_health()

    def check_health(self) -> None:
        if (
            not self._failed.is_set()
            and not self._shutdown
            and time.monotonic() - self._last_heartbeat > _HEARTBEAT_TIMEOUT_S
        ):
            self._mark_failed("helper heartbeat timed out")
        if self._failed.is_set():
            raise RuntimeError(
                "Mooncake compression session helper failed; worker must stop: "
                f"{self._failure_reason}"
            )

    def publish(self, message) -> None:
        with self._state_lock:
            self.check_health()
            self._out.put(self._encoder.encode(message))

    def _send(self, message) -> None:
        with self._state_lock:
            if not self._failed.is_set():
                self._out.put(self._encoder.encode(message))

    def _mark_failed(self, reason: str) -> None:
        with self._state_lock:
            if not self._failed.is_set():
                self._failure_reason = reason
                self._failed.set()
                self._ready.set()
        # There may still be local decodes or remote writes. Do not publish
        # finished/invalid blocks: that would let the scheduler reuse them.

    def _writer_loop(self) -> None:
        while True:
            try:
                data = self._out.get(timeout=0.2)
            except queue.Empty:
                continue
            if data is None:
                return
            try:
                self._conn.send_bytes(data)
            except (OSError, EOFError):
                self._mark_failed("helper pipe closed while sending")
                return

    def _reader_loop(self) -> None:
        while True:
            try:
                raw = self._conn.recv_bytes()
            except (EOFError, OSError):
                if not self._stopped.is_set():
                    self._mark_failed("helper process exited or pipe closed")
                return
            try:
                message = self._decoder.decode(raw)
            except Exception:
                self._mark_failed("invalid helper message")
                return
            if isinstance(message, HelperFailed):
                self._mark_failed(message.reason)
                continue
            if self._failed.is_set():
                continue
            if isinstance(message, HelperReady):
                self.side_channel_port = message.side_channel_port
                self._last_heartbeat = time.monotonic()
                self._ready.set()
            elif isinstance(message, HelperHeartbeat):
                self._last_heartbeat = time.monotonic()
            elif isinstance(message, FinalEvent):
                self._handle_final_event(message)
            elif isinstance(message, RecvDone):
                self._handle_recv_done(message)
            elif isinstance(message, Stopped):
                self._stopped.set()
            elif isinstance(
                message,
                (
                    OpenProducerJob,
                    EncodeChunkJob,
                    DecodeChunkJob,
                    DrainSlotsJob,
                ),
            ):
                with self._state_lock:
                    if not self._failed.is_set():
                        future = self._pool.submit(self._run_job, message)
                        self._jobs.add(future)
                        future.add_done_callback(self._job_finished)
            else:
                logger.warning("Unexpected session helper message %r", message)

    def _handle_final_event(self, message: FinalEvent) -> None:
        with self._state_lock:
            if self._failed.is_set():
                return
            if message.kind == "sending":
                self.finished_sending.put(message.req_id)
            elif message.kind == "sending_expired":
                self._stats.record_kv_expired_req()
                self.finished_sending.put(message.req_id)
            else:
                logger.warning("Unknown session event kind %s", message.kind)

    def _handle_recv_done(self, message: RecvDone) -> None:
        with self._state_lock:
            if self._failed.is_set():
                return
            if message.failed_blocks:
                if self.failed_recving is not None:
                    self.failed_recving.put(message.d_req_id)
                else:
                    self.invalid_block_ids.put(set(message.failed_blocks))
                self._stats.record_failed_recv()
            self.finished_recving.put(message.d_req_id)

    def _run_job(self, job) -> None:
        try:
            result = self._handler(job)
        except Exception as exc:
            logger.exception("Compression session job %s failed", type(job).__name__)
            result = JobDone(job_id=job.job_id, error=str(exc))
        self._send(result)

    def _job_finished(self, future: Future) -> None:
        with self._state_lock:
            self._jobs.discard(future)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_S
        with contextlib.suppress(Exception):
            self.publish(EngineStop())
        stopped = (
            self._stopped.wait(_SHUTDOWN_TIMEOUT_S)
            if not self._failed.is_set()
            else False
        )
        if not stopped:
            self._mark_failed("Helper stopped without proving memory quiescence")
        self._out.put(None)
        self._pool.shutdown(wait=False, cancel_futures=True)
        with self._state_lock:
            running = tuple(future for future in self._jobs if not future.done())
        if running:
            _, pending = wait(running, timeout=max(0.0, deadline - time.monotonic()))
            if pending:
                self._mark_failed("Engine jobs did not quiesce before shutdown")
        self._process.join(max(0.0, deadline - time.monotonic()))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(2.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(2.0)
        with contextlib.suppress(OSError):
            self._conn.close()
        if self._writer_thread is not None:
            self._writer_thread.join(0.1)
        self.check_health()
