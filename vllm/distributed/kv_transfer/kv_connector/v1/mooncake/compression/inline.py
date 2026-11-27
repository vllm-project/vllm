# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, in-process compressed transfers using Mooncake's worker loops."""

from __future__ import annotations

import asyncio
import concurrent.futures
import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm import envs
from vllm.logger import init_logger

from .adapter import create_chunk_io
from .protocol import (
    CompressionAbort,
    CompressionAbortAck,
    CompressionChunkPull,
    CompressionChunkReady,
    CompressionRequest,
    CompressionResponse,
    CompressionSessionOpen,
    CompressionSessionOpened,
    wait_for_abort_ack,
)
from .tokens import chunk_token_counts, chunk_window, suffix_token_counts

if TYPE_CHECKING:
    from ..mooncake_connector import MooncakeConnectorWorker, PullReqMeta, SendBlockMeta

logger = init_logger(__name__)


@dataclass
class _Transfer:
    send_meta: SendBlockMeta
    local_blocks: list[list[int]]
    valid_tokens: dict[int, int]
    count: int
    chunks: int
    remote: str
    completed: int = 0
    seen: set[int] = field(default_factory=set)
    last_activity: float = field(default_factory=time.monotonic)


class InlineCompression:
    """Run independent codec slots alongside vLLM's Mooncake transport.

    The sender and receiver each own a bounded slot pool. A receiver keeps a
    slot until its network write and GPU decode have completed. Abort waits for
    all producer operations before acknowledging that a receiver can reuse it.
    """

    def __init__(self, worker: MooncakeConnectorWorker):
        self.worker = worker
        assert worker.compression is not None
        self.runtime = worker.compression
        self.chunk_io = create_chunk_io(self.runtime)
        self.profile = f"buffers-v1:{self.chunk_io.wire_profile}"
        self.timeout = float(envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT)
        self._request_decoder = msgspec.msgpack.Decoder(CompressionRequest)
        self._response_decoder = msgspec.msgpack.Decoder(CompressionResponse)
        self._sender_slots: asyncio.Queue[int] = asyncio.Queue()
        self._receiver_slots: asyncio.Queue[int] = asyncio.Queue()
        for slot in range(len(self.runtime.slots)):
            self._sender_slots.put_nowait(slot)
            self._receiver_slots.put_nowait(slot)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.runtime.slots),
            thread_name_prefix="vllm-mooncake-codec",
            initializer=worker._bind_sender_thread_device,
        )
        self._transfers: dict[tuple[str, str, str], _Transfer] = {}
        self._cancelled: dict[tuple[str, str, str], asyncio.Event] = {}
        self._closed: dict[tuple[str, str, str], float] = {}
        self._opening: set[tuple[str, str, str]] = set()
        self._rejected: dict[tuple[str, str, str], SendBlockMeta] = {}
        self._operations: dict[tuple[str, str, str], set[asyncio.Task]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._receiver_tasks: set[asyncio.Task] = set()
        self._fatal: str | None = None

    def check_health(self) -> None:
        if self._fatal is not None:
            raise RuntimeError(self._fatal)

    def fail(self, reason: str) -> None:
        self._fatal = reason

    def dispatch(self, identity: bytes, raw: bytes, socket: Any) -> None:
        request = self._request_decoder.decode(raw)
        if isinstance(request, CompressionSessionOpen):
            if len(request.metadata.req_blocks) != 1:
                raise ValueError("Expected exactly one request per compressed pull")
            req_id, (transfer_id, _) = next(iter(request.metadata.req_blocks.items()))
        else:
            req_id, transfer_id = request.d_req_id, request.transfer_id
        key = (transfer_id, req_id, request.nonce)
        task = asyncio.create_task(self._handle(identity, socket, key, request))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if not isinstance(request, CompressionAbort):
            self._operations.setdefault(key, set()).add(task)

            def retire(done: asyncio.Task) -> None:
                operations = self._operations.get(key)
                if operations is not None:
                    operations.discard(done)
                    if not operations:
                        self._operations.pop(key, None)

            task.add_done_callback(retire)

    async def _handle(self, identity: bytes, socket: Any, key: tuple, request: Any):
        try:
            if isinstance(request, CompressionSessionOpen):
                response = await self._open(key, request)
            elif isinstance(request, CompressionChunkPull):
                response = await self._send_chunk(key, request)
            else:
                response = await self._abort(key, request)
            await socket.send_multipart(
                (identity, self.worker._encoder.encode(response))
            )
        except Exception:
            logger.exception("Compressed Mooncake request failed for %s", key)
            self._fatal = "Compressed Mooncake control path failed"

    async def _open(self, key: tuple, request: CompressionSessionOpen):
        from ..mooncake_connector import SendBlockMeta

        if key in self._opening or key in self._transfers or key in self._closed:
            return CompressionSessionOpened(0, 0, error="Duplicate compressed pull")
        self._opening.add(key)
        cancelled = self._cancelled.setdefault(key, asyncio.Event())
        send_meta = self.worker.reqs_need_send.get(key[0])
        try:
            if request.profile != self.profile:
                raise ValueError("Mooncake compression profile mismatch")
            metadata = request.metadata
            if (
                metadata.remote_tp_size != self.worker.tp_size
                or metadata.remote_pp_size != self.worker.pp_size
                or metadata.remote_tp_rank != self.worker.tp_rank
            ):
                raise ValueError("Compressed Mooncake requires matching TP and PP")
            transfer_id, req_id, _ = key
            send_meta = self.worker.reqs_need_send.get(transfer_id)
            if send_meta is None:
                if transfer_id in self.worker._expired_transfer_ids:
                    raise ValueError("Producer KV has expired")
                send_meta = SendBlockMeta("", transfer_id, [], asyncio.Event())
                self.worker.reqs_need_send[transfer_id] = send_meta
            ready = asyncio.create_task(send_meta.ready.wait())
            stopped = asyncio.create_task(cancelled.wait())
            try:
                done, _ = await asyncio.wait(
                    (ready, stopped),
                    timeout=self.timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise TimeoutError("Timed out waiting for producer KV")
                if cancelled.is_set():
                    raise ValueError("Compressed pull was cancelled")
            finally:
                for task in (ready, stopped):
                    task.cancel()
                await asyncio.gather(ready, stopped, return_exceptions=True)
            requested = metadata.req_blocks[req_id][1]
            if requested and len(requested) != len(send_meta.local_block_ids):
                raise ValueError("Producer and consumer cache groups differ")
            local, remote = self.worker._match_compression_blocks(send_meta, requested)
            if len(local) != len(remote) or any(
                len(source) != len(destination)
                for source, destination in zip(local, remote)
            ):
                raise ValueError("Producer and consumer block tables do not match")
            sizes = self.worker._compression_block_sizes()
            count = len(remote[next(iter(sizes))]) if remote else 0
            if count:
                if send_meta.num_tokens is None:
                    raise ValueError("Missing producer token count")
                valid = suffix_token_counts(send_meta.num_tokens, remote, sizes)
            else:
                valid = {group: 0 for group in sizes}
            chunks = (
                (count + self.runtime.config.logical_blocks_per_chunk - 1)
                // self.runtime.config.logical_blocks_per_chunk
            ) * 2
            send_meta.sending += 1
            send_meta.need_send = 1
            transfer = _Transfer(
                send_meta,
                local,
                valid,
                count,
                chunks,
                f"{metadata.remote_hostname}:{metadata.remote_port}",
            )
            self._transfers[key] = transfer
            if chunks == 0:
                self._finish(key)
            return CompressionSessionOpened(count, chunks, profile=self.profile)
        except Exception as exc:
            self._closed[key] = time.monotonic() + self.timeout
            if send_meta is not None and send_meta.p_req_id:
                self._rejected[key] = send_meta
            if (
                send_meta is not None
                and not send_meta.p_req_id
                and not send_meta.ready.is_set()
                and self.worker.reqs_need_send.get(key[0]) is send_meta
            ):
                self.worker.reqs_need_send.pop(key[0], None)
            return CompressionSessionOpened(0, 0, error=str(exc))
        finally:
            self._opening.discard(key)

    def _chunk(
        self, blocks: list[list[int]], valid: dict[int, int], count: int, index: int
    ):
        start, length, part = chunk_window(
            index, count, self.runtime.config.logical_blocks_per_chunk
        )
        sizes = self.worker._compression_block_sizes()
        selected = [
            group[start : start + length] if i in sizes else []
            for i, group in enumerate(blocks)
        ]
        return selected, chunk_token_counts(valid, sizes, start, length), part

    async def _execute(self, fn, *args):
        future = asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await asyncio.shield(future)
            raise

    def _compress_and_send(
        self, slot_id: int, transfer: _Transfer, request: CompressionChunkPull
    ):
        blocks, tokens, part = self._chunk(
            transfer.local_blocks,
            transfer.valid_tokens,
            transfer.count,
            request.chunk_id,
        )
        try:
            plan = self.chunk_io.plan(slot_id, blocks, part, tokens)
            chunk = self.chunk_io.compress(plan)
            self.worker._send_compressed_blocks(chunk, transfer.remote, request.buffers)
            return CompressionChunkReady(
                request.chunk_id,
                tuple(length for _, length in chunk.buffers),
                chunk.descriptor,
            )
        except BaseException:
            try:
                self.runtime.streams[slot_id].synchronize()
            except Exception:
                self._fatal = "Could not drain the failed producer GPU slot"
                raise
            raise

    async def _send_chunk(self, key: tuple, request: CompressionChunkPull):
        transfer = self._transfers.get(key)
        if transfer is None or self._cancelled[key].is_set():
            return CompressionChunkReady(request.chunk_id, error="Pull is closed")
        if (
            request.chunk_id in transfer.seen
            or not 0 <= request.chunk_id < transfer.chunks
        ):
            return CompressionChunkReady(request.chunk_id, error="Invalid chunk")
        transfer.seen.add(request.chunk_id)
        transfer.last_activity = time.monotonic()
        slot = await self._sender_slots.get()
        try:
            if self._cancelled[key].is_set():
                raise RuntimeError("Pull was cancelled")
            response = await self._execute(
                self._compress_and_send, slot, transfer, request
            )
            transfer.completed += 1
            transfer.last_activity = time.monotonic()
            if transfer.completed == transfer.chunks:
                self._finish(key)
            return response
        except Exception as exc:
            return CompressionChunkReady(request.chunk_id, error=str(exc))
        finally:
            if self._fatal is None:
                self._sender_slots.put_nowait(slot)

    def _finish(self, key: tuple) -> None:
        self._closed[key] = time.monotonic() + self.timeout
        transfer = self._transfers.pop(key, None)
        if transfer is None:
            return
        send_meta = transfer.send_meta
        send_meta.sending -= 1
        send_meta.sent += 1
        if (
            send_meta.sent == send_meta.need_send
            and self.worker.reqs_need_send.pop(send_meta.transfer_id, None) is not None
        ):
            self.worker._finished_sending.put(send_meta.p_req_id)

    async def _abort(self, key: tuple, request: CompressionAbort):
        self._cancelled.setdefault(key, asyncio.Event()).set()
        self._closed[key] = time.monotonic() + self.timeout
        pending = tuple(self._operations.get(key, ()))
        try:
            if pending:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), self.timeout
                )
            if self._fatal is not None:
                return CompressionAbortAck(
                    request.transfer_id,
                    request.d_req_id,
                    request.nonce,
                    error=self._fatal,
                )
            self._finish(key)
            send_meta = self._rejected.pop(key, None)
            if (
                send_meta is not None
                and send_meta.ready.is_set()
                and send_meta.sending == 0
                and self.worker.reqs_need_send.get(request.transfer_id) is send_meta
                and self.worker.reqs_need_send.pop(request.transfer_id, None)
                is send_meta
            ):
                self.worker._finished_sending.put(send_meta.p_req_id)
            return CompressionAbortAck(
                request.transfer_id, request.d_req_id, request.nonce
            )
        except TimeoutError:
            self._fatal = "Compressed Mooncake abort could not drain active writes"
            return CompressionAbortAck(
                request.transfer_id, request.d_req_id, request.nonce, error=self._fatal
            )

    def expire(self) -> None:
        now = time.monotonic()
        for key, transfer in tuple(self._transfers.items()):
            if now - transfer.last_activity > self.timeout and not self._operations.get(
                key
            ):
                self._cancelled[key].set()
                self._finish(key)
        for key, deadline in tuple(self._closed.items()):
            if deadline <= now and key not in self._operations:
                self._closed.pop(key, None)
                self._cancelled.pop(key, None)
                self._rejected.pop(key, None)

    def _decompress(self, slot_id, pull_meta, count, valid, ready):
        blocks, tokens, part = self._chunk(
            pull_meta.local_block_ids, valid, count, ready.chunk_id
        )
        try:
            plan = self.chunk_io.plan(slot_id, blocks, part, tokens)
            self.chunk_io.decompress(
                plan, ready.lengths, ready.descriptor
            ).synchronize()
        except BaseException:
            try:
                self.runtime.streams[slot_id].synchronize()
            except Exception:
                self._fatal = "Could not drain the failed consumer GPU slot"
                raise
            raise

    async def receive(self, worker_addr: str, pull_meta: PullReqMeta) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._receiver_tasks.add(task)
        key = (pull_meta.transfer_id, pull_meta.d_req_id, secrets.token_hex(12))
        metadata = self.worker._build_pull_metadata(
            {pull_meta.d_req_id: pull_meta}, include_layout=True
        )
        request = CompressionSessionOpen(metadata, nonce=key[2], profile=self.profile)
        sock = self.worker.async_zmq_ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int((self.timeout + 60) * 1000))
        owned: set[int] = set()
        inflight: dict[int, int] = {}
        decodes: set[asyncio.Task] = set()
        issuer = receiver = None
        safe = True
        failure: BaseException | None = None
        try:
            sock.connect(worker_addr)
            await sock.send(self.worker._encoder.encode(request))
            opened = self._response_decoder.decode(await sock.recv())
            if not isinstance(opened, CompressionSessionOpened) or opened.error:
                raise RuntimeError(
                    getattr(opened, "error", None) or "Invalid open reply"
                )
            if opened.profile != self.profile:
                raise RuntimeError("Mooncake compression profile mismatch")
            sizes = self.worker._compression_block_sizes()
            expected = (
                len(pull_meta.local_block_ids[next(iter(sizes))])
                if pull_meta.local_block_ids
                else 0
            )
            if expected:
                if pull_meta.num_tokens is None:
                    raise ValueError("Missing consumer token count")
                valid = suffix_token_counts(
                    pull_meta.num_tokens, pull_meta.local_block_ids, sizes
                )
            else:
                valid = {group: 0 for group in sizes}
            if opened.logical_block_count != expected:
                raise ValueError("Producer and consumer block counts differ")
            expected_chunks = (
                (expected + self.runtime.config.logical_blocks_per_chunk - 1)
                // self.runtime.config.logical_blocks_per_chunk
            ) * 2
            if opened.total_chunks != expected_chunks:
                raise ValueError("Producer and consumer chunk counts differ")

            async def issue():
                for chunk in range(opened.total_chunks):
                    slot_id = await self._receiver_slots.get()
                    owned.add(slot_id)
                    inflight[chunk] = slot_id
                    message = CompressionChunkPull(
                        key[0],
                        key[1],
                        chunk,
                        self.chunk_io.slot_buffers(slot_id),
                        key[2],
                    )
                    await sock.send(self.worker._encoder.encode(message))

            async def decompress_and_release(slot_id, response):
                await self._execute(
                    self._decompress, slot_id, pull_meta, expected, valid, response
                )
                owned.remove(slot_id)
                self._receiver_slots.put_nowait(slot_id)

            issuer = asyncio.create_task(issue())
            receiver = (
                asyncio.ensure_future(sock.recv()) if opened.total_chunks else None
            )
            completed = 0
            while completed < opened.total_chunks:
                pending = set(decodes)
                if issuer is not None:
                    pending.add(issuer)
                if receiver is not None:
                    pending.add(receiver)
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                if receiver in done:
                    response = self._response_decoder.decode(receiver.result())
                    if (
                        not isinstance(response, CompressionChunkReady)
                        or response.error
                    ):
                        raise RuntimeError(
                            getattr(response, "error", None) or "Invalid chunk reply"
                        )
                    slot_id = inflight.pop(response.chunk_id)
                    decodes.add(
                        asyncio.create_task(decompress_and_release(slot_id, response))
                    )
                    receiver = asyncio.ensure_future(sock.recv())
                if issuer in done:
                    issuer.result()
                    issuer = None
                for done_task in done & decodes:
                    decodes.remove(done_task)
                    done_task.result()
                    completed += 1
            self.worker._complete_pull_task(pull_meta)
        except (Exception, asyncio.CancelledError) as exc:
            failure = exc
            logger.warning("Compressed pull %s failed: %s", pull_meta.d_req_id, exc)
            for pending_task in (issuer, receiver):
                if pending_task is not None:
                    pending_task.cancel()
            await asyncio.gather(
                *(t for t in (issuer, receiver) if t is not None),
                return_exceptions=True,
            )
            try:
                abort = CompressionAbort(key[0], key[1], str(exc), key[2])
                await sock.send(self.worker._encoder.encode(abort))
                await wait_for_abort_ack(
                    sock, self._response_decoder, abort, self.timeout
                )
            except Exception:
                safe = False
                self._fatal = (
                    "Compressed Mooncake abort could not confirm remote writes stopped"
                )
                logger.exception(self._fatal)
        finally:
            for pending_task in (issuer, receiver):
                if pending_task is not None:
                    pending_task.cancel()
            await asyncio.gather(
                *(t for t in (issuer, receiver) if t is not None),
                return_exceptions=True,
            )
            if decodes:
                outcomes = await asyncio.gather(*decodes, return_exceptions=True)
                if any(isinstance(result, BaseException) for result in outcomes):
                    failure = failure or RuntimeError("Compressed GPU decode failed")
            if self._fatal is not None:
                safe = False
            if safe:
                for slot_id in owned:
                    self._receiver_slots.put_nowait(slot_id)
            if failure is not None and safe:
                self.worker._handle_failed_recv(
                    {pull_meta.d_req_id: pull_meta}, [pull_meta.d_req_id], str(failure)
                )
            sock.close(linger=0)
            self._receiver_tasks.discard(task)

    async def _close(self, tasks: set[asyncio.Task]) -> None:
        for event in self._cancelled.values():
            event.set()
        for task in tuple(tasks):
            task.cancel()
        if tasks:
            await asyncio.wait_for(
                asyncio.gather(*tuple(tasks), return_exceptions=True), self.timeout
            )

    def shutdown(self) -> None:
        for loop, tasks in (
            (self.worker.receiver_loop, self._receiver_tasks),
            (self.worker.sender_loop, self._tasks),
        ):
            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self._close(tasks), loop)
                future.result(timeout=self.timeout + 1)
        if self._fatal is not None:
            raise RuntimeError(self._fatal)
        self._executor.shutdown(wait=True)
