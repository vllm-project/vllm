# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered KV reads using the in-tree MoRI engine and explicit DMA ownership."""

import os
import queue
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Any
from uuid import uuid4

import msgspec
import torch

from vllm.logger import init_logger

from .async_load import AsyncRead
from .data import PeerMemory
from .native import description
from .registry import Domain
from .stats import PeerL1Stats

logger = init_logger(__name__)
# MoRI does not expose revocation proof for remote access to these registrations.
_PROCESS_LIFETIME: list[object] = []


@dataclass(frozen=True)
class Advertisement:
    engine: bytes
    memory: bytes
    compatibility: str


class RdmaReadSession:
    def __init__(
        self,
        wrapper: Any,
        session: Any,
        local_rows: int,
        local_segments: tuple[tuple[tuple[int, int, int], ...], ...],
        remote: PeerMemory,
        credential: bytes,
        descriptors: tuple[Any, ...] = (),
        remote_memory: Any = None,
    ):
        self.wrapper, self.session = wrapper, session
        self.local_rows, self.local_segments = local_rows, local_segments
        self.remote, self.credential = remote, credential
        self.descriptors = descriptors
        self.remote_memory = remote_memory
        self._first_post = True

    def _segments(self, read: AsyncRead) -> list[tuple[int, int, int]]:
        segments: list[tuple[int, int, int]] = []
        for group, source, target in read.job.blocks:
            if not (
                0 < source < self.remote.num_blocks and 0 < target < self.local_rows
            ):
                raise ValueError("RDMA peer row outside registered geometry")
            for src, dst in zip(
                self.remote.segments[group], self.local_segments[group], strict=True
            ):
                offset, stride, size = src
                local_offset, local_stride, local_size = dst
                if size != local_size:
                    raise ValueError("RDMA peer pages differ")
                copied = 0
                while copied < size:
                    width = min(size - copied, 16 << 20)
                    entry = (
                        local_offset + target * local_stride + copied,
                        offset + source * stride + copied,
                        width,
                    )
                    if (
                        segments
                        and segments[-1][2] + width <= 16 << 20
                        and segments[-1][0] + segments[-1][2] == entry[0]
                        and segments[-1][1] + segments[-1][2] == entry[1]
                    ):
                        local, remote, previous = segments.pop()
                        segments.append((local, remote, previous + width))
                    else:
                        segments.append(entry)
                    copied += width
        if not segments:
            raise ValueError("RDMA receive has no byte regions")
        return segments

    def run(
        self, read: AsyncRead, ready: Callable[[], bool], stop: threading.Event
    ) -> None:
        statuses: list[Any] = []
        unknown = False
        success = False
        reason = "exception"
        try:
            segments = self._segments(read)
            # A grant can outlive its owner while queued here. Reject a restarted
            # owner before posting against a cached MR/session and needing quarantine.
            try:
                hello = description(
                    self.remote.endpoint, self.remote.epoch, credential=self.credential
                )
            except ConnectionError:
                reason = "owner_changed"
                raise
            if msgspec.json.decode(hello, type=PeerMemory) != self.remote:
                reason = "owner_changed"
                raise ValueError("RDMA owner epoch or description changed")
            while not ready():
                if stop.is_set() or monotonic() >= read.deadline:
                    reason = "cancelled" if stop.is_set() else "deadline_before_post"
                    return
                stop.wait(0.001)
            for begin in range(0, len(segments), 64):
                batch = segments[begin : begin + 64]
                local, remote, sizes = map(list, zip(*batch))
                submission = self.wrapper.prepare_owned_read(
                    sizes, local, remote, self.remote_memory
                )
                read.resources.append(submission)
                if stop.is_set():
                    reason = "cancelled"
                    return
                if not read.begin(sum(sizes)):
                    return
                status = submission.status
                statuses.append(status)
                read.resources.append((self, local, remote, sizes))
                # An exception may follow partial posting without returning a status.
                unknown = True
                started = perf_counter()
                self.wrapper.submit_owned_read(submission)
                if self._first_post:
                    logger.info(
                        "Peer L1 RDMA first post owner=%s epoch=%d seconds=%.6f",
                        self.remote.endpoint,
                        self.remote.epoch,
                        perf_counter() - started,
                    )
                    self._first_post = False
                unknown = False
                while not status.Succeeded():
                    if status.Failed():
                        reason = "status_failed"
                        return
                    if stop.is_set() or monotonic() >= read.deadline:
                        reason = "cancelled" if stop.is_set() else "deadline_after_post"
                        return
                    stop.wait(0.001)
            success = monotonic() < read.deadline
        except Exception:
            logger.debug(
                "Peer RDMA receive failed request=%s",
                read.job.request_id,
                exc_info=True,
            )
        finally:
            query = (
                None
                if unknown
                else lambda: all(status.Succeeded() for status in statuses)
            )
            drained = not read.submitted or (query is not None and query())
            read.complete(success, drained=drained, drain_query=query, reason=reason)


class RdmaReadMover:
    def __init__(
        self,
        domain: Domain,
        storage: torch.UntypedStorage,
        rows: int,
        segments: tuple[tuple[tuple[int, int, int], ...], ...],
        compatibility: str,
        stats: PeerL1Stats,
    ):
        os.environ.setdefault("MORI_DISABLE_AUTO_XGMI", "1")
        if os.environ["MORI_DISABLE_AUTO_XGMI"] != "1":
            raise ValueError("RDMA peers require MORI_DISABLE_AUTO_XGMI=1")
        from mori.io import BackendType, IOEngine, IOEngineConfig

        from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
            MoRIIOWrapper,
        )

        self.domain, self.rows, self.segments = domain, rows, segments
        self.compatibility, self.stats = compatibility, stats
        self.tensor = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
            storage, 0, (storage.nbytes(),), (1,)
        )
        self.engine = IOEngine(
            f"peer-l1-{uuid4().hex}", IOEngineConfig(host=domain.host, port=0)
        )
        self.wrapper = MoRIIOWrapper(self.engine)
        self.wrapper.set_backend_type(BackendType.RDMA)
        started = perf_counter()
        memory = self.wrapper.register_local_tensor(self.tensor)
        self.advertisement = msgspec.json.encode(
            Advertisement(self.wrapper.get_agent_metadata(), memory, compatibility)
        )
        logger.info(
            "Peer L1 RDMA registered bytes=%d seconds=%.6f",
            storage.nbytes(),
            perf_counter() - started,
        )
        self._sessions: dict[tuple[str, int], RdmaReadSession] = {}
        self._reads: dict[str, AsyncRead] = {}
        self._stop = threading.Event()
        self._work: queue.Queue[tuple[AsyncRead, Callable[[], bool]] | None] = (
            queue.Queue(maxsize=128)
        )
        self._thread = threading.Thread(target=self._run, daemon=True, name="peer-rdma")
        self._thread.start()
        _PROCESS_LIFETIME.append(self)

    def _session(self, remote: PeerMemory) -> RdmaReadSession:
        key = remote.endpoint, remote.epoch
        if key not in self._sessions:
            started = perf_counter()
            advertised = msgspec.json.decode(remote.rdma, type=Advertisement)
            if advertised.compatibility != self.compatibility:
                raise ValueError("RDMA memory geometry differs")
            self.wrapper.register_remote_engine(advertised.engine)
            memory = self.wrapper.get_unpack_memory_metadata(advertised.memory)
            session = self.wrapper.build_session(
                self.wrapper.local_memory_metadata, memory
            )
            if session is None:
                raise RuntimeError("MoRI could not create an RDMA read session")
            self._sessions[key] = RdmaReadSession(
                self.wrapper,
                session,
                self.rows,
                self.segments,
                remote,
                self.domain.credential,
                descriptors=(memory, advertised),
                remote_memory=memory,
            )
            logger.info(
                "Peer L1 RDMA session owner=%s epoch=%d seconds=%.6f",
                remote.endpoint,
                remote.epoch,
                perf_counter() - started,
            )
        return self._sessions[key]

    def submit(self, read: AsyncRead, ready: Callable[[], bool]) -> None:
        self._reads[read.identity] = read
        try:
            self._work.put_nowait((read, ready))
        except queue.Full:
            read.cancel_unsubmitted("queue_full")

    def retire(self, identity: str) -> None:
        self._reads.pop(identity, None)

    def _run(self) -> None:
        torch.accelerator.set_device_index(self.tensor.device)
        while True:
            if self._stop.is_set() and self._work.empty():
                return
            item = self._work.get()
            if item is None:
                return
            read, ready = item
            if self._stop.is_set():
                read.cancel_unsubmitted()
                continue
            if read.snapshot()[0]:
                continue
            try:
                if read.job.memory is None:
                    raise ValueError("RDMA receive lacks its owner descriptor")
                session = self._session(read.job.memory)
                session.run(read, ready, self._stop)
            except Exception:
                read.complete(False, drained=not read.submitted)
                logger.debug(
                    "Peer RDMA setup failed request=%s",
                    read.job.request_id,
                    exc_info=True,
                )

    def close(self) -> None:
        self._stop.set()
        with suppress(queue.Full):
            self._work.put_nowait(None)
        # A provider may never finish an ERR-QP post. Retain ownership, not a join.
        self._thread.join(timeout=0.1)
