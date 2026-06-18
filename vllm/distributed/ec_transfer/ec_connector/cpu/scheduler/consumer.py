# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer role for ECCPUConnector."""

import threading
import time
from math import ceil
from typing import TYPE_CHECKING, Any

from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECRegionContext
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.common import (
    evict_and_alloc,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.control.zmq import (
    ZmqClientTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.data.base import (
    DataTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.session import (
    ConsumerSession,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import PeerAddr
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import AllocationError
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

CONSUMER_XFER_ACK_TIMEOUT_S = 2.0


class ECCPUConsumer:
    """Consumer role for ECCPUConnector.

    Owns a pool of ConsumerSession objects, one per producer peer. Each step:
      1. The first call to ensure_cache_available() polls the ZMQ transport
         and dispatches messages to all sessions (_poll_step).
      2. Sessions decode XferAcks, advance ConsumerXfer state machines,
         and accumulate results internally.
      3. build_loads() collects results, promotes completed reads to the local
         cache, frees failed-read blocks, and resets per-step state.

    _local_encodings and _blocks are shared with the producer role in ec_both
    deployments and are accessed under _lock.
    """

    def __init__(
        self,
        memory_context: ECRegionContext,
        transport: ZmqClientTransport,
        data: DataTransport,
        compat_hash: str,
        local_encodings: dict[str, None],
        blocks: dict[str, list[int]],
        lock: threading.Lock,
    ) -> None:
        self._memory_context = memory_context
        self._transport = transport
        self._data = data
        self._compat_hash = compat_hash

        self._lock = lock
        self._local_encodings: dict[str, None] = local_encodings
        self._blocks: dict[str, list[int]] = blocks

        # One long-lived session per producer peer.
        self._sessions: dict[PeerAddr, ConsumerSession] = {}

        # mm_hashes with an active ConsumerXfer in some session.
        self._in_flight: set[str] = set()

        # One-shot tombstones consumed by ensure_cache_available to fall
        # through to local encode without retrying the same mm_hash this step.
        self._tombstones: set[str] = set()

        # Reads that completed this step; promoted to local cache in build_loads.
        self._step_completed: set[str] = set()

        # Locally cached mm_hashes pinned for mmap->GPU re-copy this step.
        self._pending_reload: set[str] = set()

    # ── public API ────────────────────────────────────────────────────────────

    def has_cache_item(self, identifier: str) -> bool:
        return identifier in self._local_encodings

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash = (
            request.mm_features[index].mm_hash or request.mm_features[index].identifier
        )
        with self._lock:
            if mm_hash in self._local_encodings:
                if mm_hash not in self._pending_reload:
                    self._memory_context.region.pin(self._blocks[mm_hash])
                self._pending_reload.add(mm_hash)

    def ensure_cache_available(
        self,
        request: "Request",
        num_computed_tokens: int,
        first_in_batch: bool = False,
    ) -> bool:
        """Return False if any feature needs a still-in-flight remote read."""
        if first_in_batch:
            self._poll_step()

        params: dict[str, dict[str, Any]] = (
            getattr(request, "ec_transfer_params", None) or {}
        )
        if not params:
            return True

        pending = False
        for feature in request.mm_features:
            pos = feature.mm_position
            if pos.offset + pos.length <= num_computed_tokens:
                continue
            mm_hash = feature.mm_hash or feature.identifier

            with self._lock:
                is_local = mm_hash in self._local_encodings
                if is_local:
                    if mm_hash not in self._pending_reload:
                        self._memory_context.region.pin(self._blocks[mm_hash])
                    self._pending_reload.add(mm_hash)
            if is_local:
                logger.debug(
                    "EC: mm_hash=%s locally cached, queued for reload", mm_hash
                )
                continue

            if mm_hash in self._in_flight:
                logger.debug("EC: mm_hash=%s read in-flight", mm_hash)
                pending = True
                continue

            if mm_hash in self._step_completed:
                # Read finished this step but build_loads() has not promoted it
                # to _local_encodings yet. Defer one step rather than issuing a
                # redundant second read; next step it will be served locally.
                logger.debug(
                    "EC: mm_hash=%s completed this step, awaiting promote", mm_hash
                )
                pending = True
                continue

            if mm_hash in self._tombstones:
                self._tombstones.discard(mm_hash)
                logger.debug("EC: mm_hash=%s tombstone consumed; local encode", mm_hash)
                continue

            info = params.get(mm_hash)
            if info is None:
                continue

            expected_size = (
                pos.length
                * self._memory_context.hidden_dim
                * self._memory_context.element_size
            )
            if int(info.get("size_bytes", -1)) != expected_size:
                logger.warning(
                    "EC: size mismatch for mm_hash=%s; falling back to local encode",
                    mm_hash,
                )
                continue

            try:
                self._start_xfer(mm_hash, info, expected_size)
            except Exception:
                logger.exception(
                    "EC: failed to start xfer mm_hash=%s; local encode", mm_hash
                )
                continue

            self._in_flight.add(mm_hash)
            pending = True

        logger.debug(
            "EC: req=%s %s", request.request_id, "deferred" if pending else "ready"
        )
        return not pending

    def build_loads(self) -> dict[str, list[int]]:
        """Promote completed reads and re-serve reloads.

        Session results were already drained by _poll_step() earlier this step
        (the only place session.poll() advances xfer state machines), so
        _step_completed / _tombstones / etc. are up to date here.
        """
        loads: dict[str, list[int]] = {}
        for mm_hash in self._step_completed:
            if mm_hash in self._blocks:
                loads[mm_hash] = self._blocks[mm_hash]
                with self._lock:
                    self._local_encodings[mm_hash] = None
                logger.debug("EC: load issued mm_hash=%s", mm_hash)
        self._step_completed.clear()

        for mm_hash in self._pending_reload:
            if mm_hash not in loads and mm_hash in self._local_encodings:
                loads[mm_hash] = self._blocks[mm_hash]
                logger.debug("EC: local mmap re-serve mm_hash=%s", mm_hash)
            self._memory_context.region.unpin(self._blocks[mm_hash])
        self._pending_reload = set()

        return loads

    def shutdown(self) -> None:
        for mm_hash in self._pending_reload:
            blocks = self._blocks.get(mm_hash)
            if blocks is not None:
                self._memory_context.region.unpin(blocks)
        self._pending_reload = set()
        for session in list(self._sessions.values()):
            session.close()
        self._sessions.clear()
        self._transport.close()

    # ── internal ──────────────────────────────────────────────────────────────

    def _poll_step(self) -> None:
        """Poll all connections once per step and advance every session.

        Called exactly once per scheduling step: ensure_cache_available()
        invokes it only when the scheduler passes first_in_batch=True, which
        the scheduler arms once per step.
        """
        now = time.monotonic()
        all_messages = self._transport.poll()  # dict[PeerAddr, list[bytes]]

        for addr, session in list(self._sessions.items()):
            session.poll(all_messages.get(addr, []), now)

        for addr in self._transport.poll_dead():
            self._on_peer_down(addr)

        self._collect_session_results()

    def _collect_session_results(self) -> None:
        """Drain accumulated results from all active sessions."""
        for session in self._sessions.values():
            self._process_session_results(session)

    def _process_session_results(self, session: "ConsumerSession") -> None:
        """Apply one session's accumulated results to consumer state."""
        r = session.take_results()

        for mm_hash in r.completed:
            self._in_flight.discard(mm_hash)
            self._step_completed.add(mm_hash)

        for mm_hash in r.tombstoned:
            self._in_flight.discard(mm_hash)
            blocks = self._blocks.pop(mm_hash, None)
            if blocks:
                self._memory_context.region.free(blocks)
            self._tombstones.add(mm_hash)

        for mm_hash in r.quarantined:
            # DMA still running — blocks stay alive in the session's quarantine.
            self._in_flight.discard(mm_hash)
            self._tombstones.add(mm_hash)

        for mm_hash in r.cancelled:
            # Peer died while waiting for XferAck — allow retry, no tombstone.
            self._in_flight.discard(mm_hash)
            blocks = self._blocks.pop(mm_hash, None)
            if blocks:
                self._memory_context.region.free(blocks)

        for mm_hash, block_indices in r.settled:
            # Quarantine cleared — DMA finished, blocks safe to free.
            self._memory_context.region.free(block_indices)

    def _start_xfer(self, mm_hash: str, info: dict[str, Any], size_bytes: int) -> None:
        n_blocks = max(1, ceil(size_bytes / self._memory_context.block_size_bytes))
        indices = self._fifo_alloc(n_blocks)
        self._blocks[mm_hash] = indices

        addr: PeerAddr = (info["peer_host"], int(info["peer_port"]))
        if addr not in self._sessions:
            zmq_conn = self._transport.connect(addr)
            self._sessions[addr] = ConsumerSession(
                addr=addr,
                zmq_conn=zmq_conn,
                transport=self._transport,
                data=self._data,
                compat_hash=self._compat_hash,
            )

        deadline = time.monotonic() + CONSUMER_XFER_ACK_TIMEOUT_S
        try:
            self._sessions[addr].start_xfer(mm_hash, indices, deadline)
        except Exception:
            self._memory_context.region.free(self._blocks.pop(mm_hash))
            raise

        logger.debug(
            "EC: xfer started mm_hash=%s peer=%s:%d", mm_hash, addr[0], addr[1]
        )

    def _on_peer_down(self, addr: PeerAddr) -> None:
        session = self._sessions.pop(addr, None)
        if session is None:
            return
        session.on_peer_down()
        # Collect results from this specific session (already removed from _sessions,
        # so _collect_session_results won't see it).
        self._process_session_results(session)
        session.close()
        logger.info("EC: peer down addr=%s", addr)

    def _fifo_alloc(self, n_blocks: int) -> list[int]:
        try:
            return self._memory_context.region.alloc(n_blocks)
        except AllocationError:
            pass
        with self._lock:
            result = evict_and_alloc(
                n_blocks,
                self._local_encodings,
                self._blocks,
                self._memory_context.region,
                skip_pinned=True,
            )
        if result is not None:
            return result
        raise AllocationError(
            f"ECSharedRegion exhausted: cannot allocate {n_blocks} blocks"
        )
