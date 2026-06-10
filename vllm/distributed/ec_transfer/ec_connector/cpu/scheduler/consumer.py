# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer role for ECCPUConnector.

The consumer drives the transfer: for each announced ``mm_hash`` it allocates
destination blocks, sends an ``XferReq``, and on the ``XferAck`` registers the
producer with the metadata carried in that reply and issues a NIXL READ that
pulls the encoding into its mmap. A request is deferred until every feature it
needs is either locally cached or has finished reading.
"""

import time
from math import ceil
from typing import TYPE_CHECKING, Any

from vllm.distributed.ec_transfer.ec_connector.cpu.metadata import (
    XferAck,
    XferReq,
    XferStatus,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.common import (
    evict_and_alloc,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    PeerAddr,
    PendingRead,
    QuarantinedRead,
)
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

# How long to wait for an XferAck before giving up on a peer and encoding
# the feature locally instead.
CONSUMER_XFER_ACK_TIMEOUT_S = 2.0

# How long to let an in-flight READ run before giving up on it. On expiry the
# request falls back to local encode and its blocks are quarantined (NIXL has
# no abort) until the transfer reports terminal. Kept below the producer pin
# lease so the source stays pinned for the whole window.
CONSUMER_READ_TIMEOUT_S = 20.0


class ECCPUConsumer:
    """Consumer state machine over a duck-typed transport and transfer engine.

    Per-``mm_hash`` progress lives in ``_remote_encodings`` as a
    ``PendingRead`` (``read_handle is None`` ⇒ awaiting the ``XferAck``;
    set ⇒ READ in flight) or a ``None`` tombstone (a NACK that
    ``ensure_cache_available`` consumes to fall through to local encode).
    Completed reads move to ``_ready`` and then to ``_loaded`` once handed to
    the worker.
    """

    def __init__(
        self,
        region: ECSharedRegion,
        transport: Any,
        engine: Any,
        local_xfer_handle: int,
        compat_hash: str,
        hidden_dim: int,
        element_size: int,
        block_size_bytes: int,
    ) -> None:
        self._region = region
        self._transport = transport
        self._engine = engine
        self._local_xfer_handle = local_xfer_handle
        self._compat_hash = compat_hash
        self._hidden_dim = hidden_dim
        self._element_size = element_size
        self._block_size_bytes = block_size_bytes

        self._remote_encodings: dict[str, PendingRead | None] = {}
        self._ready: set[str] = set()
        self._loaded: dict[str, list[int]] = {}
        self._pending_reload: set[str] = set()
        # Reads given up on while possibly still DMA-ing; blocks held until
        # check_xfer_state reports terminal (see QuarantinedRead).
        self._quarantine: list[QuarantinedRead] = []

    def has_cache_item(self, identifier: str) -> bool:
        """True iff the bytes for `identifier` are in our local mmap, either
        because the read just completed (`_ready`, worker copies this step) or
        because the worker already copied them last step and we have not yet
        freed the blocks (`_loaded`).
        """
        return identifier in self._loaded or identifier in self._ready

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Ensure the mm_hash will be (re-)loaded into encoder_cache this step.

        Called by the scheduler after routing an EC-cached mm_hash to
        ``external_load_encoder_input``. The encoding is in the local mmap
        (``_loaded``), but ``encoder_cache`` may have been evicted by
        ``free_encoder_mm_hashes`` since it was last copied. Adding the
        mm_hash to ``_pending_reload`` makes ``build_loads`` include it in
        ``meta.loads`` so ``start_load_caches`` reloads mmap→GPU this step.
        If ``encoder_cache`` already holds it, ``start_load_caches`` skips it.
        """
        mm_hash = (
            request.mm_features[index].mm_hash or request.mm_features[index].identifier
        )
        if mm_hash in self._loaded:
            self._pending_reload.add(mm_hash)

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """Decide whether `request` can run this step, starting any reads needed.

        Returns False if the request is waiting on at least one in-flight read
        and should be deferred; True if every feature is locally cached or
        absent-from-announcement (and therefore falls through to vLLM's local
        encode).
        """
        params: dict[str, dict[str, Any]] = (
            getattr(request, "ec_transfer_params", None) or {}
        )
        if not params:
            return True
        logger.debug(
            "EC: ec_transfer_params req=%s params=%s",
            request.request_id,
            params,
        )
        logger.debug(
            "EC: ensure_cache_available req=%s num_computed=%d features=%d",
            request.request_id,
            num_computed_tokens,
            len(request.mm_features),
        )
        pending = False
        for feature in request.mm_features:
            pos = feature.mm_position
            # Make sure the requested mm object is relevant:
            if pos.offset + pos.length <= num_computed_tokens:
                continue
            mm_hash = feature.mm_hash or feature.identifier
            if self.has_cache_item(mm_hash):
                if mm_hash in self._loaded:
                    self._pending_reload.add(mm_hash)
                    logger.debug(
                        "EC: mm_hash=%s locally loaded, queued for reload",
                        mm_hash,
                    )
                else:
                    logger.debug("EC: mm_hash=%s already ready", mm_hash)
                continue
            if mm_hash in self._remote_encodings:
                if self._remote_encodings[mm_hash] is None:
                    # NACK tombstone: drop it, fall through to local
                    # encode. One-shot consumption is safe.
                    del self._remote_encodings[mm_hash]
                    logger.debug(
                        "EC: mm_hash=%s NACK tombstone consumed, "
                        "falling back to local encode",
                        mm_hash,
                    )
                    continue
                logger.debug("EC: mm_hash=%s read already in-flight", mm_hash)
                pending = True
                continue
            info = params.get(mm_hash)
            if info is None:
                # Not announced by any producer — fall through to local
                # encode.
                logger.debug(
                    "EC: mm_hash=%s not announced, falling back to local encode",
                    mm_hash,
                )
                continue
            # Verify the announced size matches the size derivable from
            # this feature's mm_position locally. The compat hash already
            # protects vllm_version/model/dtype/block_size; this guards
            # against a mismatch within otherwise-compatible peers (bug
            # in the producer or in the orchestrator's announcement).
            expected_size = pos.length * self._hidden_dim * self._element_size
            announced_size = int(info.get("size_bytes", -1))
            if announced_size != expected_size:
                logger.warning(
                    "EC: announced size_bytes=%d for mm_hash=%s does not match "
                    "locally-derived %d (length=%d, hidden_dim=%d, element_size=%d);"
                    " falling back to local encode",
                    announced_size,
                    mm_hash,
                    expected_size,
                    pos.length,
                    self._hidden_dim,
                    self._element_size,
                )
                continue
            try:
                self._start_read(mm_hash, info, expected_size)
            except Exception:
                logger.exception(
                    "EC: failed to start read for mm_hash=%s; "
                    "falling back to local encode",
                    mm_hash,
                )
                continue
            logger.debug(
                "EC: mm_hash=%s read started size_bytes=%d", mm_hash, expected_size
            )
            pending = True
        logger.debug(
            "EC: ensure_cache_available req=%s result=%s",
            request.request_id,
            "deferred" if pending else "ready",
        )
        return not pending

    def _fifo_alloc(self, n_blocks: int) -> list[int]:
        try:
            return self._region.alloc(n_blocks)
        except AllocationError:
            pass
        result = evict_and_alloc(
            n_blocks, self._loaded, self._region, protected=self._pending_reload
        )
        if result is not None:
            return result
        raise AllocationError(
            f"ECSharedRegion exhausted: cannot satisfy {n_blocks} blocks "
            f"even after evicting all evictable cache entries — operator "
            f"under-sized num_ec_blocks for the active workload"
        )

    def _start_read(self, mm_hash: str, info: dict[str, Any], size_bytes: int) -> None:
        """Allocate destination blocks and ask the producer for the source.

        The NIXL READ is issued later, in ``_handle_ack``, once the producer's
        ``XferAck`` supplies its fresh metadata and the source block indices.
        """
        n_blocks = max(1, ceil(size_bytes / self._block_size_bytes))
        indices = self._fifo_alloc(n_blocks)
        try:
            addr: PeerAddr = (info["peer_host"], int(info["peer_port"]))
            peer = self._transport.ensure_dealer(addr)
            self._transport.send_xfer_req(
                peer, XferReq(mm_hash=mm_hash, compatibility_hash=self._compat_hash)
            )
        except Exception:
            self._region.free(indices)
            raise
        self._remote_encodings[mm_hash] = PendingRead(
            addr=addr,
            dst_indices=indices,
            deadline=time.monotonic() + CONSUMER_XFER_ACK_TIMEOUT_S,
        )
        logger.debug(
            "EC: read requested mm_hash=%s n_blocks=%d peer=%s:%d",
            mm_hash,
            n_blocks,
            addr[0],
            addr[1],
        )

    def drain(self) -> None:
        """Advance every in-flight read one step: tear down dead peers, apply
        arriving acks, then poll running reads and expire stale ack waits.
        """
        for addr in self._transport.poll_dead_peers():
            self.on_peer_down(addr)
        for addr, ack in self._transport.poll_responses():
            self._handle_ack(addr, ack)
        now = time.monotonic()
        for mm_hash, pr in list(self._remote_encodings.items()):
            if pr is None:
                continue
            if pr.read_handle is not None:
                # Poll the in-flight READ; only if it is still running past its
                # budget do we give up. NIXL cannot abort it, so the blocks go
                # to quarantine (not freed) until the transfer reports terminal.
                if not self._poll_read(mm_hash, pr) and now > pr.deadline:
                    logger.warning(
                        "EC: READ for mm_hash=%s from %s:%d still in flight after "
                        "%.0fs; quarantining and falling back to local encode",
                        mm_hash,
                        pr.addr[0],
                        pr.addr[1],
                        CONSUMER_READ_TIMEOUT_S,
                    )
                    self._quarantine.append(
                        QuarantinedRead(pr.dst_indices, pr.read_handle)
                    )
                    self._remote_encodings[mm_hash] = None
            elif now > pr.deadline:
                logger.warning(
                    "EC: no XferAck for mm_hash=%s from %s:%d within %.1fs; "
                    "falling back to local encode",
                    mm_hash,
                    pr.addr[0],
                    pr.addr[1],
                    CONSUMER_XFER_ACK_TIMEOUT_S,
                )
                self._region.free(pr.dst_indices)
                self._remote_encodings[mm_hash] = None
        self._drain_quarantine()

    def _handle_ack(self, addr: PeerAddr, ack: XferAck) -> None:
        """Register the producer from the ack's fresh metadata and issue the
        READ on OK; tombstone and free the destination blocks on a NACK.
        """
        pr = self._remote_encodings.get(ack.mm_hash)
        if pr is None:
            # Unknown hash, or already tombstoned by peer-down / timeout.
            return
        if pr.read_handle is not None:
            # Read already in flight; a duplicate ack.
            return
        if ack.status != XferStatus.OK:
            if ack.status == XferStatus.NACK_INCOMPAT:
                logger.warning(
                    "EC: producer %s:%d reports an incompatible peer; "
                    "falling back to local encode for mm_hash=%s",
                    addr[0],
                    addr[1],
                    ack.mm_hash,
                )
            elif ack.status == XferStatus.NACK_MISSING:
                logger.info(
                    "EC: producer %s:%d no longer holds mm_hash=%s "
                    "(restart/evicted); falling back to local encode",
                    addr[0],
                    addr[1],
                    ack.mm_hash,
                )
            else:
                logger.warning(
                    "EC: producer %s:%d refused mm_hash=%s (status=%s); "
                    "falling back to local encode",
                    addr[0],
                    addr[1],
                    ack.mm_hash,
                    ack.status.name,
                )
            self._region.free(pr.dst_indices)
            self._remote_encodings[ack.mm_hash] = None
            return
        try:
            peer = self._transport.register_source(
                addr, ack.agent_metadata, ack.mem_descriptor
            )
            handle = self._engine.post_read(
                self._local_xfer_handle,
                pr.dst_indices,
                peer.remote_read_handle,
                ack.src_block_indices,
                notif_msg=ack.mm_hash.encode("utf-8"),
            )
        except Exception:
            logger.exception(
                "EC: failed to start READ for mm_hash=%s; falling back to local encode",
                ack.mm_hash,
            )
            self._region.free(pr.dst_indices)
            self._remote_encodings[ack.mm_hash] = None
            return
        pr.read_handle = handle
        pr.deadline = time.monotonic() + CONSUMER_READ_TIMEOUT_S
        logger.debug(
            "EC: READ started mm_hash=%s n_blocks=%d peer=%s:%d",
            ack.mm_hash,
            len(pr.dst_indices),
            addr[0],
            addr[1],
        )

    def _poll_read(self, mm_hash: str, pr: PendingRead) -> bool:
        """Check one in-flight READ. Returns True once it settles — promoted to
        `_ready` on completion, or tombstoned + freed on failure — and False
        while it is still in flight (`PROC`). A terminal state means no transfer
        can still touch the blocks, so freeing on failure here is DMA-safe.
        """
        assert pr.read_handle is not None
        try:
            state = self._engine.check_xfer_state(pr.read_handle)
        except Exception:
            logger.exception(
                "EC: check_xfer_state failed for mm_hash=%s; "
                "falling back to local encode",
                mm_hash,
            )
            self._engine.release_xfer_handle(pr.read_handle)
            self._region.free(pr.dst_indices)
            self._remote_encodings[mm_hash] = None
            return True
        if state == "DONE":
            self._engine.release_xfer_handle(pr.read_handle)
            self._ready.add(mm_hash)
            logger.debug("EC: read arrived mm_hash=%s", mm_hash)
            return True
        if state == "PROC":
            return False
        logger.warning(
            "EC: READ for mm_hash=%s in unexpected state %r; "
            "falling back to local encode",
            mm_hash,
            state,
        )
        self._engine.release_xfer_handle(pr.read_handle)
        self._region.free(pr.dst_indices)
        self._remote_encodings[mm_hash] = None
        return True

    def _drain_quarantine(self) -> None:
        """Release and free quarantined reads once their transfer is terminal.

        A terminal state (DONE/ERR, or a raised exception meaning the handle is
        no longer usable) guarantees nothing can still DMA into the blocks, so
        they are safe to release and return to the region. Entries still in
        `PROC` are kept for a later step.
        """
        still_pending: list[QuarantinedRead] = []
        for q in self._quarantine:
            try:
                state = self._engine.check_xfer_state(q.read_handle)
            except Exception:
                logger.exception(
                    "EC: check_xfer_state failed for quarantined read; "
                    "releasing %d block(s)",
                    len(q.dst_indices),
                )
                state = None
            if state == "PROC":
                still_pending.append(q)
                continue
            self._engine.release_xfer_handle(q.read_handle)
            self._region.free(q.dst_indices)
            logger.debug(
                "EC: quarantined read settled (state=%s); freed %d block(s)",
                state,
                len(q.dst_indices),
            )
        self._quarantine = still_pending

    def on_peer_down(self, addr: PeerAddr) -> None:
        """Tear down a peer and drop the reads bound to it so they retry.

        A peer going down is a transport failure, not a verdict on the data, so
        each affected read is forgotten (not tombstoned) and the next
        ensure_cache_available retries it against the — possibly restarted —
        producer; a still-dead producer then terminates via the XferAck
        timeout instead of looping. Blocks of a read that already posted its
        READ are quarantined (the transfer may still be DMA-ing); blocks of a
        read still awaiting its XferAck are freed immediately. Reads that have
        already completed (`_ready`) are left to be promoted.
        """
        evicted_agent_name = self._transport.evict_peer(addr)
        retrying = 0
        quarantined = 0
        for mm_hash, pr in list(self._remote_encodings.items()):
            if pr is None or pr.addr != addr:
                continue
            if mm_hash in self._ready:
                continue
            if pr.read_handle is not None:
                self._quarantine.append(QuarantinedRead(pr.dst_indices, pr.read_handle))
                quarantined += 1
            else:
                self._region.free(pr.dst_indices)
                retrying += 1
            del self._remote_encodings[mm_hash]
        logger.info(
            "EC: peer down addr=%s agent=%s retrying=%d quarantined=%d",
            addr,
            evicted_agent_name,
            retrying,
            quarantined,
        )

    def build_loads(self) -> dict[str, list[int]]:
        """Drain progress, promote arrived mm_hashes, re-emit cached reloads."""
        # (a) Advance in-flight reads and apply any arrivals.
        self.drain()
        # (b) Hand newly-arrived mm_hashes to the worker; move to loaded
        #     cache. Blocks stay allocated so that subsequent requests for
        #     the same mm_hash are re-served with a local mmap→GPU re-copy
        #     instead of a producer round-trip.
        loads: dict[str, list[int]] = {}
        for mm_hash in list(self._ready):
            pr = self._remote_encodings.pop(mm_hash, None)
            if pr is None:
                continue
            loads[mm_hash] = pr.dst_indices
            self._loaded[mm_hash] = pr.dst_indices
            logger.debug("EC: load issued mm_hash=%s", mm_hash)
        self._ready.clear()
        # (c) Re-serve cached entries requested this step via a local
        #     mmap→GPU re-copy.
        for mm_hash in self._pending_reload:
            if mm_hash not in loads:
                blocks = self._loaded.get(mm_hash)
                if blocks is not None:
                    loads[mm_hash] = blocks
                    logger.debug(
                        "EC: Local mmap cache hit mm_hash=%s",
                        mm_hash,
                    )
        self._pending_reload = set()
        return loads

    def shutdown(self) -> None:
        # on_peer_down may move in-flight reads into quarantine, so drain the
        # peers first and release every quarantined handle afterwards. The
        # region is torn down by the scheduler, so blocks need no freeing here.
        # release_xfer_handle logs (and absorbs) any failure itself.
        for addr in list(self._transport._peer_pool):
            self.on_peer_down(addr)
        for q in self._quarantine:
            self._engine.release_xfer_handle(q.read_handle)
        self._quarantine.clear()
        self._transport.shutdown()
