# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer role for ECCPUConnector.

The producer is the data source for a consumer-initiated NIXL READ. Its mmap
region is registered with NIXL once at startup; thereafter it serves, on each
``XferReq``, the source block indices for an ``mm_hash`` plus its own current
NIXL agent metadata, and pins those blocks until the consumer's READ completes.
A completion notification releases the pin; a per-read deadline releases it if
the notification never arrives.
"""

import threading
import time
from math import ceil
from typing import TYPE_CHECKING, Any

from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECRegionContext
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.common import (
    evict_and_alloc,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.metadata import (
    EC_CONNECTOR_VERSION,
    XferAck,
    XferReq,
    XferStatus,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import PeerAddr, PinnedEncoding
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    AllocationError,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Force-unpin backstop: a source pin whose completion notif never arrives
# (consumer crash / network loss) is reclaimed after this many seconds. Kept
# above the consumer's read budget (CONSUMER_READ_TIMEOUT_S) plus round-trip
# slack, so the source stays pinned until the consumer has stopped accepting
# the read — otherwise a reused block could feed a stale-data completion.
PRODUCER_PIN_LEASE_S = 30.0


class ECCPUProducer:
    """Producer state machine over a duck-typed transfer engine.

    The encode pipeline (``update_state_after_alloc`` → ``build_saves``) runs on
    the main thread; ``handle_xfer_req`` and ``poll`` run on the transport's
    router thread. ``_local_encodings`` (mm_hash → None ordered set) and
    ``_blocks`` (mm_hash → mmap block indices) are shared with the consumer role
    on the same node (ec_both) and guarded by ``_lock``; ``_pinned_encodings`` is
    touched only by the router thread and so needs none.
    """

    def __init__(
        self,
        memory_context: ECRegionContext,
        engine: Any,
        compat_hash: str,
        addr: PeerAddr,
        local_encodings: dict[str, None],
        blocks: dict[str, list[int]],
        lock: threading.Lock,
    ) -> None:
        self._memory_context = memory_context
        self._engine = engine
        self._compat_hash = compat_hash
        self._addr = addr

        self._pending_new_encodings: dict[str, int] = {}
        # Set of mm_hashes whose blocks have been allocated this step but not
        # yet promoted to _local_encodings. Block indices live in _blocks.
        self._pending_save: set[str] = set()

        # Shared with the consumer role (both owned by ECCPUScheduler).
        self._lock = lock
        self._local_encodings: dict[str, None] = local_encodings
        self._blocks: dict[str, list[int]] = blocks

        # Router-thread-only state. No lock needed.
        self._pinned_encodings: dict[str, PinnedEncoding] = {}

    # ── router-thread methods ─────────────────────────────────────────────────

    def has_pending_pins(self) -> bool:
        """True while any source is pinned for an in-flight read.

        The transport uses this to poll completion notifs promptly (short
        timeout) while reads are outstanding and idle otherwise.
        """
        return bool(self._pinned_encodings)

    def handle_xfer_req(self, identity: bytes, req: XferReq) -> XferAck:
        """Grant or refuse a consumer's read request.

        On grant: pin the source blocks and return them together with the
        producer's fresh NIXL metadata so the consumer can register and READ.
        The reply is synchronous — the transport sends the returned ``XferAck``
        straight back to ``identity``.
        """
        if req.connector_version != EC_CONNECTOR_VERSION:
            logger.warning(
                "EC: incompatible XferReq version theirs=%d ours=%d for "
                "mm_hash=%s; consumer will fall back to local encode",
                req.connector_version,
                EC_CONNECTOR_VERSION,
                req.mm_hash,
            )
            return XferAck(mm_hash=req.mm_hash, status=XferStatus.NACK_VERSION)

        if req.compatibility_hash != self._compat_hash:
            logger.warning(
                "EC: incompatible XferReq (compat theirs=%s ours=%s) for "
                "mm_hash=%s; consumer will fall back to local encode",
                req.compatibility_hash,
                self._compat_hash,
                req.mm_hash,
            )
            return XferAck(mm_hash=req.mm_hash, status=XferStatus.NACK_INCOMPAT)

        # Read the block list and pin under the same lock acquisition so a
        # concurrent eviction on the main thread cannot reclaim the blocks
        # between the lookup and the pin.
        with self._lock:
            if req.mm_hash not in self._local_encodings:
                logger.warning(
                    "EC: XferReq mm_hash=%s not in local cache (evicted or "
                    "never produced — likely a producer restart); NACKing, "
                    "consumer will recompute locally",
                    req.mm_hash,
                )
                return XferAck(mm_hash=req.mm_hash, status=XferStatus.NACK_MISSING)
            block_indices = self._blocks[req.mm_hash]
            self._memory_context.region.pin(block_indices)

        deadline = time.monotonic() + PRODUCER_PIN_LEASE_S
        pin = self._pinned_encodings.get(req.mm_hash)
        if pin is None:
            self._pinned_encodings[req.mm_hash] = PinnedEncoding(deadlines=[deadline])
        else:
            pin.deadlines.append(deadline)

        logger.debug(
            "EC: read granted mm_hash=%s n_blocks=%d", req.mm_hash, len(block_indices)
        )
        return XferAck(
            mm_hash=req.mm_hash,
            status=XferStatus.OK,
            src_block_indices=block_indices,
            agent_metadata=self._engine._agent_metadata,
            mem_descriptor=self._engine._mem_descriptor_bytes,
        )

    def poll(self) -> None:
        """One router-loop iteration of pin maintenance: release pins for reads
        that completed, then release pins whose deadline has passed.
        """
        self._drain_notifs()
        self._sweep_timeouts()

    def _drain_notifs(self) -> None:
        """Release one source pin per arriving completion notification."""
        try:
            notifs = self._engine.get_new_notifs()
        except Exception:
            logger.exception("EC: get_new_notifs failed")
            return
        for msgs in notifs.values():
            for msg in msgs:
                mm_hash = msg.decode("utf-8")
                logger.debug(
                    "EC: read complete mm_hash=%s; releasing source pin", mm_hash
                )
                self._unpin_once(mm_hash)

    def _unpin_once(self, mm_hash: str) -> None:
        """Release one pin reference on ``mm_hash``: unpin the blocks once and
        drop the soonest-expiring deadline, forgetting the encoding once its
        last reference is released.

        ``deadlines`` stays sorted ascending — every appended deadline is
        ``monotonic() + lease`` and entries leave from the front — so
        ``pop(0)`` removes the reference closest to timing out. A call for an
        encoding with no live pins is a no-op, which keeps the total number of
        unpins equal to the number of grants even if a completion notification
        races a deadline expiry for the same read.
        """
        pin = self._pinned_encodings.get(mm_hash)
        if pin is None:
            return
        with self._lock:
            blocks = self._blocks.get(mm_hash)
            if blocks is not None:
                self._memory_context.region.unpin(blocks)
        pin.deadlines.pop(0)
        if not pin.deadlines:
            del self._pinned_encodings[mm_hash]

    def _sweep_timeouts(self) -> None:
        """Release pins whose deadline has passed (the expired entries are the
        sorted prefix, so ``_unpin_once`` removes exactly them from the front).
        """
        now = time.monotonic()
        for mm_hash, pin in list(self._pinned_encodings.items()):
            n_expired = sum(1 for d in pin.deadlines if d < now)
            if not n_expired:
                continue
            logger.warning(
                "EC: %d read pin(s) for mm_hash=%s passed the %.1fs deadline "
                "with no completion notification (consumer crash or network "
                "loss); releasing the source",
                n_expired,
                mm_hash,
                PRODUCER_PIN_LEASE_S,
            )
            for _ in range(n_expired):
                self._unpin_once(mm_hash)

    # ── main-thread methods ───────────────────────────────────────────────────

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """Probe whether the mmap region can hold this request's encoder outputs.

        Performs a speculative alloc-then-free for each feature not already
        tracked or cached. Returns False if the region is exhausted (all blocks
        pinned by in-flight NIXL reads), causing the scheduler to defer the
        request. The actual allocation happens later in build_saves via the
        normal _fifo_alloc path.
        """
        for feature in request.mm_features:
            pos = feature.mm_position
            if pos.offset + pos.length <= num_computed_tokens:
                continue
            mm_hash = feature.mm_hash or feature.identifier
            if mm_hash in self._pending_save or mm_hash in self._pending_new_encodings:
                continue
            with self._lock:
                if mm_hash in self._local_encodings:
                    continue
            size_bytes = (
                pos.length
                * self._memory_context.hidden_dim
                * self._memory_context.element_size
            )
            n_blocks = max(1, ceil(size_bytes / self._memory_context.block_size_bytes))
            try:
                indices = self._fifo_alloc(n_blocks)
                self._memory_context.region.free(indices)
            except AllocationError:
                logger.debug(
                    "EC: producer cannot reserve %d blocks for mm_hash=%s; "
                    "deferring request",
                    n_blocks,
                    mm_hash,
                )
                return False
        return True

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        feature = request.mm_features[index]
        mm_hash = feature.mm_hash or feature.identifier
        if mm_hash in self._pending_save or mm_hash in self._pending_new_encodings:
            return
        with self._lock:
            if mm_hash in self._local_encodings:
                return
        size_bytes = (
            feature.mm_position.length
            * self._memory_context.hidden_dim
            * self._memory_context.element_size
        )
        self._pending_new_encodings[mm_hash] = size_bytes
        logger.debug("EC: save scheduled mm_hash=%s size_bytes=%d", mm_hash, size_bytes)

    def request_finished(self, request: "Request") -> dict[str, Any] | None:
        """Announce, per mm_hash, the side-channel address and encoding size a
        consumer needs to open an ``XferReq``.

        The consumer connects to ``(peer_host, peer_port)`` and learns the
        source block indices and the producer's NIXL agent metadata from the
        live ``XferAck`` reply, so this announcement carries only the routing
        address and the byte size used to size the consumer's allocation.
        """
        params: dict[str, dict[str, Any]] = {}
        with self._lock:
            local_snapshot = set(self._local_encodings)
        peer_host, peer_port = self._addr
        for feature in request.mm_features:
            mm_hash = feature.mm_hash or feature.identifier
            if mm_hash not in local_snapshot and mm_hash not in self._pending_save:
                continue
            size_bytes = (
                feature.mm_position.length
                * self._memory_context.hidden_dim
                * self._memory_context.element_size
            )
            params[mm_hash] = {
                "peer_host": peer_host,
                "peer_port": peer_port,
                "size_bytes": size_bytes,
            }
        logger.debug(
            "EC: request_finished req_id=%s params=%s",
            request.request_id,
            params,
        )
        return params or None

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
            f"ECSharedRegion exhausted: cannot satisfy {n_blocks} blocks "
            f"even after evicting all unpinned encodings — operator "
            f"under-sized num_ec_blocks for the active workload"
        )

    def build_saves(self) -> dict[str, list[int]]:
        """Advance encode pipeline; return {mm_hash: block_indices} for worker."""
        to_promote = self._pending_save
        self._pending_save = set()
        with self._lock:
            for mm_hash in to_promote:
                self._local_encodings[mm_hash] = None

        pending_new = list(self._pending_new_encodings.items())
        self._pending_new_encodings = {}
        saves: dict[str, list[int]] = {}
        for mm_hash, size_bytes in pending_new:
            if mm_hash in self._pending_save:
                continue
            with self._lock:
                if mm_hash in self._local_encodings:
                    continue
            n_blocks = max(1, ceil(size_bytes / self._memory_context.block_size_bytes))
            indices = self._fifo_alloc(n_blocks)
            self._blocks[mm_hash] = indices
            self._pending_save.add(mm_hash)
            saves[mm_hash] = indices
            logger.debug(
                "EC: save allocated mm_hash=%s n_blocks=%d",
                mm_hash,
                n_blocks,
            )
        return saves

    def shutdown(self) -> None:
        # The router thread is already stopped by the time shutdown runs, so
        # this is single-threaded. Release any still-held source pins so the
        # region's free-pool accounting is balanced before teardown.
        for mm_hash, pin in list(self._pinned_encodings.items()):
            blocks = self._blocks.get(mm_hash)
            if blocks is not None:
                for _ in pin.deadlines:
                    self._memory_context.region.unpin(blocks)
        self._pinned_encodings.clear()
