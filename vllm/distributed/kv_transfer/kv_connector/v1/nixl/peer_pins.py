# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-side lookup-and-pin for peer prefix reads over NIXL.

A peer asks "do you hold this prefix?" by block hash. A hit is pinned in the
KV cache manager (`KVCacheManager.lookup_and_pin`) and handed out as a pseudo
request: it gets an id, a lease and an entry in `reqs_to_send`, exactly like
a finished prefill, so the worker's existing read-done and lease-expiry
machinery reports when the blocks can be released.

Everything here runs on the scheduler thread.
"""

import math
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    LookupPinReply,
    LookupPinRequest,
    PeerMissReason,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    kv_lease_extension,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import PrefixPin, PrefixPinMiss

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)

PEER_PIN_ID_PREFIX = "peer-pin-"
_MAX_RELEASED_IDS = 4096

# Hash algorithms whose seed is random per process unless PYTHONHASHSEED is set
# (see ``init_none_hash``): block hashes do not match across processes.
_PROCESS_SEEDED_HASH_ALGOS = frozenset({"xxhash", "xxhash_cbor"})

_MISS_REASONS = {
    PrefixPinMiss.NONE: PeerMissReason.NONE,
    PrefixPinMiss.NOT_CACHED: PeerMissReason.NOT_CACHED,
    PrefixPinMiss.EVICTED: PeerMissReason.EVICTED,
    PrefixPinMiss.REFUSED: PeerMissReason.PIN_REFUSED,
    PrefixPinMiss.UNSUPPORTED: PeerMissReason.IDENTITY_MISMATCH,
}


def hashes_not_reproducible(hash_algo: str) -> bool:
    """Whether block hashes of `hash_algo` differ between processes here."""
    return (
        hash_algo in _PROCESS_SEEDED_HASH_ALGOS
        and os.environ.get("PYTHONHASHSEED") is None
    )


def miss_reply(reason: PeerMissReason) -> LookupPinReply:
    return LookupPinReply(
        hit_tokens=0,
        remote_block_ids=[],
        remote_request_id="",
        miss_reason=reason,
    )


@dataclass
class _PendingPin:
    token: Any
    pin: PrefixPin
    pin_id: str
    # The pin may be exposed once this many steps have been processed.
    fence: int
    lease: float
    created_at: float


@dataclass
class _ShippedPin:
    pin: PrefixPin
    pin_id: str


class PeerPinServer:
    """Answers peer lookups by pinning cached prefixes until the peer is done.

    Lifecycle of a hit: `serve` pins it and holds it as pending. A block is
    registered in the prefix cache when its step is scheduled, before the GPU
    wrote its KV, and a remote RDMA read is not ordered against that write. So
    the pin is only exposed (`take_new_sends` ships it and readies its reply)
    once every step scheduled before the lookup has been processed. It then
    lives in `reqs_to_send` like any finished prefill, until the worker
    reports it in `finished_sending` (read done, or lease expiry).

    Step accounting: the owner calls `on_step_scheduled` once per
    `build_connector_meta` and `on_step_processed` once per
    `update_from_output`. These run once per scheduled step, in order.
    """

    def __init__(
        self,
        kv_cache_manager: "KVCacheManager",
        block_size: int,
        hash_algo: str,
        lease_duration: float,
        max_pinned_fraction: float,
        min_free_fraction: float,
        clock: Callable[[], float] = time.perf_counter,
        max_pending_seconds: float | None = None,
    ):
        self._manager = kv_cache_manager
        self._block_size = block_size
        self._hash_algo = hash_algo
        self._lease_duration = lease_duration
        self._clock = clock
        # A pin that has not shipped by then is released and refused, so a
        # stalled step cannot hold blocks for a peer that gave up waiting.
        self._max_pending_seconds = max_pending_seconds
        pool = kv_cache_manager.block_pool
        self._num_gpu_blocks: int = pool.num_gpu_blocks
        self._max_pinned_blocks = int(max_pinned_fraction * self._num_gpu_blocks)
        self._min_free_blocks = math.ceil(min_free_fraction * self._num_gpu_blocks)
        # A miss is EVICTED rather than NOT_CACHED if the hash was dropped
        # recently; remembering as many hashes as the pool has blocks is enough.
        pool.enable_evicted_hash_tracking(self._num_gpu_blocks)

        self._compat_hash: str | None = None
        self._owner_thread: int | None = None
        self._warned_unseeded = False
        self._steps_scheduled = 0
        self._steps_processed = 0
        self._pending: dict[str, _PendingPin] = {}
        self._shipped: dict[str, _ShippedPin] = {}
        self._ready: deque[tuple[Any, LookupPinReply]] = deque()
        # Recently released pin ids, so a repeated report of one is dropped
        # here instead of reaching the scheduler's "known request" assert.
        self._released: OrderedDict[str, None] = OrderedDict()
        self.num_revoked_after_reply = 0

    # ------------------------------------------------------------------
    def _check_thread(self) -> None:
        ident = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = ident
        assert self._owner_thread == ident, (
            "PeerPinServer must only be used from the scheduler thread"
        )

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    @property
    def num_shipped(self) -> int:
        return len(self._shipped)

    def set_compatibility_hash(self, compat_hash: str) -> None:
        self._check_thread()
        self._compat_hash = compat_hash

    # ------------------------------------------------------------------
    def _identity_mismatch(self, request: LookupPinRequest) -> str | None:
        identity = request.identity
        if self._compat_hash is None:
            return "local compatibility hash is not known yet"
        if identity.compatibility_hash != self._compat_hash:
            return "compatibility hash differs"
        if identity.block_size != self._block_size:
            return f"block size {identity.block_size} != {self._block_size}"
        if identity.hash_algo != self._hash_algo:
            return f"hash algo {identity.hash_algo} != {self._hash_algo}"
        if hashes_not_reproducible(self._hash_algo):
            if not self._warned_unseeded:
                self._warned_unseeded = True
                logger.warning(
                    "Refusing peer prefix lookups: hash algo %s is seeded per "
                    "process because PYTHONHASHSEED is unset, so block hashes "
                    "do not match across engines. Set PYTHONHASHSEED to a "
                    "shared value or use sha256.",
                    self._hash_algo,
                )
            return "local block hashes are not reproducible across processes"
        return None

    def serve(self, token: Any, request: LookupPinRequest) -> None:
        """Handle one lookup. The reply comes out of `take_ready_replies`."""
        self._check_thread()
        mismatch = self._identity_mismatch(request)
        if mismatch is not None:
            logger.debug("Peer lookup refused: %s", mismatch)
            self._ready.append((token, miss_reply(PeerMissReason.IDENTITY_MISMATCH)))
            return

        budget = self._max_pinned_blocks - self._manager.num_pinned_blocks
        if budget <= 0:
            self._ready.append((token, miss_reply(PeerMissReason.PIN_REFUSED)))
            return

        pin = self._manager.lookup_and_pin(
            cast("list[BlockHash]", request.block_hashes),
            max_blocks=budget,
            min_free_blocks=self._min_free_blocks,
        )
        reason = _MISS_REASONS[pin.miss]
        if pin.num_blocks == 0:
            self._manager.release_pin(pin)
            logger.debug("Peer lookup_pin missed: %s", reason.value)
            self._ready.append((token, miss_reply(reason)))
            return

        pin_id = f"{PEER_PIN_ID_PREFIX}{uuid.uuid4().hex}"
        # Ids travel in "<id>:<n>" notifs and comma-separated heartbeats.
        assert ":" not in pin_id and "," not in pin_id
        requested = request.lease_duration
        lease = (
            self._lease_duration
            if not requested or requested <= 0
            else min(requested, self._lease_duration)
        )
        self._pending[pin_id] = _PendingPin(
            token=token,
            pin=pin,
            pin_id=pin_id,
            fence=self._steps_scheduled + 1,
            lease=lease,
            created_at=self._clock(),
        )
        logger.debug(
            "Peer lookup_pin pinned %d of %d blocks as %s",
            pin.num_blocks,
            len(request.block_hashes),
            pin_id,
        )

    # ------------------------------------------------------------------
    def on_step_scheduled(self) -> None:
        self._check_thread()
        self._steps_scheduled += 1

    def on_step_processed(self) -> None:
        self._check_thread()
        self._steps_processed += 1

    def take_new_sends(self) -> dict[str, float]:
        """Ship the pins whose fence has passed; returns `{pin_id: expiry}`.

        The caller adds them to `reqs_to_send` and `reqs_in_batch` of the
        metadata it is building, so the worker learns each id in the same step
        its reply leaves.
        """
        self._check_thread()
        sends: dict[str, float] = {}
        if self._max_pending_seconds is not None and self._pending:
            now = self._clock()
            for pin_id in [
                pid
                for pid, p in self._pending.items()
                if now - p.created_at > self._max_pending_seconds
            ]:
                timed_out = self._pending.pop(pin_id)
                self._manager.release_pin(timed_out.pin)
                logger.warning(
                    "Peer pin %s was not shipped within %.1fs; releasing it.",
                    pin_id,
                    self._max_pending_seconds,
                )
                self._ready.append(
                    (timed_out.token, miss_reply(PeerMissReason.PIN_REFUSED))
                )
        for pin_id in [
            pid for pid, p in self._pending.items() if p.fence <= self._steps_processed
        ]:
            pending = self._pending.pop(pin_id)
            expiry = self._clock() + pending.lease
            self._shipped[pin_id] = _ShippedPin(pending.pin, pin_id)
            sends[pin_id] = expiry
            self._ready.append(
                (
                    pending.token,
                    LookupPinReply(
                        hit_tokens=pending.pin.num_tokens,
                        remote_block_ids=[
                            list(group) for group in pending.pin.get_block_ids()
                        ],
                        remote_request_id=pin_id,
                        miss_reason=_MISS_REASONS[pending.pin.miss],
                        remote_blocks_expiry_time=expiry,
                        remote_lease_extension=kv_lease_extension(int(pending.lease)),
                    ),
                )
            )
        return sends

    def take_ready_replies(self) -> list[tuple[Any, LookupPinReply]]:
        self._check_thread()
        replies = list(self._ready)
        self._ready.clear()
        return replies

    # ------------------------------------------------------------------
    def on_finished_sending(self, ids: set[str]) -> set[str]:
        """Release the pins named in `ids`; returns `ids` without them."""
        self._check_thread()
        # Match on the pins this server shipped, not on the id prefix: request
        # ids can come from clients, and one that happens to start with the
        # prefix must still reach the scheduler, or its blocks would leak.
        rest: set[str] = set()
        for req_id in ids:
            shipped = self._shipped.pop(req_id, None)
            if shipped is not None:
                self._manager.release_pin(shipped.pin)
                self._released[req_id] = None
                if len(self._released) > _MAX_RELEASED_IDS:
                    self._released.popitem(last=False)
            elif req_id in self._released:
                logger.debug("Ignoring a repeated finished_sending for %s", req_id)
            else:
                rest.add(req_id)
        return rest

    def on_invalid_blocks(self, block_ids: set[int]) -> None:
        """Revoke pins holding blocks whose KV load failed."""
        self._check_thread()
        if not block_ids or not (self._pending or self._shipped):
            return
        for pin in self._manager.pins_holding(block_ids):
            pending_id = next(
                (pid for pid, p in self._pending.items() if p.pin is pin), None
            )
            if pending_id is not None:
                pending = self._pending.pop(pending_id)
                self._manager.release_pin(pin)
                self._ready.append(
                    (pending.token, miss_reply(PeerMissReason.NOT_CACHED))
                )
                continue
            if any(s.pin is pin for s in self._shipped.values()):
                # The peer may be reading already; it cannot be recalled from
                # here. Keep the pin until the worker is done with it.
                self.num_revoked_after_reply += 1
                logger.warning(
                    "KV blocks of an already-served peer pin failed to load; "
                    "the peer may read invalid KV."
                )

    def has_pending_work(self) -> bool:
        """Whether the engine must keep stepping for this server.

        Shipped pins count: the worker reports a pin's read-done notification
        or lease expiry only from inside a step, so an idle engine would hold
        the pin forever. Stock sends stay alive the same way, as finished
        requests still in `Scheduler.requests`.
        """
        return bool(self._pending or self._ready or self._shipped)
