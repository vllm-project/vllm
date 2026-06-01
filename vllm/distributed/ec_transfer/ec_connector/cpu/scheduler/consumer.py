# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer role for ECCPUConnector."""

from math import ceil
from typing import TYPE_CHECKING, Any

from pybase64 import b64decode

from vllm.distributed.ec_transfer.ec_connector.cpu.metadata import XferReq
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.common import (
    evict_and_alloc,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import PeerAddr
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECCPUConsumer:
    """Consumer state machine.

    Depends on a ConsumerTransport (duck-typed) and ECSharedRegion.
    Never imports zmq or NixlWrapper directly.
    """

    def __init__(
        self,
        region: ECSharedRegion,
        transport: Any,
        agent_metadata: bytes,
        mem_descriptor_bytes: bytes,
        compat_hash: str,
        engine_id: str,
        hidden_dim: int,
        element_size: int,
        block_size_bytes: int,
    ) -> None:
        self._region = region
        self._transport = transport
        self._agent_metadata = agent_metadata
        self._mem_descriptor_bytes = mem_descriptor_bytes
        self._compat_hash = compat_hash
        self._engine_id = engine_id
        self._hidden_dim = hidden_dim
        self._element_size = element_size
        self._block_size_bytes = block_size_bytes

        self._remote_encodings: dict[str, tuple[list[int], PeerAddr] | None] = {}
        self._ready: set[str] = set()
        self._loaded: dict[str, list[int]] = {}
        self._pending_reload: set[str] = set()

    def has_cache_item(self, identifier: str) -> bool:
        """True iff the bytes for `identifier` are in our local mmap, either
        because `_drain_acks` just promoted them (worker will copy this
        step) or because the worker already copied them last step and
        we have not yet freed the blocks.
        """
        return identifier in self._loaded or identifier in self._ready

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """Decide whether `request` can run this step, allocating and
        sending any XferReqs needed.

        Returns False if the request is waiting on at least one in-flight
        transfer and should be deferred; True if every feature is
        locally-cached or absent-from-announcement (and
        therefore falls through to vLLM's local encode).
        """
        params: dict[str, dict[str, Any]] = (
            getattr(request, "ec_transfer_params", None) or {}
        )
        if not params:
            return True
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
                logger.debug("EC: mm_hash=%s xfer already in-flight", mm_hash)
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
                self._alloc_and_start_xfer(mm_hash, info, expected_size)
            except Exception:
                logger.exception(
                    "EC: failed to start xfer for mm_hash=%s; "
                    "falling back to local encode",
                    mm_hash,
                )
                continue
            logger.debug(
                "EC: mm_hash=%s xfer started size_bytes=%d", mm_hash, expected_size
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

    def _alloc_and_start_xfer(
        self, mm_hash: str, info: dict[str, Any], size_bytes: int
    ) -> None:
        n_blocks = max(1, ceil(size_bytes / self._block_size_bytes))
        indices = self._fifo_alloc(n_blocks)
        try:
            peer_addr: PeerAddr = (info["peer_host"], int(info["peer_port"]))
            metadata = b64decode(info["nixl_agent_metadata_b64"])
            existing = self._transport._peer_pool.get(peer_addr)
            if existing is not None and existing.nixl_metadata_bytes != metadata:
                # Same address but the metadata is wrong - It's a new producer.
                # We need to stop the connection and start again.
                self.on_peer_down(peer_addr)
            peer = self._transport.get_or_create_peer(peer_addr, metadata)
            req = XferReq(
                mm_hash=mm_hash,
                dst_block_indices=indices,
                consumer_agent_name=self._engine_id,
                consumer_nixl_metadata=self._agent_metadata,
                consumer_mem_descriptor=self._mem_descriptor_bytes,
                compatibility_hash=self._compat_hash,
            )
            self._transport.send_xfer_req(peer, req)
        except Exception:
            self._region.free(indices)
            raise
        logger.debug(
            "EC: load requested mm_hash=%s n_blocks=%d peer=%s:%d",
            mm_hash,
            n_blocks,
            info["peer_host"],
            int(info["peer_port"]),
        )
        self._remote_encodings[mm_hash] = (indices, peer_addr)

    def drain_acks(self) -> None:
        for addr in self._transport.poll_dead_peers():
            self.on_peer_down(addr)
        for ack in self._transport.drain_acks():
            if ack.mm_hash not in self._remote_encodings:
                continue
            entry = self._remote_encodings[ack.mm_hash]
            if entry is None:
                # Already tombstoned by a prior NACK or peer-down; ignore.
                continue
            indices, _ = entry
            if ack.ok:
                self._ready.add(ack.mm_hash)
                logger.debug("EC: load arrived mm_hash=%s", ack.mm_hash)
            else:
                # NACK: free the consumer-side blocks and leave a
                # tombstone for `ensure_cache_available` to consume on
                # its next call. Consuming the tombstone there flips
                # the request from deferred to runnable so vLLM's local
                # encode path covers this feature.
                self._region.free(indices)
                self._remote_encodings[ack.mm_hash] = None

    def on_peer_down(self, addr: PeerAddr) -> None:
        evicted_agent_name = self._transport.evict_peer(addr)
        n = 0
        for mm_hash, entry in list(self._remote_encodings.items()):
            if entry is None or entry[1] != addr:
                continue
            indices, _ = entry
            if mm_hash not in self._ready:
                self._region.free(indices)
                self._remote_encodings[mm_hash] = None
                n += 1
        logger.info(
            "EC: peer down addr=%s agent=%s tombstoned=%d",
            addr,
            evicted_agent_name,
            n,
        )

    def build_loads(self) -> dict[str, list[int]]:
        """Drain acks, promote arrived mm_hashes, re-emit cached reloads."""
        # (a) Drain any fresh ack arrivals.
        self.drain_acks()
        # (b) Hand newly-arrived mm_hashes to the worker; move to loaded
        #     cache. Blocks stay allocated so that subsequent requests for
        #     the same mm_hash are re-served with a local mmap→GPU re-copy
        #     instead of a producer round-trip.
        loads: dict[str, list[int]] = {}
        for mm_hash in list(self._ready):
            entry = self._remote_encodings.pop(mm_hash, None)
            if entry is None:
                continue
            indices, _ = entry
            loads[mm_hash] = indices
            self._loaded[mm_hash] = indices
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
                        "EC: cache hit mm_hash=%s",
                        mm_hash,
                    )
        self._pending_reload = set()
        return loads

    def shutdown(self) -> None:
        for addr in list(self._transport._peer_pool):
            self.on_peer_down(addr)
        self._transport.shutdown()
