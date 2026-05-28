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
        return identifier in self._loaded or identifier in self._ready

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
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
            if self.has_cache_item(mm_hash):
                if mm_hash in self._loaded:
                    self._pending_reload.add(mm_hash)
                continue
            if mm_hash in self._remote_encodings:
                if self._remote_encodings[mm_hash] is None:
                    del self._remote_encodings[mm_hash]
                    continue
                pending = True
                continue
            info = params.get(mm_hash)
            if info is None:
                continue
            expected_size = pos.length * self._hidden_dim * self._element_size
            announced_size = int(info.get("size_bytes", -1))
            if announced_size != expected_size:
                logger.warning(
                    "ec: announced size_bytes=%d for mm_hash=%s does not match "
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
                    "ec: failed to start xfer for mm_hash=%s; "
                    "falling back to local encode",
                    mm_hash,
                )
                continue
            pending = True
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
        self._remote_encodings[mm_hash] = (indices, peer_addr)

    def drain_acks(self) -> None:
        for addr in self._transport.poll_dead_peers():
            self.on_peer_down(addr)
        for ack in self._transport.drain_acks():
            if ack.mm_hash not in self._remote_encodings:
                continue
            entry = self._remote_encodings[ack.mm_hash]
            if entry is None:
                continue
            indices, _ = entry
            if ack.ok:
                self._ready.add(ack.mm_hash)
            else:
                self._region.free(indices)
                self._remote_encodings[ack.mm_hash] = None

    def on_peer_down(self, addr: PeerAddr) -> None:
        evicted_agent = self._transport.evict_peer(addr)
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
            "ec: peer down addr=%s agent=%s tombstoned=%d",
            addr,
            evicted_agent,
            n,
        )

    def build_loads(self) -> dict[str, list[int]]:
        """Drain acks, promote arrived mm_hashes, re-emit cached reloads."""
        self.drain_acks()
        loads: dict[str, list[int]] = {}
        for mm_hash in list(self._ready):
            entry = self._remote_encodings.pop(mm_hash, None)
            if entry is None:
                continue
            indices, _ = entry
            loads[mm_hash] = indices
            self._loaded[mm_hash] = indices
        self._ready.clear()
        for mm_hash in self._pending_reload:
            if mm_hash not in loads:
                blocks = self._loaded.get(mm_hash)
                if blocks is not None:
                    loads[mm_hash] = blocks
        self._pending_reload = set()
        return loads

    def shutdown(self) -> None:
        for addr in list(self._transport._peer_pool):
            self.on_peer_down(addr)
