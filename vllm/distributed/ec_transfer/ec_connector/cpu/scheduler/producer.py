# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer role for ECCPUConnector."""

import contextlib
import threading
import uuid
from math import ceil
from typing import TYPE_CHECKING, Any

from pybase64 import b64encode

from vllm.distributed.ec_transfer.ec_connector.cpu.metadata import (
    EC_CONNECTOR_VERSION,
    XferReq,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.common import (
    evict_and_alloc,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import ProducerPeer
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECCPUProducer:
    """Producer state machine.

    Depends on a TransferEngine (duck-typed). Never imports NixlWrapper or
    zmq directly. All methods that access NIXL state run on the router thread;
    build_saves runs on the main thread.
    """

    def __init__(
        self,
        region: ECSharedRegion,
        engine: Any,
        agent_metadata: bytes,
        local_xfer_handle: int,
        compat_hash: str,
        hidden_dim: int,
        element_size: int,
        block_size_bytes: int,
        peer_host: str,
        peer_port: int,
    ) -> None:
        self._region = region
        self._engine = engine
        self._agent_metadata = agent_metadata
        self._local_xfer_handle = local_xfer_handle
        self._compat_hash = compat_hash
        self._hidden_dim = hidden_dim
        self._element_size = element_size
        self._block_size_bytes = block_size_bytes
        self._peer_host = peer_host
        self._peer_port = peer_port

        self._pending_new_encodings: dict[str, int] = {}
        self._pending_save: dict[str, list[int]] = {}

        self._lock = threading.Lock()
        self._local_encodings: dict[str, list[int]] = {}
        self._in_flight: dict[str, tuple[bytes, str, Any]] = {}
        self._pending_nacks: list[tuple[bytes, str]] = []

        # Router-thread-only state. No lock needed.
        self._remote_peers: dict[str, ProducerPeer] = {}

    # ── router-thread methods ─────────────────────────────────────────────────

    def has_in_flight(self) -> bool:
        with self._lock:
            return bool(self._in_flight)

    def handle_xfer_req(self, identity: bytes, req: XferReq) -> None:
        if req.connector_version != EC_CONNECTOR_VERSION:
            logger.warning(
                "ec: version mismatch req=%d local=%d, NACKing",
                req.connector_version,
                EC_CONNECTOR_VERSION,
            )
            with self._lock:
                self._pending_nacks.append((identity, req.mm_hash))
            return

        if req.compatibility_hash != self._compat_hash:
            with self._lock:
                self._pending_nacks.append((identity, req.mm_hash))
            return

        with self._lock:
            block_indices = self._local_encodings.get(req.mm_hash)
            if block_indices is None:
                self._pending_nacks.append((identity, req.mm_hash))
                return
            self._region.pin(block_indices)

        try:
            peer = self._ensure_remote_peer(req)
            handle = self._engine.post_write(
                self._local_xfer_handle,
                block_indices,
                peer,
                req.dst_block_indices,
            )
        except Exception:
            logger.exception(
                "ec: failed to post NIXL WRITE for mm_hash=%s", req.mm_hash
            )
            self._region.unpin(block_indices)
            with self._lock:
                self._pending_nacks.append((identity, req.mm_hash))
            return

        xfer_id = str(uuid.uuid4())
        with self._lock:
            self._in_flight[xfer_id] = (identity, req.mm_hash, handle)

    def _ensure_remote_peer(self, req: XferReq) -> ProducerPeer:
        existing = self._remote_peers.get(req.consumer_agent_name)
        if (
            existing is not None
            and existing.nixl_metadata_bytes == req.consumer_nixl_metadata
        ):
            return existing
        if existing is not None:
            self._engine.remove_remote_agent(existing.nixl_agent_name)

        peer = self._engine.add_remote_peer(
            req.consumer_nixl_metadata,
            req.consumer_mem_descriptor,
        )
        self._remote_peers[req.consumer_agent_name] = peer
        return peer

    def sweep_completions(self) -> list[tuple[bytes, str, bool]]:
        """Poll in-flight xfer state; return ack routes for the transport to send.

        Runs on the router thread.
        NIXL polling is lock-free; outcomes are applied under lock.
        Returns list of (identity, mm_hash, ok) for each completed or failed xfer,
        plus any queued NACKs.
        """
        # (1) Snapshot.
        with self._lock:
            in_flight_snapshot = [
                (xfer_id, handle) for xfer_id, (_, _, handle) in self._in_flight.items()
            ]

        # (2) Poll lock-free; release handles inline on terminal states.
        outcomes: dict[str, bool] = {}
        for xfer_id, handle in in_flight_snapshot:
            try:
                state = self._engine.check_xfer_state(handle)
            except Exception:
                logger.exception(
                    "ec: check_xfer_state raised for xfer_id=%s; treating as failure",
                    xfer_id,
                )
                outcomes[xfer_id] = False
                self._engine.release_xfer_handle(handle)
                continue
            if state == "DONE":
                outcomes[xfer_id] = True
                self._engine.release_xfer_handle(handle)
            elif state == "PROC":
                continue
            else:
                logger.warning(
                    "ec: NIXL xfer in unexpected state %r for xfer_id=%s; "
                    "treating as failure",
                    state,
                    xfer_id,
                )
                outcomes[xfer_id] = False
                self._engine.release_xfer_handle(handle)

        # (3) Apply outcomes, unpin memory, drain queued NACKs.
        ok_routes: list[tuple[bytes, str]] = []
        fail_routes: list[tuple[bytes, str]] = []
        with self._lock:
            for xfer_id, ok in outcomes.items():
                entry = self._in_flight.pop(xfer_id, None)
                if entry is None:
                    continue
                identity, mm_hash, _handle = entry
                blocks = self._local_encodings.get(mm_hash)
                if blocks is not None:
                    self._region.unpin(blocks)
                (ok_routes if ok else fail_routes).append((identity, mm_hash))
            queued_nacks = self._pending_nacks
            self._pending_nacks = []

        # (4) Assemble routes for the transport to send.
        routes: list[tuple[bytes, str, bool]] = []
        for identity, mm_hash in queued_nacks:
            routes.append((identity, mm_hash, False))
        for identity, mm_hash in ok_routes:
            routes.append((identity, mm_hash, True))
        for identity, mm_hash in fail_routes:
            routes.append((identity, mm_hash, False))
        return routes

    # ── main-thread methods ───────────────────────────────────────────────────

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        feature = request.mm_features[index]
        mm_hash = feature.mm_hash or feature.identifier
        if mm_hash in self._pending_save or mm_hash in self._pending_new_encodings:
            return
        with self._lock:
            if mm_hash in self._local_encodings:
                return
        size_bytes = feature.mm_position.length * self._hidden_dim * self._element_size
        self._pending_new_encodings[mm_hash] = size_bytes

    def request_finished(self, request: "Request") -> dict[str, Any] | None:
        params: dict[str, dict[str, Any]] = {}
        nixl_meta_b64 = b64encode(self._agent_metadata).decode("ascii")
        with self._lock:
            local_snapshot = set(self._local_encodings)
        for feature in request.mm_features:
            mm_hash = feature.mm_hash or feature.identifier
            if mm_hash not in local_snapshot and mm_hash not in self._pending_save:
                continue
            size_bytes = (
                feature.mm_position.length * self._hidden_dim * self._element_size
            )
            params[mm_hash] = {
                "peer_host": self._peer_host,
                "peer_port": self._peer_port,
                "size_bytes": size_bytes,
                "nixl_agent_metadata_b64": nixl_meta_b64,
            }
        return params or None

    def _fifo_alloc(self, n_blocks: int) -> list[int]:
        try:
            return self._region.alloc(n_blocks)
        except AllocationError:
            pass
        with self._lock:
            result = evict_and_alloc(
                n_blocks, self._local_encodings, self._region, skip_pinned=True
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
        self._pending_save = {}
        with self._lock:
            self._local_encodings.update(to_promote)

        pending_new = list(self._pending_new_encodings.items())
        self._pending_new_encodings = {}
        saves: dict[str, list[int]] = {}
        for mm_hash, size_bytes in pending_new:
            if mm_hash in self._pending_save:
                continue
            with self._lock:
                if mm_hash in self._local_encodings:
                    continue
            n_blocks = max(1, ceil(size_bytes / self._block_size_bytes))
            indices = self._fifo_alloc(n_blocks)
            self._pending_save[mm_hash] = indices
            saves[mm_hash] = indices
        return saves

    def shutdown(self) -> None:
        try:
            for _xfer_id, (_, _, handle) in list(self._in_flight.items()):
                with contextlib.suppress(Exception):
                    self._engine.release_xfer_handle(handle)
            self._in_flight.clear()
        except Exception:
            logger.debug("ec: release in_flight failed", exc_info=True)

        for peer in list(self._remote_peers.values()):
            self._engine.remove_remote_agent(peer.nixl_agent_name)
        self._remote_peers.clear()
