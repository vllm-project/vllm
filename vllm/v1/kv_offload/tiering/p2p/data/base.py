# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base class for the P2P data-plane transport.

The data plane handles RDMA (or similar) block transfers between peers.
It is independent of the control plane — the control plane establishes
who can talk to whom, then the data plane moves blocks at wire speed.

Architecture
------------

    DataTransport (one per node)
        ├── owns the local KV block memory region
        ├── registers remote peers (add_remote_peer / remove_remote_peer)
        ├── submits block writes (write_blocks → transfer_id)
        ├── polls for completion (poll → done/failed IDs)
        └── cancels inflight transfers (cancel)

Memory model
------------

The local node exposes a contiguous block region:

    base_addr ──► ┌─────────────┐  block 0
                  ├─────────────┤  block 1
                  ├─────────────┤  ...
                  └─────────────┘  block (num_blocks - 1)

    Each block is block_len bytes.

Remote peers expose the same layout. write_blocks() copies local
blocks to a remote peer's block region by index:

    write_blocks("peer:1", local_idxs=[0, 3], remote_idxs=[5, 7])
    → writes local block 0 → remote block 5
              local block 3 → remote block 7

Transfer lifecycle
------------------

1. Register remote peer: add_remote_peer(peer_id, metadata, ...)
   - Provides the remote memory layout so transfers can target it
2. Submit: write_blocks(peer_id, local_idxs, remote_idxs) → int
   - Returns a transfer_id (opaque int) for tracking
   - Returns None if peer not registered or submission fails
3. Poll: poll() → PollResult(done=[...], failed=[...])
   - Returns transfer_ids that completed or failed since last poll
   - Completed transfers are automatically cleaned up
4. Cancel: cancel(transfer_ids, mode="immediate" | "wait")
   - Best-effort cancellation of inflight transfers
   - mode="wait" returns ids still in PROC/PEND so the caller can poll
     them to completion

Implementor contracts
---------------------

- write_blocks() must not block. The transfer runs asynchronously.
- poll() must be called periodically. It drives completion checking.
- transfer_ids are unique across the lifetime of the transport.
- add_remote_peer() must be called before write_blocks() to that peer.
- get_agent_metadata() returns opaque bytes that the remote peer
  needs to call add_remote_peer() (e.g., RDMA connection info).
- config_fingerprint is a content hash of the model configuration.
  Peers with different fingerprints are incompatible and must not
  exchange blocks (validated during the control-plane handshake).
- close() releases all resources (memory registrations, handles).
  After close(), no other methods may be called.

Threading model: no background threads. All I/O driven by poll().
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Literal, NamedTuple

CancelMode = Literal["immediate", "wait"]


class PollResult(NamedTuple):
    """Result of polling inflight transfers.

    Attributes:
        done: Transfer IDs that completed successfully.
        failed: Transfer IDs that failed (error, timeout, etc.).
    """

    done: Sequence[int]
    failed: Sequence[int]


class DataTransport(ABC):
    """Abstract data-plane transport for RDMA-style block transfers.

    Owns the local KV block memory region and manages transfers to/from
    registered remote peers.

    Construction:
        view: A 2D memoryview (num_blocks × block_len bytes) over the
              local KV cache block storage.
        config_fields: Dict of model config values used to compute the
                       compatibility fingerprint. None → empty fingerprint
                       (compatible with any peer).
    """

    def __init__(self, view: memoryview, config_fields: dict | None = None) -> None:
        assert view.shape is not None
        self._view = view
        self._base_addr = ctypes.addressof(ctypes.c_char.from_buffer(view))
        self._num_blocks = view.shape[0]
        self._block_len = view.shape[1]
        self._config_fingerprint = self._compute_fingerprint(config_fields)

    @property
    def base_addr(self) -> int:
        """Base address of the local block memory region."""
        return self._base_addr

    @property
    def num_blocks(self) -> int:
        """Number of blocks in the local region."""
        return self._num_blocks

    @property
    def block_len(self) -> int:
        """Size of each block in bytes."""
        return self._block_len

    @property
    def config_fingerprint(self) -> str:
        """Content-hash of the model configuration (hex string).

        Peers must have matching fingerprints to exchange blocks.
        Empty string means no fingerprint (always compatible).
        """
        return self._config_fingerprint

    @staticmethod
    def _compute_fingerprint(config_fields: dict | None) -> str:
        if not config_fields:
            return ""
        canonical = json.dumps(config_fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @abstractmethod
    def get_agent_metadata(self) -> bytes:
        """Return opaque metadata needed by remote peers to connect.

        The returned bytes are sent during the control-plane handshake
        and passed to the remote peer's add_remote_peer().
        """
        ...

    @abstractmethod
    def add_remote_peer(
        self,
        peer_id: str,
        agent_metadata: bytes,
        base_addr: int,
        num_blocks: int,
        block_len: int,
    ) -> None:
        """Register a remote peer for block transfers.

        Must be called before write_blocks() to this peer.

        Args:
            peer_id: Unique identifier for the remote peer.
            agent_metadata: Opaque bytes from the peer's get_agent_metadata().
            base_addr: Base address of the peer's block memory region.
            num_blocks: Number of blocks in the peer's region.
            block_len: Size of each block (must match local block_len).
        """
        ...

    @abstractmethod
    def remove_remote_peer(self, peer_id: str) -> None:
        """Unregister a remote peer and release associated resources.

        Inflight transfers to this peer should be cancelled first.
        """
        ...

    @abstractmethod
    def write_blocks(
        self,
        peer_id: str,
        local_idxs: list[int],
        remote_idxs: list[int],
    ) -> int | None:
        """Submit a WRITE transfer: local blocks → remote peer's blocks.

        Args:
            peer_id: Target peer (must be registered via add_remote_peer).
            local_idxs: Indexes of local blocks to read from.
            remote_idxs: Indexes of remote blocks to write to.
                         Must be same length as local_idxs.

        Returns:
            A unique transfer_id (int) to track this transfer, or
            None if the peer is not registered or submission failed.
        """
        ...

    @abstractmethod
    def poll(self, peer_id: str | None = None) -> PollResult:
        """Poll inflight transfers for completion.

        Args:
            peer_id: If given, only poll (and drain) transfers submitted for
                this peer_id — the value passed to ``write_blocks``. This is
                required when a single transport is shared across multiple
                peer sessions: ``poll()`` pops completed handles, so an
                unscoped poll by one session would consume and discard the
                completions of its siblings, starving them. ``None`` polls
                every peer's transfers (used only for the shutdown drain).

        Returns:
            PollResult with lists of completed and failed transfer_ids.
            Completed/failed transfers are removed from the inflight set.

        Must be called periodically to drive progress checking.
        """
        ...

    @abstractmethod
    def cancel(
        self,
        transfer_ids: Iterable[int],
        mode: CancelMode = "immediate",
    ) -> list[int]:
        """Cancel inflight transfers by their IDs.

        Best-effort: transfers that already completed are ignored.

        Args:
            transfer_ids: IDs to cancel. Unknown IDs are ignored.
            mode:
                "immediate" (default): pop and release each handle and
                    return []. Matches the legacy fire-and-forget
                    behavior — the caller does not wait for the
                    underlying transfer to drain.
                "wait": attempt to release each handle. If the release
                    cannot complete because the transfer is still
                    PROC/PEND, the entry stays in the inflight set and
                    its id is included in the returned list. The
                    caller is expected to keep calling poll() until
                    every returned id surfaces in done/failed.

        Returns:
            For mode="wait", the subset of *transfer_ids* still
            tracked as inflight after the cancel attempt. For
            mode="immediate", always [].
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources (registrations, handles, memory).

        Cancels any remaining inflight transfers. Idempotent.
        After close(), no other methods may be called.
        """
        ...
