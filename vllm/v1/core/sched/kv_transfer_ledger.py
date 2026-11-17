# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side record of KV transfers that pin this engine's blocks.

Every KV transfer a connector performs on behalf of a request is one entry:

* ``EXPORT``: this engine keeps a finished request's blocks for a remote
  reader (the connector asked ``request_finished()`` to delay the free).
* ``IMPORT``: a remote engine holds KV for this request (``do_remote_prefill``)
  and, once the load starts, writes into this engine's blocks.

The ledger answers the questions the scheduler and the engine core need for
pause, sleep and shutdown decisions without inferring them from queue
lengths: does any transfer still reference a block, does any transfer need
the engine to keep stepping (heartbeats, completions), and may a request's
blocks be freed now.

An entry outlives its request: a request that finishes while its entry is
open stays referenced through the entry, so a completion that arrives later
finds it instead of a missing dictionary key.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request


class TransferDirection(enum.Enum):
    EXPORT = enum.auto()
    IMPORT = enum.auto()


class TransferState(enum.Enum):
    # IMPORT only: the remote holds KV for us; no local blocks involved yet.
    PENDING = enum.auto()
    # Bytes may be moving: local blocks are pinned by the transfer.
    OPEN = enum.auto()
    # The worker reported completion; the scheduler has not consumed it yet.
    CLOSING = enum.auto()
    # EXPORT only: dropped (abort, or a pause that clears the cache). The
    # connector fails later reads and reports the export finished once no
    # read in flight uses its blocks; until then the blocks stay pinned.
    DROPPING = enum.auto()


@dataclass
class TransferEntry:
    request: Request
    direction: TransferDirection
    state: TransferState
    # The request finished (e.g. was aborted) while the entry was open.
    request_finished: bool = False
    # EXPORT only: only the KV connector holds the blocks, so an abort may
    # ask it to cancel the export instead of waiting for the reader.
    droppable: bool = False
    # IMPORT only: the scheduler no longer wants this KV (a pause that clears
    # the cache). Once the load settles, the request prefills locally.
    retracted: bool = False

    @property
    def request_id(self) -> str:
        return self.request.request_id


class KVTransferLedger:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, TransferDirection], TransferEntry] = {}
        self._num_exports = 0
        self._num_finished_imports = 0
        # Imports that are PENDING or OPEN: the remote still holds or writes
        # KV for us, so the engine must keep stepping (renewals, completions).
        self._num_live_imports = 0
        # Imports whose load is in flight: local blocks are pinned by it.
        self._num_open_imports = 0

    # ---- open ------------------------------------------------------------

    def open_export(self, request: Request, droppable: bool = False) -> TransferEntry:
        """A finished request's blocks are kept for a remote reader or a
        connector's async save. `droppable`: only a remote reader needs them."""
        key = (request.request_id, TransferDirection.EXPORT)
        entry = self._entries.get(key)
        if entry is None:
            entry = TransferEntry(
                request,
                TransferDirection.EXPORT,
                TransferState.OPEN,
                droppable=droppable,
            )
            self._entries[key] = entry
            self._num_exports += 1
        return entry

    def open_import(self, request: Request) -> TransferEntry:
        """A remote engine holds KV for this request (not yet loading)."""
        key = (request.request_id, TransferDirection.IMPORT)
        entry = self._entries.get(key)
        if entry is None:
            entry = TransferEntry(
                request, TransferDirection.IMPORT, TransferState.PENDING
            )
            self._entries[key] = entry
            self._num_live_imports += 1
        return entry

    def import_started(self, request: Request) -> TransferEntry:
        """The load started: local blocks are now pinned by the transfer."""
        entry = self.open_import(request)
        if entry.state == TransferState.PENDING:
            entry.state = TransferState.OPEN
            self._num_open_imports += 1
        return entry

    # ---- transitions -----------------------------------------------------

    def complete(
        self, request_id: str, direction: TransferDirection
    ) -> TransferEntry | None:
        """The worker reported the transfer done (or failed)."""
        entry = self._entries.get((request_id, direction))
        if entry is None:
            return None
        if (
            direction == TransferDirection.IMPORT
            and entry.state != TransferState.CLOSING
        ):
            # Received: the blocks are the request's own now; nothing is left
            # to drive until the scheduler promotes it.
            self._num_live_imports -= 1
            if entry.state == TransferState.OPEN:
                self._num_open_imports -= 1
        entry.state = TransferState.CLOSING
        return entry

    def close(
        self, request_id: str, direction: TransferDirection
    ) -> TransferEntry | None:
        """The scheduler consumed the completion; the entry no longer pins
        anything. Returns the entry, or None if there was none."""
        entry = self._entries.pop((request_id, direction), None)
        if entry is None:
            return None
        if direction == TransferDirection.EXPORT:
            self._num_exports -= 1
        else:
            if entry.state != TransferState.CLOSING:
                self._num_live_imports -= 1
            if entry.state == TransferState.OPEN:
                self._num_open_imports -= 1
            if entry.request_finished:
                self._num_finished_imports -= 1
        return entry

    def droppable_exports(self) -> list[TransferEntry]:
        """Open exports the engine may stop holding for remote readers. The
        caller marks the ones the connector cancels DROPPING: they still pin
        their blocks until the connector reports them finished, or until the
        caller closes them."""
        return [
            entry
            for (_, direction), entry in self._entries.items()
            if direction == TransferDirection.EXPORT
            and entry.droppable
            and entry.state == TransferState.OPEN
        ]

    def live_imports(self) -> list[TransferEntry]:
        """Imports of requests that have not finished, in any state."""
        return [
            e
            for (_, direction), e in self._entries.items()
            if direction == TransferDirection.IMPORT and not e.request_finished
        ]

    def retract(self, request_id: str) -> None:
        """The scheduler no longer wants this import's KV; its load is still
        in flight and keeps pinning the blocks until it settles."""
        entry = self._entries.get((request_id, TransferDirection.IMPORT))
        if entry is not None:
            entry.retracted = True

    def is_retracted(self, request_id: str) -> bool:
        entry = self._entries.get((request_id, TransferDirection.IMPORT))
        return entry is not None and entry.retracted

    def mark_finished(self, request: Request) -> None:
        """The request finished while entries may still be open."""
        entry = self._entries.get((request.request_id, TransferDirection.IMPORT))
        if entry is not None and not entry.request_finished:
            entry.request_finished = True
            self._num_finished_imports += 1

    def drop_request(self, request_id: str) -> list[TransferEntry]:
        """Remove every entry of a request (its blocks were freed)."""
        return [
            e for d in TransferDirection if (e := self.close(request_id, d)) is not None
        ]

    # ---- queries ---------------------------------------------------------

    def is_referenced(self, request_id: str) -> bool:
        """Whether any transfer still needs this request's blocks."""
        return any((request_id, d) in self._entries for d in TransferDirection)

    def has_retained(self) -> bool:
        """Blocks of finished requests are still pinned by a transfer: either
        an export for a remote reader, or an import that was in flight when
        the request finished. This is what ``has_finished_requests()`` used
        to infer from ``len(requests) > running + waiting``."""
        return self._num_exports > 0 or self._num_finished_imports > 0

    def pins_blocks(self) -> bool:
        """Some transfer still references local blocks: an export a remote
        reads (dropped or not), or an import a remote writes. A pending import (nothing
        allocated yet) and a received one (blocks now the request's own) do
        not. This is what a cache reset or a memory release must wait for."""
        return self._num_exports > 0 or self._num_open_imports > 0

    def needs_driving(self) -> bool:
        """Some transfer needs the engine to keep stepping: to collect its
        completion, renew its lease, or start it. A received import that only
        awaits promotion does not: the scheduler promotes it when it next
        schedules, and a paused scheduler must not wait for that."""
        return self._num_exports > 0 or self._num_live_imports > 0
