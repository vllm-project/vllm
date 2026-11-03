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
import time
from dataclasses import dataclass, field
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


@dataclass
class TransferEntry:
    request: Request
    direction: TransferDirection
    state: TransferState
    opened_at: float = field(default_factory=time.monotonic)
    outcome: str | None = None
    # The request finished (e.g. was aborted) while the entry was open.
    request_finished: bool = False

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

    # ---- open ------------------------------------------------------------

    def open_export(self, request: Request) -> TransferEntry:
        """A finished request's blocks are kept for a remote reader."""
        key = (request.request_id, TransferDirection.EXPORT)
        entry = self._entries.get(key)
        if entry is None:
            entry = TransferEntry(request, TransferDirection.EXPORT, TransferState.OPEN)
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
        return entry

    # ---- transitions -----------------------------------------------------

    def complete(
        self, request_id: str, direction: TransferDirection, outcome: str = "ok"
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
        entry.state = TransferState.CLOSING
        entry.outcome = outcome
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
            if entry.request_finished:
                self._num_finished_imports -= 1
        return entry

    def drop_exports(self) -> list[TransferEntry]:
        """Forget every export: the engine stops holding blocks for remote
        readers (abort). The caller frees the blocks."""
        keys = [k for k in self._entries if k[1] == TransferDirection.EXPORT]
        return [e for k in keys if (e := self.close(*k)) is not None]

    def mark_finished(self, request: Request) -> None:
        """The request finished while entries may still be open."""
        entry = self._entries.get((request.request_id, TransferDirection.IMPORT))
        if entry is not None and not entry.request_finished:
            entry.request_finished = True
            self._num_finished_imports += 1

    def drop_request(self, request_id: str) -> list[TransferEntry]:
        """Remove every entry of a request (its blocks were freed)."""
        return [
            e
            for d in TransferDirection
            if (e := self.close(request_id, d)) is not None
        ]

    # ---- queries ---------------------------------------------------------

    def get(
        self, request_id: str, direction: TransferDirection
    ) -> TransferEntry | None:
        return self._entries.get((request_id, direction))

    def entries_for(self, request_id: str) -> list[TransferEntry]:
        return [
            e
            for d in TransferDirection
            if (e := self._entries.get((request_id, d))) is not None
        ]

    def is_referenced(self, request_id: str) -> bool:
        """Whether any transfer still needs this request's blocks."""
        return any(
            (request_id, d) in self._entries for d in TransferDirection
        )

    def has_exports(self) -> bool:
        return self._num_exports > 0

    def has_retained(self) -> bool:
        """Blocks of finished requests are still pinned by a transfer: either
        an export for a remote reader, or an import that was in flight when
        the request finished. This is what ``has_finished_requests()`` used
        to infer from ``len(requests) > running + waiting``."""
        return self._num_exports > 0 or self._num_finished_imports > 0

    def needs_driving(self) -> bool:
        """Some transfer needs the engine to keep stepping: to collect its
        completion, renew its lease, or start it. A received import that only
        awaits promotion does not: the scheduler promotes it when it next
        schedules, and a paused scheduler must not wait for that."""
        return self._num_exports > 0 or self._num_live_imports > 0

    def is_empty(self) -> bool:
        return not self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        out = {"exports": 0, "imports_pending": 0, "imports_open": 0, "closing": 0}
        for e in self._entries.values():
            if e.direction == TransferDirection.EXPORT:
                out["exports"] += 1
            elif e.state == TransferState.PENDING:
                out["imports_pending"] += 1
            elif e.state == TransferState.OPEN:
                out["imports_open"] += 1
            if e.state == TransferState.CLOSING:
                out["closing"] += 1
        return out
