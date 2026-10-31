# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-to-block index for Agent Hint context management.

The index is populated by the scheduler at successful request completion, where
the request's cached blocks are still alive. M3 exposes lookup for the future
engine-side management executor; the API-process executor cannot read this
index yet.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict


class SessionPrefixIndex:
    """Bounded LRU map of session id to cached block-hash hex strings."""

    def __init__(self, max_sessions: int = 4096, ttl_s: int = 3600) -> None:
        self._max_sessions = max(1, int(max_sessions))
        self._ttl_s = max(1, int(ttl_s))
        self._records: OrderedDict[str, tuple[float, tuple[str, ...]]] = OrderedDict()
        self._lock = threading.Lock()

    def record(self, session_id: str, block_hashes: list[str]) -> None:
        """Replace one session's block list; empty input forgets the session."""
        if not isinstance(session_id, str) or not session_id:
            return
        with self._lock:
            if not block_hashes:
                self._records.pop(session_id, None)
                return
            self._records[session_id] = (time.monotonic(), tuple(block_hashes))
            self._records.move_to_end(session_id)
            while len(self._records) > self._max_sessions:
                self._records.popitem(last=False)

    def lookup(self, session_id: str) -> tuple[str, ...]:
        """Return the current block hashes, honouring the lazy TTL."""
        if not isinstance(session_id, str) or not session_id:
            return ()
        now = time.monotonic()
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return ()
            recorded_at, hashes = record
            if now - recorded_at > self._ttl_s:
                del self._records[session_id]
                return ()
            self._records.move_to_end(session_id)
            return hashes

    def forget(self, session_id: str) -> None:
        """Drop one session entry."""
        with self._lock:
            self._records.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
