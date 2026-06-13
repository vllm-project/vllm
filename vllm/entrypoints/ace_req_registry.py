# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE request registry — maps vLLM request_id to ACE session_id (conversation_id).

This module is the bridge between two layers that cannot directly communicate:

  serving.py        — knows conversation_id, generates request_id per turn
  model_runner.py   — knows request_id, needs session_id to feed the tracker

## Lifecycle

  1. serving.py calls register(request_id, session_id) after generating request_id.
  2. model_runner.execute_model() calls get(req_id) for each request in the batch.
     If a session_id is returned, it installs ACE hooks before the forward pass
     and flushes them after.
  3. model_runner.finish_requests() calls release(req_id) when the request ends.
     This cleans up the registry entry; the AttentionImportanceTracker persists
     under session_id for the next turn.

Thread-safe: all operations hold a lock.
"""
from __future__ import annotations

import threading

_req_to_session: dict[str, str] = {}
_lock = threading.Lock()


def register(req_id: str, session_id: str) -> None:
    """Map a vLLM request_id to its ACE session_id."""
    with _lock:
        _req_to_session[req_id] = session_id


def get(req_id: str) -> str | None:
    """Return the ACE session_id for req_id, or None if not an ACE request."""
    with _lock:
        return _req_to_session.get(req_id)


def release(req_id: str) -> None:
    """Remove the registry entry for req_id (call when the request finishes)."""
    with _lock:
        _req_to_session.pop(req_id, None)
