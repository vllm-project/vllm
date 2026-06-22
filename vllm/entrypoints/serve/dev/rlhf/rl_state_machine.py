# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RL lifecycle state machine for validating endpoint call ordering.

The RL weight-transfer protocol has ordering constraints that, if violated,
produce silent failures or TMA descriptor errors in production:

  Valid sequences
  ───────────────
  Normal generation:
    (awake + not_updating) → rollout → repeat

  Full RL step:
    awake → sleep → start_weight_update → update_weights* → finish_weight_update
           → wake_up → rollout → repeat

  Pause-only (no sleep):
    awake → pause → start_weight_update → … → finish_weight_update → resume

  Constraints enforced
  ────────────────────
  • start_weight_update requires is_updating == False (no double-start)
  • finish_weight_update requires is_updating == True (must follow start)
  • update_weights     requires is_updating == True (must follow start)

This module provides ``RLStateMachineState``, a thread-safe state object
attached to ``app.state.rl_state`` at router registration time, and
``enforce_weight_update_ordering``, an async dependency that raises
``HTTPException(409)`` on ordering violations.

Design note: we deliberately do NOT enforce "must be sleeping to transfer
weights" because the valid colocate pattern (separate GPUs) transfers while
awake.  We only guard against the protocol-level ordering violations that
would corrupt the engine state.
"""
from __future__ import annotations

import asyncio
from http import HTTPStatus

from fastapi import HTTPException, Request


class RLStateMachineState:
    """Async-safe tracker for weight-update protocol state.

    Uses asyncio.Lock so it does not block the event loop — all HTTP handlers
    are async coroutines; a threading.Lock would suspend the loop while held.

    Attached to ``app.state.rl_state`` at router registration time.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None  # created lazily on first use
        self._is_updating: bool = False

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ── Public queries ─────────────────────────────────────────────────────

    @property
    def is_updating(self) -> bool:
        return self._is_updating

    # ── Transitions (async — acquire event-loop lock) ──────────────────────

    async def on_start_weight_update(self) -> None:
        """Mark a weight update as in-progress.

        Called AFTER the engine RPC succeeds so we don't leave the flag stuck
        if the engine call raises.

        Raises:
            RuntimeError: If a weight update is already active.
        """
        async with self._get_lock():
            if self._is_updating:
                raise RuntimeError(
                    "start_weight_update called while a weight update is already "
                    "in progress.  Call finish_weight_update first."
                )
            self._is_updating = True

    async def on_finish_weight_update(self) -> None:
        """Mark a weight update as finished.

        Called AFTER the engine RPC succeeds.

        Raises:
            RuntimeError: If no weight update is currently in progress.
        """
        async with self._get_lock():
            if not self._is_updating:
                raise RuntimeError(
                    "finish_weight_update called without a preceding "
                    "start_weight_update."
                )
            self._is_updating = False

    async def on_update_weights(self) -> None:
        """Assert that a weight update is in progress (update_weights ordering).

        Raises:
            RuntimeError: If start_weight_update has not been called.
        """
        async with self._get_lock():
            if not self._is_updating:
                raise RuntimeError(
                    "update_weights called without a preceding start_weight_update."
                )

    async def reset(self) -> None:
        """Force-clear the updating flag (crash recovery / test teardown)."""
        async with self._get_lock():
            self._is_updating = False

    def to_dict(self) -> dict:
        return {"weight_update_active": self._is_updating}


# ── FastAPI dependencies ────────────────────────────────────────────────────


def _sm(request: Request) -> RLStateMachineState:
    return request.app.state.rl_state


async def require_update_active(request: Request) -> None:
    """FastAPI dependency: raise 409 if no weight update is in progress."""
    sm = _sm(request)
    if not sm.is_updating:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail=(
                "update_weights requires a preceding start_weight_update. "
                "Call POST /start_weight_update first."
            ),
        )


async def require_update_inactive(request: Request) -> None:
    """FastAPI dependency: raise 409 if a weight update IS in progress."""
    sm = _sm(request)
    if sm.is_updating:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail=(
                "start_weight_update called while a weight update is already "
                "in progress.  Call POST /finish_weight_update first."
            ),
        )
