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

import threading
from http import HTTPStatus

from fastapi import HTTPException, Request


class RLStateMachineState:
    """Thread-safe tracker for weight-update protocol state.

    Attached to ``app.state.rl_state`` at router registration time.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_updating: bool = False

    # ── Public queries ─────────────────────────────────────────────────────

    @property
    def is_updating(self) -> bool:
        with self._lock:
            return self._is_updating

    # ── Transitions ────────────────────────────────────────────────────────

    def on_start_weight_update(self) -> None:
        """Mark a weight update as in-progress.

        Raises:
            RuntimeError: If a weight update is already active.
        """
        with self._lock:
            if self._is_updating:
                raise RuntimeError(
                    "start_weight_update called while a weight update is already "
                    "in progress.  Call finish_weight_update first."
                )
            self._is_updating = True

    def on_finish_weight_update(self) -> None:
        """Mark a weight update as finished.

        Raises:
            RuntimeError: If no weight update is currently in progress.
        """
        with self._lock:
            if not self._is_updating:
                raise RuntimeError(
                    "finish_weight_update called without a preceding "
                    "start_weight_update."
                )
            self._is_updating = False

    def on_update_weights(self) -> None:
        """Assert that a weight update is in progress (update_weights ordering).

        Raises:
            RuntimeError: If start_weight_update has not been called.
        """
        with self._lock:
            if not self._is_updating:
                raise RuntimeError(
                    "update_weights called without a preceding start_weight_update."
                )

    def reset(self) -> None:
        """Force-clear the updating flag (for crash recovery / test teardown)."""
        with self._lock:
            self._is_updating = False

    def to_dict(self) -> dict:
        with self._lock:
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
