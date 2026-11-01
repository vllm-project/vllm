# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

PDRole = Literal["prefill", "decode"]


class PDRoleState:
    """Coordinate a drained role change across every engine behind one frontend."""

    def __init__(
        self,
        role: PDRole,
        num_engines: int,
        call_all: Callable[..., Awaitable[list[dict[str, Any]]]],
    ) -> None:
        self.role = role
        self.epoch = 0
        self.phase = "ready"
        self.active_requests = 0
        self.idle = asyncio.Event()
        self.idle.set()
        self.num_engines = num_engines
        self.call_all = call_all
        self.task: asyncio.Task | None = None
        self.error: str | None = None
        self.target_role: PDRole | None = None
        self.started_at: float | None = None
        self.duration: float | None = None

    def status(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "epoch": self.epoch,
            "phase": self.phase,
            "target_role": self.target_role,
            "active_requests": self.active_requests,
            "error": self.error,
            "transition_seconds": self.duration,
        }

    def start(self, role: PDRole, expected_epoch: int, timeout: float) -> None:
        if self.phase != "ready":
            raise ValueError("A role transition is already in progress or failed")
        if self.role == role and expected_epoch in (self.epoch, self.epoch - 1):
            return
        if expected_epoch != self.epoch:
            raise ValueError("Stale role epoch")
        self.phase = "draining"
        self.target_role = role
        self.error = None
        self.duration = None
        self.started_at = time.monotonic()
        # Keep the transition alive after caller disconnects.
        self.task = asyncio.create_task(self._switch(role, expected_epoch, timeout))

    def _check_ranks(
        self, statuses: list[dict[str, Any]], role: PDRole, epoch: int
    ) -> None:
        if len(statuses) != self.num_engines or any(
            s["role"] != role or s["epoch"] != epoch for s in statuses
        ):
            raise RuntimeError("Engine ranks disagree on the serving role or epoch")

    async def _prepare_and_commit(self, role: PDRole, epoch: int) -> None:
        # Finish admitted HTTP work before fencing engine requests.
        await self.idle.wait()
        self.phase = "preparing"
        statuses = await self.call_all("prepare_pd_role", role, epoch)
        self._check_ranks(statuses, self.role, epoch)
        while True:
            statuses = await self.call_all("get_pd_role_status")
            self._check_ranks(statuses, self.role, epoch)
            if all(s["drained"] for s in statuses):
                break
            await asyncio.sleep(0.01)

        self.phase = "committing"
        statuses = await self.call_all("commit_pd_role", role, epoch)
        self._check_ranks(statuses, role, epoch + 1)
        self.role = role
        self.epoch = epoch + 1

    async def _switch(self, role: PDRole, epoch: int, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._prepare_and_commit(role, epoch), timeout)
            self.phase = "ready"
        except Exception as exc:
            prepared = self.phase == "preparing"
            committing = self.phase == "committing"
            self.error = f"{type(exc).__name__}: {exc}"
            # Do not reopen admission after an uncertain commit.
            self.phase = "failed" if committing else "rolling_back"
            if not committing:
                try:
                    if prepared:
                        statuses = await asyncio.wait_for(
                            self.call_all("cancel_pd_role", epoch), timeout
                        )
                        self._check_ranks(statuses, self.role, epoch)
                        if any(s["pending_role"] is not None for s in statuses):
                            raise RuntimeError("Engine ranks are still fenced")
                    self.phase = "ready"
                except Exception as rollback_exc:
                    self.phase = "failed"
                    self.error += f"; rollback failed: {rollback_exc}"
        finally:
            assert self.started_at is not None
            self.duration = time.monotonic() - self.started_at
