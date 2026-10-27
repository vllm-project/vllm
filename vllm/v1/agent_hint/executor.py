# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1 context-edit executor skeleton.

The executor owns the management job lifecycle. Block resolution and the
CPU/disk transfer path are wired in M2; M1 accepts the request, assigns an
idempotent job id, and reports ``accepted`` with an explicit skeleton detail.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from vllm.v1.agent_hint.backend import get_context_edit_backend
from vllm.v1.agent_hint.protocol import AgentHint, AgentManagementResult

_M1_DETAIL = "M2 control-plane only: no context-edit backend registered"


@dataclass(frozen=True)
class _Job:
    result: AgentManagementResult


class ContextEditExecutor:
    """Tracks management jobs by operation id and job id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._by_operation: dict[str, str] = {}

    def submit(
        self,
        hint: AgentHint,
        request_id: str,
    ) -> AgentManagementResult:
        """Validate and accept one management request.

        ``operation_id`` is idempotent: replaying the same operation returns the
        original job. Missing session/edits are rejected without creating a job.
        """
        session_id = hint.session_id or ""
        if not session_id:
            return AgentManagementResult(
                session_id="",
                status="rejected",
                detail="agent_hint.session_id is required for management requests",
            )
        if hint.context_management is None or not hint.context_management.edits:
            return AgentManagementResult(
                session_id=session_id,
                status="rejected",
                detail="context_management.edits must not be empty",
            )

        operation_id = hint.operation_id
        with self._lock:
            if operation_id:
                existing = self._by_operation.get(operation_id)
                if existing is not None:
                    return self._jobs[existing].result
            job_id = f"ctx-{uuid.uuid4().hex}"
            backend = get_context_edit_backend()
            block_count = len(hint.block_hashes or [])
            if backend is None:
                result = AgentManagementResult(
                    job_id=job_id,
                    session_id=session_id,
                    status="accepted",
                    detail=(
                        f"{_M1_DETAIL} blocks={block_count} request_id={request_id}"
                    ),
                    blocks=block_count,
                )
            else:
                action = hint.context_management.edits[0].type
                try:
                    result = backend.execute(action, hint, job_id, request_id)
                except Exception as e:  # noqa: BLE001 - backend failures are isolated
                    result = AgentManagementResult(
                        job_id=job_id,
                        session_id=session_id,
                        status="failed",
                        detail=f"context edit backend failed: {e}",
                    )
            self._jobs[job_id] = _Job(result=result)
            if operation_id:
                self._by_operation[operation_id] = job_id
            return result

    def get(self, job_id: str) -> AgentManagementResult | None:
        """Return a previously accepted job, if any."""
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else job.result


__all__ = ["ContextEditExecutor"]
