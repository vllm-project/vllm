# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agent Hint management routing.

M1 accepts management requests into a job registry. Block resolution and the
actual CPU/disk transfer are intentionally deferred to M2.
"""

from __future__ import annotations

from vllm.v1.agent_hint.executor import ContextEditExecutor
from vllm.v1.agent_hint.protocol import (
    AgentHint,
    AgentManagementResult,
)

_executor = ContextEditExecutor()


def is_management_request(hint: AgentHint | None) -> bool:
    """Return whether ``hint`` selects the management-only request path."""
    return bool(
        hint is not None
        and hint.context_management is not None
        and hint.context_management.manage_request
    )


def handle_agent_management(
    hint: AgentHint | None,
    request_id: str,
) -> AgentManagementResult | None:
    """Return the management result, or ``None`` for normal chat requests."""
    if not is_management_request(hint):
        return None
    assert hint is not None
    return _executor.submit(hint, request_id)


def get_agent_management_job(job_id: str) -> AgentManagementResult | None:
    """Return a previously accepted management job."""
    return _executor.get(job_id)


__all__ = [
    "get_agent_management_job",
    "handle_agent_management",
    "is_management_request",
]
