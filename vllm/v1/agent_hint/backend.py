# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform backend seam for Agent Hint context edits.

M2 defines the call surface and registry. Platform packages (for example
vllm-ascend) register a backend at startup; the core executor invokes it with
the validated management request. The backend owns tier selection and the
actual data movement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vllm.v1.agent_hint.protocol import AgentHint, AgentManagementResult


@runtime_checkable
class ContextEditBackend(Protocol):
    """Platform implementation of one context edit."""

    def execute(
        self,
        action: str,
        hint: "AgentHint",
        job_id: str,
        request_id: str,
    ) -> "AgentManagementResult":
        """Execute ``action`` (offload/prefetch/evict) for ``hint``."""
        ...


_backend: ContextEditBackend | None = None


def register_context_edit_backend(backend: ContextEditBackend) -> None:
    """Register or replace the process-wide context-edit backend."""
    global _backend
    _backend = backend


def get_context_edit_backend() -> ContextEditBackend | None:
    """Return the registered backend, if any."""
    return _backend


__all__ = [
    "ContextEditBackend",
    "get_context_edit_backend",
    "register_context_edit_backend",
]
