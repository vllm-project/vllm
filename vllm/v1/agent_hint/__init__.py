# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agent Hint context-management types and M0 routing."""

from vllm.v1.agent_hint.backend import (
    ContextEditBackend,
    get_context_edit_backend,
    register_context_edit_backend,
)
from vllm.v1.agent_hint.executor import ContextEditExecutor
from vllm.v1.agent_hint.manager import (
    get_agent_management_job,
    handle_agent_management,
    is_management_request,
)
from vllm.v1.agent_hint.protocol import (
    AgentContextEdit,
    AgentContextManagement,
    AgentHint,
    AgentManagementResult,
    AgentPrediction,
)
from vllm.v1.agent_hint.session_index import SessionPrefixIndex

__all__ = [
    "AgentContextEdit",
    "AgentContextManagement",
    "AgentHint",
    "AgentManagementResult",
    "AgentPrediction",
    "ContextEditBackend",
    "ContextEditExecutor",
    "SessionPrefixIndex",
    "get_context_edit_backend",
    "get_agent_management_job",
    "handle_agent_management",
    "is_management_request",
    "register_context_edit_backend",
]
