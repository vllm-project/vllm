# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed Agent Hint context-management models (M0).

The wire shape follows vLLM RFC #52113: an optional ``agent_hint`` block on the
OpenAI chat request, with ``context_management.manage_request`` selecting the
management-only path. M0 validates and routes the request; actual offload and
prefetch execution lands in later milestones.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentContextEdit(BaseModel):
    """One context edit requested by the orchestrator."""

    model_config = ConfigDict(extra="allow")

    type: Literal["offload", "prefetch", "evict"]
    target: Literal["session"] = "session"
    block_start: int | None = Field(default=None, ge=0)
    block_end: int | None = Field(default=None, ge=0)


class AgentContextManagement(BaseModel):
    """Context-management block carried on ``agent_hint``."""

    model_config = ConfigDict(extra="allow")

    manage_request: bool = False
    edits: list[AgentContextEdit] = Field(default_factory=list)


class AgentPrediction(BaseModel):
    """Prediction fields consumed by context-management policy."""

    model_config = ConfigDict(extra="allow")

    expected_resume_s: int | None = Field(default=None, ge=0)


class AgentHint(BaseModel):
    """Optional agent-hint metadata attached to an OpenAI chat request."""

    model_config = ConfigDict(extra="allow")

    session_id: str | None = None
    operation_id: str | None = None
    context_management: AgentContextManagement | None = None
    prediction: AgentPrediction | None = None
    block_hashes: list[str] | None = None


class AgentManagementResult(BaseModel):
    """Result surfaced as the ``agent_management`` response extension."""

    job_id: str | None = None
    session_id: str
    status: Literal["accepted", "unsupported", "rejected", "failed"]
    detail: str | None = None
    tier: Literal["npu_retain", "cpu", "disk"] | None = None
    blocks: int | None = None
    bytes: int | None = None


__all__ = [
    "AgentContextEdit",
    "AgentContextManagement",
    "AgentHint",
    "AgentManagementResult",
    "AgentPrediction",
]
