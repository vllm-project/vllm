# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCore-side context-edit entry.

This module runs inside the EngineCore process, where the scheduler's session
index and the live block pool are available. The API process reaches it through
``EngineCoreClient.call_utility_async("manage_context", ...)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.v1.agent_hint.executor import ContextEditExecutor
from vllm.v1.agent_hint.protocol import AgentHint, AgentManagementResult
from vllm.v1.agent_hint.session_index import SessionPrefixIndex

_ENGINE_EXECUTOR = ContextEditExecutor()


def _map_prefetch(result: Any, blocks: int) -> AgentManagementResult:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.prefetch import (
        PrefetchSubmitOutcome,
    )

    if result.outcome is PrefetchSubmitOutcome.ACCEPTED:
        return AgentManagementResult(
            session_id="",
            status="accepted",
            detail=(
                f"prefetch submitted {result.prefetch_id} "
                f"blocks={result.num_blocks_to_load}"
            ),
            tier="cpu",
            blocks=blocks,
        )
    if result.outcome is PrefetchSubmitOutcome.ALREADY_AVAILABLE:
        return AgentManagementResult(
            session_id="",
            status="accepted",
            detail="prefetch already available",
            tier="cpu",
            blocks=blocks,
        )
    if result.outcome is PrefetchSubmitOutcome.DEFERRED:
        return AgentManagementResult(
            session_id="",
            status="accepted",
            detail="prefetch deferred",
            tier="cpu",
            blocks=blocks,
        )
    if result.outcome is PrefetchSubmitOutcome.MISSING:
        return AgentManagementResult(
            session_id="",
            status="rejected",
            detail="prefetch target missing from offloaded tier",
            tier="cpu",
            blocks=blocks,
        )
    return AgentManagementResult(
        session_id="",
        status="unsupported",
        detail="request-free prefetch is disabled",
        tier="cpu",
        blocks=blocks,
    )


def execute_engine_management(
    hint: AgentHint,
    index: SessionPrefixIndex | None,
    prefetch: Callable[[list[str]], Any] | None = None,
) -> AgentManagementResult:
    """Resolve blocks from the session index and submit the edit."""
    if not hint.block_hashes and hint.session_id and index is not None:
        hint.block_hashes = list(index.lookup(hint.session_id))
    edits = hint.context_management.edits if hint.context_management else []
    if (
        edits
        and edits[0].type == "prefetch"
        and prefetch is not None
        and hint.block_hashes
    ):
        result = _map_prefetch(
            prefetch(hint.block_hashes or []),
            len(hint.block_hashes or []),
        )
        result.session_id = hint.session_id or ""
        return result
    return _ENGINE_EXECUTOR.submit(hint, "engine-manage")


__all__ = ["execute_engine_management"]
