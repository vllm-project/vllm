# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Self-preempting re-prefill at the position-range threshold.

A streaming session's mRoPE positions grow monotonically and eventually
exceed the model's trained range (`max_position_embeddings`), degrading
behaviour. To keep sessions effectively unbounded, the session self-preempts
once positions cross a configurable fraction of that range: the scheduler
frees the KV cache and re-queues the request for a fresh full prefill of its
surviving tokens, recomputing all mRoPE positions from 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request
    from vllm.v1.streaming.retention import StreamingRetentionParams


def should_trigger_reprefill(
    request: Request,
    retention: StreamingRetentionParams,
    model_max_position: int,
    highest_position: int | None = None,
) -> bool:
    """Return True once the session's highest cached position exceeds
    ``reprefill_threshold * model_max_position``.

    ``highest_position`` overrides the request's worker-reported
    ``max_cached_position`` watermark; 1D-RoPE (text) sessions pass the
    exactly-derivable ``num_tokens - 1 + position_offset``.

    A threshold >= 1.0 disables re-prefill entirely (the caller warned at
    admission; see ``StreamingRetentionParams.__post_init__``).
    """
    if retention.reprefill_threshold >= 1.0:
        return False
    if highest_position is None:
        highest_position = request.max_cached_position
    threshold = retention.reprefill_threshold * model_max_position
    return highest_position > threshold
