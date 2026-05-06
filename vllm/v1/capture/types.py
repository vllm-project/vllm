# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core types for the capture-consumer framework.

This module is torch-aware — ``CaptureChunk`` carries a ``torch.Tensor``
so consumers can dispatch on captured activations without further
round-tripping. Unit tests exercising the dataclasses stay cheap by
using small CPU tensors.

See ``docs/design/capture_consumers.md`` § "Core Types" for the
authoritative field-by-field spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NewType

import torch

# ---------------------------------------------------------------------------
# Request identity
# ---------------------------------------------------------------------------

# The unique identifier vLLM assigns internally to a request. Always
# available; never client-controlled; opaque string. Consumers that want
# to correlate with external identity should declare the appropriate
# optional sidecar field (e.g., ``client_request_id``, ``tag``).
VllmInternalRequestId = NewType("VllmInternalRequestId", str)


# ---------------------------------------------------------------------------
# Hook points and position selector
# ---------------------------------------------------------------------------

# Mirrors ``_HOOK_NAME_TO_ID`` in
# ``vllm/model_executor/layers/activation_capture.py``. Any change to the
# set of hook points must be reflected there as well.
HookName = Literal[
    "pre_attn",
    "post_attn",
    "post_mlp",
    "mlp_in",
    "mlp_out",
]

PositionSelector = (
    Literal["last_prompt", "all_prompt", "all_generated", "all"] | list[int]
)


# ---------------------------------------------------------------------------
# CaptureKey
# ---------------------------------------------------------------------------

# ``(vllm_internal_request_id, layer_idx, hook_name)``.
CaptureKey = tuple[VllmInternalRequestId, int, str]


# ---------------------------------------------------------------------------
# CaptureSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureSpec:
    """Describes which activations to capture for a request.

    ``hooks`` maps each hook point to the layer indices at which the hook
    fires. An empty list disables the hook. ``positions`` selects which
    token positions are captured at every ``(hook, layer)`` pair.

    ``CaptureSpec`` is the in-framework representation produced by a
    consumer's ``global_capture_spec()`` or its per-request
    ``validate_client_spec()``. It is not directly serializable across
    the process boundary; consumers that ship specs via IPC should go
    through their own validator.
    """

    hooks: dict[HookName, list[int]]
    positions: PositionSelector


# ---------------------------------------------------------------------------
# CaptureChunk and CaptureFinalize
# ---------------------------------------------------------------------------


@dataclass
class CaptureChunk:
    """One batch of captured rows for a ``CaptureKey``.

    Emitted by the manager after every forward step that produced rows
    for this key. For a single key, chunks arrive in ``row_offset``
    order; different keys have no ordering relationship.
    """

    key: CaptureKey
    # CPU tensor, shape ``(num_rows, hidden_size)``.
    tensor: torch.Tensor
    # Explicit dtype to avoid ``tensor.dtype`` dispatch in consumers.
    dtype: torch.dtype
    # Cumulative row index within this key's sequence.
    row_offset: int
    # Which forward step produced this chunk.
    step_index: int
    # Per-chunk context — see ``docs/design/capture_consumers.md``
    # § "Manager Runtime" for what the manager populates and
    # § "Known Limitations" for the gaps.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureFinalize:
    """Request-completion signal for a ``CaptureKey``.

    Emitted by the manager when the owning request finishes (any finish
    reason). Arrives after all ``CaptureChunk``s for the key. On receipt
    the sink should flush any buffered state for this key and produce a
    terminal ``CaptureResult`` accessible via ``get_result(key)``.
    """

    key: CaptureKey
    sidecar: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CaptureResult
# ---------------------------------------------------------------------------

CaptureStatus = Literal[
    "pending",
    "ok",
    "partial_error",
    "error",
    "not_requested",
]


@dataclass
class CaptureResult:
    """Terminal per-key result from a consumer.

    Attached to ``RequestOutput.capture_results[consumer_name]`` on
    request completion. The ``payload`` field is consumer-specific and
    opaque to the framework — filesystem returns ``list[Path]``, a
    dashboard might return ``dict[str, str]``, a silent consumer returns
    ``None``.
    """

    key: CaptureKey
    status: CaptureStatus
    error: str | None = None
    payload: Any = None


# ---------------------------------------------------------------------------
# CaptureContext
# ---------------------------------------------------------------------------


@dataclass
class CaptureContext:
    """Per-request context passed to ``validate_client_spec``.

    Contains everything a validator needs to check a client spec against
    the request's actual shape. Fields are deliberately narrow —
    validators should not poke at ``vllm_config`` beyond these. If a
    validator needs more, add it here explicitly.
    """

    vllm_internal_request_id: VllmInternalRequestId
    num_prompt_tokens: int
    # Prefix-cache hits; positions below this index are already in the
    # KV cache and cannot be re-captured.
    num_computed_tokens: int
    num_hidden_layers: int
    hidden_size: int
    element_size_bytes: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
