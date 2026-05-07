# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public API for the capture-consumer framework.

See ``docs/design/capture_consumers.md`` for the design spec and
``docs/capture_consumers/roadmap.md`` for the phase-by-phase rollout.
"""

from __future__ import annotations

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.errors import (
    CaptureValidationError,
    UnknownCaptureConsumerError,
)
from vllm.v1.capture.sink import CaptureSink
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureContext,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
    CaptureStatus,
    HookName,
    PositionSelector,
    VllmInternalRequestId,
)

__all__ = [
    "CaptureChunk",
    "CaptureConsumer",
    "CaptureContext",
    "CaptureFinalize",
    "CaptureKey",
    "CaptureResult",
    "CaptureSink",
    "CaptureSpec",
    "CaptureStatus",
    "CaptureValidationError",
    "HookName",
    "PositionSelector",
    "UnknownCaptureConsumerError",
    "VllmInternalRequestId",
]
