# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference consumer that logs one line per finalized capture key.

Useful for debugging and verifying capture pipeline correctness.
Register via entry point ``logging`` in the ``vllm.capture_consumers``
group.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal

import torch

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import CaptureKey, CaptureSpec, HookName

logger = logging.getLogger("vllm.capture.logging")


class LoggingConsumer(CaptureConsumer):
    """Reference consumer that logs one line per finalized capture key.

    Useful for debugging and verifying capture pipeline correctness.

    Params:
        hooks: dict mapping hook name to layer indices (required)
        positions: position selector (default "last_prompt")
        level: logging level name (default "INFO")
    """

    location: ClassVar[Literal["worker", "driver"]] = "worker"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        self._hooks: dict[HookName, list[int]] = params["hooks"]
        self._positions = params.get("positions", "last_prompt")
        level_name = params.get("level", "INFO")
        self._level: int = getattr(logging, level_name.upper(), logging.INFO)

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(hooks=self._hooks, positions=self._positions)

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        logger.log(
            self._level,
            "capture key=%s rows=%d dtype=%s",
            key,
            tensor.shape[0] if tensor.ndim > 0 else 0,
            tensor.dtype,
        )
