# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export capture-consumer config types for ``vllm.config`` consumers.

The canonical definitions live in :mod:`vllm.v1.capture.config`.  This
module exists so that ``vllm/config/vllm.py`` can follow the same
relative-import pattern used by all other config submodules.
"""

from __future__ import annotations

from vllm.v1.capture.config import (
    CaptureConsumersConfig,
    CaptureConsumerSpec,
)

__all__ = [
    "CaptureConsumerSpec",
    "CaptureConsumersConfig",
]
