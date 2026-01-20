# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# vllm/engine/inference_profile.py
from dataclasses import dataclass
from typing import Literal

InferenceMode = Literal["interactive", "batch", "background"]


@dataclass(frozen=True)
class InferenceProfile:
    """
    Describes high-level inference intent.

    This is a *hint* to the engine/scheduler, not a hard constraint.
    """

    mode: InferenceMode = "interactive"
    max_latency_ms: int | None = None
    max_memory_mb: int | None = None
    priority: int = 0  # higher = more important
