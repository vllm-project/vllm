# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the layer builders and launches share: the rank's context and the step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class DecodeContext:
    """One rank's config, TP group and device, plus the state the builders and
    binds keep (packed formats, caches, per-width runtimes)."""

    vllm_config: VllmConfig
    tp_size: int
    tp_rank: int
    cpu_group: Any
    device: torch.device
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeStep:
    """One routed decode step, shared by every kernel layer of the forward."""

    tokens: int
    width: int
    positions: torch.Tensor
    attn_metadata: Mapping[str, Any]
