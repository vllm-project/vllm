# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .rotary_embedding import native_rope

Rope = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        int,
        int,
        torch.Tensor,
        bool,
    ],
    tuple[torch.Tensor, torch.Tensor | None],
]


@dataclass(frozen=True)
class PlatformSpec:
    """Platform operations resolved at layer construction, with native defaults."""

    rope: Rope = native_rope
    rope_cache_dtype: torch.dtype | None = None
