# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass

import torch

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
    """Optional platform implementations resolved at layer construction."""

    rope: Rope | None = None
    rope_cache_dtype: torch.dtype | None = None
