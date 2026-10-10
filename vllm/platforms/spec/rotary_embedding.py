# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch


def custom_rope(
    kernel: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor | None, int, torch.Tensor, bool],
        None,
    ],
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    rotary_dim: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Adapt existing in-place kernels, which infer rotary_dim from the cache."""
    kernel(positions, query, key, head_size, cos_sin_cache, is_neox_style)
    return query, key
