# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch equivalents of the ``gpu/model_states/mamba_hybrid.py`` kernels.

Both run on every hybrid step, so they are the minimum a hybrid model needs on
CPU. ``idx_mapping`` may carry ``-1`` for rows filtered out under pipeline
parallelism, which the kernels skip rather than write.
"""

from typing import Any

import torch


def scatter_num_accepted(
    grid: tuple[int, ...],
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_accepted: torch.Tensor,
    **kwargs: Any,
) -> None:
    live = idx_mapping >= 0
    # Mamba treats 1 as the neutral non-spec value, and a chunked-prefill step
    # samples no token, so a zero count has to come through as 1.
    num_accepted[idx_mapping[live].long()] = (
        num_sampled[: idx_mapping.shape[0]][live].clamp_min(1).to(num_accepted.dtype)
    )


def fill_num_accepted(
    grid: tuple[int, ...],
    idx_mapping: torch.Tensor,
    num_accepted: torch.Tensor,
    num_sampled: int,
    **kwargs: Any,
) -> None:
    num_accepted[idx_mapping[idx_mapping >= 0].long()] = num_sampled
