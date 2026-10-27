# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight loaders for Inkling's own expert checkpoint layout."""

from collections.abc import Callable

import torch


def write_loader(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    write: Callable[[torch.Tensor, torch.Tensor], object],
) -> None:
    """Weight loader for Inkling's own checkpoint layout: ``write`` copies
    ``loaded_weight`` into ``param``. Loading through ``param.weight_loader``
    lets weight reloads, which defer and replay every load, see the write."""
    write(param, loaded_weight)


def deinterleave_into(
    slot: int, gate_start: int, up_start: int
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Write interleaved [g0, u0, g1, u1, ...] rows as [gate; up] into ``slot``."""

    def write(param: torch.Tensor, slab: torch.Tensor) -> None:
        # one bounded H2D upload, de-interleaved on device
        slab = slab.to(param.device)
        rows = slab.shape[0] // 2
        param.data[slot, gate_start : gate_start + rows].copy_(slab[0::2])
        param.data[slot, up_start : up_start + rows].copy_(slab[1::2])

    return write


def copy_into_slots(slots: list[int]) -> Callable[[torch.Tensor, torch.Tensor], None]:
    def write(param: torch.Tensor, weight: torch.Tensor) -> None:
        for i, slot in enumerate(slots):
            param.data[slot].copy_(weight[i])

    return write
