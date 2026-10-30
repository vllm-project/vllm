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


def _fill(dst: torch.Tensor, value: float) -> None:
    # copy_, not fill_: layerwise reload counts copied elements to know when a
    # layer is complete
    if dst.numel():
        dst.copy_(torch.full_like(dst, value))


def deinterleave_into(
    slot: int, half: int, pad: float | None = None
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Write interleaved [g0, u0, g1, u1, ...] rows as [gate; up] into ``slot``,
    each half ``half`` rows long. Rows past the checkpoint's are padding: if
    ``pad`` is given they are set to it, so a reload (which restores the layer
    uninitialized) writes the whole slot as cold start's initialization did."""

    def write(param: torch.Tensor, slab: torch.Tensor) -> None:
        # one bounded H2D upload, de-interleaved on device
        slab = slab.to(param.device)
        rows = slab.shape[0] // 2
        dst = param.data[slot]
        for start, src in ((0, slab[0::2]), (half, slab[1::2])):
            dst[start : start + rows].copy_(src)
            if pad is not None:
                _fill(dst[start + rows : start + half], pad)

    return write


def copy_padded(
    slot: int, width: int, pad: float | None = None
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Write into the first ``width`` columns of ``slot``. If ``pad`` is given,
    also set the padding columns after them to it."""

    def write(param: torch.Tensor, weight: torch.Tensor) -> None:
        dst = param.data[slot]
        dst[..., :width].copy_(weight)
        if pad is not None:
            _fill(dst[..., width:], pad)

    return write


def copy_into_slots(slots: list[int]) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Write ``weight[i]`` into slot ``slots[i]`` of the parameter's leading
    (expert) dim, after one upload of ``weight`` to the parameter's device."""

    def write(param: torch.Tensor, weight: torch.Tensor) -> None:
        weight = weight.to(param.device)
        for i, slot in enumerate(slots):
            param.data[slot].copy_(weight[i])

    return write
