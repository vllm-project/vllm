# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight loaders for Inkling's own expert checkpoint layout."""

from collections.abc import Callable

import torch

Writer = Callable[[torch.Tensor, torch.Tensor], object]


def write_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor, write: Writer
) -> None:
    """Weight loader that runs ``write(param, loaded_weight)``; loading through
    ``param.weight_loader`` lets a weight reload defer and replay the write."""
    write(param, loaded_weight)


def _fill(dst: torch.Tensor, value: float) -> None:
    # copy_, not fill_: layerwise reload counts copied elements to know when a
    # layer is complete
    if dst.numel():
        dst.copy_(torch.full_like(dst, value))


def fill(value: float) -> Writer:
    """Set every element to ``value``, in the parameter's dtype."""
    return lambda param, _: _fill(param.data, value)


def deinterleave_into(slot: int, half: int, pad: float | None = None) -> Writer:
    """Write interleaved [g0, u0, g1, u1, ...] rows as [gate; up] into ``slot``,
    each half ``half`` rows long. Rows past the checkpoint's are padding, set
    to ``pad`` when given (a reload restores the layer uninitialized)."""

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


def copy_padded(slot: int, width: int, pad: float | None = None) -> Writer:
    """Write into the first ``width`` columns of ``slot``; the columns after
    them are padding, set to ``pad`` when given."""

    def write(param: torch.Tensor, weight: torch.Tensor) -> None:
        dst = param.data[slot]
        dst[..., :width].copy_(weight)
        if pad is not None:
            _fill(dst[..., width:], pad)

    return write


def copy_into_slots(slots: list[int]) -> Writer:
    """Write ``weight[i]`` into slot ``slots[i]`` of the parameter's leading
    (expert) dim, after one upload of ``weight`` to the parameter's device."""

    def write(param: torch.Tensor, weight: torch.Tensor) -> None:
        weight = weight.to(param.device)
        for i, slot in enumerate(slots):
            param.data[slot].copy_(weight[i])

    return write
