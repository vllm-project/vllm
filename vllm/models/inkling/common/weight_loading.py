# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch


def load_expert_scales(
    param: torch.nn.Parameter, weight: torch.Tensor, slots: dict[int, int]
) -> None:
    """Copy scale views in one batch for a contiguous expert partition."""
    if not slots:
        return

    width = math.prod(param.shape[1:])
    source_width = math.prod(weight.shape[1:])
    if source_width not in (1, width):
        raise ValueError(f"cannot load {tuple(weight.shape)} into {tuple(param.shape)}")

    first_gid, first_lid = next(iter(slots.items()))
    contiguous = all(
        gid == first_gid + offset and lid == first_lid + offset
        for offset, (gid, lid) in enumerate(slots.items())
    )
    if not contiguous:
        for gid, lid in slots.items():
            src = weight[gid].reshape(-1)
            dst = param.data[lid].view(-1)
            if source_width == width:
                dst.copy_(src)
            else:
                for half in range(width):
                    dst.narrow(0, half, 1).copy_(src)
        return

    count = len(slots)
    src = weight.narrow(0, first_gid, count).view(count, source_width)
    dst = param.data.narrow(0, first_lid, count).view(count, width)
    if source_width == width:
        dst.copy_(src)
    else:
        # The checkpoint scalar supplies both gate/up halves.
        for half in range(width):
            dst.narrow(1, half, 1).copy_(src)
