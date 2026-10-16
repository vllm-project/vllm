# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as
# vendored by ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/packing.py).
# The original source code was licensed under the Apache License 2.0 and included the
# following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths
#   rewritten to this package; reduced to the FP8 tile packer the GLM-5 MonoKernel uses.

"""Host-side weight packing for the MonoKernel launch wrapper."""

from __future__ import annotations

import torch


def pack_fp8(q: torch.Tensor) -> torch.Tensor:
    """Pack FP8 ``[..., N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""
    *lead, rows, k = q.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"FP8 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w8 = q.view(torch.uint8).reshape(*lead, rows // 16, 16, k // 64, 2, 4, 8)
    nlead = len(lead)
    order = list(range(nlead)) + [nlead + position for position in (0, 2, 4, 1, 3, 5)]
    return w8.permute(*order).contiguous().view(-1)
