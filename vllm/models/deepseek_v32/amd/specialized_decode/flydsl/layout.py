# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as
# vendored by ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/layout.py).
# The original source code was licensed under the Apache License 2.0 and included the
# following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths
#   rewritten to this package; reduced to the constants the GLM-5 MonoKernel uses.

"""Shared compile-time constants of the MonoKernel."""

from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.config import (
    MAX_LAYERS_PER_STEP,
)

BLOCKS = 256
LAYER_SLOTS = MAX_LAYERS_PER_STEP
THREADS = 512
WAVES = THREADS // 64
QKV_A_TILE = 16
Q_B_TILE = 16
UK_TILE = 128
UV_TILE = 64
ROW_TILE = 32
ROUTER_TILE = 8
UG_TILE = 16
NEG = -1.0e30

CM_DEV = 16
CM_SYS = 17
POLL_MAX = 12


def atom_mxfp4_scale_index(row, col, cols):
    """Byte of E8M0 scale (row, 32-group col) in aiter's ``e8m0_shuffle`` layout.

    The [rows, cols] scale matrix is
    ``view(rows / 32, 2, 16, cols / 8, 2, 4).permute(0, 3, 5, 2, 4, 1)``.
    """
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, e = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * ((cols + 7) // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a
