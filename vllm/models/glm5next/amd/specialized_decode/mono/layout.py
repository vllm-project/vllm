# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the GLM-5 MonoKernel of ROCm/FlyDSL#1204 (@coderfeli),
# kernels/monokernel/layout.py, and ROCm/ATOM#2435 (@XiaobingSuper),
# atom/model_ops/monokernel/layout.py (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors.

"""Grid, tile and cache-modifier constants of the persistent launch."""

from __future__ import annotations

BLOCKS = 256
LAYER_SLOTS = 128
THREADS = 512
WAVES = THREADS // 64
ROW_TILE = 32
ROUTER_TILE = 8
UG_TILE = 16
NEG = -1.0e30

CM_DEV = 16
CM_SYS = 17
POLL_MAX = 12


def atom_mxfp4_scale_index(row, col, cols):
    """Byte of an e8m0 scale in AITER's shuffled (e8m0_shuffle) layout."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, e = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * ((cols + 7) // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a
