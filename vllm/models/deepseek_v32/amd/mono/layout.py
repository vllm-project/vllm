# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at c39b56c36 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/layout.py

"""Shared compile-time layouts and schedules for fused model-layer kernels."""

from __future__ import annotations

from vllm.models.deepseek_v32.amd.mono.config import (
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
TL_COLS = 8


def atom_mxfp4_scale_index(row, col, cols):
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, e = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * ((cols + 7) // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a
