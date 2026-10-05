# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as vendored by
# ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/layout.py). The original source code was
# licensed under the Apache License 2.0 and included the following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths rewritten to this package;
#   reduced to the constants the GLM-5 MonoKernel uses.

"""Shared compile-time constants of the MonoKernel."""

from vllm.models.deepseek_v32.amd.mono.kernel.config import MAX_LAYERS_PER_STEP

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
