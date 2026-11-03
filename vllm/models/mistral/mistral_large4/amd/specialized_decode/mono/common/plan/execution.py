# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The execution model every mono kernel is written for.

One CTA per CU of an MI355X, all of them co-resident -- a spin wait on another
CTA can only make progress if that CTA is running, so the runner refuses a GPU
with another CU count -- and THREADS threads each.
"""

BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64
LDS_BYTES = 160 * 1024  # a CU's LDS: one CTA's whole


def first_task(bid, base):
    """CTA ``bid``'s first task of a stage placed from CTA ``base`` (any count of
    CTAs past 0: placements that add task counts run past the grid), then every
    BLOCKS-th."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS
