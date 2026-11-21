# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident CTA grid for the V4.1 kernel modules, imported on the bound worker."""

from .plan import BLOCKS as GRID_MAX  # noqa: F401 (the 256-CTA layout's grid)
from .plan import device_grid

_, BLOCKS = device_grid()


def first_task(bid, base):
    """CTA bid's first task in a round-robin stage starting at base."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS
