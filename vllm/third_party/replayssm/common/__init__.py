# SPDX-License-Identifier: Apache-2.0
"""Local helpers for the pinned ReplaySSM kernels."""

import math

from cutlass import cute

from .cute_compat import simple_tma_copy

__all__ = ["simple_tma_copy", "make_fake_tensor"]


def make_fake_tensor(dtype, shape, *, divisibility=16, slot_stride=None):
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    if slot_stride is not None:
        strides[0] = slot_stride
    return cute.runtime.make_fake_tensor(
        dtype, shape, tuple(strides), assumed_align=divisibility
    )
