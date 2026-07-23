# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-only fake tensor for TileLang warmup.

TileLang's ``JITImpl.compile()`` only inspects ``.shape`` / ``.stride()`` /
``.dtype`` to build the TIR PrimFunc; it never calls ``.data_ptr()``. So a
fake tensor exposing just those three is enough to trigger JIT compilation
without allocating GPU memory or launching the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class TileLangWarmupTensor:
    dtype: torch.dtype
    shape: tuple[int, ...] = (1,)
    strides: tuple[int, ...] | None = field(default=None)

    def stride(self) -> tuple[int, ...]:
        if self.strides is not None:
            return self.strides
        strides: list[int] = []
        s = 1
        for size in reversed(self.shape):
            strides.append(s)
            s *= size
        return tuple(reversed(strides))
