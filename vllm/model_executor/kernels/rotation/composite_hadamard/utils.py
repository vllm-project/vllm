# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Shape selection for composite Hadamard transforms."""

from __future__ import annotations


def is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def resolve_hadamard_dim(
    hidden_size: int,
    k: int | None = None,
    p: int | None = None,
) -> tuple[int, int]:
    if k is None:
        if p is None:
            raise ValueError("K or P must be specified")
        if p <= 0 or hidden_size % p:
            raise ValueError("P must divide hidden_size")
        k = hidden_size // p
    elif p is None:
        if k <= 0 or hidden_size % k:
            raise ValueError("K must divide hidden_size")
        p = hidden_size // k
    if k * p != hidden_size:
        raise ValueError("hidden_size must equal K * P")
    if not is_pow2(p):
        raise ValueError("P must be a power of two")
    return k, p
