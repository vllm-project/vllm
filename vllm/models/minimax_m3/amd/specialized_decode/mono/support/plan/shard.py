# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/mono/plan/shard.py
# Original code is MIT-licensed; see mono/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A TP rank's share of a model's widths.

A mono kernel is built for one TP size: every per-rank width it loops over,
places tasks by or lays a region out with is a build-time constant. A model
derives all of them from ``Shard`` in one place (its ``Dims``), so a TP size
either yields every width exactly or is refused by name before anything loads.
"""

from dataclasses import dataclass


class ShardError(ValueError):
    """A width the TP size does not divide, or a TP size the model refuses."""


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class Shard:
    tp: int

    def __post_init__(self):
        if self.tp < 1:
            raise ShardError(f"TP {self.tp}")

    def split(self, full: int, what: str) -> int:
        """``full`` / tp, refusing a remainder: a rank's share of ``what``."""
        if full % self.tp:
            raise ShardError(f"{what} {full} is not divisible by TP {self.tp}")
        return full // self.tp


def padded(n: int, align: int) -> int:
    return cdiv(n, align) * align


@dataclass(frozen=True)
class Tiles:
    """``n`` items (a rank's heads) in tiles of ``width`` (an MFMA's N): the
    last tile ``ragged`` when ``width`` does not divide ``n``, its items past
    ``n`` dead -- a kernel reads a live one in their place and writes nothing."""

    n: int
    width: int

    @property
    def count(self) -> int:
        return cdiv(self.n, self.width)

    @property
    def ragged(self) -> bool:
        return self.n % self.width != 0


def shard_refusal(make, tp: int) -> str | None:
    """Why ``make(tp)`` (a model's ``Dims``) refuses TP ``tp``, or None."""
    try:
        make(tp)
    except ShardError as e:
        return str(e)
    return None
