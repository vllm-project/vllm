# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ATOM's MXFP4 expert-storage index arithmetic, and the size of a symmetric
all-reduce buffer.

Model-free: the MXFP4 functions describe ATOM's storage ABI, which every mono
kernel reading ATOM expert weights has to agree with, and the buffer size
follows from the TP group rather than from any model's geometry. A kernel's own
tile sizes and stage list are NOT here -- those are named after that kernel's
stages and live with it.
"""

from __future__ import annotations


def atom_mxfp4_weight_index(row_group, k_chunk, lane, step, k_size):
    row = row_group * 16 + lane % 16
    return (
        (row_group * (k_size // 64) + k_chunk * 2 + step // 2) * 128
        + (step % 2) * 64
        + (row % 16) * 4
        + lane // 16
    )


def atom_mxfp4_scale_index(row, col, cols):
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, e = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * ((cols + 7) // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a


def atom_mxfp4_scale_pair(row, col, cols):
    """``(i32 word, byte offset)`` holding the scales of ``col`` and ``col + 4``.

    ``atom_mxfp4_scale_index`` is ``word * 4 + d * 2 + a``, where ``d`` is the
    only term that moves when ``col`` advances by four and ``a`` is the only
    one below it.  So the scale bytes of two k-chunks four columns apart live
    in **one naturally aligned i32**, at byte offsets ``a`` and ``a + 2``, and
    a single dword load replaces two 1-byte gathers.  Caller must pass the
    lower of the two columns, i.e. one with ``(col // 4) % 2 == 0``.
    """
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, e = col // 8, col % 4
    return (((r32 * ((cols + 7) // 8) + c8) * 4 + e) * 16 + b, a)


def symmetric_allreduce_nbytes(sizes: tuple[int, ...], npes: int) -> int:
    """Return bytes for two tagged-mailbox epoch slots per reduce region."""
    if not sizes or any(size <= 0 or size % 2 for size in sizes):
        raise ValueError("all BF16 reduce sizes must be positive and even")
    if npes not in {2, 4, 8}:
        raise ValueError(f"npes must be one of {{2, 4, 8}}, got {npes}")
    max_pairs = max(sizes) // 2
    slot_bytes = npes * max_pairs * 8
    return len(sizes) * 2 * slot_bytes
