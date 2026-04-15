# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared @triton.jit helpers for packed sub-byte KV layouts.

Both the reshape (write) and attention (read) kernels for INT4 and INT2
per-token-head modes pack/unpack multiple low-bit values into a single
``uint8``.  Keeping the pack/unpack arithmetic in one place avoids drift
between the two kernels and keeps the per-mode files focused on the
quantization math rather than bit-twiddling.
"""

from __future__ import annotations

from vllm.triton_utils import triton

# ---------------------------------------------------------------------------
# INT4: two 4-bit values per uint8.  Even index in the low nibble, odd in
# the high nibble.  Used by INT4 per-token-head reshape + attention.
# ---------------------------------------------------------------------------


@triton.jit
def pack_int4_nibbles(lo, hi):
    """Pack two uint8 values (each in [0, 15]) into one byte."""
    return (lo & 0xF) | ((hi & 0xF) << 4)


@triton.jit
def unpack_int4_nibbles(packed):
    """Split one packed byte into the (low, high) nibble pair as uint8."""
    return packed & 0xF, (packed >> 4) & 0xF


# ---------------------------------------------------------------------------
# INT2: four 2-bit values per uint8.  Quartet (q0..q3) is laid out
# little-endian: q0 in bits [0:2], q1 in [2:4], q2 in [4:6], q3 in [6:8].
# Used by INT2 per-token-head reshape + attention.
# ---------------------------------------------------------------------------


@triton.jit
def pack_int2_quartet(q0, q1, q2, q3):
    """Pack four uint8 values (each in [0, 3]) into one byte."""
    return (q0 & 0x3) | ((q1 & 0x3) << 2) | ((q2 & 0x3) << 4) | ((q3 & 0x3) << 6)


@triton.jit
def unpack_int2_quartet(packed):
    """Split one packed byte into the (q0, q1, q2, q3) quartet as uint8."""
    return (
        packed & 0x3,
        (packed >> 2) & 0x3,
        (packed >> 4) & 0x3,
        (packed >> 6) & 0x3,
    )
