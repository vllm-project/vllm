# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sub-byte packing helpers shared by the INT4 and INT2 KV cache backends.

Two helpers per layout (pack / unpack) for the INT4-nibble and INT2-quartet
formats.  Shared by the reshape (write) and attention (read) kernels to
prevent drift between the two sides of the cache.

Layouts:

* **INT4**: two 4-bit values per uint8 — low nibble = even index, high
  nibble = odd index.
* **INT2**: four 2-bit values per uint8 in little-endian quartet order
  (q0 in bits ``[0:2]``, q3 in ``[6:8]``).
"""

from __future__ import annotations

from vllm.triton_utils import triton


@triton.jit
def pack_int4_nibbles(lo, hi):
    """Pack two uint8 values (each in [0, 15]) into one byte."""
    return (lo & 0xF) | ((hi & 0xF) << 4)


@triton.jit
def unpack_int4_nibbles(packed):
    """Split one packed byte into the (low, high) nibble pair as uint8."""
    return packed & 0xF, (packed >> 4) & 0xF


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
