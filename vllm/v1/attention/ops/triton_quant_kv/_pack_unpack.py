# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sub-byte packing helpers shared by the INT4 KV cache backend.

Two helpers (pack / unpack) for the INT4-nibble format.  Shared by the
reshape (write) and attention (read) kernels to prevent drift between the
two sides of the cache.

Layout:

* **INT4**: two 4-bit values per uint8 — low nibble = even index, high
  nibble = odd index.
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
