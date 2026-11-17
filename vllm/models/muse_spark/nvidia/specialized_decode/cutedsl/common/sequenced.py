# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The protocol of the sequence-tagged mailboxes (`CandidateMailbox`,
`PacketMailbox`).

A writer multicasts 64-bit slots, each 32-bit word tagged with the row's
sequence number, into the generation of the number's parity; a reader polls
its own copy with 64-bit relaxed system-scope loads, the size of the stores,
until every word carries the number, then advances it. The numbers of `bits`
tag bits run 1 .. `sequences(bits)`: never the empty slots' 0, and an even
count, so the generation alternates across the wrap.
"""

import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, dsl_user_op

from .primitives import _stmc_words, _words_of, global_ptr, predicated


def sequences(bits: int) -> int:
    """How many sequence numbers `bits` tag bits hold."""
    return (1 << bits) - 2


def next_sequence(seq: Uint32, bits: int) -> Uint32:
    """The sequence number after `seq`, wrapping to 1."""
    return seq % Uint32(sequences(bits)) + Uint32(1)


def generation(seq: Uint32) -> Uint32:
    """The generation of the slots that sequence number `seq` uses."""
    return seq & Uint32(1)


@dsl_user_op
def stmc_b64(
    address: Int64,
    lo: Uint32,
    hi: Uint32,
    predicate: Int32 | None = None,
    *,
    loc=None,
    ip=None,
) -> None:
    """One 64-bit multicast store of `{lo, hi}` (if `predicate`); a reader
    may see either word before the other."""
    predicated(
        predicate,
        lambda ip: _stmc_words(address, [lo, hi], dtype=Int64, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


def _load_relaxed_sys_b64(address: Int64, count: int, *, loc, ip) -> tuple:
    """`count` (1 or 2) 64-bit relaxed system-scope loads in one access, as
    32-bit words in memory order."""
    loaded = cute.arch.load(
        global_ptr(Int64, Int64(address), 8 * count),
        Int64 if count == 1 else ir.VectorType.get([count], T.i64(), loc=loc),
        sem="relaxed",
        scope="sys",
        loc=loc,
        ip=ip,
    )
    if count == 1:
        loaded = loaded.ir_value(loc=loc, ip=ip)
    return _words_of(loaded, 2 * count, loc=loc, ip=ip)


@dsl_user_op
def load_relaxed_sys_b64(address: Int64, *, loc=None, ip=None) -> tuple:
    """One 64-bit relaxed system-scope load, as `(lo, hi)` words."""
    return _load_relaxed_sys_b64(address, 1, loc=loc, ip=ip)


@dsl_user_op
def load_relaxed_sys_b64x2(
    address: Int64, predicate: Int32, *, loc=None, ip=None
) -> tuple:
    """Two 64-bit relaxed system-scope loads in one vector access, as four
    32-bit words in memory order; zeros if not `predicate`."""
    return predicated(
        predicate,
        lambda ip: _load_relaxed_sys_b64(address, 2, loc=loc, ip=ip),
        (Uint32(0),) * 4,
        loc=loc,
        ip=ip,
    )
