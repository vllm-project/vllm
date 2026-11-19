# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the logits kernels, beside the read and boundary kernels'
own (`..hc._ptx`, `..boundary._ptx`).

Memory accesses whose position matters are side-effecting assembly, so the
compiler keeps them where the kernel issues them. The token choice's keys are
built here instruction by instruction (`pack_key`), so that the order they
define is exactly the one stated in `choice.py`.
"""

from cutlass import Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..hc._ptx import Words4, _asm, _words

# A token key: [39:24] the ordered bits of the BF16 logit, [23:0] 0xFFFFFF
# minus the vocabulary index (ties go to the lower index).
INDEX_BITS = 24
INDEX_MASK = (1 << INDEX_BITS) - 1


def _key_block(word: str, half: int, offset: str) -> str:
    """PTX of one BF16 value's chunk key (`pack_key`): `word`'s `half`
    (0: lower) at chunk position `offset` (a register), folded into
    `best`."""
    shift = "0" if half == 0 else "16"
    return (
        f"shr.b32 v, {word}, {shift};\n\tand.b32 v, v, 0xFFFF;\n\t"
        # NaN orders as -inf (greedy_argmax treats NaN as -inf).
        "and.b32 t, v, 0x7FFF;\n\tsetp.gt.u32 p, t, 0x7F80;\n\t"
        "@p mov.b32 v, 0xFF80;\n\t"
        # -0 orders as +0 (they compare equal).
        "setp.eq.u32 p, v, 0x8000;\n\t@p mov.b32 v, 0;\n\t"
        # Order-preserving bits: flip all of a negative value, else the sign.
        "and.b32 t, v, 0x8000;\n\tsetp.ne.u32 p, t, 0;\n\t"
        "xor.b32 t, v, 0xFFFF;\n\tor.b32 v, v, 0x8000;\n\tselp.b32 v, t, v, p;\n\t"
        # Key: the bits, then 0xFFFF minus the chunk position.
        f"mov.b32 t, 0xFFFF;\n\tsub.u32 t, t, {offset};\n\t"
        "shl.b32 v, v, 16;\n\tor.b32 v, v, t;\n\t"
        # Only positions below the chunk's limit count.
        f"setp.lt.u32 p, {offset}, $6;\n\t@!p mov.b32 v, 0;\n\t"
        "max.u32 best, best, v;\n\t"
        f"add.u32 {offset}, {offset}, 1;\n\t"
    )


_PACK_KEY = (
    "{\n\t.reg .b32 v, t, best, o;\n\t.reg .pred p;\n\t"
    "mov.b32 best, 0;\n\tmov.b32 o, $5;\n\t"
    + "".join(
        _key_block(word, half, "o")
        for word in ("$1", "$2", "$3", "$4")
        for half in (0, 1)
    )
    + "mov.b32 $0, best;\n\t}"
)


@dsl_user_op
def pack_key(
    words: Words4, offset: Uint32, limit: Uint32, *, loc=None, ip=None
) -> Uint32:
    """The largest chunk key of a 16-byte pack of eight BF16 logits at chunk
    positions `offset .. offset + 7`: `(ordered bits << 16) | (0xFFFF -
    position)` over the positions below `limit`, or 0 if none is."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(w).ir_value(loc=loc, ip=ip) for w in words]
            + [
                Uint32(offset).ir_value(loc=loc, ip=ip),
                Uint32(limit).ir_value(loc=loc, ip=ip),
            ],
            _PACK_KEY,
            "=r,r,r,r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


def _full_word_block(word: str, index: int) -> str:
    """PTX of the chunk keys of one 32-bit word (two BF16 values, the
    pack's values `2 * index` and `2 * index + 1`) for `pack_key_full`,
    folded into `best`: two values an instruction where the order allows."""
    low, high = 2 * index, 2 * index + 1
    return (
        # -0 + +0 is +0 (round to nearest), every other value is unchanged;
        # max with -inf turns NaN into -inf (max returns the non-NaN input).
        f"add.rn.bf16x2 c, {word}, z;\n\tmax.bf16x2 c, c, ni;\n\t"
        # Order-preserving bits of both halves: flip all of a negative half,
        # else its sign bit.
        "shr.b32 s, c, 15;\n\tand.b32 s, s, 0x00010001;\n\t"
        "mul.lo.u32 s, s, 0x7FFF;\n\tor.b32 s, s, 0x80008000;\n\txor.b32 c, c, s;\n\t"
        # Keys: the bits, then 0xFFFF minus the chunk position.
        f"shl.b32 k, c, 16;\n\tadd.u32 k, k, base;\n\tsub.u32 k, k, {low};\n\t"
        "max.u32 best, best, k;\n\t"
        f"and.b32 k, c, 0xFFFF0000;\n\tadd.u32 k, k, base;\n\tsub.u32 k, k, {high};\n\t"
        "max.u32 best, best, k;\n\t"
    )


_PACK_KEY_FULL = (
    "{\n\t.reg .b32 c, s, k, best, base, z, ni;\n\t"
    "mov.b32 best, 0;\n\tmov.b32 z, 0;\n\tmov.b32 ni, 0xFF80FF80;\n\t"
    "mov.b32 base, 0xFFFF;\n\tsub.u32 base, base, $5;\n\t"
    + "".join(
        _full_word_block(word, index)
        for index, word in enumerate(("$1", "$2", "$3", "$4"))
    )
    + "mov.b32 $0, best;\n\t}"
)


@dsl_user_op
def pack_key_full(words: Words4, offset: Uint32, *, loc=None, ip=None) -> Uint32:
    """`pack_key` for a pack whose eight positions all count (a chunk
    wholly inside the vocabulary): the same keys, from packed arithmetic on
    two values at a time instead of one value at a time."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(w).ir_value(loc=loc, ip=ip) for w in words]
            + [Uint32(offset).ir_value(loc=loc, ip=ip)],
            _PACK_KEY_FULL,
            "=r,r,r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """The larger of two unsigned words."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.u32 $0, $1, $2;",
            "=r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def min_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """The smaller of two unsigned words."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "min.u32 $0, $1, $2;",
            "=r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_u64(a: Int64, b: Int64, *, loc=None, ip=None) -> Int64:
    """The larger of two unsigned 64-bit words."""
    return Int64(
        _asm(
            T.i64(),
            [Int64(a).ir_value(loc=loc, ip=ip), Int64(b).ir_value(loc=loc, ip=ip)],
            "max.u64 $0, $1, $2;",
            "=l,l,l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def warp_max_u32(value: Uint32, *, loc=None, ip=None) -> Uint32:
    """The largest `value` of the warp, in every lane."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atomic_max_u64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    """`max` into a GPU-scope word (relaxed)."""
    _asm(
        T.i64(),
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int64(value).ir_value(loc=loc, ip=ip),
        ],
        "atom.relaxed.gpu.global.max.u64 $0, [$1], $2;",
        "=l,l,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def arrive_acq_rel_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    """Add one to a GPU-scope counter with acquire-release semantics: this
    thread's earlier accesses happen before the add, and its later ones see
    what threads that added before it did before their add. Returns the
    count before."""
    return Uint32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], 1;",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def exchange_u64(address: Int64, value: Int64, *, loc=None, ip=None) -> Int64:
    """Swap `value` into a GPU-scope word (relaxed); returns the old one."""
    return Int64(
        _asm(
            T.i64(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Int64(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.relaxed.gpu.global.exch.b64 $0, [$1], $2;",
            "=l,l,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_u64(address: Int64, *, loc=None, ip=None) -> Int64:
    """One 64-bit word an earlier kernel (or this thread) wrote."""
    return Int64(
        _asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.u64 $0, [$1];",
            "=l,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_u64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    """Store one 64-bit word."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u64 [$0], $1;",
        "l,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_record(address: Int64, *, loc=None, ip=None) -> Int64:
    """One exchange record of this rank's mailbox, re-read on every call
    (peers store it over NVLink)."""
    return Int64(
        _asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.sys.global.u64 $0, [$1];",
            "=l,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_record(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    """Store one exchange record into a peer's mailbox: one 8-byte store, so
    its key and its tag land together."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.relaxed.sys.global.u64 [$0], $1;",
        "l,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_x4_peer(address: Int64, words: Words4, *, loc=None, ip=None) -> None:
    """Store 16 bytes into a peer's mailbox over NVLink (or a local one)."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in words],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_volatile_words(address: Int64, *, loc=None, ip=None) -> Words4:
    """16 bytes of a Lamport mailbox slot, re-read on every call."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.volatile.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, loc, ip)
