# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the QSA decode select.

Memory accesses are side-effecting assembly, so the compiler keeps them where
the kernel issues them relative to the grid-dependency wait and the CTA and
cluster barriers that order them. Addresses are plain integers: 64-bit global
addresses, 32-bit shared-memory addresses of this CTA, and 32-bit
shared::cluster addresses of a CTA of the cluster (from `peer_address`).
"""

from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..hc._ptx import _asm

Words4 = tuple[Uint32, Uint32, Uint32, Uint32]


def _v(value, loc, ip):
    """IR value of a DSL scalar."""
    return value.ir_value(loc=loc, ip=ip)


@dsl_user_op
def load_global_x4(address: Int64, *, loc=None, ip=None) -> Words4:
    """16 bytes an earlier kernel wrote (a coherent load)."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(Int64(address), loc, ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return tuple(  # type: ignore[return-value]
        Uint32(llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def load_global_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    """A 32-bit integer an earlier kernel wrote."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int64(address), loc, ip)],
            "ld.global.s32 $0, [$1];",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_global_s64(address: Int64, *, loc=None, ip=None) -> Int64:
    """A 64-bit integer an earlier kernel wrote."""
    return Int64(
        _asm(
            T.i64(),
            [_v(Int64(address), loc, ip)],
            "ld.global.s64 $0, [$1];",
            "=l,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_global_x4(
    address: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None
) -> None:
    """Store four 32-bit integers at a 16-byte aligned global address."""
    _asm(
        None,
        [_v(Int64(address), loc, ip)]
        + [_v(Int32(w), loc, ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.s32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_s32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    """Store a 32-bit integer to global memory."""
    _asm(
        None,
        [_v(Int64(address), loc, ip), _v(Int32(value), loc, ip)],
        "st.global.s32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shared_load(address: Int32, *, loc=None, ip=None) -> Uint32:
    """Load a 32-bit word of this CTA's shared memory."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Int32(address), loc, ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shared_store(address: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    """Store a 32-bit word to this CTA's shared memory."""
    _asm(
        None,
        [_v(Int32(address), loc, ip), _v(Uint32(value), loc, ip)],
        "st.shared.u32 [$0], $1;",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shared_zero_x4(address: Int32, *, loc=None, ip=None) -> None:
    """Zero 16 bytes at a 16-byte aligned address of this CTA's shared memory."""
    _asm(
        None,
        [_v(Int32(address), loc, ip)],
        "st.shared.v4.u32 [$0], {0, 0, 0, 0};",
        "r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shared_red_add(address: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    """Atomically add `value` to a word of this CTA's shared memory."""
    _asm(
        None,
        [_v(Int32(address), loc, ip), _v(Uint32(value), loc, ip)],
        "red.shared.add.u32 [$0], $1;",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def peer_address(address: Int32, rank, *, loc=None, ip=None) -> Int32:
    """The shared::cluster address of this CTA's shared `address` in the
    cluster's CTA `rank`."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(address), loc, ip), _v(Int32(rank), loc, ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cluster_red_add(address: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    """Atomically add `value` to a shared::cluster word."""
    _asm(
        None,
        [_v(Int32(address), loc, ip), _v(Uint32(value), loc, ip)],
        "red.shared::cluster.add.u32 [$0], $1;",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cluster_store_x2(
    address: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None
) -> None:
    """Store two words at an 8-byte aligned shared::cluster address."""
    _asm(
        None,
        [_v(Int32(address), loc, ip), _v(Uint32(a), loc, ip), _v(Uint32(b), loc, ip)],
        "st.shared::cluster.v2.u32 [$0], {$1, $2};",
        "r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cluster_store_x4(
    address: Int32, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None
) -> None:
    """Store four words at a 16-byte aligned shared::cluster address."""
    _asm(
        None,
        [_v(Int32(address), loc, ip)] + [_v(Uint32(w), loc, ip) for w in (a, b, c, d)],
        "st.shared::cluster.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shfl_up(value: Uint32, delta: int, *, loc=None, ip=None) -> Uint32:
    """`value` of the lane `delta` below (own value in the lowest lanes)."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Uint32(value), loc, ip)],
            f"shfl.sync.up.b32 $0, $1, {delta}, 0, 0xffffffff;",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_idx(value: Uint32, lane, *, loc=None, ip=None) -> Uint32:
    """`value` of the full warp's lane `lane`."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Uint32(value), loc, ip), _v(Int32(lane), loc, ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;",
            "=r,r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def match_any(value: Uint32, *, loc=None, ip=None) -> Uint32:
    """The lanes of the full warp whose `value` equals this lane's."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Uint32(value), loc, ip)],
            "match.any.sync.b32 $0, $1, 0xffffffff;",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def lanes_below(*, loc=None, ip=None) -> Uint32:
    """The mask of the lanes below this one."""
    return Uint32(
        _asm(
            T.i32(),
            [],
            "mov.u32 $0, %lanemask_lt;",
            "=r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def popc(value: Uint32, *, loc=None, ip=None) -> Uint32:
    """The number of set bits of `value`."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Uint32(value), loc, ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cluster_atom_add(address: Int32, value: Uint32, *, loc=None, ip=None) -> Uint32:
    """Atomically add `value` to a shared::cluster word; its old value."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Int32(address), loc, ip), _v(Uint32(value), loc, ip)],
            "atom.shared::cluster.add.u32 $0, [$1], $2;",
            "=r,r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shared_load_x4(address: Int32, *, loc=None, ip=None) -> Words4:
    """Load 16 bytes at a 16-byte aligned address of this CTA's shared memory."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(Int32(address), loc, ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return tuple(  # type: ignore[return-value]
        Uint32(llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip))
        for index in range(4)
    )
