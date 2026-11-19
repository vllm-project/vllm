# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the tensor-core (tcgen05) read kernels: mbarriers, tensor
memory, the UMMA instruction and its descriptors, and stores into a cluster
peer's shared memory that complete on the peer's mbarrier.

Shared-memory addresses are 32-bit integers, tensor-memory addresses 32-bit
integers (lane in the upper 16 bits, column in the lower 16). Every
instruction whose position matters is side-effecting assembly.

Operand layout (`sw128_desc`): an operand tile of R rows by 64 BF16 values
(one 128-byte swizzle atom along K) is R rows of 128 bytes, the 16-byte chunk
`c` of row `r` stored at chunk `c ^ (r % 8)`: what a TMA load with
128-byte swizzle writes into a 1024-byte aligned buffer, and the K-major
128-byte-swizzled canonical layout of the UMMA descriptors (8-row groups 1024
bytes apart). A 16-value K step inside the atom advances the start address by
32 bytes.
"""

from cutlass import Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# Descriptor bits above the start address: stride between 8-row groups 1024
# bytes (bits 32-45, in 16-byte units), descriptor version 1 (bit 46), layout
# SWIZZLE_128B (2 in bits 61-63). The leading byte offset is unused for
# swizzled K-major operands.
_SW128_DESC = (1024 >> 4) << 32 | 1 << 46 | 2 << 61


def mma_idesc(m: int, n: int) -> int:
    """The instruction descriptor of a BF16 x BF16 -> FP32 UMMA of `m` x
    `n`, both operands K-major."""
    assert m in (64, 128) and n % 16 == 0 and 16 <= n <= 256
    return 1 << 4 | 1 << 7 | 1 << 10 | (n >> 3) << 17 | (m >> 4) << 24


def _asm(
    result: ir.Type | None,
    operands: list[ir.Value],
    text: str,
    constraints: str,
    *,
    loc: ir.Location | None,
    ip: ir.InsertionPoint | None,
) -> ir.Value:
    """One side-effecting PTX inline assembly operation."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _values(struct: ir.Value, count: int, loc, ip) -> list[Float32]:
    """The FP32 members of an inline-assembly result struct."""
    return [
        Float32(llvm.extractvalue(T.f32(), struct, [index], loc=loc, ip=ip))
        for index in range(count)
    ]


@dsl_user_op
def sw128_desc(address: Int32, *, loc=None, ip=None) -> Int64:
    """The UMMA shared-memory descriptor of a K-major 128-byte-swizzled
    operand starting at `address` (module docstring)."""
    return Int64(_SW128_DESC) | (Int64(address) >> 4)


@dsl_user_op
def mbarrier_init(barrier: Int32, count: int, *, loc=None, ip=None) -> None:
    """Initialize the mbarrier at `barrier` for `count` arrivals."""
    _asm(
        None,
        [Int32(barrier).ir_value(loc=loc, ip=ip)],
        f"mbarrier.init.shared::cta.b64 [$0], {count};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_mbarrier_init(*, loc=None, ip=None) -> None:
    """Make this thread's mbarrier initializations visible to the cluster
    (and the asynchronous proxy)."""
    _asm(None, [], "fence.mbarrier_init.release.cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def mbarrier_arrive_expect(barrier: Int32, size: Int32, *, loc=None, ip=None) -> None:
    """Arrive on `barrier` and expect `size` transaction bytes."""
    _asm(
        None,
        [Int32(barrier).ir_value(loc=loc, ip=ip), Int32(size).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_wait(barrier: Int32, phase: Int32, *, loc=None, ip=None) -> None:
    """Wait until phase `phase` (parity) of `barrier` completed (acquire
    at CTA scope: TMA loads and tensor-core commits of this CTA)."""
    _asm(
        None,
        [
            Int32(barrier).ir_value(loc=loc, ip=ip),
            Int32(phase).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tWAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n\t"
        "@!p bra WAIT;\n\t}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_wait_cluster(barrier: Int32, phase: Int32, *, loc=None, ip=None) -> None:
    """`mbarrier_wait` with acquire at cluster scope: for bytes that cluster
    peers stored with `store_async_peer`."""
    _asm(
        None,
        [
            Int32(barrier).ir_value(loc=loc, ip=ip),
            Int32(phase).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tWAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$0], $1;\n\t"
        "@!p bra WAIT;\n\t}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_alloc(slot: Int32, columns: int, *, loc=None, ip=None) -> None:
    """Allocate `columns` tensor-memory columns (warp-wide) and write their
    address to shared memory at `slot`; then give up the right to allocate,
    so that CTAs of later kernels on this SM can allocate."""
    _asm(
        None,
        [Int32(slot).ir_value(loc=loc, ip=ip)],
        f"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], {columns};\n\t"
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_dealloc(address: Int32, columns: int, *, loc=None, ip=None) -> None:
    """Free the `columns` tensor-memory columns at `address` (warp-wide)."""
    _asm(
        None,
        [Int32(address).ir_value(loc=loc, ip=ip)],
        f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, {columns};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_before_sync(*, loc=None, ip=None) -> None:
    """Order this thread's tensor-core operations before a thread barrier."""
    _asm(None, [], "tcgen05.fence::before_thread_sync;", "", loc=loc, ip=ip)


@dsl_user_op
def fence_after_sync(*, loc=None, ip=None) -> None:
    """Order this thread's tensor-core operations after a thread barrier or
    an mbarrier wait."""
    _asm(None, [], "tcgen05.fence::after_thread_sync;", "", loc=loc, ip=ip)


@dsl_user_op
def mma(
    accumulator: Int32,
    a_desc: Int64,
    b_desc: Int64,
    idesc: Int32,
    accumulate: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """`D = A B^T (+ D if accumulate)` on the tensor cores: D in tensor
    memory at `accumulator`, A and B in shared memory (descriptors). One
    thread issues it."""
    _asm(
        None,
        [
            Int32(accumulator).ir_value(loc=loc, ip=ip),
            Int64(a_desc).ir_value(loc=loc, ip=ip),
            Int64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Int32(accumulate).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, p;\n\t}",
        "r,l,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_commit(barrier: Int32, *, loc=None, ip=None) -> None:
    """Arrive on `barrier` once every tensor-core operation this thread
    issued so far completed."""
    _asm(
        None,
        [Int32(barrier).ir_value(loc=loc, ip=ip)],
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_load(address: Int32, count: int, *, loc=None, ip=None) -> list[Float32]:
    """`count` consecutive FP32 columns of this thread's tensor-memory lane
    (warp-wide `32x32b` load: thread `i` of the warp reads lane
    `(address >> 16) + i`). Wait with `tmem_wait_load` before use."""
    outputs = ", ".join(f"${i}" for i in range(count))
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * count),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        f"tcgen05.ld.sync.aligned.32x32b.x{count}.b32 {{{outputs}}}, [${count}];",
        ",".join(["=f"] * count) + ",r",
        loc=loc,
        ip=ip,
    )
    return _values(loaded, count, loc, ip)


@dsl_user_op
def tmem_wait_load(*, loc=None, ip=None) -> None:
    """Wait for this thread's tensor-memory loads."""
    _asm(None, [], "tcgen05.wait::ld.sync.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def store_async_peer(
    address: Int32,
    values: tuple[Float32, Float32, Float32, Float32],
    barrier: Int32,
    rank: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store four FP32 values at `address` in the shared memory of the
    cluster's CTA `rank`; the 16 bytes complete on that CTA's mbarrier at
    `barrier` (both addresses as in this CTA)."""
    _asm(
        None,
        [
            Int32(address).ir_value(loc=loc, ip=ip),
            *(Float32(v).ir_value(loc=loc, ip=ip) for v in values),
            Int32(rank).ir_value(loc=loc, ip=ip),
            Int32(barrier).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 a, b;\n\t"
        "mapa.shared::cluster.u32 a, $0, $5;\n\t"
        "mapa.shared::cluster.u32 b, $6, $5;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
        "[a], {$1, $2, $3, $4}, [b];\n\t}",
        "r,f,f,f,f,r,r",
        loc=loc,
        ip=ip,
    )
