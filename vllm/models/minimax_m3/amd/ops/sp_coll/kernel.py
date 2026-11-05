# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL pull collectives (TP4, gfx950). ``fence_publish`` writes every XCD's L2
back (a coarse-grained buffer's dirty lines are not flushed for peers' xGMI reads at
a kernel boundary), then the last workgroup stores the sequence into the flag;
``pull`` is a small persistent grid: CU slots are scarce next to compute kernels."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

WORLD = 4
NT = 256
UNROLL = 8  # 16 B per thread per step
STEP = NT * 4 * UNROLL  # dwords per workgroup per loop step
SYS = 0x11  # sc0 | sc1: system-coherent loads / stores
# 8 workgroups per XCD (w runs on XCD w % 8, each with its own L2)
FENCE_WGS = 64
PER_PEER = 256  # chunks per rank block
# 4 workgroups per rank reach the link rate (50 MB: 0.87 vs RCCL 0.95 ms)
PULL_WGS = 16
SUM_WGS = 1024
# bytes per set: the flag word, then the fence ticket counter
FLAG_STRIDE = 16


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _flag_read(addr_i64):
    """System-scope atomic add of 0: never hoisted out of the spin loop."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(0).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


def _flag_add(addr_i64, v):
    """System-scope atomic add; returns the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(v).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


def _sel4(p, a0, a1, a2, a3):
    return (p == 0).select(a0, (p == 1).select(a1, (p == 2).select(a2, a3)))


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _pk_bf16(a, b):
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def compile_publish():
    """Store a value into the flag, system-coherent (the init's zeroing)."""

    @flyc.kernel(name="sp_coll_publish", known_block_size=[64, 1, 1])
    def kernel(arg_flag: fx.Int64, i32_seq: fx.Int32):
        fr = buffer_ops.create_buffer_resource_from_addr(arg_flag, num_records_bytes=64)
        buffer_ops.buffer_store(
            i32_seq, fr, 0, mask=fx.thread_idx.x == 0, cache_modifier=SYS
        )

    @flyc.jit
    def launch(arg_flag: fx.Int64, i32_seq: fx.Int32, stream: fx.Stream):
        kernel(arg_flag, i32_seq).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.kernel_name = "sp_coll_publish"
    return launch


def compile_fence_publish():
    """Every workgroup releases its XCD's L2 and takes a ticket; the last one
    resets the counter and stores ``seq`` into the flag."""

    @flyc.kernel(name="sp_coll_fence_publish", known_block_size=[64, 1, 1])
    def kernel(arg_flag: fx.Int64, i32_seq: fx.Int32):
        _llvm.FenceOp(_llvm.AtomicOrdering.release, syncscope="")
        if fx.thread_idx.x == 0:
            t = _flag_add(arg_flag + 4, 1)
            if t == FENCE_WGS - 1:
                _flag_add(arg_flag + 4, -FENCE_WGS)
                fr = buffer_ops.create_buffer_resource_from_addr(
                    arg_flag, num_records_bytes=64
                )
                buffer_ops.buffer_store(i32_seq, fr, 0, cache_modifier=SYS)

    @flyc.jit
    def launch(arg_flag: fx.Int64, i32_seq: fx.Int32, stream: fx.Stream):
        kernel(arg_flag, i32_seq).launch(
            grid=(FENCE_WGS, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.kernel_name = "sp_coll_fence_publish"
    return launch


def compile_pull():
    """dst[p * block, (p + 1) * block) <- rank p's bytes [src_off, src_off + block).
    Workgroup w serves rank w % 4: waits for its flag to reach ``seq``, then copies
    chunks w // 4, w // 4 + PULL_WGS // 4, ..."""

    @flyc.kernel(name="sp_coll_pull", known_block_size=[NT, 1, 1])
    def kernel(
        arg_dst: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        arg_f0: fx.Int64,
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        i64_block: fx.Int64,  # bytes per rank block (multiple of 16)
        i64_src_off: fx.Int64,  # byte offset of the block inside a rank's region
        i32_seq: fx.Int32,
        i32_me: fx.Int32,
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        p = bid % WORLD
        if (tx == 0) & (p != i32_me):
            fa = _sel4(p, arg_f0, arg_f1, arg_f2, arg_f3)
            f = _flag_read(fa)
            while f < i32_seq:
                f = _flag_read(fa)
        fx.gpu.barrier()
        _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")
        src_a = _sel4(p, arg_s0, arg_s1, arg_s2, arg_s3) + i64_src_off
        src = buffer_ops.create_buffer_resource_from_addr(
            src_a, num_records_bytes=i64_block
        )
        dst = buffer_ops.create_buffer_resource_from_addr(
            arg_dst + fx.Int64(p) * i64_block, num_records_bytes=i64_block
        )
        n_dw = fx.Int32(i64_block // 4)
        per_block = (n_dw + (PER_PEER - 1)) // PER_PEER
        per_block = (per_block + (STEP - 1)) // STEP * STEP
        pb = bid // WORLD
        while pb < PER_PEER:
            start = pb * per_block
            end = start + per_block
            off = start + tx * 4
            while off < end:
                vals = []
                for u in range_constexpr(UNROLL):
                    o = off + u * NT * 4
                    vals.append(
                        Vec(
                            buffer_ops.buffer_load(
                                src,
                                o,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=o < n_dw,
                                cache_modifier=SYS,
                            )
                        )
                    )
                for u in range_constexpr(UNROLL):
                    o = off + u * NT * 4
                    buffer_ops.buffer_store(vals[u], dst, o, mask=o < n_dw)
                off = off + STEP
            pb = pb + (PULL_WGS // WORLD)

    @flyc.jit
    def launch(
        arg_dst: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        arg_f0: fx.Int64,
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        i64_block: fx.Int64,
        i64_src_off: fx.Int64,
        i32_seq: fx.Int32,
        i32_me: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_dst,
            arg_s0,
            arg_s1,
            arg_s2,
            arg_s3,
            arg_f0,
            arg_f1,
            arg_f2,
            arg_f3,
            i64_block,
            i64_src_off,
            i32_seq,
            i32_me,
        ).launch(grid=(PULL_WGS, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = "sp_coll_pull"
    return launch


def compile_pull_unpack():
    """``pull`` that also unpacks (``sp_pack.mxfp8_unpack``): rank p's rows of ``d``
    codes + ``d / 32`` scales land in codes[:, p * d :] and scales[:, p * d / 32 :].
    A 16 B unit never straddles codes and scales since d % 16 == 0."""

    @flyc.kernel(name="sp_coll_pull_unpack", known_block_size=[NT, 1, 1])
    def kernel(
        arg_codes: fx.Int64,
        arg_scales: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        arg_f0: fx.Int64,
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        i64_block: fx.Int64,  # bytes per rank block = rows * row_bytes
        i64_src_off: fx.Int64,
        i32_seq: fx.Int32,
        i32_me: fx.Int32,
        i32_rows: fx.Int32,
        i32_d: fx.Int32,  # codes per row (bytes); scales per row = d / 32
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        p = bid % WORLD
        if (tx == 0) & (p != i32_me):
            fa = _sel4(p, arg_f0, arg_f1, arg_f2, arg_f3)
            f = _flag_read(fa)
            while f < i32_seq:
                f = _flag_read(fa)
        fx.gpu.barrier()
        _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")
        src_a = _sel4(p, arg_s0, arg_s1, arg_s2, arg_s3) + i64_src_off
        src = buffer_ops.create_buffer_resource_from_addr(
            src_a, num_records_bytes=i64_block
        )
        row_bytes = i32_d + i32_d // 32
        codes = buffer_ops.create_buffer_resource_from_addr(
            arg_codes, num_records_bytes=fx.Int64(i32_rows) * fx.Int64(i32_d * WORLD)
        )
        scales = buffer_ops.create_buffer_resource_from_addr(
            arg_scales,
            num_records_bytes=fx.Int64(i32_rows) * fx.Int64((i32_d // 32) * WORLD),
        )
        n_dw = fx.Int32(i64_block // 4)
        per_block = (n_dw + (PER_PEER - 1)) // PER_PEER
        per_block = (per_block + (STEP - 1)) // STEP * STEP
        pb = bid // WORLD
        while pb < PER_PEER:
            start = pb * per_block
            end = start + per_block
            off = start + tx * 4
            while off < end:
                vals = []
                for u in range_constexpr(UNROLL):
                    o = off + u * NT * 4
                    vals.append(
                        Vec(
                            buffer_ops.buffer_load(
                                src,
                                o,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=o < n_dw,
                                cache_modifier=SYS,
                            )
                        )
                    )
                for u in range_constexpr(UNROLL):
                    o = off + u * NT * 4
                    byte = o * 4
                    row = byte // row_bytes
                    b = byte - row * row_bytes
                    is_code = b < i32_d
                    off_c = (row * (i32_d * WORLD) + p * i32_d + b) // 4
                    off_s = (
                        row * ((i32_d // 32) * WORLD) + p * (i32_d // 32) + (b - i32_d)
                    ) // 4
                    ok = o < n_dw
                    buffer_ops.buffer_store(vals[u], codes, off_c, mask=ok & is_code)
                    buffer_ops.buffer_store(vals[u], scales, off_s, mask=ok & ~is_code)
                off = off + STEP
            pb = pb + (PULL_WGS // WORLD)

    @flyc.jit
    def launch(
        arg_codes: fx.Int64,
        arg_scales: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        arg_f0: fx.Int64,
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        i64_block: fx.Int64,
        i64_src_off: fx.Int64,
        i32_seq: fx.Int32,
        i32_me: fx.Int32,
        i32_rows: fx.Int32,
        i32_d: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_codes,
            arg_scales,
            arg_s0,
            arg_s1,
            arg_s2,
            arg_s3,
            arg_f0,
            arg_f1,
            arg_f2,
            arg_f3,
            i64_block,
            i64_src_off,
            i32_seq,
            i32_me,
            i32_rows,
            i32_d,
        ).launch(grid=(PULL_WGS, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = "sp_coll_pull_unpack"
    return launch


def compile_sum4():
    """Sum the four staged bf16 blocks in fp32, in rank order 0..3."""

    @flyc.kernel(name="sp_coll_sum4", known_block_size=[NT, 1, 1])
    def kernel(
        arg_out: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        i64_block: fx.Int64,  # bytes per block (bf16, multiple of 16)
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        srcs = [
            buffer_ops.create_buffer_resource_from_addr(a, num_records_bytes=i64_block)
            for a in (arg_s0, arg_s1, arg_s2, arg_s3)
        ]
        out = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=i64_block
        )
        n_dw = fx.Int32(i64_block // 4)
        per_block = (n_dw + (SUM_WGS - 1)) // SUM_WGS
        per_block = (per_block + (STEP - 1)) // STEP * STEP
        start = bid * per_block
        end = start + per_block
        off = start + tx * 4
        while off < end:
            words = []
            for u in range_constexpr(UNROLL):
                o = off + u * NT * 4
                words.append(
                    [
                        Vec(
                            buffer_ops.buffer_load(
                                srcs[r],
                                o,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=o < n_dw,
                            )
                        )
                        for r in range_constexpr(WORLD)
                    ]
                )
            for u in range_constexpr(UNROLL):
                o = off + u * NT * 4
                packed = []
                for i in range_constexpr(4):
                    lo = fx.Float32(0.0)
                    hi = fx.Float32(0.0)
                    for r in range_constexpr(WORLD):
                        w = fx.Int32(words[u][r][i])
                        lo = lo + _as_f32(w << 16)
                        hi = hi + _as_f32(w & fx.Int32(-65536))
                    packed.append(_pk_bf16(lo, hi))
                buffer_ops.buffer_store(
                    Vec.from_elements(packed, fx.Int32), out, o, mask=o < n_dw
                )
            off = off + STEP

    @flyc.jit
    def launch(
        arg_out: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        i64_block: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(arg_out, arg_s0, arg_s1, arg_s2, arg_s3, i64_block).launch(
            grid=(SUM_WGS, 1, 1), block=(NT, 1, 1), stream=stream
        )

    launch.kernel_name = "sp_coll_sum4"
    return launch
