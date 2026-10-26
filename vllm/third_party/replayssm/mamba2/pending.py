# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accepted-prefix ReplaySSM for SM121, using register-based warp MMA.

Group-level B*C products are shared by all heads in each group. The update
kernel commits accepted history before reading the quantized checkpoint.
Only the commit path reads FP16 recurrence state. BF16 MMA operands and
candidate readout accumulate in FP32.
"""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint8, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.third_party.replayssm.common import make_fake_tensor
from vllm.third_party.replayssm.common.cvt_fp4 import (
    e4m3_byte_to_f32,
    f32_to_e4m3_byte,
)
from vllm.third_party.replayssm.common.philox import philox4x32


@cute.jit
def _maximum(a, b):
    result = a
    if b > a:
        result = b
    return result


@dsl_user_op
def _atomic_max_positive(address, value, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 bits; mov.b32 bits,$2; atom.global.max.u32 $0,[$1],bits; }",
        "=r,l,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _pack_bf16(a, b, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.i32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _mma(a0, a1, a2, a3, b0, b1, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)],
        "{ .reg .f32 z; mov.f32 z, 0f00000000; "
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {z,z,z,z}; }",
        "=f,=f,=f,=f,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def _sr_half(value, random, *, loc=None, ip=None):
    # Choose adjacent finite FP16 values using 13 Philox bits. Unlike a
    # mantissa-bit addition, this also handles the FP16 subnormal interval.
    result = llvm.inline_asm(
        T.f16(),
        [
            Float32(value).ir_value(loc=loc, ip=ip),
            Uint32(random).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .f32 x,lo,hi,p,u; .reg .b16 h,h1; .reg .b32 r,s; .reg .pred q; "
        "abs.f32 x,$1; min.f32 x,x,0f477FE000; "
        "cvt.rz.f16.f32 h,x; cvt.f32.f16 lo,h; "
        "add.u16 h1,h,1; cvt.f32.f16 hi,h1; "
        "sub.f32 p,x,lo; sub.f32 hi,hi,lo; div.rn.f32 p,p,hi; "
        "and.b32 r,$2,8191; cvt.rn.f32.u32 u,r; mul.f32 u,u,0f39000000; "
        "setp.gt.f32 q,p,u; selp.b16 h,h1,h,q; "
        "mov.b32 s,$1; shr.u32 s,s,16; and.b32 s,s,32768; "
        "cvt.u16.u32 h1,s; or.b16 $0,h,h1; }",
        "=h,f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float16(result)


@dsl_user_op
def _store_e4m3_pair(address, a, b, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 q; cvt.rn.satfinite.e4m3x2.f32 q,$2,$1; st.global.b16 [$0],q; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _e2m1_pair(a, b, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.i32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q,$2,$1; cvt.u32.u8 $0,q; }",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _unpack_e2m1(packed, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        "{ .reg .b8 q; .reg .b32 p; .reg .b16 lo,hi; "
        "cvt.u8.u32 q,$2; cvt.rn.f16x2.e2m1x2 p,q; mov.b32 {lo,hi},p; "
        "cvt.f32.f16 $0,lo; cvt.f32.f16 $1,hi; }",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@cute.jit
def _scale_offset(head, row, block):
    r = (head % 2) * 64 + row
    return head // 2 * 1024 + block // 4 * 512 + r % 32 * 16 + r // 32 * 4 + block % 4


class PendingKernel:
    def __init__(self, heads, groups, length, max_tokens, nvfp4):
        self.heads = heads
        self.groups = groups
        self.length = length
        self.max_tokens = max_tokens
        self.nvfp4 = nvfp4

    @cute.jit
    def __call__(
        self,
        state,
        read,
        scale,
        meta,
        parity,
        xc,
        dc,
        bc,
        x,
        dt,
        a,
        b,
        c,
        d,
        bias,
        ids,
        cu,
        control,
        cb_products,
        out,
        seed,
        stream: CUstream,
    ):
        self.products_kernel(bc, b, c, ids, cu, control, cb_products).launch(
            grid=(ids.shape[0], self.groups, 24), block=(32, 1, 1), stream=stream
        )
        self.update_kernel(
            state,
            parity,
            seed,
            read,
            scale,
            meta,
            xc,
            dc,
            bc,
            x,
            dt,
            a,
            b,
            c,
            d,
            bias,
            ids,
            cu,
            control,
            cb_products,
            out,
        ).launch(grid=(ids.shape[0], self.heads, 1), block=(128, 1, 1), stream=stream)

    @cute.jit
    def _commit_state(
        self, state, read, scale, meta, parity, xc, dc, bc, a, ids, control, seed
    ):
        batch, head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        slot = ids[batch]
        if slot > 0 and control[batch, 2] != 0:
            warp = tid // 32
            lane = tid % 32
            group = head // (self.heads // self.groups)
            old = Int32(control[batch, 0])
            smem = cutlass.utils.SmemAllocator()
            weights = smem.allocate_array(Float32, 16)
            total = smem.allocate_array(Float32, 2)
            if tid == 0:
                accum = Float32(0.0)
                for i in cutlass.range(16, unroll=1):
                    ri = i if i < old else 16 + i - old
                    accum += dc[slot, head, ri] * Float32(a[head])
                    weights[i] = accum
                total[0] = cute.math.exp(accum, fastmath=True)
                total[1] = accum
            cute.arch.sync_threads()
            if tid < 16:
                ri = tid if tid < old else 16 + tid - old
                weights[tid] = dc[slot, head, ri] * cute.math.exp(
                    total[1] - weights[tid], fastmath=True
                )
            cute.arch.sync_threads()
            row = warp * 16 + lane // 4
            k = lane % 4 * 2
            ar = cute.make_rmem_tensor(4, Uint32)
            for j in cutlass.range_constexpr(4):
                rr = row + j % 2 * 8
                kk = k + j // 2 * 8
                i0 = kk if kk < old else 16 + kk - old
                i1 = kk + 1 if kk + 1 < old else 17 + kk - old
                ar[j] = _pack_bf16(xc[slot, head, i0, rr], xc[slot, head, i1, rr])
            values = cute.make_rmem_tensor((16, 4), Float32)
            maximum = cute.make_rmem_tensor(2, Float32)
            maximum.fill(0.0)
            for tile in cutlass.range_constexpr(16):
                col = tile * 8 + lane % 4 * 2
                kk = lane % 4 * 2
                br = cute.make_rmem_tensor(2, Uint32)
                for j in cutlass.range_constexpr(2):
                    k0 = kk + j * 8
                    i0 = k0 if k0 < old else 16 + k0 - old
                    i1 = k0 + 1 if k0 + 1 < old else 17 + k0 - old
                    cc = tile * 8 + lane // 4
                    br[j] = _pack_bf16(
                        Float32(bc[slot, group, i0, cc]) * weights[k0],
                        Float32(bc[slot, group, i1, cc]) * weights[k0 + 1],
                    )
                result = _mma(ar[0], ar[1], ar[2], ar[3], br[0], br[1])
                for j in cutlass.range_constexpr(4):
                    rr = row + j // 2 * 8
                    v = (
                        Float32(state[slot, head, rr, col + j % 2]) * total[0]
                        + result[j]
                    )
                    values[tile, j] = v
                    maximum[j // 2] = _maximum(maximum[j // 2], cute.math.absf(v))
            if cutlass.const_expr(self.nvfp4):
                global_scale = meta[slot, 2, group]
                thread_max = _maximum(maximum[0], maximum[1])
                warp_max = cute.arch.warp_reduction_max(thread_max)
                if lane == 0:
                    offset = cute.crd2idx(
                        (slot, Int32(parity[slot, group]), group), meta.layout
                    )
                    _atomic_max_positive((meta.iterator + offset).toint(), warp_max)
            for r in cutlass.range_constexpr(2):
                row_max = cute.arch.warp_reduction_max(maximum[r], threads_in_group=4)
                row_scale = row_max / Float32(448.0)
                if row_scale <= 0.0:
                    row_scale = Float32(1.0)
                if cutlass.const_expr(not self.nvfp4):
                    if lane % 4 == 0:
                        scale[slot, head * 64 + row + r * 8] = row_scale
                for block in cutlass.range_constexpr(8):
                    block_scale = Float32(1.0)
                    if cutlass.const_expr(self.nvfp4):
                        maximum_block = Float32(0.0)
                        for j in cutlass.range_constexpr(2):
                            for q in cutlass.range_constexpr(2):
                                maximum_block = _maximum(
                                    maximum_block,
                                    cute.math.absf(values[block * 2 + j, r * 2 + q]),
                                )
                        maximum_block = cute.arch.warp_reduction_max(
                            maximum_block, threads_in_group=4
                        )
                        sf = f32_to_e4m3_byte(
                            maximum_block / (Float32(6.0) * global_scale)
                        )
                        block_scale = e4m3_byte_to_f32(sf)
                        if lane % 4 == 0:
                            scale[slot, _scale_offset(head, row + r * 8, block)] = (
                                Uint8(sf)
                            )
                    for j in cutlass.range_constexpr(2):
                        tile = block * 2 + j
                        col = tile * 8 + lane % 4 * 2
                        v0 = values[tile, r * 2]
                        v1 = values[tile, r * 2 + 1]
                        event = control[batch, 4] // 16
                        rand = philox4x32(
                            Uint32(slot),
                            Uint32((head * 64 + row + r * 8) * 64 + col // 2),
                            Uint32(event),
                            Uint32(event >> 32),
                            seed,
                            Uint32(0),
                            7,
                        )
                        state[slot, head, row + r * 8, col] = _sr_half(v0, rand[0])
                        state[slot, head, row + r * 8, col + 1] = _sr_half(v1, rand[1])
                        if cutlass.const_expr(self.nvfp4):
                            inverse = Float32(1.0)
                            if block_scale > 0.0:
                                inverse = Float32(1.0) / block_scale
                            inverse /= global_scale
                            read[slot, (head * 64 + row + r * 8) * 64 + col // 2] = (
                                Uint8(_e2m1_pair(v0 * inverse, v1 * inverse))
                            )
                        else:
                            offset = cute.crd2idx(
                                (slot, (head * 64 + row + r * 8) * 128 + col),
                                read.layout,
                            )
                            _store_e4m3_pair(
                                (read.iterator + offset).toint(),
                                v0 / row_scale,
                                v1 / row_scale,
                            )

    @cute.kernel
    def products_kernel(self, bc, b, c, ids, cu, control, cb_products):
        batch, group, i = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        slot = ids[batch]
        start = cu[batch]
        tokens = cu[batch + 1] - start
        remain = Int32(control[batch, 3])
        if slot > 0 and i < remain + tokens:
            old = Int32(control[batch, 0])
            crossed = control[batch, 2] != 0
            bv = cute.make_rmem_tensor(4, Float32)
            for n in cutlass.range_constexpr(4):
                col = lane + n * 32
                if i < remain:
                    ri = i if i < old else 16 + i - old
                    if crossed:
                        ri = 32 - old + i
                    bv[n] = Float32(bc[slot, group, ri, col])
                else:
                    bv[n] = Float32(b[start + i - remain, group, col])
            for t in cutlass.range(tokens, unroll=1):
                if i <= remain + t:
                    acc = Float32(0.0)
                    for n in cutlass.range_constexpr(4):
                        acc += bv[n] * Float32(c[start + t, group, lane + n * 32])
                    acc = cute.arch.warp_reduction_sum(acc)
                    if lane == 0:
                        cb_products[batch, group, i, t] = acc

    @cute.kernel
    def update_kernel(
        self,
        state,
        parity,
        seed,
        read,
        scale,
        meta,
        xc,
        dc,
        bc,
        x,
        dt,
        a,
        b,
        c,
        d,
        bias,
        ids,
        cu,
        control,
        cb_products,
        out,
    ):
        self._commit_state(
            state, read, scale, meta, parity, xc, dc, bc, a, ids, control, seed
        )
        cute.arch.sync_threads()
        batch, head, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        slot = ids[batch]
        start = cu[batch]
        tokens = cu[batch + 1] - start
        if slot > 0:
            warp = tid // 32
            lane = tid % 32
            group = head // (self.heads // self.groups)
            old = Int32(control[batch, 0])
            crossed = control[batch, 2] != 0
            remain = Int32(control[batch, 3])
            smem = cutlass.utils.SmemAllocator()
            cumulative = smem.allocate_array(Float32, 24)
            deltas = smem.allocate_array(Float32, 24)
            products = smem.allocate_array(Float32, 24 * self.max_tokens)
            if tid < remain + tokens:
                i = tid
                delta = Float32(0.0)
                if i < remain:
                    ri = i if i < old else 16 + i - old
                    if crossed:
                        ri = 32 - old + i
                    delta = dc[slot, head, ri]
                else:
                    value = Float32(dt[start + i - remain, head]) + Float32(bias[head])
                    delta = value
                    if value <= 20.0:
                        delta = cute.math.log(
                            Float32(1.0) + cute.math.exp(value, fastmath=True),
                            fastmath=True,
                        )
                deltas[i] = delta
            cute.arch.sync_threads()
            if tid == 0:
                accum = Float32(0.0)
                for i in cutlass.range(remain + tokens, unroll=1):
                    accum += deltas[i] * Float32(a[head])
                    cumulative[i] = accum
            cute.arch.sync_threads()
            for index in cutlass.range(
                tid, (remain + tokens) * self.max_tokens, 128, unroll=1
            ):
                i = index // self.max_tokens
                t = index % self.max_tokens
                if t < tokens and i <= remain + t:
                    weight = deltas[i] * cute.math.exp(
                        cumulative[remain + t] - cumulative[i], fastmath=True
                    )
                    products[index] = cb_products[batch, group, i, t] * weight
            cute.arch.sync_threads()
            row = warp * 16 + lane // 4
            accumulators = cute.make_rmem_tensor(4, Float32)
            accumulators.fill(0.0)
            # FP8 values and NVFP4 value times block scale are exact in BF16.
            # Keep the arbitrary FP32 row/global scale outside the MMA.
            for tile in cutlass.range_constexpr(8):
                ar = cute.make_rmem_tensor(4, Uint32)
                for j in cutlass.range_constexpr(4):
                    rr = row + j % 2 * 8
                    col = tile * 16 + lane % 4 * 2 + j // 2 * 8
                    if cutlass.const_expr(self.nvfp4):
                        packed = Uint32(read[slot, (head * 64 + rr) * 64 + col // 2])
                        v0, v1 = _unpack_e2m1(packed)
                        sf = e4m3_byte_to_f32(
                            Uint32(scale[slot, _scale_offset(head, rr, tile)])
                        )
                        ar[j] = _pack_bf16(v0 * sf, v1 * sf)
                    else:
                        ar[j] = _pack_bf16(
                            Float32(read[slot, (head * 64 + rr) * 128 + col]),
                            Float32(read[slot, (head * 64 + rr) * 128 + col + 1]),
                        )
                br = cute.make_rmem_tensor(2, Uint32)
                br.fill(0)
                if lane // 4 < tokens:
                    for j in cutlass.range_constexpr(2):
                        col = tile * 16 + lane % 4 * 2 + j * 8
                        br[j] = _pack_bf16(
                            c[start + lane // 4, group, col],
                            c[start + lane // 4, group, col + 1],
                        )
                result = _mma(ar[0], ar[1], ar[2], ar[3], br[0], br[1])
                for j in cutlass.range_constexpr(4):
                    accumulators[j] += result[j]
            for j in cutlass.range_constexpr(4):
                rr = row + j // 2 * 8
                t = lane % 4 * 2 + j % 2
                read_scale = Float32(1.0)
                if cutlass.const_expr(self.nvfp4):
                    read_scale = meta[slot, 2, group]
                else:
                    read_scale = scale[slot, head * 64 + rr]
                if t < tokens:
                    acc = accumulators[j] * read_scale
                    acc *= cute.math.exp(cumulative[remain + t], fastmath=True)
                    for i in cutlass.range(remain + t + 1, unroll=1):
                        xv = Float32(0.0)
                        if i < remain:
                            ri = i if i < old else 16 + i - old
                            if crossed:
                                ri = 32 - old + i
                            xv = Float32(xc[slot, head, ri, rr])
                        else:
                            xv = Float32(x[start + i - remain, head, rr])
                        acc += xv * products[i * self.max_tokens + t]
                    acc += Float32(d[head, rr]) * Float32(x[start + t, head, rr])
                    out[start + t, head, rr] = BFloat16(acc)


@cache
def _compile(heads, groups, length, max_tokens, nvfp4, strides, inputs):
    batch, slots, tokens = cute.sym_int(), cute.sym_int(), cute.sym_int()
    read_dtype = Uint8 if nvfp4 else cutlass.Float8E4M3FN
    scale_dtype = Uint8 if nvfp4 else Float32
    shapes = [
        (Float16, (slots, heads, 64, 128)),
        (read_dtype, (slots, heads * 64 * (64 if nvfp4 else 128))),
        (scale_dtype, (slots, heads * (512 if nvfp4 else 64))),
        (Float32, (slots, 3, groups)),
        (Int32, (slots, groups)),
        (BFloat16, (slots, heads, length, 64)),
        (Float32, (slots, heads, length)),
        (BFloat16, (slots, groups, length, 128)),
    ]
    arguments = [
        make_fake_tensor(dtype, shape, divisibility=1, slot_stride=strides[i])
        for i, (dtype, shape) in enumerate(shapes)
    ]
    input_shapes = (
        (tokens, heads, 64),
        (tokens, heads),
        (heads,),
        (tokens, groups, 128),
        (tokens, groups, 128),
        (heads, 64),
        (heads,),
    )
    arguments += [
        cute.runtime.make_fake_tensor(
            BFloat16 if dtype == torch.bfloat16 else Float32,
            shape,
            tensor_strides,
            assumed_align=1,
        )
        for shape, (dtype, tensor_strides) in zip(input_shapes, inputs)
    ]
    arguments += [
        make_fake_tensor(Int32, (batch,), divisibility=1),
        make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1),
        make_fake_tensor(Int64, (batch, 5), divisibility=1),
        make_fake_tensor(Float32, (batch, groups, 24, max_tokens), divisibility=1),
        make_fake_tensor(BFloat16, (tokens, heads, 64)),
        Uint32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    ]
    return cute.compile(
        PendingKernel(heads, groups, length, max_tokens, nvfp4),
        *arguments,
        options="--enable-tvm-ffi",
    )


def run_pending_kernels(
    cache, x, dt, a, b, c, d, bias, ids, cu, control, out, *, nvfp4, max_tokens, seed
):
    slots, heads = cache[1].shape[:2]
    groups, length = cache[4].shape[1:3]
    meta = (
        cache[7]
        if nvfp4
        else cache[6].as_strided((slots, 3, groups), (cache[6].stride(0), groups, 1))
    )
    parity = (
        cache[8]
        if nvfp4
        else cache[6]
        .view(torch.int32)
        .as_strided((slots, groups), (cache[6].stride(0), 1))
    )
    tensors = [
        cache[1],
        cache[5].flatten(1),
        cache[6].flatten(1),
        meta,
        parity,
        *cache[2:5],
    ]
    strides = tuple(t.stride(0) for t in tensors)
    inputs = tuple((t.dtype, tuple(t.stride())) for t in (x, dt, a, b, c, d, bias))
    cb_products = torch.empty(
        (ids.numel(), groups, 24, max_tokens), dtype=torch.float32, device=x.device
    )
    args = (
        *tensors,
        x,
        dt,
        a,
        b,
        c,
        d,
        bias,
        ids,
        cu,
        control,
        cb_products,
        out,
        seed & 0xFFFFFFFF,
    )
    # The first eager call compiles both kernels before graph capture.
    _compile(heads, groups, length, max_tokens, nvfp4, strides, inputs)(*args)
