# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4-specific FlyDSL preshuffle helpers."""

from contextlib import contextmanager
from typing import Any

import flydsl.expr as fx
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    buffer_copy_gmem16_dwordx4,
    crd2idx,
    lds_store_4b_xor16,
    lds_store_8b_xor16,
    lds_store_16b_xor16,
    swizzle_xor16,
)
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects._rocdl_ops_gen import cvt_scalef32_pk_bf16_fp4
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T


@contextmanager
def _if_then(if_op):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def ptr_buffer_resource(ptr, num_records_bytes, *, arith, buffer_ops):
    """Create a buffer resource from a FlyDSL pointer argument."""
    address = fx.ptrtoint(ptr)
    address_i64 = arith.index_cast(T.i64, address)
    return buffer_ops.create_buffer_resource_from_addr(
        address_i64, num_records_bytes=num_records_bytes
    )


def flatten_nvfp4_b_tile(tile):
    """Serialize packed NVFP4 values and block scales for loop-carried state."""
    flattened: list[Any] = []
    for entry in tile:
        flattened.extend(value[0] for value in entry)
        flattened.extend(value[1] for value in entry)
    return flattened


def unflatten_nvfp4_b_tile(values, *, k_unroll: int, num_acc_n: int):
    """Reconstruct packed NVFP4 values and block scales from loop state."""
    tile: list[Any] = []
    index = 0
    for _ in range_constexpr(k_unroll):
        packed = list(values[index : index + num_acc_n])
        index += num_acc_n
        scales = list(values[index : index + num_acc_n])
        index += num_acc_n
        tile.append([(packed[ni], scales[ni]) for ni in range_constexpr(num_acc_n)])
    return tile


def _i64_to_v4i16(x_i64, vector):
    v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
    return vector.bitcast(T.i16x4, v1)


def _i64x2_to_v8bf16(lo, hi, vector):
    v2 = vector.from_elements(T.i64x2, [lo, hi])
    return vector.bitcast(T.bf16x8, v2)


def mfma_k64(
    acc,
    a0,
    a1,
    b0,
    b1,
    *,
    mfma_fn,
    mfma_res_ty,
    use_mfma_k32: bool,
    vector,
):
    """Run the BF16 MFMA sequence for one 64-byte K fragment."""
    if const_expr(use_mfma_k32):
        av = _i64x2_to_v8bf16(a0, a1, vector)
        bv = _i64x2_to_v8bf16(b0, b1, vector)
        return mfma_fn(mfma_res_ty, [av, bv, acc, 0, 0, 0])

    a0v = _i64_to_v4i16(a0, vector)
    a1v = _i64_to_v4i16(a1, vector)
    b0v = _i64_to_v4i16(b0, vector)
    b1v = _i64_to_v4i16(b1, vector)
    acc_mid = mfma_fn(mfma_res_ty, [a0v, b0v, acc, 0, 0, 0])
    return mfma_fn(mfma_res_ty, [a1v, b1v, acc_mid, 0, 0, 0])


def load_bf16_x(idx_i32, *, buffer_ops, vector, x_elem, x_rsrc, vec16_elems):
    """Load the fixed 16-byte BF16 activation fragment."""
    return buffer_copy_gmem16_dwordx4(
        buffer_ops,
        vector,
        elem_type=x_elem,
        idx_i32=idx_i32 * fx.Index(2),
        rsrc=x_rsrc,
        vec_elems=vec16_elems,
        elem_bytes=2,
    )


def lds_load_bf16_k64(
    curr_row,
    col_base_bytes,
    lds_base,
    *,
    arith,
    vector,
    k_blocks16,
    layout_lds,
    vec16_x,
    lds_x,
):
    """Load the two BF16 MFMA operands for one K64 fragment from LDS."""
    col_swizzled = swizzle_xor16(curr_row, col_base_bytes, k_blocks16)
    col = col_swizzled // arith.index(2)
    idx = crd2idx((fx.Int32(curr_row), fx.Int32(col)), layout_lds) + lds_base
    loaded = vector.load_op(vec16_x, lds_x, [idx])
    values = vector.bitcast(T.i64x2, loaded)
    return (
        vector.extract(values, static_position=[0], dynamic_position=[]),
        vector.extract(values, static_position=[1], dynamic_position=[]),
    )


def store_x_tile_to_lds(
    values,
    lds_base,
    *,
    arith,
    vector,
    x_row_local,
    x_col_local_i32,
    num_x_loads: int,
    x_load_bytes: int,
    lds_x,
    vec16_x,
    vec8_x,
    vec4_x,
    layout_lds,
    k_blocks16,
    elem_bytes: int,
):
    """Store one prefetched activation tile into the ping-pong LDS buffer."""
    for i in range_constexpr(num_x_loads):
        common = dict(
            lds_memref=lds_x,
            layout_lds=layout_lds,
            row_local=x_row_local[i],
            col_local_i32=x_col_local_i32[i],
            tx_c4=fx.Index(4),
            k_blocks16=k_blocks16,
            lds_base=lds_base,
        )
        if const_expr(x_load_bytes == 16):
            lds_store_16b_xor16(
                arith,
                vector,
                vec16_ty=vec16_x,
                vec_part_i32x4=values[i],
                elem_bytes=elem_bytes,
                **common,
            )
        elif const_expr(x_load_bytes == 8):
            lds_store_8b_xor16(
                arith,
                vector,
                vec8_ty=vec8_x,
                vec_part_i32x2=values[i],
                **common,
            )
        else:
            lds_store_4b_xor16(
                arith,
                vector,
                vec4_ty=vec4_x,
                vec_part_i32x1=values[i],
                **common,
            )


def load_b_raw_nvfp4(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ku: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 8,
):
    """Load one packed NVFP4 dword for the two-phase MFMA pipeline."""
    if kpack_bytes != 8:
        raise ValueError(f"NVFP4 requires kpack_bytes=8, got {kpack_bytes!r}")

    c64 = fx.Index(64)
    half_bytes = kpack_bytes // 2
    c2_idx = fx.Index(2)
    c4_idx = fx.Index(4)

    k0_base = base_k // c64

    k1_layout_offset = ku * 2
    lane_div_32 = lane_div_16 // c2_idx
    total_k1 = fx.Index(k1_layout_offset) + lane_div_32
    k0 = k0_base + (total_k1 // c4_idx)
    k1_local = total_k1 % c4_idx
    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * fx.Index(half_bytes)

    coord_pack = (n_blk, k0, k1_local, n_intra, fx.Index(0))
    idx_pack = crd2idx(tuple(fx.Int32(c) for c in coord_pack), layout_b)
    idx_bytes = idx_pack + k2_base

    b4 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_bytes,
        elem_type=elem_type,
        vec_elems=4,
        elem_bytes=1,
        offset_in_bytes=True,
    )
    packed32 = vector.extract(
        vector.bitcast(T.vec(1, T.i32), b4),
        static_position=[0],
        dynamic_position=[],
    )
    return packed32


def _load_fp8_block_scale(
    buffer_ops,
    arith,
    *,
    scale_rsrc,
    expert_idx,
    n_blk,
    n_intra,
    k_pos,
    num_groups: int,
    group_size: int,
    n_per_expert: int,
):
    """Load one fp8_e4m3 block scale from logical MoE [E, G, N] layout."""
    c16 = fx.Index(16)
    c_npe = fx.Index(n_per_expert)
    n_local = (n_blk * c16 + n_intra) % c_npe
    group_idx = k_pos // fx.Index(group_size)
    elem_idx = (expert_idx * fx.Index(num_groups) + group_idx) * c_npe + n_local
    return buffer_ops.buffer_load(
        scale_rsrc, arith.index_cast(T.i32, elem_idx), vec_width=1, dtype=T.i8
    )


def load_b_raw_nvfp4_groupwise(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k,
    ku: int,
    n_blk,
    n_intra,
    lane_div_16,
    elem_type,
    scale_rsrc,
    expert_idx,
    num_groups: int,
    n_per_expert: int,
    kpack_bytes: int = 8,
):
    """Load packed fp4 weights plus the fp8 block scale for this lane's K group."""
    packed32 = load_b_raw_nvfp4(
        buffer_ops,
        arith,
        vector,
        arg_b=arg_b,
        b_rsrc=b_rsrc,
        layout_b=layout_b,
        base_k=base_k,
        ku=ku,
        n_blk=n_blk,
        n_intra=n_intra,
        lane_div_16=lane_div_16,
        elem_type=elem_type,
        kpack_bytes=kpack_bytes,
    )
    # A ku step covers 32 K elements across the wave, but each lane's packed32
    # contributes only 8 fp4 values. lane_div_16 0/1 map to the first K16 scale
    # group and lane_div_16 2/3 map to the second.
    k_pos = base_k + fx.Index(ku * 32) + (lane_div_16 // fx.Index(2)) * fx.Index(16)
    scale = _load_fp8_block_scale(
        buffer_ops,
        arith,
        scale_rsrc=scale_rsrc,
        expert_idx=expert_idx,
        n_blk=n_blk,
        n_intra=n_intra,
        k_pos=k_pos,
        num_groups=num_groups,
        group_size=16,
        n_per_expert=n_per_expert,
    )
    return packed32, scale


def load_nvfp4_b_tile(
    base_k,
    n_blk,
    n_intra,
    *,
    buffer_ops,
    arith,
    vector,
    arg_w,
    w_rsrc,
    layout_b,
    lane_div_16,
    w_elem,
    sw_rsrc,
    expert_idx,
    num_groups: int,
    n_per_expert: int,
    kpack_bytes: int,
    k_unroll: int,
    num_acc_n: int,
):
    """Prefetch one NVFP4 weight tile and its E4M3 block scales."""
    raw_data = []
    for ku in range_constexpr(k_unroll):
        raw_ku = []
        for ni in range_constexpr(num_acc_n):
            packed32, scale = load_b_raw_nvfp4_groupwise(
                buffer_ops,
                arith,
                vector,
                arg_b=arg_w,
                b_rsrc=w_rsrc,
                layout_b=layout_b,
                base_k=base_k,
                ku=ku,
                n_blk=n_blk[ni],
                n_intra=n_intra[ni],
                lane_div_16=lane_div_16,
                elem_type=w_elem,
                scale_rsrc=sw_rsrc,
                expert_idx=expert_idx,
                num_groups=num_groups,
                n_per_expert=n_per_expert,
                kpack_bytes=kpack_bytes,
            )
            raw_ku.append((packed32, scale))
        raw_data.append(raw_ku)
    return raw_data


def pack_bf16x4_to_i64(elems, arith, vector):
    v_i16x4 = vector.from_elements(
        T.i16x4, [arith.bitcast(T.i16, elem) for elem in elems]
    )
    v_i32x2 = vector.bitcast(T.vec(2, T.i32), v_i16x4)
    v_i64x1 = vector.bitcast(T.vec(1, T.i64), v_i32x2)
    return vector.extract(v_i64x1, static_position=[0], dynamic_position=[])


def _unpack_b_nvfp4_gfx950(packed32, scale_f32, arith, vector):
    unity = arith.constant(1.0, type=T.f32)
    bf16x2_ty = T.vec(2, T.bf16)

    def fp4_bytes_to_i64(sel_a: int, sel_b: int, block_scale):
        elems = []
        for byte_sel in (sel_a, sel_b):
            # Use hardware v_cvt_scalef32_pk_bf16_fp4 with scale=1.0
            # (f32 1.0 has exponent bits 0x7F=127, so E8M0 factor = 2^(127-127) = 1)
            # for pure FP4->bf16 conversion, then scale separately in bf16.
            pair = cvt_scalef32_pk_bf16_fp4(bf16x2_ty, packed32, unity, byte_sel)
            for elem_idx in range(2):
                v = vector.extract(
                    pair, static_position=[elem_idx], dynamic_position=[]
                )
                v_f32 = arith.extf(T.f32, v)
                elems.append(arith.truncf(T.bf16, v_f32 * block_scale))
        return pack_bf16x4_to_i64(elems, arith, vector)

    b0 = fp4_bytes_to_i64(0, 1, scale_f32)
    b1 = fp4_bytes_to_i64(2, 3, scale_f32)
    return b0, b1


def _unpack_b_nvfp4_gfx942(packed32, scale_f32, arith, vector):
    def decode_e2m1_to_f32(nibble_i32):
        magnitude = arith.andi(nibble_i32, arith.constant(0x07, type=T.i32))
        sign_bit = arith.andi(
            arith.shrui(nibble_i32, arith.constant(3, type=T.i32)),
            arith.constant(1, type=T.i32),
        )
        fp32_bits = arith.addi(
            arith.constant(0x3F000000, type=T.i32),
            arith.shli(magnitude, arith.constant(22, type=T.i32)),
        )
        val = arith.bitcast(T.f32, fp32_bits)
        is_zero = arith.cmpi(CmpIPredicate.eq, magnitude, arith.constant(0, type=T.i32))
        val = arith.select(is_zero, arith.constant(0.0, type=T.f32), val)
        is_one = arith.cmpi(CmpIPredicate.eq, magnitude, arith.constant(1, type=T.i32))
        val = arith.select(is_one, arith.constant(0.5, type=T.f32), val)
        neg_val = arith.negf(val)
        is_negative = arith.cmpi(
            CmpIPredicate.ne, sign_bit, arith.constant(0, type=T.i32)
        )
        return arith.select(is_negative, neg_val, val)

    # NVFP4 block scales are stored as OCP e4m3fn. On gfx942, v_cvt_f32_fp8
    # decodes the same byte as AMD e4m3fnuz (bias 8), which is half the OCP
    # value (bias 7). Compensate in the software FP4 fallback.
    scale_f32 = scale_f32 * fx.Float32(2.0)
    packed_vec = vector.bitcast(
        T.vec(4, T.i8), vector.from_elements(T.vec(1, T.i32), [packed32])
    )

    def fp4_bytes_to_i64(sel_a: int, sel_b: int, block_scale):
        elems = []
        for byte_sel in (sel_a, sel_b):
            raw_byte = vector.extract(
                packed_vec, static_position=[byte_sel], dynamic_position=[]
            )
            raw_byte_i32 = arith.extui(T.i32, raw_byte)

            low_nibble = arith.andi(raw_byte_i32, arith.constant(0x0F, type=T.i32))
            low_bf16 = arith.truncf(
                T.bf16, arith.mulf(decode_e2m1_to_f32(low_nibble), block_scale)
            )

            high_nibble = arith.andi(
                arith.shrui(raw_byte_i32, arith.constant(4, type=T.i32)),
                arith.constant(0x0F, type=T.i32),
            )
            high_bf16 = arith.truncf(
                T.bf16, arith.mulf(decode_e2m1_to_f32(high_nibble), block_scale)
            )

            elems.append(low_bf16)
            elems.append(high_bf16)
        return pack_bf16x4_to_i64(elems, arith, vector)

    b0 = fp4_bytes_to_i64(0, 1, scale_f32)
    b1 = fp4_bytes_to_i64(2, 3, scale_f32)
    return b0, b1


def unpack_b_nvfp4(packed32, scale_i8, arith, vector, use_gfx950_cvt=False):
    """Dequantize packed fp4 to two bf16x4 i64 fragments for BF16 MFMA.

    The fp8 block scale is applied here. The global f32 scale is intentionally
    left for the epilogue so accumulation remains in f32 before global scaling.
    """
    scale_f32 = rocdl.cvt_f32_fp8(T.f32, arith.extui(T.i32, scale_i8), 0)

    if use_gfx950_cvt:
        return _unpack_b_nvfp4_gfx950(packed32, scale_f32, arith, vector)
    else:
        return _unpack_b_nvfp4_gfx942(packed32, scale_f32, arith, vector)
