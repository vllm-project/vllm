# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# once we have more CuteDSL kernels in vLLM, we can refactor small helper functions
# to a separate file
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from vllm.vllm_flash_attn.cute import utils as cute_utils

# MXFP4: 32 elements per block, packed 2 nibbles per byte, ue8m0 block scale.
MXFP4_BLOCK_SIZE = 32

_TORCH_TO_CUTE = {
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
}


def fused_indexer_q_rope_quant_mxfp4_cutedsl(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    index_q_packed: torch.Tensor,
    index_q_scale: torch.Tensor,
    index_weights_out: torch.Tensor,
) -> None:
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]
    compiled = IndexerQMxFp4Kernel.compile(
        index_q_head_dim,
        index_q_cos_sin_cache.shape[-1],
        num_index_q_heads,
        _TORCH_TO_CUTE[index_q_cos_sin_cache.dtype],
    )
    scale = float(index_weights_softmax_scale * index_weights_head_scale)
    compiled(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_q_packed,
        index_q_scale,
        index_weights_out,
        scale,
    )


@dsl_user_op
def _fp32x2_to_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _bf16x2_to_fp32(data: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [data.ir_value(loc=loc, ip=ip)],
        "shl.b32 $0, $2, 16;\n\tand.b32 $1, $2, 0xFFFF0000;\n",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _bf16x2_abs(a: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip)],
        "abs.bf16x2 $0, $1;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "max.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _fp32x8_to_fp4x8(
    vals: cute.Tensor,
    offset: cutlass.Constexpr[int],
    *,
    loc=None,
    ip=None,
) -> Uint32:
    # Pack eight scaled FP32 values into four E2M1x2 bytes, returned as one b32.
    assert vals.element_type is Float32
    out = llvm.inline_asm(
        T.i32(),
        [vals[offset + i].ir_value(loc=loc, ip=ip) for i in range(8)],
        "{\n\t"
        ".reg .b8 x0, x1, x2, x3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x0, $2, $1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x1, $4, $3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x2, $6, $5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x3, $8, $7;\n\t"
        "mov.b32 $0, {x0, x1, x2, x3};\n\t"
        "}\n",
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


# Custom vectorized load to support cache modifiers. For some reason,
# cute.autovec_copy() does not currently emit the requested modifiers.
# tensor and coord is only used to select the base pointer. actual load
# is done using out_dtype
@dsl_user_op
def _ldg_vec(
    tensor: cute.Tensor,
    coord: cute.Coord,
    vec_size: cutlass.Constexpr[int],
    modifier: cutlass.Constexpr[str] = "",
    ld_type: cutlass.Constexpr[type[cutlass.Numeric] | None] = None,
    *,
    loc=None,
    ip=None,
) -> cute.TensorSSA:
    if ld_type is None:
        ld_type = tensor.element_type
    if const_expr(ld_type is Float32):
        ptx_ty = "f32"
        constraint = "=f"
    elif const_expr(ld_type is Uint32):
        ptx_ty = "u32"
        constraint = "=r"
    else:
        raise TypeError(f"_ldg_vec only supports Uint32 and Float32, got {ld_type}")

    # compute base pointer
    base_ptr = (
        tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)
    ).toint()

    # build PTX string
    ptx_str = f"ld.global{modifier}.v{vec_size}.{ptx_ty}"
    ptx_str += "{" + ", ".join(f"${i}" for i in range(vec_size)) + "}"
    ptx_str += f", [${vec_size}];"

    out = llvm.inline_asm(
        llvm.StructType.get_literal([ld_type.mlir_type] * vec_size),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        ptx_str,
        ",".join([constraint] * vec_size + ["l"]),
        has_side_effects=False,
        is_align_stack=False,
    )
    vec = vector.from_elements(
        ir.VectorType.get([vec_size], ld_type.mlir_type, loc=loc),
        [
            llvm.extractvalue(ld_type.mlir_type, out, [i], loc=loc, ip=ip)
            for i in range(vec_size)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, vec_size, ld_type)


@dsl_user_op
def _stg_vec(
    tensor: cute.Tensor,
    coord: cute.Coord,
    values: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    modifier: cutlass.Constexpr[str] = "",
    *,
    loc=None,
    ip=None,
) -> None:
    # NOTE: st_type is derived from values tensor
    st_type = values.element_type
    if const_expr(st_type is Float32):
        ptx_ty = "f32"
        constraint = "f"
    elif const_expr(st_type is Uint32):
        ptx_ty = "u32"
        constraint = "r"
    else:
        raise TypeError(f"_stg_vec only supports Uint32 and Float32, got {st_type}")

    # compute base pointer
    base_ptr = (
        tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)
    ).toint()

    # build PTX string
    ptx_str = f"st.global{modifier}.v{vec_size}.{ptx_ty} [$0], "
    ptx_str += "{" + ", ".join(f"${i + 1}" for i in range(vec_size)) + "};"

    llvm.inline_asm(
        None,
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)]
        + [values[i].ir_value(loc=loc, ip=ip) for i in range(vec_size)],
        ptx_str,
        ",".join(["l"] + [constraint] * vec_size),
        has_side_effects=True,
        is_align_stack=False,
    )


class IndexerQMxFp4Kernel:
    """Eight-thread subwarps process one ``(token, head)`` row."""

    def __init__(
        self,
        head_dim: int = 128,
        rope_dim: int = 64,
        num_heads: int = 64,
        cos_sin_dtype: type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.nope_dim = head_dim - rope_dim
        self.num_heads = num_heads
        self.cos_sin_dtype = cos_sin_dtype

        # later we will use 32B load = 16 BF16 elems
        # thus, head_dim=128 requires 8 threads to handle.
        # let's call subwarp = 8 threads.
        self.subwarp_size = head_dim // 16
        self.tb_size = 256

    @cute.jit
    def __call__(
        self,
        positions: cute.Tensor,
        q: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        weights: cute.Tensor,
        q_fp4: cute.Tensor,
        q_scale: cute.Tensor,
        weights_out: cute.Tensor,
        scale: Float32,
        stream: CUstream,
    ):
        num_tokens, num_heads, _ = q.shape
        total_threads = num_tokens * num_heads * self.subwarp_size
        grid = [cute.ceil_div(total_threads, self.tb_size), 1, 1]
        self.kernel(
            positions,
            q,
            cos_sin_cache,
            weights,
            q_fp4,
            q_scale,
            weights_out,
            scale,
        ).launch(grid=grid, block=[self.tb_size, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        positions: cute.Tensor,
        q: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        weights: cute.Tensor,
        q_fp4: cute.Tensor,
        q_scale: cute.Tensor,
        weights_out: cute.Tensor,
        scale: Float32,
    ):
        block_id, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        num_token_heads = q.shape[0] * self.num_heads
        global_tid = block_id * self.tb_size + tidx

        global_subwarp_id = global_tid // self.subwarp_size
        sublane = tidx % self.subwarp_size

        token_id = global_subwarp_id // self.num_heads
        head_id = global_subwarp_id - token_id * self.num_heads

        # each thread loads 16 BF16 elems
        elem_base = sublane * 16

        # q layout: [num_tokens, num_heads, head_dim]
        _q_bf16x2 = _ldg_vec(
            q,
            (token_id, head_id, elem_base),
            8,
            ".relaxed.cta.L1::no_allocate",
            ld_type=Uint32,
        )
        q_bf16x2 = cute.make_rmem_tensor(8, Uint32)
        q_bf16x2.store(_q_bf16x2)  # copy to make it mutable

        # RoPE applies only to the trailing rope_dim values. We keep the rounded
        # BF16 result in q_bits so the later amax and quantization see BF16.
        # cos_sin_cache layout: [max_pos, rope_dim]
        if elem_base >= self.nope_dim:
            pos = positions[token_id]
            rope_idx = (elem_base - self.nope_dim) // 2
            if const_expr(self.cos_sin_dtype is Float32):
                # fp32x8 loads
                cos_vals = _ldg_vec(cos_sin_cache, (pos, rope_idx), 8)
                sin_vals = _ldg_vec(
                    cos_sin_cache, (pos, self.rope_dim // 2 + rope_idx), 8
                )
            else:
                # bf16x2x4 loads
                cos_bf16x2 = _ldg_vec(
                    cos_sin_cache,
                    (pos, rope_idx),
                    4,
                    ld_type=Uint32,
                )
                sin_bf16x2 = _ldg_vec(
                    cos_sin_cache,
                    (pos, self.rope_dim // 2 + rope_idx),
                    4,
                    ld_type=Uint32,
                )
                cos_vals = cute.make_rmem_tensor(8, Float32)
                sin_vals = cute.make_rmem_tensor(8, Float32)
                for i in cutlass.range_constexpr(4):
                    cos_vals[i * 2], cos_vals[i * 2 + 1] = _bf16x2_to_fp32(
                        cos_bf16x2[i]
                    )
                    sin_vals[i * 2], sin_vals[i * 2 + 1] = _bf16x2_to_fp32(
                        sin_bf16x2[i]
                    )

            for i in cutlass.range_constexpr(8):
                q0, q1 = _bf16x2_to_fp32(q_bf16x2[i])
                rot0 = q0 * cos_vals[i] - q1 * sin_vals[i]
                rot1 = q0 * sin_vals[i] + q1 * cos_vals[i]
                # convert back to BF16 to match numerics
                q_bf16x2[i] = _fp32x2_to_bf16x2(rot0, rot1)

        # compute amax in packed bf16x2 to save instructions
        # Each thread holds 16 elems. Two adjacent threads form one 32-elem
        # MXFP4 block, so a width-2 shuffle gives the block amax.
        local_amax = _bf16x2_abs(q_bf16x2[0])
        for i in cutlass.range_constexpr(1, 8):
            local_amax = _bf16x2_max(local_amax, _bf16x2_abs(q_bf16x2[i]))
        amax_bits = cute_utils.warp_reduce(
            local_amax, _bf16x2_max, width=MXFP4_BLOCK_SIZE // 16
        )
        amax0, amax1 = _bf16x2_to_fp32(amax_bits)
        amax = cute_utils.fmax(amax0, amax1)

        # compute block scale with bit manipulation
        # UE8M0 stores ceil(log2(fp4_scale)) + 127. Adding the mantissa mask
        # increments the exponent whenever fp4_scale is not exactly a power of 2.
        fp4_scale = cute_utils.fmax(amax, float.fromhex("0x6p-126")) * (1.0 / 6.0)
        bits = Uint32(llvm.bitcast(T.i32(), fp4_scale.ir_value()))
        ue8m0 = cute_utils.shr_u32(bits + Uint32(0x7FFFFF), Uint32(23)) & Uint32(0xFF)

        # Only one of the two threads in an MXFP4 block writes the shared scale.
        if tidx % 2 == 0:
            mx_block = sublane // (MXFP4_BLOCK_SIZE // 16)
            q_scale[token_id, head_id, mx_block] = Uint8(ue8m0)

        # If scale = 2^A and ue8m0 = A + 127, then inverse scale has exponent
        # -A + 127 = 254 - ue8m0.
        inv_scale_bits = (Uint32(254) - ue8m0) << Uint32(23)
        inv_fp4_scale = Float32(llvm.bitcast(T.f32(), inv_scale_bits.ir_value()))

        vals = cute.make_rmem_tensor(16, Float32)
        for i in cutlass.range_constexpr(8):
            vals[i * 2], vals[i * 2 + 1] = _bf16x2_to_fp32(q_bf16x2[i])
            vals[i * 2] = vals[i * 2] * inv_fp4_scale
            vals[i * 2 + 1] = vals[i * 2 + 1] * inv_fp4_scale

        # pack to FP4
        packed = cute.make_rmem_tensor(2, Uint32)
        packed[0] = _fp32x8_to_fp4x8(vals, 0)
        packed[1] = _fp32x8_to_fp4x8(vals, 8)
        # Each thread writes the eight packed bytes corresponding to its 16 Q values.
        _stg_vec(q_fp4, (token_id, head_id, elem_base // 2), packed, 2, ".cs")

        # Weight scaling is independent of the Q subwarp work. The first
        # num_tokens * num_heads logical threads cover one weight each.
        if global_tid < num_token_heads:
            weight_token_id = global_tid // self.num_heads
            weight_head_id = global_tid - weight_token_id * self.num_heads
            weights_out[weight_token_id, weight_head_id] = (
                weights[weight_token_id, weight_head_id].to(Float32) * scale
            )

    @cache
    @staticmethod
    def compile(
        head_dim: int,
        rope_dim: int,
        num_heads: int,
        cos_sin_dtype: type[cutlass.Numeric],
    ):
        num_tokens = cute.sym_int()
        max_pos = cute.sym_int()

        q = make_fake_tensor(
            BFloat16, (num_tokens, num_heads, head_dim), divisibility=8
        )
        positions = make_fake_tensor(Int64, (num_tokens,), divisibility=1)
        cos_sin_cache = make_fake_tensor(
            cos_sin_dtype,
            (max_pos, rope_dim),
            divisibility=8,
        )
        weights = make_fake_tensor(BFloat16, (num_tokens, num_heads), divisibility=8)
        q_fp4 = make_fake_tensor(
            Uint8,
            (num_tokens, num_heads, head_dim // 2),
            divisibility=16,
        )
        q_scale = make_fake_tensor(
            Uint8,
            (num_tokens, num_heads, head_dim // MXFP4_BLOCK_SIZE),
            divisibility=4,
        )
        weights_out = make_fake_tensor(Float32, (num_tokens, num_heads), divisibility=4)

        kernel = IndexerQMxFp4Kernel(head_dim, rope_dim, num_heads, cos_sin_dtype)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            positions,
            q,
            cos_sin_cache,
            weights,
            q_fp4,
            q_scale,
            weights_out,
            Float32(0.0),
            stream,
            options="--enable-tvm-ffi",
        )
