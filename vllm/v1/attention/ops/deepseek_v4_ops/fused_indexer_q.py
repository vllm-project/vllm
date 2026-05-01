# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from vllm.triton_utils import tl, triton
from vllm.vllm_flash_attn.cute import utils as cute_utils

# MXFP4: 32 elements per block, packed 2 nibbles per byte, ue8m0 block scale.
MXFP4_BLOCK_SIZE = 32

_TORCH_TO_CUTE = {
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
}


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _quantize_mxfp4_pair(x_lo, x_hi):
    """Quantize a block of MXFP4_BLOCK_SIZE fp32 values given as two
    interleaved halves (x_lo = values at even positions in the block,
    x_hi = values at odd positions). Returns:
        - packed : uint8[BLOCK/2]  (low nibble = quant(x_lo), high = quant(x_hi))
        - ue8m0  : scalar uint8    (block scale = 2^(ue8m0 - 127))
    """
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    # 6 * 2^-126 is from https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py#L163
    amax = tl.maximum(amax, 6.0 * (2**-126))
    # ue8m0 block scale: 2^ceil(log2(amax/6.0)).
    log2_ratio = tl.math.ceil(tl.math.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    scale = tl.math.exp2(log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)

    inv_scale = 1.0 / scale
    packed = _fp32x2_to_fp4x2(x_lo * inv_scale, x_hi * inv_scale)
    return packed, ue8m0


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    pos_ptr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # Index Q Quantize
    index_q_fp8_ptr,
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    # Index weights
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
):
    # Layout matches the unfused reference (DeepseekV4ScalingRotaryEmbedding
    # + per_token_group_quant_fp8): GPT-J interleaved RoPE applied to the
    # LAST rope_dim dims of each head; the leading [0, NOPE_DIM) is passed
    # through unchanged.
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        pos,
        INDEX_Q_HALF_ROT_DIM,
    )
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    # Interleaved (GPT-J) RoPE on dims [NOPE_DIM, HEAD_DIM):
    #   even = q[NOPE_DIM + 2*i],  odd = q[NOPE_DIM + 2*i + 1]
    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin

    # Match reference numerics: fp32 → bf16 → fp32 before the ue8m0 absmax.
    # Same pattern as the K-side compressor kernel (fused_compress_quant_cache.py).
    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
    index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
    index_q_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))

    # Store quantized values to index_q_fp8
    fp8_base_ptr = (
        index_q_fp8_ptr + tok_idx * index_q_fp8_stride0 + head_idx * index_q_fp8_stride1
    )
    if INDEX_Q_NOPE_DIM > 0:
        tl.store(
            fp8_base_ptr + nope_offset,
            tl.div_rn(x_nope, index_q_scale).to(tl.float8e4nv),
        )
    fp8_rot_base = fp8_base_ptr + INDEX_Q_NOPE_DIM
    tl.store(
        fp8_rot_base + half_offset * 2,
        tl.div_rn(r_even, index_q_scale).to(tl.float8e4nv),
    )
    tl.store(
        fp8_rot_base + half_offset * 2 + 1,
        tl.div_rn(r_odd, index_q_scale).to(tl.float8e4nv),
    )

    # FP8 weight-fold contract:
    #   index_weights_out = index_weights * q_scale * softmax_scale * head_scale
    # The per-token-per-head q_scale (fp32) IS folded into the output weights
    # here because FP8 Q is stored WITHOUT a companion scale tensor — the
    # downstream fp8_fp4_mqa_logits/fp8_fp4_paged_mqa_logits kernels use `weights` to
    # apply per-token Q scale inline. See the MXFP4 kernel below for the
    # contrasting convention (scales live with the Q values, weights are NOT
    # q-scaled).
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    )
    index_weights = index_weights.to(tl.float32)
    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    # Index weights
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    """Fused RoPE + quantize Q for the sparse indexer.

    Weight-fold semantics (important — the two paths differ):

    FP8 path (use_fp4=False, default):
        q_fp8      : (T, H, HEAD_DIM) float8_e4m3fn, per-token-per-head
                     scalar scale (NOT stored — folded into weights below)
        weights_out = weights * q_scale * softmax_scale * head_scale
        Rationale: a single per-token q_scale is a scalar the downstream FP8
        logits kernel would otherwise multiply in. Folding it into `weights`
        avoids emitting a separate tensor and is free for the logits kernel.

    MXFP4 path (use_fp4=True):
        q_packed   : (T, H, HEAD_DIM // 2) uint8 (2 E2M1 nibbles per byte)
        q_scale    : (T, H, HEAD_DIM // MXFP4_BLOCK_SIZE) uint8 ue8m0 bytes
        weights_out = weights * softmax_scale * head_scale
        Rationale: MXFP4 has PER-BLOCK (32-element) scales that live with
        the Q values — they cannot be folded into a per-token weight
        scalar, so `weights` carries only the softmax and head scales.

    Returns (q_quant, weights_out) where q_quant is either a Tensor (FP8) or
    a (values, scales) tuple (MXFP4). This matches the union type accepted
    by `SparseAttnIndexer.forward_*`.
    """
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    if use_fp4:
        assert index_q_head_dim % MXFP4_BLOCK_SIZE == 0, (
            f"head_dim={index_q_head_dim} must be a multiple of MXFP4 block "
            f"size {MXFP4_BLOCK_SIZE}"
        )
        num_scale_blocks = index_q_head_dim // MXFP4_BLOCK_SIZE
        index_q_packed = index_q.new_empty(
            (num_tokens, num_index_q_heads, index_q_head_dim // 2),
            dtype=torch.uint8,
        )
        index_q_scale = index_q.new_empty(
            (num_tokens, num_index_q_heads, num_scale_blocks),
            dtype=torch.uint8,
        )
        compiled = _compile_indexer_q_mxfp4(
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
        # Values stay uint8 (2 E2M1 nibbles per byte). Scales are 4 ue8m0
        # bytes per (token, head) reinterpreted as one int32, then squeezed
        # from (T, H, 1) to (T, H) to match DeepGEMM's expected q_sf rank
        # (prefill wants 2-D (seq_len, num_heads); decode reshapes this to
        # 3-D (batch, next_n, num_heads)).
        return (
            index_q_packed,
            index_q_scale.view(torch.int32).squeeze(-1),
        ), index_weights_out

    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
    _fused_indexer_q_rope_quant_kernel[(num_tokens, num_index_q_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
        index_q_fp8,
        index_q_fp8.stride(0),
        index_q_fp8.stride(1),
        index_q_head_dim,
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
        index_weights_out.stride(0),
        num_warps=1,  # TODO: Tune this
    )
    return index_q_fp8, index_weights_out


@dsl_user_op
def _fp32x2_to_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _bf16x2_to_fp32(data: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(data).ir_value(loc=loc, ip=ip)],
        "shl.b32 $0, $2, 16;\n\tand.b32 $1, $2, 0xFFFF0000;\n",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _bf16x2_abs(a: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip)],
            "abs.bf16x2 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
            ],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fp32x8_to_fp4x8(
    vals: cute.Tensor,
    offset: cutlass.Constexpr[int],
    *,
    loc=None,
    ip=None,
) -> Uint32:
    # Pack eight scaled FP32 values into four E2M1x2 bytes, returned as one b32.
    operands = [Float32(vals[offset + i]).ir_value(loc=loc, ip=ip) for i in range(8)]
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            operands,
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
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


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
    out_dtype: cutlass.Constexpr[type[cutlass.Numeric]] = Uint32,
    *,
    loc=None,
    ip=None,
) -> cute.TensorSSA:
    if const_expr(out_dtype is Float32):
        mlir_ty = T.f32()
        ptx_ty = "f32"
        constraint = "=f"
    elif const_expr(out_dtype is Uint32):
        mlir_ty = T.i32()
        ptx_ty = "b32"
        constraint = "=r"
    else:
        raise TypeError(f"_ldg_vec only supports Uint32 and Float32, got {out_dtype}")

    # compute base pointer
    base_ptr = (
        tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)
    ).toint()

    # build PTX string
    ptx_str = f"ld.global{modifier}.v{vec_size}.{ptx_ty}"
    ptx_str += "{" + ", ".join(f"${i}" for i in range(vec_size)) + "}"
    ptx_str += f", [${vec_size}];"
    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * vec_size),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        ptx_str,
        ",".join([constraint] * vec_size + ["l"]),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    vec = vector.from_elements(
        ir.VectorType.get([vec_size], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(vec_size)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, vec_size, out_dtype)


@dsl_user_op
def _stg_u32xN(
    tensor: cute.Tensor,
    coord: cute.Coord,
    values: cute.Tensor,
    vec_size: cutlass.Constexpr[int],
    modifier: cutlass.Constexpr[str] = "",
    *,
    loc=None,
    ip=None,
) -> None:
    base_ptr = (
        tensor.iterator + cute.crd2idx(coord, tensor.layout, loc=loc, ip=ip)
    ).toint()
    value_operands = ", ".join(f"${i + 1}" for i in range(vec_size))
    llvm.inline_asm(
        None,
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)]
        + [Uint32(values[i]).ir_value(loc=loc, ip=ip) for i in range(vec_size)],
        f"st.global{modifier}.v{vec_size}.u32 [$0], {{{value_operands}}};",
        ",".join(["l"] + ["r"] * vec_size),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
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
        ).launch(grid=grid, block=[self.tb_size, 1, 1])

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
        _q_bits = _ldg_vec(
            q, (token_id, head_id, elem_base), 8, ".relaxed.cta.L1::no_allocate"
        )
        q_bits = cute.make_rmem_tensor(8, Uint32)
        q_bits.store(_q_bits)  # copy to make it mutable

        # RoPE applies only to the trailing rope_dim values. We keep the rounded
        # BF16 result in q_bits so the later amax and quantization see BF16.
        # cos_sin_cache layout: [max_pos, rope_dim]
        if elem_base >= self.nope_dim:
            pos = positions[token_id]
            rope_idx = (elem_base - self.nope_dim) // 2
            if const_expr(self.cos_sin_dtype is Float32):
                cos_vals = _ldg_vec(
                    cos_sin_cache,
                    (pos, rope_idx),
                    8,
                    out_dtype=Float32,
                )
                sin_vals = _ldg_vec(
                    cos_sin_cache,
                    (pos, self.rope_dim // 2 + rope_idx),
                    8,
                    out_dtype=Float32,
                )
            else:
                # Each BF16 cache load lane contains two adjacent values.
                cos_loaded = _ldg_vec(cos_sin_cache, (pos, rope_idx), 4)
                sin_loaded = _ldg_vec(
                    cos_sin_cache,
                    (pos, self.rope_dim // 2 + rope_idx),
                    4,
                )
                cos_vals = cute.make_rmem_tensor(8, Float32)
                sin_vals = cute.make_rmem_tensor(8, Float32)
                for i in cutlass.range_constexpr(4):
                    cos_vals[i * 2], cos_vals[i * 2 + 1] = _bf16x2_to_fp32(
                        cos_loaded[i]
                    )
                    sin_vals[i * 2], sin_vals[i * 2 + 1] = _bf16x2_to_fp32(
                        sin_loaded[i]
                    )

            for i in cutlass.range_constexpr(8):
                q0, q1 = _bf16x2_to_fp32(q_bits[i])
                cos = cos_vals[i]
                sin = sin_vals[i]
                rot0 = q0 * cos - q1 * sin
                rot1 = q0 * sin + q1 * cos
                # convert back to BF16 to match numerics
                q_bits[i] = _fp32x2_to_bf16x2(rot0, rot1)

        # compute amax in packed bf16x2 to save instructions
        # Each thread holds 16 elems. Two adjacent threads form one 32-elem
        # MXFP4 block, so a width-2 shuffle gives the block amax.
        local_amax = _bf16x2_abs(q_bits[0])
        for i in cutlass.range_constexpr(1, 8):
            local_amax = _bf16x2_max(local_amax, _bf16x2_abs(q_bits[i]))
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
            vals[i * 2], vals[i * 2 + 1] = _bf16x2_to_fp32(q_bits[i])
            vals[i * 2] = vals[i * 2] * inv_fp4_scale
            vals[i * 2 + 1] = vals[i * 2 + 1] * inv_fp4_scale

        # pack to FP4
        packed = cute.make_rmem_tensor(2, Uint32)
        packed[0] = _fp32x8_to_fp4x8(vals, 0)
        packed[1] = _fp32x8_to_fp4x8(vals, 8)
        # Each thread writes the eight packed bytes corresponding to its 16 Q values.
        _stg_u32xN(q_fp4, (token_id, head_id, elem_base // 2), packed, 2, ".cs")

        # Weight scaling is independent of the Q subwarp work. The first
        # num_tokens * num_heads logical threads cover one weight each.
        if global_tid < num_token_heads:
            weight_token_id = global_tid // self.num_heads
            weight_head_id = global_tid - weight_token_id * self.num_heads
            weights_out[weight_token_id, weight_head_id] = (
                weights[weight_token_id, weight_head_id].to(Float32) * scale
            )


@cache
def _compile_indexer_q_mxfp4(
    head_dim: int, rope_dim: int, num_heads: int, cos_sin_dtype: type[cutlass.Numeric]
):
    num_tokens = cute.sym_int()
    max_pos = cute.sym_int()

    q = make_fake_tensor(BFloat16, (num_tokens, num_heads, head_dim), divisibility=8)
    positions = make_fake_tensor(Int64, (num_tokens,), divisibility=1)
    cos_sin_cache = make_fake_tensor(cos_sin_dtype, (max_pos, rope_dim), divisibility=8)
    weights = make_fake_tensor(BFloat16, (num_tokens, num_heads), divisibility=8)
    q_fp4 = make_fake_tensor(
        Uint8, (num_tokens, num_heads, head_dim // 2), divisibility=16
    )
    q_scale = make_fake_tensor(
        Uint8,
        (num_tokens, num_heads, head_dim // MXFP4_BLOCK_SIZE),
        divisibility=4,
    )
    weights_out = make_fake_tensor(Float32, (num_tokens, num_heads), divisibility=4)

    kernel = IndexerQMxFp4Kernel(head_dim, rope_dim, num_heads, cos_sin_dtype)
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
        options="--enable-tvm-ffi",
    )
