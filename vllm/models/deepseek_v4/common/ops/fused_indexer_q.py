# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_cutedsl

# MXFP4: 32 elements per block, packed 2 nibbles per byte, ue8m0 block scale.
MXFP4_BLOCK_SIZE = 32


@triton.jit
def _hadamard_stage(x, N: tl.constexpr, S: tl.constexpr):
    a, b = tl.split(tl.trans(tl.reshape(x, (N // (2 * S), 2, S)), 0, 2, 1))
    return tl.reshape(tl.trans(tl.join(a + b, a - b), 0, 2, 1), (N,))


@triton.jit
def _hadamard_rotate(x, N: tl.constexpr):
    """Sylvester Hadamard rotation y = (H @ x) * N**-0.5 as a matrix-free
    fp32 butterfly: log2(N) fixed-order stages of single fp32 add/subs, so
    the result is bitwise reproducible with elementwise torch ops."""
    tl.static_assert(N == 128)
    x = _hadamard_stage(x, N, 64)
    x = _hadamard_stage(x, N, 32)
    x = _hadamard_stage(x, N, 16)
    x = _hadamard_stage(x, N, 8)
    x = _hadamard_stage(x, N, 4)
    x = _hadamard_stage(x, N, 2)
    x = _hadamard_stage(x, N, 1)
    return x * 0.08838834764831845  # 128**-0.5


@triton.jit
def _rope_hadamard_full(
    x,
    cos_sin_ptr,
    cos_sin_stride,
    pos,
    HEAD_DIM: tl.constexpr,
    ROT_DIM: tl.constexpr,
    ROTATE: tl.constexpr = True,
):
    """Full-width GPT-J RoPE (nope pairs masked to identity) with pinned FMA
    contraction, a bf16 roundtrip matching the unfused reference numerics,
    and — when ROTATE — the Hadamard rotation. Returns the full-width fp32
    vector. Shared by all four indexer quant kernels (Q fp8/mxfp4, K
    fp8/mxfp4) so the RoPE/roundtrip/rotation semantics stay bit-identical.
    """
    HALF_ROT: tl.constexpr = ROT_DIM // 2
    NOPE_PAIRS: tl.constexpr = (HEAD_DIM - ROT_DIM) // 2
    pairs = tl.reshape(x, (HEAD_DIM // 2, 2))
    ev, od = tl.split(pairs)  # [HEAD_DIM // 2] fp32 each
    pidx = tl.arange(0, HEAD_DIM // 2)
    is_rope = pidx >= NOPE_PAIRS
    cs_i = tl.maximum(pidx - NOPE_PAIRS, 0)
    row_ptr = cos_sin_ptr + pos * cos_sin_stride
    cos_v = tl.load(row_ptr + cs_i, mask=is_rope, other=1.0).to(tl.float32)
    sin_v = tl.load(row_ptr + HALF_ROT + cs_i, mask=is_rope, other=0.0).to(tl.float32)
    # Pinned FMA contraction, matching the fused (NVCC/HIP) contraction of
    # the unfused rotary_embedding flow on every platform.
    new_ev = tl.fma(ev, cos_v, -(od * sin_v))
    new_od = tl.fma(od, cos_v, ev * sin_v)
    x_rope = tl.interleave(new_ev, new_od)  # [HEAD_DIM] fp32
    # Match reference numerics: fp32 → bf16 → fp32 before rotation/quant.
    # Post-interleave; equivalent to roundtripping the halves since the
    # interleave is a permutation and the roundtrip is elementwise.
    x_rope = x_rope.to(tl.bfloat16).to(tl.float32)
    if ROTATE:
        # Hadamard rotation (reference rotate_activation) after RoPE, before
        # quantization.
        tl.static_assert(HEAD_DIM == 128)
        tl.static_assert(ROT_DIM == 64)
        x_rope = _hadamard_rotate(x_rope, HEAD_DIM)
    return x_rope


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
    FP8_MAX: tl.constexpr = 448.0,
    USE_FNUZ: tl.constexpr = False,
    HADAMARD: tl.constexpr = False,
):
    # Layout matches the unfused reference (DeepseekV4ScalingRotaryEmbedding
    # + per_token_group_quant_fp8): GPT-J interleaved RoPE applied to the
    # LAST rope_dim dims of each head; the leading [0, NOPE_DIM) is passed
    # through unchanged. With HADAMARD (production indexer dims 128/64), the
    # full vector is Hadamard-rotated after a bf16 roundtrip and before
    # quantization, matching the reference rotate_activation.
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
    if HADAMARD:
        # Full-width flow: rope with the nope region masked to identity,
        # bf16 roundtrip, Hadamard rotation, full-width ue8m0 FP8 quant.
        full_idx = tl.arange(0, INDEX_Q_HEAD_DIM)
        x_full = tl.load(base_ptr + full_idx).to(tl.float32)
        x_h = _rope_hadamard_full(
            x_full,
            index_q_cos_sin_ptr,
            index_q_cos_sin_stride,
            pos,
            INDEX_Q_HEAD_DIM,
            INDEX_Q_ROT_DIM,
        )
        amax = tl.max(tl.abs(x_h))
    else:
        # Interleaved (GPT-J) RoPE on dims [NOPE_DIM, HEAD_DIM):
        #   even = q[NOPE_DIM + 2*i],  odd = q[NOPE_DIM + 2*i + 1]
        cos, sin = _get_cos_sin(
            index_q_cos_sin_ptr,
            index_q_cos_sin_stride,
            pos,
            INDEX_Q_HALF_ROT_DIM,
        )
        half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
        rot_base = base_ptr + INDEX_Q_NOPE_DIM
        x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
        x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
        # Pinned FMA contraction, matching the fused (NVCC/HIP) contraction
        # of the unfused rotary_embedding flow on every platform.
        r_even = tl.fma(x_even, cos, -(x_odd * sin))
        r_odd = tl.fma(x_odd, cos, x_even * sin)
        # Match reference numerics: fp32 → bf16 → fp32 before the ue8m0
        # absmax. Same pattern as the K-side kernel.
        r_even = r_even.to(tl.bfloat16).to(tl.float32)
        r_odd = r_odd.to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
        if INDEX_Q_NOPE_DIM > 0:
            nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
            x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
    index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), FP8_MAX)
    index_q_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))

    # Store quantized values to index_q_fp8. FNUZ (e4m3fnuz) on gfx942,
    # OCP (e4m3fn) elsewhere -- matches the K cache.
    fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv
    fp8_base_ptr = (
        index_q_fp8_ptr + tok_idx * index_q_fp8_stride0 + head_idx * index_q_fp8_stride1
    )
    if HADAMARD:
        tl.store(
            fp8_base_ptr + full_idx,
            tl.div_rn(x_h, index_q_scale).to(fp8_dtype),
        )
    else:
        if INDEX_Q_NOPE_DIM > 0:
            tl.store(
                fp8_base_ptr + nope_offset,
                tl.div_rn(x_nope, index_q_scale).to(fp8_dtype),
            )
        fp8_rot_base = fp8_base_ptr + INDEX_Q_NOPE_DIM
        tl.store(
            fp8_rot_base + half_offset * 2,
            tl.div_rn(r_even, index_q_scale).to(fp8_dtype),
        )
        tl.store(
            fp8_rot_base + half_offset * 2 + 1,
            tl.div_rn(r_odd, index_q_scale).to(fp8_dtype),
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


@triton.jit
def _fused_indexer_q_rope_mxfp4_kernel(
    pos_ptr,
    # Index Q RoPE input (fp/bf16)
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # MXFP4 Q outputs
    index_q_mxfp4_ptr,  # uint8, (T, H, HEAD_DIM // 2)
    index_q_mxfp4_stride0,
    index_q_mxfp4_stride1,
    index_q_scale_ptr,  # uint8 ue8m0, (T, H, HEAD_DIM // BLOCK)
    index_q_scale_stride0,
    index_q_scale_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    MXFP4_BLOCK: tl.constexpr,
    # Weights (NO per-token q_scale fold for MXFP4; per-block scales stay
    # with the Q values in the output scale tensor).
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
    HADAMARD: tl.constexpr = False,
):
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    HALF_BLOCK: tl.constexpr = MXFP4_BLOCK // 2
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)
    tl.static_assert(MXFP4_BLOCK % 2 == 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)

    q_base = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
    out_base = (
        index_q_mxfp4_ptr
        + tok_idx * index_q_mxfp4_stride0
        + head_idx * index_q_mxfp4_stride1
    )
    scale_base = (
        index_q_scale_ptr
        + tok_idx * index_q_scale_stride0
        + head_idx * index_q_scale_stride1
    )

    if HADAMARD:
        # Full-width flow: GPT-J RoPE (nope masked to identity) → bf16
        # roundtrip → Hadamard rotation → per-32-block MXFP4 quant.
        full_idx = tl.arange(0, INDEX_Q_HEAD_DIM)
        x_full = tl.load(q_base + full_idx).to(tl.float32)
        x_h = _rope_hadamard_full(
            x_full,
            index_q_cos_sin_ptr,
            index_q_cos_sin_stride,
            pos,
            INDEX_Q_HEAD_DIM,
            INDEX_Q_ROT_DIM,
        )
        # Per-block MXFP4 quant on the rotated vector, vectorized over the
        # blocks: each block of MXFP4_BLOCK consecutive elements is
        # HALF_BLOCK (even, odd) pairs.
        NQB: tl.constexpr = INDEX_Q_HEAD_DIM // MXFP4_BLOCK
        x2 = tl.reshape(x_h, (INDEX_Q_HEAD_DIM // 2, 2))
        lo, hi = tl.split(x2)  # lo[k]=x_h[2k], hi[k]=x_h[2k+1]
        lo2 = tl.reshape(lo, (NQB, HALF_BLOCK))
        hi2 = tl.reshape(hi, (NQB, HALF_BLOCK))
        amax = tl.maximum(tl.max(tl.abs(lo2), 1), tl.max(tl.abs(hi2), 1))
        # 6 * 2^-126 is from https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py#L163
        amax = tl.maximum(amax, 6.0 * (2**-126))
        # ue8m0 block scale: 2^ceil(log2(amax/6.0)).
        log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
        log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
        inv_scale = tl.exp2(-log2_ratio)
        ue8m0 = (log2_ratio + 127.0).to(tl.uint8)  # [NQB]
        inv_col = tl.reshape(inv_scale, (NQB, 1))
        packed = _fp32x2_to_fp4x2(lo2 * inv_col, hi2 * inv_col)
        packed_flat = tl.reshape(packed, (INDEX_Q_HEAD_DIM // 2,))
        tl.store(out_base + tl.arange(0, INDEX_Q_HEAD_DIM // 2), packed_flat)
        tl.store(scale_base + tl.arange(0, NQB), ue8m0)
    else:
        NUM_NOPE_BLOCKS: tl.constexpr = INDEX_Q_NOPE_DIM // MXFP4_BLOCK
        NUM_ROPE_BLOCKS: tl.constexpr = INDEX_Q_ROT_DIM // MXFP4_BLOCK
        tl.static_assert(INDEX_Q_NOPE_DIM % MXFP4_BLOCK == 0)
        tl.static_assert(INDEX_Q_ROT_DIM % MXFP4_BLOCK == 0)
        half_off = tl.arange(0, HALF_BLOCK)

        # ---- NoPE blocks: direct load, pair as (even, odd) values ----
        for b in tl.static_range(NUM_NOPE_BLOCKS):
            base = b * MXFP4_BLOCK
            x_lo = tl.load(q_base + base + half_off * 2).to(tl.float32)
            x_hi = tl.load(q_base + base + half_off * 2 + 1).to(tl.float32)
            packed, ue8m0 = _quantize_mxfp4_pair(x_lo, x_hi)
            tl.store(out_base + base // 2 + half_off, packed)
            tl.store(scale_base + b, ue8m0)

        # ---- RoPE blocks: GPT-J interleaved RoPE on the block's 16 pairs,
        # then quantize. Each block covers HALF_BLOCK cos/sin pairs. ----
        rot_q_base = q_base + INDEX_Q_NOPE_DIM
        for b in tl.static_range(NUM_ROPE_BLOCKS):
            pair_off = b * HALF_BLOCK + half_off  # indices in [0, HALF_ROT_DIM)
            cos_b = tl.load(
                index_q_cos_sin_ptr + pos * index_q_cos_sin_stride + pair_off
            ).to(tl.float32)
            sin_b = tl.load(
                index_q_cos_sin_ptr
                + pos * index_q_cos_sin_stride
                + pair_off
                + INDEX_Q_HALF_ROT_DIM
            ).to(tl.float32)
            x_even = tl.load(rot_q_base + pair_off * 2).to(tl.float32)
            x_odd = tl.load(rot_q_base + pair_off * 2 + 1).to(tl.float32)
            # Pinned FMA contraction, matching the fused (NVCC/HIP)
            # contraction of the unfused rotary_embedding flow.
            r_even = tl.fma(x_even, cos_b, -(x_odd * sin_b))
            r_odd = tl.fma(x_odd, cos_b, x_even * sin_b)
            # bf16 roundtrip for parity with the FP8 kernel / reference.
            r_even = r_even.to(tl.bfloat16).to(tl.float32)
            r_odd = r_odd.to(tl.bfloat16).to(tl.float32)
            packed, ue8m0 = _quantize_mxfp4_pair(r_even, r_odd)
            rope_byte_off = (INDEX_Q_NOPE_DIM + b * MXFP4_BLOCK) // 2
            tl.store(out_base + rope_byte_off + half_off, packed)
            tl.store(scale_base + NUM_NOPE_BLOCKS + b, ue8m0)

    # MXFP4 weight-fold contract:
    #   index_weights_out = index_weights * softmax_scale * head_scale
    # NOTE: q_scale is NOT folded here (contrast with the FP8 kernel above).
    # MXFP4 Q emits a separate ue8m0 scale tensor of shape
    # (T, H, HEAD_DIM // MXFP4_BLOCK) alongside the packed values, so each
    # per-block scale is applied by the downstream MXFP4 logits kernel when
    # dequantizing Q — there is no per-token scalar to fold into `weights`.
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    ).to(tl.float32)
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

    For the production indexer dims (head_dim=128, rope_dim=64) a Sylvester
    Hadamard rotation (scaled by head_dim**-0.5) follows RoPE and precedes
    quantization, matching the reference implementation's rotate_activation.
    Being orthogonal, it preserves indexer QK dot products. Other dims take
    the original unrotated path.

    Weight-fold semantics (important — the two paths differ):

    FP8 path (use_fp4=False, default):
        q_fp8      : (T, H, HEAD_DIM) platform fp8 (e4m3fnuz on gfx942,
                     e4m3fn elsewhere); per-token-per-head scalar scale
                     (NOT stored — folded into weights below)
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
    # The Hadamard rotation applies to the production indexer dims only
    # (head_dim=128, rope_dim=64); other dims take the original path.
    hadamard = index_q_head_dim == 128 and index_q_cos_sin_cache.shape[-1] == 64

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    if use_fp4:
        assert index_q_head_dim % MXFP4_BLOCK_SIZE == 0, (
            f"head_dim={index_q_head_dim} must be a multiple of MXFP4 block "
            f"size {MXFP4_BLOCK_SIZE}"
        )
        num_scale_blocks = index_q_head_dim // MXFP4_BLOCK_SIZE
        index_q_packed = torch.empty(
            (num_tokens, num_index_q_heads, index_q_head_dim // 2),
            dtype=torch.uint8,
            device=index_q.device,
        )
        index_q_scale = torch.empty(
            (num_tokens, num_index_q_heads, num_scale_blocks),
            dtype=torch.uint8,
            device=index_q.device,
        )
        if not hadamard and has_cutedsl():
            # lazily import, otherwise some tests fail due to CUDA driver init failure.
            from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
                fused_indexer_q_rope_quant_mxfp4_cutedsl,
            )

            fused_indexer_q_rope_quant_mxfp4_cutedsl(
                positions,
                index_q,
                index_q_cos_sin_cache,
                index_weights,
                index_weights_softmax_scale,
                index_weights_head_scale,
                index_q_packed,
                index_q_scale,
                index_weights_out,
            )
        elif current_platform.is_xpu():
            torch.ops.vllm.xpu_deepseek_fused_indexer_q_rope_mxfp4(
                index_q,
                positions,
                index_q_cos_sin_cache,
                index_weights,
                index_weights_softmax_scale,
                index_weights_head_scale,
                index_q_packed,
                index_q_scale,
                index_weights_out,
            )
        else:
            # NOTE: the cutedsl indexer-Q kernels predate the Hadamard
            # rotation and would skip it; when the rotation applies
            # (head_dim=128, rope_dim=64) always launch the Triton kernel
            # until they implement the rotation.
            _fused_indexer_q_rope_mxfp4_kernel[(num_tokens, num_index_q_heads)](
                positions,
                index_q,
                index_q.stride(0),
                index_q.stride(1),
                index_q_cos_sin_cache,
                index_q_cos_sin_cache.stride(0),
                index_q_cos_sin_cache.shape[-1] // 2,
                index_q_packed,
                index_q_packed.stride(0),
                index_q_packed.stride(1),
                index_q_scale,
                index_q_scale.stride(0),
                index_q_scale.stride(1),
                index_q_head_dim,
                MXFP4_BLOCK_SIZE,
                index_weights,
                index_weights.stride(0),
                index_weights_softmax_scale,
                index_weights_head_scale,
                index_weights_out,
                index_weights_out.stride(0),
                HADAMARD=hadamard,
                num_warps=1,
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

    fp8_dtype = current_platform.fp8_dtype()
    use_fnuz = fp8_dtype == torch.float8_e4m3fnuz
    fp8_max = 224.0 if use_fnuz else 448.0
    index_q_fp8 = torch.empty_like(index_q, dtype=fp8_dtype)
    if not hadamard and has_cutedsl():
        # lazily import, otherwise some tests fail due to CUDA driver init failure.
        from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
            fused_indexer_q_rope_quant_fp8_cutedsl,
        )

        fused_indexer_q_rope_quant_fp8_cutedsl(
            positions,
            index_q,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            index_q_fp8,
            index_weights_out,
        )
    elif current_platform.is_xpu():
        torch.ops.vllm.xpu_deepseek_fused_indexer_q_rope_fp8(
            index_q,
            positions,
            index_q_cos_sin_cache,
            index_weights,
            index_weights_softmax_scale,
            index_weights_head_scale,
            index_q_fp8,
            index_weights_out,
        )
    else:
        # NOTE: the cutedsl indexer-Q kernels predate the Hadamard rotation
        # and would skip it; when the rotation applies (head_dim=128,
        # rope_dim=64) always launch the Triton kernel until they implement
        # the rotation.
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
            FP8_MAX=fp8_max,
            USE_FNUZ=use_fnuz,
            HADAMARD=hadamard,
            num_warps=1,
        )
    return index_q_fp8, index_weights_out
