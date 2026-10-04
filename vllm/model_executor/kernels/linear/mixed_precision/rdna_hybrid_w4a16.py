# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid W4A16 kernel: Triton for prefill, HIP skinny for decode.

Routes based on batch size M:
  M <= MAX_SKINNY_BATCH_SIZE: HIP skinny GEMM (wvSplitK_int4_g)
  M > MAX_SKINNY_BATCH_SIZE:  Triton W4A16 fused dequant GEMM

Stores the weights ONCE as int8 [N, K//2] (ExLlama shuffle packed). Both
paths read this single buffer: the HIP skinny kernel uses it directly, and
the triton kernel reinterprets it as int32 [N, K//8] via a view (and
transposes tiles in-register). No dual weight storage.
"""

from contextlib import nullcontext

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    permute_param_layout_,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

SUPPORTED_GROUP_SIZES = [32, 64, 128]


def _on_gfx12x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx12x

    return on_gfx12x()


def _on_gfx1x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1x

    return on_gfx1x()


def _on_gfx1151() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1151

    return on_gfx1151()


# Maximum batch size M for the HIP skinny kernel path (C++ supports N_in
# up to 5).  When M is below this AND K*M fits in LDS, the skinny kernel is
# used; otherwise the Triton prefill path handles the GEMM.
MAX_SKINNY_BATCH_SIZE = 5
# 64 KiB per-workgroup LDS limit expressed in fp16 elements.
# (AMD RDNA has 128 KiB total LDS per CU, but 64 KiB per workgroup.)
LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements


# ---------------------------------------------------------------------------
# Triton kernel for the prefill path (reads skinny-format weights [N, K//8])
# ---------------------------------------------------------------------------


@triton.constexpr_function
def _target_is_gfx11() -> bool:
    """True when the kernel is being compiled for RDNA3 (gfx11)."""
    target = tl.target_info.current_target()
    if target is None or target.backend != "hip":
        return False
    return str(target.arch).startswith("gfx11")


@triton.jit
def _int4_pair_to_fp16x2(x):
    """Unpack two packed int4 nibbles into a uint32 holding two fp16 lanes,
    each equal to 1024 + nibble, with one ``v_and_or_b32``
    (``(x & 0x000F000F) | 0x64006400``).

    OR-ing a 4-bit nibble into the low mantissa of fp16 1024.0 (0x6400)
    bitcasts to exactly 1024+n. Doing it on a full 32-bit lane dequants two
    nibbles per instruction, vs the scalar v_and_b16 + v_or_b16 pair Triton
    emits from the elementwise form.
    """
    mask = tl.full(x.shape, 0x000F000F, tl.int32)
    return tl.inline_asm_elementwise(
        asm="v_and_or_b32 $0, $1, $2, 0x64006400",
        constraints="=v,v,v",
        args=[x, mask],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _triton_w4a16_skinny_fmt_kernel(
    # Pointers
    a_ptr,  # [M, K]  fp16/bf16 activations
    b_ptr,  # [N, K//8]  int32 packed (ExLlama shuffle, K is packed dim)
    scales_ptr,  # [N, K//G]  fp16/bf16 scales (skinny layout)
    zp_ptr,  # [N//8, K//G]  int32 zero-points (when HAS_ZP=True)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    K8,  # K // 8
    num_groups,  # K // group_size
    stride_bn,  # b_ptr row stride; >= K8, the rows may be padded
    stride_am,  # a_ptr row stride in elements; >= K, rows may be padded
    # Quantization parameters
    group_size,
    ZP_BIAS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused W4A16 GEMM reading weights from skinny format [N, K//8].

    B is stored as [N, K//8] int32 using ExLlama shuffle packing:
      each int32 packs 8 K-values with interleave [0,2,4,6,1,3,5,7]:
        packed = val[0] | (val[2]<<4) | (val[4]<<8) | (val[6]<<12)
               | (val[1]<<16) | (val[3]<<20) | (val[5]<<24) | (val[7]<<28)

    Scales are [N, K//G] (skinny layout, NOT transposed).
    When HAS_ZP=True, zp_ptr holds [N//8, K//G] int32 with row n's raw
    zero-point at word[n//8] bits 4*(n%8), and dequant is
    (nibble - zp_raw) * scale.
    When HAS_ZP=False, only the constant ZP_BIAS is subtracted (symmetric).

    On the fp16 path the nibble arrives as ``b_raw`` = 1024 + nibble (the
    magic-constant unpack), so the subtrahend absorbs the 1024: the arithmetic
    is unchanged and every intermediate stays exact, since fp16 represents every
    integer below 2048.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ExLlama unshuffle shifts: shift[j] = (j//2)*4 + (j%2)*16
    # For 8 values: [0, 16, 4, 20, 8, 24, 12, 28]
    exllama_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    shifts_1d = tl.reshape(
        tl.broadcast_to(exllama_shifts_row[None, :], (BLOCK_K // 8, 8)),
        (BLOCK_K,),
    )
    shifts_full = tl.broadcast_to(shifts_1d[None, :], (BLOCK_N, BLOCK_K))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k8[None, :]
        mask_b = (offs_n[:, None] < N) & (offs_k8[None, :] < K8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        if a.dtype == tl.float16 and _target_is_gfx11():
            # The ExLlama int32 holds the paired nibbles val[2p] @ bits[4p:4p+4]
            # and val[2p+1] @ bits[16+4p:20+4p], so for pre-shift 4p (p=0..3),
            #   (x >> 4p) & 0x000F000F | 0x64006400
            # is one v_and_or_b32 producing a half2 = (1024+val[2p],
            # 1024+val[2p+1]) in K order (signed shift is fine: the sign fill
            # lands above bit 20 and is masked out). The interleave(lo, hi) lays
            # b_raw out as half2 so the downstream affine also packs into
            # v_pk_fma_f16. The dequant inner loop is VALU-issue-bound on gfx11,
            # so this ~halves the dequant instruction count per WMMA.
            shifts4 = (tl.arange(0, 4) * 4)[None, None, :]
            bp_shift = tl.reshape(
                b_packed[:, :, None] >> shifts4, (BLOCK_N, BLOCK_K // 2)
            )
            packed_hl = _int4_pair_to_fp16x2(bp_shift)  # u32 half2: 1024+nibble
            lo = (packed_hl & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
            hi = (packed_hl >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
            b_raw = tl.interleave(lo, hi)  # [BLOCK_N, BLOCK_K] fp16 = 1024+nibble
        else:
            # ExLlama unshuffle: replicate each int32 8x then per-lane shift+mask.
            b = tl.interleave(b_packed, b_packed)
            b = tl.interleave(b, b)
            b = tl.interleave(b, b)
            b = (b >> shifts_full) & 0xF  # [BLOCK_N, BLOCK_K]

        group_idx = (k_start * BLOCK_K) // group_size
        scale_ptrs = scales_ptr + offs_n * num_groups + group_idx
        scale_mask = offs_n < N
        scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0)

        if HAS_ZP:
            # Zero points stay in their packed 4-bit form: row n's nibble lives
            # at word[n//8], bits 4*(n%8).
            zp_ptrs = zp_ptr + (offs_n // 8) * num_groups + group_idx
            zp_word = tl.load(zp_ptrs, mask=scale_mask, other=0)
            zp_raw = (zp_word >> (4 * (offs_n % 8))) & 0xF

        if a.dtype == tl.float16:
            # The magic unpack yields b_raw = 1024 + nibble, so fold the 1024
            # into the subtrahend: (b_raw - (1024 + zp)) == (nibble - zp),
            # exactly, and the multiply that follows rounds once as before.
            # 1024..1039 are all exact in fp16, so the cast cannot round.
            if not _target_is_gfx11():
                b_raw = (b | 0x6400).to(tl.uint16).to(tl.float16, bitcast=True)
            c1024 = tl.full((), 1024.0, tl.float16)
            if HAS_ZP:
                zp_off = (c1024 + zp_raw.to(tl.float16))[:, None]
                b_fp = (b_raw - zp_off) * scales[:, None]
            else:
                b_fp = (b_raw - (c1024 + ZP_BIAS)) * scales[:, None]
        else:
            # bf16 keeps the subtract in the int domain and casts once: zp_raw is
            # an int32 nibble, so casting b first would promote the whole
            # expression to fp32 and break the tl.dot dtype match.
            if HAS_ZP:
                b_fp = (b - zp_raw[:, None]).to(scales.dtype) * scales[:, None]
            else:
                b_fp = (b - ZP_BIAS).to(scales.dtype) * scales[:, None]

        b_fp_t = tl.trans(b_fp)
        accumulator += tl.dot(a, b_fp_t, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


# Per-shape (group_size, K, N) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps,
# num_stages) tile-config overrides for prefill (M <= 128) on gfx1151. Applies
# to the SCALAR (bf16) dequant path only; the packed fp16 path is tuned by the
# ladder in _select_skinny_gfx1151_config and needs no per-shape entries.
# Picked by sweeping benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py + a
# per-config sweep script; only added when better than the generic heuristic
# by > 20% at M=128. Re-run benchmarks after edits.
_GFX1151_BF16_PREFILL_OVERRIDES: dict[
    tuple[int, int, int], tuple[int, int, int, int, int]
] = {
    # SmolLM2-1.7B-Instruct-AWQ (gs=32, K=2048; gs forces BLOCK_K to 32 so
    # widen BLOCK_M and let Triton pipeline 4 stages to amortize the small
    # K-tile).
    (32, 2048, 6144): (128, 32, 32, 4, 4),  # qkv_proj
    (32, 2048, 2048): (128, 32, 32, 4, 4),  # o_proj
    (32, 2048, 16384): (128, 32, 32, 4, 4),  # gate_up_proj
    (32, 8192, 2048): (128, 64, 32, 8, 2),  # down_proj
    # Qwen3-8B-quantized.w4a16 (gs=128, K=4096 / 12288). For these K's a
    # full BLOCK_K=group_size with no software pipelining beats the generic
    # 64x64x64 — single-stage keeps register pressure down.
    (128, 4096, 6144): (128, 64, 128, 8, 1),  # qkv_proj
    (128, 4096, 4096): (128, 32, 128, 4, 1),  # o_proj
    (128, 4096, 24576): (64, 32, 128, 2, 1),  # gate_up_proj
    (128, 12288, 4096): (128, 64, 128, 8, 1),  # down_proj
}


# Explicit gfx1151 prefill tile selection -- DTYPE-AWARE. The kernel takes the
# packed v_and_or/v_pk_fma dequant for fp16 and the scalar dequant for bf16, and
# the two paths want different tiles (most visibly BLOCK_N at deep M: 256 for
# packed fp16 vs 64 for scalar bf16).
#
# fp16 (packed), M > 128 only -- tuned with rotating cold weights over a broad
# shape catalog:
#   * 129..256: BLOCK_N=128.
#   * 257..2047: the wide distilled BLOCK_N=256/BLOCK_M=128 tile.
#   * M >= 2048: distilled BLOCK_N=256; BLOCK_M=64 for narrow+deep K (N<=2048
#     and K>=4096), else 128.
#
# bf16 (scalar) -- byte-for-byte the pre-existing scalar-tuned ladder, its
# per-shape overrides, and its pipeline depth, so bf16 is bit-for-bit unchanged:
# the packed fp16 table regresses bf16 by up to ~40% at deep M, where scalar bf16
# wants BLOCK_N=64, not 256. num_stages=1 is deliberately NOT applied here --
# measured at -15.8% end-to-end prefill on an asymmetric bf16 model. The packed
# fp16 path issues one per-group load and does not miss the pipelining; the
# scalar asymmetric path issues two (scale and zero point) and needs the software
# pipeline to hide the second gather.
#
# BLOCK_K is capped to group_size so a K-block never straddles a quant group
# (scale aliasing); gs=128 -- the bulk -- passes the table BLOCK_K through.
# The packed-fp16 tile table covers deep prefill only. At M <= 128 both dtypes
# use the pre-existing scalar-tuned ladder: the fp16 table was swept at deep
# prefill, and applying it at M=128 measured as a regression (up to +19% TTFT on
# a K=2048 model, and far worse on shapes whose packed row stride sits on the
# gfx11 cache cliff). A dedicated small-M sweep did remove those regressions but
# bought nothing -- at short prompts the prefill GEMMs are a small share of TTFT
# -- so the tiles there are left alone. The packed dequant itself applies at
# every M; only tile selection is gated.
_FP16_TILE_MIN_M = 128


def _select_skinny_gfx1151_config(
    M: int, N: int, K: int, group_size: int, dtype: torch.dtype
) -> tuple[int, int, int, int, int | None]:
    """Return (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for gfx1151.

    num_stages None means "leave Triton's default pipeline depth alone".
    """
    num_stages: int | None = None
    if dtype == torch.float16 and M > _FP16_TILE_MIN_M:
        # The packed path issues a single per-group load, so the pipeline buys
        # nothing and only costs registers. Not applied to bf16 -- see above.
        num_stages = 1
        if M <= 256:
            block_m, block_n, block_k, num_warps = 128, 128, 32, 8
        elif M < 2048:  # 257..2047 (mostly 512, 1024): wide distilled tile
            block_m, block_n, block_k, num_warps = 128, 256, 32, 8
        else:  # M >= 2048 (deep prefill)
            if N <= 2048 and K >= 4096:  # narrow + deep: halved BM saturates
                block_m, block_n, block_k, num_warps = 64, 256, 32, 8
            else:
                block_m, block_n, block_k, num_warps = 128, 256, 32, 8
        # Very narrow N at small/mid M: a wide BLOCK_N leaves too few N-tiles to
        # fill the CUs, so clamp it. At M>=1024 the M-tiles already saturate.
        if N <= 1024 and M <= 512:
            block_n = min(block_n, 32)
    else:
        # Scalar-dequant path (bf16): the pre-existing scalar-tuned ladder.
        key = (group_size, K, N)
        override = _GFX1151_BF16_PREFILL_OVERRIDES.get(key) if M <= 128 else None
        if override is not None:
            block_m, block_n, block_k, num_warps, num_stages = override
        elif M <= 32:
            block_m, block_n, block_k, num_warps = 32, 32, 128, 4
        elif M <= 64:
            block_m, block_n, block_k, num_warps = 64, 64, 32, 4
        elif M <= 128:
            if K >= 2 * N:  # tall K (down_proj)
                block_m, block_n, block_k, num_warps = 64, 16, 64, 1
            elif N > K:  # wide N (qkv / gate_up)
                block_m, block_n, block_k, num_warps = 64, 64, 64, 4
            else:  # N ~= K (o_proj)
                block_m, block_n, block_k, num_warps = 64, 32, 64, 4
        elif M <= 1024:
            if K >= 2 * N:  # tall K (down_proj)
                block_m, block_n, block_k, num_warps = 64, 64, 64, 4
            elif N >= 4 * K:  # very wide N (gate_up)
                block_m, block_n, block_k, num_warps = 128, 64, 64, 8
            else:
                block_m, block_n, block_k, num_warps = 64, 128, 32, 4
        else:  # M > 1024
            if K >= 2 * N:  # tall K (down_proj)
                block_m, block_n, block_k, num_warps = 128, 512, 32, 16
            else:
                block_m, block_n, block_k, num_warps = 128, 64, 64, 8
    return block_m, block_n, min(block_k, group_size), num_warps, num_stages


def triton_w4a16_skinny_fmt_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [N, K//8] int32 (ExLlama shuffle packed)
    scales: torch.Tensor,  # [N, K//G] fp16/bf16
    group_size: int,
    zp_bias: int = 8,
    zp: torch.Tensor | None = None,  # [N//8, K//G] int32 zero-points
) -> torch.Tensor:
    """Fused W4A16 GEMM reading from skinny weight format [N, K//8].

    Args:
        a:          Activation matrix [M, K], float16 or bfloat16.
        b_q:        Packed weight matrix [N, K//8], int32 (ExLlama shuffle).
        scales:     Per-group scales [N, K//G], same dtype as a.
        group_size: Quantization group size (resolved from -1 to K by caller).
        zp_bias:    Constant zero bias (default 8 for unsigned int4).
        zp:         Raw per-group zero-points [N//8, K//G] int32, row n at
                    word[n//8] bits 4*(n%8) (asymmetric). When provided,
                    dequant is (nibble - zp_raw) * scale.

    Returns:
        Output matrix [M, N], same dtype as a.

    """
    assert a.stride(1) == 1, "Activation rows must be contiguous"
    assert b_q.stride(1) == 1, "Weight rows must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[0]
    K8 = K // 8
    num_groups = K // group_size
    stride_bn = b_q.stride(0)
    stride_am = a.stride(0)

    assert b_q.shape == (N, K8), f"b_q shape mismatch: {b_q.shape} vs ({N}, {K8})"
    assert scales.shape == (N, num_groups), (
        f"scales shape mismatch: {scales.shape} vs ({N}, {num_groups})"
    )
    if zp is not None:
        assert zp.is_contiguous(), "Zero-points must be contiguous"
        assert N % 8 == 0, f"N must be divisible by 8 for packed zp, got {N}"
        assert zp.shape == (N // 8, num_groups), (
            f"zp shape mismatch: {zp.shape} vs ({N // 8}, {num_groups})"
        )
    has_zp = zp is not None

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # num_stages stays None unless the tile table sets it, so the generic
    # heuristics fall back to Triton's default pipeline depth.
    num_stages: int | None = None
    if _on_gfx12x():
        # Tuned on gfx1201 (Radeon AI PRO R9700, 32 CUs, 32-wide wavefronts)
        # using Llama-3.1-8B AWQ weight shapes with group_size=128.
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 16, 16, 128, 4
        elif M <= 64:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 32, 128, 8
            elif N > K:  # wide N (e.g. qkv_proj, gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 32, 64, 8
            else:  # N ~= K (e.g. o_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 32, 64, 128, 4
        elif M <= 128:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 16, 64, 1
            elif N >= 2 * K:  # very wide N (e.g. gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 128, 64, 8
            else:  # N ~= K (e.g. o_proj, qkv_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 64, 64, 8
        elif M <= 512:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 64, 64, 8
            elif N >= 4 * K:  # very wide N (e.g. gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 128, 64, 8
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 128, 64, 8
        else:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 64, 64, 8
            elif N >= 4 * K:  # very wide N (e.g. gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 256, 64, 64, 8
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 128, 32, 8
    elif _on_gfx1151():
        # gfx1151 (Strix Halo, 40 CUs, 32-wide wavefronts): per-(M, N, K) tile
        # config from the dtype-aware table, since the packed fp16 dequant and
        # the scalar bf16 dequant want different tiles. See
        # _select_skinny_gfx1151_config; re-run
        # benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py after edits.
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = (
            _select_skinny_gfx1151_config(M, N, K, group_size, a.dtype)
        )
    else:
        num_warps = 4
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        elif M <= 64:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    # The kernel loads one scale per BLOCK_K tile, so BLOCK_K must not
    # exceed group_size — otherwise elements in the tile that belong to
    # a different group would get the wrong scale.
    BLOCK_K = min(BLOCK_K, group_size)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    extra_kwargs = {} if num_stages is None else {"num_stages": num_stages}
    _triton_w4a16_skinny_fmt_kernel[grid](
        a,
        b_q,
        scales,
        zp if has_zp else scales,  # dummy pointer when no zp (unused)
        c,
        M,
        N,
        K,
        K8,
        num_groups,
        stride_bn,
        stride_am,
        group_size=group_size,
        ZP_BIAS=zp_bias,
        HAS_ZP=has_zp,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        **extra_kwargs,
    )
    return c


# ---------------------------------------------------------------------------
# Weight packing
# ---------------------------------------------------------------------------


def pack_int4_exllama_shuffle(w_uint4: torch.Tensor) -> torch.Tensor:
    """Pack uint4 values into ExLlama shuffle format: [N, K] -> [N, K//8] int32.

    Each int32 packs 8 K-values with interleave order [0,2,4,6,1,3,5,7].
    """
    N_dim, K_dim = w_uint4.shape
    assert K_dim % 8 == 0
    g = w_uint4.to(torch.uint8).view(N_dim, K_dim // 8, 8).to(torch.int32)
    return (
        g[:, :, 0]
        | (g[:, :, 2] << 4)
        | (g[:, :, 4] << 8)
        | (g[:, :, 6] << 12)
        | (g[:, :, 1] << 16)
        | (g[:, :, 3] << 20)
        | (g[:, :, 5] << 24)
        | (g[:, :, 7] << 28)
    )


# gfx11 row strides alias in the vector caches when they land on a power-of-two
# multiple.
_STRIDE_CLIFF_BYTES = 1024  # packed weight row, K/2 B
_ACT_CLIFF_BYTES = 2048  # activation row, K*2 B
_STRIDE_PAD_BYTES = 128  # one cache line


def _cliff_pad_bytes(row_bytes: int) -> int:
    """Bytes to add to a packed weight row stride to move it off the cliff.

    Only strides that are a multiple of ``_STRIDE_CLIFF_BYTES`` are moved, and
    only by one cache line. Padding is not free: it costs ``pad / row_bytes`` of
    weight memory, which on an APU comes straight out of KV-cache space, and
    measurements show it is a real loss on strides that are already off the
    cliff (-14% at M=1 on a 4864 B row).
    """
    if row_bytes % _STRIDE_CLIFF_BYTES:
        return 0
    return _STRIDE_PAD_BYTES


def pack_skinny_int4(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack [N, K] uint4 into the skinny weight layout the kernels consume.

    Single source of truth for the skinny weight memory layout: ExLlama shuffle
    to [N, K//8] int32, viewed as int8 [N, K//2], with the row stride nudged
    off the gfx11 cliff (see ``_cliff_pad_bytes``) where it lands on one.
    """
    shuffled = pack_int4_exllama_shuffle(unpacked)
    n_rows, k8 = shuffled.shape
    pad_int32 = _cliff_pad_bytes(k8 * 4) // 4
    if not (pad_int32 and _on_gfx1151()):
        return shuffled.contiguous().view(torch.int8)
    padded = torch.empty(
        (n_rows, k8 + pad_int32), dtype=torch.int32, device=shuffled.device
    )
    padded[:, :k8].copy_(shuffled)
    # The int8 view keeps stride(0) = 4 * (k8 + pad_int32) bytes, and
    # .view(torch.int32) at apply time recovers the [N, K//8] int32 view.
    return padded.view(torch.int8)[:, : k8 * 4]


# ---------------------------------------------------------------------------
# Hybrid dispatch logic
# ---------------------------------------------------------------------------


def _pad_activation_rows(x_2d: torch.Tensor) -> torch.Tensor:
    """Copy ``x_2d`` into a row-padded buffer when its row stride is on the cliff.

    The packed weight is padded once at load time, but activations are produced
    fresh every step, so this materialises a padded copy. Only the Triton
    prefill path calls it: the skinny decode kernel derives the activation row
    stride from K, and at M <= 5 the row stride cannot alias anything anyway.
    """
    row_bytes = x_2d.shape[1] * x_2d.element_size()
    on_cliff = _ACT_CLIFF_BYTES and row_bytes % _ACT_CLIFF_BYTES == 0
    if not (on_cliff and _on_gfx1151()):
        return x_2d
    pad_elems = _STRIDE_PAD_BYTES // x_2d.element_size()
    buf = torch.empty(
        (x_2d.shape[0], x_2d.shape[1] + pad_elems),
        dtype=x_2d.dtype,
        device=x_2d.device,
    )
    padded = buf[:, : x_2d.shape[1]]
    padded.copy_(x_2d)
    return padded


def _rdna_hybrid_w4a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch between skinny GEMM and Triton based on batch size M.

    ``w_zp`` is [N//8, K//G] int32 for asymmetric layers, None for symmetric.
    """
    import vllm._custom_ops as ops

    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N = w_q.shape[0]

    if M <= MAX_SKINNY_BATCH_SIZE and K * M <= LDS_CAPACITY_ELEMENTS:
        # record_function is not torch.compile-safe; use nullcontext when
        # compiling to keep the op traceable.
        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"wvsplitk_int4 {M}x{N}x{K}")
        )
        with ctx:
            return ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu_count, group_size, w_zp, bias)

    ctx = (
        nullcontext()
        if torch.compiler.is_compiling()
        else torch.profiler.record_function(f"hybrid_triton_w4a16 {M}x{N}x{K}")
    )
    with ctx:
        output = triton_w4a16_skinny_fmt_gemm(
            a=_pad_activation_rows(x_2d),
            b_q=w_q.view(torch.int32),
            scales=w_s,
            group_size=group_size,
            zp=w_zp,
        )
        if bias is not None:
            output.add_(bias)
    return output


def _rdna_hybrid_w4a16_apply_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    M = x_2d.size(0)
    N = w_q.size(0)
    return torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)


direct_register_custom_op(
    op_name="rdna_hybrid_w4a16_apply",
    op_func=_rdna_hybrid_w4a16_apply_impl,
    mutates_args=[],
    fake_impl=_rdna_hybrid_w4a16_apply_fake,
)


class RDNAHybridW4A16LinearKernel(MPLinearKernel):
    """Hybrid W4A16 kernel: HIP skinny for decode, Triton for prefill.

    Stores the weights once as int8 [N, K//2] (ExLlama shuffle packed). The
    HIP skinny kernel reads it directly; the triton kernel reinterprets the
    same buffer as int32 [N, K//8] via a view, so there is no dual weight
    storage.
    """

    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
        scalar_types.uint4,  # asymmetric (zero_points)
    ]

    @classmethod
    def get_min_capability(cls) -> int:
        # Arch filtering is handled by can_implement (_on_gfx1x check)
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNAHybridW4A16LinearKernel only targets ROCm"

        if not _on_gfx1x():
            return False, "RDNAHybridW4A16LinearKernel only targets gfx11/gfx12"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"

        gs = c.group_size
        if gs not in SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size {gs} not supported; supported: {SUPPORTED_GROUP_SIZES}",
            )

        K = c.partition_weight_shape[0]
        if K % 16 != 0:
            return False, f"K={K} must be divisible by 16"

        if K % gs != 0:
            return (
                False,
                f"K={K} must be divisible by group_size={gs}",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        w_q_raw = getattr(layer, self.w_q_name)
        w_s_raw = getattr(layer, self.w_s_name)

        unpacked = unpack_quantized_values_into_int32(
            w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
        )
        # AWQ weights arrive as (K, N) with output_dim=1;
        # compressed-tensors arrive as (N, K) with output_dim=0.
        if getattr(w_q_raw, "output_dim", 0) != 0:
            unpacked = unpacked.t().contiguous()

        # Store as int8; Triton reinterprets via .view(torch.int32) at apply time.
        w_q_skinny = pack_skinny_int4(unpacked)

        permute_param_layout_(w_s_raw, input_dim=1, output_dim=0)
        w_s_skinny = w_s_raw.data.contiguous()

        if c.zero_points:
            assert self.w_zp_name is not None
            w_zp_raw = getattr(layer, self.w_zp_name)
            permute_param_layout_(w_zp_raw, input_dim=1, output_dim=0, packed_dim=0)
            w_zp = w_zp_raw.data.contiguous()
            self._transform_param(layer, self.w_zp_name, lambda x: w_zp)

        self._transform_param(layer, self.w_q_name, lambda x: w_q_skinny)
        self._transform_param(layer, self.w_s_name, lambda x: w_s_skinny)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        c = self.config
        w_q, w_s, w_zp = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1])
        N = w_q.shape[0]
        out_shape = x.shape[:-1] + (N,)

        cu_count = num_compute_units()
        output = torch.ops.vllm.rdna_hybrid_w4a16_apply(
            x_2d,
            w_q,
            w_s,
            w_zp,
            bias,
            cu_count,
            c.group_size,
        )
        return output.reshape(out_shape)
