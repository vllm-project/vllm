# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hybrid W4A16 kernel: Triton for prefill, HIP skinny for decode.

Routes based on batch size M:
  M <= MAX_SKINNY_BATCH_SIZE: HIP skinny GEMM (wvSplitK_int4_g)
  M > MAX_SKINNY_BATCH_SIZE:  Triton W4A16 fused dequant GEMM

Stores weights ONCE in skinny layout [N, K//8] int32 (ExLlama shuffle).
Both the HIP skinny kernel and the triton kernel read from this single
weight copy. The triton kernel transposes tiles in-register.
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
from vllm.platforms.rocm import on_gfx1x, on_gfx1151
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

SUPPORTED_GROUP_SIZES = [32, 64, 128]

# Maximum batch size M for the HIP skinny kernel path (C++ supports N_in
# up to 5).  When M exceeds this AND K*M fits in LDS, the skinny kernel is
# used; otherwise the Triton prefill path handles the GEMM.
MAX_SKINNY_BATCH_SIZE = 5
LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements


# ---------------------------------------------------------------------------
# Triton kernel for the prefill path (reads skinny-format weights [N, K//8])
# ---------------------------------------------------------------------------


@tl.target_info.constexpr_function
def _target_is_gfx1x() -> bool:
    """Compile-time True on RDNA gfx11/gfx12 (where the v_and_or_b32 packed
    dequant is validated/tuned)."""
    target = tl.target_info.current_target()
    if target is None or target.backend != "hip":
        return False
    arch = str(target.arch)
    return arch.startswith("gfx11") or arch.startswith("gfx12")


@triton.jit
def _int4_pair_to_fp16x2(x):
    """Unpack two packed int4 nibbles into a uint32 holding two fp16 lanes,
    each equal to 1024 + nibble, with one ``v_and_or_b32``
    (``(x & 0x000F000F) | 0x64006400``).

    OR-ing a 4-bit nibble into the low mantissa of fp16 1024.0 (0x6400)
    bitcasts to exactly 1024+n (CK's i4_to_half trick). Doing it on a full
    32-bit lane dequants two nibbles per instruction, vs the scalar
    v_and_b16 + v_or_b16 pair Triton emits from the elementwise form.
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
    scales_ptr,  # [N, K//G]  fp16/bf16 scales (sym path, HAS_ZP=False)
    packed_scale_zp_ptr,  # [N, K//G]  int32 scale/zp carrier (asym, HAS_ZP)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    K8,  # K // 8
    stride_bn,  # per-row stride of b_ptr (in int32 elements)
    num_groups,  # K // group_size
    group_size,
    HAS_ZP: tl.constexpr,  # asym: read the scale/zp carrier; sym: scales + (-8)
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W4A16 GEMM reading skinny weights [N, K//8].

    B is stored as [N, K//8] int32 using ExLlama shuffle packing:
      each int32 packs 8 K-values with interleave [0,2,4,6,1,3,5,7]:
        packed = val[0] | (val[2]<<4) | (val[4]<<8) | (val[6]<<12)
               | (val[1]<<16) | (val[3]<<20) | (val[5]<<24) | (val[7]<<28)

    Two dequant paths, chosen at the layer's sym/asym nature:
      - HAS_ZP=True (asymmetric): read the carrier ``packed_scale_zp_ptr``
        [N, K//G] (one fp32 per (n, group)) — it folds the per-group scale AND
        the zero-point offset into a single load, replacing the separate scale +
        zp loads. Layout: fp16 = scale | bias_eff (= -8*scale - scaled_zp),
        dequant (nibble-1024)*scale + bias_eff via the magic-const fp16 unpack;
        bf16 = scale | zp_int, dequant (nibble - zp_int)*scale.
      - HAS_ZP=False (symmetric): the -8 offset is a constant, so there is
        no second load to fold — read ``scales_ptr`` directly and subtract the
        constant 8. fp16: (nibble - 1032)*scale via the magic unpack; bf16:
        (nibble - 8)*scale. (No carrier overhead for the sym fast path.)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ExLlama unshuffle shifts: shift[j] = (j//2)*4 + (j%2)*16
    # For 8 values: [0, 16, 4, 20, 8, 24, 12, 28]
    exllama_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    # Tile across BLOCK_K: repeat the 8-element pattern BLOCK_K//8 times
    shifts_1d = tl.reshape(
        tl.broadcast_to(exllama_shifts_row[None, :], (BLOCK_K // 8, 8)),
        (BLOCK_K,),
    )
    # Broadcast to [BLOCK_N, BLOCK_K]
    shifts_full = tl.broadcast_to(shifts_1d[None, :], (BLOCK_N, BLOCK_K))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load activations A: [BLOCK_M, BLOCK_K] ----
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # ---- Load packed weights B: [BLOCK_N, BLOCK_K//8] int32 ----
        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k8[None, :]
        mask_b = (offs_n[:, None] < N) & (offs_k8[None, :] < K8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # ---- Unpack int4 weights ----
        # The packed v_and_or_b32 / v_pk_fma dequant is fp16-only (the 1024+n
        # magic trick needs fp16's mantissa) and only validated/tuned on RDNA
        # gfx11/gfx12. Decided here at compile time (dtype + target arch) so
        # callers pass no flag; everything else uses the scalar unpack. The
        # condition is written inline (not via a local) so Triton constexpr-
        # eliminates the dead arm — the per-dtype vars below are only defined
        # on the taken path.
        if (a.dtype == tl.float16) and _target_is_gfx1x():
            # Packed dequant (fp16). The ExLlama int32 holds the
            # paired nibbles val[2p] @ bits[4p:4p+4] and val[2p+1] @
            # bits[16+4p:20+4p], so for pre-shift 4p (p=0..3),
            #   (x >> 4p) & 0x000F000F | 0x64006400
            # is one v_and_or_b32 producing a half2 = (1024+val[2p],
            # 1024+val[2p+1]) in K order (signed shift is fine: the sign fill
            # lands above bit 20, masked out). This dequants TWO nibbles per
            # instruction; the elementwise form lowers to scalar v_and_b16 +
            # v_or_b16 (1 nibble each). The interleave(lo, hi) lays b_raw out as
            # half2 so the downstream affine also packs into v_pk_fma_f16. The
            # dequant inner loop is VALU-issue-bound on gfx11, so this ~halves
            # the dequant instruction count per WMMA and matches CK.
            shifts4 = (tl.arange(0, 4) * 4)[None, None, :]
            bp_shift = tl.reshape(
                b_packed[:, :, None] >> shifts4, (BLOCK_N, BLOCK_K // 2)
            )
            packed_hl = _int4_pair_to_fp16x2(bp_shift)  # u32 half2: 1024+nibble pair
            lo = (packed_hl & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
            hi = (packed_hl >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
            b_raw = tl.interleave(lo, hi)  # [BLOCK_N, BLOCK_K] fp16 = 1024+nibble
        else:
            # ExLlama unshuffle: replicate each int32 8x then per-lane shift+mask.
            b = tl.interleave(b_packed, b_packed)
            b = tl.interleave(b, b)
            b = tl.interleave(b, b)
            b = (b >> shifts_full) & 0xF  # [BLOCK_N, BLOCK_K]

        # ---- Per-group quant params from [N, K//G] layout ----
        g_idx = (k_start * BLOCK_K) // group_size
        scale_mask = offs_n < N

        # ---- Dequantize ----
        if HAS_ZP:
            # Asymmetric: packed scale/zp carrier (one fp32/group folds scale + zp).
            psz = tl.load(
                packed_scale_zp_ptr + offs_n * num_groups + g_idx,
                mask=scale_mask,
                other=0,
            )
            psz_u = psz.to(tl.uint32, bitcast=True)
            if a.dtype == tl.float16:
                # fp16: low16 = scale, high16 = bias_eff (= -8*scale - scaled_zp).
                # ONE fp16 FMA per group via the magic-constant i4->fp16 unpack.
                scale = (psz_u & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
                bias_eff = (psz_u >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
                if not _target_is_gfx1x():
                    b_raw = (b | 0x6400).to(tl.uint16).to(tl.float16, bitcast=True)
                c1024 = tl.full((), 1024.0, tl.float16)
                b_fp = (b_raw - c1024) * scale[:, None] + bias_eff[:, None]
            else:
                # bf16: low16 = scale, high16 = zp_int. Cheap int-domain subtract
                # before the single bf16 multiply (RDNA3 has no v_pk_fma_bf16).
                scale = (psz_u & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
                zp_int = ((psz_u >> 16) & 0xFFFF).to(b.dtype)
                b_fp = (b - zp_int[:, None]).to(scale.dtype) * scale[:, None]
        else:
            # Symmetric: the -8 offset is constant (no zp to fold), so read the
            # scale directly — no carrier overhead.
            scales = tl.load(
                scales_ptr + offs_n * num_groups + g_idx, mask=scale_mask, other=1.0
            )
            if a.dtype == tl.float16:
                # (nibble - 8) * scale == (b_raw - (1024+8)) * scale, via magic.
                if not _target_is_gfx1x():
                    b_raw = (b | 0x6400).to(tl.uint16).to(tl.float16, bitcast=True)
                c = tl.full((), float(1024 + 8), tl.float16)
                b_fp = (b_raw - c) * scales[:, None]
            else:
                # bf16: (nibble - 8) * scale, int subtract before the cast.
                b_fp = (b - 8).to(scales.dtype) * scales[:, None]

        # ---- Transpose to [BLOCK_K, BLOCK_N] for matmul ----
        b_fp_t = tl.trans(b_fp)

        # ---- Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] ----
        accumulator += tl.dot(a, b_fp_t, out_dtype=tl.float32)

    # ---- Store output C: [BLOCK_M, BLOCK_N] ----
    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


# Explicit gfx11 prefill tile selection, tuned for the *packed-dequant* kernel
# across the full M range under do_bench_cudagraph (the production path) over the
# F.2 GEMM catalog. Four M tiers; BLOCK_K=128 dominates the small/mid tiers
# (long per-block K work, occupancy-friendly narrow BLOCK_N), the distilled wide
# BLOCK_N=256 tile takes over only at deep M:
#
#   * M <= 64: small tile BLOCK_M=32, BLOCK_K=128. A wide BLOCK_N here leaves
#     only ceil(N/BN) workgroups -- e.g. 8 for N=2048/BN=256 -- starving the 40
#     CUs (the M-blind BLOCK_N=256 was a 1.6-3x regression at M<=32). BLOCK_N=64
#     for tall K (down_proj) gives more B-reuse; 32 otherwise.
#   * 65 <= M <= 256: very wide N (>=16384) saturates with BLOCK_M=128/BN=64;
#     tall K likes BLOCK_M=32/BN=64; else BLOCK_M=64/BN=32. BLOCK_K=128/64.
#   * 257 <= M < 2048: the wide distilled tile (BN=256) wins for tall, narrow-N,
#     non-cliff K (e.g. Qwen3-4B down 2560x9728); BLOCK_M=128/BN=64 elsewhere.
#     The K%2048==0 cliff collapses BN=256 at these M (down 4096x12288 lost ~28%
#     at M=1024), so it is excluded from the distilled branch.
#   * M >= 2048 (deep prefill): the distilled BLOCK_N=256 / BLOCK_M=128 tile
#     maximizes B-reuse and ds_read_b128 LDS readback; +13-50% over gfx11.
#
# (The earlier distilled table was M-blind, autotuned only at M in {2048,3968},
# so it regressed every M<512 shape.) BLOCK_K is capped to group_size so a
# K-block never straddles a quant group (scale aliasing); gs128 layers -- the
# bulk -- pass the table BLOCK_K through unchanged, gs64/gs32 clamp to 64/32.
def _select_skinny_gfx11_config(
    M: int, N: int, K: int, group_size: int
) -> tuple[int, int, int, int]:
    """Return (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) for the gfx11 skinny GEMM."""
    tall = K >= 2 * N  # tall-K (down_proj-like)
    if M <= 64:
        block_m, block_n, block_k, num_warps = 32, (64 if tall else 32), 128, 4
    elif M <= 256:
        if N >= 16384:  # very wide N (gate_up / lm_head)
            block_m, block_n, block_k, num_warps = 128, 64, 64, 8
        elif tall:  # tall K (down_proj)
            block_m, block_n, block_k, num_warps = 32, 64, 128, 4
        else:  # N ~= K (o_proj / qkv)
            block_m, block_n, block_k, num_warps = 64, 32, 128, 4
    elif M < 2048:
        if tall and N < 4096 and K % 2048 != 0:  # narrow tall non-cliff K
            block_m, block_n, block_k, num_warps = 128, 256, 32, 8
        else:
            block_m, block_n, block_k, num_warps = 128, 64, 64, 8
    else:  # M >= 2048 (deep prefill)
        block_m, block_n, block_k, num_warps = 128, 256, 32, 8
    # Very narrow N (e.g. the L2 N=512 microbench shapes) at small/mid M: a wide
    # BLOCK_N leaves too few N-tiles (ceil(512/256)=2) to fill the CUs, so clamp
    # it -- BLOCK_N=32 restores ~16 N-tiles. At M>=1024 the M-tiles already
    # saturate the CUs, so the wide tile is kept (clamping there costs ~5x).
    if N <= 1024 and M <= 512:
        block_n = min(block_n, 32)
    return block_m, block_n, min(block_k, group_size), num_warps


def triton_w4a16_skinny_fmt_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [N, K//8] int32 (ExLlama shuffle packed)
    scales: torch.Tensor,  # [N, K//G] fp16/bf16 (used for the symmetric path)
    group_size: int,
    out: torch.Tensor | None = None,  # [M, N] optional pre-allocated output
    packed_scale_zp: torch.Tensor | None = None,  # [N, K//G] fp32 carrier (asym only)
) -> torch.Tensor:
    """
    Fused W4A16 GEMM reading skinny weights [N, K//8].

    Asymmetric layers pass ``packed_scale_zp`` (the carrier that folds scale +
    zero-point into one load); symmetric layers leave it None and the kernel
    reads ``scales`` directly with a constant -8 offset (no carrier overhead —
    sym has no second load to fold).

    Args:
        a:          Activation matrix [M, K], float16 or bfloat16.
        b_q:        Packed weight matrix [N, K//8], int32 (ExLlama shuffle).
        scales:     Per-group scales [N, K//G], same dtype as a (symmetric path).
        group_size: Quantization group size (resolved from -1 to K by caller).
        out:        Optional pre-allocated [M, N] output.
        packed_scale_zp:  Optional packed scale/zp carrier [N, K//G] fp32 for asymmetric
                    layers; layout is dtype-specific (fp16: scale|bias_eff; bf16:
                    scale|zp_int) — see the kernel docstring. When None, the
                    symmetric path is used.

    Returns:
        Output matrix [M, N], same dtype as a.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    # b_q may be a row-padded view (stride(0) > K//8) when the K_packed%2048
    # cliff workaround is active; only require the last dim to be unit-stride.
    assert b_q.stride(1) == 1, "Weight matrix must be unit-stride on K axis"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[0]
    K8 = K // 8
    num_groups = K // group_size

    assert b_q.shape == (N, K8), f"b_q shape mismatch: {b_q.shape} vs ({N}, {K8})"
    stride_bn = b_q.stride(0)
    assert scales.shape == (N, num_groups), (
        f"scales shape mismatch: {scales.shape} vs ({N}, {num_groups})"
    )
    has_zp = packed_scale_zp is not None
    if packed_scale_zp is not None:
        assert packed_scale_zp.is_contiguous(), "packed_scale_zp must be contiguous"
        assert packed_scale_zp.shape == (N, num_groups), (
            f"packed_scale_zp shape mismatch: {packed_scale_zp.shape} "
            f"vs ({N}, {num_groups})"
        )
        packed_scale_zp_i32 = packed_scale_zp.view(torch.int32)
    else:
        packed_scale_zp_i32 = scales  # dummy pointer (unused when HAS_ZP=False)

    if out is None:
        c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    else:
        assert out.shape == (M, N), f"out shape mismatch: {out.shape} vs ({M}, {N})"
        assert out.dtype == a.dtype, f"out dtype mismatch: {out.dtype} vs {a.dtype}"
        assert out.device == a.device, (
            f"out device mismatch: {out.device} vs {a.device}"
        )
        assert out.is_contiguous(), "out must be contiguous"
        c = out

    # On gfx11x, select the tile config per (M, N, K) from the per-M table
    # (see _select_skinny_gfx11_config). BLOCK_K is capped to group_size there
    # so a K-block never straddles a quant group (no scale aliasing).
    if on_gfx1x():
        block_m, block_n, block_k, num_warps = _select_skinny_gfx11_config(
            M, N, K, group_size
        )
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        # The kernel picks the packed (fp16/gfx1x) vs scalar unpack itself.
        _triton_w4a16_skinny_fmt_kernel[grid](
            a,
            b_q,
            scales,
            packed_scale_zp_i32,
            c,
            M,
            N,
            K,
            K8,
            stride_bn,
            num_groups,
            group_size=group_size,
            HAS_ZP=has_zp,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,  # >1 regresses badly for this kernel (no SW pipeline)
        )
        return c

    # AMD-specific scheduling hint; only consumed by the HIP backend below
    # (see compiler.py amdgpu-waves-per-eu attribute). Set to 0 by default
    # (no constraint); per-shape branches may override.
    waves_per_eu = 0

    cap = current_platform.get_device_capability()
    if cap is not None and cap.major >= 12:
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

    _triton_w4a16_skinny_fmt_kernel[grid](
        a,
        b_q,
        scales,
        packed_scale_zp_i32,
        c,
        M,
        N,
        K,
        K8,
        stride_bn,
        num_groups,
        group_size=group_size,
        HAS_ZP=has_zp,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        **({"waves_per_eu": waves_per_eu} if waves_per_eu else {}),
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


# ---------------------------------------------------------------------------
# Hybrid dispatch logic
# ---------------------------------------------------------------------------


def _hybrid_w4a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_q_i32: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    packed_scale_zp: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch between skinny GEMM and Triton based on batch size M.

    Both paths read from the same skinny-format weights:
      w_q:     [N, K//8] int8 (ExLlama shuffle, for skinny kernel)
      w_q_i32: [N, K//8] int32 (same data viewed as int32, for triton)
      w_s:     [N, K//G] fp16/bf16 (skinny-layout scales)
      w_zp:    [N, K//G] raw zero-points (zp_raw) in act dtype,
               or None for symmetric. Both HIP skinny and Triton use this
               single format: dequant = (nibble - zp_raw) * scale.
      packed_scale_zp: [N, K//G] fp32 carrier packing scale + zero-point per
               group (Triton prefill, asymmetric only), or None for symmetric.

    Registered as a custom op so torch.compile treats it as opaque.
    """
    import vllm._custom_ops as ops

    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N = w_q.shape[0]

    # Use the HIP skinny kernel for small batch sizes (fast decode path),
    # but only when K*M fits in LDS.  Otherwise fall through to Triton.
    if M <= MAX_SKINNY_BATCH_SIZE and K * M <= LDS_CAPACITY_ELEMENTS:
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
        # Asymmetric layers carry the packed scale/zp carrier (scale + zero-point folded
        # into one load); symmetric layers pass packed_scale_zp=None and the kernel
        # reads scales directly with a constant -8 offset (no carrier overhead).
        output = triton_w4a16_skinny_fmt_gemm(
            x_2d,
            w_q_i32,
            w_s,
            group_size,
            packed_scale_zp=packed_scale_zp,
        )
        if bias is not None:
            output.add_(bias)
    return output


def _hybrid_w4a16_apply_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_q_i32: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    packed_scale_zp: torch.Tensor | None = None,
) -> torch.Tensor:
    M = x_2d.size(0)
    N = w_q.size(0)
    return torch.empty((M, N), dtype=x_2d.dtype, device=x_2d.device)


direct_register_custom_op(
    op_name="hybrid_w4a16_apply",
    op_func=_hybrid_w4a16_apply_impl,
    mutates_args=[],
    fake_impl=_hybrid_w4a16_apply_fake,
)


class HybridW4A16LinearKernel(MPLinearKernel):
    """Hybrid W4A16 kernel: HIP skinny for decode, Triton for prefill.

    Stores weights once in skinny layout [N, K//8] (ExLlama shuffle packed).
    Both the HIP skinny kernel and the triton kernel read from this single
    weight copy, eliminating the memory overhead of dual weight storage.
    """

    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
        scalar_types.uint4,  # asymmetric (zero_points)
    ]

    @classmethod
    def get_min_capability(cls) -> int:
        return 110

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "HybridW4A16LinearKernel only targets ROCm"

        # Check HIP skinny op availability
        try:
            if not hasattr(torch.ops, "_rocm_C") or not hasattr(
                torch.ops._rocm_C, "wvSplitK_int4_g"
            ):
                return False, "wvSplitK_int4_g op not available in this build"
        except Exception:
            return False, "ROCm ops not available"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"

        if c.has_g_idx:
            return False, "does not support g_idx reordering"

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

        # Unpack raw weights and normalize to [N, K] int32
        unpacked = unpack_quantized_values_into_int32(
            w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
        )
        # AWQ-converted weights arrive as (K, N) with output_dim=1;
        # compressed-tensors arrive as (N, K) with output_dim=0.
        if getattr(w_q_raw, "output_dim", 0) != 0:
            unpacked = unpacked.t().contiguous()

        # ---- Pack into skinny format: [N, K//8] ExLlama shuffle ----
        shuffled = pack_int4_exllama_shuffle(unpacked)

        # ---- Pad K axis by +128 B per row on the gfx1151 cliff ----
        # On gfx1151 (Strix Halo) the int4 wvSplitK skinny kernel hits a sharp
        # BW cliff when K_packed = K/2 is a multiple of 2048 B -- multi-row
        # weight loads collide on memory-subsystem hash bits (DRAM channel
        # and/or MALL slice) downstream of L2.  Adding 128 B (one cache line /
        # 32 int32 cols) to the row stride breaks the collision.  Other gfx11x
        # parts (Strix Point gfx1150, Krackan gfx115{2,3}) lack the
        # multi-channel / MALL combination that produces the cliff and see no
        # benefit from the pad, so the 3% weight-memory overhead is gated to
        # gfx1151 only.
        N_rows, K8 = shuffled.shape
        K_packed_bytes = K8 * 4  # int32 -> bytes
        # +128 B per row = one cache line, keeping each row cache-line aligned.
        pad_int32 = 32
        if (
            on_gfx1151()
            and K_packed_bytes % 2048 == 0
            and shuffled.device.type == "cuda"
        ):
            padded = torch.empty(
                (N_rows, K8 + pad_int32),
                dtype=torch.int32,
                device=shuffled.device,
            )
            padded[:, :K8].copy_(shuffled)
            # Both views inherit stride(0) = K8 + pad_int32.
            w_q_skinny_i32 = padded[:, :K8]
            w_q_skinny = padded.view(torch.int8)[:, :K_packed_bytes]
        else:
            # Store as int8 for skinny kernel, keep int32 view for triton kernel
            w_q_skinny_i32 = shuffled.contiguous()
            w_q_skinny = w_q_skinny_i32.view(torch.int8)

        # ---- Prepare skinny scales: normalize to [N, K//G] ----
        permute_param_layout_(w_s_raw, input_dim=1, output_dim=0)
        w_s_skinny = w_s_raw.data.contiguous()

        # ---- Process zero-points for asymmetric quantization ----
        if c.zero_points:
            assert self.w_zp_name is not None
            w_zp_raw = getattr(layer, self.w_zp_name)
            # Normalize zp layout to (N, num_groups)
            permute_param_layout_(w_zp_raw, input_dim=1, output_dim=0, packed_dim=0)
            zp_unpacked = unpack_quantized_values_into_int32(
                w_zp_raw.data, c.weight_type, packed_dim=0
            )
            # zp_unpacked: [N, num_groups] with raw uint4 values [0..15]
            # Store raw zero-points in activation dtype.
            # Both kernels dequant as (nibble - zp_raw) * scale.
            w_zp = zp_unpacked.to(c.act_type).contiguous()
            self._transform_param(layer, self.w_zp_name, lambda x: w_zp)

        # ---- Store on layer ----
        # Replace w_q with skinny int8 (primary weights for skinny kernel)
        self._transform_param(layer, self.w_q_name, lambda x: w_q_skinny)
        # Replace w_s with skinny scales
        self._transform_param(layer, self.w_s_name, lambda x: w_s_skinny)

        # Store int32 view for triton kernel
        layer.register_parameter(
            "_hybrid_w_q_i32",
            torch.nn.Parameter(w_q_skinny_i32, requires_grad=False),
        )

        # Packed scale/zp carrier for the Triton prefill path — built ONLY
        # for asymmetric layers, where it folds the two per-group loads (scale +
        # zp) into one. Symmetric layers skip it: the -8 offset is a constant, so
        # there is no second load to fold and the carrier would be pure overhead
        # (measured ~+8% on fp16 sym); sym reads scales directly instead.
        # Layout (matches the kernel's HAS_ZP dequant):
        #   fp16: low16 = scale, high16 = bias_eff (= -8*scale - (zp-8)*scale).
        #         Consumed via one fp16 FMA with the magic-constant i4->fp16 unpack.
        #   bf16: low16 = scale (bf16 bits), high16 = zp_int (raw zp 0..15), as a
        #         plain integer. Consumed by the int-domain subtract (RDNA3 has no
        #         v_pk_fma_bf16). Bit-identical to the separate scale+zp loads.
        if c.zero_points and c.act_type in (torch.float16, torch.bfloat16):
            scale_u16 = w_s_skinny.view(torch.uint16).to(torch.int32) & 0xFFFF
            if c.act_type == torch.float16:
                w_s_f32 = w_s_skinny.to(torch.float32)
                scaled_zp_f32 = (w_zp.to(torch.float32) - 8.0) * w_s_f32
                bias_eff = (-(8.0 * w_s_f32 + scaled_zp_f32)).to(c.act_type)
                bias_u16 = bias_eff.contiguous().view(torch.uint16)
                hi_u16 = bias_u16.to(torch.int32) & 0xFFFF
            else:
                hi_u16 = w_zp.to(torch.int32) & 0xFFFF  # raw zp 0..15
            packed_scale_zp = (
                ((hi_u16 << 16) | scale_u16).view(torch.float32).contiguous()
            )
            layer.register_parameter(
                "_hybrid_w_packed_scale_zp",
                torch.nn.Parameter(packed_scale_zp, requires_grad=False),
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)
        w_q_i32 = layer._hybrid_w_q_i32
        # Packed scale/zp carrier (asymmetric layers only; None for sym).
        packed_scale_zp = getattr(layer, "_hybrid_w_packed_scale_zp", None)

        x_2d = x.reshape(-1, x.shape[-1])
        N = w_q.shape[0]
        out_shape = x.shape[:-1] + (N,)

        cu_count = num_compute_units()
        output = torch.ops.vllm.hybrid_w4a16_apply(
            x_2d,
            w_q,
            w_s,
            w_q_i32,
            w_zp,
            bias,
            cu_count,
            c.group_size,
            packed_scale_zp,
        )
        return output.reshape(out_shape)
