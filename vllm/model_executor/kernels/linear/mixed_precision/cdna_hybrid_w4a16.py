# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CDNA W4A16 hybrid linear kernel (MI200 / MI300, gfx90a / gfx942 / gfx950).

Mirrors the structure of HybridW4A16LinearKernel (PR #40977) but targets the
CDNA family of GPUs (wave64, MFMA) instead of RDNA (wave32, WMMA). PR #40977's
HIP skinny GEMM is wave32-locked (DPP masks and __shfl_xor lane widths are
hard-coded for 16-lane half-waves), so a CDNA mirror cannot reuse it as-is.

This kernel is Triton-only: a single fused W4A16 GEMM is launched with two
distinct tile-size regimes selected by the batch dimension M:

  * Decode regime (M <= DECODE_M_THRESHOLD): small BLOCK_M, wide BLOCK_N to
    amortise the per-output-column scale and zero-point loads. This regime
    dominates single-token autoregressive serving.
  * Prefill regime (M >  DECODE_M_THRESHOLD): MFMA-shaped tiles (multiples of
    16 in each dim). num_stages=1 (no software pipelining) is the default
    after observing HSA_STATUS_ERROR_INVALID_PACKET_FORMAT with num_stages>1
    on large-K (>=3584) shapes during decode JIT-compile on MI300. Pipelining
    can be opted into per-shape via _CDNA_PREFILL_OVERRIDES once a particular
    (group_size, K, N) tuple is known to be stable.

The weight format is shared with PR #40977: [N, K//8] int32 with the ExLlama
shuffle (nibbles 0,2,4,6,1,3,5,7 -> bit shifts 0,4,8,12,16,20,24,28). This
keeps the per-output-row scales/zero-points contiguous in DRAM, which matters
much more on wave64 (where each wave reads 64 K-lanes per cycle) than on wave32.
"""

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import permute_param_layout_
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_mi3xx
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

SUPPORTED_GROUP_SIZES = [32, 64, 128]

# Batch dimension below which we use the decode-shaped tile.
DECODE_M_THRESHOLD = 16


# ---------------------------------------------------------------------------
# Triton kernel (shared by both regimes; tile sizes are picked at launch time)
# ---------------------------------------------------------------------------


@triton.jit
def _cdna_w4a16_skinny_kernel(
    a_ptr,           # [M, K]      fp16 / bf16 activations
    b_ptr,           # [N, K//8]   int32  ExLlama-shuffle packed weights
    scales_ptr,      # [N, K//G]   fp16 / bf16
    zp_ptr,          # [N, K//G]   fp16 / bf16 (raw zero-points, asymmetric)
    c_ptr,           # [M, N]      fp16 / bf16 output
    M, N, K,
    K8,              # K // 8
    num_groups,      # K // group_size
    group_size,
    ZP_BIAS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """W4A16 GEMM with ExLlama-shuffle-packed weights in [N, K//8] layout.

    Dequant: (nibble - zero) * scale, where `zero` is either ZP_BIAS (e.g. 8
    for uint4b8) or a per-group raw zero-point loaded from zp_ptr.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ExLlama unshuffle: nibble j has shift (j//2)*4 + (j%2)*16
    # -> for j in 0..7: shifts = [0, 16, 4, 10, 8, 14, 12, 28]
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

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]
        mask_b = (offs_n[:, None] < N) & (offs_k8[None, :] < K8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # Replicate each packed int32 8 times along K, then shift+mask.
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts_full) & 0xF

        g_idx = (k_start * BLOCK_K) // group_size
        scale_ptrs = scales_ptr + offs_n * num_groups + g_idx
        scale_mask = offs_n < N
        scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0)

        if HAS_ZP:
            zp_ptrs = zp_ptr + offs_n * num_groups + g_idx
            zp_raw = tl.load(zp_ptrs, mask=scale_mask, other=0.0)
            b_fp = (b.to(scales.dtype) - zp_raw[:, None]) * scales[:, None]
        else:
            b_fp = (b - ZP_BIAS).to(scales.dtype) * scales[:, None]

        # [BLOCK_N, BLOCK_K] -> [BLOCK_K, BLOCK_N] for the matmul.
        b_fp_t = tl.trans(b_fp)
        accumulator += tl.dot(a, b_fp_t, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


# Per-shape overrides for prefill (M > DECODE_M_THRESHOLD) on CDNA. Populated
# from sweeps over the shapes that show up in Qwen / Llama / Mistral W4A16
# checkpoints; only kept where they beat the generic heuristic by >15% at
# M=128. Re-tune by running benchmark_cdna_hybrid_w4a16_gemm.py.
_CDNA_PREFILL_OVERRIDES: dict[
    tuple[int, int, int], tuple[int, int, int, int, int]
] = {
    # (group_size, K, N) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
}


def _pick_tile_sizes(
    M: int, K: int, N: int, group_size: int
) -> tuple[int, int, int, int, int]:
    """Return (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for CDNA.

    Tuned for gfx942 (MI300X: 304 CUs, wave64, 16x16x16 MFMA, 64KB LDS/CU).
    The gfx90a / gfx950 paths share these picks until per-arch sweeps justify
    splitting them; this matches how PR #40977 treats gfx1101/gfx1102 as a
    single bucket.

    BLOCK_K is later clamped to group_size by the caller, so all entries here
    represent the upper bound. All tile sizes are multiples of 16 so Triton
    lowers tl.dot to MFMA 16x16x16.
    """
    override = (
        _CDNA_PREFILL_OVERRIDES.get((group_size, K, N))
        if M > DECODE_M_THRESHOLD
        else None
    )
    if override is not None:
        return override

    # Decode regime: tiny BLOCK_M, wide BLOCK_N. Each wave (64 lanes) handles
    # one or two rows; the bottleneck is weight + scale DRAM traffic, so we
    # widen N to reuse loaded activations across many output columns.
    if M <= DECODE_M_THRESHOLD:
        # 16x128 tile, BLOCK_K up to 64 (clamped to group_size below). 4 warps
        # = 256 threads = 4 waves, which is a good occupancy point on MI300
        # (8 waves/SIMD limit, 4 SIMDs/CU).
        return 16, 128, 64, 4, 1

    # Prefill regime: MFMA-shaped tiles. Branch on aspect ratio of (N, K).
    if M <= 64:
        if K >= 2 * N:                      # tall K, e.g. down_proj
            return 32, 64, 64, 4, 1
        if N >= 2 * K:                      # wide N, e.g. gate_up_proj
            return 32, 128, 64, 4, 1
        return 64, 64, 64, 4, 1             # square-ish, e.g. o_proj

    if M <= 128:
        if K >= 2 * N:
            return 64, 64, 64, 4, 1
        if N >= 2 * K:
            return 64, 128, 64, 8, 1
        return 64, 128, 64, 4, 1

    if M <= 512:
        if K >= 2 * N:
            return 128, 64, 64, 8, 1
        if N >= 2 * K:
            return 128, 128, 64, 8, 1
        return 128, 128, 64, 4, 1

    # Large prefill: maximize per-CTA work, accept higher register pressure.
    if K >= 2 * N:
        return 128, 64, 64, 8, 1
    if N >= 2 * K:
        return 128, 256, 32, 8, 1
    return 128, 128, 32, 8, 1


def _cdna_w4a16_gemm(
    a: torch.Tensor,
    b_q_i32: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    zp_bias: int,
    zp: torch.Tensor | None,
) -> torch.Tensor:
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q_i32.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q_i32.shape[0]
    K8 = K // 8
    num_groups = K // group_size

    assert b_q_i32.shape == (N, K8), (
        f"b_q shape mismatch: {b_q_i32.shape} vs ({N}, {K8})"
    )
    assert scales.shape == (N, num_groups), (
        f"scales shape mismatch: {scales.shape} vs ({N}, {num_groups})"
    )
    has_zp = zp is not None
    if has_zp:
        assert zp.is_contiguous(), "Zero-points must be contiguous"
        assert zp.shape == (N, num_groups), (
            f"zp shape mismatch: {zp.shape} vs ({N}, {num_groups})"
        )

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _pick_tile_sizes(
        M, K, N, group_size
    )

    # One scale load per BLOCK_K tile: tile must not span more than one group.
    BLOCK_K = min(BLOCK_K, group_size)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _cdna_w4a16_skinny_kernel[grid](
        a,
        b_q_i32,
        scales,
        zp if has_zp else scales,  # dummy pointer when HAS_ZP=False (unused)
        c,
        M, N, K,
        K8,
        num_groups,
        group_size=group_size,
        ZP_BIAS=zp_bias,
        HAS_ZP=has_zp,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


# ---------------------------------------------------------------------------
# Weight repack (ExLlama shuffle, shared format with PR #40977)
# ---------------------------------------------------------------------------


def _pack_int4_exllama_shuffle(w_uint4: torch.Tensor) -> torch.Tensor:
    """[N, K] uint4 -> [N, K//8] int32 with ExLlama shuffle.

    Each int32 packs 8 K-values in interleave order [0,2,4,6,1,3,5,7] so the
    Triton reader can unshuffle them with the constant shift table above.
    """
    N_dim, K_dim = w_uint4.shape
    assert K_dim % 8 == 0, f"K={K_dim} must be divisible by 8"
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
# MPLinearKernel implementation
# ---------------------------------------------------------------------------


class CDNAHybridW4A16LinearKernel(MPLinearKernel):
    """W4A16 linear kernel for AMD CDNA (MI200 / MI300, gfx9 family).

    Single Triton kernel with two tile-size regimes (decode vs prefill) and
    a shared ExLlama-shuffle weight layout [N, K//8]. Targets the gap left
    by PRs #40977 (gfx11/gfx12) and #41394 (gfx1100) on issue #34008.
    """

    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
        scalar_types.uint4,    # asymmetric (per-group zero-points)
    ]

    @classmethod
    def get_min_capability(cls) -> int:
        # Arch filtering is handled by can_implement().
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "CDNAHybridW4A16LinearKernel only targets ROCm"

        if not on_mi3xx():
            return (
                False,
                "CDNAHybridW4A16LinearKernel only targets MI300-class GPUs "
                "(gfx942 / gfx950)",
            )

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"

        if c.has_g_idx:
            return False, "does not support g_idx activation reordering"

        gs = c.group_size
        if gs not in SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size {gs} not supported; supported: "
                f"{SUPPORTED_GROUP_SIZES}",
            )

        K = c.partition_weight_shape[0]
        if K % 16 != 0:
            return False, f"K={K} must be divisible by 16"
        if K % gs != 0:
            return False, f"K={K} must be divisible by group_size={gs}"

        N = c.partition_weight_shape[1]
        if N % 16 != 0:
            return False, f"N={N} must be divisible by 16"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        w_q_raw = getattr(layer, self.w_q_name)
        w_s_raw = getattr(layer, self.w_s_name)

        # Normalize qweight to [N, K] int32 with one uint4 value per slot.
        unpacked = unpack_quantized_values_into_int32(
            w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
        )
        # compressed-tensors: output_dim=0, [N, K//8] (already N-major)
        # AWQ-converted:      output_dim=1, [K, N//8] (transpose needed)
        if getattr(w_q_raw, "output_dim", 0) != 0:
            unpacked = unpacked.t().contiguous()

        shuffled = _pack_int4_exllama_shuffle(unpacked).contiguous()
        self._transform_param(layer, self.w_q_name, lambda x: shuffled)

        # Scales: bring to [N, K//G].
        permute_param_layout_(w_s_raw, input_dim=1, output_dim=0)
        w_s_skinny = w_s_raw.data.contiguous()
        self._transform_param(layer, self.w_s_name, lambda x: w_s_skinny)

        if c.zero_points:
            assert self.w_zp_name is not None
            w_zp_raw = getattr(layer, self.w_zp_name)
            permute_param_layout_(
                w_zp_raw, input_dim=1, output_dim=0, packed_dim=0
            )
            zp_unpacked = unpack_quantized_values_into_int32(
                w_zp_raw.data, c.weight_type, packed_dim=0
            )
            # Cast raw nibbles into activation dtype; the kernel subtracts
            # zp_raw from each dequantised weight directly.
            w_zp = zp_unpacked.to(c.act_type).contiguous()
            self._transform_param(layer, self.w_zp_name, lambda x: w_zp)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        N = w_q.shape[0]
        out_shape = x.shape[:-1] + (N,)

        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0

        output = _cdna_w4a16_gemm(
            a=x_2d,
            b_q_i32=w_q,
            scales=w_s,
            group_size=c.group_size,
            zp_bias=zp_bias,
            zp=w_zp,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
