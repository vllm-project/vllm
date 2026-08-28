# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hybrid W4A16 kernel: Triton for prefill, HIP skinny for decode.

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


@triton.jit
def _triton_w4a16_skinny_fmt_kernel(
    # Pointers
    a_ptr,  # [M, K]  fp16/bf16 activations
    b_ptr,  # [N, K//8]  int32 packed (ExLlama shuffle, K is packed dim)
    scales_ptr,  # [N, K//G]  fp16/bf16 scales (skinny layout)
    zp_ptr,  # [N, K//G]  fp16/bf16 raw zero-points (when HAS_ZP=True)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    K8,  # K // 8
    num_groups,  # K // group_size
    # Quantization parameters
    group_size,
    ZP_BIAS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W4A16 GEMM reading weights from skinny format [N, K//8].

    B is stored as [N, K//8] int32 using ExLlama shuffle packing:
      each int32 packs 8 K-values with interleave [0,2,4,6,1,3,5,7]:
        packed = val[0] | (val[2]<<4) | (val[4]<<8) | (val[6]<<12)
               | (val[1]<<16) | (val[3]<<20) | (val[5]<<24) | (val[7]<<28)

    Scales are [N, K//G] (skinny layout, NOT transposed).
    When HAS_ZP=True, raw zero-points zp_raw are loaded from zp_ptr [N, K//G]
    and subtracted directly: (nibble - zp_raw) * scale.
    When HAS_ZP=False, only the constant ZP_BIAS is subtracted (symmetric).
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

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]
        mask_b = (offs_n[:, None] < N) & (offs_k8[None, :] < K8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

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

        b_fp_t = tl.trans(b_fp)
        accumulator += tl.dot(a, b_fp_t, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


# Per-shape (group_size, K, N) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps,
# num_stages) tile-config overrides for prefill (M <= 128) on gfx1x.
# Picked by sweeping benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py + a
# per-config sweep script; only added when better than the generic heuristic
# by > 20% at M=128. Re-run benchmarks after edits.
_GFX1X_PREFILL_OVERRIDES: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {
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


def triton_w4a16_skinny_fmt_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [N, K//8] int32 (ExLlama shuffle packed)
    scales: torch.Tensor,  # [N, K//G] fp16/bf16
    group_size: int,
    zp_bias: int = 8,
    zp: torch.Tensor | None = None,  # [N, K//G] per-group zero-points
) -> torch.Tensor:
    """
    Fused W4A16 GEMM reading from skinny weight format [N, K//8].

    Args:
        a:          Activation matrix [M, K], float16 or bfloat16.
        b_q:        Packed weight matrix [N, K//8], int32 (ExLlama shuffle).
        scales:     Per-group scales [N, K//G], same dtype as a.
        group_size: Quantization group size (resolved from -1 to K by caller).
        zp_bias:    Constant zero bias (default 8 for unsigned int4).
        zp:         Raw per-group zero-points [N, K//G] (asymmetric),
                    stored as zp_raw in activation dtype. When provided,
                    dequant is (nibble - zp_raw) * scale.

    Returns:
        Output matrix [M, N], same dtype as a.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[0]
    K8 = K // 8
    num_groups = K // group_size

    assert b_q.shape == (N, K8), f"b_q shape mismatch: {b_q.shape} vs ({N}, {K8})"
    assert scales.shape == (N, num_groups), (
        f"scales shape mismatch: {scales.shape} vs ({N}, {num_groups})"
    )
    if zp is not None:
        assert zp.is_contiguous(), "Zero-points must be contiguous"
        assert zp.shape == (N, num_groups), (
            f"zp shape mismatch: {zp.shape} vs ({N}, {num_groups})"
        )
    has_zp = zp is not None

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # num_stages stays None unless a per-shape override sets it, so the
    # generic heuristics fall back to Triton's default pipeline depth.
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
        # Tuned on gfx1151 (Strix Halo, 40 CUs, 32-wide wavefronts)
        # using Qwen3-4B weight shapes with group_size=128.
        # Per-shape overrides for known prefill regressions live in a small
        # lookup table — see _GFX1X_PREFILL_OVERRIDES below. Re-run
        # benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py after edits.
        override = (
            _GFX1X_PREFILL_OVERRIDES.get((group_size, K, N)) if M <= 128 else None
        )
        if override is not None:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = override
        elif M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 32, 32, 128, 4
        elif M <= 64:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 64, 32, 4
        elif M <= 128:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 16, 64, 1
            elif N > K:  # wide N (e.g. qkv_proj, gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 64, 64, 4
            else:  # N ~= K (e.g. o_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 32, 64, 4
        elif M <= 1024:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 64, 64, 4
            elif N >= 4 * K:  # very wide N (e.g. gate_up_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 64, 64, 8
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 64, 128, 32, 4
        else:
            if K >= 2 * N:  # tall K (e.g. down_proj)
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 512, 32, 16
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps = 128, 64, 64, 8
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


# ---------------------------------------------------------------------------
# Hybrid dispatch logic
# ---------------------------------------------------------------------------


def _rdna_hybrid_w4a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch between skinny GEMM and Triton based on batch size M."""
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
            a=x_2d,
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

        unpacked = unpack_quantized_values_into_int32(
            w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
        )
        # AWQ weights arrive as (K, N) with output_dim=1;
        # compressed-tensors arrive as (N, K) with output_dim=0.
        if getattr(w_q_raw, "output_dim", 0) != 0:
            unpacked = unpacked.t().contiguous()

        shuffled = pack_int4_exllama_shuffle(unpacked)
        # Store as int8; Triton reinterprets via .view(torch.int32) at apply time.
        w_q_skinny = shuffled.contiguous().view(torch.int8)

        permute_param_layout_(w_s_raw, input_dim=1, output_dim=0)
        w_s_skinny = w_s_raw.data.contiguous()

        if c.zero_points:
            assert self.w_zp_name is not None
            w_zp_raw = getattr(layer, self.w_zp_name)
            permute_param_layout_(w_zp_raw, input_dim=1, output_dim=0, packed_dim=0)
            zp_unpacked = unpack_quantized_values_into_int32(
                w_zp_raw.data, c.weight_type, packed_dim=0
            )
            w_zp = zp_unpacked.to(c.act_type).contiguous()
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
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

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
