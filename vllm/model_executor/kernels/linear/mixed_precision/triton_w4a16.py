# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based W4A16 GEMM kernel for ROCm MI300.

Implements fused int4-weight dequantization + fp16 GEMM in a single kernel,
using GPTQ sequential packing (8 int4 values per int32, shifts [0,4,...,28]).
Plugs into the MPLinearKernel selection system and is preferred over
MarlinLinearKernel/ExllamaLinearKernel on ROCm.

Weight layout expected by this kernel (post-process_weights_after_loading):
  qweight: [K, N//8]  int32  — rows=K (input), cols=N//8 (N is packed)
  scales:  [K//G, N]  fp16/bf16
  qzeros:  [K//G, N//8]  int32  (optional; None for symmetric uint4b8)

Checkpoint layout from compressed_tensors_wNa16 create_weights:
  weight_packed:     [N, K//8]  int32  (output_dim=0, input_dim=1, packed_dim=1)
  weight_scale:      [N, K//G]  fp16   (output_dim=0, input_dim=1)
  weight_zero_point: [N//8, K//G]  int32 (output_dim=0, packed_dim=0)
"""

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

TRITON_W4A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]
TRITON_W4A16_SUPPORTED_QUANT_TYPES = [
    scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
    scalar_types.uint4,  # asymmetric with explicit zeros
]


@triton.jit
def triton_w4a16_gemm_kernel(
    # Pointers
    a_ptr,  # [M, K]  fp16/bf16 activations
    b_ptr,  # [K, N//8]  int32 packed 4-bit weights (N is the packed dim)
    scales_ptr,  # [K//G, N]  fp16/bf16 scales
    zeros_ptr,  # [K//G, N//8]  int32 packed zeros (unused when HAS_ZP=False)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,  # stride in b along the packed N//8 dim
    stride_cm,
    stride_cn,
    # Quantization parameters
    group_size,
    # Whether explicit zero points are provided
    HAS_ZP: tl.constexpr,
    # Zero bias used when HAS_ZP is False (e.g. 8 for uint4b8)
    ZP_BIAS: tl.constexpr,
    # Block sizes (tuned for MI300 wavefront=64)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W4A16 GEMM: C[M,N] = A[M,K] @ dequant(B)[K,N]

    B is stored as [K, N//8] int32 using GPTQ sequential packing:
      each int32 packs 8 consecutive N-values at bit offsets [0,4,8,12,16,20,24,28].

    Dequant: w_fp = (w_int4 - zero) * scale
      HAS_ZP=True:  zero is loaded from zeros_ptr and unpacked
      HAS_ZP=False: zero = ZP_BIAS constant (e.g. 8 for uint4b8 symmetric)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # b/zeros are stored with N packed: N//8 int32 columns per K row
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)

    # GPTQ sequential shifts tiled across BLOCK_N:
    #   [0,4,8,...,28] repeating for every group of 8 N-values.
    # Build 1D shifts_1d of length BLOCK_N: column j gets shift (j % 8) * 4.
    shifts_row = tl.arange(0, 8) * 4  # [8]
    shifts_1d_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8))
    shifts_1d = tl.reshape(shifts_1d_2d, (BLOCK_N,))  # [BLOCK_N]
    # Broadcast to [BLOCK_K, BLOCK_N] for weight unpacking
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))

    # Scales column offsets: full N-width (one scale per output neuron)
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load activations A: [BLOCK_M, BLOCK_K] ----
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # ---- Load packed weights B: [BLOCK_K, BLOCK_N//8] int32 ----
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // 8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # ---- Unpack int4 weights → [BLOCK_K, BLOCK_N] ----
        # tl.interleave(x, x) doubles the last dim by interleaving.
        # Starting from [BLOCK_K, BLOCK_N//8], three interleaves give
        # [BLOCK_K, BLOCK_N], where each int32 is replicated 8 times.
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        # Extract the correct 4-bit nibble for each output column
        b = (b >> shifts) & 0xF

        # ---- Compute scale/zero group row index ----
        g_idx = (k_start * BLOCK_K) // group_size

        # ---- Load scales: [BLOCK_N] → broadcast to [BLOCK_K, BLOCK_N] ----
        scale_offset = g_idx * N + offs_sn
        scale_mask = offs_sn < N
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))

        # ---- Load / compute zeros ----
        if HAS_ZP:
            # Load packed zeros row: [BLOCK_N//8] int32
            zero_offset = g_idx * (N // 8) + offs_bn
            zero_mask = offs_bn < N // 8
            z_packed = tl.load(zeros_ptr + zero_offset, mask=zero_mask, other=0)
            # Unpack to [BLOCK_N] using same interleave+shift pattern
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xF
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
        else:
            z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)

        # ---- Dequantize: (w - zero) * scale ----
        b_fp = (b - z).to(a.dtype) * scales

        # ---- Accumulate ----
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

    # ---- Store output C: [BLOCK_M, BLOCK_N] ----
    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def triton_w4a16_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [K, N//8] int32
    scales: torch.Tensor,  # [K//G, N] fp16/bf16
    qzeros: torch.Tensor | None,  # [K//G, N//8] int32, or None
    group_size: int,
    zp_bias: int = 8,  # bias for uint4b8 when qzeros is None
) -> torch.Tensor:
    """
    Fused W4A16 GEMM using GPTQ-packed int4 weights.

    Args:
        a:          Activation matrix [M, K], float16 or bfloat16.
        b_q:        Packed weight matrix [K, N//8], int32 (GPTQ sequential).
        scales:     Per-group scales [K//G, N], same dtype as a.
        qzeros:     Per-group packed zero points [K//G, N//8] int32, or None
                    for symmetric quantization (uses zp_bias instead).
        group_size: Quantization group size (resolved from -1 to K by caller).
        zp_bias:    Constant zero used when qzeros is None (default 8 for uint4b8).

    Returns:
        Output matrix [M, N], same dtype as a.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[1] * 8

    assert b_q.shape == (K, N // 8), (
        f"b_q shape mismatch: {b_q.shape} vs ({K}, {N // 8})"
    )
    assert scales.shape == (K // group_size, N), (
        f"scales shape mismatch: {scales.shape} vs ({K // group_size}, {N})"
    )
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N // 8), (
            f"qzeros shape mismatch: {qzeros.shape}"
        )

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    has_zp = qzeros is not None
    # Provide a dummy pointer when HAS_ZP=False (Triton requires a valid ptr)
    zeros_ptr = qzeros if has_zp else b_q

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1x

        if on_gfx1x():
            # Tuned for RDNA 3.5 (gfx1151, 40 CUs, 32-wide wavefronts).
            if M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 32, 64
        else:
            # Tuned for MI300 (gfx942, 304 CUs, 64-wide wavefronts).
            if M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    else:
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        elif M <= 64:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    # The kernel loads scales/zeros for a single group per BLOCK_K tile
    # (one g_idx per iteration). If BLOCK_K > group_size, rows at the tail
    # of the tile dequantize with the wrong group's scales, silently
    # corrupting the output. Clamp BLOCK_K to group_size to keep one
    # scale group per tile.
    if group_size < BLOCK_K:
        BLOCK_K = group_size

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    triton_w4a16_gemm_kernel[grid](
        a,
        b_q,
        scales,
        zeros_ptr,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_q.stride(0),
        b_q.stride(1),
        c.stride(0),
        c.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


class TritonW4A16LinearKernel(MPLinearKernel):
    """
    Triton-based W4A16 GEMM kernel for ROCm (MI300 and newer).

    Supports GPTQ-format int4 weights (uint4b8 symmetric, uint4 asymmetric)
    with grouped quantization. Weight tensors are transposed from the
    compressed-tensors checkpoint layout to the kernel's [K, N//8] layout.
    """

    SUPPORTED_QUANT_TYPES = TRITON_W4A16_SUPPORTED_QUANT_TYPES

    @classmethod
    def get_min_capability(cls) -> int:
        # Triton handles capability checks itself
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "TritonW4A16LinearKernel only targets ROCm"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        N = c.partition_weight_shape[1]
        if N % 8 != 0:
            return (
                False,
                f"Output features ({N}) must be divisible by 8 "
                "(8 int4 values packed per int32)",
            )

        if c.has_g_idx:
            return (
                False,
                "Activation reordering (g_idx) is not supported by "
                "TritonW4A16LinearKernel",
            )

        gs = c.group_size
        if (
            gs not in TRITON_W4A16_SUPPORTED_GROUP_SIZES
            and gs != c.full_weight_shape[0]
        ):
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: {TRITON_W4A16_SUPPORTED_GROUP_SIZES} "
                f"or full K ({c.full_weight_shape[0]})",
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return (False, f"Input features {K} not divisible by group size {eff_gs}")

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Convert compressed-tensors checkpoint layout to kernel layout.

        Checkpoint (from compressed_tensors_wNa16.create_weights):
          weight_packed:     [N, K//8]  int32   input_dim=1, output_dim=0, packed_dim=1
          weight_scale:      [N, K//G]  fp16    input_dim=1, output_dim=0
          weight_zero_point: [N//8, K//G] int32  output_dim=0, packed_dim=0

        Kernel needs:
          qweight: [K, N//8]  int32   (transpose weight_packed)
          scales:  [K//G, N]  fp16    (transpose weight_scale)
          qzeros:  [K//G, N//8] int32 (transpose weight_zero_point)
        """

        # ---- Transform qweight: [N, K//8] → [K//8, N] → back to [K, N//8] ----
        # permute_param_layout_(x, input_dim=0, output_dim=1) rearranges so that
        # the input(K) dimension is at physical dim 0 and output(N) at dim 1.
        # Checkpoint has input_dim=1, output_dim=0, packed_dim=1 (K is packed).
        # After permute we get [K//8, N] (K packed at dim 0, N at dim 1).
        # The kernel wants [K, N//8] (K at dim 0, N packed at dim 1), so we
        # then transpose: [K//8, N].T = [N, K//8] — that's not right.
        #
        # Actually we need to change WHAT is packed:
        #   Original packing: K packed into K//8 (8 K-values per int32)
        #   Kernel packing:   N packed into N//8 (8 N-values per int32)
        # These require a full repack, not just a transpose.
        #
        # Simple approach: unpack → transpose the full [N, K] → repack as [K, N//8].
        # This is done CPU-side at load time (one-time cost).
        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            # x.data is [N, K//8] int32, K packed (GPTQ checkpoint format)
            # Step 1: bring to [N, K//8] with output(N) at dim 0
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//8] int32

            N_dim, K8 = w.shape
            K_dim = K8 * 8
            # Step 2: unpack to [N, K] int32 (vectorized)
            shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
            w_unpacked = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(N_dim, K_dim)
            # Step 3: transpose to [K, N] int32
            w_KN = w_unpacked.t().contiguous()
            # Step 4: repack N into N//8 int32 values → [K, N//8] (vectorized)
            N8 = N_dim // 8
            w_repacked = torch.sum(
                (w_KN.view(K_dim, N8, 8) & 0xF) << shifts,
                dim=2,
                dtype=torch.int32,
            )
            x.data = w_repacked.contiguous()
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            # x.data is [N, K//G] fp16, bring to [K//G, N]
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                # Checkpoint: [N//8, K//G] int32 (N packed at dim 0, K//G at dim 1)
                # Kernel needs: [K//G, N//8] — just transpose
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(zp.data.t().contiguous(), requires_grad=False),
                )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        K = c.partition_weight_shape[0]
        group_size = c.group_size if c.group_size != -1 else K

        # For symmetric types (uint4b8), use the scalar bias; no zeros tensor
        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0

        output = triton_w4a16_gemm(
            a=x_2d,
            b_q=w_q,
            scales=w_s,
            qzeros=w_zp,
            group_size=group_size,
            zp_bias=zp_bias,
        )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
