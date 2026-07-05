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

_logged = set()

@triton.jit
def triton_w4a16_gemm_kernel(
    # Pointers
    a_ptr,       # [M, K]  fp16/bf16 activations
    b_ptr,       # [K, N//8] int32 packed 4-bit weights
    scales_ptr,  # [K//G, N]  fp16/bf16 scales
    zeros_ptr,   # [K//G, N//8] int32 packed zeros
    c_ptr,       # [M, N]  fp16/bf16 output
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,  # Stride for original int32 tensor
    stride_cm, stride_cn,
    # Quantization parameters
    group_size: tl.constexpr,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    num_k_blocks_per_pid = tl.cdiv(num_k_blocks, SPLIT_K)
    k_start_idx = pid_k * num_k_blocks_per_pid
    k_end_idx = min(k_start_idx + num_k_blocks_per_pid, num_k_blocks)

    # 1. Initialize Block Pointers for A and C (O(1) register overhead)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, k_start_idx * BLOCK_K),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    # 2. Reinterpret b_ptr as uint8 to drastically reduce interleave footprint
    # N//8 int32 columns equals N//2 uint8 columns
    b_ptr_u8 = b_ptr.to(tl.pointer_type(tl.uint8))
    stride_bk_u8 = stride_bk * 4 
    
    # Initialize Block Pointer for B in int8 view
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr_u8,
        shape=(K, N // 2),
        strides=(stride_bk_u8, 1), 
        offsets=(k_start_idx * BLOCK_K, pid_n * (BLOCK_N // 2)),
        block_shape=(BLOCK_K, BLOCK_N // 2),
        order=(1, 0)
    )

    # 3. Setup 1D offsets for scale and zero vector loading
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scale_mask = offs_sn < N
    
    offs_zn = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    z_mask = offs_zn < N // 2

    # Accumulator in FP32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K-loop using block pointer advanced scaling
    for k_idx in range(k_start_idx, k_end_idx):
        # ---- Load A ----
        a = tl.load(a_block_ptr, boundary_check=(0, 1))

        # ---- Load B (uint8 mode, half the data footprint in registers) ----
        # b_packed_u8 shape: [BLOCK_K, BLOCK_N // 2]
        b_packed_u8 = tl.load(b_block_ptr, boundary_check=(0, 1))

        # ---- Fast Unpacking via single int8 Interleave ----
        # Extract low 4-bit and high 4-bit nibbles separately
        # Cast to int8 because later it will be substracted by zero points
        # Merge back to [BLOCK_K, BLOCK_N] using only ONE interleave operation
        b_low = (b_packed_u8 & 0x0F).to(tl.int8)
        b_high = ((b_packed_u8 >> 4) & 0x0F).to(tl.int8)  
        b = tl.interleave(b_low, b_high) 

        # ---- Compute scale/zero group row index ----
        g_idx = (k_idx * BLOCK_K) // group_size

        # ---- Load Scales ----
        # Others set to 0 to avoid overflowed N
        scale_offset = g_idx * N + offs_sn
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=0.0)

        # ---- Load / Compute ZP (Optimized int8 handling) ----
        if HAS_ZP:
            zeros_ptr_u8 = zeros_ptr.to(tl.pointer_type(tl.uint8))
            
            z_offset = g_idx * (N // 2) + offs_zn
            z_packed_u8 = tl.load(zeros_ptr_u8 + z_offset, mask=z_mask, other=0)
            
            z_low = (z_packed_u8 & 0x0F).to(tl.int8)
            z_high = ((z_packed_u8 >> 4) & 0x0F).to(tl.int8)
            z = tl.interleave(z_low, z_high)
        else:
            z = ZP_BIAS # Zero extra register cost as it handles via scalar broadcast

        # ---- Dequantize ----
        # Keep calculations in int8 up to subtraction, then cast to activation type
        z_val = z[None, :] if HAS_ZP else z
        b_fp = (b - z_val).to(a.dtype) * scales[None, :]
        

        # ---- GEMM Tensor Core Dot ----
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

        # ---- Advance Block Pointers for next tile ----
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # ---- Store Output C ----
    c = accumulator.to(c_ptr.dtype.element_ty)
    
    if SPLIT_K == 1:
        tl.store(c_block_ptr, c, boundary_check=(0, 1))
    else:
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
        c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
        tl.atomic_add(c_ptrs, c, mask=c_mask)

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

    c = torch.zeros((M, N), dtype=a.dtype, device=a.device)

    has_zp = qzeros is not None
    # Provide a dummy pointer when HAS_ZP=False (Triton requires a valid ptr)
    zeros_ptr = qzeros if has_zp else b_q
    
    BLOCK_M = 64 if M >= 64 else 1 << (M - 1).bit_length()
    BLOCK_N = 128 #if BLOCK_M >= 64 else 256
    BLOCK_K = group_size if group_size <=32 else 64 #gs=K when gs==-1
    
    num_cu = torch.cuda.get_device_properties(a.device).multi_processor_count
    
    active_cu = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    pra_ratio = active_cu / num_cu
    if pra_ratio <= 0.125:
        split_k = 8
    elif pra_ratio <= 0.4:
        split_k = 4
    elif pra_ratio <= 0.8:
        split_k = 2
    else:
        split_k = 1

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)
    
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
        SPLIT_K=split_k,
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
        if not (current_platform.is_rocm() or current_platform.is_cuda()):
            return False, "TritonW4A16LinearKernel requires CUDA or ROCm"

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
        qzeros = None if c.weight_type.has_bias() else w_zp
        output = triton_w4a16_gemm(
            a=x_2d,
            b_q=w_q,
            scales=w_s,
            qzeros=qzeros,
            group_size=group_size,
            zp_bias=zp_bias,
        )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
