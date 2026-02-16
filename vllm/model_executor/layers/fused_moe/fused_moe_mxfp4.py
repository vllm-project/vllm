# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Custom Triton fused MoE kernel for MXFP4 (FP4 e2m1f) weights with inline
dequantization. Designed for hardware lacking tl.dot_scaled (e.g. RDNA4/gfx12).

Weights stay packed as uint8 in VRAM (~half the bf16 size) and are dequantized
per-tile inside the GEMM loop to bf16 in registers, then use standard
tl.dot(bf16, bf16).

Uses a "two half-dots" strategy:
  - Each uint8 stores 2 FP4 nibbles: lo = byte & 0x0F, hi = byte >> 4
  - Dequant lo/hi separately -> two [BLOCK_K//2, BLOCK_N] bf16 tiles
  - Load A with stride-2 for even/odd K columns -> two [BLOCK_M, BLOCK_K//2]
  - acc += tl.dot(a_even, lo_bf16) + tl.dot(a_odd, hi_bf16)
"""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    moe_align_block_size,
    try_get_optimal_moe_config,
    write_zeros_to_output,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Triton JIT helpers
# ---------------------------------------------------------------------------


@triton.jit
def dequant_mxfp4_nibble_to_bf16(nibble):
    """
    Dequantize a 4-bit FP4 e2m1f nibble (stored in the low 4 bits of a
    uint8/int32) to a bf16 value via bit manipulation.

    FP4 e2m1f layout (4 bits): S EE M
      sign = bit 3, exp = bits 2:1, mant = bit 0

    BF16 layout (16 bits): S EEEEEEEE MMMMMMM
      bias = 127, mantissa = 7 bits

    Conversion:
      - Normal (exp > 0):  new_exp = exp - 1 + 127 = exp + 126
                           new_mant = mant (placed at bit 6 of bf16 mantissa)
      - Subnormal (exp==0, mant==1): value = 0.5 -> exp=126, mant=0
      - Zero (exp==0, mant==0): all zero
    """
    sign = (nibble >> 3) & 1
    exp = (nibble >> 1) & 3
    mant = nibble & 1

    # Determine if the value is nonzero
    is_nonzero = (exp > 0) | (mant > 0)

    # For normals (exp>0): bf16_exp = exp + 126
    # For subnormal (exp==0, mant==1): bf16_exp = 126, bf16_mant = 0
    # We unify: bf16_exp = (exp + 126) when nonzero, but for subnormal
    # exp==0 gives 126 which is correct.
    new_exp = tl.where(is_nonzero, exp + 126, 0)

    # Mantissa bit: only set when exp > 0 AND mant == 1
    new_mant = tl.where((exp > 0) & (mant > 0), 1, 0)

    # Assemble bf16 bits: sign(1) | exponent(8) | mantissa(7)
    bf16_bits = (sign << 15) | (new_exp << 7) | (new_mant << 6)
    bf16_bits = bf16_bits.to(tl.uint16)
    return bf16_bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def e8m0_scale_to_bf16(scale_uint8):
    """
    Convert E8M0 scale (raw uint8 biased exponent) to bf16.

    E8M0 stores a biased exponent representing 2^(val - 127).
    In bf16, the exponent field is bits [14:7] with bias 127.
    So we just place the raw uint8 value as the bf16 exponent:
      bf16_bits = uint16(scale) << 7
    This gives sign=0, exponent=scale, mantissa=0 = 2^(scale-127).
    """
    bf16_bits = scale_uint8.to(tl.uint16) << 7
    return bf16_bits.to(tl.bfloat16, bitcast=True)


# ---------------------------------------------------------------------------
# Main fused MoE MXFP4 kernel
# ---------------------------------------------------------------------------


@triton.jit
def fused_moe_mxfp4_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,  # packed uint8 weights [E, N, K//2]
    c_ptr,
    b_bias_ptr,  # optional bias [E, N]
    b_scale_ptr,  # E8M0 scales [E, N, K//32]
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,  # logical (unpacked) K dimension
    EM,
    num_valid_tokens,
    # Strides for A [M, K]
    stride_am: tl.int64,
    stride_ak: tl.int64,
    # Strides for B [E, N, K//2] (packed)
    stride_be: tl.int64,
    stride_bn: tl.int64,
    stride_bk: tl.int64,
    # Strides for C [M, topk, N]
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    # Strides for B_scale [E, N, K//32]
    stride_bse: tl.int64,
    stride_bsn: tl.int64,
    stride_bsk: tl.int64,
    # Strides for B_bias [E, N]
    stride_bbe: tl.int64,
    stride_bbn: tl.int64,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # must be multiple of 64
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    # Map program id to tile
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Token routing
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    # N-dimension offsets
    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    if HAS_BIAS:
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)

    # Half-K for packed dimension
    HALF_K: tl.constexpr = BLOCK_SIZE_K // 2

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K in steps of BLOCK_SIZE_K (logical unpacked elements)
    num_k_iters = tl.cdiv(K, BLOCK_SIZE_K)
    for k_iter in range(0, num_k_iters):
        k_start = k_iter * BLOCK_SIZE_K  # logical K offset
        k_packed_start = k_start // 2     # packed K offset

        # Remaining elements mask
        k_remaining = K - k_start

        # --- Load B packed: [HALF_K, BLOCK_N] uint8 ---
        offs_bk = tl.arange(0, HALF_K)
        b_ptrs = (b_ptr
                  + off_experts * stride_be
                  + offs_bn[None, :] * stride_bn
                  + (k_packed_start + offs_bk[:, None]) * stride_bk)
        b_mask = offs_bk[:, None] < (k_remaining // 2)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0)

        # Unpack nibbles
        lo_nibble = b_packed & 0x0F          # even K indices
        hi_nibble = (b_packed >> 4) & 0x0F   # odd K indices

        # Dequant to bf16
        lo_bf16 = dequant_mxfp4_nibble_to_bf16(lo_nibble.to(tl.int32))
        hi_bf16 = dequant_mxfp4_nibble_to_bf16(hi_nibble.to(tl.int32))

        # --- Load and apply E8M0 scales ---
        # Scale shape: [E, N, K//32], one scale per 32 logical elements.
        # In packed space, 32 logical = 16 packed rows.
        # Each packed row j maps to scale group (k_start // 32 + j // 16).
        # Load scales directly as [HALF_K, BLOCK_N] by computing per-row
        # scale pointers.
        scale_k_start = k_start // 32
        scale_k_offs = offs_bk // 16  # [HALF_K] - scale group for each row
        scale_ptrs = (b_scale_ptr
                      + off_experts * stride_bse
                      + offs_bn[None, :] * stride_bsn
                      + (scale_k_start + scale_k_offs[:, None]) * stride_bsk)
        scale_mask = (scale_k_start + scale_k_offs[:, None]) < tl.cdiv(K, 32)
        raw_scales = tl.load(scale_ptrs, mask=scale_mask, other=127)
        # Convert E8M0 -> bf16: [HALF_K, BLOCK_N]
        scales_bf16 = e8m0_scale_to_bf16(raw_scales.to(tl.int32))

        # Apply scales to dequantized values
        lo_scaled = lo_bf16 * scales_bf16
        hi_scaled = hi_bf16 * scales_bf16

        # --- Load A even/odd columns ---
        # A is [M, K] bf16, we need even columns [0,2,4,...] and odd [1,3,5,...]
        # for this K-block starting at k_start
        offs_a_even = k_start + tl.arange(0, HALF_K) * 2      # [0,2,4,...]
        offs_a_odd = k_start + tl.arange(0, HALF_K) * 2 + 1   # [1,3,5,...]

        a_even_ptrs = (a_ptr
                       + (offs_token[:, None] // top_k) * stride_am
                       + offs_a_even[None, :] * stride_ak)
        a_odd_ptrs = (a_ptr
                      + (offs_token[:, None] // top_k) * stride_am
                      + offs_a_odd[None, :] * stride_ak)

        a_even_mask = token_mask[:, None] & (offs_a_even[None, :] < K)
        a_odd_mask = token_mask[:, None] & (offs_a_odd[None, :] < K)

        a_even = tl.load(a_even_ptrs, mask=a_even_mask,
                         other=0.0).to(tl.bfloat16)
        a_odd = tl.load(a_odd_ptrs, mask=a_odd_mask,
                        other=0.0).to(tl.bfloat16)

        # --- Two half-dots ---
        accumulator += tl.dot(a_even, lo_scaled)
        accumulator += tl.dot(a_odd, hi_scaled)

    # Match fused_moe semantics: add bias before router weighting.
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router weight multiplication (in float32 for stability)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    # Write output
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr
              + stride_cm * offs_token[:, None]
              + stride_cn * offs_cn[None, :])
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------


def invoke_fused_moe_mxfp4_kernel(
    A: torch.Tensor,       # [M, K] bf16
    B: torch.Tensor,       # [E, N, K//2] uint8 packed
    C: torch.Tensor,       # output
    B_scale: torch.Tensor,  # [E, N, K//32] uint8
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    K: int,  # logical (unpacked) K dimension
    config: dict,
    B_bias: torch.Tensor | None = None,
):
    assert topk_weights is not None or not mul_routed_weight

    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(
            sorted_token_ids.size(0),
            A.size(0) * top_k * config["BLOCK_SIZE_M"],
        )

    N = B.size(1)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    config = config.copy()
    # Ensure BLOCK_SIZE_K is a multiple of 64 for scale group alignment
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K", 64)
    BLOCK_SIZE_K = max(64, (BLOCK_SIZE_K // 64) * 64)

    # Only pass meta-parameters that our kernel accepts; the generic config
    # may contain extra keys like SPLIT_K that fused_moe_kernel uses.
    BLOCK_SIZE_M = config.get("BLOCK_SIZE_M", 64)
    BLOCK_SIZE_N = config.get("BLOCK_SIZE_N", 64)
    GROUP_SIZE_M = config.get("GROUP_SIZE_M", 8)
    HAS_BIAS = B_bias is not None

    fused_moe_mxfp4_kernel[grid](
        A,
        B,
        C,
        B_bias,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        # Dimensions
        N,
        K,
        EM,
        num_tokens,
        # A strides
        A.stride(0),
        A.stride(1),
        # B strides [E, N, K//2]
        B.stride(0),
        B.stride(1),
        B.stride(2),
        # C strides
        C.stride(1),
        C.stride(2),
        # B_scale strides [E, N, K//32]
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        # B_bias strides [E, N]
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        # Meta-parameters
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        HAS_BIAS=HAS_BIAS,
        top_k=top_k,
        compute_type=tl.bfloat16,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )


# ---------------------------------------------------------------------------
# Standalone MoE forward (legacy apply path)
# ---------------------------------------------------------------------------


def _add_expert_bias(
    output: torch.Tensor,
    expert_bias: torch.Tensor | None,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> None:
    """Add routed expert bias to a tensor of shape [M, top_k, D]."""
    if expert_bias is None:
        return

    if expert_map is None:
        routed_expert_ids = topk_ids
    else:
        routed_expert_ids = expert_map[topk_ids.to(torch.long)]

    valid_expert = (
        (routed_expert_ids >= 0) & (routed_expert_ids < expert_bias.size(0))
    )
    safe_expert_ids = torch.where(valid_expert, routed_expert_ids, 0).to(torch.long)
    bias = expert_bias[safe_expert_ids].to(output.dtype)
    output.add_(bias * valid_expert.unsqueeze(-1).to(output.dtype))


def mxfp4_dequant_fused_experts(
    hidden_states: torch.Tensor,  # [M, K] bf16
    w1: torch.Tensor,             # [E, N, K//2] uint8 packed
    w2: torch.Tensor,             # [E, K, N//2] uint8 packed
    w1_scale: torch.Tensor,       # [E, N, K//32] uint8
    w2_scale: torch.Tensor,       # [E, K, N//32] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Full MoE forward pass using MXFP4 dequant Triton kernels.
    Used by the legacy apply() path in Mxfp4MoEMethod.
    """
    assert hidden_states.dtype == torch.bfloat16
    assert hidden_states.is_contiguous()

    M = hidden_states.size(0)
    K = hidden_states.size(1)
    E, N, _ = w1.size()
    top_k = topk_ids.size(1)

    if global_num_experts == -1:
        global_num_experts = E

    config = try_get_optimal_moe_config(
        w1.size(),
        w2.size(),
        top_k,
        "mxfp4_dequant",
        M,
    )
    if "BLOCK_SIZE_K" not in config or config["BLOCK_SIZE_K"] < 64:
        config["BLOCK_SIZE_K"] = 64

    # Determine activation output dim (gated activations halve N)
    is_no_mul = activation.endswith("_no_mul")
    activation_out_dim = N if is_no_mul else N // 2

    # Allocate workspace
    intermediate_cache1 = torch.empty(
        (M, top_k, N), device=hidden_states.device, dtype=torch.bfloat16
    )
    intermediate_cache2 = torch.empty(
        (M * top_k, activation_out_dim),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    intermediate_cache3 = torch.empty(
        (M, top_k, K), device=hidden_states.device, dtype=torch.bfloat16
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
        )
    )

    # --- First GEMM: hidden_states × w1 ---
    invoke_fused_moe_mxfp4_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        w1_scale,
        topk_weights if apply_router_weight_on_input else None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        top_k,
        K,
        config,
        B_bias=w13_bias,
    )

    # --- Activation ---
    from vllm.model_executor.layers.fused_moe.utils import apply_moe_activation

    apply_moe_activation(
        activation,
        intermediate_cache2,
        intermediate_cache1.view(-1, N),
    )

    # --- Second GEMM: intermediate_cache2 × w2 ---
    invoke_fused_moe_mxfp4_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        activation_out_dim,
        config,
        B_bias=w2_bias,
    )

    # --- Reduction ---
    out = hidden_states if inplace else torch.empty_like(hidden_states)

    ops.moe_sum(intermediate_cache3, out)

    return out


# ---------------------------------------------------------------------------
# Modular kernel class
# ---------------------------------------------------------------------------


class Mxfp4DequantTritonExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    Non-monolithic MoE expert implementation for MXFP4 weights with inline
    dequantization via custom Triton kernel. Keeps weights packed as uint8
    in VRAM and dequantizes per-tile to bf16 inside the GEMM loop.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (None, None)

    @staticmethod
    def _supports_activation(activation: str) -> bool:
        return activation in ["silu", "gelu", "swigluoai", "swiglustep"]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_fi_all2allv_kernels

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # w1 is [E, N, K//2] packed, so we get K from activations
        assert w1.dim() == 3 and w2.dim() == 3
        E, N, _ = w1.size()
        K = a1.size(-1)
        if a1.dim() == 2:
            assert topk_ids.size(0) == a1.size(0)
            M = a1.size(0)
        else:
            assert a1.dim() == 3
            assert a1.size(0) == E
            M = a1.size(1)
        topk = topk_ids.size(1)
        return E, M, N, K, topk

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert hidden_states.is_contiguous()
        assert hidden_states.dim() == 2
        assert hidden_states.dtype == torch.bfloat16

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            "mxfp4_dequant",
            num_tokens,
        )

        # Ensure BLOCK_SIZE_K is compatible
        if "BLOCK_SIZE_K" not in config or config["BLOCK_SIZE_K"] < 64:
            config["BLOCK_SIZE_K"] = 64

        intermediate_cache1 = _resize_cache(
            workspace2, (num_tokens, top_k_num, N)
        )
        cache2_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, cache2_dim)
        )
        intermediate_cache3 = _resize_cache(
            workspace2, (num_tokens, top_k_num, K)
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(
                topk_ids,
                config["BLOCK_SIZE_M"],
                global_num_experts,
                expert_map,
            )
        )

        # --- First GEMM: hidden_states × w1 -> intermediate_cache1 ---
        invoke_fused_moe_mxfp4_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            self.w1_scale,
            topk_weights if apply_router_weight_on_input else None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            K,  # logical K
            config,
            B_bias=self.w1_bias,
        )

        # --- Activation (SiLU/SwiGLU etc.) ---
        self.activation(
            activation,
            intermediate_cache2,
            intermediate_cache1.view(-1, N),
        )

        # --- Second GEMM: intermediate_cache2 × w2 -> intermediate_cache3 ---
        # w2 is [E, hidden_size, intermediate_size//2]
        # logical K for second gemm = cache2_dim (activation output dim)
        invoke_fused_moe_mxfp4_kernel(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            self.w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,  # top_k=1 for second gemm
            cache2_dim,  # logical K
            config,
            B_bias=self.w2_bias,
        )

        # --- Reduction ---
        ops.moe_sum(intermediate_cache3, output)
