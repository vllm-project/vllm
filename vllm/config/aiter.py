# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for ROCm AITER operations."""

from pydantic import Field

import vllm.envs as envs
from vllm.config.utils import config, get_hash_factors, hash_factors


@config
class AITERConfig:
    """Configuration for ROCm AITER (AI Tensor Engine for ROCm) operations.

    AITER provides optimized kernels for AMD GPUs (attention, MoE, GEMM,
    normalization, ...). Historically every feature was toggled by its own
    `VLLM_ROCM_USE_AITER*` environment variable, read once at module import.
    This object is the typed replacement: it is built once with the rest of
    `VllmConfig`, travels to every worker by value, and is part of the
    compilation cache hash.

    Every field defaults to its corresponding environment variable, so leaving
    the config unset reproduces the previous behaviour exactly. `enabled`
    remains the master switch - when it is `False` no AITER path is taken
    regardless of the per-feature fields.

    Only used on ROCm; inert (and unused) on other platforms.
    """

    enabled: bool = Field(default_factory=lambda: envs.VLLM_ROCM_USE_AITER)
    """Master switch for all AITER operations.
    Corresponds to `VLLM_ROCM_USE_AITER`."""

    linear: bool = Field(default_factory=lambda: envs.VLLM_ROCM_USE_AITER_LINEAR)
    """AITER GEMM / linear / quantization ops.
    Corresponds to `VLLM_ROCM_USE_AITER_LINEAR`."""

    linear_hipbmm: bool = Field(
        default_factory=lambda: envs.VLLM_ROCM_USE_AITER_LINEAR_HIPBMM
    )
    """AITER hipBLASLt batched-matmul linear path (CDNA > 2).
    Corresponds to `VLLM_ROCM_USE_AITER_LINEAR_HIPBMM`."""

    moe: bool = Field(default_factory=lambda: envs.VLLM_ROCM_USE_AITER_MOE)
    """AITER fused Mixture-of-Experts ops.
    Corresponds to `VLLM_ROCM_USE_AITER_MOE`."""

    moe_shared_experts: bool = Field(
        default_factory=lambda: envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
    )
    """AITER fused shared-expert path (requires `moe`).
    Corresponds to `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`."""

    moe_situv2_a8w4: bool = Field(
        default_factory=lambda: envs.VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4
    )
    """AITER SITU v2 a8w4 fused-MoE variant (requires `moe`).
    Corresponds to `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4`."""

    moe_dispatch_policy: int = Field(
        default_factory=lambda: envs.VLLM_ROCM_AITER_MOE_DISPATCH_POLICY
    )
    """MoE sorting dispatch policy for AITER fused-MoE kernels.
    Corresponds to `VLLM_ROCM_AITER_MOE_DISPATCH_POLICY`."""

    mla: bool = Field(default_factory=lambda: envs.VLLM_ROCM_USE_AITER_MLA)
    """AITER Multi-head Latent Attention ops.
    Corresponds to `VLLM_ROCM_USE_AITER_MLA`."""

    mha: bool = Field(default_factory=lambda: envs.VLLM_ROCM_USE_AITER_MHA)
    """AITER Multi-Head Attention ops (incl. flash_attn_varlen).
    Corresponds to `VLLM_ROCM_USE_AITER_MHA`."""

    unified_attention: bool = Field(
        default_factory=lambda: envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION
    )
    """AITER Triton unified attention for V1 attention.
    Corresponds to `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION`."""

    fp8bmm: bool = Field(default_factory=lambda: envs.VLLM_ROCM_USE_AITER_FP8BMM)
    """AITER FP8 batched matrix multiply.
    Corresponds to `VLLM_ROCM_USE_AITER_FP8BMM`."""

    fp4bmm: bool = Field(default_factory=lambda: envs.VLLM_ROCM_USE_AITER_FP4BMM)
    """AITER FP4 batched matrix multiply (CDNA 4).
    Corresponds to `VLLM_ROCM_USE_AITER_FP4BMM`."""

    triton_rope: bool = Field(
        default_factory=lambda: envs.VLLM_ROCM_USE_AITER_TRITON_ROPE
    )
    """AITER Triton rotary position embeddings.
    Corresponds to `VLLM_ROCM_USE_AITER_TRITON_ROPE`."""

    triton_gemm: bool = Field(
        default_factory=lambda: envs.VLLM_ROCM_USE_AITER_TRITON_GEMM
    )
    """AITER Triton unquantized GEMM.
    Corresponds to `VLLM_ROCM_USE_AITER_TRITON_GEMM`."""

    custom_all_reduce: bool = Field(
        default_factory=lambda: envs.VLLM_ROCM_USE_AITER_CUSTOM_AR
    )
    """Use AITER's CustomAllreduce as the custom-allreduce backend.
    Corresponds to `VLLM_ROCM_USE_AITER_CUSTOM_AR`."""

    shuffle_kv_cache_layout: bool = Field(
        default_factory=lambda: envs.VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT
    )
    """Shuffle the ROCm KV-cache layout for AITER MLA kernels.
    Corresponds to `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`."""

    def compute_hash(self) -> str:
        """Every field selects kernels / fusion passes / weight layout, so all
        of them feed the compilation cache key."""
        return hash_factors(get_hash_factors(self, set()))
