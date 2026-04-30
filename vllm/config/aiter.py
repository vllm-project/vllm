# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for ROCm AITER operations."""

from typing import Any

from pydantic import Field

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

logger = init_logger(__name__)


def _get_aiter_enabled() -> bool:
    return envs.VLLM_ROCM_USE_AITER


def _get_paged_attn() -> bool:
    return envs.VLLM_ROCM_USE_AITER_PAGED_ATTN


def _get_linear() -> bool:
    return envs.VLLM_ROCM_USE_AITER_LINEAR


def _get_moe() -> bool:
    return envs.VLLM_ROCM_USE_AITER_MOE


def _get_rmsnorm() -> bool:
    return envs.VLLM_ROCM_USE_AITER_RMSNORM


def _get_mla() -> bool:
    return envs.VLLM_ROCM_USE_AITER_MLA


def _get_mha() -> bool:
    return envs.VLLM_ROCM_USE_AITER_MHA


def _get_fp4_asm_gemm() -> bool:
    return envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM


def _get_triton_rope() -> bool:
    return envs.VLLM_ROCM_USE_AITER_TRITON_ROPE


def _get_fp8bmm() -> bool:
    return envs.VLLM_ROCM_USE_AITER_FP8BMM


def _get_fp4bmm() -> bool:
    return envs.VLLM_ROCM_USE_AITER_FP4BMM


def _get_unified_attention() -> bool:
    return envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION


def _get_fusion_shared_experts() -> bool:
    return envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS


def _get_triton_gemm() -> bool:
    return envs.VLLM_ROCM_USE_AITER_TRITON_GEMM


@config
class AITERConfig:
    """Configuration for ROCm AITER (AMD Inference TEnsor library for Rocm) operations.

    This class centralizes the configuration for AITER operations on AMD GPUs.
    AITER provides optimized kernels for various operations like attention, MoE,
    GEMM, and normalization on supported AMD hardware (gfx9 architectures).

    All options default to their corresponding environment variable values,
    allowing backward compatibility while providing a structured config interface.
    """

    enabled: bool = Field(default_factory=_get_aiter_enabled)
    """Main toggle for all AITER operations. When False, no AITER operations
    are used regardless of other settings. Corresponds to VLLM_ROCM_USE_AITER."""

    paged_attn: bool = Field(default_factory=_get_paged_attn)
    """Enable AITER paged attention operations.
    Corresponds to VLLM_ROCM_USE_AITER_PAGED_ATTN."""

    linear: bool = Field(default_factory=_get_linear)
    """Enable AITER GEMM and linear/quantization operations.
    Corresponds to VLLM_ROCM_USE_AITER_LINEAR."""

    moe: bool = Field(default_factory=_get_moe)
    """Enable AITER Mixture of Experts (MoE) operations.
    Corresponds to VLLM_ROCM_USE_AITER_MOE."""

    rmsnorm: bool = Field(default_factory=_get_rmsnorm)
    """Enable AITER RMSNorm operations.
    Corresponds to VLLM_ROCM_USE_AITER_RMSNORM."""

    mla: bool = Field(default_factory=_get_mla)
    """Enable AITER Multi-head Latent Attention (MLA) operations.
    Corresponds to VLLM_ROCM_USE_AITER_MLA."""

    mha: bool = Field(default_factory=_get_mha)
    """Enable AITER Multi-Head Attention operations including flash_attn_varlen.
    Corresponds to VLLM_ROCM_USE_AITER_MHA."""

    fp4_asm_gemm: bool = Field(default_factory=_get_fp4_asm_gemm)
    """Enable AITER FP4 assembly GEMM operations.
    Corresponds to VLLM_ROCM_USE_AITER_FP4_ASM_GEMM."""

    triton_rope: bool = Field(default_factory=_get_triton_rope)
    """Enable AITER Triton rotary position embeddings.
    Corresponds to VLLM_ROCM_USE_AITER_TRITON_ROPE."""

    fp8bmm: bool = Field(default_factory=_get_fp8bmm)
    """Enable AITER FP8 batched matrix multiply operations.
    Corresponds to VLLM_ROCM_USE_AITER_FP8BMM."""

    fp4bmm: bool = Field(default_factory=_get_fp4bmm)
    """Enable AITER FP4 batched matrix multiply operations.
    Corresponds to VLLM_ROCM_USE_AITER_FP4BMM."""

    unified_attention: bool = Field(default_factory=_get_unified_attention)
    """Enable AITER Triton unified attention for V1 attention.
    Corresponds to VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION."""

    fusion_shared_experts: bool = Field(default_factory=_get_fusion_shared_experts)
    """Enable AITER fused shared expert operations.
    Corresponds to VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS."""

    triton_gemm: bool = Field(default_factory=_get_triton_gemm)
    """Enable AITER Triton unquantized GEMM operations.
    Corresponds to VLLM_ROCM_USE_AITER_TRITON_GEMM."""

    def compute_hash(self) -> str:
        """Compute a hash of the AITER configuration.

        This hash does not affect the computation graph since AITER settings
        are ROCm-specific runtime optimizations.
        """
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str
