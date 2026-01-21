# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AITER-based batched MoE implementation using DeepGemm masked-M grouped GEMM.

This module provides an optimized batched experts implementation for AMD ROCm
GPUs using AITER's deepgemm kernel. The kernel operates on the masked-M
batched format where:
  - Activations are shaped (E, T, K) where E=num_experts, T=max_tokens, K=hidden
  - Each expert processes a variable number of tokens tracked by tokens_per_expert
  - Weights must be shuffled using AITER's shuffle_weight for optimal performance

NOTE: Currently only supported on MI300X (gfx942). Support for MI325X (gfx950)
is pending in AITER.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    persistent_masked_m_silu_mul_quant,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT


class AiterBatchedExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    AITER-based batched experts using DeepGemm masked-M grouped GEMM.

    This implementation uses AITER's deepgemm kernel for the GEMMs and
    a Triton kernel for the fused SiLU+Mul+Quantize activation.

    The workflow for each forward pass is:
    1. GEMM1: input @ w1 -> intermediate (gate, up projections)
    2. Activation: SiLU(gate) * up, then quantize to FP8
    3. GEMM2: activated @ w2 -> output (down projection)
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        """
        Initialize AITER batched experts.

        Args:
            moe_config: MoE configuration
            quant_config: Quantization configuration
            max_num_tokens: Maximum number of tokens from a DP Rank
            num_dispatchers: The number of DP dispatchers
        """
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        # AITER deepgemm requires 128x128 block quantization
        assert self.block_shape == [128, 128], (
            f"AITER DeepGemm requires block_shape=[128, 128], got {self.block_shape}"
        )
        assert self.quant_config.use_fp8_w8a8, (
            "AITER DeepGemm requires FP8 W8A8 quantization"
        )

        # Track if weights have been shuffled
        self._weights_shuffled = False

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        # AITER deepgemm only supports MI300X (gfx942) currently
        # TODO: Add gfx950 support when AITER adds it
        if not rocm_aiter_ops.is_deepgemm_enabled():
            return False
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_mi3xx
        # Check for gfx942 specifically (MI300X)
        # on_mi3xx() returns True for gfx942 and gfx950, but deepgemm
        # only supports gfx942 currently
        if not on_mi3xx():
            return False
        try:
            gpu_arch = torch.cuda.get_device_properties("cuda").gcnArchName
            return "gfx942" in gpu_arch
        except Exception:
            return False

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [(kFp8Static128BlockSym, kFp8Dynamic128Sym)]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: str) -> bool:
        return activation in ["silu"]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

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
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dispatchers = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = M if self.max_num_tokens is None else self.max_num_tokens
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (num_experts, max_num_tokens * num_dispatchers, max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, activation_out_dim)
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output)

    def _shuffle_weights_if_needed(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shuffle weights for AITER deepgemm if not already shuffled.

        AITER's deepgemm kernel requires weights to be in a specific
        16x16 tiled layout for optimal memory access patterns.
        """
        if self._weights_shuffled:
            return w1, w2

        # Check if weights are already marked as shuffled
        if hasattr(w1, "is_shuffled") and w1.is_shuffled:
            self._weights_shuffled = True
            return w1, w2

        # Shuffle weights using AITER's shuffle_weight
        w1_shuffled, w2_shuffled = rocm_aiter_ops.shuffle_weights(
            w1, w2, layout=(16, 16)
        )
        self._weights_shuffled = True
        return w1_shuffled, w2_shuffled

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
        """
        Execute batched experts using AITER deepgemm.

        The forward pass consists of:
        1. GEMM1: hidden_states @ w1 -> workspace1 (gate, up projections)
        2. Activation: SiLU(gate) * up, quantize -> a2q, a2q_scale
        3. GEMM2: a2q @ w2 -> output (down projection)
        """
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3, (
            f"Expected 3D hidden_states (E, T, K), got {hidden_states.ndim}D"
        )
        assert self.block_shape is not None

        a1q = hidden_states
        E, _, K = hidden_states.size()
        _, N, _ = w1.size()  # w1 is (E, N, K) for AITER

        E, max_num_tokens, N, K, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        # Allocate output tensor for GEMM1
        # Note: AITER deepgemm writes to Y in-place
        workspace1.zero_()

        # GEMM1: a1q @ w1 -> workspace1
        # Shapes: a1q (E, T, K), w1 (E, N, K), workspace1 (E, T, N)
        rocm_aiter_ops.deepgemm(
            XQ=a1q,
            WQ=w1,
            Y=workspace1,
            group_layout=expert_num_tokens.to(torch.int32),
            x_scale=a1q_scale,
            w_scale=self.w1_scale,
        )

        # Activation: SiLU(gate) * up, then quantize to FP8
        # Use the same Triton kernel as BatchedDeepGemmExperts
        quant_scale_fmt = DeepGemmQuantScaleFMT.FLOAT32
        a2q, a2q_scale = persistent_masked_m_silu_mul_quant(
            workspace1,
            expert_num_tokens,
            quant_scale_fmt=quant_scale_fmt,
        )

        # Zero output before GEMM2
        output.zero_()

        # GEMM2: a2q @ w2 -> output
        # Shapes: a2q (E, T, H), w2 (E, K, H), output (E, T, K)
        rocm_aiter_ops.deepgemm(
            XQ=a2q,
            WQ=w2,
            Y=output,
            group_layout=expert_num_tokens.to(torch.int32),
            x_scale=a2q_scale,
            w_scale=self.w2_scale,
        )
