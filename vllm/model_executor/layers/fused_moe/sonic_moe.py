# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Sonic MoE integration for vLLM.

Sonic MoE is a high-performance Mixture-of-Experts implementation optimized
for NVIDIA Hopper GPUs (H100/H200). This module provides integration with
vLLM's modular MoE kernel infrastructure.

Reference: https://github.com/Dao-AILab/sonic-moe
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

logger = init_logger(__name__)

# Lazy import for optional Sonic MoE dependency
_SONICMOE_AVAILABLE: bool | None = None


def _check_sonicmoe_available() -> bool:
    """Check if Sonic MoE is available."""
    global _SONICMOE_AVAILABLE
    if _SONICMOE_AVAILABLE is None:
        try:
            import sonicmoe  # noqa: F401

            _SONICMOE_AVAILABLE = True
        except ImportError:
            _SONICMOE_AVAILABLE = False
            logger.warning_once(
                "Sonic MoE is not installed. Install it from "
                "https://github.com/Dao-AILab/sonic-moe for Hopper GPU "
                "optimized MoE kernels."
            )
    return _SONICMOE_AVAILABLE


def _is_hopper_gpu() -> bool:
    """Check if the current GPU is a Hopper architecture GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 9  # Hopper is compute capability 9.0
    except Exception:
        return False


def is_sonic_moe_supported(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> bool:
    """
    Check if Sonic MoE can be used for the given configuration.

    Requirements:
    - Sonic MoE package must be installed
    - Must be running on Hopper GPU (H100/H200)
    - Hidden states must be bfloat16 or float16
    - Weights must be in compatible format
    """
    if not _check_sonicmoe_available():
        logger.debug_once("SonicMoE disabled: sonicmoe package not available.")
        return False

    if not _is_hopper_gpu():
        logger.debug_once("SonicMoE disabled: requires Hopper GPU (H100/H200).")
        return False

    # Check data types
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        logger.debug_once(
            f"SonicMoE disabled: hidden_states must be bfloat16 or float16, "
            f"got {hidden_states.dtype}."
        )
        return False

    # Check weight dimensions (should be 3D: [E, N, K])
    if w1.dim() != 3 or w2.dim() != 3:
        logger.debug_once(
            f"SonicMoE disabled: weights must be 3D, got w1.dim={w1.dim()}, "
            f"w2.dim={w2.dim()}."
        )
        return False

    return True


def _convert_weights_to_sonic_format(
    w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert vLLM weight format to Sonic MoE format.

    vLLM format: [E, N, K] (num_experts, intermediate_size, hidden_size)
    Sonic format: [I, H, E] where the weight is accessed via permute(1, 2, 0)

    For Sonic MoE's functional API, weights are passed as [I, H, E] format
    which is then internally permuted to [E, H, I] for computation.
    """
    # vLLM: [E, N, K] -> Sonic expects [N, K, E] which it then permutes
    # w1: [E, intermediate*2, hidden] for GLU activations
    # w2: [E, hidden, intermediate]
    w1_sonic = w1.permute(1, 2, 0).contiguous()  # [N, K, E] = [I*2, H, E]
    w2_sonic = w2.permute(1, 2, 0).contiguous()  # [K, N, E] = [H, I, E]
    return w1_sonic, w2_sonic


class SonicMoEExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    Sonic MoE experts implementation for vLLM.

    This class wraps Sonic MoE's high-performance kernels and integrates them
    with vLLM's modular MoE infrastructure. Sonic MoE is optimized for
    NVIDIA Hopper GPUs and provides significant speedups for MoE inference.

    Key features:
    - Optimized for H100/H200 GPUs with tensor core acceleration
    - Supports GLU activations (SwiGLU, GeGLU, etc.)
    - Fused routing, permutation, and expert computation
    """

    # Activation string to Sonic MoE ActivationType mapping
    _ACTIVATION_MAP = {
        "silu": "SWIGLU",
        "gelu": "GEGLU",
        "relu": "RELU",
    }

    def __init__(
        self,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config)
        if not _check_sonicmoe_available():
            raise ImportError(
                "Sonic MoE is not installed. Please install it from "
                "https://github.com/Dao-AILab/sonic-moe"
            )

        # Import Sonic MoE components
        from sonicmoe.count_cumsum import count_cumsum
        from sonicmoe.enums import ActivationType
        from sonicmoe.functional import (
            TC_topk_router_metadata,
            _DownProjection,
            _UpProjection,
        )

        self._ActivationType = ActivationType
        self._TC_topk_router_metadata = TC_topk_router_metadata
        self._UpProjection = _UpProjection
        self._DownProjection = _DownProjection
        self._count_cumsum = count_cumsum

        # Cache for converted weights (avoid repeated conversions)
        self._w1_sonic_cache: torch.Tensor | None = None
        self._w2_sonic_cache: torch.Tensor | None = None
        self._w1_cache_id: int = -1
        self._w2_cache_id: int = -1
        self._stream_id = torch.cuda.current_stream().cuda_stream

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return False  # Sonic MoE handles its own chunking

    def supports_expert_map(self) -> bool:
        return False  # Sonic MoE doesn't support expert parallelism yet

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Sonic MoE applies weights and reduces internally
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Sonic MoE manages its own workspace internally
        # We just need to allocate the output buffer
        workspace13 = (0,)  # Not needed
        workspace2 = (0,)  # Not needed
        output_shape = (M, K)
        return (workspace13, workspace2, output_shape)

    def _get_activation_type(self, activation: str):
        """Convert vLLM activation string to Sonic MoE ActivationType."""
        sonic_act_name = self._ACTIVATION_MAP.get(activation, "SWIGLU")
        return getattr(self._ActivationType, sonic_act_name)

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        Apply Sonic MoE experts to the hidden states.

        This method converts the weights to Sonic MoE format and uses
        the optimized _UpProjection and _DownProjection kernels.
        """
        M, K = hidden_states.shape
        num_experts = w1.size(0)
        topk = topk_ids.size(1)

        # Convert weights to Sonic format (cached for efficiency)
        if self._w1_cache_id != id(w1) or self._w2_cache_id != id(w2):
            self._w1_sonic_cache, self._w2_sonic_cache = (
                _convert_weights_to_sonic_format(w1, w2)
            )
            self._w1_cache_id = id(w1)
            self._w2_cache_id = id(w2)
        w1_sonic = self._w1_sonic_cache
        w2_sonic = self._w2_sonic_cache

        # Get Sonic MoE activation type
        activation_type = self._get_activation_type(activation)

        # Compute routing metadata using Sonic MoE's utilities
        topk_indices_flat = topk_ids.view(-1)
        expert_frequency, expert_frequency_offset = self._count_cumsum(
            topk_indices_flat, num_experts, do_cumsum=True
        )

        (
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        ) = self._TC_topk_router_metadata(topk_ids, expert_frequency_offset, topk)

        total_expert_freq = M * topk

        # Up projection
        y1, z = self._UpProjection.apply(
            hidden_states,  # x
            w1_sonic,  # w1
            None,  # b1 (no bias)
            expert_frequency_offset,
            total_expert_freq,
            topk,
            self._stream_id,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            False,  # is_varlen_K
            activation_type,
            True,  # is_inference_mode_enabled
        )

        # Apply router weights if needed
        if not apply_router_weight_on_input:
            router_scores = topk_weights
        else:
            router_scores = torch.ones_like(topk_weights)

        # Down projection
        o = self._DownProjection.apply(
            y1,
            z,
            w2_sonic,
            None,  # b2 (no bias)
            router_scores,
            expert_frequency_offset,
            M,
            topk,
            self._stream_id,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            False,  # is_varlen_K
            activation_type,
        )

        # Copy result to output
        output.copy_(o)


def create_sonic_moe_experts(
    quant_config: FusedMoEQuantConfig,
) -> SonicMoEExperts:
    """Factory function to create SonicMoEExperts instance."""
    return SonicMoEExperts(quant_config)


def sonic_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """
    High-level API to run Sonic MoE on the given inputs.

    This is a convenience function that creates the modular kernel
    and runs it on the given inputs.

    Args:
        hidden_states: Input tensor of shape (M, K)
        w1: First expert weight tensor of shape (E, N, K)
        w2: Second expert weight tensor of shape (E, K, N)
        topk_weights: Router weights of shape (M, topk)
        topk_ids: Router expert indices of shape (M, topk)
        quant_config: Quantization configuration
        inplace: If True, perform the operation in-place
        activation: Activation function name
        global_num_experts: Total number of experts
        expert_map: Expert mapping for expert parallelism
        apply_router_weight_on_input: Whether router weight is applied on input

    Returns:
        Output tensor of shape (M, K)
    """
    fused_experts = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(defer_input_quant=True),
        SonicMoEExperts(quant_config),
    )

    return fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
