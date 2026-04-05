# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
B12x MoE expert backend for NVFP4 on SM120 (Blackwell) GPUs.

This backend replaces FlashInfer CUTLASS for fused MoE FP4 operations,
providing significantly higher throughput on Blackwell-class hardware.
It reuses the same weight preparation path as FLASHINFER_CUTLASS
(swizzled block-scales, [w3,w1] ordering for gated activations) and
delegates the actual computation to ``b12x.integration.tp_moe.b12x_moe_fp4``.
"""

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Per-device workspace pool (mirrors SGLang's pattern).
# The pool is stream-aware inside b12x, so one pool per device suffices.
# ---------------------------------------------------------------------------
_B12X_MOE_WORKSPACE_POOLS: dict[int, Any] = {}


def _get_b12x_workspace_pool(device: torch.device):
    """Return (or lazily create) the b12x workspace pool for *device*."""
    from b12x.integration.tp_moe import allocate_tp_moe_workspace_pool

    device_idx = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    pool = _B12X_MOE_WORKSPACE_POOLS.get(device_idx)
    if pool is None:
        pool = allocate_tp_moe_workspace_pool()
        _B12X_MOE_WORKSPACE_POOLS[device_idx] = pool
    return pool


def _has_b12x() -> bool:
    """Return True when the b12x MoE kernel is importable."""
    try:
        from b12x.integration.tp_moe import b12x_moe_fp4  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Expert implementation
# ---------------------------------------------------------------------------


class B12xExperts(mk.FusedMoEExpertsModular):
    """
    NVFP4 fused-MoE expert backend powered by b12x kernels.

    Weight contract
    ~~~~~~~~~~~~~~~
    This class relies on the *same* weight preparation that
    ``FLASHINFER_CUTLASS`` uses (handled by
    ``prepare_nvfp4_moe_layer_for_fi_or_cutlass`` in the quantisation
    layer):

    * Weights are packed uint8 (two FP4 values per byte).
    * Block-scales are swizzled via ``swizzle_blockscale``.
    * For gated activations the W13 tensor is reordered to ``[w3, w1]``.

    The ``FusedMoEQuantConfig`` supplies the following tensors consumed
    here (via the property helpers inherited from ``FusedMoEExperts``):

    ==================  ==========================================
    quant_config attr   b12x parameter
    ==================  ==========================================
    ``a1_gscale``       ``a1_gscale``   (reciprocal input scale)
    ``g1_alphas``       ``w1_alphas``   (alpha = input_scale * weight_scale_2)
    ``w1_scale``        ``w1_blockscale`` (swizzled, viewed as int32 for FI; raw fp8 for b12x)
    ``a2_gscale``       ``a2_gscale``
    ``g2_alphas``       ``w2_alphas``
    ``w2_scale``        ``w2_blockscale``
    ==================  ==========================================
    """

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        assert quant_config.weight_quant_dtype == "nvfp4", (
            f"B12xExperts only supports nvfp4 weights, got "
            f"{quant_config.weight_quant_dtype}"
        )

        self.device = moe_config.device
        self.num_experts = moe_config.num_local_experts
        self.out_dtype = moe_config.in_dtype

    # ------------------------------------------------------------------
    # process_weights_after_loading: fuse input scales into g{1,2}_alphas
    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Fuse per-expert activation scales into the alpha tensors.

        This matches the FlashInferExperts behaviour: after this call
        ``g1_alphas == w13_weight_scale_2 * w13_input_scale`` and
        ``g2_alphas == w2_weight_scale_2 * w2_input_scale``.
        The quant_config tensors are updated in-place so that the EPLB
        rearrangement pathway stays in sync.
        """
        if self.quant_config.use_nvfp4_w4a4:
            layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
            layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    # ------------------------------------------------------------------
    # Static capabilities
    # ------------------------------------------------------------------
    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and _has_b12x()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # b12x supports SiLU-gated (SwiGLU) activations.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
        ]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # b12x performs its own FP4 quantisation internally.
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # b12x fuses topk-weight application and expert reduction internally.
        return TopKWeightAndReduceNoOP()

    # ------------------------------------------------------------------
    # workspace_shapes
    # ------------------------------------------------------------------
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # b12x manages its own internal workspace via the pool.
        # We only need to declare the output shape for the framework.
        workspace1 = (M, K)
        workspace2 = (0,)
        # nvfp4 output is packed int8 -> hidden dim is 2 * K
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    # ------------------------------------------------------------------
    # apply -- the hot path
    # ------------------------------------------------------------------
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        from b12x.integration.tp_moe import b12x_moe_fp4

        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for B12xExperts"
        )
        assert self.g1_alphas is not None and self.g2_alphas is not None, (
            "g1_alphas and g2_alphas must not be None for B12xExperts"
        )
        assert self.a1_gscale is not None and self.a2_gscale is not None, (
            "a1_gscale and a2_gscale must not be None for B12xExperts"
        )

        workspace_pool = _get_b12x_workspace_pool(hidden_states.device)

        b12x_moe_fp4(
            a=hidden_states,
            a1_gscale=self.a1_gscale,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_alphas=self.g1_alphas,
            a2_gscale=self.a2_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_alphas=self.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=(
                apply_router_weight_on_input
                if apply_router_weight_on_input is not None
                else False
            ),
            workspace=workspace_pool,
            output=output,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for B12xExperts")
