# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MXFP4 W4A16 fused-MoE method (BF16 activations × MXFP4 weights)."""

from __future__ import annotations

import torch

from vllm.model_executor.hw_agnostic.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel import (
    FusedMoEKernel,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.runner.shared_experts import (  # noqa: E501
    SharedExperts,
)
from vllm.model_executor.hw_agnostic.quantization.mxfp4_utils import _swizzle_mxfp4
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.math_utils import round_up

_MXFP4_BLOCK = 32
# matmul_ogs picks BLOCK_N tiles down to 64 on the Triton path; round
# the per-rank intermediate dim up so every TP rank's slice tiles cleanly.
_MXFP4_INTERMEDIATE_ALIGN = 64


def _select_experts_cls():
    """Return the modular MXFP4 expert class and raise if unsupported.

    Imported lazily so the module-level ``triton_kernels`` monkey-patches
    inside the experts file only run once an MXFP4 layer is actually built.
    """
    from vllm.model_executor.hw_agnostic.layers.fused_moe.experts.triton_mxfp4 import (  # noqa: E501
        OAITritonMxfp4Experts,
    )

    if not OAITritonMxfp4Experts._supports_current_device():
        raise NotImplementedError(
            "No MXFP4 MoE backend supports the current device. The OAI "
            "Triton kernel requires CUDA SM (9, 0) <= cap < (11, 0) or "
            "ROCm gfx9/gfx1x."
        )
    return OAITritonMxfp4Experts


def _swizzle_weights(
    layer: RoutedExperts,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    object,  # PrecisionConfig
    object,  # PrecisionConfig
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Swizzle MXFP4 weights + scales into the ``matmul_ogs`` layout.

    Biases are cast to FP32 because ``matmul_ogs`` reads them as FP32. The
    swizzled scales come back inside ``PrecisionConfig`` objects (one per
    GEMM); the originals on ``layer`` are no longer read and are deleted
    to free their memory.
    """
    from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

    if w13_bias is not None:
        w13_bias = w13_bias.to(torch.float32)
    if w2_bias is not None:
        w2_bias = w2_bias.to(torch.float32)

    w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(w13_weight, w13_weight_scale)
    w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(w2_weight, w2_weight_scale)

    w13_precision_config = PrecisionConfig(
        weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex)
    )
    w2_precision_config = PrecisionConfig(
        weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)
    )

    del layer.w13_weight
    del layer.w2_weight
    del layer.w13_weight_scale
    del layer.w2_weight_scale

    return (
        w13_weight,
        w2_weight,
        w13_precision_config,
        w2_precision_config,
        w13_bias,
        w2_bias,
    )


class Mxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE method (BF16 activations, MXFP4 W4A16 weights)."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "mxfp4"
        self.experts_cls = _select_experts_cls()

        # The swizzled scales live in ``PrecisionConfig`` objects on ``self``
        # because the wrapped tensors returned by ``triton_kernels`` cannot
        # round-trip through ``nn.Parameter`` (no ``.detach()`` support).
        self.w13_precision_config: object | None = None
        self.w2_precision_config: object | None = None

    @property
    def supports_eplb(self) -> bool:
        return True

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        intermediate_size_per_partition = round_up(
            intermediate_size_per_partition, _MXFP4_INTERMEDIATE_ALIGN
        )
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts

        # Fused gate_up_proj (column parallel): uint8 storage for 2x FP4.
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // _MXFP4_BLOCK,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "block"

        # down_proj (row parallel): uint8 storage for 2x FP4.
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // _MXFP4_BLOCK,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _validate_weight_shapes(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None,
        w2_bias: torch.Tensor | None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size

        assert (
            w13.dim() == 3
            and w13.shape[0] == num_experts
            and w13.shape[1] == intermediate_size * 2
            and w13.shape[2] == hidden_size // 2
        )
        assert (
            w13_scale.dim() == 3
            and w13_scale.shape[0] == num_experts
            and w13_scale.shape[1] == intermediate_size * 2
            and w13_scale.shape[2] == hidden_size // _MXFP4_BLOCK
        )
        assert (
            w2.dim() == 3
            and w2.shape[0] == num_experts
            and w2.shape[1] == hidden_size
            and w2.shape[2] == intermediate_size // 2
        )
        assert (
            w2_scale.dim() == 3
            and w2_scale.shape[1] == hidden_size
            and w2_scale.shape[2] == intermediate_size // _MXFP4_BLOCK
        )
        if w13_bias is not None:
            assert (
                w13_bias.dim() == 2
                and w13_bias.shape[0] == num_experts
                and w13_bias.shape[1] == intermediate_size * 2
            )
        if w2_bias is not None:
            assert (
                w2_bias.dim() == 2
                and w2_bias.shape[0] == num_experts
                and w2_bias.shape[1] == hidden_size
            )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)
        self._validate_weight_shapes(w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = _swizzle_weights(
            layer=layer,
            w13_weight=w13,
            w2_weight=w2,
            w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
        )

        layer.w13_weight = w13
        layer.w2_weight = w2
        self.w13_precision_config = w13_scale
        self.w2_precision_config = w2_scale
        if w13_bias is not None:
            layer.w13_bias = w13_bias
        if w2_bias is not None:
            layer.w2_bias = w2_bias

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        prepare_finalize = maybe_make_prepare_finalize(self.moe)
        experts = self.experts_cls(
            moe_config=self.moe, quant_config=self.moe_quant_config
        )
        self.moe_kernel = FusedMoEKernel(prepare_finalize, experts)

    def get_fused_moe_quant_config(
        self,
        layer: RoutedExperts,
    ) -> FusedMoEQuantConfig | None:
        # The swizzle step moved the scale tensors into ``PrecisionConfig``
        # objects on ``self`` (see ``_swizzle_weights``).
        assert self.w13_precision_config is not None
        assert self.w2_precision_config is not None
        return mxfp4_w4a16_moe_quant_config(
            w1_scale=self.w13_precision_config,
            w2_scale=self.w2_precision_config,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
