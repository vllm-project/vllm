# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online MXFP4 (microscaling FP4, block-32) quantization methods."""

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

if TYPE_CHECKING:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe import (
        FusedMoEQuantConfig,
        RoutedExperts,
    )

from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    _Fp8OnlineLinearBase,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    mxfp4_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.utils import replace_parameter

# MXFP4 block size (elements per shared e8m0 scale), same as the checkpoint
# and MoE oracle paths.
MXFP4_BLOCK_SIZE = 32


def _quantize_mxfp4_moe_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch quantization: bf16/fp16 stacked expert weights -> MXFP4.

    Returns packed FP4 weights (uint8, two values per byte) and per-block
    (group-32) e8m0 scales (uint8), one pair per expert.
    """
    num_experts = weight.size(0)
    first_quant, first_scale = mxfp4_quantize(weight[0])
    # Pre-allocate the output tensors rather than stacking.
    # This is important for consistent memory layout.
    w_quant = torch.empty(
        (num_experts, *first_quant.shape),
        dtype=first_quant.dtype,
        device=weight.device,
    )
    w_scales = torch.empty(
        (num_experts, *first_scale.shape),
        dtype=first_scale.dtype,
        device=weight.device,
    )
    w_quant[0] = first_quant
    w_scales[0] = first_scale
    for i in range(1, num_experts):
        w_quant[i], w_scales[i] = mxfp4_quantize(weight[i])

    return w_quant, w_scales


class Mxfp4OnlineLinearMethod(_Fp8OnlineLinearBase):
    """Online MXFP4 linear method.
    Loads bf16/fp16 checkpoints and quantizes weights to MXFP4 (microscaling
    FP4 with block-32 scales) during weight loading.
    """

    def __init__(self):
        super().__init__()
        self.kernel = init_mxfp4_linear_kernel()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % MXFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP4 requires input_size_per_partition "
                f"({input_size_per_partition}) to be divisible by "
                f"{MXFP4_BLOCK_SIZE}."
            )

        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        weight_fp4, weight_scale = mxfp4_quantize(layer.weight.contiguous())

        layer.input_scale = None
        replace_parameter(layer, "weight", weight_fp4.data)
        replace_parameter(layer, "weight_scale", weight_scale.data)

        self.kernel.process_weights_after_loading(layer)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class Mxfp4OnlineMoEMethod(OnlineMoEMethodBase):
    """MoE method for online MXFP4 (block) quantization."""

    mxfp4_backend: Mxfp4MoeBackend
    experts_cls: "type[mk.FusedMoEExperts] | None"
    activation_quant_key = kMxfp4Dynamic

    def __init__(self, *, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.weight_block_size: list[int] = [1, MXFP4_BLOCK_SIZE]
        self.weight_scale_name = "weight_scale"

        self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(
            config=self.moe, activation_key=self.activation_quant_key
        )

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if (
            hidden_size % MXFP4_BLOCK_SIZE != 0
            or intermediate_size_per_partition % MXFP4_BLOCK_SIZE != 0
        ):
            raise ValueError(
                "Online MXFP4 MoE requires hidden/intermediate sizes divisible "
                f"by {MXFP4_BLOCK_SIZE}."
            )

        super().create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        layer.weight_block_size = [1, MXFP4_BLOCK_SIZE]

    def _setup_kernel(
        self,
        layer: "RoutedExperts",
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None,
        w2_bias: torch.Tensor | None,
    ) -> None:
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
            )
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale)
        replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale)
        if w13_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
        if w2_bias is not None:
            replace_parameter(layer, "w2_bias", w2_bias)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> "FusedMoEQuantConfig | None":
        w1_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            layer=layer,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            layer._already_called_process_weights_after_loading = True
            return

        layer.w13_input_scale = None
        layer.w2_input_scale = None

        w13, w13_scale = _quantize_mxfp4_moe_weight(layer.w13_weight)
        w2, w2_scale = _quantize_mxfp4_moe_weight(layer.w2_weight)

        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            getattr(layer, "w13_bias", None),
            getattr(layer, "w2_bias", None),
        )

        layer._already_called_process_weights_after_loading = True
