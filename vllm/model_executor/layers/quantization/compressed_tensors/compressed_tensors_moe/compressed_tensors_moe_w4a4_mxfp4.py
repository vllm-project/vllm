# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


class CompressedTensorsW4A4Mxfp4MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.group_size = 32

        # Select mxfp4 MoE backend
        self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(
            moe,
            activation_key=kMxfp4Dynamic,
        )
        self.use_cutlass_mxfp4 = self.mxfp4_backend == Mxfp4MoeBackend.CUTLASS_MXFP4

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        if self.use_cutlass_mxfp4:
            # Swizzle weight scales from flat checkpoint layout [E, N, K//32]
            # to CUTLASS tiled layout [E, numMTiles*numKTiles*512].
            from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
                swizzle_mxfp4_scales,
            )

            E = layer.w13_weight_scale.shape[0]
            w13_N = layer.w13_weight_scale.shape[1]
            w13_scale_K = layer.w13_weight_scale.shape[2]
            w13_K = w13_scale_K * 32

            w2_M = layer.w2_weight_scale.shape[1]
            w2_scale_N = layer.w2_weight_scale.shape[2]
            w2_N = w2_scale_N * 32

            swizzled_w13 = []
            swizzled_w2 = []
            for e_idx in range(E):
                s13 = layer.w13_weight_scale[e_idx]
                sw13 = swizzle_mxfp4_scales(s13, w13_N, w13_K)
                swizzled_w13.append(sw13.reshape(w13_N, w13_scale_K))
                s2 = layer.w2_weight_scale[e_idx]
                sw2 = swizzle_mxfp4_scales(s2, w2_M, w2_N)
                swizzled_w2.append(sw2.reshape(w2_M, w2_scale_N))
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.stack(swizzled_w13), requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                torch.stack(swizzled_w2), requires_grad=False
            )
        else:
            # Allow for accessing weights and scales in standard way.
            w13 = layer.w13_weight
            w2 = layer.w2_weight
            w13_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale
            w13_bias = getattr(layer, "w13_bias", None)
            w2_bias = getattr(layer, "w2_bias", None)

            # Convert weights to kernel format
            w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
                convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
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

            # For TRITON backends, weights are wrapped tensors from triton_kernels
            # that don't support .detach(). Manually assign parameters.
            if self.mxfp4_backend not in TRITON_BACKENDS:
                replace_parameter(layer, "w13_weight", w13)
                replace_parameter(layer, "w2_weight", w2)
                replace_parameter(layer, "w13_weight_scale", w13_scale)
                replace_parameter(layer, "w2_weight_scale", w2_scale)
            else:
                layer.w13_weight = w13
                layer.w2_weight = w2
                self.w13_precision_config = w13_scale
                self.w2_precision_config = w2_scale

            if w13_bias is not None and w2_bias is not None:
                replace_parameter(layer, "w13_bias", w13_bias)
                replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config and kernel
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                mxfp4_backend=self.mxfp4_backend,
                shared_experts=layer.shared_experts,
                routing_tables=layer._maybe_init_expert_routing_tables(),
            )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
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
            shared_experts_input=shared_experts_input,
        )
