# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    MoEActivation,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    int8_w8a8_moe_quant_config,
    mxfp4_w4a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
    ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    backend_to_kernel_cls,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_fp8_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
    OCP_MX_Scheme,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d,
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = [
    "QuarkMoEMethod",
    "QuarkW8A8Fp8MoEMethod",
    "QuarkOCP_MX_MoEMethod",
    "QuarkNvfp4MoEMethod",
]


class QuarkMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.has_bias = self.moe.has_bias

    @staticmethod
    def get_moe_method(
        quant_config: "QuarkConfig",  # type: ignore # noqa E501 # noqa F821
        module: RoutedExperts,
        layer_name: str,
    ) -> "QuarkMoEMethod":
        layer_quant_config = quant_config._find_matched_config(layer_name, module)

        if layer_quant_config.get("output_tensors") or layer_quant_config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with "
                "output_tensors and bias "
                "quantized are not supported"
            )

        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if quant_config._is_fp8_w4a8(weight_config, input_config):
            return QuarkW4A8Fp8MoEMethod(weight_config, input_config, module.moe_config)
        elif quant_config._is_nvfp4(weight_config, input_config):
            return QuarkNvfp4MoEMethod(
                weight_config, input_config, module.moe_config, quant_config
            )
        elif quant_config._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8Fp8MoEMethod(weight_config, input_config, module.moe_config)
        elif quant_config._is_w_ocp_mx_a_x(weight_config, input_config):
            # All OCP MX schemes (W4A16, W4A8, etc.) handled by QuarkOCP_MX_MoEMethod
            # Backend selection happens inside via oracle
            return QuarkOCP_MX_MoEMethod(weight_config, input_config, module.moe_config)
        elif quant_config._is_static_tensor_w8a8(
            weight_config, input_config
        ) or quant_config._is_dynamic_per_token_w8a8(weight_config, input_config):
            return QuarkW8A8Int8MoEMethod(
                weight_config, input_config, module.moe_config
            )
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")


class QuarkW8A8Fp8MoEMethod(QuarkMoEMethod):
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.weight_quant = weight_config
        self.input_quant = input_config

        self.weight_qscheme = self.weight_quant.get("qscheme")
        self.input_qscheme = self.input_quant.get("qscheme")
        self.weight_dtype = self.weight_quant.get("dtype", "").replace(
            "fp8_e4m3", "fp8"
        )
        self.input_dtype = self.input_quant.get("dtype", "").replace("fp8_e4m3", "fp8")
        per_tensor = (
            self.weight_qscheme == "per_tensor" and self.input_qscheme == "per_tensor"
        )
        per_channel = (
            self.weight_qscheme == "per_channel" and self.input_qscheme == "per_channel"
        )
        self.act_quant_group_shape = (
            GroupShape.PER_TOKEN if per_channel else GroupShape.PER_TENSOR
        )
        if not (per_tensor or per_channel):
            raise ValueError(
                "For FP8 Fused MoE layers, only per-tensor and per-channel "
                "scales for weights and activations are supported. Found "
                f"{self.weight_qscheme}, {self.input_qscheme}"
            )  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization."
            )

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (
            not current_platform.has_device_capability(89)
            or envs.VLLM_TEST_FORCE_FP8_MARLIN
        )
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        self.rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

        self.model_type = getattr(
            get_current_vllm_config().model_config.hf_config, "model_type", None
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.weight_qscheme == "per_tensor":
            # Allocate 2 scales for w1 and w3 respectively.
            # They are combined to a single scale after weight loading.
            if self.model_type != "gpt_oss":
                w13_weight_scale = torch.nn.Parameter(
                    torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
                )
            else:
                # For gpt_oss, the w1(gate) & w3(up) are fused as one.
                # Therefore, only one weight scale for each expert.
                w13_weight_scale = torch.nn.Parameter(
                    torch.ones(num_experts, 1, dtype=torch.float32), requires_grad=False
                )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-TENSOR quantization for RoutedExperts.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        elif self.weight_qscheme == "per_channel":
            # quark's scale is 1 dim.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for RoutedExperts.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.w13_bias, layer.w2_bias = None, None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer. "
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        if current_platform.is_fp8_fnuz():
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = (
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(
                    w13_input_scale, requires_grad=False
                )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(
                    w2_input_scale, requires_grad=False
                )

        # For per-tensor case, Fp8 moe kernel needs single weight scale
        # for w13 per expert. Use max then dequant and requant each expert.
        if self.weight_qscheme == "per_tensor":
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values

            # For gpt_oss, w1 and w3 are fused into a single combined
            # gate_up_proj tensor with size 2*intermediate_size_per_partition
            # and only one scale per expert.
            # Process the entire weight tensor as one shard.
            if self.model_type == "gpt_oss":
                for expert_id in range(layer.local_num_experts):
                    # Process all 2*intermediate_size_per_partition rows at once
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id],
                        layer.w13_weight_scale[expert_id][0],
                    )
                    layer.w13_weight[expert_id], _ = ops.scaled_fp8_quant(
                        dq_weight, max_w13_scales[expert_id]
                    )
            else:
                # For non-gpt_oss, process w1 and w3 shards separately
                for expert_id in range(layer.local_num_experts):
                    start = 0
                    for shard_id in range(2):
                        dq_weight = per_tensor_dequantize(
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            layer.w13_weight_scale[expert_id][shard_id],
                        )
                        (
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            _,
                        ) = ops.scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                        start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

        # quark's scale is 1 dim.
        elif self.weight_qscheme == "per_channel":
            if self.act_quant_group_shape == GroupShape.PER_TOKEN:
                w13_weight_scale = layer.w13_weight_scale.unsqueeze(-1)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False
                )
                w2_weight_scale = layer.w2_weight_scale.unsqueeze(-1)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_weight_scale, requires_grad=False
                )
        # Property to determine if AITER is used
        if self.rocm_aiter_moe_enabled:
            # reshaping weights is required for aiter moe kernel.
            shuffled_w13, shuffled_w2 = rocm_aiter_ops.shuffle_weights(
                layer.w13_weight.data, layer.w2_weight.data
            )

            layer.w13_weight = torch.nn.Parameter(shuffled_w13, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(shuffled_w2, requires_grad=False)

        elif self.use_marlin:
            w13_weight, w2_weight, w13_weight_scale, w2_weight_scale = (
                prepare_fp8_moe_layer_for_marlin(
                    layer,
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                )
            )
            # TODO(rob): once we apply refactor to Quark, switch to using
            # replace_parameter for compatibility with reloading in RL.
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return fp8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            per_act_token_quant=self.input_qscheme == "per_channel",
            per_out_ch_quant=self.weight_qscheme == "per_channel",
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
        if self.rocm_aiter_moe_enabled:
            from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
                rocm_aiter_fused_experts,
            )

            return rocm_aiter_fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                quant_config=self.moe_quant_config,
                moe_config=layer.moe_config,
                expert_map=layer.expert_map,
            )
        elif self.use_marlin:
            assert layer.activation == MoEActivation.SILU, (
                f"{layer.activation} not supported for Marlin MoE."
            )
            return fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                None,
                None,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                topk_weights,
                topk_ids,
                quant_type_id=scalar_types.float8_e4m3fn.id,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                inplace=not self.moe.disable_inplace,
            )
        else:
            from vllm.model_executor.layers.fused_moe import fused_experts

            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=not self.moe.disable_inplace,
                activation=layer.activation,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                quant_config=self.moe_quant_config,
            )


class QuarkW8A8Int8MoEMethod(QuarkMoEMethod):
    """Quark W8A8 INT8 MoE method."""

    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.weight_quant = weight_config
        self.input_quant = input_config
        self.weight_qscheme = self.weight_quant.get("qscheme", "per_tensor")
        self.static_input_scales = not self.input_quant.get("is_dynamic", False)

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
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.weight_qscheme == "per_channel":
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        else:
            # per-tensor: one scalar per expert
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

        # ZERO POINTS (loaded but discarded after loading; kernel uses symmetric)
        w13_input_zero_point = torch.nn.Parameter(
            torch.zeros(num_experts, 2, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_zero_point", w13_input_zero_point)
        set_weight_attrs(w13_input_zero_point, extra_weight_attrs)

        w2_input_zero_point = torch.nn.Parameter(
            torch.zeros(num_experts, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w2_input_zero_point", w2_input_zero_point)
        set_weight_attrs(w2_input_zero_point, extra_weight_attrs)

        if self.weight_qscheme == "per_channel":
            w13_weight_zero_point = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            w2_weight_zero_point = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.int8),
                requires_grad=False,
            )
        else:
            w13_weight_zero_point = torch.nn.Parameter(
                torch.zeros(num_experts, 2, dtype=torch.int8),
                requires_grad=False,
            )
            w2_weight_zero_point = torch.nn.Parameter(
                torch.zeros(num_experts, dtype=torch.int8),
                requires_grad=False,
            )
        layer.register_parameter("w13_weight_zero_point", w13_weight_zero_point)
        set_weight_attrs(w13_weight_zero_point, extra_weight_attrs)
        layer.register_parameter("w2_weight_zero_point", w2_weight_zero_point)
        set_weight_attrs(w2_weight_zero_point, extra_weight_attrs)

        # BIAS
        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.w13_bias, layer.w2_bias = None, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Discard zero points (INT8 fused MoE kernel uses symmetric quant)
        for attr in (
            "w13_input_zero_point",
            "w2_input_zero_point",
            "w13_weight_zero_point",
            "w2_weight_zero_point",
        ):
            if hasattr(layer, attr):
                delattr(layer, attr)

        # For static input scales, collapse per-expert scales to single max
        if self.static_input_scales:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "INT8 MoE layer. Using the maximum across experts "
                    "for each layer."
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        # Per-channel scales: 2D [E, N] -> 3D [E, N, 1] for the int8 MoE kernel.
        if self.weight_qscheme == "per_channel":
            for attr in ("w13_weight_scale", "w2_weight_scale"):
                param = getattr(layer, attr, None)
                if param is not None and param.dim() == 2:
                    replace_parameter(
                        layer,
                        attr,
                        torch.nn.Parameter(
                            param.data.unsqueeze(-1).contiguous(),
                            requires_grad=False,
                        ),
                    )

        # For per-tensor weights, merge w1/w3 scales into single per-expert
        if self.weight_qscheme == "per_tensor":
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values

            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    layer.w13_weight[expert_id][start : start + shard_size, :], _, _ = (
                        ops.scaled_int8_quant(
                            dq_weight,
                            scale=max_w13_scales[expert_id],
                        )
                    )
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.weight_qscheme == "per_channel" and not self.static_input_scales:
            return int8_w8a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                per_act_token_quant=True,
            )
        is_dynamic = not self.static_input_scales
        is_per_channel = self.weight_qscheme == "per_channel"
        return FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            per_act_token_quant=is_dynamic,
            per_out_ch_quant=is_per_channel,
            block_shape=None,
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
        from vllm.model_executor.layers.fused_moe import fused_experts

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=not self.moe.disable_inplace,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )


class QuarkW4A8Fp8MoEMethod(QuarkMoEMethod):
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.weight_quant = weight_config
        self.input_quant = input_config

        assert rocm_aiter_ops.is_fused_moe_enabled(), (
            "W4A8 FP8 MoE requires ROCm AITER fused MoE support."
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        params_dtype = torch.uint32
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 8,  # INT32 packing for W4
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 8,  # INT32 packing for W4
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Per-tensor fp8 weight scales
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Per-channel int4 weight scales
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        shuffled_w13, shuffled_w2 = rocm_aiter_ops.shuffle_weights(
            layer.w13_weight.data, layer.w2_weight.data
        )
        layer.w13_weight = torch.nn.Parameter(shuffled_w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(shuffled_w2, requires_grad=False)

        # INT4-FP8 : offset INT4 w13_weight_scale1 to single w13_weight_scale
        # Fp8 moe kernel needs single fp8 w13_weight_scale for w13 per expert.
        # We won't do requant each expert's fp8 weight (not direct available),
        # instead we adjust half of INT4 w13_weight_scale1 numbers
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values
        assert torch.all(max_w13_scales != 0), "fp8 weight scale cannot be zero."
        for expert_id in range(layer.local_num_experts):
            start = 0
            max_w13_scale_fp8 = max_w13_scales[expert_id]
            for shard_id in range(2):
                if layer.w13_weight_scale[expert_id][shard_id] != max_w13_scale_fp8:
                    int4_rescale = (
                        layer.w13_weight_scale[expert_id][shard_id] / max_w13_scale_fp8
                    )
                    layer.w13_weight_scale_2[expert_id][start : start + shard_size] *= (
                        int4_rescale
                    )
                start += shard_size

        layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)

        # special hack to asm_moe, which takes (weight_scale1 * weight_scale) as post
        # GEMM scaling optimal design - shall apply per-column weight_scale1 before
        # GEMM, and weight_scale post
        for expert_id in range(layer.local_num_experts):
            layer.w13_weight_scale_2[expert_id] *= max_w13_scales[expert_id]
            layer.w2_weight_scale_2[expert_id] *= layer.w2_weight_scale[expert_id]

    def get_fused_moe_quant_config(self, layer):
        return fp8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale_2,
            w2_scale=layer.w2_weight_scale_2,
            per_out_ch_quant=True,
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
        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
            rocm_aiter_fused_experts,
        )

        return rocm_aiter_fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            quant_config=self.moe_quant_config,
            moe_config=layer.moe_config,
            expert_map=layer.expert_map,
        )


class QuarkOCP_MX_MoEMethod(QuarkMoEMethod):
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any] | None,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        if not weight_qscheme == "per_group":
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                f"for weights are supported. Found {weight_qscheme}."
            )  # noqa E501

        self.weight_dtype = self.weight_quant["dtype"].replace("fp", "mxfp")
        if self.input_quant is not None:
            input_quant = self.input_quant["dtype"]
            if input_quant in ["fp4", "fp6_e3m2", "fp6_e2m3"]:
                self.input_dtype = input_quant.replace("fp", "mxfp")
            elif input_quant == "fp8_e4m3":
                self.input_dtype = input_quant.replace("fp8_e4m3", "fp8")
            else:
                raise NotImplementedError(
                    f"Current input dtype {input_quant} is not compatible \
                        with OCP MX (weight) MoE quantization. Please open an issue"
                )
        else:
            self.input_dtype = None

        self.fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype
        )

        if self.ocp_mx_scheme is None:
            raise ValueError(
                f"Unsupported OCP MX dtype combination for MoE: "
                f"input_dtype={self.input_dtype}, weight_dtype={self.weight_dtype}. "
                f"Please check that the combination is supported in OCP_MX_Scheme."
            )

        # TODO(bowenbao): refactor and introduce backends for other OCP MX schemes,
        # use kernel abstraction for all OCP MX MOE implementations.
        self.mxfp4_backend: Mxfp4MoeBackend = Mxfp4MoeBackend.NONE
        self.experts_cls: type[mk.FusedMoEExperts] | None = None
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs (W4A8, TRITON backends)
        self.w13_precision_config = None
        self.w2_precision_config = None

        if self.input_quant is not None:
            self.static_input_scales = not self.input_quant.get("is_dynamic")
        else:
            self.static_input_scales = False

        # Select backend based on OCP MX scheme
        if self.ocp_mx_scheme == "w_mxfp4":
            # W4A16: weight-only MXFP4
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)
        elif self.ocp_mx_scheme == "w_mxfp4_a_fp8" and self.static_input_scales:
            # W4A8: MXFP4 weights + static FP8 activations
            self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(
                moe, activation_key=kFp8StaticTensorSym
            )

        # Validation for unsupported schemes
        if any(
            self.ocp_mx_scheme.endswith(a_scheme)
            for a_scheme in ["a_mxfp4", "a_mxfp6_e3m2", "a_mxfp6_e2m3"]
        ):
            if self.static_input_scales:
                raise NotImplementedError(
                    "QuarkOCP_MX_MoEMethod with static input scales is currently "
                    f"not implemented for OCP MX scheme {self.ocp_mx_scheme}. "
                    "Please open an issue."
                )
        elif self.ocp_mx_scheme.endswith("a_fp8") and not self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX_MoEMethod with dynamic input scales is currently "
                f"not implemented for OCP MX scheme {self.ocp_mx_scheme}. "
                "Please open an issue."
            )

        self.use_rocm_aiter_moe = rocm_aiter_ops.is_fused_moe_enabled()

        self.model_type = getattr(
            get_current_vllm_config().model_config.hf_config, "model_type", None
        )

        # TODO: Remove once all OCP MX schemes use the kernel abstraction
        _AITER_NATIVE_OCP_MX_SCHEMES = ("w_mxfp4", "w_mxfp4_a_mxfp4", "w_mxfp4_a_fp8")
        self.emulate = (
            not current_platform.supports_mx()
            or self.ocp_mx_scheme not in _AITER_NATIVE_OCP_MX_SCHEMES
        ) and (
            self.mxfp4_backend is Mxfp4MoeBackend.NONE or not self.use_rocm_aiter_moe
        )

        if self.emulate:
            # We use the same code path between MXFP4/MXFP6 emulation.
            self.mxfp4_backend = Mxfp4MoeBackend.EMULATION

        # TODO: Remove `self.mxfp4_backend != Mxfp4MoeBackend.NONE` and make it so that
        # all MXFP4 backends use the kernel abstraction.
        if self.mxfp4_backend != Mxfp4MoeBackend.NONE:
            self.experts_cls = backend_to_kernel_cls(self.mxfp4_backend)[0]

        # Log backend selection
        if self.mxfp4_backend != Mxfp4MoeBackend.NONE:
            logger.info_once(
                f"Using {self.mxfp4_backend.value} backend for {self.ocp_mx_scheme}"
            )
        elif self.emulate:
            logger.warning_once(
                f"The current mode (supports_mx={current_platform.supports_mx()}, "
                f"use_rocm_aiter_moe={self.use_rocm_aiter_moe}, "
                f"ocp_mx_scheme={self.ocp_mx_scheme}) "
                "does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

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
        # In case quantization emulation backend is used, there is no need to apply
        # MXFP4-specific padding logic as the compute happens in higher precision.
        if (
            self.mxfp4_backend is not None
            and self.mxfp4_backend != Mxfp4MoeBackend.EMULATION
        ):
            hidden_size, intermediate_size_per_partition = (
                mxfp4_round_up_hidden_size_and_intermediate_size(
                    self.mxfp4_backend, hidden_size, intermediate_size_per_partition
                )
            )
        return hidden_size, intermediate_size_per_partition

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype == "mxfp4":
            assert dim % 2 == 0
            return dim // 2
        else:
            # FP6 packs 4 * 6 = 24 bits on 3 bytes.
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        params_dtype = torch.uint8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                self.get_packed_dim(hidden_size, self.weight_dtype),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                self.get_packed_dim(intermediate_size_per_partition, self.weight_dtype),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)

        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.w13_bias, layer.w2_bias = None, None

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: RoutedExperts):
        # For MXFP4 schemes with native backend, use oracle
        if self.mxfp4_backend != Mxfp4MoeBackend.NONE:
            self._setup_kernel(layer)
            return

        if self.static_input_scales and self.input_dtype == "fp8":
            # firstly, process activations if fp8 static input
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer. "
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

            if current_platform.is_fp8_fnuz():
                # Normalize the weights and scales
                _, _, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    torch.empty_like(layer.w13_weight, dtype=torch.float8_e4m3fn),
                    torch.empty_like(
                        layer.w13_weight_scale, dtype=layer.w13_weight_scale.dtype
                    ),
                    layer.w13_input_scale,
                )
                _, _, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    torch.empty_like(layer.w2_weight, dtype=torch.float8_e4m3fn),
                    torch.empty_like(
                        layer.w2_weight_scale, dtype=layer.w13_weight_scale.dtype
                    ),
                    layer.w2_input_scale,
                )
                # Reset the parameter
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False
                    )
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False
                    )

        # TODO(bowenbao): gradually migrate to oracles.
        # Existing AITER path for w_mxfp4_a_mxfp4 and other schemes
        from aiter.utility.fp4_utils import e8m0_shuffle

        # Pre-shuffle weight scales
        s0, s1, _ = layer.w13_weight_scale.shape
        w13_weight_scale = layer.w13_weight_scale.view(s0 * s1, -1)
        w13_weight_scale = e8m0_shuffle(w13_weight_scale)
        layer.w13_weight_scale.data = w13_weight_scale.view(s0, s1, -1)

        s0, s1, _ = layer.w2_weight_scale.shape
        w2_weight_scale = layer.w2_weight_scale.view(s0 * s1, -1)
        w2_weight_scale = e8m0_shuffle(w2_weight_scale)
        layer.w2_weight_scale.data = w2_weight_scale.view(s0, s1, -1)

        if self.fp4_dtype is not None:
            layer.w13_weight = torch.nn.Parameter(
                layer.w13_weight.view(self.fp4_dtype),
                requires_grad=layer.w13_weight.requires_grad,
            )
            layer.w2_weight = torch.nn.Parameter(
                layer.w2_weight.view(self.fp4_dtype),
                requires_grad=layer.w2_weight.requires_grad,
            )
        # Pre-shuffle weight
        shuffled_w13, shuffled_w2 = rocm_aiter_ops.shuffle_weights(
            layer.w13_weight.data, layer.w2_weight.data
        )

        layer.w13_weight = torch.nn.Parameter(shuffled_w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(shuffled_w2, requires_grad=False)
        layer.w13_weight.is_shuffled = True
        layer.w2_weight.is_shuffled = True

        # Build quant config for AITER path
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        torch.accelerator.empty_cache()

    def _setup_kernel(self, layer: RoutedExperts):
        """Setup kernel using oracle functions for MXFP4 schemes (W4A16, W4A8)."""
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        # Convert weights to kernel format (handles all backend-specific logic)
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_weight_scale=layer.w13_weight_scale,
                w2_weight_scale=layer.w2_weight_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
            )
        )

        # Handle weight/scale assignment based on backend type
        if self.mxfp4_backend in TRITON_BACKENDS or self.mxfp4_backend in (
            Mxfp4MoeBackend.AITER_MXFP4_FP8,
        ):
            # Triton-based backends: w13/w2 are triton_kernels.tensor.Tensor
            # Store on layer for apply(), scales are PrecisionConfig
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale
        else:
            # Standard backends: replace parameters
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        torch.accelerator.empty_cache()

        # Build quant config and kernel
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        # For oracle-based backends (W4A16, W4A8), use make_mxfp4_moe_quant_config
        if self.mxfp4_backend not in (Mxfp4MoeBackend.NONE, Mxfp4MoeBackend.EMULATION):
            # Determine scale source based on backend type
            if self.mxfp4_backend in TRITON_BACKENDS or self.mxfp4_backend in (
                Mxfp4MoeBackend.AITER_MXFP4_FP8,
            ):
                w1_scale = self.w13_precision_config
                w2_scale = self.w2_precision_config
            else:
                w1_scale = layer.w13_weight_scale
                w2_scale = layer.w2_weight_scale

            return make_mxfp4_moe_quant_config(
                mxfp4_backend=self.mxfp4_backend,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
                a1_scale=getattr(layer, "w13_input_scale", None),
                a2_scale=getattr(layer, "w2_input_scale", None),
            )

        # Emulation and other schemes
        if self.ocp_mx_scheme == "w_mxfp4":
            return mxfp4_w4a16_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
            )
        elif self.ocp_mx_scheme == "w_mxfp4_a_fp8":
            return mxfp4_w4a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                block_shape=None,
            )
        elif self.ocp_mx_scheme in ["w_mxfp6_e3m2_a_fp8", "w_mxfp6_e2m3_a_fp8"]:
            raise NotImplementedError(
                "Currently there is no corresponding fused moe quant config configured "
                f"in vLLM for OCP MX scheme {self.ocp_mx_scheme}. Please open an issue."
            )
        else:
            return ocp_mx_moe_quant_config(
                quant_dtype=self.input_dtype,
                weight_dtype=self.weight_dtype,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
            )

    @property
    def is_monolithic(self) -> bool:
        if self.moe_kernel is not None:
            return self.moe_kernel.is_monolithic
        return False

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # For oracle-based kernels (W4A16, W4A8) or emulation kernel
        if self.moe_kernel is not None:
            return self.moe_kernel.apply(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                expert_map=layer.expert_map,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )

        # AITER path
        # TODO: Refactor this to use modular MOE kernel as well.
        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
            rocm_aiter_fused_experts,
        )

        return rocm_aiter_fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            quant_config=self.moe_quant_config,
            moe_config=layer.moe_config,
            expert_map=layer.expert_map,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )


class QuarkNvfp4MoEMethod(QuarkMoEMethod):
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
        quant_config: "QuarkConfig",  # type: ignore # noqa E501 # noqa F821
    ):
        super().__init__(moe)
        self.weight_quant = weight_config
        self.input_quant = input_config
        self.quant_config = quant_config
        self.group_size = 16

        # Select experts implementation.
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=kNvfp4Dynamic,
        )

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
        layer.quant_config = self.quant_config
        weight_dtype = torch.uint8
        weight_scale_dtype = torch.float8_e4m3fn
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        # GEMM 1 - w13 weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # GEMM 2 - w2 weight
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight scales (per-group FP8 scales)
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
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
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Global weight scales (per-tensor FP32 scales)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        w13_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input global scales (per-tensor FP32 scales)
        w13_input_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale_2", w13_input_scale_2)
        set_weight_attrs(w13_input_scale_2, extra_weight_attrs)

        w2_input_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale_2", w2_input_scale_2)
        set_weight_attrs(w2_input_scale_2, extra_weight_attrs)

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        """
        Convert NVFP4 MoE weights into kernel format and setup the kernel.
        """

        if not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            raise ValueError("Different global scales for w1 and w3 is not supported.")

        # Use a single gscale for w13
        w13_weight_scale_2 = torch.maximum(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ).contiguous()

        w2_weight_scale_2 = layer.w2_weight_scale_2

        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=w13_weight_scale_2,
            a13_scale=layer.w13_input_scale_2,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=w2_weight_scale_2,
            a2_scale=layer.w2_input_scale_2,
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w13_input_scale_2", a13_scale)

        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
        replace_parameter(layer, "w2_input_scale_2", a2_scale)

        # Setup modular kernel.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_mk = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                shared_experts=layer.shared_experts,
                routing_tables=layer._maybe_init_expert_routing_tables(),
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale_2,
            a2_scale=layer.w2_input_scale_2,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: Any | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.moe_mk is not None
        return self.moe_mk.apply(
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
