# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    mxfp4_w4a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
    ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.mxfp4 import (
    Mxfp4Backend,
    get_mxfp4_backend,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_fp8_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
    OCP_MX_Scheme,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d,
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

__all__ = ["QuarkMoEMethod", "QuarkW8A8Fp8MoEMethod", "QuarkOCP_MX_MoEMethod"]


class QuarkMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.has_bias = self.moe.has_bias

    @staticmethod
    def get_moe_method(
        quant_config: "QuarkConfig",  # type: ignore # noqa E501 # noqa F821
        module: torch.nn.Module,
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
        elif quant_config._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8Fp8MoEMethod(weight_config, input_config, module.moe_config)
        elif quant_config._is_w_ocp_mx_a_x(weight_config, input_config):
            return QuarkOCP_MX_MoEMethod(weight_config, input_config, module.moe_config)
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
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        params_dtype = torch.float8_e4m3fn

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
            # Add PER-TENSOR quantization for FusedMoE.weight_loader.
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
            # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
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
        self, layer: torch.nn.Module
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
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.rocm_aiter_moe_enabled:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
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
                expert_map=layer.expert_map,
            )
        elif self.use_marlin:
            assert layer.activation == "silu", (
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
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        params_dtype = torch.uint32
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 8,  # INT32 packing for W4
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
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
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
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

        self.mxfp4_backend: Mxfp4Backend | None = None
        if self.ocp_mx_scheme == "w_mxfp4":
            self.mxfp4_backend = get_mxfp4_backend(moe.is_lora_enabled)

        if self.input_quant is not None:
            self.static_input_scales = not self.input_quant.get("is_dynamic")
        else:
            self.static_input_scales = False

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

        self._emulate = (
            not current_platform.supports_mx()
            or not self.ocp_mx_scheme.startswith("w_mxfp4")
        ) and (self.mxfp4_backend is None or not self.use_rocm_aiter_moe)

        self.emulate = True if self.model_type == "gpt_oss" else self._emulate

        if self.emulate:
            logger.warning_once(
                f"The current mode (supports_mx={current_platform.supports_mx()}, "
                f"use_mxfp4_aiter_moe={self.use_rocm_aiter_moe}, "
                f"ocp_mx_scheme={self.ocp_mx_scheme}) "
                "does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )
        else:
            logger.warning_once(
                "The current mode supports native MoE MXFP4 computation"
            )

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
        layer: torch.nn.Module,
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
        if self.model_type == "gpt_oss":
            if current_platform.is_rocm():
                intermediate_size_per_partition_after_pad = round_up(
                    intermediate_size_per_partition, 256
                )
            else:
                intermediate_size_per_partition_after_pad = round_up(
                    intermediate_size_per_partition, 64
                )
        else:
            intermediate_size_per_partition_after_pad = intermediate_size_per_partition

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                self.get_packed_dim(hidden_size, self.weight_dtype),
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
                self.get_packed_dim(
                    intermediate_size_per_partition_after_pad, self.weight_dtype
                ),
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
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // OCP_MX_BLOCK_SIZE,
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
                    2 * intermediate_size_per_partition_after_pad,
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

    def process_weights_after_loading(self, layer):
        if self.static_input_scales:
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
                    torch.empty_like(layer.w13_weight, dtype=torch.float8_e4m3fnuz),
                    torch.empty_like(
                        layer.w13_weight_scale, dtype=layer.w13_weight_scale.dtype
                    ),
                    layer.w13_input_scale,
                )
                _, _, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    torch.empty_like(layer.w2_weight, dtype=torch.float8_e4m3fnuz),
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

        # secondly, process mxfp weights
        if self.emulate:
            torch.cuda.empty_cache()
            return

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

        torch.cuda.empty_cache()

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
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

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not self.emulate:
            if (
                self.model_type == "gpt_oss"
                and self.mxfp4_backend == Mxfp4Backend.TRITON
            ):
                raise NotImplementedError(
                    "Triton kernel implemented fused MoE for GPT_OSS model "
                    "in Quark(MoE) format is not integrated or provided yet."
                )

            else:
                from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
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
                    expert_map=layer.expert_map,
                )
        else:
            from vllm.model_executor.layers.fused_moe import fused_experts

            return fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=not self.moe.disable_inplace,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                expert_map=layer.expert_map,
                quant_config=self.moe_quant_config,
            )
