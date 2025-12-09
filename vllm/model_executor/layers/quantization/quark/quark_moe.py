# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Callable, Optional, Union

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEConfig,
                                                  FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig, fp8_w8a8_moe_quant_config,
    mxfp4_w4a4_moe_quant_config)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    OCP_MX_BLOCK_SIZE, _swizzle_mxfp4)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import round_up
import aiter
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    is_rocm_aiter_moe_enabled,
    use_mxfp4_aiter_moe,
)
logger = init_logger(__name__)

__all__ = [
    "QuarkMoEMethod", "QuarkW8A8Fp8MoEMethod", "QuarkW4A4MXFp4MoEMethod"
]


class QuarkMoEMethod(FusedMoEMethodBase):

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    @staticmethod
    def get_moe_method(
            quant_config: "QuarkConfig",  # type: ignore # noqa E501 # noqa F821
            module: torch.nn.Module,
            layer_name: str) -> "QuarkMoEMethod":
        layer_quant_config = quant_config._find_matched_config(
            layer_name, module)

        if (layer_quant_config.get("output_tensors")
                or layer_quant_config.get("bias")):
            raise NotImplementedError("Currently, Quark models with "
                                      "output_tensors and bias "
                                      "quantized are not supported")
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if quant_config._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8Fp8MoEMethod(weight_config, input_config,
                                         module.moe_config)
        elif quant_config._is_mx_fp4(weight_config, input_config):
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
            model_type = getattr(vllm_config.model_config.hf_config,
                                 "model_type", None)
            if model_type == "gpt_oss":
                return QuarkW4MXFp4MoEMethod_OSS(weight_config, input_config,
                                                 module.moe_config)
            else:
                return QuarkW4MXFp4MoEMethod(weight_config, input_config,
                                             module.moe_config)
        elif quant_config._is_ocp_mx(weight_config, input_config):
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

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_tensor"
                and input_qscheme == "per_tensor"):
            raise ValueError(
                "For FP8 Fused MoE layers, only per-tensor scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}")  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                         2,
                                                         dtype=torch.float32),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None.")
            if (not all_close_1d(layer.w13_input_scale)
                    or not all_close_1d(layer.w2_input_scale)):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer. ")
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False)
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False)

        if current_platform.is_fp8_fnuz():
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = \
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale,
                    layer.w13_input_scale)
            w2_weight, w2_weight_scale, w2_input_scale = \
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w2_weight, layer.w2_weight_scale,
                    layer.w2_input_scale)
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                  requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale,
                                                        requires_grad=False)
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(w13_input_scale,
                                                           requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                 requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale,
                                                       requires_grad=False)
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(w2_input_scale,
                                                          requires_grad=False)

        # Fp8 moe kernel needs single weight scale for w13 per expert.
        # We take the max then dequant and requant each expert.
        assert layer.w13_weight_scale is not None
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values
        for expert_id in range(layer.local_num_experts):
            start = 0
            for shard_id in range(2):
                dq_weight = per_tensor_dequantize(
                    layer.w13_weight[expert_id][start:start + shard_size, :],
                    layer.w13_weight_scale[expert_id][shard_id])
                layer.w13_weight[expert_id][
                    start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                        dq_weight, max_w13_scales[expert_id])
                start += shard_size

        layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                    requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.fused_experts is None

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `QuarkW8A8Fp8MoEMethod` yet.")

        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype)

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=True,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            activation=activation)


class QuarkOCP_MX_MoEMethod(QuarkMoEMethod):
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_group" and input_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}"
            )  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")

        self.weight_dtype = self.weight_quant["dtype"].replace("fp", "mxfp")
        self.input_dtype = self.input_quant["dtype"].replace("fp", "mxfp")
        self.fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype
        )

        if self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX_MoEMethod with static input scales is currently "
                "not implemented. Please open an issue."
            )

        self.use_rocm_aiter_moe = rocm_aiter_ops.is_fused_moe_enabled()

        self.emulate = not current_platform.supports_mx() or not (
            self.use_rocm_aiter_moe and self.ocp_mx_scheme == "w_mxfp4_a_mxfp4"
        )
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

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
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
            torch.empty(
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

    def process_weights_after_loading(self, layer):
        if self.emulate:
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
        return ocp_mx_moe_quant_config(
            quant_dtype=self.input_dtype,
            weight_dtype=self.weight_dtype,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
        )

    @property
    def allow_inplace(self) -> bool:
        return True

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, _ = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        if not self.emulate:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
                rocm_aiter_fused_experts,
            )

            out = rocm_aiter_fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                quant_config=self.moe_quant_config,
                expert_map=expert_map,
            )
        else:
            from vllm.model_executor.layers.fused_moe import fused_experts

            out = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
                quant_config=self.moe_quant_config,
            )
        return out


class QuarkW4MXFp4MoEMethod(QuarkMoEMethod):

    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_group"
                and input_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}")  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")

        if self.static_input_scales:
            raise NotImplementedError(
                "QuarkW4MXFp4MoEMethod with static input scales is currently "
                "not implemented. Please open an issue.")

        self.emulate = not current_platform.supports_mx() or not (
            use_mxfp4_aiter_moe()
        )

        if self.emulate:
            logger.warning_once(
                f"The current mode (supports_mx={current_platform.supports_mx()}, "
                f"use_mxfp4_aiter_moe={use_mxfp4_aiter_moe()}, "
                "does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision.")
        else:
            self.emulate = False
            logger.info_once(
                "The current mode supports native MoE MXFP4 computation"
            )

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})

        params_dtype = torch.uint8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // 2,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // 2,
            dtype=params_dtype),
                                       requires_grad=False)
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
        
    def process_weights_after_loading(self, layer):
        if self.emulate:
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
        torch.cuda.empty_cache()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.fused_experts is None

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `QuarkW4MXFp4MoEMethod` yet.")

        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype)

        if not self.emulate:
            from aiter import ActivationType, QuantType
            from aiter.fused_moe import fused_moe

            aiter_acts = {
                ActivationType.No.name.lower(): ActivationType.No,
                ActivationType.Silu.name.lower(): ActivationType.Silu,
                ActivationType.Gelu.name.lower(): ActivationType.Gelu,
            }
            assert activation in aiter_acts, (
                f"Aiter CK fp4 MoE doesn't support activation {activation}"
            )
            if hasattr(torch, "float4_e2m1fn_x2"):
                w13_weight = layer.w13_weight.view(torch.float4_e2m1fn_x2)
                w2_weight = layer.w2_weight.view(torch.float4_e2m1fn_x2)
            else:
                w13_weight = layer.w13_weight
                w2_weight = layer.w2_weight

            out = fused_moe(
                x,
                w13_weight,
                w2_weight,
                topk_weights,
                topk_ids,
                quant_type=QuantType.per_1x32,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                activation=aiter_acts[activation],
                doweight_stage1=False,
            )
        else:
            from vllm.model_executor.layers.fused_moe import fused_experts

            out = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                global_num_experts=global_num_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
                quant_config=self.moe_quant_config,
            )
        return out


class QuarkW4MXFp4MoEMethod_OSS(QuarkMoEMethod):

    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}")  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")
        if not current_platform.supports_mx():
            self.emulate = True
            logger.warning_once(
                "The current platform does not support native MXFP4 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision.")
        else:
            self.emulate = True
            logger.warning_once(
                "The current platform supports native MXFP4 "
                "computation, but kernels are not yet integrated in vLLM. "
                "Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision.")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, unpadded_hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        self.num_experts = num_experts
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
        mxfp4_block = 32
        weight_dtype = torch.uint8
        weight_scale_dtype = torch.uint8
        per_tensor_fp8_act_scale_dtype = torch.bfloat16
        self.intermediate_size_per_partition = intermediate_size_per_partition
        intermediate_size_per_partition_after_pad = \
            intermediate_size_per_partition

        if current_platform.is_rocm():
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 256)  # 2880 -> 2944
        else:
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 64)

        self.unpadded_hidden_size = unpadded_hidden_size
        self.hidden_pad = hidden_size - unpadded_hidden_size
        self.intermediate_pad = intermediate_size_per_partition_after_pad - intermediate_size_per_partition
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    dtype=per_tensor_fp8_act_scale_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)
            w2_input_scale = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    dtype=per_tensor_fp8_act_scale_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer):

        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

        w13_bias = layer.w13_bias.to(torch.float32)
        w2_bias = layer.w2_bias.to(torch.float32)

        layer.w13_bias = Parameter(w13_bias, requires_grad=False)
        layer.w2_bias = Parameter(w2_bias, requires_grad=False)

        # FIXME warp need to be adjusted based on batch size
        # only apply to  batched mode
        if self.moe.use_ep:
            num_warps = 4 if envs.VLLM_MOE_DP_CHUNK_SIZE <= 512 else 8
        else:
            num_warps = 8

        if envs.VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4:
            from aiter.ops.shuffle import shuffle_mxfp4_weight, shuffle_mxfp4_scale

            w13_aiter_weight = layer.w13_weight.contiguous()
            w13_aiter_scale = layer.w13_weight_scale.contiguous()
            w2_aiter_weight = layer.w2_weight.contiguous()
            w2_aiter_scale = layer.w2_weight_scale.contiguous()
            
            e, n, k = w13_aiter_weight.shape
            w13_aiter_weight = w13_aiter_weight.view(e, n // 2, 2, k).permute(0, 2, 1, 3).contiguous().view(e, n, k)
            w13_aiter_scale = w13_aiter_scale.view(e, n // 2, 2, -1).permute(0, 2, 1, 3).contiguous().view(e, n, -1)
            
            self.w13_weight_aiter_tensor = shuffle_mxfp4_weight(w13_aiter_weight, 16, True)
            self.w13_scale_aiter_tensor = shuffle_mxfp4_scale(w13_aiter_scale, True)
            self.w2_weight_aiter_tensor = shuffle_mxfp4_weight(w2_aiter_weight, 16, False)
            self.w2_scale_aiter_tensor = shuffle_mxfp4_scale(w2_aiter_scale, False)
            self.w13_bias_aiter_tensor = layer.w13_bias.view(-1, n // 2, 2).permute(0, 2, 1).contiguous().view(-1, n)
        else:
            w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
                layer.w13_weight, layer.w13_weight_scale, num_warps)
            w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(layer.w2_weight,
                                                        layer.w2_weight_scale,
                                                        num_warps)

            self.w13_weight_triton_tensor = w13_weight
            self.w2_weight_triton_tensor = w2_weight

        # need to delete the original weights to save memory on single GPU
        del layer.w13_weight
        del layer.w2_weight
        layer.w13_weight = None
        layer.w2_weight = None
        torch.cuda.empty_cache()

        if not envs.VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4:
            if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None.")
            if (not all_close_1d(layer.w13_input_scale)
                    or not all_close_1d(layer.w2_input_scale)):
                logger.warning_once(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer.")
            # layer.w13_input_scale = torch.nn.Parameter(
            #     layer.w13_input_scale.max(), requires_grad=False)
            # layer.w2_input_scale = torch.nn.Parameter(
            #     layer.w2_input_scale.max(), requires_grad=False)

            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max().to(torch.float32),
                requires_grad=False)
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max().to(torch.float32),
                requires_grad=False)

            from triton_kernels.numerics import InFlexData
            lhs_data13 = InFlexData(scale=layer.w13_input_scale)
            lhs_data2 = InFlexData(scale=layer.w2_input_scale)

            self.w13_precision_config = PrecisionConfig(
                weight_scale=w13_scale,
                flex_ctx=FlexCtx(rhs_data=w13_flex, lhs_data=lhs_data13))
            self.w2_precision_config = PrecisionConfig(weight_scale=w2_scale,
                                                       flex_ctx=FlexCtx(
                                                           rhs_data=w2_flex,
                                                           lhs_data=lhs_data2))

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        if envs.VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4:
            return mxfp4_w4a4_moe_quant_config(
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=self.w13_scale_aiter_tensor,
                w2_scale=self.w2_scale_aiter_tensor,
            )
        else:
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config

            # TODO: how to set scale?
            return mxfp4_w4a4_moe_quant_config(
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert self.fused_experts is None

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `QuarkW4MXFp4MoEMethod_OSS` yet.")

        if envs.VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4:
            from aiter.fused_moe import fused_topk, moe_sorting
    
            token_num = x.shape[0]
            BLOCKM = 16 if token_num < 2048 else 32
            topk_weights, topk_ids = fused_topk(x, router_logits, top_k, True)
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = moe_sorting(
                topk_ids,
                topk_weights,
                self.num_experts,
                x.shape[1],
                torch.bfloat16,
                BLOCKM
            )
            _, n1, k1 = self.w13_weight_aiter_tensor.shape
            _, k2, n2 = self.w2_weight_aiter_tensor.shape
            D = n2 if k2 == k1 else n2*2
            cktile_moe_out1 = torch.empty((token_num, top_k, D), dtype=torch.bfloat16, device=x.device)
            aiter.moe_cktile2stages_gemm1(
                x,
                self.w13_weight_aiter_tensor,
                cktile_moe_out1,
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                top_k,
                self.intermediate_pad // 64 * 64 * 2,
                self.hidden_pad // 128 * 128, # k_pad_zeros
                None, # sorted_weights
                None,
                self.w13_scale_aiter_tensor,
                self.w13_bias_aiter_tensor,
                BLOCKM, # block_size
            )
            aiter.moe_cktile2stages_gemm2(
                cktile_moe_out1,
                self.w2_weight_aiter_tensor,
                moe_out,
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                top_k,
                self.hidden_pad // 64 * 64, # n_pad_zeros
                self.intermediate_pad // 128 * 128,
                sorted_weights, # sorted_weights
                None,
                self.w2_scale_aiter_tensor,
                layer.w2_bias,
                BLOCKM, # block_size
            )
            return moe_out

        else:
            from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (  # noqa: E501
                triton_kernel_moe_forward)

            return triton_kernel_moe_forward(
                hidden_states=x,
                w1=self.w13_weight_triton_tensor,
                w2=self.w2_weight_triton_tensor,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                unpadded_N_w1 = self.intermediate_size_per_partition * 2,
                unpadded_K_w1 = self.unpadded_hidden_size,
                unpadded_N_w2 = self.unpadded_hidden_size,
                unpadded_K_w2 = self.intermediate_size_per_partition
            )
