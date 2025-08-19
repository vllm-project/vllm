# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE, FusedMoEActivationFormat, FusedMoEConfig, FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize,
    FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, all_close_1d, maybe_create_device_identity,
    normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize,
    requantize_with_max_scale)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW8A8Fp8", "CompressedTensorsW8A8Fp8MoEMethod"]


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme):

    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.out_dtype = torch.get_default_dtype()
        self.is_static_input_scheme = is_static_input_scheme
        self.act_q_group_shape = GroupShape.PER_TENSOR \
            if is_static_input_scheme else GroupShape.PER_TOKEN
        self.fp8_linear = Fp8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            act_quant_group_shape=self.act_q_group_shape)

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer) -> None:
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per tensor
        if self.strategy == QuantizationStrategy.TENSOR:
            max_w_scale, weight = requantize_with_max_scale(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                logical_widths=layer.logical_widths,
            )

            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, 'input_scale', None)

                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight,
                    weight_scale=max_w_scale,
                    input_scale=input_scale)
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale,
                                                  requires_grad=False)

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # If channelwise, scales are already lined up, so just transpose.
        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight = layer.weight

            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, 'input_scale', None)

                weight, weight_scale, input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=weight,
                        weight_scale=layer.weight_scale,
                        input_scale=input_scale)
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale,
                                                  requires_grad=False)
            else:
                weight_scale = layer.weight_scale.data

            layer.weight = Parameter(weight.t(), requires_grad=False)
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(f"Unknown quantization strategy {self.strategy}")

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, 'input_scale'):
            layer.input_scale = Parameter(layer.input_scale.max(),
                                          requires_grad=False)
        else:
            layer.input_scale = None

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.float8_e4m3fn),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        # TODO: update create_xxx_parameter functions to return
        # the newly added parameters
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                 dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader)
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)

        # min requirement for fp8 kernels
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                  weight_loader=weight_loader)
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        return self.fp8_linear.apply(input=x,
                                     weight=layer.weight,
                                     weight_scale=layer.weight_scale,
                                     out_dtype=self.out_dtype,
                                     input_scale=layer.input_scale,
                                     bias=bias)


class CompressedTensorsW8A8Fp8MoEMethod(FusedMoEMethodBase):

    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        per_tensor = (self.weight_quant.strategy == QuantizationStrategy.TENSOR
                      and self.input_quant.strategy
                      == QuantizationStrategy.TENSOR)
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not (per_tensor or per_channel):
            raise ValueError(
                "For FP8 Fused MoE layers, we require per tensor "
                "or channelwise, dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization.")

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (not current_platform.has_device_capability(89)
                           or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            is_rocm_aiter_moe_enabled)

        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()

        # cutlass path
        self.is_fp8_w8a8_sm100 = quant_config._is_fp8_w8a8_sm100(
            self.weight_quant, self.input_quant)
        self.use_cutlass = (quant_config._is_fp8_w8a8_sm90(
            self.weight_quant, self.input_quant) or self.is_fp8_w8a8_sm100)
        self.disable_expert_map = False

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

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
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Allocate 2 scales for w1 and w3 respectively.
            # They are combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, 2, dtype=torch.float32),
                                                  requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-TENSOR quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=torch.float32),
                                                  requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, hidden_size, 1, dtype=torch.float32),
                                                 requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
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
            assert self.input_quant.strategy == QuantizationStrategy.TENSOR
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

        # For Per-TENSOR case, Fp8 moe kernel needs single weight scale
        # for w13 per expert. Use max then dequant and requant each expert.
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start +
                                                    shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size
            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales,
                                                        requires_grad=False)

        # Property to determine if AITER is used
        if self.rocm_aiter_moe_enabled:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa E501
                rocm_aiter_fused_experts, shuffle_weights)

            # reshaping weights is required for aiter moe kernel.
            shuffled_w13, shuffled_w2 = shuffle_weights(
                layer.w13_weight.data, layer.w2_weight.data)

            layer.w13_weight = torch.nn.Parameter(shuffled_w13,
                                                  requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(shuffled_w2,
                                                 requires_grad=False)

            self.rocm_aiter_fused_experts_func = rocm_aiter_fused_experts
        elif self.use_marlin:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (  # noqa
                prepare_moe_fp8_layer_for_marlin)
            prepare_moe_fp8_layer_for_marlin(layer, False)
            # Activations not quantized for marlin.
            del layer.w13_input_scale
            del layer.w2_input_scale
            self.fused_experts_func = None
        else:
            from vllm.model_executor.layers.fused_moe import fused_experts
            self.fused_experts_func = fused_experts

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        moe: FusedMoEConfig,
    ) -> FusedMoEPermuteExpertsUnpermute:
        # cutlass path
        if self.use_cutlass:
            from vllm.model_executor.layers.fused_moe import (
                CutlassBatchedExpertsFp8, CutlassExpertsFp8)

            experts: FusedMoEPermuteExpertsUnpermute

            num_dispatchers = prepare_finalize.num_dispatchers()

            if (prepare_finalize.activation_format ==
                    FusedMoEActivationFormat.BatchedExperts):
                logger.debug("CutlassBatchedExpertsFp8(%s)",
                             self.__class__.__name__)
                experts = CutlassBatchedExpertsFp8(
                    moe.num_local_experts,
                    num_dispatchers,
                    moe.in_dtype,
                    self.input_quant.strategy == QuantizationStrategy.TOKEN,
                    self.weight_quant.strategy == QuantizationStrategy.CHANNEL,
                )
            else:
                logger.debug("CutlassExpertsFp8(%s)", self.__class__.__name__)
                experts = CutlassExpertsFp8(
                    moe.in_dtype,
                    self.input_quant.strategy == QuantizationStrategy.TOKEN,
                    self.weight_quant.strategy == QuantizationStrategy.CHANNEL,
                )

            self.disable_expert_map = (num_dispatchers > 1
                                       or not experts.supports_expert_map())

            return experts

        # triton path
        from vllm.model_executor.layers.fused_moe import TritonExperts
        from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
            BatchedTritonExperts)

        assert not self.rocm_aiter_moe_enabled and not self.use_marlin

        logger.debug("BatchedTritonExperts(%s)", self.__class__.__name__)

        if (prepare_finalize.activation_format ==
                FusedMoEActivationFormat.BatchedExperts):
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank(
            )
            assert max_num_tokens_per_rank is not None

            return BatchedTritonExperts(
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                use_fp8_w8a8=True,
                block_shape=self.quant_config.weight_block_size,
                per_act_token_quant=(
                    self.input_quant.strategy == QuantizationStrategy.TOKEN),
            )
        else:
            return TritonExperts(
                use_fp8_w8a8=True,
                block_shape=self.quant_config.weight_block_size,
                per_act_token_quant=(
                    self.input_quant.strategy == QuantizationStrategy.TOKEN),
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
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for "
                "`CompressedTensorsW8A8Fp8MoEMethod` yet.")

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
            indices_type=self.topk_indices_dtype,
        )

        # cutlass path
        if self.use_cutlass:
            per_act_token = (
                self.input_quant.strategy == QuantizationStrategy.TOKEN)
            per_channel_quant = (
                self.weight_quant.strategy == QuantizationStrategy.CHANNEL)

            # small-batch fallback on SM100
            if self.is_fp8_w8a8_sm100 and topk_ids.shape[0] <= 8:
                from vllm.model_executor.layers.fused_moe import fused_experts
                return fused_experts(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    inplace=True,
                    activation=activation,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    use_fp8_w8a8=True,
                    per_channel_quant=per_channel_quant,
                    global_num_experts=global_num_experts,
                    expert_map=None if self.disable_expert_map else expert_map,
                    w1_scale=layer.w13_weight_scale,
                    w2_scale=layer.w2_weight_scale,
                    a1_scale=layer.w13_input_scale,
                    a2_scale=layer.w2_input_scale)

            if self.fused_experts is None:
                from vllm.model_executor.layers.fused_moe.cutlass_moe import (
                    cutlass_moe_fp8)
                return cutlass_moe_fp8(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    per_act_token=per_act_token,
                    activation=activation,
                    global_num_experts=global_num_experts,
                    expert_map=None if self.disable_expert_map else expert_map,
                    w1_scale=layer.w13_weight_scale,
                    w2_scale=layer.w2_weight_scale,
                    a1_scale=layer.w13_input_scale,
                    a2_scale=layer.w2_input_scale,
                )
            else:
                return self.fused_experts(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    activation=activation,
                    global_num_experts=global_num_experts,
                    expert_map=None if self.disable_expert_map else expert_map,
                    w1_scale=layer.w13_weight_scale,
                    w2_scale=layer.w2_weight_scale,
                    a1_scale=layer.w13_input_scale,
                    a2_scale=layer.w2_input_scale,
                )

        if self.rocm_aiter_moe_enabled:
            return self.rocm_aiter_fused_experts_func(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                use_fp8_w8a8=True,
                per_channel_quant=self.weight_quant.strategy ==
                QuantizationStrategy.CHANNEL,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                expert_map=expert_map)
        if self.use_marlin:
            assert activation == "silu", (
                f"{activation} not supported for Marlin MoE.")
            return torch.ops.vllm.fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                None,
                None,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                router_logits,
                topk_weights,
                topk_ids,
                quant_type_id=scalar_types.float8_e4m3fn.id,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map)

        assert self.fused_experts_func is not None

        return self.fused_experts_func(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=True,
            per_channel_quant=self.weight_quant.strategy ==
            QuantizationStrategy.CHANNEL,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale)
