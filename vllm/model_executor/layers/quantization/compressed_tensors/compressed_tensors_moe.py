# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from enum import Enum
from typing import Callable, Optional

import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (ActivationOrdering,
                                             QuantizationStrategy)

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE, FusedMoEActivationFormat, FusedMoEConfig, FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize,
    FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (  # noqa
    FlashInferCutlassMoEPrepareAndFinalize)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS, WNA16_SUPPORTED_TYPES_MAP)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    build_flashinfer_fp4_cutlass_moe_kernel,
    flashinfer_fp4_cutlass_moe_forward, reorder_w1w3_to_w3w1)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer, marlin_make_workspace_new,
    marlin_moe_permute_scales)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_moe_fp8_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    swizzle_blockscale)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


__all__ = [
    "CompressedTensorsMoEMethod", "CompressedTensorsW8A8Fp8MoEMethod",
    "CompressedTensorsW8A8Int8MoEMethod",
    "CompressedTensorsWNA16MarlinMoEMethod", "CompressedTensorsWNA16MoEMethod",
    "CompressedTensorsW4A4MoeMethod"
]


class CompressedTensorsMoEMethod(FusedMoEMethodBase):

    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
    ) -> "CompressedTensorsMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            # group_size=None means channelwise
            group_size = weight_quant.group_size or -1
            # Prefer to use the MarlinMoE kernel when it is supported.
            if not check_moe_marlin_supports_layer(layer, group_size):
                if (weight_quant.strategy in QuantizationStrategy.GROUP and
                        weight_quant.actorder in (ActivationOrdering.GROUP,
                                                  ActivationOrdering.DYNAMIC)):
                    raise ValueError(
                        "WNA16MoE is not supported with actorder=group/dynamic."
                    )
                logger.info_once("Using CompressedTensorsWNA16MoEMethod")
                return CompressedTensorsWNA16MoEMethod(quant_config)
            else:
                logger.info_once("Using CompressedTensorsWNA16MarlinMoEMethod")
                return CompressedTensorsWNA16MarlinMoEMethod(quant_config)
        elif quant_config._is_fp4a4_nvfp4(weight_quant, input_quant):
            return CompressedTensorsW4A4MoeMethod()
        elif (quant_config._is_fp8_w8a8_sm90(weight_quant, input_quant)
              or quant_config._is_fp8_w8a8_sm100(weight_quant, input_quant)
              or quant_config._is_fp8_w8a8(weight_quant, input_quant)):
            return CompressedTensorsW8A8Fp8MoEMethod(quant_config)
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8MoEMethod(quant_config)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")


class CompressedTensorsW4A4MoeMethod(CompressedTensorsMoEMethod):

    def __init__(self):
        from vllm.model_executor.layers.quantization.utils.nvfp4_moe_support import (  # noqa: E501
            detect_nvfp4_moe_support)
        _nvfp4 = detect_nvfp4_moe_support(self.__class__.__name__)
        self.cutlass_nvfp4_supported = _nvfp4.cutlass_supported
        self.allow_flashinfer = _nvfp4.allow_flashinfer
        self.use_marlin = _nvfp4.use_marlin
        self.group_size = 16
        self.fused_experts = None  # type: ignore[assignment]

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8),
            requires_grad=False)
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8),
            requires_grad=False)
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn),
            requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn),
            requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value})
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales
        w13_weight_scale_2 = torch.nn.Parameter(torch.empty(
            num_experts, 2, dtype=torch.float32),
                                                requires_grad=False)
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(torch.empty(
            num_experts, dtype=torch.float32),
                                               requires_grad=False)
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales
        w13_input_scale = torch.nn.Parameter(torch.empty(num_experts,
                                                         2,
                                                         dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(torch.empty(num_experts,
                                                        dtype=torch.float32),
                                            requires_grad=False)
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # From packed to weight
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight_packed.data,
                                              requires_grad=False)

        layer.w2_weight = torch.nn.Parameter(layer.w2_weight_packed.data,
                                             requires_grad=False)

        # reorder GEMM1 weights and block scales for FlashInfer CUTLASS kernel.
        if self.allow_flashinfer:
            w, s = reorder_w1w3_to_w3w1(layer.w13_weight.data,
                                        layer.w13_weight_scale.data,
                                        dim=-2)
            layer.w13_weight = torch.nn.Parameter(w, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(s, requires_grad=False)

        if not torch.allclose(layer.w13_weight_global_scale[:, 0],
                              layer.w13_weight_global_scale[:, 1]):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected.")

        # Take inverse of global scale saved to disk
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w13_weight_global_scale[:, 0], requires_grad=False)

        layer.w2_weight_scale_2 = torch.nn.Parameter(
            1 / layer.w2_weight_global_scale.data, requires_grad=False)

        if self.use_marlin:
            prepare_moe_fp4_layer_for_marlin(layer)
            return

        # swizzle weight scales
        layer.w13_blockscale_swizzled = torch.nn.Parameter(swizzle_blockscale(
            layer.w13_weight_scale),
                                                           requires_grad=False)

        layer.w2_blockscale_swizzled = torch.nn.Parameter(swizzle_blockscale(
            layer.w2_weight_scale),
                                                          requires_grad=False)

        # w13
        w13_input_global_scale = layer.w13_input_global_scale.max(
            dim=1).values.to(torch.float32)

        layer.g1_alphas = torch.nn.Parameter(
            ((1 / w13_input_global_scale) * layer.w13_weight_scale_2),
            requires_grad=False)

        layer.w13_input_scale_quant = torch.nn.Parameter(
            (w13_input_global_scale), requires_grad=False)

        # w2
        layer.g2_alphas = torch.nn.Parameter(
            ((1 / layer.w2_input_global_scale) * layer.w2_weight_scale_2).to(
                torch.float32),
            requires_grad=False)

        layer.w2_input_scale_quant = torch.nn.Parameter(
            (layer.w2_input_global_scale), requires_grad=False)

    def maybe_swap_experts_impl(self, moe_parallel_config):
        if not self.allow_flashinfer:
            return
        self.fused_experts = build_flashinfer_fp4_cutlass_moe_kernel(
            moe_parallel_config)

    def select_gemm_impl(self, prepare_finalize, moe):
        """Return the appropriate GEMM experts implementation."""
        assert moe is not None and prepare_finalize is not None
        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (  # noqa: E501
            select_nvfp4_gemm_impl)

        return select_nvfp4_gemm_impl(self.allow_flashinfer, moe, logger)

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
            raise NotImplementedError("EPLB not supported for "
                                      "`CompressedTensorsW4A4MoeMethod` yet.")
        assert activation == "silu", "Only SiLU activation is supported."

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
        )

        if self.use_marlin:
            return torch.ops.vllm.fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                router_logits,
                topk_weights,
                topk_ids,
                global_scale1=layer.w13_weight_scale_2,
                global_scale2=layer.w2_weight_scale_2,
                quant_type_id=scalar_types.float4_e2m1f.id,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map)

        # FlashInfer fused experts path
        if self.fused_experts is not None:
            return flashinfer_fp4_cutlass_moe_forward(
                self.fused_experts,
                layer,
                x,
                topk_weights,
                topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        assert expert_map is None, ("Expert Parallelism / expert_map "
                                    "is currently not supported for "
                                    "CompressedTensorsW4A4MoeMethod.")
        from vllm.model_executor.layers.fused_moe.cutlass_moe import (
            cutlass_moe_fp4)

        # Cutlass moe takes in activations in BF16/Half precision
        # and fp4 quantized weights loaded from the checkpoint
        return cutlass_moe_fp4(
            a=x,
            w1_fp4=layer.w13_weight,
            w2_fp4=layer.w2_weight,
            w1_blockscale=layer.w13_blockscale_swizzled,
            w2_blockscale=layer.w2_blockscale_swizzled,
            g1_alphas=layer.g1_alphas,
            g2_alphas=layer.g2_alphas,
            a1_gscale=layer.w13_input_scale_quant,
            a2_gscale=layer.w2_input_scale_quant,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            m=x.shape[0],
            n=layer.w2_weight.shape[2] * 2,
            k=x.shape[1],
            e=layer.w13_weight.shape[0],
            device=x.device,
            apply_router_weight_on_input=apply_router_weight_on_input).to(
                x.dtype)


class CompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsMoEMethod):

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")
        self.topk_indices_dtype = None

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
        self.fused_experts = None  # type: ignore[assignment]
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
            from vllm.model_executor.layers.fused_moe import CutlassExpertsFp8

            use_batched_format = (prepare_finalize.activation_format ==
                                  FusedMoEActivationFormat.BatchedExperts)

            num_dispatchers = prepare_finalize.num_dispatchers()
            num_experts = (moe.num_local_experts
                           if use_batched_format else moe.num_experts)

            logger.debug("CutlassExpertsFp8(%s)", self.__class__.__name__)

            experts = CutlassExpertsFp8(
                num_experts,
                moe.in_dtype,
                self.input_quant.strategy == QuantizationStrategy.TOKEN,
                self.weight_quant.strategy == QuantizationStrategy.CHANNEL,
                num_dispatchers=num_dispatchers,
                use_batched_format=use_batched_format,
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


class CompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales.")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        params_dtype = torch.int8

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
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=torch.float32),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        hidden_size,
                                                        1,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

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
                "`CompressedTensorsW8A8Int8MoEMethod` yet.")

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
            e_score_correction_bias=e_score_correction_bias)

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale)


class CompressedTensorsWNA16MarlinMoEMethod(CompressedTensorsMoEMethod):

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        self.group_size = config.group_size
        self.actorder = config.actorder
        assert config.symmetric, (
            "Only symmetric quantization is supported for MoE")

        if not (self.quant_config.quant_format
                == CompressionFormat.pack_quantized.value
                and self.num_bits in WNA16_SUPPORTED_BITS):
            raise ValueError("For Fused MoE layers, only ",
                             f"{CompressionFormat.pack_quantized.value} ",
                             "is supported for the following bits: ",
                             f"{WNA16_SUPPORTED_BITS}")
        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[self.num_bits]

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        intermediate_size_full = extra_weight_attrs.pop(
            "intermediate_size_full")

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update({
            "is_transposed": True,
            "quant_method": self.strategy
        })
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size // self.packed_factor,
            2 * intermediate_size_per_partition,
            dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            intermediate_size_per_partition // self.packed_factor,
            hidden_size,
            dtype=torch.int32),
                                       requires_grad=False)
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1
        w2_scales_size = (intermediate_size_full
                          if load_full_w2 else intermediate_size_per_partition)

        self.is_k_full = (not self.actorder) or (
            intermediate_size_per_partition == intermediate_size_full)

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                 num_groups_w2,
                                                 hidden_size,
                                                 dtype=params_dtype),
                                      requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        w2_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                              requires_grad=False)

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices",
                                 w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices",
                                 w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None
        layer.marlin_state = GPTQMarlinState.REPACK

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(
                    layer.w13_weight_g_idx[e]).to(torch.int32)
                w2_g_idx_sort_indices[e] = torch.argsort(
                    layer.w2_weight_g_idx[e]).to(torch.int32)
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][
                    w13_g_idx_sort_indices[e]]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][
                    w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices",
                              w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices",
                              w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32,
                            device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = ops.gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
        )
        replace_parameter(layer, "w13_weight_packed", marlin_w13_qweight)
        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
        )
        replace_parameter(layer, "w2_weight_packed", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_weight_scale,
            size_k=layer.w13_weight_packed.shape[2],
            size_n=layer.w13_weight_scale.shape[2],
            group_size=self.group_size,
        )
        replace_parameter(layer, "w13_weight_scale", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_weight_scale,
            size_k=layer.w2_weight_scale.shape[1] *
            (self.group_size if self.group_size != -1 else self.packed_factor),
            size_n=layer.w2_weight_scale.shape[2],
            group_size=self.group_size,
        )
        replace_parameter(layer, "w2_weight_scale", marlin_w2_scales)

        layer.workspace = marlin_make_workspace_new(device, 4)

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
                "`CompressedTensorsWNA16MarlinMoEMethod` yet.")

        assert activation == "silu", (
            f"{activation} not supported for Marlin MoE.")

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
            e_score_correction_bias=e_score_correction_bias)

        return torch.ops.vllm.fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            quant_type_id=self.quant_type.id,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            workspace=layer.workspace,
            is_k_full=self.is_k_full)


class CompressedTensorsWNA16MoEMethod(CompressedTensorsMoEMethod):

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        # channelwise is not supported by this kernel
        assert config.strategy == "group"
        self.group_size = config.group_size
        # grouped actorder isn't supported by this kernel
        assert config.actorder != "group"
        assert config.symmetric, (
            "Only symmetric quantization is supported for MoE")

        if not (self.quant_config.quant_format
                == CompressionFormat.pack_quantized.value
                and self.num_bits in WNA16_SUPPORTED_BITS):
            raise ValueError("For Fused MoE layers, only ",
                             f"{CompressionFormat.pack_quantized.value} ",
                             "is supported for the following bits: ",
                             f"{WNA16_SUPPORTED_BITS}")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update({
            "is_transposed": True,
            "quant_method": self.strategy
        })
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size // self.packed_factor,
            2 * intermediate_size_per_partition,
            dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            intermediate_size_per_partition // self.packed_factor,
            hidden_size,
            dtype=torch.int32),
                                       requires_grad=False)
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_scales_size = intermediate_size_per_partition

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            num_groups_w13,
            2 * intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                 num_groups_w2,
                                                 hidden_size,
                                                 dtype=params_dtype),
                                      requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": False})

        w2_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2),
                                              requires_grad=False)

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices",
                                 w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices",
                                 w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reconfigure packed weights and scales to match moe_wna16 format
        layer.w13_weight_packed = torch.nn.Parameter(
            layer.w13_weight_packed.transpose(1, 2).contiguous().view(
                torch.uint8),
            requires_grad=False)
        layer.w2_weight_packed = torch.nn.Parameter(
            layer.w2_weight_packed.transpose(1,
                                             2).contiguous().view(torch.uint8),
            requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.transpose(1, 2).contiguous(),
            requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.transpose(1, 2).contiguous(),
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
        if enable_eplb:
            raise NotImplementedError("EPLB not supported for "
                                      "`CompressedTensorsWNA16MoEMethod` yet.")

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
            e_score_correction_bias=e_score_correction_bias)

        return fused_experts(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            use_int4_w4a16=self.num_bits == 4,
            use_int8_w8a16=self.num_bits == 8,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size])
