import enum
from enum import Enum
from typing import Callable, List, Optional

import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy

import vllm.model_executor.layers.fused_moe  # noqa
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    WNA16_SUPPORTED_BITS)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, normalize_e4m3fn_to_e4m3fnuz, per_tensor_dequantize)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


__all__ = [
    "CompressedTensorsMoEMethod", "CompressedTensorsW8A8Fp8MoEMethod",
    "CompressedTensorsWNA16MoEMethod"
]


class CompressedTensorsMoEMethod(FusedMoEMethodBase):

    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ) -> "CompressedTensorsMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            return CompressedTensorsWNA16MoEMethod(quant_config)
        elif quant_config._is_fp8_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Fp8MoEMethod(quant_config)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")


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

        if not (self.weight_quant.strategy == QuantizationStrategy.TENSOR
                and self.input_quant.strategy == QuantizationStrategy.TENSOR):
            raise ValueError(
                "For FP8 Fused MoE layers, only per-tensor scales"
                "for weights and activations are supported. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic

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
                    "for each layer.")
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False)
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False)

        # If rocm, normalize the weights and scales to e4m3fnuz
        if current_platform.is_rocm():
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
        for expert_id in range(layer.num_experts):
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
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        return fused_experts(x,
                             layer.w13_weight,
                             layer.w2_weight,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids,
                             inplace=True,
                             use_fp8_w8a8=True,
                             w1_scale=layer.w13_weight_scale,
                             w2_scale=layer.w2_weight_scale,
                             a1_scale=layer.w13_input_scale,
                             a2_scale=layer.w2_input_scale)


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

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        assert params_dtype == torch.float16, (
            "float16 is required for MoE compressed models. Set dtype=torch.float16"  # noqa: E501
        )

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

        def replace_tensor(name, new_t):
            # It is important to use resize_() here since it ensures
            # the same buffer is reused
            getattr(layer, name).resize_(new_t.shape)
            getattr(layer, name).copy_(new_t)
            del new_t

        def get_scale_perms(num_bits: int):
            scale_perm: List[int] = []
            for i in range(8):
                scale_perm.extend([i + 8 * j for j in range(8)])
            scale_perm_single: List[int] = []
            for i in range(4):
                scale_perm_single.extend(
                    [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
            return scale_perm, scale_perm_single

        def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                                  group_size: int, num_bits: int):
            scale_perm, scale_perm_single = get_scale_perms(num_bits)
            if group_size < size_k and group_size != -1:
                s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
            else:
                s = s.reshape((-1, len(scale_perm_single)))[:,
                                                            scale_perm_single]
            s = s.reshape((-1, size_n)).contiguous()
            return s

        def marlin_moe_permute_scales(s: torch.Tensor, size_k: int,
                                      size_n: int, group_size: int,
                                      num_bits: int):
            num_experts = s.shape[0]
            output = torch.empty((num_experts, s.shape[1], s.shape[2]),
                                 device=s.device,
                                 dtype=s.dtype)
            for e in range(num_experts):
                output[e] = marlin_permute_scales(s[e], size_k, size_n,
                                                  group_size, num_bits)
            return output

        size_k2 = layer.w2_weight_packed.shape[2]
        size_k13 = layer.w13_weight_packed.shape[2]

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
        replace_tensor("w13_weight_packed", marlin_w13_qweight)
        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w2_weight_packed", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            layer.w13_weight_scale,
            size_k13,
            layer.w13_weight_scale.shape[2],
            self.group_size,
            self.num_bits,
        )
        replace_tensor("w13_weight_scale", marlin_w13_scales)
        marlin_w2_scales = marlin_moe_permute_scales(
            layer.w2_weight_scale,
            layer.w2_weight_scale.shape[1] * self.packed_factor,
            size_k2,
            self.group_size,
            self.num_bits,
        )
        replace_tensor("w2_weight_scale", marlin_w2_scales)

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
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

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
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full)
