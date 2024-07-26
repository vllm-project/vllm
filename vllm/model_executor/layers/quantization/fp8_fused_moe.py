from typing import TYPE_CHECKING, Optional

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, fused_moe
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d, per_tensor_dequantize)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils import print_warning_once

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: 'Fp8Config'):
        self.quant_config = quant_config

    def create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                       intermediate_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size,
                                                    hidden_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   hidden_size,
                                                   intermediate_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                  2,
                                                  dtype=torch.float32),
                                       requires_grad=False)
        layer.register_parameter("w13_scale", w13_scale)

        w2_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                 dtype=torch.float32),
                                      requires_grad=False)
        layer.register_parameter("w2_scale", w2_scale)

        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_scale, extra_weight_attrs)
            set_weight_attrs(w2_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8.")

            a13_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                      dtype=torch.float32),
                                           requires_grad=False)
            layer.register_parameter("a13_scale", a13_scale)
            set_weight_attrs(a13_scale, extra_weight_attrs)

            a2_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                     dtype=torch.float32),
                                          requires_grad=False)
            layer.register_parameter("a2_scale", a2_scale)
            set_weight_attrs(a2_scale, extra_weight_attrs)
        else:
            layer.a13_scale = None
            layer.a2_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:

        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            w13_weight = torch.empty_like(layer.w13_weight.data,
                                          dtype=torch.float8_e4m3fn)
            w2_weight = torch.empty_like(layer.w2_weight.data,
                                         dtype=torch.float8_e4m3fn)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_scale = torch.nn.Parameter(torch.ones(
                layer.num_experts,
                dtype=torch.float32,
                device=w13_weight.device),
                                                 requires_grad=False)
            for expert in range(layer.num_experts):
                w13_weight[expert, :, :], layer.w13_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w13_weight.data[expert, :, :])
                w2_weight[expert, :, :], layer.w2_scale[
                    expert] = ops.scaled_fp8_quant(
                        layer.w2_weight.data[expert, :, :])
            layer.w13_weight = torch.nn.Parameter(w13_weight,
                                                  requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight,
                                                 requires_grad=False)
            return

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if layer.a13_scale is None or layer.a2_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None.")
                if (not all_close_1d(layer.a13_scale)
                        or not all_close_1d(layer.a2_scale)):
                    print_warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer. ")
                layer.a13_scale = torch.nn.Parameter(layer.a13_scale.max(),
                                                     requires_grad=False)
                layer.a2_scale = torch.nn.Parameter(layer.a2_scale.max(),
                                                    requires_grad=False)

            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_scale.max(dim=1).values
            for expert_id in range(layer.num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start:start +
                                                    shard_size, :],
                        layer.w13_scale[expert_id][shard_id])
                    layer.w13_weight[expert_id][
                        start:start + shard_size, :], _ = ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id])
                    start += shard_size

            layer.w13_scale = torch.nn.Parameter(max_w13_scales,
                                                 requires_grad=False)
            return

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool = True,
              use_grouped_topk: bool = False,
              num_expert_group: Optional[int] = None,
              topk_group: Optional[int] = None) -> torch.Tensor:

        return fused_moe(x,
                         layer.w13_weight,
                         layer.w2_weight,
                         router_logits,
                         top_k,
                         renormalize=renormalize,
                         inplace=True,
                         use_fp8=True,
                         w1_scale=layer.w13_scale,
                         w2_scale=layer.w2_scale,
                         a1_scale=layer.a13_scale,
                         a2_scale=layer.a2_scale,
                         use_grouped_topk=use_grouped_topk,
                         num_expert_group=num_expert_group,
                         topk_group=topk_group)
