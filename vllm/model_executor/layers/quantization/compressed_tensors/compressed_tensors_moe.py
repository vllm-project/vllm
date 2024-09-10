import enum
from enum import Enum
from typing import Callable, List, Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    CompressionFormat)
from vllm.model_executor.utils import set_weight_attrs


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


__all__ = ["CompressedTensorsMoEMethod"]


class CompressedTensorsMoEMethod(FusedMoEMethodBase):

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
        self.strategy = config.strategy.value
        self.group_size = config.group_size
        assert config.symmetric, (
            "Only symmetric quantization is supported for MoE")

        if not (self.quant_config.quant_format
                == CompressionFormat.pack_quantized.value
                and self.num_bits == 4):
            raise ValueError("For Fused MoE layers, only ",
                             f"{CompressionFormat.pack_quantized.value} ",
                             "is supported for 4 bits")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update({
            "is_transposed": True,
            "quant_method": self.strategy
        })
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    hidden_size //
                                                    self.packed_factor,
                                                    2 * intermediate_size,
                                                    dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   intermediate_size //
                                                   self.packed_factor,
                                                   hidden_size,
                                                   dtype=torch.int32),
                                       requires_grad=False)
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = intermediate_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                  num_groups_w13,
                                                  2 * intermediate_size,
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
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
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
                intermediate_size,
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

        num_experts = layer.w13_g_idx.shape[0]
        device = layer.w13_g_idx.device
        layer.w13_g_idx = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w2_g_idx = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
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
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:

        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            fused_marlin_moe)

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function)

        return fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            router_logits,
            layer.w13_g_idx,
            layer.w2_g_idx,
            layer.w13_g_idx_sort_indices,
            layer.w2_g_idx_sort_indices,
            topk_weights,
            topk_ids,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )
