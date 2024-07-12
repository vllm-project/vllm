from typing import Callable, List, Optional

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_marlin_linear, marlin_is_k_full, marlin_make_empty_g_idx,
    marlin_make_workspace, marlin_permute_scales,
    marlin_repeat_scales_on_all_ranks, marlin_sort_g_idx, replace_tensor,
    verify_marlin_supported, verify_marlin_supports_shape)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_BITS = [4, 8]


class CompressedTensorsWNA16(CompressedTensorsScheme):

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 act_order: bool,
                 group_size: Optional[int] = None):
        self.num_bits = num_bits
        self.pack_factor = 32 // self.num_bits
        self.pack_factor = 32 // self.num_bits
        self.strategy = strategy

        self.group_size: int
        if group_size is None:
            if self.strategy != "channel":
                raise ValueError(
                    "Marlin kernels require group quantization or "
                    "channelwise quantization, but found no group "
                    "size and strategy is not channelwise.")
            self.group_size = -1
        else:
            self.group_size = group_size

        self.group_size: int
        if group_size is None:
            if self.strategy != "channel":
                raise ValueError(
                    "Marlin kernels require group quantization or "
                    "channelwise quantization, but found no group "
                    "size and strategy is not channelwise.")
            self.group_size = -1
        else:
            self.group_size = group_size

        if act_order and self.group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            act_order = False
        self.act_order = act_order

        # Verify supported on platform.
        verify_marlin_supported(num_bits=self.num_bits,
                                group_size=self.group_size,
                                is_sym=True)


    def create_weights(self, layer: torch.nn.Module, input_size: int,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition

        # If group_size is -1, we are in channelwise case.
        group_size = input_size if self.group_size == -1 else self.group_size

        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size)

        # WEIGHT
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.pack_factor,
                "pack_factor": self.pack_factor,
                "weight_loader": weight_loader
            })
        layer.register_parameter("weight_packed", weight)

        # Shape of the weights before packing (not used in inference)
        weight_shape = Parameter(torch.empty(2, dtype=torch.int64),
                                 requires_grad=False)

        layer.register_parameter("weight_shape", weight_shape)
        set_weight_attrs(weight_shape, {
            "weight_loader": weight_loader,
            "ignore_warning": True,
        })

        # WEIGHT SCALE
        if marlin_repeat_scales_on_all_ranks(self.act_order, self.group_size,
                                             is_row_parallel):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scale_input_dim = None
            scale_input_size = input_size // group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scale_input_dim = 1
            scale_input_size = input_size_per_partition // group_size

        weight_scale = Parameter(torch.empty(output_size_per_partition,
                                             scale_input_size,
                                             dtype=params_dtype),
                                 requires_grad=False)
        set_weight_attrs(
            weight_scale, {
                "weight_loader": weight_loader,
                "input_dim": scale_input_dim,
                "output_dim": 0
            })
        layer.register_parameter("weight_scale", weight_scale)

        # G_IDX (for activation reordering)
        g_idx = Parameter(torch.empty(input_size_per_partition,
                                      dtype=torch.int32),
                          requires_grad=False)
        layer.register_parameter("weight_g_idx", g_idx)
        set_weight_attrs(g_idx, {
            "weight_loader": weight_loader,
            "input_dim": 0,
            "ignore_warning": True
        })

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.is_k_full = marlin_is_k_full(self.act_order, is_row_parallel)


    # Checkpoints are serialized in compressed-tensors format, which is
    # different from marlin format. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight_packed.device

        # Allocate marlin workspace.
        layer.workspace = marlin_make_workspace(
            layer.output_size_per_partition, device)

        # Handle sorting for activation reordering if needed.
        self.act_order = False
        if self.act_order:
            breakpoint()
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(layer.weight_g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
            replace_tensor(layer, "weight_g_idx", g_idx)
        else:
            layer.weight_g_idx = marlin_make_empty_g_idx(device)
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # Repack weights from compressed-tensors format to marlin format.
        marlin_qweight = ops.gptq_marlin_repack(
            layer.weight_packed.t().contiguous(),
            perm=layer.g_idx_sort_indices,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.num_bits)
        replace_tensor(layer, "weight_packed", marlin_qweight)

        # Permute scales from compressed-tensors format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.weight_scale.squeeze().t().contiguous(),
            size_k=(layer.input_size if self.act_order else
                    layer.input_size_per_partition),
            size_n=layer.output_size_per_partition,
            group_size=self.group_size)
        replace_tensor(layer, "weight_scale", marlin_scales)


    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        return apply_marlin_linear(
            input=x,
            weight=layer.weight_packed,
            weight_scale=layer.weight_scale,
            g_idx=layer.weight_g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            num_bits=self.num_bits,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            is_k_full=layer.is_k_full)
