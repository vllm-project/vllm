from typing import Callable, List, Optional

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    ActivationOrdering)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear, marlin_make_empty_g_idx, marlin_make_workspace,
    marlin_permute_scales, marlin_repeat_scales_on_all_ranks,
    marlin_sort_g_idx, replace_tensor, verify_marlin_supported,
    verify_marlin_supports_shape)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)
from vllm.scalar_type import scalar_types

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128
}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsWNA16(CompressedTensorsScheme):

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None,
                 actorder: Optional[ActivationOrdering] = None):

        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError("Marlin kernels require group quantization or "
                             "channelwise quantization, but found no group "
                             "size and strategy is not channelwise.")

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}")

        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[num_bits]

        # Verify supported on platform.
        verify_marlin_supported(quant_type=self.quant_type,
                                group_size=self.group_size)

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        output_size_per_partition = sum(output_partition_sizes)

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = (input_size != input_size_per_partition)
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            self.has_g_idx, self.group_size, row_parallel)

        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size)

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader,
                                     packed_factor=self.pack_factor,
                                     packed_dim=1,
                                     data=torch.empty(
                                         output_size_per_partition,
                                         input_size_per_partition //
                                         self.pack_factor,
                                         dtype=torch.int32,
                                     ))

        weight_scale_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            )
        }
        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)
        else:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(data=torch.empty(2,
                                                          dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        # group index (for activation reordering)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
                                            input_dim=0,
                                            weight_loader=weight_loader)
            layer.register_parameter("weight_g_idx", weight_g_idx)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.group_size = group_size

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from marlin format. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight_packed.device

        # Allocate marlin workspace.
        layer.workspace = marlin_make_workspace(
            layer.output_size_per_partition, device)

        # Handle sorting for activation reordering if needed.
        if self.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(layer.weight_g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
            replace_tensor(layer, "weight_g_idx", g_idx)
        else:
            layer.weight_g_idx = marlin_make_empty_g_idx(device)
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        layer.weight_zp = marlin_make_empty_g_idx(device)
        # Update for kernel
        layer.weight_packed = torch.nn.Parameter(
            layer.weight_packed.t().contiguous(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.squeeze().t().contiguous(), requires_grad=False)

        # Repack weights from compressed-tensors format to marlin format.
        marlin_qweight = ops.gptq_marlin_repack(
            layer.weight_packed,
            perm=layer.g_idx_sort_indices,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_type.size_bits)
        replace_tensor(layer, "weight_packed", marlin_qweight)

        # Permute scales from compressed-tensors format to marlin format.
        # scale is required on all partitions if activation reordering
        marlin_scales = marlin_permute_scales(
            layer.weight_scale,
            size_k=(layer.input_size
                    if self.has_g_idx else layer.input_size_per_partition),
            size_n=layer.output_size_per_partition,
            group_size=layer.group_size)
        replace_tensor(layer, "weight_scale", marlin_scales)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:

        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.weight_packed,
            weight_scale=layer.weight_scale,
            weight_zp=layer.weight_zp,
            g_idx=layer.weight_g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            wtype=self.quant_type,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            is_k_full=True,
            bias=bias)
