from typing import Callable, List, Optional

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear, marlin_make_empty_g_idx, marlin_make_workspace,
    marlin_permute_scales, replace_tensor, verify_marlin_supported,
    verify_marlin_supports_shape)
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import scalar_types

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128,
}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsWNA16(CompressedTensorsScheme):

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None):

        self.pack_factor = 32 // num_bits
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
        channelwise = (self.group_size == -1)
        group_size = input_size if channelwise else self.group_size
        row_parallel = (input_size != input_size_per_partition)
        # In the case of channelwise quantization, we need to replicate the
        # scales across all gpus.
        partition_scales = (row_parallel and not channelwise)

        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size)

        weight_scale_dim = None
        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            weight_scale_dim = 1
            scales_and_zp_size = input_size_per_partition // group_size

        weight = Parameter(
            torch.empty(
                output_size_per_partition,
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
                "weight_loader": weight_loader
            })
        layer.register_parameter("weight_packed", weight)

        weight_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        set_weight_attrs(
            weight_scale, {
                "weight_loader": weight_loader,
                "input_dim": weight_scale_dim,
                "output_dim": 0
            })
        layer.register_parameter("weight_scale", weight_scale)

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = Parameter(torch.empty(2, dtype=torch.int64),
                                 requires_grad=False)

        layer.register_parameter("weight_shape", weight_shape)
        set_weight_attrs(weight_shape, {
            "weight_loader": weight_loader,
            "ignore_warning": True,
        })

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

        # Act-order not supported in compressed-tensors yet, so set to empty.
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        layer.weight_zp = marlin_make_empty_g_idx(device)

        # Repack weights from compressed-tensors format to marlin format.
        marlin_qweight = ops.gptq_marlin_repack(
            layer.weight_packed.t().contiguous(),
            perm=layer.g_idx_sort_indices,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_type.size_bits)
        replace_tensor(layer, "weight_packed", marlin_qweight)

        # Permute scales from compressed-tensors format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.weight_scale.squeeze().t().contiguous(),
            size_k=layer.input_size_per_partition,
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
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            wtype=self.quant_type,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            is_k_full=True,
            bias=bias)
