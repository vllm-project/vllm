from typing import Callable, List, Optional

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N, GPTQMarlinState,
    marlin_permute_scales)
from vllm.model_executor.utils import set_weight_attrs

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_BITS = [4, 8]


class CompressedTensorsWNA16(CompressedTensorsScheme):

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None):
        self.num_bits = num_bits
        self.strategy = strategy
        self.group_size = group_size

        if self.strategy == "group" and self.group_size is None:
            raise ValueError(
                "group_size must be given when using strategy group")

    def create_weights(self, layer: torch.nn.Module, input_size: int,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        pack_factor = 32 // self.num_bits
        output_size_per_partition = sum(output_partition_sizes)

        if self.group_size is not None:
            group_size = self.group_size
        else:
            group_size = input_size

        weight_scale_dim = None
        scales_and_zp_size = input_size // group_size

        if (input_size != input_size_per_partition
                and self.group_size is not None):
            weight_scale_dim = 1
            scales_and_zp_size = input_size_per_partition // group_size

        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": pack_factor,
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
        layer.marlin_state = GPTQMarlinState.REPACK
        layer.is_k_full = True
        layer.group_size = group_size

        max_workspace_size = (
            output_size_per_partition //
            GPTQ_MARLIN_MIN_THREAD_N) * GPTQ_MARLIN_MAX_PARALLEL

        workspace = torch.zeros(max_workspace_size,
                                dtype=torch.int,
                                requires_grad=False)
        layer.workspace = workspace

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor):
        reshaped_x = x.reshape(-1, x.shape[-1])

        size_m = reshaped_x.shape[0]
        part_size_n = layer.output_size_per_partition
        part_size_k = layer.input_size_per_partition

        out_shape = x.shape[:-1] + (part_size_n, )

        if layer.marlin_state == GPTQMarlinState.REPACK:
            layer.marlin_state = GPTQMarlinState.READY

            # Newly generated tensors need to replace existing tensors that are
            # already registered as parameters by vLLM (and won't be freed)
            def replace_tensor(name, new_t):
                # It is important to use resize_() here since it ensures
                # the same buffer is reused
                getattr(layer, name).resize_(new_t.shape)
                getattr(layer, name).copy_(new_t)
                del new_t

            cur_device = layer.weight_packed.device

            # Reset g_idx related tensors
            layer.g_idx = Parameter(torch.empty(0,
                                                dtype=torch.int,
                                                device=cur_device),
                                    requires_grad=False)
            layer.g_idx_sort_indices = Parameter(torch.empty(
                0, dtype=torch.int, device=cur_device),
                                                 requires_grad=False)

            # Repack weights
            marlin_qweight = ops.gptq_marlin_repack(
                layer.weight_packed.t().contiguous(), layer.g_idx_sort_indices,
                part_size_k, part_size_n, self.num_bits)

            replace_tensor("weight_packed", marlin_qweight)

            # Permute scales
            scales_size_k = part_size_k
            scales_size_n = part_size_n

            marlin_scales = marlin_permute_scales(
                layer.weight_scale.squeeze().t().contiguous(), scales_size_k,
                scales_size_n, layer.group_size, self.num_bits)
            replace_tensor("weight_scale", marlin_scales)

        output = ops.gptq_marlin_gemm(reshaped_x, layer.weight_packed,
                                      layer.weight_scale, layer.g_idx,
                                      layer.g_idx_sort_indices,
                                      layer.workspace, self.num_bits, size_m,
                                      part_size_n, part_size_k,
                                      layer.is_k_full)
        return output.reshape(out_shape)
