# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
import math

import torch

from vllm import _custom_ops as ops, envs
from vllm.model_executor.layers.quantization.quark.schemes.quark_scheme import (
    QuarkScheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)


class QuarkW4A16Int4(QuarkScheme):
    """Quark packed INT4 weight-only linear scheme.

    Quark export pack order is canonicalized to the layout used by ``awq_*`` ops.
    This is a kernel layout detail only; checkpoints still load via ``quark``.
    """

    _AWQ_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)

    def __init__(self, group_size: int, pack_method: str, is_symmetric: bool):
        self.group_size = group_size
        self.pack_factor = 8
        self.pack_reorder = pack_method == "reorder"
        self.is_symmetric = is_symmetric

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        group_size = (
            self.group_size if self.group_size != -1 else input_size_per_partition
        )
        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        packed_output_size_per_partition = math.ceil(
            output_size_per_partition / self.pack_factor
        )
        layer.output_size_per_partition = output_size_per_partition
        layer.packed_output_size_per_partition = packed_output_size_per_partition

        def weight_scale_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            *args,
            **kwargs,
        ) -> None:
            if loaded_weight.shape[1] < param.data.shape[1]:
                padded_weight = loaded_weight.new_zeros(param.data.shape)
                padded_weight[:, : loaded_weight.shape[1]] = loaded_weight
                loaded_weight = padded_weight
            weight_loader(param, loaded_weight, *args, **kwargs)

        weight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                packed_output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        num_groups = input_size_per_partition // group_size
        weight_zero_point = PackedvLLMParameter(
            data=torch.zeros(
                num_groups,
                packed_output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                packed_output_size_per_partition * self.pack_factor,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_scale_loader,
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_zero_point", weight_zero_point)
        layer.register_parameter("weight_scale", weight_scale)

    def _pack_order(self, device: torch.device) -> torch.Tensor:
        if self.pack_reorder:
            return torch.tensor(
                self._AWQ_PACK_ORDER,
                device=device,
                dtype=torch.int32,
            )
        return torch.arange(self.pack_factor, device=device, dtype=torch.int32)

    def _awq_pack_order(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            self._AWQ_PACK_ORDER,
            device=device,
            dtype=torch.int32,
        )

    def _canonicalize_packed_weight(self, packed_weight: torch.Tensor) -> torch.Tensor:
        source_shifts = self._pack_order(packed_weight.device) * 4
        target_shifts = self._awq_pack_order(packed_weight.device) * 4

        values = (packed_weight.to(torch.int32)[..., None] >> source_shifts) & 0xF
        if self.is_symmetric:
            values = values ^ 0x8
        packed = (values.to(torch.int64) << target_shifts.to(torch.int64)).sum(dim=-1)
        return packed.to(torch.int32)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = self._canonicalize_packed_weight(layer.weight.data)
        layer.weight_zero_point.data = self._canonicalize_packed_weight(
            layer.weight_zero_point.data
        )
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_zero_point = torch.nn.Parameter(
            layer.weight_zero_point.data, requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )
        output_size = layer.output_size_per_partition
        packed_output_size = layer.packed_output_size_per_partition * self.pack_factor
        if output_size < packed_output_size:
            layer.weight_scale.data[:, output_size:].zero_()

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        qweight = layer.weight
        scales = layer.weight_scale
        qzeros = layer.weight_zero_point
        output_size = layer.output_size_per_partition
        reshaped_x = x.reshape(-1, x.shape[-1])

        is_output_padded = output_size < qweight.shape[-1] * self.pack_factor
        if is_output_padded:
            shifts = self._awq_pack_order(qweight.device) * 4

            weight = (qweight.to(torch.int32)[..., None] >> shifts) & 0xF
            weight = weight.reshape(qweight.shape[0], -1)[:, :output_size]

            zeros = (qzeros.to(torch.int32)[..., None] >> shifts) & 0xF
            zeros = zeros.reshape(qzeros.shape[0], -1)[:, :output_size]
            group_size = qweight.shape[0] // scales.shape[0]
            zeros = zeros.repeat_interleave(group_size, dim=0)

            scale = scales[:, :output_size].repeat_interleave(group_size, dim=0)
            weight = (weight - zeros).to(scale.dtype) * scale
            out = torch.matmul(reshaped_x, weight)
        elif (
            x.dtype == torch.bfloat16
            or x.shape[:-1].numel() >= 256
            or envs.VLLM_BATCH_INVARIANT
        ):
            out = ops.awq_dequantize(
                qweight,
                scales,
                qzeros,
                0,
                0,
                0,
            )
            out = torch.matmul(reshaped_x, out)
        else:
            out = ops.awq_gemm(
                reshaped_x,
                qweight,
                scales,
                qzeros,
                self.pack_factor,
            )
        if bias is not None:
            out[:, :output_size].add_(bias)
        return out.reshape(x.shape[:-1] + (out.shape[-1],))[..., :output_size]
