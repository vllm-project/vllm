# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)

from .base import INCLinearScheme

if TYPE_CHECKING:
    from ..resolver import INCLayerConfig


class INCXPUW4A16LinearScheme(INCLinearScheme):
    def __init__(self, layer_config: "INCLayerConfig") -> None:
        self.weight_bits = layer_config.bits
        self.group_size = layer_config.group_size
        self.sym = layer_config.sym
        self.pack_factor = 32 // self.weight_bits

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size  # Unused.
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        scales_and_zp_size = input_size_per_partition // self.group_size

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        scales = GroupQuantScaleParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )
        qzeros = PackedvLLMParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        g_idx = RowvLLMParameter(
            data=torch.tensor(
                [i // self.group_size for i in range(input_size_per_partition)],
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("g_idx", g_idx)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.data.device

        qweight_ct = layer.qweight.data.t().contiguous()
        layer.qweight = Parameter(qweight_ct.t(), requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)
        layer.qzeros = Parameter(
            torch.tensor([8], dtype=torch.int8, device=device),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.qweight.shape[1],)
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            layer.qweight,
            bias,
            layer.scales,
            layer.qzeros,
            self.group_size,
            None,
        )
        return out.reshape(out_shape)
