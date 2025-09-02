# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsScheme, CompressedTensorsW4A4Fp4)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod, TransformTuple)

__all__ = ["is_qutlass_fp4_scheme", "QutlassFP4LinearMethod"]


def is_qutlass_fp4_scheme(quant_scheme: CompressedTensorsScheme,
                          input_tfms: dict[int, TransformTuple]) -> bool:
    return isinstance(
        quant_scheme,
        (CompressedTensorsW4A4Fp4, )) and len(input_tfms) == 1 and input_tfms[
            0].scheme.head_dim == quant_scheme.group_size


class QutlassFP4LinearMethod(CompressedTensorsLinearTransformMethod):

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        # initializes fp4 qparams
        assert isinstance(layer.scheme, (CompressedTensorsW4A4Fp4, ))
        ret = super().create_weights(layer, input_size_per_partition,
                                     output_partition_sizes, input_size,
                                     output_size, params_dtype,
                                     **extra_weight_attrs)

        assert self.input_transform is not None
        assert len(self.input_transform.weight) == 1
        assert self.input_transform.weight[0].size(
            0) == layer.scheme.group_size

        return ret

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert bias is None

        # x_e2m1, x_e8m0, clip_mask = fusedQuantizeNv(
        #     x, self.input_transform.weight[0], method='quest')
        # x_scale_block = to_blocked(x_e8m0)
        # out = matmul_nvf4_bf16_tn(x_e2m1, layer.weight_scale, x_scale_block,
        #                           layer.weight)

        # # TODO (@ksayers): Confirm that this is done in parallel
        # if self.output_transform is not None:
        #     for part_id, (start, length) in enumerate(self.partition_ranges):
        #         x[:, start:start + length] = self.output_transform(
        #             x[:, start:start + length].contiguous(), part_id=part_id)

        return x
