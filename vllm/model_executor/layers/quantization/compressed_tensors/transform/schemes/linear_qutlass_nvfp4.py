# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsScheme,
    CompressedTensorsW4A4Fp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
    TransformTuple,
)

__all__ = ["is_qutlass_fp4_scheme", "QutlassNvFP4LinearMethod"]


def is_qutlass_fp4_scheme(
    quant_scheme: CompressedTensorsScheme | None,
    input_tfms: dict[int, TransformTuple],
) -> bool:
    return (
        isinstance(quant_scheme, (CompressedTensorsW4A4Fp4,))
        and len(input_tfms) == 1
        and input_tfms[0].scheme.head_dim == quant_scheme.group_size
    )


class QutlassNvFP4LinearMethod(CompressedTensorsLinearTransformMethod):
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        # initializes fp4 qparams
        assert isinstance(layer.scheme, (CompressedTensorsW4A4Fp4,))
        ret = super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

        assert self.input_transform is not None
        assert len(self.input_transform.weight) == 1
        assert self.input_transform.weight[0].size(0) == layer.scheme.group_size

        return ret

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError()
