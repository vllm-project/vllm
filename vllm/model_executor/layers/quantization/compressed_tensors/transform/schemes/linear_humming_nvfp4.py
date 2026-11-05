# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear import (
    HummingHadamardNvFp4LinearKernel,
    _get_linear_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsScheme,
    CompressedTensorsW4A4Fp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
    TransformTuple,
)

__all__ = ["is_humming_hadamard_fp4_scheme", "HummingNvFP4HadamardLinearMethod"]

HADAMARD_BLOCK_SIZES = (16, 32, 64, 128)


def is_humming_hadamard_fp4_scheme(
    quant_scheme: CompressedTensorsScheme | None,
    input_tfms: dict[int, TransformTuple],
    output_tfms: dict[int, TransformTuple],
) -> bool:
    """Whether Humming can fuse the input Hadamard with W4A4 quantization.

    Only deterministic Sylvester transforms qualify: Humming computes them
    without reading the checkpoint matrix.
    """
    if (
        not isinstance(quant_scheme, CompressedTensorsW4A4Fp4)
        or quant_scheme.use_a16
        or output_tfms
        or not input_tfms
    ):
        return False
    schemes = [input_tfm.scheme for input_tfm in input_tfms.values()]
    return (
        len({scheme.head_dim for scheme in schemes}) == 1
        and all(
            scheme.type == "hadamard" and scheme.head_dim in HADAMARD_BLOCK_SIZES
            for scheme in schemes
        )
        and _get_linear_backend(quantization="nvfp4_w4a4") in ("auto", "humming")
        and HummingHadamardNvFp4LinearKernel.is_supported()[0]
    )


class HummingNvFP4HadamardLinearMethod(CompressedTensorsLinearTransformMethod):
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
        assert isinstance(layer.scheme, CompressedTensorsW4A4Fp4)
        layer.scheme.kernel = HummingHadamardNvFp4LinearKernel(
            layer.scheme.kernel.config,
            hadamard_block_size=next(iter(self.input_tfms.values())).scheme.head_dim,
        )
        return super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, x, bias)
