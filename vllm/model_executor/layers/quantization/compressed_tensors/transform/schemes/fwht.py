# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.utils import (  # noqa: E501
    TransformTuple)
from vllm.model_executor.parameter import MissingParameter


def is_deterministic_hadamard_scheme(input_tfms: dict[int, TransformTuple],
                                     output_tfms: dict[int, TransformTuple]):
    return all(tfm.scheme.type == "hadamard"
               for tfm in (list(input_tfms.values()) +
                           list(output_tfms.values())))


# Because qutlass fuses hadamard with quantization, it cannot automatically be
# composed with kernels in the way CompressedTensorsLinearTransformMethod does.
# Therefore, a separate scheme must be created for each quantized dtype
class FastWalshHadamardTransform(CompressedTensorsLinearTransformMethod):

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs):
        self.quant_method.create_weights(layer, *args, **kwargs)

        if len(self.input_tfms) > 0:
            scheme_name = list(self.input_tfms.values())[0].scheme_name
            location = list(self.input_tfms.values())[0].args.location
            transform_name = f"{scheme_name}_{location}"

            transform = DeterministicHadamardTransform(self.input_tfms, layer)
            layer.register_module(transform_name, transform)
            self.input_transform = transform

        if len(self.output_tfms) > 0:
            scheme_name = list(self.output_tfms.values())[0].scheme_name
            location = list(self.output_tfms.values())[0].args.location
            transform_name = f"{scheme_name}_{location}"

            transform = DeterministicHadamardTransform(self.output_tfms, layer)
            layer.register_module(transform_name, transform)
            self.output_transform = transform


class DeterministicHadamardTransform(torch.nn.Module):
    weight: MissingParameter

    def __init__(self, transforms: dict[int, TransformTuple],
                 layer: torch.nn.Module):
        super().__init__()

        self.weight = MissingParameter()

    def forward(self, value: torch.Tensor, part_id: int = 0) -> torch.Tensor:
        # TODO
        return value
