# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Callable, Hashable

import torch
from compressed_tensors.transform import (
    TransformArgs,
    TransformLocation,
    TransformScheme,
)
from torch import Tensor

import vllm._custom_ops as ops
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.compressed_tensors.transform.utils import (  # noqa: E501
    TransformTuple,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parameter import SharedWeightParameter


class HadamardTransform(torch.nn.Module):
    """
    Class which handles weight loading, postprocessing, and application of
    transforms. Meant to be used with `CompressedTensorsLinearTransformMethod`
    and attention transforms method (not implemented yet)
    """

    transforms: dict[int, TransformTuple]  # info parsed from transforms config
    weight: SharedWeightParameter  # container for shared tensors

    scales: dict[int, float]  # hadamard scale, usually sqrt(matrix.size(0))

    def __init__(
        self,
        transforms: dict[int, TransformTuple],
        layer: torch.nn.Module,
        weight_loader: Callable,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
    ):
        super().__init__()
        self.transforms = transforms
        self.scales = {}

        if get_tensor_model_parallel_world_size() > 1:
            raise NotImplementedError(
                "Online transforms with tensor parallelism is not supported"
            )

        # Similar to row/col parallel params, but tensors are separate
        # to allow for loading with shared memory
        self.weight = SharedWeightParameter(weight_loader=weight_loader)

        # create shared partition data for each partition of the original weight
        input_size = input_size_per_partition
        for part_index, (_scheme_name, scheme, args) in self.transforms.items():
            output_size = output_partition_sizes[part_index]
            weight_size = self._get_weight_size(
                layer, scheme, args, input_size, output_size
            )

            data_key = self._get_data_key(scheme, weight_size)
            self.weight.add_partition(
                part_index,
                data_key,
                size=(weight_size, weight_size),
                dtype=scheme.precision,
            )

        # validate that shared tensors and schemes are correct
        self._validate_input_transforms()

    def process_weights_after_loading(self):
        for part_id in self.weight.partitions:
            data = self.weight.partitions[part_id].data

            # required by torch.compile
            self.weight.process_weights_after_loading()

            # precompute scale as a runtime multiply, not division
            # do not fold into weight in order to utilize FWHT
            self.scales[part_id] = 1 / math.sqrt(data.size(0))

            # FUTURE: avoid runtime transpose by processing weights
            # prior to apply

    def forward(self, value: Tensor, part_id: int = 0) -> Tensor:
        if part_id not in self.weight.partitions:
            return value

        # use hadacore if possible
        if self.transforms[part_id].scheme.type == "hadamard":
            if self.transforms[part_id].scheme.head_dim is not None:
                weight_size = self.transforms[part_id].scheme.head_dim
                value = value.unflatten(-1, (-1, weight_size))
                value = ops.hadacore_transform(value)
                value = value.flatten(-2, -1)

                return value

            # sylvester transforms are symmetric, inv => transpose => original
            return ops.hadacore_transform(value)

        # fall back to dense
        else:
            weight = self.weight.partitions[part_id]
            weight = (
                weight if self.transforms[part_id].args.inverse else weight.T
            )  # linear := x(W.T)
            scale = self.scales[part_id]

            if self.transforms[part_id].scheme.head_dim is not None:
                value = value.unflatten(-1, (-1, weight.size(0)))
                value = (
                    dispatch_unquantized_gemm()(
                        self, value.to(weight.dtype), weight, None
                    ).to(value.dtype)
                    * scale
                )
                value = value.flatten(-2, -1)

                return value

            return (
                dispatch_unquantized_gemm()(
                    self, value.to(weight.dtype), weight, None
                ).to(value.dtype)
                * scale
            )

    def _get_data_key(self, scheme: TransformScheme, weight_size: int) -> Hashable:
        return (id(scheme), weight_size)

    def _get_weight_size(
        self,
        layer: torch.nn.Module,
        scheme: TransformScheme,
        args: TransformArgs,
        input_size: int,
        output_size: int,
    ) -> int:
        if scheme.head_dim is not None:
            return scheme.head_dim

        if isinstance(layer, LinearBase):
            if args.location == TransformLocation.INPUT:
                return input_size

            elif args.location == TransformLocation.OUTPUT:
                return output_size

        elif isinstance(layer, VocabParallelEmbedding):
            if args.location == TransformLocation.INPUT:
                return output_size

            elif args.location == TransformLocation.OUTPUT:
                return input_size

        raise ValueError()

    def _validate_input_transforms(self):
        assert len(self.transforms) > 0
        location = list(self.transforms.values())[0].args.location

        if location == TransformLocation.INPUT:
            first_data = self.weight.partitions[0].data
            for partition in self.weight.partitions.values():
                if partition.data.data_ptr() != first_data.data_ptr():
                    raise ValueError("")
