# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator
from itertools import accumulate
from typing import Callable, Optional

import torch
from compressed_tensors.transform import (TransformArgs, TransformConfig,
                                          TransformLocation, TransformScheme)
from compressed_tensors.utils import is_match

from vllm.model_executor.layers.linear import (WEIGHT_LOADER_V2_SUPPORTED,
                                               LinearMethodBase,
                                               QKVCrossParallelLinear)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.module import (  # noqa: E501
    HadamardTransform)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.utils import (  # noqa: E501
    TransformTuple)


class CompressedTensorsLinearTransformMethod(LinearMethodBase):
    """
    Wraps `CompressedTensorsLinearMethod` or `UnquantizedLinearMethod` and adds
    input and output transforms to either side of the original apply method
    """

    @classmethod
    def from_schemes(
        cls,
        quant_method: LinearMethodBase,
        quant_scheme: Optional[CompressedTensorsScheme],
        input_tfms: dict[int, TransformTuple],
        output_tfms: dict[int, TransformTuple],
    ) -> "CompressedTensorsLinearTransformMethod":
        from vllm.model_executor.layers.quantization.compressed_tensors.transform.schemes.linear_qutlass_nvfp4 import (  # noqa: E501
            QutlassNvFP4LinearMethod, is_qutlass_fp4_scheme)

        assert input_tfms or output_tfms

        if is_qutlass_fp4_scheme(quant_scheme, input_tfms):
            return QutlassNvFP4LinearMethod(quant_method, input_tfms,
                                            output_tfms)

        # hadacore or dense gemm is selected by Transform module

        return cls(quant_method, input_tfms, output_tfms)

    def __init__(self, quant_method: LinearMethodBase,
                 input_tfms: dict[int, TransformTuple],
                 output_tfms: dict[int, TransformTuple]):
        self.quant_method = quant_method
        self.input_tfms = input_tfms
        self.output_tfms = output_tfms

        self.input_transform: Optional[HadamardTransform] = None
        self.output_transform: Optional[HadamardTransform] = None

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):

        # get weight loader for transforms
        weight_loader: Callable = extra_weight_attrs.get(
            "weight_loader")  # type: ignore[assignment]

        # HACK: UnquantizedLinearMethod does not support weight loader v2, but
        # transforms (specifically SharedWeightParameter) requires
        # weight loader v2. Until UnquantizedLinearMethod supports v2, we must
        # hack around this by getting weight loader v1 so ULM can load correctly
        quant_method_name = self.quant_method.__class__.__name__
        if quant_method_name not in WEIGHT_LOADER_V2_SUPPORTED:
            if isinstance(layer, QKVCrossParallelLinear):
                weight_loader_v1 = layer.weight_loader_v1
            else:
                weight_loader_v1 = layer.weight_loader
            extra_weight_attrs["weight_loader"] = weight_loader_v1

        self.quant_method.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs)

        # validate schemes
        num_partitions = len(output_partition_sizes)
        self._validate_tfm_schemes(num_partitions)

        # create submodules for weight loading
        if len(self.input_tfms) > 0:
            scheme_name = list(self.input_tfms.values())[0].scheme_name
            location = list(self.input_tfms.values())[0].args.location
            transform_name = f"{scheme_name}_{location}"

            transform = HadamardTransform(self.input_tfms, layer,
                                          weight_loader,
                                          input_size_per_partition,
                                          output_partition_sizes)
            layer.register_module(transform_name, transform)
            self.input_transform = transform

        if len(self.output_tfms) > 0:
            scheme_name = list(self.output_tfms.values())[0].scheme_name
            location = list(self.output_tfms.values())[0].args.location
            transform_name = f"{scheme_name}_{location}"

            transform = HadamardTransform(self.output_tfms, layer,
                                          weight_loader,
                                          input_size_per_partition,
                                          output_partition_sizes)
            layer.register_module(transform_name, transform)
            self.output_transform = transform

        # compute partition ranges for slicing activations
        starts = [0] + list(accumulate(output_partition_sizes))[:-1]
        self.partition_ranges = list(zip(starts, output_partition_sizes))

    def process_weights_after_loading(self, layer):
        self.quant_method.process_weights_after_loading(layer)

        for submodule in layer.children():
            if isinstance(submodule, HadamardTransform):
                submodule.process_weights_after_loading()

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.input_transform is not None:
            x = self.input_transform(x)

        assert bias is None
        x = self.quant_method.apply(layer, x, bias)

        # In most cases, input transforms are preferred over output transforms
        # (@ksayers): confirm that this is done concurrently
        if self.output_transform is not None:
            for part_id, (start, length) in enumerate(self.partition_ranges):
                x[:, start:start + length] = self.output_transform(
                    x[:, start:start + length].contiguous(), part_id=part_id)

        return x

    def _validate_tfm_schemes(self, num_partitions: int):
        if len(self.input_tfms) > 0:
            if 0 not in self.input_tfms:
                raise ValueError("Must have same input")

            for part_index in range(num_partitions):
                if self.input_tfms[part_index] != self.input_tfms[0]:
                    raise ValueError("Must have same input")

        if len(self.output_tfms) > 0:
            scheme_name = list(self.output_tfms.values())[0].scheme_name
            location = list(self.output_tfms.values())[0].args.location

            for tfm in self.output_tfms.values():
                if tfm.scheme_name != scheme_name:
                    raise ValueError("Must have same scheme name")
                if tfm.args.location != location:
                    raise ValueError("Must have same location")

        return self.input_tfms, self.output_tfms


def get_linear_transform_schemes(
    layer: torch.nn.Module, layer_name: str,
    transform_config: Optional[TransformConfig],
    packed_modules_mapping: dict[str, list[str]]
) -> tuple[dict[int, TransformTuple], dict[
        int, TransformTuple]]:  # [input_transform, [output_transform, ...]]
    # there can only be one transform input scheme per (fused) module
    input_tfms = {}
    output_tfms = {}

    partition_names = get_layer_partition_names(layer_name,
                                                packed_modules_mapping)

    for scheme_name, scheme, args in get_schemes_args(transform_config):
        for part_index, part_name in enumerate(partition_names):
            if is_match(part_name, layer, args.targets,
                        args.ignore) and args.is_online():
                if args.location == TransformLocation.INPUT:
                    input_tfms[part_index] = TransformTuple(
                        scheme_name, scheme, args)

                elif args.location == TransformLocation.OUTPUT:
                    output_tfms[part_index] = TransformTuple(
                        scheme_name, scheme, args)

                else:
                    raise ValueError(f"Cannot apply `{args.location}` "
                                     f"transform to `{layer_name}`")

    return (input_tfms, output_tfms)


def get_schemes_args(
    transform_config: Optional[TransformConfig]
) -> Generator[tuple[str, TransformScheme, TransformArgs]]:
    if transform_config is None:
        return

    for scheme_name, scheme in transform_config.config_groups.items():
        for args in scheme.apply:
            yield (scheme_name, scheme, args)


def get_layer_partition_names(
        layer_name: str, packed_modules_mapping: dict[str,
                                                      list[str]]) -> list[str]:
    """
    Get all partition names associated with this layer.
    Names are returned in order of their partition indices.
    
    ```python
    mapping = {"gate_up_proj", "gate_proj", "up_proj"}

    assert get_layer_partition_names(
        "mlp.gate_up_proj", mapping) == ["gate_proj", "up_proj"]
    assert get_layer_partition_names(
        "mlp.down_proj", mapping) == ["down_proj"]
    """
    for fused_suffix, part_suffixes in packed_modules_mapping.items():
        if layer_name.endswith(fused_suffix):
            return [
                layer_name.removesuffix(fused_suffix) + part_suffix
                for part_suffix in part_suffixes
            ]

    return [layer_name]
