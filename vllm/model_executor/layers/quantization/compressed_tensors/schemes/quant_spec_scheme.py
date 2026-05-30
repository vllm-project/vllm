# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Protocol

import torch

from vllm.config.quantization import QuantSpec
from vllm.model_executor.kernels.linear import init_linear_kernel_for_spec
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (  # noqa: E501
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

__all__ = [
    "ACTIVATION_BUILDERS",
    "WEIGHT_BUILDERS",
    "ActivationBuilder",
    "QuantSpecScheme",
    "WeightBuilder",
]


class WeightBuilder(Protocol):
    group_size: int

    def create(
        self,
        *,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None: ...

    def post_load(self, layer: torch.nn.Module) -> None: ...


class ActivationBuilder(Protocol):
    def create(
        self,
        *,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None: ...

    def post_load(self, layer: torch.nn.Module) -> None: ...


WEIGHT_BUILDERS: dict[QuantKey, WeightBuilder] = {}
ACTIVATION_BUILDERS: dict[QuantKey, ActivationBuilder] = {}


class QuantSpecScheme(CompressedTensorsScheme):
    """Dispatch-only compressed-tensors scheme backed by a QuantSpec."""

    group_size: int
    min_capability = 75

    def __init__(self, spec: QuantSpec):
        if spec.weight is None:
            raise ValueError("QuantSpecScheme requires a weight quantization key")

        self.spec = spec
        try:
            self._weight = WEIGHT_BUILDERS[spec.weight]
        except KeyError:
            raise NotImplementedError(
                f"No compressed-tensors weight builder registered for {spec.weight}"
            ) from None

        if spec.activation is None:
            self._act = None
        else:
            try:
                self._act = ACTIVATION_BUILDERS[spec.activation]
            except KeyError:
                raise NotImplementedError(
                    "No compressed-tensors activation builder registered for "
                    f"{spec.activation}"
                ) from None

        self.group_size = self._weight.group_size
        self.kernel = init_linear_kernel_for_spec(spec)

    @classmethod
    def get_min_capability(cls) -> int:
        return cls.min_capability

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None:
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)

        self._weight.create(
            layer=layer,
            output_partition_sizes=output_partition_sizes,
            input_size_per_partition=input_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
            **kwargs,
        )
        if self._act is not None:
            self._act.create(
                layer=layer,
                output_partition_sizes=output_partition_sizes,
                input_size_per_partition=input_size_per_partition,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
                **kwargs,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._weight.post_load(layer)
        if self._act is not None:
            self._act.post_load(layer)
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)


# Register built-in builders after the registries exist.
from .builders import nvfp4  # noqa: E402,F401
