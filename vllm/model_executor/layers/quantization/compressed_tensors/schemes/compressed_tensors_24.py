# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
)

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)

__all__ = ["CompressedTensors24"]


class CompressedTensors24(CompressedTensorsScheme):
    def __init__(
        self,
        quantized: bool = False,
        weight_quant: QuantizationArgs | None = None,
        input_quant: QuantizationArgs | None = None,
        model_compression_config: dict[str, Any] | None = None,
    ):
        raise NotImplementedError("Sparse24 models are no longer supported by vLLM")

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError("Sparse24 models are no longer supported by vLLM")

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        raise NotImplementedError("Sparse24 models are no longer supported by vLLM")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError("Sparse24 models are no longer supported by vLLM")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Sparse24 models are no longer supported by vLLM")
