# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MarlinFP8ScaledMMLinearKernel,
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.model_executor.parameter import BlockQuantScaleParameter

logger = init_logger(__name__)


class QuarkW8A8Fp8Block(QuarkScheme):
    """Quark FP8 block-scaled W8A8 linear scheme.

    DeepSeek V4 Pro uses Quark layer overrides for some attention projections
    with FP8 block-scaled weights and dynamic per-128 activation scales.
    """

    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
    ):
        self.weight_config = weight_config
        self.input_config = input_config
        self.weight_block_size = list(weight_config.get("block_size") or [128, 128])
        self.input_dtype = get_current_vllm_config().model_config.dtype
        self.out_dtype = torch.get_default_dtype()
        self.fp8_linear = None

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
        input_size = kwargs.get("input_size", input_size_per_partition)
        output_size = kwargs.get("output_size", sum(output_partition_sizes))
        output_size_per_partition = sum(output_partition_sizes)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size

        validate_fp8_block_shape(
            layer,
            input_size,
            output_size,
            input_size_per_partition,
            output_partition_sizes,
            self.weight_block_size,
        )

        weight = create_fp8_weight_parameter(
            output_size_per_partition,
            input_size_per_partition,
            weight_loader,
        )
        layer.register_parameter("weight", weight)

        scale_dtype = (
            torch.float8_e8m0fnu
            if self.weight_config.get("scale_type") == "float8_e8m0fnu"
            else None
        )
        weight_scale = create_fp8_scale_parameter(
            BlockQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            self.weight_block_size,
            weight_loader,
            scale_dtype=scale_dtype,
        )
        # DeepSeek V4 weight mappers route checkpoint ".scale" tensors here.
        layer.register_parameter("weight_scale_inv", weight_scale)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=kFp8Dynamic128Sym,
            weight_quant_key=kFp8Static128BlockSym,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )
        logger.info_once(
            "Selected %s for QuarkW8A8Fp8Block",
            type(self.fp8_linear).__name__,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.fp8_linear is None:
            raise RuntimeError("FP8 block linear kernel was not initialized.")
        if isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel):
            self.fp8_linear.process_weights_after_loading(layer)
            return
        layer.input_scale = None
        self.fp8_linear.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fp8_linear is None:
            raise RuntimeError("FP8 block linear kernel was not initialized.")
        return self.fp8_linear.apply_weights(layer, x, bias)
