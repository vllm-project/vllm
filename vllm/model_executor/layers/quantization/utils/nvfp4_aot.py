# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearKernel
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    MarlinNvFp4LinearKernel,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)


class NvFp4LinearRuntime:
    """Select the execution method after an NVFP4 checkpoint is loaded."""

    def __init__(self, kernel: NvFp4LinearKernel) -> None:
        self.kernel = kernel
        self.unquantized_method = (
            UnquantizedLinearMethod()
            if envs.VLLM_NVFP4_DEQUANT_AT_LOAD
            and isinstance(kernel, MarlinNvFp4LinearKernel)
            else None
        )

    @property
    def uses_unquantized_linear(self) -> bool:
        return self.unquantized_method is not None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.unquantized_method is None:
            self.kernel.process_weights_after_loading(layer)
            return

        logger.warning_once(
            "VLLM_NVFP4_DEQUANT_AT_LOAD is enabled. Expanding NVFP4 weights "
            "to BF16/FP16 at model load and using the unquantized linear "
            "runtime. This uses roughly four times the weight memory."
        )
        weight = dequantize_to_dtype(
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.params_dtype,
            block_size=16,
            swizzle=False,
        )
        replace_parameter(layer, "weight", weight.contiguous())
        self.unquantized_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.unquantized_method is not None:
            return self.unquantized_method.apply(layer, x, bias)
        return self.kernel.apply_weights(layer=layer, x=x, bias=bias)
