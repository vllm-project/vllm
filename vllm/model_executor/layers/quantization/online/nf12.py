# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Module

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
from vllm.model_executor.layers.quantization.utils.quant_utils import kNf12
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Nf12OnlineLinearMethod(OnlineLinearBase):
    """Losslessly pack BF16 weights during loading and use native NF12 GEMMs."""

    weight_quant_key = kNf12
    activation_quant_key = None

    def __init__(self):
        super().__init__()
        if not current_platform.is_cuda() or not current_platform.has_device_capability(
            80
        ):
            raise ValueError(
                "NF12 requires an NVIDIA GPU with compute capability >= 8.0"
            )
        if self.input_dtype != torch.bfloat16:
            raise ValueError("NF12 requires BF16 weights and activations")
        self.unquantized = UnquantizedLinearMethod()
        self.fallback = False

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        weight = self.get_weight_for_quantization(layer)
        if weight.dtype != torch.bfloat16:
            raise ValueError("NF12 requires BF16 weights and activations")
        n, k = weight.shape
        self.fallback = n % 16 != 0 or k % 64 != 0
        if self.fallback:
            logger.warning_once(
                "Keeping an NF12-targeted linear layer BF16: "
                "shape (%s, %s) is unsupported",
                n,
                k,
            )
            replace_parameter(layer, "weight", weight.data)
            self.unquantized.process_weights_after_loading(layer)
        else:
            import nf12

            encoded = nf12.encode(weight.contiguous())
            replace_parameter(layer, "weight", encoded.packed)
            replace_parameter(layer, "weight_escapes", encoded.escapes)
            layer.weight_exp_base = encoded.exp_base
        self.release_requantization_source_weights(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self, layer: Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.fallback:
            return self.unquantized.apply(layer, x, bias)
        return torch.ops.nf12.linear(
            x, layer.weight, layer.weight_escapes, layer.weight_exp_base, bias
        )
