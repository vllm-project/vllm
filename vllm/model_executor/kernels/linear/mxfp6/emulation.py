# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    dequant_mxfp6,
    quant_dequant_mxfp6,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Dynamic,
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
)

from .base import MxFp6LinearKernel, MxFp6LinearLayerConfig

_WEIGHT_DEQUANT_FUNCS: dict[QuantKey, Callable[..., torch.Tensor]] = {
    kMxfp6E3M2Static: partial(dequant_mxfp6, quant_dtype="fp6_e3m2"),
    kMxfp6E2M3Static: partial(dequant_mxfp6, quant_dtype="fp6_e2m3"),
}

_ACTIVATION_QUANT_DEQUANT_FUNCS: dict[
    QuantKey, Callable[[torch.Tensor], torch.Tensor]
] = {
    kMxfp4Dynamic: quant_dequant_mxfp4,
    kMxfp6E3M2Dynamic: partial(quant_dequant_mxfp6, quant_dtype="fp6_e3m2"),
    kMxfp6E2M3Dynamic: partial(quant_dequant_mxfp6, quant_dtype="fp6_e2m3"),
}


class EmulationMxfp6LinearKernel(MxFp6LinearKernel):
    """Software emulation fallback for OCP MXFP4/MXFP6 (dequant + F.linear)."""

    def __init__(self, config: MxFp6LinearLayerConfig) -> None:
        super().__init__(config)
        self.dequant_func = _WEIGHT_DEQUANT_FUNCS[config.weight_quant_key]
        if config.activation_quant_key is None:
            self.quant_dequant_func: Callable[[torch.Tensor], torch.Tensor] = (
                lambda x: x
            )  # no input Q/DQ for weight-only
        else:
            self.quant_dequant_func = _ACTIVATION_QUANT_DEQUANT_FUNCS[
                config.activation_quant_key
            ]

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp6LinearLayerConfig) -> tuple[bool, str | None]:
        if config.weight_quant_key not in (
            kMxfp6E2M3Static,
            kMxfp6E3M2Static,
        ):
            return False, "only supports MXFP6 weights"
        if config.activation_quant_key not in (
            None,
            kMxfp4Dynamic,
            kMxfp6E3M2Dynamic,
            kMxfp6E2M3Dynamic,
        ):
            return False, "only supports MXFP4 or MXFP6 or unquantized activations"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dq_w = self.dequant_func(layer.weight, layer.weight_scale, x.dtype)
        qdq_x = self.quant_dequant_func(x)
        return F.linear(qdq_x, dq_w, bias)
