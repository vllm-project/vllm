# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    maybe_round_up_hidden_size_and_intermediate_size,
    pad_fp8_moe_weights,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


class FusedFp8MoEMethodBase(FusedMoEMethodBase):
    """Base for FP8 MoE quant methods; pads the loaded weights to the FP8
    backend's alignment. Subclasses set ``fp8_backend`` and ``weight_quant_key``.
    """

    fp8_backend: Fp8MoeBackend
    weight_quant_key: QuantKey

    def update_kernel_sizes(self) -> tuple[int, int]:
        """Round the MoE config's hidden and intermediate sizes up to what the
        FP8 backend needs and return them."""
        hidden_size, intermediate_size = (
            maybe_round_up_hidden_size_and_intermediate_size(
                self.fp8_backend,
                self.weight_quant_key,
                self.moe.activation,
                self.moe.hidden_dim,
                self.moe.intermediate_size_per_partition,
            )
        )
        self.moe.hidden_dim = hidden_size
        self.moe.intermediate_size_per_partition = intermediate_size
        return hidden_size, intermediate_size

    def maybe_pad_weights(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zero-pad the loaded weights to the FP8 backend's sizes."""
        hidden_size, intermediate_size = self.update_kernel_sizes()
        if (hidden_size, intermediate_size) == (w2.shape[1], w2.shape[2]):
            return w13, w2, w13_scale, w2_scale
        return pad_fp8_moe_weights(
            self.moe, w13, w2, w13_scale, w2_scale, hidden_size, intermediate_size
        )
