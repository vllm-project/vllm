# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms import current_platform

from .base_linear import BaseLinearLayerWithLoRA


class ReplicatedLinearWithLoRA(BaseLinearLayerWithLoRA):
    def __init__(self, base_layer: ReplicatedLinear) -> None:
        super().__init__(
            base_layer,
        )
        # To ensure interface compatibility, set to 1 always.
        self.output_size = self.base_layer.output_size
        self.n_slices = 1

    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Forward of ReplicatedLinearWithLoRA

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None

        # Matrix multiply.
        output = self.apply(input_, bias)

        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None

        if not self.base_layer.return_bias:
            return output

        return output, output_bias

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        # Use the base layer's forward to preserve any subclass-specific
        # behavior (e.g., specialized router kernels and output dtype handling).
        base_output = self.base_layer(x)
        output = base_output[0] if isinstance(base_output, tuple) else base_output

        original_shape = output.shape if output.ndim == 3 else None
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
            output,
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            1.0,
            self.output_slices,
        )
        if not current_platform.can_update_inplace():
            output = lora_output

        if original_shape is not None:
            output = output.reshape(original_shape)

        return output

    # ReplicatedLinear should always be replaced, regardless of the fully
    # sharded LoRAs setting, because it is, by definition, copied per GPU.
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return isinstance(source_layer, ReplicatedLinear)

    def slice_lora_a(
        self, lora_a: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice lora a if splitting for tensor parallelism."""
        return lora_a

    def slice_lora_b(
        self, lora_b: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice lora b if splitting with tensor parallelism."""
        return lora_b
