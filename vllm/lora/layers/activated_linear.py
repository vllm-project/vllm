# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Optional

import torch

from vllm.forward_context import get_forward_context
from vllm.lora.punica_wrapper import PunicaWrapperBase

from .base_linear import BaseLinearLayerWithLoRA

if TYPE_CHECKING:
    from .base import BaseLayerWithLoRA


class LinearLayerWithActivatedLoRAMixin:

    base_layer: BaseLinearLayerWithLoRA
    punica_wrapper: PunicaWrapperBase
    lora_a_stacked: torch.tensor
    lora_b_stacked: torch.tensor
    lora_bias_stacked: Optional[tuple[torch.Tensor, ...]]
    output_slices: tuple[int, ...]

    @classmethod
    def maybe_mixin(cls, lora_cls: "type[BaseLayerWithLoRA]"):
        if issubclass(lora_cls, BaseLinearLayerWithLoRA):
            return type(lora_cls.__name__.replace("LoRA", "ActivatedLoRA"),
                        (cls, lora_cls), {})
        else:
            return lora_cls

    def apply(self,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        # Extract aLoRA batch metadata from forward context
        alora_metadata = get_forward_context().alora_metadata

        mask1d = alora_metadata.mask1d
        mask2d = mask1d.unsqueeze(1).to(output.dtype)

        # Clone base layer output before running LoRA
        # TODO(tdoublep): pass in mask1d and only operate on valid entries
        orig_out = output.clone()

        # Apply LoRA in‐place on `output`:
        self.punica_wrapper.add_lora_linear(output, x, self.lora_a_stacked,
                                            self.lora_b_stacked,
                                            self.lora_bias_stacked, 1.0,
                                            self.output_slices)
        # Apply alora mask
        final_output = orig_out.mul(mask2d) + output.mul(1.0 - mask2d)
        return final_output
