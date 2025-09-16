# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import ReplicatedLinear

from .base import BaseLayerWithLoRA
from .utils import _get_layer_weight, _get_lora_device


class ClassifierWithLoRA(BaseLayerWithLoRA):
    """
    TODO: Add docs
    """

    def __init__(self, base_layer: ReplicatedLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.dtype = _get_layer_weight(base_layer).dtype
        self.input_size = base_layer.input_size
        self._label_slot: list = []
        self.device = _get_lora_device(base_layer)

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.lora_config = lora_config

        self.max_class_label = lora_config.max_num_labels
        self._label_slot = [-1] * max_loras
        self.lora_a_stacked = torch.zeros(
            max_loras,
            1,
            self.max_class_label,
            self.input_size,
            dtype=self.dtype,
            device=self.device,
        )
        pass

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self._label_slot[index] = -1

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        lora_bias: Optional[torch.Tensor] = None,
    ):
        self.reset_lora(index)

        self.lora_a_stacked[index,
                            0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                lora_a.T, non_blocking=True)
        self._label_slot[index] = lora_a.shape[1]

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward of ClassifierWithLoRA

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
    
        """
        # We set the buffer to be float32 by default, refer to:
        # https://github.com/triton-lang/triton/issues/1387
        y = torch.zeros((self.input_size, self.max_class_label),
                        dtype=torch.float32,
                        device=input_.device)
        lora_a = (self.lora_a_stacked, )
        self.punica_wrapper.add_shrink(y,
                                       input_,
                                       lora_a,
                                       scale=1.0,
                                       add_input=True)
        # Cast y using self._label_slot
        return y
