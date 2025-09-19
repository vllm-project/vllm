# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import ReplicatedLinear

from .base import BaseLayerWithLoRA
from .utils import _get_layer_device, _get_layer_dtype


class ClassifierWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: ReplicatedLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.input_size = base_layer.input_size
        self._label_slot: list = []
        self.device = _get_layer_device(base_layer)

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
            dtype=_get_layer_dtype(self.base_layer),
            device=self.device,
        )

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
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        org_output = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias)

        y = torch.zeros(self.input_size,
                        self.max_class_label,
                        device=input_.device)
        lora_weight = tuple(self.lora_a_stacked, )

        self.punica_wrapper.add_shrink(y.unsqueeze(dim=0),
                                       input_,
                                       lora_weight,
                                       scale=1.0)
        #TODO Cast y using self._label_slot
        # Keep consistent with the base_layer output
        return y, None

    # ReplicatedLinear should always be replaced, regardless of the fully
    # sharded LoRAs setting, because it is, by definition, copied per GPU.
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is ReplicatedLinear
