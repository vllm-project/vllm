# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig

from .replicated_linear import ReplicatedLinearWithLoRA


class ClassificationHeadWithLoRA(ReplicatedLinearWithLoRA):
    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        # Preserve ordinary LoRA A/B support for classification heads.
        super().create_lora_weights(max_loras, lora_config, model_config)
        dtype = self.base_layer.params_dtype
        self.full_weight_stacked = torch.zeros(
            max_loras,
            1,
            self.output_size,
            self.input_size,
            dtype=dtype,
            device=self.device,
        )
        self.full_bias_stacked = torch.zeros(
            max_loras,
            self.output_size,
            dtype=dtype,
            device=self.device,
        )
        self.full_module_enabled = torch.zeros(
            max_loras, dtype=torch.bool, device=self.device
        )

    def reset_module_to_save(self, index: int) -> None:
        self.full_weight_stacked[index].zero_()
        self.full_bias_stacked[index].zero_()
        self.full_module_enabled[index] = False

    def set_module_to_save(
        self,
        index: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        self.reset_module_to_save(index)
        self.full_weight_stacked[index, 0].copy_(weight, non_blocking=True)
        if bias is not None:
            self.full_bias_stacked[index].copy_(bias, non_blocking=True)
        self.full_module_enabled[index] = True
