# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.platforms import current_platform

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

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # TODO base_result maybe don't need compute truly.
        base_result = super().forward(input_)
        output = base_result[0] if isinstance(base_result, tuple) else base_result

        full_output = self.punica_wrapper.apply_lora_full_linear(
            output,
            input_,
            self.full_weight_stacked,
            self.full_bias_stacked,
            self.full_module_enabled,
        )

        if full_output is not None:
            output = full_output
        return output

    def _apply_lora_to_output(
        self, x: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        original_shape = output.shape if output.ndim == 3 else None
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        # Classification outputs are request-level, so use prompt-based LoRA routing.
        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_logits(
            output, x, self.lora_a_stacked[0], self.lora_b_stacked[0], 1.0
        )
        if not current_platform.can_update_inplace():
            output = lora_output

        if original_shape is not None:
            output = output.reshape(original_shape)

        return output
