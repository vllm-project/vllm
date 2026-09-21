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

        self.max_lora_cls_labels = lora_config.max_lora_cls_labels or self.output_size
        self.padded_num_labels = max(self.output_size, self.max_lora_cls_labels)
        self.full_weight_stacked = torch.zeros(
            max_loras,
            1,
            self.padded_num_labels,
            self.input_size,
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.full_bias_stacked = torch.zeros(
            max_loras,
            self.padded_num_labels,
            dtype=self.base_layer.params_dtype,
            device=self.device,
        )
        self.full_module_enabled = torch.zeros(
            max_loras, dtype=torch.bool, device=self.device
        )
        # Zero marks an unused slot; positive values are unpadded label counts.
        self.full_module_num_labels = [0] * max_loras
        self._output_lora_indices: tuple[int, ...] = ()

    def set_output_mapping(self, slot_indices: tuple[int, ...]) -> None:
        self._output_lora_indices = slot_indices

    def reset_module_to_save(self, index: int) -> None:
        self.full_weight_stacked[index].zero_()
        self.full_bias_stacked[index].zero_()
        self.full_module_enabled[index] = False
        self.full_module_num_labels[index] = 0

    def set_module_to_save(
        self,
        index: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        self.reset_module_to_save(index)
        num_labels = weight.size(0)
        self.full_weight_stacked[index, 0, :num_labels].copy_(weight, non_blocking=True)
        if bias is not None:
            self.full_bias_stacked[index, :num_labels].copy_(bias, non_blocking=True)
        self.full_module_num_labels[index] = num_labels
        self.full_module_enabled[index] = True

    def forward(self, input_: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        # TODO base_result maybe don't need compute truly.
        base_result = super().forward(input_)
        base_result = base_result[0] if isinstance(base_result, tuple) else base_result

        slot_indices = self._output_lora_indices
        assert len(slot_indices) == base_result.size(0), (
            "Classification rows do not match LoRA request mapping"
        )
        num_labels = [
            self.full_module_num_labels[index] if index >= 0 else 0
            for index in slot_indices
        ]

        if not any(num_labels):
            return base_result

        output = base_result.new_zeros(base_result.size(0), self.padded_num_labels)
        output[:, : self.output_size].copy_(base_result)
        self.punica_wrapper.apply_lora_full_linear(
            output,
            input_.to(self.full_weight_stacked.dtype),
            self.full_weight_stacked,
            self.full_bias_stacked,
            self.full_module_enabled,
        )

        # Rows without a full head keep the base classifier output size.
        num_labels = [size or self.output_size for size in num_labels]
        if len(set(num_labels)) == 1:
            return output[:, : num_labels[0]]

        output = [output[row, :size] for row, size in enumerate(num_labels)]
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
            output,
            x.to(self.lora_a_stacked[0].dtype),
            self.lora_a_stacked[0],
            self.lora_b_stacked[0],
            1.0,
        )
        if not current_platform.can_update_inplace():
            output = lora_output

        if original_shape is not None:
            output = output.reshape(original_shape)

        return output
