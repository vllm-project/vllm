
from typing import Optional, Union, cast

import torch
from vllm.config.lora import LoRAConfig
from transformers import PretrainedConfig
import torch.nn as nn
import torch.nn.functional as F

from vllm.lora.layers.replicated_linear import ReplicatedLinearWithLoRA
from vllm.model_executor.layers.linear import ReplicatedLinear

class MergedReplicatedLinearWithLoRA(ReplicatedLinearWithLoRA):
    
    """ReplicatedLinear layer that is composed of multiple sublayers (slices)
    packed together (eg. gate_proj + up_proj -> gate_up_proj).

    This means we have multiple LoRAs, each applied to one slice of the layer.
    """
    # MergedReplicatedLinear
    def __init__(self, base_layer: ReplicatedLinear) -> None:
        super().__init__(base_layer)
        # Multiple LoRA layers based on output_sizes
        self.output_slices = tuple(self.base_layer.output_sizes)
        self.n_slices = len(self.output_slices)

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        """
        Create LoRA weights for each slice of the merged layer.
        """
        self.lora_config = lora_config

        # For ReplicatedLinear, lora_a and lora_b are not sharded
        lora_a_out_size = lora_config.max_lora_rank
        
        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_out_size,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(self.n_slices))
        
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                output_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for output_size in self.output_slices)
        
        if lora_config.bias_enabled:
            self.lora_bias_stacked = tuple(
                torch.zeros(
                    max_loras,
                    1,
                    output_size,
                    dtype=lora_config.lora_dtype,
                    device=self.device,
                ) for output_size in self.output_slices)

    def slice_lora_a(
        self, lora_a: list[Union[torch.Tensor, None]]
    ) -> list[Union[torch.Tensor, None]]:
        # No slicing needed for ReplicatedLinear
        return lora_a

    def slice_lora_b(
        self, lora_b: list[Union[torch.Tensor, None]]
    ) -> list[Union[torch.Tensor, None]]:
        # No slicing needed for ReplicatedLinear
        return lora_b

    def slice_bias(
        self, bias: list[Union[torch.Tensor, None]]
    ) -> list[Union[torch.Tensor, None]]:
        # No slicing needed for ReplicatedLinear
        return bias

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        lora_bias: Optional[torch.Tensor] = None,
    ):
        self.reset_lora(index)

        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)
            if lora_bias is not None:
                lora_bias = self.slice_bias(lora_bias)

        for i in range(self.n_slices):
            if (lora_a_i := lora_a[i]) is not None:
                self.lora_a_stacked[i][
                    index, 0, :lora_a_i.shape[1], :lora_a_i.shape[0]].copy_(
                        lora_a_i.T, non_blocking=True)
            if (lora_b_i := lora_b[i]) is not None:
                self.lora_b_stacked[i][
                    index, 0, :lora_b_i.shape[1], :lora_b_i.shape[0]].copy_(
                        lora_b_i.T, non_blocking=True)

        if lora_bias is not None:
            self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                          self.lora_bias_stacked)
            for i in range(self.n_slices):
                if (lora_bias_i := lora_bias[i]) is not None:
                    self.lora_bias_stacked[i][index, 0, :lora_bias_i.shape[0]].copy_(
                        lora_bias_i.T, non_blocking=True)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        # print(f"Checking if {source_layer} can be replaced by {cls.__name__}, {packed_modules_list} packed modules")

        return (type(source_layer) is ReplicatedLinear #MergedReplicatedLinear
                and len(packed_modules_list) >= 2)
