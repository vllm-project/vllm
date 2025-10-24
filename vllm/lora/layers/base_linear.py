# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, cast

import torch
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed.utils import divide
# yapf: disable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, ReplicatedLinear,
                                               RowParallelLinear)
from vllm.platforms import current_platform

from .base import BaseLayerWithLoRA
from .utils import _get_lora_device


class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        self.device = _get_lora_device(self.base_layer)
        self.lora_bias_stacked: Optional[tuple[torch.Tensor, ...]] = None

        self.output_slices: tuple[int, ...]
        self.tp_size: int
        self.output_size: int
        self.n_slices: int

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.lora_config = lora_config
        #
        if isinstance(self.base_layer, ReplicatedLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, ColumnParallelLinear):
            lora_a_out_size = (lora_config.max_lora_rank if
                               not lora_config.fully_sharded_loras else divide(
                                   lora_config.max_lora_rank, self.tp_size))
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, RowParallelLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = (self.output_size if
                               not lora_config.fully_sharded_loras else divide(
                                   self.output_size, self.tp_size))
        else:
            raise NotImplementedError

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
                lora_b_out_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(self.n_slices))
        if lora_config.bias_enabled:
            lora_bias_out_size = lora_b_out_size
            self.lora_bias_stacked = tuple(
                torch.zeros(
                    max_loras,
                    1,
                    lora_bias_out_size,
                    dtype=lora_config.lora_dtype,
                    device=self.device,
                ) for _ in range(self.n_slices))
        self.output_slices = (self.lora_b_stacked[0].shape[2], )

    def reset_lora(self, index: int):
        for s_index in range(self.n_slices):
            self.lora_a_stacked[s_index][index] = 0
            self.lora_b_stacked[s_index][index] = 0
            if self.lora_config.bias_enabled:
                # Make mypy happy
                self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                              self.lora_bias_stacked)
                self.lora_bias_stacked[s_index][index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        lora_bias: Optional[torch.Tensor] = None,
    ):
        # Except for QKVParallelLinearWithLoRA and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert (len(self.lora_a_stacked) == len(self.lora_b_stacked) ==
                self.n_slices == 1)

        from vllm.lora.training_manager import TrainingManager
        training_manager = TrainingManager.get_instance()
        if not training_manager.is_registered_by_index(index):
            # Only reset LoRA if not registered in training manager (i.e., not training mode)
            self.reset_lora(index)

        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)
            if lora_bias is not None:
                lora_bias = self.slice_bias(lora_bias)

        self.lora_a_stacked[0][index,
                               0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                   lora_a.T, non_blocking=True)
        self.lora_b_stacked[0][index,
                               0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                   lora_b.T, non_blocking=True)
        if lora_bias is not None:

            self.lora_bias_stacked = cast(tuple[torch.Tensor, ...],
                                          self.lora_bias_stacked)
            assert len(self.lora_bias_stacked)
            self.lora_bias_stacked[0][index, 0, :lora_bias.shape[0]].copy_(
                lora_bias.T, non_blocking=True)

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

        if x.requires_grad or output.requires_grad:
            return self._training_apply(x, output)
        else:
            return self._inference_apply(x, output)

    def _inference_apply(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        lora_output: Optional[
            torch.Tensor] = self.punica_wrapper.add_lora_linear(
                output, x, self.lora_a_stacked, self.lora_b_stacked,
                self.lora_bias_stacked, 1.0, self.output_slices)
        if not current_platform.can_update_inplace():
            output = lora_output
        return output

    def _training_apply(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if len(self.lora_a_stacked) <= 0:
            raise ValueError("No LoRA weights found")

        from vllm.lora.training_manager import TrainingManager
        training_manager = TrainingManager.get_instance()

        # Apply ALL LoRAs in the stacked tensors (supports parallel training)
        # We apply even zero-initialized LoRAs to enable gradient flow from scratch
        max_loras = self.lora_a_stacked[0].shape[0]
        for lora_idx in range(max_loras):
            # Check if this slot might be active by looking at tensor shapes
            # We don't check for zero values because LoRAs can start from zero during training
            first_lora_a = self.lora_a_stacked[0][lora_idx, 0, :, :]

            # Apply LoRA if the slot is properly shaped (not uninitialized)
            if first_lora_a.numel() > 0:
                # Apply this LoRA using PyTorch ops (supports autograd)
                # For packed layers (e.g., QKV), apply each component to its output slice
                output_offset = 0
                # Determine which slices to apply (Q/V only for packed QKV)
                try:
                    from vllm.lora.layers.column_parallel_linear import (
                        MergedQKVParallelLinearWithLoRA,
                        QKVParallelLinearWithLoRA,
                    )
                    is_merged_qkv = isinstance(self, MergedQKVParallelLinearWithLoRA)
                    is_single_qkv = isinstance(self, QKVParallelLinearWithLoRA)
                except Exception:
                    is_merged_qkv, is_single_qkv = False, False

                if is_merged_qkv:
                    indices = training_manager.get_qkv_indices_for_training()
                elif is_single_qkv:
                    raise ValueError(f"Single-slice QKV LoRA layer training is not supported")
                else:
                    raise ValueError(f"Unsupported LoRA layer: {self.__class__.__name__}")

                # LoRA scale matching PEFT: alpha / r
                lora_scale = 1.0
                if hasattr(self, 'lora_config') and getattr(self, 'lora_config') is not None:
                    alpha = getattr(self.lora_config, 'lora_alpha', 1.0)
                    rank = getattr(self.lora_config, 'max_lora_rank', 1.0)
                    if rank:
                        lora_scale = float(alpha) / float(rank)

                for i in indices:
                    lora_a = self.lora_a_stacked[i][lora_idx, 0, :, :]  # [rank, input_size]
                    lora_b = self.lora_b_stacked[i][lora_idx, 0, :, :]  # [output_size, rank]

                    # Convert to bfloat16 to match PEFT
                    target_dtype = output.dtype if output.dtype in [torch.bfloat16, torch.float16] else torch.bfloat16

                    # Note:
                    # For training LoRAs, scaling is applied here, not in models.py (lora.optimize()).
                    # (x @ A^T) @ B^T * scaling
                    lora_hidden = x.to(target_dtype) @ lora_a.T.to(target_dtype)
                    lora_output = (lora_hidden @ lora_b.T.to(target_dtype)) * float(lora_scale)
                    lora_output = lora_output.to(output.dtype)

                    # Get the output slice size for this component
                    slice_size = self.output_slices[i]

                    # Add LoRA output to corresponding slice (build delta to avoid in-place on view)
                    delta = torch.zeros_like(output)
                    delta[:, output_offset:output_offset + slice_size] = lora_output
                    output = output + delta
                    output_offset += slice_size

        return output

    @property
    def weight(self) -> torch.Tensor:

        # unquantizedLinear
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        # Compressed Tensor
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        # GPTQ/AWQ
        elif hasattr(self.base_layer, "qweight"):
            return self.base_layer.qweight
        # marlin
        elif hasattr(self.base_layer, "B"):
            return self.base_layer.B
        # HQQ marlin
        elif hasattr(self.base_layer, "W_q"):
            return self.base_layer.W_q
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None
