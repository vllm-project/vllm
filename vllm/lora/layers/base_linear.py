# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, cast

import torch
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed.utils import divide
# yapf: disable
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, ReplicatedLinear,
                                               RowParallelLinear)
from vllm.platforms import current_platform

from .base import BaseLayerWithLoRA
from .utils import _get_lora_device


logger = init_logger(__name__)

IS_LOGGED = False


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
                # requires_grad=True,
            ) for _ in range(self.n_slices))
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_b_out_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
                # requires_grad=True,
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
                    # requires_grad=True,
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
        is_trainable: bool = False,
        trainable_slices: Optional[list[int]] = None,
    ):
        # Except for QKVParallelLinearWithLoRA and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert (len(self.lora_a_stacked) == len(self.lora_b_stacked) ==
                self.n_slices == 1)

        if not is_trainable:
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
        if is_trainable:
            self.lora_a_stacked[0].requires_grad_(True)
            self.lora_b_stacked[0].requires_grad_(True)
            if lora_bias is not None:
                self.lora_bias_stacked[0].requires_grad_(True)

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

        if torch.is_grad_enabled():
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

    # TODO(girfan): Add tests to verify if this is functionally correct and matches punica.
    def _training_apply(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        # save q_old, k_old, v_old to separate csv files
        global IS_LOGGED

        # if not IS_LOGGED:
        #     import pandas as pd

        #     q_old, k_old, v_old = output.split([2048, 512, 512], dim=-1)
        #     df = pd.DataFrame({
        #         "q": q_old.flatten().tolist(),
        #     })
        #     df.to_csv(f"vllm_q_old.csv", index=False)
            
        #     df = pd.DataFrame({
        #         "k": k_old.flatten().tolist(),
        #     })
        #     df.to_csv(f"vllm_k_old.csv", index=False)

        #     df = pd.DataFrame({
        #         "v": v_old.flatten().tolist(),
        #     })
        #     df.to_csv(f"vllm_v_old.csv", index=False)

        lora_indices = self.punica_wrapper.token_lora_indices

        # Group tokens by LoRA index
        unique_lora_ids = set(lora_indices.tolist())

        scaling = self.lora_config.lora_alpha / self.lora_config.max_lora_rank
        # TODO(girfan): We should use the ACTUAL rank of this LoRA, not max rank.
        assert scaling == 2.0, f"Scaling factor is {scaling}, expected 2.0 (temp: matching PEFT example)"

        for lora_idx in unique_lora_ids:
            if lora_idx == -1:
                continue

            # Get all tokens using this LoRA
            token_mask = (lora_indices == lora_idx)
            token_indices = torch.where(token_mask)[0]

            if len(token_indices) == 0:
                continue

            # Process all tokens for this LoRA as a batch
            x_batch = x[token_indices]  # [num_tokens, hidden_dim]

            output_offset = 0
            for slice_idx in range(len(self.lora_a_stacked)):
                lora_a = self.lora_a_stacked[slice_idx][lora_idx, 0, :, :]  # [rank, input_size]
                lora_b = self.lora_b_stacked[slice_idx][lora_idx, 0, :, :]  # [output_size, rank]

                # Batch operation: [num_tokens, hidden_dim] @ [hidden_dim, rank] @ [rank, output_size]
                # This matches PEFT's computation: result = result + lora_B(lora_A(x)) * scaling
                lora_hidden = x_batch @ lora_a.T  # [num_tokens, rank]
                lora_output = lora_hidden @ lora_b.T  # [num_tokens, output_size]
                lora_output_scaled = lora_output * scaling  # Apply scaling to match PEFT

                slice_size = self.output_slices[slice_idx]

                # print(f"Updating output indices: {token_indices}")
                # print(f"Updating output offset: {output_offset} to {output_offset + slice_size}")
                # print(f"Updating slice size: {slice_size}")

                output[token_indices, output_offset:output_offset + slice_size] += lora_output_scaled
                output_offset += slice_size

                if self.lora_bias_stacked is not None and self.lora_bias_stacked[slice_idx] is not None:
                    bias = self.lora_bias_stacked[slice_idx][lora_idx, 0, :]
                    output[token_indices, output_offset-slice_size:output_offset] += bias

                # if not IS_LOGGED:
                #     import pandas as pd

                #     q_old, k_old, v_old = output.split([2048, 512, 512], dim=-1)
                #     df = pd.DataFrame({
                #         "q": q_old.flatten().tolist(),
                #     })
                #     df.to_csv(f"vllm_q_one_accumulation.csv", index=False)

                #     df = pd.DataFrame({
                #         "lora_a": lora_a.flatten().tolist(),
                #     })
                #     df.to_csv(f"vllm_lora_a_{slice_idx}_{lora_idx}.csv", index=False)

                #     df = pd.DataFrame({
                #         "lora_b": lora_b.flatten().tolist(),
                #     })
                #     df.to_csv(f"vllm_lora_b_{slice_idx}_{lora_idx}.csv", index=False)

                #     df = pd.DataFrame({
                #         "x": x_batch.flatten().tolist(),
                #     })
                #     df.to_csv(f"vllm_x_batch_{slice_idx}_{lora_idx}.csv", index=False)

                #     df = pd.DataFrame({
                #         "q": lora_output_scaled.flatten().tolist(),
                #     })
                #     df.to_csv(f"vllm_lora_result_{slice_idx}_{lora_idx}.csv", index=False)
                #     ss

                #     IS_LOGGED = True

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
