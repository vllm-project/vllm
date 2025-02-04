# SPDX-License-Identifier: Apache-2.0

import torch

from ..torch_ops import bgmv_expand, bgmv_expand_slice, bgmv_shrink


def sgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                b_seq_start_loc: torch.Tensor,
                seq_len_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                batches: int,
                max_seq_length: int,
                token_nums: int,
                add_inputs: bool = False):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor,
                                               inputs.size(0))

    bgmv_expand(inputs, lora_b_weights, output_tensor, exploded_indices,
                add_inputs)


def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor,
                                               inputs.size(0))

    print("SGMV", lora_indices_tensor, lora_a_weights)
    bgmv_shrink(inputs, lora_a_weights, output_tensor, exploded_indices,
                scaling)


def sgmv_expand_slice(inputs: torch.Tensor,
                      lora_b_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      b_seq_start_loc: torch.Tensor,
                      seq_len_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      batches: int,
                      max_seq_length: int,
                      token_nums: int,
                      slice_offset: int,
                      slice_size: int,
                      add_inputs: bool = False):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor,
                                               inputs.size(0))

    bgmv_expand_slice(inputs, lora_b_weights, output_tensor, exploded_indices,
                      slice_offset, slice_size, add_inputs)
