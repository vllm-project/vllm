# SPDX-License-Identifier: Apache-2.0

import torch

# Required to register the custom ops
import vllm.lora.ops.xla_ops.pallas  # noqa # pylint: disable=unused-import


def bgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                add_inputs: bool = True):
    """
    Args:
        inputs (torch.Tensor): Input tensor of shape [num_tokens, hidden_size].
        
        lora_b_weights (torch.Tensor): LoRA weights of shape 
            [num_loras, lora_rank, hidden_size].
        
        output_tensor (torch.Tensor): output tensor of shape 
            [num_tokens, hidden_size * num_slices].
        
        lora_indices_tensor (torch.Tensor): Tensor of shape [num_tokens] 
            indicating which LoRA matrix to use for each token.
        add_inputs (bool): Whether or not to add the input tensor to the output 
            tensor.
    """

    outputs = torch.ops.xla.bgmv(inputs, lora_b_weights, lora_indices_tensor)
    n_tokens = outputs.size(0)

    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1

    outputs = torch.cat(
        (outputs,
         torch.zeros((n_tokens, output_tensor.shape[1] - outputs.shape[1]),
                     device=outputs.device)),
        dim=1)

    if add_inputs:
        return output_tensor + outputs[:limit, :]
    else:
        return outputs[:limit, :]


def bgmv_shrink(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                scaling: float = 1.0):
    """
    Args:
        inputs (torch.Tensor): Input tensor of shape [num_tokens, hidden_size].
        lora_b_weights (torch.Tensor): LoRA weights of shape 
            [num_loras, lora_rank, hidden_size].
        output_tensor (torch.Tensor): (Unused) output tensor (placeholder).
        lora_indices_tensor (torch.Tensor): Tensor of shape [num_tokens] 
            indicating which LoRA matrix to use for each token.
        scaling (float, optional): Scalar multiplier applied to the output.
    """

    return scaling * torch.ops.xla.bgmv(inputs, lora_b_weights,
                                        lora_indices_tensor)


def bgmv_expand_slice(inputs: torch.Tensor,
                      lora_b_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      slice_offset: int,
                      slice_size: int,
                      add_inputs: bool = True):
    """
    Args:
        inputs (torch.Tensor): Input tensor of shape [num_tokens, hidden_size].
        
        lora_b_weights (torch.Tensor): LoRA weights of shape 
            [num_loras, lora_rank, hidden_size].
        
        output_tensor (torch.Tensor): output tensor of shape 
            [num_tokens, hidden_size * num_slices].
        
        lora_indices_tensor (torch.Tensor): Tensor of shape [num_tokens] 
            indicating which LoRA matrix to use for each token.
        add_inputs (bool): Whether or not to add the input tensor to the output 
            tensor.
    """
    outputs = torch.ops.xla.bgmv(inputs, lora_b_weights, lora_indices_tensor)
    n_tokens = outputs.size(0)

    outputs = torch.cat((
        torch.zeros((n_tokens, slice_offset), device=outputs.device),
        outputs,
        torch.zeros(
            (n_tokens, output_tensor.shape[1] - (slice_offset + slice_size)),
            device=outputs.device),
    ),
                        dim=1)

    if add_inputs:
        return output_tensor + outputs
    else:
        return outputs
