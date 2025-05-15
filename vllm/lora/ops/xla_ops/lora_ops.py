# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


def bgmv(inputs: torch.Tensor, loras: torch.Tensor, idxs: torch.IntTensor):
    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    N = loras.shape[0]

    selected_loras = torch.einsum(
        "tn,nld->tld",
        torch.nn.functional.one_hot(idxs.to(torch.long), N).to(inputs.dtype),
        loras,
    )

    return torch.einsum("td,tld->tl", inputs, selected_loras)


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
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

    outputs = bgmv(inputs, lora_b_weights, lora_indices_tensor)

    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1

    if output_tensor.shape[1] > outputs.shape[1]:
        outputs = F.pad(outputs,
                        (0, output_tensor.shape[1] - outputs.shape[1], 0, 0))

    if add_inputs:
        return output_tensor + outputs[:limit, :output_tensor.shape[1]]
    else:
        return outputs[:limit, :output_tensor.shape[1]]


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
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

    return scaling * bgmv(inputs, lora_b_weights, lora_indices_tensor)


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
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
    outputs = bgmv(inputs, lora_b_weights, lora_indices_tensor)

    outputs = F.pad(
        outputs,
        (
            slice_offset,
            output_tensor.shape[1] - (slice_offset + slice_size),
            0,
            0,
        ),
    )

    if add_inputs:
        return output_tensor + outputs
    else:
        return outputs
