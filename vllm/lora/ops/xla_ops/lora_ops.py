# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
import torch_xla.core.xla_builder as xb
from torch.library import impl
from torch_xla.experimental.custom_kernel import XLA_LIB, jax_import_guard


@jax.jit
def bgmv_jax(inputs, loras, idxs):
    return jnp.einsum(
        "td,tX,Xld->tl",
        inputs,
        jax.nn.one_hot(idxs, loras.shape[0], dtype=inputs.dtype),
        loras,
    )


XLA_LIB.define("bgmv(Tensor inputs, Tensor loras, Tensor idxs) -> Tensor")


@impl(XLA_LIB, "bgmv", "XLA")
def bgmv_xla(inputs: torch.Tensor, loras: torch.Tensor, idxs: torch.IntTensor):
    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    jax_import_guard()
    return xb.call_jax(bgmv_jax, (inputs, loras, idxs))


@impl(XLA_LIB, "bgmv", "CompositeExplicitAutograd")
def bgmv_non_xla(inputs: torch.Tensor, loras: torch.Tensor, idxs: torch.IntTensor):
    T, _ = inputs.shape
    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)
    _, L, _ = loras.shape

    return torch.empty((T, L), device=inputs.device)


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

    outputs = torch.ops.xla.bgmv(inputs, lora_b_weights, lora_indices_tensor)

    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1

    if output_tensor.shape[1] > outputs.shape[1]:
        outputs = F.pad(outputs, (0, output_tensor.shape[1] - outputs.shape[1], 0, 0))

    if add_inputs:
        return output_tensor + outputs[:limit, : output_tensor.shape[1]]
    else:
        return outputs[:limit, : output_tensor.shape[1]]


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
        lora_indices_tensor (torch.Tensor): Tensor of shape [num_tokens]
            indicating which LoRA matrix to use for each token.
        scaling (float, optional): Scalar multiplier applied to the output.
    """

    return scaling * torch.ops.xla.bgmv(inputs, lora_b_weights, lora_indices_tensor)


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
    outputs = torch.ops.xla.bgmv(inputs, lora_b_weights, lora_indices_tensor)

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
