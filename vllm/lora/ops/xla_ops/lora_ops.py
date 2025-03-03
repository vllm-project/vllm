# SPDX-License-Identifier: Apache-2.0

import torch
import vllm.lora.ops.xla_ops.pallas # Required to register the custom ops

def bgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                add_inputs: bool = True):
    inputs = inputs.to(dtype=output_tensor.dtype)

    outputs = torch.ops.xla.bgmv(inputs, lora_b_weights, lora_indices_tensor)
    batch_size = outputs.size(0)

    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1

    outputs = torch.cat(
        (outputs,
         torch.zeros((batch_size, output_tensor.shape[1] - outputs.shape[1]),
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


    inputs = inputs.to(dtype=output_tensor.dtype)

    return scaling * torch.ops.xla.bgmv(inputs, lora_b_weights, lora_indices_tensor)

def bgmv_expand_slice(inputs: torch.Tensor,
                      lora_b_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      slice_offset: int,
                      slice_size: int,
                      add_inputs: bool = True):

    inputs = inputs.to(dtype=output_tensor.dtype)

    outputs = torch.ops.xla.bgmv(inputs, lora_b_weights, lora_indices_tensor)
    batch_size = outputs.size(0)

    outputs = torch.cat((
        torch.zeros((batch_size, slice_offset), device=outputs.device),
        outputs,
        torch.zeros(
            (batch_size, output_tensor.shape[1] - (slice_offset + slice_size)),
            device=outputs.device),
    ),
                        dim=1)

    if add_inputs:
        return output_tensor + outputs
    else:
        return outputs
