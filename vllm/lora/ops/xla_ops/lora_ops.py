# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

# Required to register the custom ops
import vllm.lora.ops.xla_ops.pallas  # noqa # pylint: disable=unused-import


def bgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                add_inputs: bool = True):

    outputs = torch.ops.xla.bgmv_expand(inputs, lora_b_weights.transpose(2, 3),
                                        lora_indices_tensor)

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


def bgmv_shrink(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                scaling: float = 1.0):

    return scaling * torch.ops.xla.bgmv_shrink(inputs, lora_b_weights,
                                               lora_indices_tensor)


def bgmv_expand_slice(inputs: torch.Tensor,
                      lora_b_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      slice_offset: int,
                      slice_size: int,
                      add_inputs: bool = True):
    outputs = torch.ops.xla.bgmv_expand(inputs, lora_b_weights.transpose(2, 3),
                                        lora_indices_tensor)

    outputs = F.pad(outputs, (slice_offset, output_tensor.shape[1] -
                              (slice_offset + slice_size), 0, 0))

    if add_inputs:
        return output_tensor + outputs
    else:
        return outputs
