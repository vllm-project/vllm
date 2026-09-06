# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# ---------------------------------------------------------------------------
# Multi-slice ops: process all LoRA slices in a single kernel launch.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _lora_shrink(
    inputs: torch.Tensor,  #  shape [num_tokens, hidden_size]
    lora_a_weights: list[torch.Tensor],  # shape [num_loras, lora_rank, hidden_size]
    output_tensor: torch.Tensor,  # shape [num_slices, num_tokens, lora_rank]
    token_lora_mapping: torch.Tensor,  # shape [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens]
    num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
    lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
    lora_ids: torch.Tensor,  # shape [max-loras + 1]
    no_lora_flag_cpu: torch.Tensor,  # shape [1]
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    scaling: float,
) -> None:
    """Shrink op with triton-compatible signature.
    Extra metadata args are accepted but unused on XPU."""
    if no_lora_flag_cpu is not None:
        assert no_lora_flag_cpu.numel() == 1
        if no_lora_flag_cpu.item():
            return

    if isinstance(lora_a_weights, torch.Tensor):
        lora_a_weights = [lora_a_weights]
        if output_tensor.dim() == 2:
            output_tensor = output_tensor.unsqueeze(0)

    assert inputs.dtype in [torch.float16, torch.bfloat16]
    assert inputs.dtype == lora_a_weights[0].dtype
    for weight in lora_a_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]
    assert inputs.size(1) == lora_a_weights[0].size(-1)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    M = inputs.size(0)
    assert token_lora_mapping.size(0) >= M

    torch.ops._xpu_C.lora_shrink(
        inputs, list(lora_a_weights), output_tensor, token_lora_mapping, scaling
    )


def _lora_shrink_fake(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    scaling: float,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="xpu_lora_shrink",
        op_func=_lora_shrink,
        mutates_args=["output_tensor"],
        fake_impl=_lora_shrink_fake,
    )
    lora_shrink = torch.ops.vllm.xpu_lora_shrink

except AttributeError:
    lora_shrink = _lora_shrink


@torch.inference_mode()
def _lora_expand(
    inputs: torch.Tensor,  # shape [num_slices, num_tokens, lora_rank]
    lora_b_weights: list[torch.Tensor],  # shape [num_lora, hidden_size, lora_rank]
    output_tensor: torch.Tensor,  # shape [num_tokens, hidden_size * num_slices]
    token_lora_mapping: torch.Tensor,  # shape [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens]
    num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
    lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
    lora_ids: torch.Tensor,  # shape [max-loras + 1]
    no_lora_flag_cpu: torch.Tensor,  # shape [1]
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    """Expand op with triton-compatible signature.
    Extra metadata args are accepted but unused on XPU."""
    if no_lora_flag_cpu is not None:
        assert no_lora_flag_cpu.numel() == 1
        if no_lora_flag_cpu.item():
            return

    if isinstance(lora_b_weights, torch.Tensor):
        lora_b_weights = [lora_b_weights]
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    for weight in lora_b_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]
    assert inputs.size(0) == len(lora_b_weights)
    assert output_tensor.is_contiguous()

    M = inputs.size(1)
    assert token_lora_mapping.size(0) >= M

    torch.ops._xpu_C.lora_expand(
        inputs,
        lora_b_weights,
        output_tensor,
        token_lora_mapping,
        offset_start,
        add_inputs,
    )


def _lora_expand_fake(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="xpu_lora_expand",
        op_func=_lora_expand,
        mutates_args=["output_tensor"],
        fake_impl=_lora_expand_fake,
    )
    lora_expand = torch.ops.vllm.xpu_lora_expand

except AttributeError:
    lora_expand = _lora_expand
