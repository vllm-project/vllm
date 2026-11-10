# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def _bgmv_shrink_impl(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float,
) -> None:
    torch.ops._xpu_C.bgmv_shrink(
        output_tensor, inputs, lora_a_weights, lora_indices_tensor, scaling
    )


def _bgmv_expand_impl(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool,
) -> None:
    weight_out_dim = lora_b_weights.size(-2)
    output_dim = output_tensor.size(1)

    if weight_out_dim == output_dim:
        torch.ops._xpu_C.bgmv_expand(
            output_tensor,
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            add_inputs,
        )
    elif weight_out_dim < output_dim:
        # LoRA weight output dim can be smaller than the output tensor
        # (e.g. vocab_size vs padded logits). Use expand_slice to write
        # only the matching portion, mirroring torch_ops common_len logic.
        torch.ops._xpu_C.bgmv_expand_slice(
            output_tensor,
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            0,
            weight_out_dim,
            add_inputs,
        )
    else:
        # Weight output dim larger than output tensor: truncate weights.
        lora_b_weights = lora_b_weights[..., :output_dim, :].contiguous()
        torch.ops._xpu_C.bgmv_expand_slice(
            output_tensor,
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            0,
            output_dim,
            add_inputs,
        )


def _bgmv_expand_slice_impl(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool,
) -> None:
    assert slice_size == lora_b_weights.size(-2)
    assert slice_offset + slice_size <= output_tensor.size(1)
    torch.ops._xpu_C.bgmv_expand_slice(
        output_tensor,
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        slice_offset,
        slice_size,
        add_inputs,
    )


def _lora_linear_impl(
    output_tensor: torch.Tensor,
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    lora_b_weights: list[torch.Tensor],
    lora_indices_tensor: torch.Tensor,
    single_lora_slot: torch.Tensor,
    output_slices: list[int],
    scaling: float,
    add_inputs: bool,
    gemm_min_tokens: int,
) -> None:
    num_tokens = inputs.size(0)
    # `single_lora_slot` is a host tensor, so reading it does not synchronize.
    # It is only consulted above the largest captured graph size, so captured
    # graphs never bake in a slot.
    slot = int(single_lora_slot[0]) if num_tokens >= gemm_min_tokens else -1
    if slot >= 0:
        # Every token uses the same adapter: two dense GEMMs per slice are much
        # faster than per-token BGMV for prefill-sized batches.
        offset = 0
        for lora_a, lora_b, slice_size in zip(
            lora_a_weights, lora_b_weights, output_slices
        ):
            hidden = torch.mm(inputs, lora_a[slot, 0].t())
            if scaling != 1.0:
                hidden.mul_(scaling)
            output_tensor[:, offset : offset + slice_size].addmm_(
                hidden, lora_b[slot, 0].t(), beta=1.0 if add_inputs else 0.0
            )
            offset += slice_size
        return

    rank = lora_b_weights[0].size(-1)
    buffer = torch.zeros(
        (len(lora_a_weights), num_tokens, rank),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    for i, lora_a in enumerate(lora_a_weights):
        _bgmv_shrink_impl(inputs, lora_a, buffer[i], lora_indices_tensor, scaling)
    offset = 0
    for i, (lora_b, slice_size) in enumerate(zip(lora_b_weights, output_slices)):
        _bgmv_expand_slice_impl(
            buffer[i],
            lora_b,
            output_tensor,
            lora_indices_tensor,
            offset,
            slice_size,
            add_inputs,
        )
        offset += slice_size


direct_register_custom_op(
    op_name="xpu_bgmv_shrink",
    op_func=_bgmv_shrink_impl,
    mutates_args=["output_tensor"],
)

direct_register_custom_op(
    op_name="xpu_bgmv_expand",
    op_func=_bgmv_expand_impl,
    mutates_args=["output_tensor"],
)

direct_register_custom_op(
    op_name="xpu_bgmv_expand_slice",
    op_func=_bgmv_expand_slice_impl,
    mutates_args=["output_tensor"],
)

direct_register_custom_op(
    op_name="xpu_lora_linear",
    op_func=_lora_linear_impl,
    mutates_args=["output_tensor"],
)


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
) -> None:
    torch.ops.vllm.xpu_bgmv_shrink(
        inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
) -> None:
    torch.ops.vllm.xpu_bgmv_expand(
        inputs, lora_b_weights, output_tensor, lora_indices_tensor, add_inputs
    )


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
) -> None:
    torch.ops.vllm.xpu_bgmv_expand_slice(
        inputs,
        lora_b_weights,
        output_tensor,
        lora_indices_tensor,
        slice_offset,
        slice_size,
        add_inputs,
    )


def lora_linear(
    output_tensor: torch.Tensor,
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    lora_b_weights: list[torch.Tensor],
    lora_indices_tensor: torch.Tensor,
    single_lora_slot: torch.Tensor,
    output_slices: list[int],
    scaling: float,
    add_inputs: bool,
    gemm_min_tokens: int,
) -> None:
    """Apply shrink and expand for all slices of a linear layer.

    If `single_lora_slot` (a one-element host tensor) holds a slot index and the
    batch has at least `gemm_min_tokens` tokens, every token is assumed to use
    that adapter and dense GEMMs are used; otherwise BGMV is used.
    """
    torch.ops.vllm.xpu_lora_linear(
        output_tensor,
        inputs,
        lora_a_weights,
        lora_b_weights,
        lora_indices_tensor,
        single_lora_slot,
        output_slices,
        scaling,
        add_inputs,
        gemm_min_tokens,
    )
