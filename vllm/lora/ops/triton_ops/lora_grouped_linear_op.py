# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.lora.ops.triton_ops.kernel_utils import do_expand_kernel
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


# The token bound and head-major group strides can vary between batches.
@triton.jit(do_not_specialize=["num_tokens"])
def _lora_grouped_matmul_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    num_tokens,
    N: tl.constexpr,
    K: tl.constexpr,
    input_token_stride: tl.constexpr,
    input_group_stride,
    input_feature_stride: tl.constexpr,
    lora_slot_stride: tl.constexpr,
    lora_group_stride: tl.constexpr,
    lora_n_stride: tl.constexpr,
    lora_k_stride: tl.constexpr,
    output_token_stride: tl.constexpr,
    output_group_stride,
    output_feature_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
):
    lora_idx = tl.program_id(2)
    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id < 0:
        return

    group_idx = tl.program_id(1).to(tl.int64)
    feature_tiles = tl.cdiv(N, BLOCK_N)
    tile = tl.program_id(0)
    row_start = tile // feature_tiles * BLOCK_M
    pid_n = tile % feature_tiles
    lora_num_tokens = tl.load(num_tokens_per_lora + lora_idx)
    if row_start >= lora_num_tokens:
        return

    lora_token_start = tl.load(lora_token_start_loc + lora_idx)
    sorted_tile_ptr = token_indices_sorted_by_lora_ids + lora_token_start + row_start
    # Stable sorting keeps token indices ascending within each adapter.
    # This lets a kernel processing a batch prefix skip out-of-range tiles.
    first_token = tl.load(sorted_tile_ptr)
    if first_token >= num_tokens:
        return
    rows = tl.arange(0, BLOCK_M)
    token_indices = tl.load(
        sorted_tile_ptr + rows,
        mask=row_start + rows < lora_num_tokens,
        other=num_tokens,
    ).to(tl.int64)
    valid_tokens = token_indices < num_tokens

    # The helper loads a full tile and writes only its leading valid rows.
    do_expand_kernel(
        pid_n=pid_n,
        lora_index=lora_id.to(tl.int64),
        slice_id=0,
        input_ptr=input_ptr + group_idx * input_group_stride,
        lora_ptr=lora_ptr + group_idx * lora_group_stride,
        out_ptr=out_ptr + group_idx * output_group_stride,
        N=N,
        K=K,
        M_LEN=tl.sum(valid_tokens.to(tl.int32), axis=0),
        ram=tl.where(valid_tokens, token_indices, first_token),
        slice_start_loc=0,
        input_d0_stride=0,
        input_d1_stride=input_token_stride,
        input_d2_stride=input_feature_stride,
        ls_d0_ptr=lora_slot_stride,
        ls_d1_ptr=lora_n_stride,
        ls_d2_ptr=lora_k_stride,
        output_d0_stride=output_token_stride,
        output_d1_stride=output_feature_stride,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SAME_STRIDE=True,
        SLICE_NUM=1,
        EVEN_K=K % BLOCK_K == 0,
        CAST_TYPE=False,
        ADD_INPUTS=ADD_INPUTS,
        USE_GDC=False,
    )


def _launch_lora_grouped_matmul(
    inputs: torch.Tensor,
    lora_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    grid_lora_dim: int,
    *,
    add_inputs: bool,
) -> None:
    num_tokens, num_groups, input_size = inputs.shape
    output_size = output_tensor.shape[-1]
    block_m, block_n, block_k = 16, 32, 32
    grid = (
        triton.cdiv(num_tokens, block_m) * triton.cdiv(output_size, block_n),
        num_groups,
        grid_lora_dim,
    )
    _lora_grouped_matmul_kernel[grid](
        inputs,
        lora_weights,
        output_tensor,
        *metadata,
        num_tokens,
        output_size,
        input_size,
        *inputs.stride(),
        *lora_weights.stride(),
        *output_tensor.stride(),
        block_m,
        block_n,
        block_k,
        add_inputs,
        num_stages=1,
    )


def _lora_grouped_linear(
    output_tensor: torch.Tensor,
    inputs: torch.Tensor,
    lora_a_stacked: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
) -> None:
    if inputs.numel() == 0 or no_lora_flag_cpu.item():
        return
    grid_lora_dim = int(num_active_loras.item())
    if grid_lora_dim == 0:
        return

    num_tokens, num_groups, _ = inputs.shape
    rank = lora_a_stacked.shape[-2]
    metadata = (
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
    )
    buffer = inputs.new_empty((num_tokens, num_groups, rank))
    _launch_lora_grouped_matmul(
        inputs, lora_a_stacked, buffer, metadata, grid_lora_dim, add_inputs=False
    )
    _launch_lora_grouped_matmul(
        buffer, lora_b_stacked, output_tensor, metadata, grid_lora_dim, add_inputs=True
    )


def _lora_grouped_linear_fake(*args, **kwargs) -> None:
    pass


direct_register_custom_op(
    op_name="lora_grouped_linear",
    op_func=_lora_grouped_linear,
    mutates_args=["output_tensor"],
    fake_impl=_lora_grouped_linear_fake,
)
lora_grouped_linear = torch.ops.vllm.lora_grouped_linear
