# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.lora.ops.triton_ops.kernel_utils import do_expand_kernel
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _grouped_lora_matmul(
    x_ptr,
    weight_ptr,
    out_ptr,
    sorted_rows_ptr,
    group_sizes_ptr,
    group_starts_ptr,
    lora_ids_ptr,
    T: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    x_token_stride: tl.constexpr,
    x_group_stride: tl.constexpr,
    x_feature_stride: tl.constexpr,
    weight_slot_stride: tl.constexpr,
    weight_group_stride: tl.constexpr,
    weight_n_stride: tl.constexpr,
    weight_k_stride: tl.constexpr,
    out_token_stride: tl.constexpr,
    out_group_stride: tl.constexpr,
    out_feature_stride: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
):
    lora_group = tl.program_id(2)
    slot = tl.load(lora_ids_ptr + lora_group)
    if slot < 0:
        return

    group = tl.program_id(1).to(tl.int64)
    feature_tiles = tl.cdiv(N, BLOCK_N)
    tile = tl.program_id(0)
    row_start = tile // feature_tiles * BLOCK_M
    pid_n = tile % feature_tiles
    group_size = tl.load(group_sizes_ptr + lora_group)
    if row_start >= group_size:
        return

    group_start = tl.load(group_starts_ptr + lora_group)
    sorted_tile_ptr = sorted_rows_ptr + group_start + row_start
    # Stable sorting leaves original token row IDs in ascending order per group.
    first_token = tl.load(sorted_tile_ptr)
    if first_token >= T:
        return
    rows = tl.arange(0, BLOCK_M)
    tokens = tl.load(
        sorted_tile_ptr + rows, mask=row_start + rows < group_size, other=T
    ).to(tl.int64)
    valid_tokens = tokens < T

    # The helper loads a full tile and writes only its leading valid rows.
    do_expand_kernel(
        pid_n=pid_n,
        lora_index=slot.to(tl.int64),
        slice_id=0,
        input_ptr=x_ptr + group * x_group_stride,
        lora_ptr=weight_ptr + group * weight_group_stride,
        out_ptr=out_ptr + group * out_group_stride,
        N=N,
        K=K,
        M_LEN=tl.sum(valid_tokens.to(tl.int32), axis=0),
        ram=tl.where(valid_tokens, tokens, first_token),
        slice_start_loc=0,
        input_d0_stride=0,
        input_d1_stride=x_token_stride,
        input_d2_stride=x_feature_stride,
        ls_d0_ptr=weight_slot_stride,
        ls_d1_ptr=weight_n_stride,
        ls_d2_ptr=weight_k_stride,
        output_d0_stride=out_token_stride,
        output_d1_stride=out_feature_stride,
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


def _launch_grouped_lora_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    lora_group_count: int,
    *,
    add_inputs: bool,
) -> None:
    tokens, groups, input_size = x.shape
    output_size = out.shape[-1]
    block_m, block_n, block_k = 16, 32, 32
    grid = (
        triton.cdiv(tokens, block_m) * triton.cdiv(output_size, block_n),
        groups,
        lora_group_count,
    )
    _grouped_lora_matmul[grid](
        x,
        weight,
        out,
        *metadata,
        tokens,
        output_size,
        input_size,
        *x.stride(),
        *weight.stride(),
        *out.stride(),
        block_m,
        block_n,
        block_k,
        add_inputs,
        num_stages=1,
    )


def _lora_grouped_linear(
    y: torch.Tensor,
    x: torch.Tensor,
    shrink_weights: torch.Tensor,
    expand_weights: torch.Tensor,
    sorted_rows: torch.Tensor,
    group_sizes: torch.Tensor,
    group_starts: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag: torch.Tensor,
    num_lora_groups: torch.Tensor,
) -> None:
    if x.numel() == 0 or no_lora_flag.item():
        return
    lora_group_count = int(num_lora_groups.item())
    if lora_group_count == 0:
        return

    tokens, groups, _ = x.shape
    rank = shrink_weights.shape[-2]
    metadata = (sorted_rows, group_sizes, group_starts, lora_ids)
    tmp = x.new_empty((tokens, groups, rank))
    _launch_grouped_lora_matmul(
        x, shrink_weights, tmp, metadata, lora_group_count, add_inputs=False
    )
    _launch_grouped_lora_matmul(
        tmp, expand_weights, y, metadata, lora_group_count, add_inputs=True
    )


def _lora_grouped_linear_fake(*args, **kwargs) -> None:
    pass


direct_register_custom_op(
    op_name="lora_grouped_linear",
    op_func=_lora_grouped_linear,
    mutates_args=["y"],
    fake_impl=_lora_grouped_linear_fake,
)
lora_grouped_linear = torch.ops.vllm.lora_grouped_linear
