# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import helion
import helion.language as hl
import torch


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    allow_warp_specialize=True,
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
)
def static_scaled_fp8_quant_helion(
    input: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    group_m: int,
    group_n: int,
):
    assert scale.dim() == 2

    num_tokens, hidden_size = input.shape
    num_groups_m, num_groups_n = scale.shape
    hl.specialize(hidden_size)
    hl.specialize(group_n)
    hl.specialize(num_groups_n)

    for tile_gm, tile_gn, tile_m, tile_n in hl.tile(
        [num_groups_m, num_groups_n, group_m, group_n]
    ):
        gm_idx = tile_gm.index
        gn_idx = tile_gn.index

        # shape: [tile_gm, tile_gn]
        scale_blk = scale[gm_idx[:, None], gn_idx[None, :]]
        inv_scale_blk = (1.0 / scale_blk).to(dtype=torch.float32)

        # offset inside group
        m_offset = tile_m.index
        n_offset = tile_n.index

        # Global indices
        # m_idx: [tile_gm, tile_m]
        # n_idx: [tile_gn, tile_n]
        m_idx = gm_idx[:, None] * group_m + m_offset[None, :]
        n_idx = gn_idx[:, None] * group_n + n_offset[None, :]

        m_blk = m_idx[:, None, :, None]
        n_blk = n_idx[None, :, None, :]

        # input tile shape:  [tile_gm, tile_gn, tile_m, tile_n]
        x_blk = input[m_blk, n_blk].to(torch.float32)

        # scale tile shape:  [tile_gm, tile_gn, 1, 1]
        y_blk = x_blk * inv_scale_blk[:, :, None, None]

        output[m_blk, n_blk] = y_blk.to(output.dtype)


def static_scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    group_shape: tuple[int, int] | None = None,
):
    assert input.stride(-1) == 1
    assert output.stride(-1) == 1

    num_tokens, hidden_size = input.shape

    if scale.dim() == 0 or scale.numel() == 1:
        # per tensor
        group_m = num_tokens
        group_n = hidden_size
        scale_2d = scale.reshape(1, 1)
    elif scale.dim() == 1:
        assert group_shape is not None
        group_shape_m, group_shape_n = group_shape
        assert group_shape_m == -1 or group_shape_n == -1
        group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
        group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)
        scale_2d = scale[None, :] if group_shape_m == -1 else scale[:, None]
        inferred_group_m = num_tokens // scale_2d.size(0)
        inferred_group_n = hidden_size // scale_2d.size(1)
        assert group_m == inferred_group_m and group_n == inferred_group_n
    elif scale.dim() == 2:
        scale_2d = scale
        scale_size_0, scale_size_1 = scale.shape
        assert num_tokens % scale_size_0 == 0
        assert hidden_size % scale_size_1 == 0
        inferred_group_m = num_tokens // scale_size_0
        inferred_group_n = hidden_size // scale_size_1

        if group_shape is not None:
            group_shape_m, group_shape_n = group_shape
            group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
            group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)
            assert group_m == inferred_group_m and group_n == inferred_group_n
        else:
            group_m, group_n = inferred_group_m, inferred_group_n

    static_scaled_fp8_quant_helion(input, scale_2d, output, group_m, group_n)
