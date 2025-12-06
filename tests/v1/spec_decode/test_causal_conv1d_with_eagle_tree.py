# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.platforms import current_platform


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    retrieve_parent_token: torch.Tensor | None,
):
    """
    x: (dim, seqlen)
    conv_state: (chunk_size, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    out: (dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    dim, width = weight.shape
    state_len = width - 1
    output = torch.zeros_like(x)
    conv_state_ori = conv_state.clone()
    num_reqs = query_start_loc.shape[0] - 1
    for j in range(num_reqs):
        # update conv_state
        con_seq_len = query_start_loc[j + 1] - query_start_loc[j]
        x_new = torch.cat(
            [
                conv_state[
                    conv_state_indices[j], :, : num_accepted_tokens[j] + state_len - 1
                ],
                x[:, query_start_loc[j] : query_start_loc[j + 1]],
            ],
            dim=-1,
        ).to(weight.dtype)
        update_state_len = state_len + con_seq_len - 1
        conv_state[conv_state_indices[j], :, :update_state_len].copy_(
            x_new[:, num_accepted_tokens[j] :]
        )
        # update output
        for i in range(con_seq_len):
            con_x = x[:, query_start_loc[j] + i : query_start_loc[j] + i + 1]
            con_index = i
            if retrieve_parent_token is not None:
                while retrieve_parent_token[j, con_index] != -1:
                    con_index = retrieve_parent_token[j, con_index]
                    con_x = torch.cat(
                        [
                            x[
                                :,
                                query_start_loc[j] + con_index : query_start_loc[j]
                                + con_index
                                + 1,
                            ],
                            con_x,
                        ],
                        dim=-1,
                    )
            else:
                con_x = x[:, query_start_loc[j] : query_start_loc[j] + i + 1]
            con_x = torch.cat(
                [
                    conv_state_ori[
                        conv_state_indices[j],
                        :,
                        : num_accepted_tokens[j] + state_len - 1,
                    ],
                    con_x,
                ],
                dim=-1,
            ).to(weight.dtype)

            con_x = con_x[:, -width:]

            if unsqueeze:
                con_x = con_x.unsqueeze(0)
            out = F.conv1d(con_x, weight.unsqueeze(1), bias, padding=0, groups=dim)[
                :, :, -1:
            ]
            if unsqueeze:
                out = out.squeeze(0)
            output[:, query_start_loc[j] + i : query_start_loc[j] + i + 1] = out

    return (output if activation is None else F.silu(output)).to(dtype=dtype_in)


@pytest.mark.parametrize("itype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("is_eagle_tree", [False, True])
def test_causal_conv1d_update(
    has_bias: bool, silu_activation: bool, itype: torch.dtype, is_eagle_tree: bool
):
    device = "cuda"
    # set seed
    current_platform.seed_everything(0)
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (1e-2, 5e-2)
    # num_reqs = 4, max_spec_len = 5
    batch_size = 4
    num_speculative_tokens = 5
    chunk_size = 64
    dim = 2048
    width = 4
    # shape is [batch_size, max_spec_len + 1]
    retrieve_parent_token = torch.tensor(
        [
            # Tree1:
            #    0
            #   / \
            #  1   2
            # /
            # 3
            [-1, 0, 0, 1, -1, -1],
            # Tree2:
            #    0
            #   /
            #  1
            # /
            # 2
            [-1, 0, 1, -1, -1, -1],
            # Tree3:
            #    0
            #   / \
            #  1   2
            # / \
            # 3  4
            [-1, 0, 0, 1, 1, -1],
            # Tree4:
            #    0
            #   / \
            #  1   2
            # / \  /
            # 3  4 5
            [-1, 0, 0, 1, 1, 2],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    spec_query_start_loc = torch.tensor(
        [0, 4, 7, 12, 18],
        device="cuda",
        dtype=torch.int32,
    )
    spec_state_indices_tensor = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
        ],
        device=device,
        dtype=torch.int32,
    )
    num_accepted_tokens = torch.tensor(
        [1, 2, 1, 2],
        device="cuda",
        dtype=torch.int32,
    )
    seqlen = spec_query_start_loc[-1].item()
    x = torch.rand(seqlen, dim, device=device, dtype=itype)
    x_ref = x.clone().transpose(0, 1)

    conv_state = torch.randn(
        chunk_size, dim, width + num_speculative_tokens - 1, device=device, dtype=itype
    )
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    conv_state_ref = conv_state.detach().clone()
    activation = None if not silu_activation else "silu"

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=spec_state_indices_tensor[:, 0][:batch_size],
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=spec_query_start_loc,
        max_query_len=spec_state_indices_tensor.size(-1),
        retrieve_parent_token=retrieve_parent_token if is_eagle_tree else None,
        validate_data=False,
    )
    out_ref = causal_conv1d_update_ref(
        x_ref,
        conv_state_ref,
        weight,
        bias,
        activation=activation,
        conv_state_indices=spec_state_indices_tensor[:, 0][:batch_size],
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=spec_query_start_loc,
        retrieve_parent_token=retrieve_parent_token if is_eagle_tree else None,
    ).transpose(0, 1)
    assert torch.equal(conv_state, conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
