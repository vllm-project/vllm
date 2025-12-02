# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F
from einops import repeat

from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    retrieve_parent_token: torch.Tensor = None,
    initial_state: torch.Tensor = None,
    scale: float | None = None,
):
    o = torch.zeros(*v.shape).to(v)
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )

    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    num_reqs = cu_seq_lens.shape[0] - 1
    for j in range(num_reqs):
        h = initial_state[ssm_state_indices[j][num_accepted_tokens[j] - 1]]
        T = cu_seq_lens[j + 1] - cu_seq_lens[j]
        for i in range(T):
            if retrieve_parent_token is not None and i != 0:
                h = initial_state[ssm_state_indices[j][retrieve_parent_token[j][i]]]
            b_q = q[:, :, cu_seq_lens[j] + i]
            b_k = k[:, :, cu_seq_lens[j] + i]
            b_v = v[:, :, cu_seq_lens[j] + i].clone()
            h = h.clone() * g[:, :, cu_seq_lens[j] + i].exp()[..., None, None]
            b_beta = beta[:, :, cu_seq_lens[j] + i]
            b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
            b_v = b_v * b_beta[..., None]
            h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
            o[:, cu_seq_lens[j] + i, :] = torch.einsum("bhd,bhdm->bhm", b_q, h)
            initial_state[ssm_state_indices[j][i]] = h
    return o, initial_state


@pytest.mark.parametrize("has_eagle_tree_state", [False, True])
def test_fused_recurrent(has_eagle_tree_state: bool):
    torch.manual_seed(42)
    H = 4
    K = 128
    V = 128
    HV = 8

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
    # num_reqs = 4, max_spec_len = 5
    cu_seq_lens = torch.tensor(
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
        device="cuda",
        dtype=torch.int32,
    )
    num_accepted_tokens = torch.tensor(
        [2, 1, 1, 2],
        device="cuda",
        dtype=torch.int32,
    )
    # for variable-length inputs,
    # the batch size `B` is expected to be 1 and `cu_seqlens` is required
    B = 1
    T = cu_seq_lens.max()
    chunk_size = 64
    q = torch.randn(B, T, H, K, dtype=torch.float16)
    k = torch.randn(B, T, H, K, dtype=torch.float16)
    v = torch.randn(B, T, HV, V, dtype=torch.float16)
    beta = torch.randn(B, T, HV, dtype=torch.float16).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32))
    h0 = torch.randn(chunk_size, HV, K, V, dtype=torch.float32)
    q, k, v, beta, g, h0 = map(
        lambda x: x.to("cuda").requires_grad_(), (q, k, v, beta, g, h0)
    )

    ref, ref_ht = recurrent_gated_delta_rule_ref(
        q=F.normalize(
            repeat(q.clone(), "b t h d -> b t (h g) d", g=HV // H), p=2, dim=-1
        ),
        k=F.normalize(
            repeat(k.clone(), "b t h d -> b t (h g) d", g=HV // H), p=2, dim=-1
        ),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        cu_seq_lens=cu_seq_lens,
        ssm_state_indices=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        retrieve_parent_token=retrieve_parent_token if has_eagle_tree_state else None,
        initial_state=h0.clone(),
    )
    tri, tri_ht = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seq_lens,
        ssm_state_indices=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        retrieve_parent_token=retrieve_parent_token if has_eagle_tree_state else None,
        inplace_final_state=True,
    )

    assert torch.allclose(ref, tri, atol=1e-3, rtol=1e-4)
    assert torch.allclose(ref_ht, tri_ht, atol=1e-3, rtol=1e-4)
