# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import torch

from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda


@pytest.fixture
def dummy_inputs():
    B, T, H, K, V = 1, 4, 2, 16, 16
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    g = torch.randn(B, T, H, K)
    beta = torch.randn(B, T, H)
    return q, k, v, g, beta


def test_fused_recurrent_kda_default_num_accepted_tokens_is_none(dummy_inputs):
    q, k, v, g, beta = dummy_inputs
    fake_o = torch.empty_like(k)
    fake_final_state = torch.empty(1, 2, 16, 16)

    with patch(
        "vllm.third_party.flash_linear_attention.ops.kda.fused_recurrent_kda_fwd",
        return_value=(fake_o, fake_final_state),
    ) as mock_fwd:
        o, final_state = fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta)
        assert mock_fwd.called
        assert mock_fwd.call_args.kwargs["num_accepted_tokens"] is None
        assert o is fake_o
        assert final_state is fake_final_state


def test_fused_recurrent_kda_passes_num_accepted_tokens_keyword(dummy_inputs):
    q, k, v, g, beta = dummy_inputs
    fake_o = torch.empty_like(k)
    fake_final_state = torch.empty(1, 2, 16, 16)
    num_accepted = torch.tensor([2, 3], dtype=torch.int32)

    with patch(
        "vllm.third_party.flash_linear_attention.ops.kda.fused_recurrent_kda_fwd",
        return_value=(fake_o, fake_final_state),
    ) as mock_fwd:
        o, final_state = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            num_accepted_tokens=num_accepted,
        )
        assert mock_fwd.called
        assert mock_fwd.call_args.kwargs["num_accepted_tokens"] is num_accepted
        assert o is fake_o
        assert final_state is fake_final_state


def test_fused_recurrent_kda_passes_num_accepted_tokens_positional(
    dummy_inputs,
):
    q, k, v, g, beta = dummy_inputs
    fake_o = torch.empty_like(k)
    fake_final_state = torch.empty(1, 2, 16, 16)
    scale = 0.25
    initial_state = torch.empty(1, 2, 16, 16)
    inplace_final_state = True
    use_qk_l2norm = True
    cu_seqlens = None
    ssm_state_indices = None
    num_accepted = torch.tensor([1], dtype=torch.int32)

    with patch(
        "vllm.third_party.flash_linear_attention.ops.kda.fused_recurrent_kda_fwd",
        return_value=(fake_o, fake_final_state),
    ) as mock_fwd:
        o, final_state = fused_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            inplace_final_state,
            use_qk_l2norm,
            cu_seqlens,
            ssm_state_indices,
            num_accepted,
        )
        assert mock_fwd.called
        assert mock_fwd.call_args.kwargs["num_accepted_tokens"] is num_accepted
        assert mock_fwd.call_args.kwargs["scale"] == scale
        assert o is fake_o
        assert final_state is fake_final_state


def test_fused_recurrent_kda_invalid_cu_seqlens_batch_size():
    q = torch.randn(2, 4, 2, 16)
    k = torch.randn(2, 4, 2, 16)
    v = torch.randn(2, 4, 2, 16)
    g = torch.randn(2, 4, 2, 16)
    beta = torch.randn(2, 4, 2)
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)

    with pytest.raises(ValueError, match="The batch size is expected to be 1"):
        fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
        )
