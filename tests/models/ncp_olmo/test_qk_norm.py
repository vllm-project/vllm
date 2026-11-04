# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Full-hidden Q/K normalization tests for token and HLM attention."""

import pytest
import torch


@pytest.mark.parametrize("tower", ["token", "hlm"])
def test_qk_norm_uses_full_hidden_dimension(ncp_model, tower: str) -> None:
    layers = (
        ncp_model.token_backbone.encoder.layers
        if tower == "token"
        else ncp_model.highlevel.concept_predictor.hlm_block.layers
    )
    attention = layers[0].self_attention
    hidden_size = ncp_model.backend_config.hidden_size
    torch.manual_seed(42)
    query = torch.randn(2, hidden_size, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)

    with torch.inference_mode():
        actual_q, actual_k = attention._apply_qk_norm(query, key)
        expected_q = attention.q_layernorm(query)
        expected_k = attention.k_layernorm(key)

    assert attention.q_layernorm.weight.shape == (hidden_size,)
    assert attention.k_layernorm.weight.shape == (hidden_size,)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
