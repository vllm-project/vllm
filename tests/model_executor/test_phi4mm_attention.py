# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.phi4mm_utils import MultiHeadedAttention


def test_eager_attention_does_not_retain_probability_matrix():
    module = MultiHeadedAttention(
        n_head=2,
        n_feat=16,
        dropout_rate=0.0,
        use_pt_scaled_dot_product_attention=False,
    )
    module.eval()
    batch, time, feat = 2, 8, 16
    x = torch.randn(batch, time, feat)
    with torch.no_grad():
        out = module(x, x, x, None, None, None)

    assert out.shape == (batch, time, feat)
    assert getattr(module, "attn", None) is None


def test_multiheaded_attention_defaults_to_scaled_dot_product():
    module = MultiHeadedAttention(n_head=2, n_feat=16, dropout_rate=0.0)
    assert module.use_pt_scaled_dot_product_attention is True
