# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.cohere_asr import RelPositionMultiHeadAttention


@pytest.mark.cpu_test
def test_rel_position_attention_casts_bias_to_query_dtype() -> None:
    n_head = 2
    d_model = 4
    attention = RelPositionMultiHeadAttention(
        n_head=n_head,
        n_feat=d_model,
        pos_bias_u=torch.zeros(n_head, d_model // n_head),
        pos_bias_v=torch.zeros(n_head, d_model // n_head),
    ).half()

    attention.pos_bias_u = attention.pos_bias_u.float()
    attention.pos_bias_v = attention.pos_bias_v.float()

    output = attention(
        query=torch.randn(1, 3, d_model, dtype=torch.float16),
        key=torch.randn(1, 3, d_model, dtype=torch.float16),
        value=torch.randn(1, 3, d_model, dtype=torch.float16),
        mask=None,
        pos_emb=torch.randn(1, 5, d_model, dtype=torch.float16),
    )

    assert output.dtype == torch.float16
    assert output.shape == (1, 3, d_model)
