# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.models.cohere_asr import (
    CohereASRModel,
    RelPositionMultiHeadAttention,
)
from vllm.utils.torch_utils import set_default_torch_dtype


class _CohereASRWeightLoadingModel(CohereASRModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.register_buffer("pos_bias_u", torch.zeros(2, 2))
        self.register_buffer("pos_bias_v", torch.zeros(2, 2))
        self.attention = RelPositionMultiHeadAttention(
            n_head=2,
            n_feat=4,
            pos_bias_u=self.pos_bias_u,
            pos_bias_v=self.pos_bias_v,
        )


@pytest.mark.cpu_test
def test_load_weights_preserves_runtime_bias_dtype() -> None:
    with set_default_torch_dtype(torch.float16):
        model = _CohereASRWeightLoadingModel()

    loaded_weight = torch.ones_like(model.pos_bias_u, dtype=torch.float32)
    model.load_weights([("pos_bias_u", loaded_weight), ("pos_bias_v", loaded_weight)])

    assert model.pos_bias_u.dtype == torch.float16
    assert model.pos_bias_v.dtype == torch.float16
    assert model.attention.pos_bias_u.dtype == torch.float16
    assert model.attention.pos_bias_v.dtype == torch.float16
    torch.testing.assert_close(model.pos_bias_u, loaded_weight.half())

    output = model.attention(
        query=torch.randn(1, 3, 4, dtype=torch.float16),
        key=torch.randn(1, 3, 4, dtype=torch.float16),
        value=torch.randn(1, 3, 4, dtype=torch.float16),
        mask=None,
        pos_emb=torch.randn(1, 5, 4, dtype=torch.float16),
    )

    assert output.dtype == torch.float16
    assert output.shape == (1, 3, 4)
