# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4EmbeddingMethod,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)


def test_nvfp4_embedding_matches_reference_dequant():
    torch.manual_seed(0)
    vocab, hidden = 37, 64
    layer = torch.nn.Module()
    method = ModelOptNvFp4EmbeddingMethod()
    method.create_weights(layer, hidden, [vocab], hidden, vocab, torch.bfloat16)
    assert layer.weight.shape == (vocab, hidden // 2)
    assert layer.weight_scale.shape == (vocab, hidden // 16)

    layer.weight.data.copy_(torch.randint(0, 256, (vocab, hidden // 2)))
    layer.weight_scale.data.copy_(
        torch.rand(vocab, hidden // 16).mul(4).to(torch.float8_e4m3fn)
    )
    layer.weight_scale_2.data.fill_(0.03)
    method.process_weights_after_loading(layer)

    ids = torch.tensor([[0, 5, 36], [5, 5, 1]])
    out = method.embedding(layer, ids)

    table = break_fp4_bytes(layer.weight.data, torch.float32).view(vocab, -1, 16)
    table = table * (layer.weight_scale.data.float() * 0.03).unsqueeze(-1)
    ref = table.view(vocab, hidden)[ids].to(torch.bfloat16)
    assert out.shape == (2, 3, hidden) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
