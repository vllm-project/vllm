# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.minicpmv import Resampler2_5, Resampler4_5


@pytest.mark.parametrize("resampler_cls", [Resampler2_5, Resampler4_5])
def test_minicpmv_resampler_device(resampler_cls):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_queries = 8
    embed_dim = 16
    num_heads = 2
    resampler = resampler_cls(
        num_queries=num_queries,
        embed_dim=embed_dim,
        num_heads=num_heads,
    ).to(device)

    bs = 2
    h, w = 10, 10
    x = torch.randn((bs, h * w, embed_dim), device=device)
    tgt_sizes = torch.tensor([[h, w], [h, w]], device=device)

    if resampler_cls is Resampler4_5:
        temporal_ids = [[-1], [-1]]
        out = resampler(x, tgt_sizes, temporal_ids=temporal_ids)
    else:
        out = resampler(x, tgt_sizes)

    # Verify output shape and device
    assert out.device == x.device
    assert out.shape == (bs, num_queries, embed_dim)

    # Verify that buffers are on the target device
    assert resampler.pos_embed.device == x.device
    if resampler_cls is Resampler4_5:
        assert resampler.temporal_pos_embed.device == x.device
