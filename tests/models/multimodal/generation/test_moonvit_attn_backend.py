# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from vllm.model_executor.models.moonvit import (multihead_attention,
                                                sdpa_attention,
                                                xformers_attention)


def random_create_qk_cu_seqlens(tot_seqlens, batch_size, device):
    cuts = sorted(random.sample(range(1, tot_seqlens), batch_size - 1))
    segs = [a - b for a, b in zip(cuts + [tot_seqlens], [0] + cuts)]

    # self-attention, q,k,v are the same shape
    cu_seqlens = torch.nn.functional.pad(torch.tensor(segs).cumsum(0), (1, 0))
    cu_seqlens = cu_seqlens.to(dtype=torch.int32, device=device)
    return cu_seqlens, cu_seqlens


@pytest.mark.parametrize("qkv_shape", [(65340, 16, 72)])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_moonvit_attn_backend(qkv_shape, batch_size, dtype, device):
    q_cu_lens, k_cu_lens = random_create_qk_cu_seqlens(qkv_shape[0],
                                                       batch_size, device)

    q = torch.randn(*qkv_shape, dtype=dtype, device=device)
    k = torch.randn(*qkv_shape, dtype=dtype, device=device)
    v = torch.randn(*qkv_shape, dtype=dtype, device=device)

    flash_attn_out = multihead_attention(q, k, v, q_cu_lens, k_cu_lens)
    xformers_attn_out = xformers_attention(q, k, v, q_cu_lens, k_cu_lens)
    sdpa_attn_out = sdpa_attention(q, k, v, q_cu_lens, k_cu_lens)

    rtol, atol = 1e-3, 5e-3
    torch.testing.assert_close(flash_attn_out,
                               sdpa_attn_out,
                               rtol=rtol,
                               atol=atol)
    torch.testing.assert_close(flash_attn_out,
                               xformers_attn_out,
                               rtol=rtol,
                               atol=atol)
