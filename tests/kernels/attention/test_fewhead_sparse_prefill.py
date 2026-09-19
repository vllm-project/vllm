# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fewhead_matches_padded_flashmla():
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd,
        is_flashmla_sparse_supported,
    )

    ok, reason = is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason or "FlashMLA sparse unavailable")

    from vllm.models.deepseek_v41.nvidia.triton_fewhead_sparse_prefill import (
        fewhead_sparse_mla_fwd,
    )

    torch.manual_seed(0)
    device = "cuda"
    d, topk, h_local, h_pad, s_q = 512, 640, 8, 64, 64
    s_kv = max(s_q, topk)
    sm_scale = d**-0.5
    kv = torch.randn((s_kv, 1, d), dtype=torch.bfloat16, device=device)
    q64 = torch.randn((s_q, h_pad, d), dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, s_kv, (s_q, topk), dtype=torch.int32, device=device)
    lens = torch.full((s_q,), 512, dtype=torch.int32, device=device)
    sink64 = torch.zeros((h_pad,), dtype=torch.float32, device=device)
    sink64[h_local:] = float("-inf")
    flash_out = flash_mla_sparse_fwd(
        q=q64,
        kv=kv,
        indices=indices.unsqueeze(1),
        sm_scale=sm_scale,
        attn_sink=sink64,
        topk_length=lens,
    )
    flash_q = flash_out[0] if isinstance(flash_out, (tuple, list)) else flash_out
    tri = fewhead_sparse_mla_fwd(
        q64[:, :h_local].contiguous(),
        kv,
        indices,
        sm_scale,
        attn_sink=sink64[:h_local].contiguous(),
        topk_length=lens,
    )
    err = (tri.float() - flash_q[:, :h_local].float()).abs()
    assert bool(torch.isfinite(tri).all())
    assert float(err.mean()) < 2e-3
    assert float(err.max()) < 2e-2
