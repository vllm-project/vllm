# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable (bf16-dequant) Triton paged-MQA-logits kernel vs torch reference.

The portable kernel backs the DeepSeek-V4 sparse indexer on platforms whose
Triton target cannot compile fp8 tl.dot (e.g. ROCm gfx1151 / RDNA3.5). It
must produce the same logits as fp8_paged_mqa_logits_torch.
"""
import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "batch_size,next_n,ctx_lens",
    [
        (3, 1, [37, 811, 5120]),
        (2, 6, [64, 1930]),
        (1, 1, [1]),
        (1, 6, [64]),
    ],
)
def test_portable_fp8_paged_mqa_logits_matches_torch(
    batch_size, next_n, ctx_lens
):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
        portable_fp8_paged_mqa_logits,
    )

    torch.manual_seed(7)
    device = "cuda"
    fp8 = current_platform.fp8_dtype()
    num_heads, dim, block_size, num_blocks = 64, 128, 64, 300
    max_model_len = 8192

    vals = (torch.randn(num_blocks, block_size, dim, device=device) * 0.25).to(fp8)
    scales = torch.rand(num_blocks, block_size, device=device) * 1.5 + 0.25
    flat = torch.empty(
        num_blocks, block_size * (dim + 4), dtype=torch.uint8, device=device
    )
    flat[:, : block_size * dim] = vals.reshape(
        num_blocks, block_size * dim
    ).view(torch.uint8)
    flat[:, block_size * dim :] = (
        scales.contiguous().view(torch.uint8).reshape(num_blocks, block_size * 4)
    )
    kv_cache = flat.view(num_blocks, block_size, 1, dim + 4)

    q = (
        torch.randn(batch_size, next_n, num_heads, dim, device=device) * 0.3
    ).to(fp8)
    weights = torch.rand(batch_size * next_n, num_heads, device=device) * 0.2
    context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device=device)
    block_tables = torch.randint(
        0,
        num_blocks,
        (batch_size, max_model_len // block_size),
        dtype=torch.int32,
        device=device,
    )

    ref = fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )
    out = portable_fp8_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )

    ref_inf = torch.isinf(ref) & (ref < 0)
    assert torch.equal(ref_inf, torch.isinf(out) & (out < 0))
    finite = ~ref_inf
    if finite.any():
        r, o = ref[finite], out[finite]
        rel = ((o - r).abs() / r.abs().clamp_min(1e-3)).max().item()
        assert rel < 0.05
        for i in range(ref.shape[0]):
            k = min(64, int((~ref_inf[i]).sum().item()))
            if k == 0:
                continue
            ti = set(torch.topk(ref[i], k).indices.tolist())
            to = set(torch.topk(out[i], k).indices.tolist())
            assert len(ti & to) / k > 0.93
