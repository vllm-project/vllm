# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 (Turing) portable fallback for the DeepSeek-V4 sparse indexer."""

import pytest
import torch
import torch.nn.functional as F

from vllm.models.deepseek_v4.turing.indexer_fallback import (
    supports_turing_indexer_fallback,
)
from vllm.models.deepseek_v4.turing.indexer_logits import fp8_mqa_logits_triton
from vllm.platforms import current_platform

_capability = current_platform.get_device_capability()

pytestmark = pytest.mark.skipif(
    _capability is None or (_capability.major, _capability.minor) != (7, 5),
    reason="SM75 only",
)


def _sm75_capability() -> int:
    cap = current_platform.get_device_capability()
    assert cap is not None
    return cap.major


def test_turing_indexer_supports_sm75():
    assert _sm75_capability() == 7
    assert supports_turing_indexer_fallback()


def test_turing_indexer_persistent_topk_portable():
    num_rows, num_cols, topk = 4, 2048, 512
    scores = torch.randn(num_rows, num_cols, device="cuda", dtype=torch.float32)
    seq_lens = torch.full((num_rows,), num_cols, dtype=torch.int32, device="cuda")
    topk_indices = torch.empty((num_rows, topk), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.ops._C.persistent_topk(
        scores,
        seq_lens,
        topk_indices,
        workspace,
        topk,
        num_cols,
    )
    assert topk_indices.shape == (num_rows, topk)
    assert (topk_indices >= 0).all()
    assert (topk_indices < num_cols).all()


def test_turing_prefill_logits_matches_torch():
    torch.manual_seed(0)
    m, h, d, n = 4, 8, 128, 2048
    q = (torch.randn(m, h, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    k = (torch.randn(n, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    kv_scales = torch.rand(n, device="cuda", dtype=torch.float32) + 0.5
    weights = torch.randn(m, h, device="cuda", dtype=torch.float32)
    cu_starts = torch.tensor([0, 100, 500, 900], dtype=torch.int32, device="cuda")
    cu_ends = torch.tensor([64, 800, 1800, 2048], dtype=torch.int32, device="cuda")

    logits = fp8_mqa_logits_triton(q, k, kv_scales, weights, cu_starts, cu_ends)

    score = torch.einsum("mhd,nd->mhn", q.float(), k.float()) * kv_scales[None, None, :]
    score = F.relu(score) * weights[:, :, None]
    ref = score.sum(1)

    assert logits.shape == (m, n)
    for row in range(m):
        start = int(cu_starts[row].item())
        end = int(cu_ends[row].item())
        assert torch.allclose(
            logits[row, start:end],
            ref[row, start:end],
            rtol=1e-3,
            atol=1e-3,
        )
        outside = torch.cat([logits[row, :start], logits[row, end:]])
        assert not torch.isfinite(outside).any()


def test_turing_paged_logits_fallback_matches_torch():
    torch.manual_seed(1)
    h, d = 8, 128
    num_blocks, block_size, seq_len, max_model_len = 2, 16, 3, 64

    q = (torch.randn(1, 1, h, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    weights = torch.randn(1, h, device="cuda", dtype=torch.float32)

    k = torch.randn(num_blocks, block_size, d, device="cuda") * 0.5
    scales = torch.rand(num_blocks, block_size, 1, device="cuda") + 0.5

    # The indexer insert kernel packs each block value-major: block_size*dim
    # FP8 value bytes followed by block_size*4 FP32 scale bytes. The torch
    # fallback views each block row the same way, so the cache must be packed
    # accordingly (a natural 3D interleaved [b, t, d+4] view does not match).
    row_bytes = block_size * (d + 4)
    flat = torch.zeros(num_blocks * row_bytes, dtype=torch.uint8, device="cuda")
    for b in range(num_blocks):
        row = flat[b * row_bytes : (b + 1) * row_bytes]
        row[: block_size * d] = (
            k[b].to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)
        )
        row[block_size * d :] = (
            scales[b].to(torch.float32).view(torch.uint8).reshape(-1)
        )
    kv_cache = flat.reshape(num_blocks, block_size, d + 4).unsqueeze(-2)

    context_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")

    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
    )

    logits = fp8_paged_mqa_logits_torch(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )

    assert logits.shape == (1, max_model_len)
    assert torch.isfinite(logits[0, :seq_len]).all()
    assert not torch.isfinite(logits[0, seq_len:]).any()

    k_flat = k.reshape(-1, d)[:seq_len].to(torch.float8_e4m3fn).to(torch.float32)
    scale_flat = scales.reshape(-1)[:seq_len].to(torch.float32)
    score = k_flat @ q[0, 0].float().T  # [seq_len, h]
    score = F.relu(score) * weights[0][None, :]
    ref = score.sum(1) * scale_flat
    assert torch.allclose(logits[0, :seq_len], ref, rtol=1e-3, atol=1e-3)


def test_turing_paged_logits_fallback_spec_decode():
    # next_n > 1 (native speculative-decode) branch of the torch fallback.
    torch.manual_seed(2)
    h, d, next_n = 8, 128, 2
    num_blocks, block_size, seq_len, max_model_len = 2, 16, 3, 64

    q = (torch.randn(1, next_n, h, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    weights = torch.randn(1 * next_n, h, device="cuda", dtype=torch.float32)

    k = torch.randn(num_blocks, block_size, d, device="cuda") * 0.5
    scales = torch.rand(num_blocks, block_size, 1, device="cuda") + 0.5
    cache = torch.zeros(num_blocks, block_size, d + 4, dtype=torch.uint8, device="cuda")
    cache[..., :d] = k.to(torch.float8_e4m3fn).view(torch.uint8)
    cache[..., d:] = scales.to(torch.float32).view(torch.uint8)
    kv_cache = cache.unsqueeze(-2)

    context_lens = torch.tensor([[seq_len]], dtype=torch.int32, device="cuda")
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")

    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
    )

    logits = fp8_paged_mqa_logits_torch(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )

    assert logits.shape == (1 * next_n, max_model_len)
    assert torch.isfinite(logits[:, :seq_len]).all()
    assert not torch.isfinite(logits[:, seq_len:]).any()
