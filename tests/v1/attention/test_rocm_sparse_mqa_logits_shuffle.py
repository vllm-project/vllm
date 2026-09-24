# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The paged-MQA-logits Triton kernel must read both indexer cache layouts.

DeepSeek-V4's C4A cache is plain pos-major; DeepSeek-V3.2 / GLM-5.x are written
16x16 tiled (SHUFFLE) by the in-tree writer. Speculative decode routes both to
this kernel, so a layout misread silently corrupts the indexer top-k rather than
failing loudly -- hence a numerical check against an explicit reference.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only sparse MLA kernel", allow_module_level=True)

from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (  # noqa: E402
    rocm_fp8_paged_mqa_logits_triton,
)

D = 128
H = 32
BLOCK_SIZE = 64
NUM_BLOCKS = 32


def _build(layout: str, batch: int, next_n: int, ctx: int, device: str):
    """Pack values/scales into the paged cache in the requested layout."""
    fp8 = current_platform.fp8_dtype()
    torch.manual_seed(0)
    q = (torch.randn(batch, next_n, H, D, device=device) * 0.4).to(fp8)
    vals = (torch.randn(NUM_BLOCKS, BLOCK_SIZE, D, device=device) * 0.4).to(fp8)
    scales = torch.rand(NUM_BLOCKS, BLOCK_SIZE, device=device).float() * 0.5 + 0.5

    kv = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, D + 4, dtype=torch.uint8, device=device)
    flat = kv.reshape(NUM_BLOCKS, -1)
    if layout == "SHUFFLE":
        p = torch.arange(BLOCK_SIZE, device=device)
        d = torch.arange(D, device=device)
        off = ((p // 16) * (16 * D) + (p % 16) * 16)[:, None] + (
            (d // 16) * (16 * 16) + (d % 16)
        )[None, :]
        tiled = torch.zeros(NUM_BLOCKS, BLOCK_SIZE * D, dtype=fp8, device=device)
        tiled.scatter_(
            1, off.reshape(1, -1).expand(NUM_BLOCKS, -1), vals.reshape(NUM_BLOCKS, -1)
        )
        flat[:, : BLOCK_SIZE * D] = tiled.view(torch.uint8)
    else:
        flat[:, : BLOCK_SIZE * D] = vals.reshape(NUM_BLOCKS, -1).view(torch.uint8)
    flat.view(torch.float32)[:, (BLOCK_SIZE * D) // 4 :] = scales

    weights = torch.randn(batch * next_n, H, device=device).float().abs()
    ctx_lens = torch.full((batch * next_n,), ctx, dtype=torch.int32, device=device)
    n_blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.arange(
        batch * n_blocks, dtype=torch.int32, device=device
    ).reshape(batch, n_blocks)
    return q, kv, weights, ctx_lens, block_tables, vals, scales


def _reference(q, weights, vals, scales, block_tables, batch, next_n, ctx):
    """logits[r, p] = sum_h relu(q[b,n,h] . k[p]) * w[r,h] * scale[p]."""
    out = torch.zeros(batch * next_n, ctx, device=q.device, dtype=torch.float32)
    for r in range(batch * next_n):
        b, n = r // next_n, r % next_n
        qv = q[b, n].float()
        for p in range(ctx):
            blk = int(block_tables[b, p // BLOCK_SIZE])
            k = vals[blk, p % BLOCK_SIZE].float()
            out[r, p] = (torch.relu(qv @ k) * weights[r]).sum() * scales[
                blk, p % BLOCK_SIZE
            ]
    return out


@pytest.mark.parametrize("layout", ["NORMAL", "SHUFFLE"])
@pytest.mark.parametrize("next_n", [1, 2, 8])
def test_paged_mqa_logits_matches_reference(layout: str, next_n: int):
    """Both layouts must reproduce the reference, and pick the same top-k."""
    from vllm.v1.worker.workspace import init_workspace_manager

    device = "cuda"
    init_workspace_manager(torch.device(f"{device}:0"))
    batch, ctx, max_model_len = 1, 256, 1024

    q, kv, weights, ctx_lens, block_tables, vals, scales = _build(
        layout, batch, next_n, ctx, device
    )
    got = rocm_fp8_paged_mqa_logits_triton(
        q, kv, weights, ctx_lens, block_tables, max_model_len, cache_layout=layout
    )[:, :ctx].float()
    ref = _reference(q, weights, vals, scales, block_tables, batch, next_n, ctx)

    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    # The indexer consumes the ranking, not the values, so check it explicitly.
    k = 64
    for row in range(got.shape[0]):
        assert set(got[row].topk(k).indices.tolist()) == set(
            ref[row].topk(k).indices.tolist()
        )
