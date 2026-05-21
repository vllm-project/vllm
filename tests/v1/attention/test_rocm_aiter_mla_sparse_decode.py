# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sanity check for the ROCm sparse-MLA indexer decode logits MB budget.

When VLLM_SPARSE_INDEXER_DECODE_MAX_MB forces a per-call split, the chunked
path must produce bit-identical logits to a single full-batch call. The
underlying aiter kernel is deterministic on a fixed shape, so a tiny budget
that triggers chunking is enough to catch any indexing slip.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Only used by ROCm"
)


def _aiter_decode_available() -> bool:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        paged_mqa_logits_module,
    )

    return rocm_aiter_ops.is_enabled() and paged_mqa_logits_module() is not None


def _build_decode_inputs(
    batch_size: int,
    next_n: int,
    heads: int,
    head_dim: int,
    block_size: int,
    max_model_len: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    fp8_dtype = current_platform.fp8_dtype()

    q = torch.randn(
        (batch_size, next_n, heads, head_dim),
        device=device,
        dtype=torch.bfloat16,
    ).to(fp8_dtype)

    num_blocks_per_batch = max_model_len // block_size
    num_blocks = batch_size * num_blocks_per_batch + 4
    kv_cache = torch.randint(
        0,
        255,
        (num_blocks, block_size, 1, head_dim + 4),
        device=device,
        dtype=torch.uint8,
    )
    weights = torch.randn(
        (batch_size * next_n, heads), device=device, dtype=torch.float32
    )
    context_lens = torch.full(
        (batch_size,),
        max_model_len // 2,
        device=device,
        dtype=torch.int32,
    )
    block_tables = torch.zeros(
        (batch_size, num_blocks_per_batch),
        device=device,
        dtype=torch.int32,
    )
    cursor = 0
    for i in range(batch_size):
        block_tables[i].copy_(
            torch.arange(
                cursor,
                cursor + num_blocks_per_batch,
                device=device,
                dtype=torch.int32,
            )
        )
        cursor += num_blocks_per_batch

    return dict(
        q_fp8=q,
        kv_cache_fp8=kv_cache,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_tables,
        schedule_metadata=torch.empty(0, device=device, dtype=torch.int32),
        max_model_len=max_model_len,
    )


def test_decode_chunked_matches_full(monkeypatch):
    if not _aiter_decode_available():
        pytest.skip("AITER paged_mqa_logits module not available")

    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops

    # B*next_n*max_model_len*4 = 2 MiB per call on the fused 2D path; a 1 MiB
    # budget forces a 2-way split. On the 3D else path the same shape gives
    # 8 MiB per call (heads=4), so the chunking still fires.
    inputs = _build_decode_inputs(
        batch_size=8,
        next_n=1,
        heads=4,
        head_dim=128,
        block_size=64,
        max_model_len=65536,
        seed=0,
    )

    monkeypatch.setenv("VLLM_SPARSE_INDEXER_DECODE_MAX_MB", "0")
    full = ops.rocm_fp8_paged_mqa_logits(**inputs).clone()

    monkeypatch.setenv("VLLM_SPARSE_INDEXER_DECODE_MAX_MB", "1")
    chunked = ops.rocm_fp8_paged_mqa_logits(**inputs).clone()

    assert torch.equal(full, chunked), (
        "chunked decode logits diverged from full-batch call: "
        f"max |delta| = {(full - chunked).abs().max().item()}"
    )


def test_decode_workspace_rounding():
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops

    device = torch.device("cuda")
    ops._decode_ws_2d.clear()
    ops._decode_ws_3d.clear()

    # 5 and 7 both round up to alloc_rows=8 so they share one cache entry.
    ws_a = ops._decode_workspace_2d(device, rows=5, cols=128)
    ws_b = ops._decode_workspace_2d(device, rows=7, cols=128)

    assert len(ops._decode_ws_2d) == 1
    (key2d,) = ops._decode_ws_2d.keys()
    assert key2d == (device, 8, 128)
    assert ws_a.shape == (5, 128)
    assert ws_b.shape == (7, 128)
    assert ws_a.is_contiguous() and ws_b.is_contiguous()
    neg_inf = torch.tensor(float("-inf"), device=device)
    assert torch.all(ws_a == neg_inf)
    assert torch.all(ws_b == neg_inf)

    ws3_a = ops._decode_workspace_3d(device, heads=4, rows=5, cols=128)
    ws3_b = ops._decode_workspace_3d(device, heads=4, rows=7, cols=128)

    assert len(ops._decode_ws_3d) == 1
    (key3d,) = ops._decode_ws_3d.keys()
    assert key3d == (device, 4, 8, 128)
    assert ws3_a.shape == (4, 8, 128)
    assert ws3_b.shape == (4, 8, 128)
    assert torch.all(ws3_a[:, :5, :] == neg_inf)
    assert torch.all(ws3_b[:, :7, :] == neg_inf)
