# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for MiniMax M3 sparse prefill attention kernels."""

import pytest
import torch

from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_topk,
    minimax_m3_index_topk_decode,
)
from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

if not current_platform.is_cuda():
    pytest.skip("MiniMax M3 attention kernels require CUDA.", allow_module_level=True)


NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
BLOCK_SIZE = 128
DTYPE = torch.bfloat16
SM_SCALE = HEAD_DIM**-0.5
TOPK = 16


# Index top-k kernels.
def _reference_index_topk(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float,
) -> torch.Tensor:
    total_q, num_idx_heads, _ = idx_q.shape
    out = torch.full(
        (num_idx_heads, total_q, topk), -1, device=idx_q.device, dtype=torch.int32
    )

    q_start = 0
    for req_id, (q_len, seq_len, prefix_len) in enumerate(
        zip(q_lens.tolist(), seq_lens.tolist(), prefix_lens.tolist())
    ):
        q_end = q_start + q_len
        q = idx_q[q_start:q_end]
        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages = block_table[req_id, :num_blocks]
        k = index_kv_cache[pages].reshape(num_blocks * BLOCK_SIZE, -1)
        score = torch.einsum("qhd,kd->hqk", q.float(), k.float()) * sm_scale

        q_pos = prefix_len + torch.arange(q_len, device=idx_q.device)
        k_pos = torch.arange(k.shape[0], device=idx_q.device)
        score.masked_fill_(k_pos[None, :] > q_pos[:, None], -float("inf"))
        score = score.reshape(num_idx_heads, q_len, num_blocks, BLOCK_SIZE)
        score_tensor = score.max(dim=3).values

        valid_blocks = (q_pos + BLOCK_SIZE) // BLOCK_SIZE
        for local_q, num_valid_blocks in enumerate(valid_blocks.tolist()):
            end = min(init_blocks, num_valid_blocks)
            score_tensor[:, local_q, :end] = 1e30
            start = max(0, num_valid_blocks - local_blocks)
            score_tensor[:, local_q, start:num_valid_blocks] = 1e29

            k = min(topk, num_valid_blocks)
            topk_idx = score_tensor[:, local_q].topk(k, dim=1).indices
            out[:, q_start + local_q, :k] = topk_idx
        q_start = q_end

    return out


def test_prefill_index_topk_correctness():
    topk = 6
    init_blocks = 0
    local_blocks = 1
    num_idx_heads = 2
    head_dim = 16
    q_lens = torch.tensor((4, 3), device="cuda", dtype=torch.int32)
    prefix_lens = torch.tensor((0, 1024), device="cuda", dtype=torch.int32)
    seq_lens = prefix_lens + q_lens
    batch = q_lens.numel()
    max_seq_len = seq_lens.max().item()
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = batch * max_blocks

    cu_seqlens = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = q_lens.cumsum(0)
    block_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        batch, max_blocks
    )
    idx_q = torch.ones(q_lens.sum().item(), num_idx_heads, head_dim, device="cuda")
    index_kv_cache = torch.empty(num_pages, BLOCK_SIZE, head_dim, device="cuda")
    for req_id in range(batch):
        for block_id in range(max_blocks):
            page = block_table[req_id, block_id]
            index_kv_cache[page].fill_(block_id + 1)

    actual = minimax_m3_index_topk(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_query_len=q_lens.max().item(),
        max_seq_len=max_seq_len,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_kv_heads=num_idx_heads,
        sm_scale=head_dim**-0.5,
    )
    expected = _reference_index_topk(
        idx_q,
        index_kv_cache,
        block_table,
        q_lens,
        seq_lens,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        head_dim**-0.5,
    )
    assert torch.equal(actual, expected)


def test_decode_index_topk_correctness():
    topk = 6
    init_blocks = 0
    local_blocks = 1
    num_idx_heads = 2
    head_dim = 16
    seq_lens = torch.tensor((7, 129, 1025), device="cuda", dtype=torch.int32)
    q_lens = torch.ones_like(seq_lens)
    prefix_lens = seq_lens - 1
    batch = seq_lens.numel()
    max_seq_len = seq_lens.max().item()
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = batch * max_blocks

    block_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        batch, max_blocks
    )
    idx_q = torch.ones(batch, num_idx_heads, head_dim, device="cuda")
    index_kv_cache = torch.empty(num_pages, BLOCK_SIZE, head_dim, device="cuda")
    for req_id in range(batch):
        for block_id in range(max_blocks):
            page = block_table[req_id, block_id]
            index_kv_cache[page].fill_(block_id + 1)

    actual = minimax_m3_index_topk_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len=max_seq_len,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_kv_heads=num_idx_heads,
        sm_scale=head_dim**-0.5,
    )
    expected = _reference_index_topk(
        idx_q,
        index_kv_cache,
        block_table,
        q_lens,
        seq_lens,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        head_dim**-0.5,
    )
    assert torch.equal(actual, expected)


# Sparse attention kernels.
def _reference_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q, dtype=torch.float32)
    gqa_group_size = NUM_Q_HEADS // NUM_KV_HEADS
    q_start = 0
    for req_id, (q_len, seq_len, prefix_len) in enumerate(
        zip(q_lens.tolist(), seq_lens.tolist(), prefix_lens.tolist())
    ):
        q_end = q_start + q_len
        q_req = q[q_start:q_end]
        positions = torch.arange(seq_len, device="cuda")
        pages = block_table[req_id, positions // BLOCK_SIZE]
        rows = positions % BLOCK_SIZE
        k_req = kv_cache[pages, 0, rows]
        v_req = kv_cache[pages, 1, rows].float()

        q_pos = prefix_len + torch.arange(q_len, device="cuda")
        key_blocks = positions // BLOCK_SIZE
        causal_mask = positions.unsqueeze(0) <= q_pos.unsqueeze(1)

        for kv_head in range(NUM_KV_HEADS):
            selected = topk_idx[kv_head, q_start:q_end]
            selected_mask = (key_blocks[None, :, None] == selected[:, None, :]).any(-1)
            mask = causal_mask & selected_mask
            head_start = kv_head * gqa_group_size
            head_end = head_start + gqa_group_size

            q_heads = q_req[:, head_start:head_end].transpose(0, 1)
            k_head = k_req[:, kv_head].T.expand(gqa_group_size, -1, -1)
            scores = torch.bmm(q_heads, k_head, out_dtype=torch.float32)
            scores = scores.transpose(0, 1) * SM_SCALE
            probs = torch.softmax(
                scores.masked_fill(~mask[:, None, :], -float("inf")), -1
            )
            out[q_start:q_end, head_start:head_end] = torch.einsum(
                "qhk,kd->qhd", probs, v_req[:, kv_head]
            )
        q_start += q_len
    return out.to(q.dtype)


@pytest.mark.parametrize("backend", ["triton", "cutedsl"])
@pytest.mark.parametrize(
    ("q_lens", "kv_lens"),
    [
        ((129, 257), (129, 257)),
        ((65, 129, 257), (129, 257, 385)),
    ],
)
def test_prefill_sparse_attention_correctness(
    backend: str,
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
):
    if backend == "cutedsl":
        if not current_platform.is_device_capability_family(100):
            pytest.skip("MiniMax M3 CuteDSL prefill requires CUDA SM10x.")
        if not has_cutedsl():
            pytest.skip("cutedsl (cutlass) is not installed")

    assert len(q_lens) == len(kv_lens)
    assert all(kv_len >= q_len for q_len, kv_len in zip(q_lens, kv_lens))

    # Build paged-KV metadata, including a non-identity page order.
    batch = len(q_lens)
    pages_per_req = [(kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE for kv_len in kv_lens]
    max_blocks = max(pages_per_req)
    num_pages = sum(pages_per_req)
    physical_pages = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
    block_table = torch.zeros(batch, max_blocks, device="cuda", dtype=torch.int32)
    base_page = 0
    for req_id, num_req_pages in enumerate(pages_per_req):
        block_table[req_id, :num_req_pages] = physical_pages[
            base_page : base_page + num_req_pages
        ]
        base_page += num_req_pages

    q_lens_t = torch.tensor(q_lens, device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    prefix_lens = seq_lens - q_lens_t
    cu_seqlens = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = q_lens_t.cumsum(0)
    cu_seqlens_k = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    cu_seqlens_k[1:] = seq_lens.cumsum(0)
    total_q = sum(q_lens)
    max_seqlen_q = max(q_lens)
    max_seqlen_k = max(kv_lens)

    q_shape = (total_q, NUM_Q_HEADS, HEAD_DIM)
    kv_shape = (num_pages, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    q = torch.randn(q_shape, device="cuda", dtype=DTYPE)
    kv_cache = torch.randn(kv_shape, device="cuda", dtype=DTYPE)

    # Build sparse block indices with the same contract as the real M3 indexer:
    # one forced local block, then score-selected older causal blocks.
    topk_shape = (NUM_KV_HEADS, total_q, TOPK)
    topk_idx = torch.full(topk_shape, -1, device="cuda", dtype=torch.int32)
    q_start = 0
    for q_len, prefix_len in zip(q_lens_t.tolist(), prefix_lens.tolist()):
        for local_q in range(q_len):
            current_block = (prefix_len + local_q) // BLOCK_SIZE
            older_blocks = torch.randperm(
                current_block, device="cuda", dtype=torch.int32
            )
            selected = torch.cat(
                [
                    torch.tensor([current_block], device="cuda", dtype=torch.int32),
                    older_blocks[: TOPK - 1],
                ]
            )
            topk_idx[:, q_start + local_q, : selected.numel()] = selected
        q_start += q_len

    actual = torch.empty_like(q)
    if backend == "triton":
        minimax_m3_sparse_attn(
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            max_seqlen_q,
            NUM_KV_HEADS,
            SM_SCALE,
            actual,
        )
    else:
        from vllm.models.minimax_m3.nvidia.ops.prefill_gqa_sparse import (
            minimax_m3_sparse_attn_cutedsl,
        )

        minimax_m3_sparse_attn_cutedsl(
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens,
            cu_seqlens_k,
            seq_lens,
            max_seqlen_q,
            max_seqlen_k,
            NUM_KV_HEADS,
            SM_SCALE,
            actual,
            total_kv_blocks=num_pages,
        )

    expected = _reference_sparse_attn(
        q,
        kv_cache,
        topk_idx,
        block_table,
        q_lens_t,
        seq_lens,
        prefix_lens,
    )
    torch.accelerator.synchronize()

    error = (actual.float() - expected.float()).abs()
    assert error.mean().item() < 2.5e-4
    assert error.max().item() < 1.7e-2
