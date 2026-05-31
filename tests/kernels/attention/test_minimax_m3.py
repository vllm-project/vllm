# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for MiniMax M3 sparse prefill attention kernels."""

import pytest
import torch

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

    torch.manual_seed(0)
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
