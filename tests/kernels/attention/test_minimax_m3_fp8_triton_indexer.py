# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 index-K side cache on the Triton indexer: top-k must agree with bf16.

The MiniMax M3 lightning-indexer scores only feed a top-k block ranking (the
attention itself reads the main KV cache), so an fp8 (e4m3) side cache trades
~2% relative score error for half the per-step index-K read bandwidth — the
dominant linear-in-context decode cost at long context. #45892 enabled fp8
index caches on the SM100 MSA path; this covers the Triton impl used
everywhere else (e.g. SM120), for both the decode and prefill scorers.
"""
import pytest
import torch

from vllm.models.minimax_m3.common.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm.platforms import current_platform

PAGE = SPARSE_BLOCK_SIZE
HEAD_DIM = 128
HEADS = 1
TOPK = 16
NUM_PAGES = 100
SEQ_LEN = 96 * PAGE + 37
PLANTED = (7, 40, 88)


def _make_inputs(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    keys = torch.randn(
        NUM_PAGES * PAGE, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    # RMS-normalize like the real (post index_k_norm) keys.
    keys = keys * keys.float().pow(2).mean(-1, keepdim=True).add(
        1e-6
    ).rsqrt().to(torch.bfloat16)
    q = torch.randn(1, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    for b in PLANTED:
        keys[b * PAGE + 5] = (q[0, 0] * 3).to(torch.bfloat16)
    return q, keys


def _check_selection(top_bf16: set[int], top_fp8: set[int]) -> None:
    # Planted (clearly relevant) blocks must never be lost.
    assert all(b in top_fp8 for b in PLANTED)
    # Selection may differ only at the noise floor (near-tied filler blocks).
    assert len(top_bf16 & top_fp8) >= TOPK - 2


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@torch.inference_mode()
def test_fp8_index_cache_decode_topk_matches_bf16() -> None:
    dev = "cuda"
    q, keys = _make_inputs(dev)
    block_table = torch.arange(
        NUM_PAGES, dtype=torch.int32, device=dev
    ).unsqueeze(0)
    seq_lens = torch.tensor([SEQ_LEN], dtype=torch.int32, device=dev)

    def select(cache: torch.Tensor) -> set[int]:
        idx = minimax_m3_index_decode(
            q,
            cache.view(NUM_PAGES, PAGE, HEAD_DIM).contiguous(),
            block_table,
            seq_lens,
            SEQ_LEN,
            TOPK,
            0,
            0,
            HEADS,
            1,
            1,
        )
        return {int(b) for b in idx[0, 0].tolist() if b >= 0}

    _check_selection(select(keys), select(keys.to(torch.float8_e4m3fn)))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@torch.inference_mode()
def test_fp8_index_cache_prefill_topk_matches_bf16() -> None:
    dev = "cuda"
    q, keys = _make_inputs(dev)
    block_table = torch.arange(
        NUM_PAGES, dtype=torch.int32, device=dev
    ).unsqueeze(0)
    seq_lens = torch.tensor([SEQ_LEN], dtype=torch.int32, device=dev)
    # single decode-like query token at the end of the sequence
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    prefix_lens = torch.tensor([SEQ_LEN - 1], dtype=torch.int32, device=dev)

    def select(cache: torch.Tensor) -> set[int]:
        score = minimax_m3_index_score(
            q,
            cache.view(NUM_PAGES, PAGE, HEAD_DIM).contiguous(),
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            1,
            SEQ_LEN,
            HEADS,
        )
        idx = minimax_m3_index_topk(
            score, cu_seqlens_q, prefix_lens, 1, TOPK, 0, 0
        )
        return {int(b) for b in idx[0, 0].tolist() if b >= 0}

    _check_selection(select(keys), select(keys.to(torch.float8_e4m3fn)))
