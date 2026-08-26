# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from vllm.model_executor.layers.sparse_attn_indexer import (
    _top_k_per_row_prefill,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_bi_dispatch_uses_standalone_topk_with_request_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.ops.load_library(library)

    top_k = 64
    row_starts = torch.tensor([0, 17, 65], dtype=torch.int32, device="cuda")
    row_ends = row_starts + 512
    logits = torch.ones((3, 1024), dtype=torch.float32, device="cuda")
    expected = row_ends[:, None] - 1 - torch.arange(
        top_k, dtype=torch.int32, device="cuda"
    )

    for _ in range(3):
        indices = torch.empty((3, top_k), dtype=torch.int32, device="cuda")
        _top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            indices,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            top_k,
        )
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_standalone_topk_matches_vllm_score_and_tie_order() -> None:
    library = os.environ.get("DS4_BI_TOPK_LIB")
    if not library:
        pytest.skip("DS4_BI_TOPK_LIB is not configured")
    torch.ops.load_library(library)

    logits = torch.zeros((1, 16), dtype=torch.float32, device="cuda")
    logits[0, 5] = 3
    logits[0, 2] = 4
    logits[0, 7] = 4
    logits[0, 1] = 2
    row_starts = torch.tensor([0], dtype=torch.int32, device="cuda")
    row_ends = torch.tensor([16], dtype=torch.int32, device="cuda")
    expected = torch.tensor([[7, 2, 5, 1]], dtype=torch.int32, device="cuda")

    for _ in range(3):
        indices = torch.empty((1, 4), dtype=torch.int32, device="cuda")
        torch.ops.ds4_bi.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            indices,
            1,
            logits.stride(0),
            logits.stride(1),
            4,
        )
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)
