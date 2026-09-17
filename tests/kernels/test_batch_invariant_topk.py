# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    require_batch_invariant_kernel,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    _top_k_per_row_prefill,
)
from vllm.platforms import current_platform


def _require_unified_kernel() -> None:
    try:
        require_batch_invariant_kernel()
    except RuntimeError as error:
        pytest.skip(str(error))


def _run_topk(
    logits: torch.Tensor, row_start: int, row_length: int, top_k: int
) -> torch.Tensor:
    row_starts = torch.tensor([row_start], dtype=torch.int32, device="cuda")
    row_ends = row_starts + row_length
    indices = torch.empty((1, top_k), dtype=torch.int32, device="cuda")
    torch.ops.vllm_batch_invariant.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        1,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )
    return indices


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_bi_dispatch_uses_unified_kernel_with_request_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    _require_unified_kernel()

    top_k = 64
    row_starts = torch.tensor([0, 17, 65], dtype=torch.int32, device="cuda")
    row_ends = row_starts + 512
    logits = torch.ones((3, 1024), dtype=torch.float32, device="cuda")
    expected = (
        (row_ends - row_starts)[:, None]
        - 1
        - torch.arange(top_k, dtype=torch.int32, device="cuda")
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
def test_batch_invariant_topk_matches_vllm_score_and_tie_order() -> None:
    _require_unified_kernel()

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
        torch.ops.vllm_batch_invariant.top_k_per_row_prefill(
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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("width", [4095, 4096, 4097, 262144])
@torch.inference_mode()
def test_batch_invariant_topk_has_no_packed_width_limit(width: int) -> None:
    _require_unified_kernel()

    top_k = 64
    logits = torch.zeros((2, width), dtype=torch.float32, device="cuda")
    # Exercise a short offset row as well as a row spanning the full packed
    # width. Equal scores make the expected descending local-index tie break
    # exact and cheap to construct even for a 256K context.
    row_starts = torch.tensor([17, 0], dtype=torch.int32, device="cuda")
    row_ends = torch.tensor([min(width, 529), width], dtype=torch.int32, device="cuda")
    indices = torch.empty((2, top_k), dtype=torch.int32, device="cuda")

    torch.ops.vllm_batch_invariant.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        2,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )

    lengths = row_ends - row_starts
    expected = (
        lengths[:, None] - 1 - torch.arange(top_k, dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_batch_invariant_topk_matches_teacher_forcing_across_packing() -> None:
    _require_unified_kernel()

    row_length = 3500
    top_k = 512
    # Quantized buckets intentionally create many ties, making source-index
    # ordering part of the train/inference contract rather than an accident of
    # launch order.
    logical_scores = ((torch.arange(row_length) * 37) % 257).float()
    expected = torch.tensor(
        sorted(
            range(row_length),
            key=lambda index: (float(logical_scores[index]), index),
            reverse=True,
        )[:top_k],
        dtype=torch.int32,
        device="cuda",
    ).unsqueeze(0)

    teacher_logits = torch.full((1, 4095), -torch.inf, device="cuda")
    teacher_logits[0, 17 : 17 + row_length] = logical_scores.cuda()
    rollout_logits = torch.full((1, 6144), -torch.inf, device="cuda")
    rollout_logits[0, 513 : 513 + row_length] = logical_scores.cuda()

    teacher_indices = _run_topk(teacher_logits, 17, row_length, top_k)
    rollout_indices = _run_topk(rollout_logits, 513, row_length, top_k)

    torch.testing.assert_close(teacher_indices, expected, rtol=0, atol=0)
    torch.testing.assert_close(rollout_indices, teacher_indices, rtol=0, atol=0)
