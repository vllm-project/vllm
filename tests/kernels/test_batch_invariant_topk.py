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
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    combine_topk_swa_indices,
    fill_c128_topk,
    zero_invalid_lens,
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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("top_k", [0, 128, 129, 512])
@torch.inference_mode()
def test_fused_c4_decode_indices_match_existing_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    top_k: int,
) -> None:
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    _require_unified_kernel()

    rows = 32
    window_size = 128
    compress_ratio = 4
    compressed_width = 4096
    row_stride = compressed_width + window_size
    output_width = ((top_k + window_size + 127) // 128) * 128
    boundary_lens = torch.tensor(
        [0, 1, 3, 4, 127, 128, 511, 512, 2047, 2048, 2049, 16048],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = boundary_lens.repeat(3)[:rows]
    query_start_loc = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    gather_lens = seq_lens.clamp_max(window_size)
    topk_indices = torch.full((rows, top_k), -1, dtype=torch.int32, device="cuda")
    for row, seq_len in enumerate(seq_lens.cpu().tolist()):
        length = min(seq_len // compress_ratio, top_k)
        if length:
            topk_indices[row, :length] = torch.randperm(
                length, dtype=torch.int32, device="cuda"
            )
    is_valid = torch.arange(rows, device="cuda") % 7 != 0

    expected_indices = torch.empty(
        (rows, output_width), dtype=torch.int32, device="cuda"
    )
    expected_lens = torch.empty(rows, dtype=torch.int32, device="cuda")
    combine_topk_swa_indices(
        topk_indices.clone(),
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        top_k,
        row_stride,
        compressed_width,
        out=(expected_indices, expected_lens),
    )
    zero_invalid_lens(expected_lens, is_valid)

    actual_indices = torch.empty_like(expected_indices)
    actual_lens = torch.empty_like(expected_lens)
    torch.ops.vllm_batch_invariant.combine_topk_swa_decode(
        actual_indices,
        actual_lens,
        topk_indices,
        seq_lens,
        is_valid,
        row_stride,
        compressed_width,
        top_k,
        compress_ratio,
        window_size,
    )
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_lens, expected_lens, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_fused_c128_decode_indices_match_existing_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    _require_unified_kernel()

    rows = 32
    top_k = 128
    window_size = 128
    compress_ratio = 128
    compressed_width = 126
    row_stride = compressed_width + window_size
    output_width = 256
    boundary_lens = torch.tensor(
        [0, 1, 127, 128, 129, 255, 256, 2047, 2048, 2049, 16047, 16048],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = boundary_lens.repeat(3)[:rows]
    compressed_lens = torch.div(
        seq_lens, compress_ratio, rounding_mode="floor"
    ).clamp_max(top_k)
    query_start_loc = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    gather_lens = seq_lens.clamp_max(window_size)
    is_valid = torch.arange(rows, device="cuda") % 7 != 0

    local_topk = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    fill_c128_topk(local_topk, compressed_lens)
    expected_indices = torch.empty(
        (rows, output_width), dtype=torch.int32, device="cuda"
    )
    expected_lens = torch.empty(rows, dtype=torch.int32, device="cuda")
    combine_topk_swa_indices(
        local_topk,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        top_k,
        row_stride,
        compressed_width,
        out=(expected_indices, expected_lens),
    )
    zero_invalid_lens(expected_lens, is_valid)

    actual_indices = torch.empty_like(expected_indices)
    actual_lens = torch.empty_like(expected_lens)
    torch.ops.vllm_batch_invariant.combine_c128_swa_decode(
        actual_indices,
        actual_lens,
        seq_lens,
        is_valid,
        row_stride,
        compressed_width,
        top_k,
        compress_ratio,
        window_size,
    )
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_lens, expected_lens, rtol=0, atol=0)
