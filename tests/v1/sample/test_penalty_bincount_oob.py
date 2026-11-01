# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for OOB prompt-presence bitset write in penalty bincount.

An out-of-range token ID (>= vocab_size) in processed prompts must not
corrupt the adjacent request's penalty state row."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

VOCAB_SIZE = 128
NUM_ROWS = 2
MAX_MODEL_LEN = 64


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Triton kernel requires compute capability >= 8.0",
)
def test_oob_prompt_token_no_cross_row_bitset_corruption():
    """Token ID == vocab_size must not write to the adjacent request's
    prompt_bin_mask row."""
    from vllm.v1.worker.gpu.sample.penalties import bincount

    device = torch.device(f"{current_platform.device_type}:0")

    num_words = cdiv(VOCAB_SIZE, 32)
    prompt_bin_mask = torch.zeros(NUM_ROWS, num_words, dtype=torch.int32, device=device)
    output_bin_counts = torch.zeros(
        NUM_ROWS, VOCAB_SIZE, dtype=torch.int32, device=device
    )

    all_token_ids = torch.zeros(
        NUM_ROWS, MAX_MODEL_LEN, dtype=torch.int32, device=device
    )
    # Row 0: prompt with valid token (5) and OOB token (== vocab_size)
    all_token_ids[0, 0] = 5
    all_token_ids[0, 1] = VOCAB_SIZE  # OOB: one-past-end

    prompt_len = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    prompt_len[0] = 2  # 2 prompt tokens for row 0

    prefill_len = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    prefill_len[0] = 2  # same as prompt_len (no output tokens in prefill)

    expanded_idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)

    bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        max_prefill_len=2,
    )

    # Row 0: valid token 5 should be set (word 0, bit 5)
    assert (prompt_bin_mask[0, 0].item() & (1 << 5)) != 0, (
        "Valid token 5 was not recorded in row 0's prompt_bin_mask"
    )

    # Row 1: must be completely untouched
    assert torch.all(prompt_bin_mask[1] == 0), (
        "Row 1's prompt_bin_mask was corrupted by row 0's OOB token"
    )
    assert torch.all(output_bin_counts[1] == 0), (
        "Row 1's output_bin_counts was corrupted"
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Triton kernel requires compute capability >= 8.0",
)
def test_oob_output_token_no_cross_row_counts_corruption():
    """Token ID == vocab_size in the output portion must not write to
    the adjacent request's output_bin_counts row."""
    from vllm.v1.worker.gpu.sample.penalties import bincount

    device = torch.device(f"{current_platform.device_type}:0")

    num_words = cdiv(VOCAB_SIZE, 32)
    prompt_bin_mask = torch.zeros(NUM_ROWS, num_words, dtype=torch.int32, device=device)
    output_bin_counts = torch.zeros(
        NUM_ROWS, VOCAB_SIZE, dtype=torch.int32, device=device
    )

    all_token_ids = torch.zeros(
        NUM_ROWS, MAX_MODEL_LEN, dtype=torch.int32, device=device
    )
    # Row 0: 1 prompt token (valid), then 1 output token (OOB)
    all_token_ids[0, 0] = 10  # prompt token (valid)
    all_token_ids[0, 1] = VOCAB_SIZE  # output token (OOB)

    prompt_len = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    prompt_len[0] = 1

    prefill_len = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    prefill_len[0] = 2  # includes 1 output token

    expanded_idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)

    bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        max_prefill_len=2,
    )

    # Row 0: prompt token 10 should be set
    assert (prompt_bin_mask[0, 0].item() & (1 << 10)) != 0, (
        "Valid prompt token 10 was not recorded"
    )

    # Row 0: OOB output token should NOT be counted
    # (If it were written at index vocab_size, it would be in row 1's space)
    assert output_bin_counts[0].sum().item() == 0, (
        "OOB output token should not be counted in row 0"
    )

    # Row 1: must be completely untouched
    assert torch.all(output_bin_counts[1] == 0), (
        "Row 1's output_bin_counts was corrupted by row 0's OOB output token"
    )
    assert torch.all(prompt_bin_mask[1] == 0), "Row 1's prompt_bin_mask was corrupted"


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Triton kernel requires compute capability >= 8.0",
)
def test_boundary_valid_token_correctly_recorded():
    """Token ID == vocab_size - 1 (the last valid token) should be
    correctly recorded without any issues."""
    from vllm.v1.worker.gpu.sample.penalties import bincount

    device = torch.device(f"{current_platform.device_type}:0")

    num_words = cdiv(VOCAB_SIZE, 32)
    prompt_bin_mask = torch.zeros(NUM_ROWS, num_words, dtype=torch.int32, device=device)
    output_bin_counts = torch.zeros(
        NUM_ROWS, VOCAB_SIZE, dtype=torch.int32, device=device
    )

    all_token_ids = torch.zeros(
        NUM_ROWS, MAX_MODEL_LEN, dtype=torch.int32, device=device
    )
    # Row 0: prompt with the last valid token
    last_valid = VOCAB_SIZE - 1  # 127
    all_token_ids[0, 0] = last_valid

    prompt_len = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    prompt_len[0] = 1

    prefill_len = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    prefill_len[0] = 1

    expanded_idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)

    bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        max_prefill_len=1,
    )

    # Last valid token should be set: word = 127//32 = 3, bit = 127%32 = 31
    expected_word = last_valid // 32
    expected_bit = last_valid % 32
    assert (prompt_bin_mask[0, expected_word].item() & (1 << expected_bit)) != 0, (
        f"Last valid token {last_valid} was not recorded in prompt_bin_mask"
    )

    # Row 1: must be untouched
    assert torch.all(prompt_bin_mask[1] == 0), (
        "Row 1 was modified when only row 0 was processed"
    )
