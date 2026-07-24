# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for GHSA-wr27-mx79-rg6c: out-of-range allowed_token_ids
must not corrupt adjacent logits rows via the Triton _bias_kernel."""

import pytest
import torch

from vllm.platforms import current_platform

VOCAB_SIZE = 128
NUM_ROWS = 2
MAX_ALLOWED = 4


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Triton kernel requires compute capability >= 8.0",
)
def test_oob_allowed_token_id_no_cross_row_corruption():
    """An out-of-range allowed_token_id (== vocab_size) must not alter the
    adjacent row's logits. The kernel's id_mask should prevent any
    load/store at the invalid index."""
    from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias

    device = torch.device(f"{current_platform.device_type}:0")

    logits = torch.randn(NUM_ROWS, VOCAB_SIZE, dtype=torch.float32, device=device)
    row1_original = logits[1].clone()

    expanded_idx_mapping = torch.arange(NUM_ROWS, dtype=torch.int32, device=device)
    pos = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)

    num_allowed = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    num_allowed[0] = 1

    allowed_ids = torch.zeros(NUM_ROWS, MAX_ALLOWED, dtype=torch.int32, device=device)
    # Row 0: one-past-end ID (== vocab_size), which would index into row 1
    allowed_ids[0, 0] = VOCAB_SIZE

    num_logit_bias = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    logit_bias_token_ids = torch.zeros(
        NUM_ROWS, MAX_ALLOWED, dtype=torch.int32, device=device
    )
    logit_bias_values = torch.zeros(
        NUM_ROWS, MAX_ALLOWED, dtype=torch.float32, device=device
    )

    min_lens = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    num_stop = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    stop_ids = torch.zeros(NUM_ROWS, MAX_ALLOWED, dtype=torch.int32, device=device)

    apply_logit_bias(
        logits,
        expanded_idx_mapping,
        pos,
        num_allowed,
        allowed_ids,
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias_values,
        min_lens,
        num_stop,
        stop_ids,
    )

    # Row 0 should be all -inf (no valid allowed IDs after bounds check)
    assert torch.all(logits[0] == float("-inf")), (
        "Row 0 should be all -inf since the only allowed ID is out of range"
    )

    # Row 1 must be completely untouched
    assert torch.equal(logits[1], row1_original), (
        "Row 1 was corrupted by row 0's out-of-range allowed_token_id"
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Triton kernel requires compute capability >= 8.0",
)
def test_valid_allowed_token_ids_still_work():
    """Sanity check: valid allowed_token_ids correctly restrict output."""
    from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias

    device = torch.device(f"{current_platform.device_type}:0")

    logits = torch.randn(NUM_ROWS, VOCAB_SIZE, dtype=torch.float32, device=device)
    original_val_row0_token5 = logits[0, 5].item()

    expanded_idx_mapping = torch.arange(NUM_ROWS, dtype=torch.int32, device=device)
    pos = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)

    num_allowed = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    num_allowed[0] = 1

    allowed_ids = torch.zeros(NUM_ROWS, MAX_ALLOWED, dtype=torch.int32, device=device)
    allowed_ids[0, 0] = 5  # Valid token ID

    num_logit_bias = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    logit_bias_token_ids = torch.zeros(
        NUM_ROWS, MAX_ALLOWED, dtype=torch.int32, device=device
    )
    logit_bias_values = torch.zeros(
        NUM_ROWS, MAX_ALLOWED, dtype=torch.float32, device=device
    )

    min_lens = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    num_stop = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    stop_ids = torch.zeros(NUM_ROWS, MAX_ALLOWED, dtype=torch.int32, device=device)

    apply_logit_bias(
        logits,
        expanded_idx_mapping,
        pos,
        num_allowed,
        allowed_ids,
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias_values,
        min_lens,
        num_stop,
        stop_ids,
    )

    # Token 5 should be preserved, all others should be -inf
    assert logits[0, 5].item() == pytest.approx(original_val_row0_token5), (
        "Valid allowed token's logit was not preserved"
    )
    for i in range(VOCAB_SIZE):
        if i != 5:
            assert logits[0, i].item() == float("-inf"), (
                f"Token {i} should be -inf but got {logits[0, i].item()}"
            )
