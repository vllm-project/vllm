# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the min_p Triton sampling kernel."""

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.min_p import apply_min_p

DEVICE = current_platform.device_type


def _apply_min_p_reference(
    logits: torch.Tensor,
    min_p: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of min_p filtering."""
    out = logits.clone()
    for token_idx in range(logits.shape[0]):
        req_idx = expanded_idx_mapping[token_idx].item()
        p = min_p[req_idx].item()
        if p == 0.0:
            continue
        row = out[token_idx]
        max_val = row.max().item()
        threshold = max_val + math.log(p)
        row[row < threshold] = float("-inf")
    return out


def _make_inputs(
    logits: torch.Tensor,
    min_p_values: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper to create kernel inputs from logits and per-token min_p values.

    Assumes a 1:1 mapping between tokens and requests (each token belongs
    to a separate request).
    """
    num_tokens = logits.shape[0]
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    min_p = torch.tensor(min_p_values, dtype=torch.float32, device=DEVICE)
    return logits.to(DEVICE), expanded_idx_mapping, min_p


@pytest.mark.parametrize("vocab_size", [128, 1024, 32000])
def test_min_p_basic(vocab_size: int):
    """Tokens with probability below min_p * max_prob are masked to -inf."""
    num_tokens = 4
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32)
    min_p_values = [0.1, 0.05, 0.5, 0.2]
    logits_gpu, idx_map, min_p = _make_inputs(logits, min_p_values)

    expected = _apply_min_p_reference(logits.to(DEVICE), min_p, idx_map)
    apply_min_p(logits_gpu, idx_map, min_p)

    torch.testing.assert_close(logits_gpu, expected)


def test_min_p_zero_is_noop():
    """min_p=0.0 should leave logits completely unchanged."""
    vocab_size = 512
    logits = torch.randn(2, vocab_size, dtype=torch.float32)
    original = logits.clone()
    logits_gpu, idx_map, min_p = _make_inputs(logits, [0.0, 0.0])

    apply_min_p(logits_gpu, idx_map, min_p)

    torch.testing.assert_close(logits_gpu, original.to(DEVICE))


def test_min_p_one_keeps_only_max():
    """min_p=1.0 should only keep the maximum logit(s)."""
    vocab_size = 256
    logits = torch.randn(1, vocab_size, dtype=torch.float32)
    logits_gpu, idx_map, min_p = _make_inputs(logits, [1.0])

    apply_min_p(logits_gpu, idx_map, min_p)

    max_val = logits.max().item()
    result = logits_gpu.cpu()
    for i in range(vocab_size):
        if logits[0, i].item() == max_val:
            assert result[0, i].item() == max_val
        else:
            assert result[0, i].item() == float("-inf")


def test_min_p_all_equal_logits():
    """When all logits are equal, none should be filtered regardless of min_p.

    threshold = max_val + log(min_p). For equal logits, all values equal
    max_val, so they all pass the threshold (assuming min_p <= 1.0).
    """
    vocab_size = 128
    logits = torch.full((1, vocab_size), 2.0, dtype=torch.float32)
    logits_gpu, idx_map, min_p = _make_inputs(logits, [0.5])

    apply_min_p(logits_gpu, idx_map, min_p)

    # All logits are equal to max, so all should survive
    assert (logits_gpu == 2.0).all()


def test_min_p_shared_request_mapping():
    """Multiple tokens can map to the same request's min_p value.

    This tests the expanded_idx_mapping indirection.
    """
    vocab_size = 256
    num_tokens = 4
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32)
    logits_gpu = logits.clone().to(DEVICE)

    # Tokens 0,1 -> request 0 (min_p=0.1), tokens 2,3 -> request 1 (min_p=0.5)
    expanded_idx_mapping = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=DEVICE)
    min_p = torch.tensor([0.1, 0.5], dtype=torch.float32, device=DEVICE)

    expected = _apply_min_p_reference(logits.to(DEVICE), min_p, expanded_idx_mapping)
    apply_min_p(logits_gpu, expanded_idx_mapping, min_p)

    torch.testing.assert_close(logits_gpu, expected)


def test_min_p_mixed_zero_and_nonzero():
    """Some requests use min_p=0 (noop) while others use nonzero values."""
    vocab_size = 512
    logits = torch.randn(3, vocab_size, dtype=torch.float32)
    min_p_values = [0.0, 0.3, 0.0]
    logits_gpu, idx_map, min_p = _make_inputs(logits, min_p_values)

    expected = _apply_min_p_reference(logits.to(DEVICE), min_p, idx_map)
    apply_min_p(logits_gpu, idx_map, min_p)

    torch.testing.assert_close(logits_gpu, expected)


def test_min_p_already_neg_inf():
    """Logits that are already -inf should remain -inf."""
    vocab_size = 128
    logits = torch.randn(1, vocab_size, dtype=torch.float32)
    logits[0, :10] = float("-inf")
    logits_gpu, idx_map, min_p = _make_inputs(logits, [0.1])

    expected = _apply_min_p_reference(logits.to(DEVICE), min_p, idx_map)
    apply_min_p(logits_gpu, idx_map, min_p)

    torch.testing.assert_close(logits_gpu, expected)


@pytest.mark.parametrize("min_p_val", [0.01, 0.1, 0.5, 0.9])
def test_min_p_threshold_correctness(min_p_val: float):
    """Verify the exact threshold: tokens survive iff logit >= max + log(min_p)."""
    vocab_size = 64
    logits = torch.randn(1, vocab_size, dtype=torch.float32)
    logits_gpu, idx_map, min_p = _make_inputs(logits, [min_p_val])

    apply_min_p(logits_gpu, idx_map, min_p)

    max_val = logits.max().item()
    threshold = max_val + math.log(min_p_val)
    result = logits_gpu.cpu()
    for i in range(vocab_size):
        if logits[0, i].item() >= threshold:
            assert result[0, i].item() == logits[0, i].item(), (
                f"Token {i} should have survived (logit={logits[0, i].item()}, "
                f"threshold={threshold})"
            )
        else:
            assert result[0, i].item() == float("-inf"), (
                f"Token {i} should have been filtered (logit={logits[0, i].item()}, "
                f"threshold={threshold})"
            )
