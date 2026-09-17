# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for apply_top_k_top_p_pytorch: verify the scatter step is
cudagraph-safe (out-of-place) and numerically correct (issue #42745).
"""
import torch
import pytest

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch


def _make_logits(batch: int, vocab: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, vocab)


@pytest.mark.parametrize("batch,vocab", [(1, 64), (4, 128), (8, 256)])
def test_top_k_only_does_not_modify_original(batch: int, vocab: int) -> None:
    """scatter must be out-of-place: original logits tensor is unchanged."""
    logits = _make_logits(batch, vocab)
    original = logits.clone()
    k = torch.full((batch,), vocab // 2, dtype=torch.long)
    result = apply_top_k_top_p_pytorch(logits, k=k, p=None)
    # The returned tensor should NOT be the same object as logits
    assert result.data_ptr() != logits.data_ptr(), (
        "scatter should be out-of-place: result must be a new tensor"
    )
    # And logits must be unchanged
    assert torch.equal(logits, original), (
        "apply_top_k_top_p_pytorch must not modify the input logits in-place"
    )


@pytest.mark.parametrize("batch,vocab", [(1, 64), (4, 128)])
def test_top_p_only_does_not_modify_original(batch: int, vocab: int) -> None:
    logits = _make_logits(batch, vocab, seed=1)
    original = logits.clone()
    p = torch.full((batch,), 0.9)
    result = apply_top_k_top_p_pytorch(logits, k=None, p=p)
    assert result.data_ptr() != logits.data_ptr()
    assert torch.equal(logits, original)


@pytest.mark.parametrize("batch,vocab", [(2, 128), (4, 256)])
def test_top_k_top_p_combined_does_not_modify_original(
    batch: int, vocab: int
) -> None:
    logits = _make_logits(batch, vocab, seed=2)
    original = logits.clone()
    k = torch.full((batch,), vocab // 4, dtype=torch.long)
    p = torch.full((batch,), 0.95)
    result = apply_top_k_top_p_pytorch(logits, k=k, p=p)
    assert result.data_ptr() != logits.data_ptr()
    assert torch.equal(logits, original)


@pytest.mark.parametrize("batch,vocab", [(1, 64), (4, 128)])
def test_top_k_masks_correct_tokens(batch: int, vocab: int) -> None:
    """Top-k result keeps exactly k finite values per row."""
    logits = _make_logits(batch, vocab, seed=3)
    k_val = vocab // 4
    k = torch.full((batch,), k_val, dtype=torch.long)
    result = apply_top_k_top_p_pytorch(logits, k=k, p=None)
    for row in result:
        finite_count = torch.isfinite(row).sum().item()
        assert finite_count == k_val, (
            f"Expected {k_val} finite logits, got {finite_count}"
        )


@pytest.mark.parametrize("batch,vocab", [(2, 64), (4, 128)])
def test_top_p_at_least_one_finite(batch: int, vocab: int) -> None:
    """Top-p always keeps at least one finite logit per row."""
    logits = _make_logits(batch, vocab, seed=4)
    p = torch.full((batch,), 0.01)  # very tight threshold
    result = apply_top_k_top_p_pytorch(logits, k=None, p=p)
    for row in result:
        finite_count = torch.isfinite(row).sum().item()
        assert finite_count >= 1


def test_no_op_returns_input_unchanged() -> None:
    """k=None, p=None → returns the original logits object."""
    logits = _make_logits(2, 64)
    result = apply_top_k_top_p_pytorch(logits, k=None, p=None)
    assert result.data_ptr() == logits.data_ptr()


def test_result_shape_preserved() -> None:
    """Output shape matches input shape."""
    logits = _make_logits(3, 200)
    k = torch.full((3,), 50, dtype=torch.long)
    p = torch.full((3,), 0.9)
    result = apply_top_k_top_p_pytorch(logits, k=k, p=p)
    assert result.shape == logits.shape
