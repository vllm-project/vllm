# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused row statistics of the DiffusionGemma sampler against the PyTorch
reference: argmax, entropy and softmax must match. A zero temperature must
sample the argmax and a positive one must sample from the distribution."""

import math

import pytest
import torch

from vllm.model_executor.models.diffusion_gemma_sampler import (
    sample_row_stats,
    sample_row_stats_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused kernel needs a GPU"
)


@pytest.mark.parametrize("vocab", [5000, 9001])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stats_match_reference(vocab: int, dtype: torch.dtype):
    torch.manual_seed(0)
    num_decode, cl = 9, 4
    logits = (torch.randn(num_decode * cl, vocab, device="cuda") * 3).to(dtype)
    temps = torch.tensor([0.0, 0.7, 2.0, 1.0, 0.0, 0.3, 1.5, 1.0, 0.0], device="cuda")
    argmax, sample, entropy, probs = sample_row_stats(
        logits, temps, cl, seed=1234, probs_dtype=torch.bfloat16
    )
    ref_argmax, _, ref_entropy, ref_probs = sample_row_stats_reference(
        logits, temps, cl, probs_dtype=torch.bfloat16
    )
    assert torch.equal(argmax, ref_argmax)
    torch.testing.assert_close(entropy, ref_entropy, atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(probs.float(), ref_probs.float(), atol=5e-3, rtol=2e-2)
    assert probs.dtype == torch.bfloat16
    assert probs.shape == logits.shape
    # A padded (all-zero) row is uniform: max entropy, argmax 0.
    zero = torch.zeros(1, vocab, device="cuda")
    a0, _, e0, _ = sample_row_stats(zero, temps[:1] + 1.0, 1, 1, None)
    assert a0.item() == 0
    assert abs(e0.item() - torch.log(torch.tensor(float(vocab))).item()) < 1e-3


def test_zero_temperature_is_greedy_and_probs_optional():
    torch.manual_seed(0)
    logits = torch.randn(12, 4097, device="cuda")
    temps = torch.zeros(12, device="cuda")
    argmax, sample, entropy, probs = sample_row_stats(logits, temps, 1, 7, None)
    assert torch.equal(sample, argmax)
    assert probs is None
    assert torch.equal(argmax, logits.argmax(dim=-1))
    # Greedy entropy is that of the reference's clamped temperature.
    _, _, ref_entropy, _ = sample_row_stats_reference(logits, temps, 1, None)
    torch.testing.assert_close(entropy, ref_entropy, atol=2e-3, rtol=1e-3)


def test_positive_temperature_samples_from_the_distribution():
    torch.manual_seed(0)
    rows, vocab = 2000, 4099
    # A peaked row samples its argmax almost always and a flat row rarely.
    peaked = torch.zeros(rows, vocab, device="cuda")
    peaked[:, 17] = 16.0  # p(17) = e^16 / (e^16 + 4098) > 0.999
    flat = torch.zeros(rows, vocab, device="cuda")
    temps = torch.ones(rows, device="cuda")
    _, s_peaked, _, _ = sample_row_stats(peaked, temps, 1, 99, None)
    _, s_flat, _, _ = sample_row_stats(flat, temps, 1, 99, None)
    assert (s_peaked == 17).float().mean().item() > 0.99
    assert s_flat.unique().numel() > rows // 2
    # The same seed repeats the draws and another seed changes them.
    _, s_again, _, _ = sample_row_stats(flat, temps, 1, 99, None)
    _, s_other, _, _ = sample_row_stats(flat, temps, 1, 100, None)
    assert torch.equal(s_flat, s_again)
    assert not torch.equal(s_flat, s_other)


def test_empty_batch():
    logits = torch.empty(0, 128, device="cuda")
    temps = torch.empty(0, device="cuda")
    argmax, sample, entropy, probs = sample_row_stats(
        logits, temps, 1, 1, torch.bfloat16
    )
    assert argmax.numel() == sample.numel() == entropy.numel() == 0
    assert probs is not None and probs.shape == (0, 128)


def test_masked_logits_keep_a_finite_entropy():
    """top_k/top_p leave -inf logits. Masked columns gave NaN entropy, and a
    row with one live column has zero entropy."""
    logits = torch.full((4, 5000), float("-inf"), device="cuda")
    logits[:, 3] = 0.0
    logits[2, 9] = 0.0  # two live columns: entropy log 2
    temps = torch.tensor([1.0, 0.0, 1.0, 0.7], device="cuda")
    argmax, sample, entropy, probs = sample_row_stats(
        logits, temps, 1, 5, torch.float32
    )
    assert torch.isfinite(entropy).all()
    assert entropy[0].item() < 1e-6 and entropy[1].item() < 1e-6
    assert abs(entropy[2].item() - math.log(2.0)) < 1e-4
    assert torch.equal(argmax[[0, 1, 3]], torch.tensor([3, 3, 3], device="cuda"))
    assert probs[0, 3].item() == 1.0 and probs[0].sum().item() == 1.0
    _, _, ref_entropy, _ = sample_row_stats_reference(logits, temps, 1, None)
    assert torch.isfinite(ref_entropy).all()
