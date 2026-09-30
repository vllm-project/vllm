# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tracemalloc

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

import vllm._C  # noqa: F401, E402

VOCAB_SIZES = [32000, 49152, 128256]
BATCH_SIZES = [1, 4, 16]


def reference_greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1).view(-1)


class TestGreedyArgmax:
    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_exact_match(self, vocab_size: int, batch_size: int):
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        expected = reference_greedy_sample(logits)
        result = torch.ops._C.greedy_argmax(logits)
        torch.testing.assert_close(result, expected)

    def test_single_dominant(self):
        logits = torch.full((1, 50000), -1e9, dtype=torch.float32)
        logits[0, 42] = 100.0
        assert torch.ops._C.greedy_argmax(logits).item() == 42

    def test_negative_logits(self):
        logits = torch.randn(8, 32000, dtype=torch.float32) - 10.0
        expected = reference_greedy_sample(logits)
        result = torch.ops._C.greedy_argmax(logits)
        torch.testing.assert_close(result, expected)


class TestFusedGumbelArgmax:
    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    def test_distribution_chi_squared(self, vocab_size: int):
        """Verify sampling distribution via chi-squared goodness of fit."""
        small_vocab = min(vocab_size, 100)
        logits = torch.randn(1, small_vocab, dtype=torch.float32)
        probs = logits.softmax(dim=-1).squeeze(0)

        n_samples = 100_000
        counts = torch.zeros(small_vocab)
        for trial in range(n_samples):
            seed = torch.tensor([trial * 7 + 13], dtype=torch.long)
            tile = logits.expand(1, -1).contiguous()
            idx = torch.ops._C.fused_gumbel_argmax(tile, seed)
            counts[idx.item()] += 1

        expected = probs * n_samples
        mask = expected > 5
        chi2 = ((counts[mask] - expected[mask]) ** 2 / expected[mask]).sum()
        dof = mask.sum().item() - 1
        from scipy.stats import chi2 as chi2_dist

        p_value = 1.0 - chi2_dist.cdf(chi2.item(), dof)
        assert p_value > 0.001, (
            f"Chi-squared test failed: chi2={chi2.item():.1f}, "
            f"dof={dof}, p={p_value:.6f}"
        )

    def test_deterministic_same_seed(self):
        """Same seed produces same result."""
        logits = torch.randn(4, 32000, dtype=torch.float32)
        seeds = torch.tensor([42, 123, 456, 789], dtype=torch.long)
        r1 = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        r2 = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        torch.testing.assert_close(r1, r2)

    def test_different_seeds_differ(self):
        logits = torch.zeros(16, 50000, dtype=torch.float32)
        seeds_a = torch.arange(16, dtype=torch.long)
        seeds_b = torch.arange(16, dtype=torch.long) + 1_000_000
        r_a = torch.ops._C.fused_gumbel_argmax(logits, seeds_a)
        r_b = torch.ops._C.fused_gumbel_argmax(logits, seeds_b)
        assert not torch.equal(r_a, r_b)

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_output_in_range(self, vocab_size: int, batch_size: int):
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        seeds = torch.arange(batch_size, dtype=torch.long)
        result = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        assert result.min() >= 0
        assert result.max() < vocab_size


class TestMemory:
    def test_fused_no_intermediate_allocs(self):
        """Verify that the fused kernel does not allocate large intermediates."""
        logits = torch.randn(16, 128256, dtype=torch.float32)
        seeds = torch.arange(16, dtype=torch.long)

        torch.ops._C.fused_gumbel_argmax(logits, seeds)

        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()
        for _ in range(50):
            torch.ops._C.fused_gumbel_argmax(logits, seeds)
        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snap_after.compare_to(snap_before, "lineno")
        total_new_bytes = sum(s.size_diff for s in diff if s.size_diff > 0)
        vocab_bytes = 16 * 128256 * 4
        assert total_new_bytes < vocab_bytes, (
            f"Fused kernel allocated {total_new_bytes} bytes, "
            f"expected < {vocab_bytes} (one intermediate tensor)"
        )
