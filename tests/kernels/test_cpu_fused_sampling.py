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
    def test_distribution_chi_squared(self):
        """Verify sampling distribution via chi-squared goodness of fit."""
        # Small support is enough for chi-squared power; large-vocab bias
        # is covered by test_noise_not_sliding_window (#59786).
        small_vocab = 100
        logits = torch.randn(1, small_vocab, dtype=torch.float32)
        probs = logits.softmax(dim=-1).squeeze(0)

        n_samples = 100_000
        seeds = torch.arange(n_samples, dtype=torch.long) * 7 + 13
        idx = torch.ops._C.fused_gumbel_argmax(
            logits.expand(n_samples, -1).contiguous(), seeds
        )
        counts = torch.bincount(idx, minlength=small_vocab).float()

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

    def test_noise_not_sliding_window(self):
        """Table[(seed+i)&MASK] made consecutive seeds a 1-token shift.

        On uniform logits that yields P(w(s+1) == w(s)-1) ~ 1-2/V. Independent
        per-token noise yields ~1/V. See #59786.
        """
        vocab, n = 128, 2048
        logits = torch.zeros(n, vocab, dtype=torch.float32)
        seeds = torch.arange(n, dtype=torch.long)
        winners = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        shifted = (winners[:-1] - 1) % vocab
        frac = (winners[1:] == shifted).float().mean().item()
        assert frac < 0.05, (
            f"consecutive-seed winners follow a table shift (frac={frac:.3f})"
        )

    def test_zipf_head_matches_softmax(self):
        """Zipf head probability should match softmax (#59786)."""
        vocab, n_samples, batch = 8192, 16384, 256
        logits = -1.1 * torch.arange(1, vocab + 1, dtype=torch.float32).log()
        p0 = torch.softmax(logits.double(), dim=-1)[0].item()
        row = logits.view(1, -1).expand(batch, vocab).contiguous()
        n_top = 0
        for start in range(0, n_samples, batch):
            seeds = torch.arange(start, start + batch, dtype=torch.long)
            tokens = torch.ops._C.fused_gumbel_argmax(row, seeds)
            n_top += int((tokens == 0).sum())
        q0 = n_top / n_samples
        se = (p0 * (1.0 - p0) / n_samples) ** 0.5
        assert abs(q0 - p0) < 8.0 * se, (
            f"P(top-1) kernel={q0:.4f} exact={p0:.4f} se={se:.4f}"
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

    def test_samples_only_from_finite_logits(self):
        """top-k/top-p mask with -inf; those IDs must never win."""
        allowed = torch.tensor([3, 17, 99, 1024, 8191], dtype=torch.long)
        logits = torch.full((64, 32000), float("-inf"), dtype=torch.float32)
        logits[:, allowed] = torch.randn(64, allowed.numel())
        tokens = torch.ops._C.fused_gumbel_argmax(
            logits, torch.arange(64, dtype=torch.long)
        )
        assert torch.isin(tokens, allowed).all()

    def test_single_unmasked_token(self):
        logits = torch.full((8, 4096), float("-inf"), dtype=torch.float32)
        logits[:, 123] = 0.0
        seeds = torch.tensor([0, -1, -(2**63), 2**63 - 1, 42, 99, 1000, 7])
        tokens = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        assert (tokens == 123).all()

    def test_tokens_beyond_gumbel_max_cannot_win(self):
        """A gap larger than GUMBEL_MAX is unreachable."""
        logits = torch.zeros(256, 4, dtype=torch.float32)
        logits[:, 1:] = -80.0
        tokens = torch.ops._C.fused_gumbel_argmax(
            logits, torch.arange(256, dtype=torch.long)
        )
        assert (tokens == 0).all()

    def test_tied_huge_logits_not_index_biased(self):
        """Noise must still break ties when logits sit at float32 max."""
        logits = torch.full(
            (4096, 2), torch.finfo(torch.float32).max, dtype=torch.float32
        )
        tokens = torch.ops._C.fused_gumbel_argmax(
            logits, torch.arange(4096, dtype=torch.long)
        )
        frac0 = (tokens == 0).float().mean().item()
        assert set(tokens.tolist()) == {0, 1}, tokens.unique().tolist()
        assert 0.35 < frac0 < 0.65, f"P(index 0)={frac0:.3f} expected ~0.5"

    def test_serial_batch1_matches_batched(self):
        """OpenMP rows must match the batch=1 serial path."""
        logits = torch.randn(8, 2048, dtype=torch.float32)
        seeds = torch.arange(8, dtype=torch.long)
        batched = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        serial = torch.stack(
            [
                torch.ops._C.fused_gumbel_argmax(logits[i : i + 1], seeds[i : i + 1])[0]
                for i in range(8)
            ]
        )
        torch.testing.assert_close(batched, serial)

    def test_noncontiguous_matches_contiguous(self):
        logits = torch.randn(8, 1024, dtype=torch.float32)[:, ::2]
        seeds = torch.arange(16, dtype=torch.long)[::2]
        assert not logits.is_contiguous() and not seeds.is_contiguous()
        expected = torch.ops._C.fused_gumbel_argmax(
            logits.contiguous(), seeds.contiguous()
        )
        actual = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        torch.testing.assert_close(actual, expected)

    def test_empty_batch(self):
        tokens = torch.ops._C.fused_gumbel_argmax(
            torch.empty(0, 7), torch.empty(0, dtype=torch.long)
        )
        assert tokens.shape == (0,) and tokens.dtype == torch.int64

    def test_masked_support_matches_softmax(self):
        """Finite subset after -inf mask should match softmax."""
        from scipy.stats import chisquare

        logits = torch.tensor(
            [0.0, -0.2, -0.7, -1.5, -3.0, float("-inf")], dtype=torch.float32
        )
        n_samples = 20_000
        tokens = torch.ops._C.fused_gumbel_argmax(
            logits.expand(n_samples, -1).contiguous(),
            torch.arange(n_samples, dtype=torch.long),
        )
        counts = torch.bincount(tokens, minlength=logits.numel()).double()
        assert counts[-1] == 0
        probs = logits.double().softmax(-1)
        _, p_value = chisquare(counts[:-1].numpy(), (probs[:-1] * n_samples).numpy())
        assert p_value > 1e-7, f"masked-support p-value: {p_value}"

    def test_zipf_tail_matches_softmax(self):
        """Independent 53-bit Gumbel must not starve the Zipf tail (#59786)."""
        vocab, n_samples, batch = 8192, 16384, 256
        logits = -1.1 * torch.arange(1, vocab + 1, dtype=torch.float32).log()
        p_tail = torch.softmax(logits.double(), dim=-1)[vocab // 2 :].sum().item()
        row = logits.view(1, -1).expand(batch, vocab).contiguous()
        n_tail = 0
        for start in range(0, n_samples, batch):
            seeds = torch.arange(start, start + batch, dtype=torch.long)
            tokens = torch.ops._C.fused_gumbel_argmax(row, seeds)
            n_tail += int((tokens >= vocab // 2).sum())
        q_tail = n_tail / n_samples
        se = (p_tail * (1.0 - p_tail) / n_samples) ** 0.5
        assert abs(q_tail - p_tail) < 8.0 * se, (
            f"P(tail) kernel={q_tail:.4f} exact={p_tail:.4f} se={se:.4f}"
        )


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
