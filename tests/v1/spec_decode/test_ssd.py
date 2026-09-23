# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SSD (speculative speculative decoding) core algorithms."""

import math

import pytest
import torch

from vllm.v1.spec_decode.ssd import (
    SpeculationCache,
    compute_geometric_fanout,
    predict_bonus_tokens,
    saguaro_adjust_logits,
)


class TestGeometricFanout:
    def test_respects_budget_and_length(self):
        fanout = compute_geometric_fanout(
            budget=32, acceptance_rate=0.7, num_speculative_tokens=5
        )
        assert len(fanout) == 6
        assert sum(fanout) <= 32
        assert all(f >= 1 for f in fanout)

    def test_decreasing_over_positions(self):
        # Geometric decay: earlier accept-lengths are more likely, so they
        # get more bonus-token guesses (last position may be boosted).
        fanout = compute_geometric_fanout(
            budget=64, acceptance_rate=0.6, num_speculative_tokens=4
        )
        assert all(fanout[i] >= fanout[i + 1] for i in range(len(fanout) - 2))

    def test_last_position_boosted_at_high_acceptance(self):
        # With high acceptance rate, all-accepted is the most likely outcome
        # so F_K gets the (1 - a)^(-1/(1+r)) boost.
        fanout = compute_geometric_fanout(
            budget=64, acceptance_rate=0.9, num_speculative_tokens=4
        )
        assert fanout[-1] > fanout[-2]

    def test_budget_smaller_than_positions(self):
        fanout = compute_geometric_fanout(
            budget=3, acceptance_rate=0.7, num_speculative_tokens=5
        )
        assert len(fanout) == 6
        assert sum(fanout) == 3
        assert all(f in (0, 1) for f in fanout)

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            compute_geometric_fanout(0, 0.7, 5)
        with pytest.raises(ValueError):
            compute_geometric_fanout(8, 0.7, 0)
        with pytest.raises(ValueError):
            compute_geometric_fanout(8, 0.7, 5, power_law_r=0.0)


class TestSaguaroSampling:
    def test_c_equal_one_is_noop(self):
        logits = torch.randn(4, 32)
        out = saguaro_adjust_logits(logits, fanout=4, downweight_c=1.0)
        assert torch.equal(out, logits)

    def test_downweights_only_top_f(self):
        logits = torch.randn(2, 64)
        c = 0.25
        out = saguaro_adjust_logits(logits, fanout=8, downweight_c=c)
        top = logits.topk(8, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, top, True)
        assert torch.allclose(out[mask], logits[mask] + math.log(c))
        assert torch.equal(out[~mask], logits[~mask])

    def test_c_zero_masks_top_f(self):
        logits = torch.randn(3, 16)
        out = saguaro_adjust_logits(logits, fanout=2, downweight_c=0.0)
        top = logits.topk(2, dim=-1).indices
        assert torch.isinf(out.gather(-1, top)).all()

    def test_residual_mass_moves_to_top_f(self):
        # Theorem 15: as C -> 0 the residual distribution concentrates on
        # the cached (top-F) tokens, increasing the cache hit rate.
        torch.manual_seed(0)
        target_probs = torch.softmax(torch.randn(128), dim=-1)
        draft_logits = torch.randn(128)
        fanout = 8
        top = draft_logits.topk(fanout).indices

        def residual_mass_on_top(c: float) -> float:
            draft_probs = torch.softmax(
                saguaro_adjust_logits(draft_logits, fanout, c), dim=-1
            )
            residual = (target_probs - draft_probs).clamp(min=0)
            residual /= residual.sum()
            return residual[top].sum().item()

        assert residual_mass_on_top(0.1) >= residual_mass_on_top(1.0)

    def test_invalid_c(self):
        with pytest.raises(ValueError):
            saguaro_adjust_logits(torch.randn(8), fanout=2, downweight_c=1.5)


class TestPredictBonusTokens:
    def test_excludes_drafted_token(self):
        vocab = 16
        logits = torch.zeros(3, vocab)
        # Make token 5 the argmax everywhere; it is also the drafted token
        # at every position, so it must never be predicted as bonus.
        logits[:, 5] = 10.0
        logits[:, 7] = 5.0
        draft_tokens = torch.tensor([5, 5, 5])
        predictions = predict_bonus_tokens(logits, draft_tokens, [1, 1, 1])
        assert predictions == [[7], [7], [7]]

    def test_no_exclusion_with_negative_token(self):
        logits = torch.zeros(1, 8)
        logits[0, 3] = 10.0
        predictions = predict_bonus_tokens(logits, torch.tensor([-1]), [1])
        assert predictions == [[3]]

    def test_fanout_sizes(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 64)
        draft_tokens = torch.tensor([0, 1, 2, -1])
        fanout = [3, 2, 1, 4]
        predictions = predict_bonus_tokens(logits, draft_tokens, fanout)
        assert [len(p) for p in predictions] == fanout
        for k, guesses in enumerate(predictions):
            assert draft_tokens[k].item() not in guesses
            assert len(set(guesses)) == len(guesses)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            predict_bonus_tokens(torch.randn(2, 8), torch.tensor([0]), [1, 1])


class TestSpeculationCache:
    def test_hit_and_miss(self):
        cache = SpeculationCache()
        cache.put(2, 17, [4, 8, 15])
        assert cache.get(2, 17) == [4, 8, 15]
        assert cache.get(2, 99) is None
        assert cache.hits == 1
        assert cache.misses == 1
        assert cache.hit_rate == 0.5

    def test_clear_keeps_stats(self):
        cache = SpeculationCache()
        cache.put(0, 1, [2])
        cache.get(0, 1)
        cache.clear()
        assert len(cache) == 0
        assert cache.hits == 1
        assert cache.get(0, 1) is None

    def test_empty_hit_rate(self):
        assert SpeculationCache().hit_rate == 0.0
