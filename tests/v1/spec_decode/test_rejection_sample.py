# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from scipy import stats

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.rejection_sample import (
    sample_recovered_and_bonus_tokens,
)

device = current_platform.device_type


@pytest.mark.skipif(device != "cuda", reason="Requires CUDA")
def test_sample_recovered_and_bonus_tokens_correctness():
    """Verify that sample_recovered_and_bonus_tokens produces samples whose
    empirical distribution matches the theoretical distribution.

    For each non-bonus (recovered) token position the correct distribution is
    max(0, target - draft) renormalized; for the bonus token position it is
    target directly.  We draw N samples with independent seeds and run a
    chi-squared goodness-of-fit test against those expected distributions.
    """
    # 3 requests with 2 / 1 / 2 draft tokens each, giving 8 total token
    # req 0 (2 draft + 1 bonus): tok 0, 1, 2
    # req 1 (1 draft + 1 bonus): tok 3, 4
    # req 2 (2 draft + 1 bonus): tok 5, 6, 7
    num_reqs = 3
    num_tokens = 8
    vocab_size = 3

    target_probs = torch.tensor(
        [
            [0.6, 0.3, 0.1],  # req 0 recovered
            [0.3, 0.4, 0.3],  # req 0 recovered
            [0.1, 0.7, 0.2],  # req 0 bonus
            [0.5, 0.3, 0.2],  # req 1 recovered
            [0.2, 0.5, 0.3],  # req 1 bonus
            [0.4, 0.4, 0.2],  # req 2 recovered
            [0.2, 0.6, 0.2],  # req 2 recovered
            [0.3, 0.3, 0.4],  # req 2 bonus
        ],
        dtype=torch.float32,
        device=device,
    )
    draft_probs = torch.tensor(
        [
            [0.1, 0.2, 0.7],  # req 0 draft 0
            [0.1, 0.2, 0.7],  # req 0 draft 1
            [0.3, 0.1, 0.6],  # req 1 draft 0
            [0.1, 0.3, 0.6],  # req 2 draft 0
            [0.1, 0.2, 0.7],  # req 2 draft 1
        ],
        dtype=torch.float32,
        device=device,
    )

    # Expected distributions (recovered = max(0, target-draft) normalised):
    #   tok 0: max(0, [0.5,  0.1, -0.6]) → [5/6, 1/6,   0]
    #   tok 1: max(0, [0.2,  0.2, -0.4]) → [1/2, 1/2,   0]
    #   tok 2: bonus                      → [0.1, 0.7, 0.2]
    #   tok 3: max(0, [0.2,  0.2, -0.4]) → [1/2, 1/2,   0]
    #   tok 4: bonus                      → [0.2, 0.5, 0.3]
    #   tok 5: max(0, [0.3,  0.1, -0.4]) → [3/4, 1/4,   0]
    #   tok 6: max(0, [0.1,  0.4, -0.5]) → [1/5, 4/5,   0]
    #   tok 7: bonus                      → [0.3, 0.3, 0.4]
    expected_dists = torch.tensor(
        [
            [5 / 6, 1 / 6, 0.0],  # req 0 recovered
            [0.5, 0.5, 0.0],  # req 0 recovered
            [0.1, 0.7, 0.2],  # req 0 bonus
            [0.5, 0.5, 0.0],  # req 1 recovered
            [0.2, 0.5, 0.3],  # req 1 bonus
            [3 / 4, 1 / 4, 0.0],  # req 2 recovered
            [1 / 5, 4 / 5, 0.0],  # req 2 recovered
            [0.3, 0.3, 0.4],  # req 2 bonus
        ],
        dtype=torch.float32,
        device=device,
    )

    cu_num_logits = torch.tensor([0, 3, 5, 8], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor(
        [0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32, device=device
    )
    # Exact values don't matter, only used for seeding
    pos = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Draw N samples with varying seeds to build empirical distributions
    N = 30_000
    counts = torch.zeros(num_tokens, vocab_size, dtype=torch.int64, device=device)
    for trial in range(N):
        # Give each request a distinct seed that changes every trial.
        seed = torch.arange(
            trial * num_reqs + 1,
            (trial + 1) * num_reqs + 1,
            dtype=torch.int64,
            device=device,
        )
        samples = sample_recovered_and_bonus_tokens(
            target_probs, draft_probs, cu_num_logits, idx_mapping, seed, pos
        )
        counts.scatter_add_(
            1,
            samples.unsqueeze(1),
            torch.ones(num_tokens, 1, dtype=torch.int64, device=device),
        )

    # Chi-squared test to compare empirical and expected distributions
    # Chi-squared test cannot handle zero frequencies,
    # so we check those explicitly first
    expected_freq = expected_dists * N  # [num_tokens, vocab_size]
    nonzero = expected_freq > 0  # [num_tokens, vocab_size]
    assert torch.where(~nonzero, counts, 0).sum() == 0, "Sampled a zero-prob token"

    alpha = 1e-4
    for tok_pos in range(num_tokens):
        obs = counts[tok_pos, nonzero[tok_pos]].cpu().numpy().astype(float)
        exp = expected_freq[tok_pos, nonzero[tok_pos]].cpu().numpy().astype(float)
        _, p_value = stats.chisquare(obs, f_exp=exp, sum_check=False)
        assert p_value > alpha, (
            f"Token position {tok_pos}: empirical distribution significantly "
            f"differs from expected ({p_value=:.6f})"
        )
