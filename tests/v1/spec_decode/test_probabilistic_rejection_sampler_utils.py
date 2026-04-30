# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.probabilistic_rejection_sampler_utils import (
    probabilistic_rejection_sample,
)

VOCAB_SIZE = 4096

# Skip if no CUDA - Triton kernel requires GPU
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for rejection sampler tests", allow_module_level=True)


def _build_rejection_sample_inputs(
    target_logits_1d: torch.Tensor,
    draft_logits_1d: torch.Tensor,
    num_speculative_steps: int,
    temperature: float,
    num_trials: int,
) -> dict:
    device = target_logits_1d.device
    vocab_size = target_logits_1d.shape[0]
    K = num_speculative_steps
    num_logits = num_trials * (K + 1)

    target_logits = target_logits_1d.unsqueeze(0).expand(num_logits, -1).contiguous()
    draft_logits = (
        draft_logits_1d.view(1, 1, vocab_size).expand(num_trials, K, -1).contiguous()
    )

    draft_probs = torch.softmax(draft_logits_1d, dim=0)
    draft_tokens = torch.multinomial(
        draft_probs.expand(num_trials, -1), K, replacement=True
    )
    draft_sampled_2d = torch.zeros(num_trials, K + 1, dtype=torch.int64, device=device)
    draft_sampled_2d[:, 1:] = draft_tokens
    draft_sampled = draft_sampled_2d.reshape(-1)

    cu_num_logits = torch.arange(num_trials + 1, dtype=torch.int32, device=device) * (
        K + 1
    )
    pos = torch.arange(num_logits, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(num_trials, dtype=torch.int32, device=device)
    expanded_idx_mapping = torch.arange(
        num_trials, dtype=torch.int32, device=device
    ).repeat_interleave(K + 1)
    expanded_local_pos = torch.arange(K + 1, dtype=torch.int32, device=device).repeat(
        num_trials
    )
    temp_tensor = torch.full(
        (num_trials,), temperature, dtype=torch.float32, device=device
    )
    seed = torch.arange(num_trials, dtype=torch.int64, device=device)

    return dict(
        target_logits=target_logits,
        draft_logits=draft_logits,
        draft_sampled=draft_sampled,
        cu_num_logits=cu_num_logits,
        pos=pos,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        temperature=temp_tensor,
        seed=seed,
    )


def _assert_distribution_match(
    sampled_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    device: str,
    label: str = "",
    min_expected: float = 5.0,
):
    """
    Assert sampled tokens match the target distribution via a
    chi-squared goodness-of-fit test. This is done by computing
    observed vs expected token counts (target_probs * num_samples),
    then checking that the chi-squared statistic is below a conservative
    threshold. The threshold is set at df + 10*sqrt(2*df), which
    corresponds to ~10 sigma under the chi-squared distribution's
    normal approximation, effectively disallowing false positives.

    NOTE: Tokens with expected count < min_expected are merged into
    a single "other" bin to minimize chi-squared noise.
    """
    num_samples = sampled_tokens.shape[0]
    vocab_size = target_probs.shape[0]

    observed = torch.zeros(vocab_size, device=device, dtype=torch.float32)
    observed.scatter_add_(0, sampled_tokens, torch.ones(num_samples, device=device))
    expected = target_probs * num_samples

    sufficient = expected >= min_expected
    obs_main = observed[sufficient]
    exp_main = expected[sufficient]

    obs_other = observed[~sufficient].sum().unsqueeze(0)
    exp_other = expected[~sufficient].sum().unsqueeze(0)

    if exp_other.item() >= min_expected:
        obs_all = torch.cat([obs_main, obs_other])
        exp_all = torch.cat([exp_main, exp_other])
    else:
        obs_all = obs_main
        exp_all = exp_main

    chi2 = ((obs_all - exp_all) ** 2 / exp_all).sum().item()
    df = obs_all.shape[0] - 1
    if df < 1:
        # All samples were merged into < 2 bins, which is too
        # few to evaluate.
        return

    threshold = df + 10 * math.sqrt(2 * df)
    prefix = f"[{label}] " if label else ""
    assert chi2 < threshold, (
        f"{prefix}Chi-squared test failed: chi2={chi2:.1f}, "
        f"df={df}, threshold={threshold:.1f}. "
        f"Output distribution does not match target distribution."
    )


@pytest.mark.parametrize(
    "num_speculative_steps,temperature",
    [
        (1, 0.6),
        (3, 0.6),
        (1, 1.0),
        (3, 1.0),
    ],
)
def test_stochastic_rejection_sample(num_speculative_steps: int, temperature: float):
    """
    Verify that rejection sampling produces the target distribution.
    This is done by simulating many independent trials of speculative
    decoding (from a fixed target and draft distribution). We then
    run rejection sample on all of the trials (requests), and verify
    that the sampled tokens at every position follow the target
    distribution p(x).
    """

    torch.manual_seed(42)
    device = "cuda"
    num_trials = 10 * VOCAB_SIZE

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)

    if temperature > 0:
        target_logits_1d /= temperature
        draft_logits_1d /= temperature

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=temperature,
        num_trials=num_trials,
    )

    sampled, num_sampled = probabilistic_rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps
    )

    target_probs = torch.softmax(target_logits_1d, dim=0)
    for pos in range(num_speculative_steps + 1):
        accepted_mask = num_sampled >= pos + 1
        _assert_distribution_match(
            sampled[accepted_mask, pos], target_probs, device, label=f"position {pos}"
        )


@pytest.mark.parametrize("num_speculative_steps", [1, 3])
def test_greedy_rejection_sample(num_speculative_steps: int):
    """
    Verify that greedy (temperature=0) always outputs the target argmax
    at every accepted position.
    """

    torch.manual_seed(42)
    device = "cuda"
    num_trials = 10 * VOCAB_SIZE

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=0.0,
        num_trials=num_trials,
    )

    sampled, num_sampled = probabilistic_rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps
    )

    target_argmax = target_logits_1d.argmax().item()

    steps = torch.arange(num_speculative_steps + 1, device=device).unsqueeze(0)
    accepted_mask = steps < num_sampled.unsqueeze(1)

    assert (sampled[accepted_mask] == target_argmax).all(), (
        "Greedy sampling produced tokens that are not the target argmax"
    )
