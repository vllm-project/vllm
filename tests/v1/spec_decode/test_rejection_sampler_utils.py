# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
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
    """Build rejection_sample kwargs from a fixed target and draft distribution.

    target_logits_1d must already have temperature applied (the sampler applies
    sampling params before verification), whereas draft_logits_1d must not:
    rejection_sample divides the draft logits by the temperature on load.
    """
    device = target_logits_1d.device
    vocab_size = target_logits_1d.shape[0]
    K = num_speculative_steps
    num_logits = num_trials * (K + 1)

    target_logits = target_logits_1d.unsqueeze(0).expand(num_logits, -1).contiguous()
    draft_logits = (
        draft_logits_1d.view(1, 1, vocab_size).expand(num_trials, K, -1).contiguous()
    )

    scaled_draft_logits_1d = draft_logits_1d.float()
    if temperature > 0:
        scaled_draft_logits_1d = scaled_draft_logits_1d / temperature
    draft_probs = torch.softmax(scaled_draft_logits_1d, dim=0)
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
@pytest.mark.parametrize("draft_logits_dtype", [torch.float32, torch.bfloat16])
def test_stochastic_rejection_sample(
    num_speculative_steps: int, temperature: float, draft_logits_dtype: torch.dtype
):
    """
    Verify that rejection sampling produces the target distribution.
    This is done by simulating many independent trials of speculative
    decoding (from a fixed target and draft distribution). We then
    run rejection sample on all of the trials (requests), and verify
    that the sampled tokens at every position follow the target
    distribution p(x).

    Parametrized over the draft-logits dtype: storing them in the draft head's
    dtype must not bias the output distribution.
    """

    torch.manual_seed(42)
    device = "cuda"
    num_trials = 10 * VOCAB_SIZE

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32).to(
        draft_logits_dtype
    )

    if temperature > 0:
        target_logits_1d /= temperature

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=temperature,
        num_trials=num_trials,
    )

    sampled, num_sampled = rejection_sample(
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

    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps
    )

    target_argmax = target_logits_1d.argmax().item()

    steps = torch.arange(num_speculative_steps + 1, device=device).unsqueeze(0)
    accepted_mask = steps < num_sampled.unsqueeze(1)

    assert (sampled[accepted_mask] == target_argmax).all(), (
        "Greedy sampling produced tokens that are not the target argmax"
    )


@pytest.mark.parametrize(
    "num_speculative_steps,temperature,unconditional_rates",
    [
        (3, 1.0, [0.9, 0.5, 0.2]),
        (3, 0.0, [0.9, 0.5, 0.2]),
        (3, 1.0, [1.0, 1.0, 1.0]),
        (3, 0.0, [1.0, 1.0, 1.0]),
        (3, 1.0, [0.0, 0.0, 0.0]),
        (3, 0.0, [0.0, 0.0, 0.0]),
        (1, 1.0, [0.7]),
        (1, 0.0, [0.7]),
    ],
)
def test_synthetic_rejection_sample(
    num_speculative_steps: int,
    temperature: float,
    unconditional_rates: list[float],
):
    """
    Verify that synthetic rejection sampling produces the expected
    per-position acceptance rates. The unconditional rate at position i
    is P(all draft steps 0..i accepted) = product(conditional_rates[0:i+1]).
    This is approximately mean(num accepted >= i + 1) over many trials.
    """
    from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

    torch.manual_seed(42)
    device = "cuda"
    num_trials = 10 * VOCAB_SIZE
    deviation_tol = 1e-2

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)

    if temperature > 0:
        target_logits_1d /= temperature

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=temperature,
        num_trials=num_trials,
    )

    conditional_rates = unconditional_to_conditional_rates(unconditional_rates)
    synthetic_conditional_rates = torch.tensor(
        conditional_rates, dtype=torch.float32, device=device
    )

    _, num_sampled = rejection_sample(
        **inputs,
        num_speculative_steps=num_speculative_steps,
        synthetic_conditional_rates=synthetic_conditional_rates,
    )

    # num_sampled includes the resampled/bonus token.
    num_accepted = num_sampled - 1
    for i, expected_rate in enumerate(unconditional_rates):
        observed_rate = (num_accepted >= i + 1).float().mean().item()
        assert abs(observed_rate - expected_rate) < deviation_tol, (
            f"Step {i}: observed rate {observed_rate:.4f} deviates from "
            f"expected rate {expected_rate:.4f} by more than {deviation_tol}."
        )


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_all_nan_target_logits_in_range(temperature: float):
    """Regression test for NaN breaking tl.argmax index bounds.

    An all-NaN target logits row makes every per-block local max NaN.
    tl.argmax over such a block returns an index into the padded region
    (>= num_blocks), causing an OOB read of the local-argmax tensors in
    _compute_global_target_argmax (greedy) and _insert_resampled_kernel
    (stochastic). The vocab size below gives a non-power-of-2 block count
    (3 blocks of 8192, padded to 4) so the padded region exists. Post-fix,
    NaN is mapped to -inf, so the kernels must complete without error and
    emit in-range token ids.
    """
    torch.manual_seed(0)
    device = "cuda"
    num_trials = 4
    K = 1
    vocab_size = 20000  # 3 vocab blocks of 8192, padded to 4

    target_logits_1d = torch.full(
        (vocab_size,), float("nan"), device=device, dtype=torch.float32
    )
    draft_logits_1d = torch.randn(vocab_size, device=device, dtype=torch.float32)

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        K,
        temperature=temperature,
        num_trials=num_trials,
    )

    sampled, num_sampled = rejection_sample(**inputs, num_speculative_steps=K)

    assert ((num_sampled >= 1) & (num_sampled <= K + 1)).all()
    steps = torch.arange(K + 1, device=device).unsqueeze(0)
    valid = steps < num_sampled.unsqueeze(1)
    assert (sampled[valid] >= 0).all()
    assert (sampled[valid] < vocab_size).all()


def test_placeholder_draft_token_rejected():
    """A placeholder draft id (-1) must be rejected without reading the logit
    tensors out of bounds, for any sampling method.
    """
    torch.manual_seed(0)
    device = "cuda"
    num_trials = 64
    K = 1
    temperature = 0.6

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device) / temperature
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device)

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        K,
        temperature=temperature,
        num_trials=num_trials,
    )
    inputs["draft_sampled"].view(num_trials, K + 1)[:, 1:] = -1

    sampled, num_sampled = rejection_sample(**inputs, num_speculative_steps=K)

    assert torch.equal(num_sampled, torch.ones_like(num_sampled))
    recovered = sampled[:, 0]
    assert (recovered >= 0).all() and (recovered < VOCAB_SIZE).all()


@pytest.mark.parametrize("has_draft_logits", [True, False])
@pytest.mark.parametrize("num_placeholders", [1, 2])
def test_block_verification_placeholder_truncates_block(
    has_draft_logits: bool, num_placeholders: int
):
    """Trailing placeholder (-1) draft ids end the block early. The last real
    draft token must then be verified against the closed-form threshold rather
    than a residual mass derived from the placeholder position, whose logits
    describe a token that was never drafted.
    """
    torch.manual_seed(42)
    device = "cuda"
    num_trials = 20 * VOCAB_SIZE
    K = 3
    temperature = 1.0

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device) / temperature
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device)

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        K,
        temperature=temperature,
        num_trials=num_trials,
    )
    if not has_draft_logits:
        inputs["draft_logits"] = None
    inputs["draft_sampled"].view(num_trials, K + 1)[:, K + 1 - num_placeholders :] = -1

    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=K, use_block_verification=True
    )

    num_real = K - num_placeholders
    assert (num_sampled <= num_real + 1).all(), (
        "Accepted a draft token at or after a placeholder."
    )
    target_probs = torch.softmax(target_logits_1d, dim=0)
    for pos in range(num_real + 1):
        accepted_mask = num_sampled >= pos + 1
        _assert_distribution_match(
            sampled[accepted_mask, pos], target_probs, device, label=f"position {pos}"
        )


def test_greedy_placeholder_emits_target_argmax():
    """Greedy sampling skips resampling and relies on the rejection kernel
    storing the target argmax at the rejected position. A placeholder must not
    bypass that store, or the output slot is returned uninitialized.
    """
    torch.manual_seed(0)
    device = "cuda"
    num_trials = 512
    K = 3

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device)

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        K,
        temperature=0.0,
        num_trials=num_trials,
    )
    target_argmax = int(target_logits_1d.argmax())
    draft_sampled = inputs["draft_sampled"].view(num_trials, K + 1)
    # Accept the first draft so that the rejection lands on the placeholder.
    draft_sampled[:, 1] = target_argmax
    draft_sampled[:, 2:] = -1

    sampled, num_sampled = rejection_sample(**inputs, num_speculative_steps=K)

    assert torch.equal(num_sampled, torch.full_like(num_sampled, 2))
    steps = torch.arange(K + 1, device=device).unsqueeze(0)
    emitted = sampled[steps < num_sampled.unsqueeze(1)]
    assert (emitted == target_argmax).all(), (
        "Greedy sampling emitted a token that is not the target argmax"
    )


@pytest.mark.parametrize("use_block_verification", [False, True])
def test_placeholder_blocks_later_draft_tokens(use_block_verification: bool):
    """A placeholder is not necessarily the final draft. Nothing at or after
    one may be accepted, even when a valid draft follows it.
    """
    torch.manual_seed(0)
    device = "cuda"
    num_trials = 4 * VOCAB_SIZE
    K = 3
    temperature = 1.0

    # An identical draft is accepted with probability ~1, so any token after
    # the placeholder would be accepted unless it is explicitly blocked. The
    # draft logits are stored pre-temperature, so pass the unscaled base.
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device)
    target_logits_1d = draft_logits_1d / temperature

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        K,
        temperature=temperature,
        num_trials=num_trials,
    )
    inputs["draft_sampled"].view(num_trials, K + 1)[:, 2] = -1

    _, num_sampled = rejection_sample(
        **inputs,
        num_speculative_steps=K,
        use_block_verification=use_block_verification,
    )

    # Only the first draft is verifiable, plus one resampled token.
    assert (num_sampled <= 2).all(), "Accepted a draft token past a placeholder."
    assert (num_sampled == 2).float().mean().item() > 0.9, (
        "The first draft was rarely accepted; the test is not exercising "
        "acceptance past the placeholder."
    )


@pytest.mark.parametrize(
    "num_speculative_steps,temperature",
    [
        (1, 0.6),
        (3, 0.6),
        (1, 1.0),
        (3, 1.0),
        (5, 1.0),
    ],
)
@pytest.mark.parametrize("has_draft_logits", [True, False])
def test_block_verification_rejection_sample(
    num_speculative_steps: int, temperature: float, has_draft_logits: bool
):
    """
    Verify that block verification (Sun et al.) preserves the target
    distribution at every accepted position, for both the full draft-logits
    case and the one-hot (no draft logits) case. Block verification changes
    *which* prefix is accepted, but the marginal of every output position must
    still match the target distribution p(x).
    """

    torch.manual_seed(42)
    device = "cuda"
    num_trials = 10 * VOCAB_SIZE

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    draft_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)

    if temperature > 0:
        target_logits_1d /= temperature

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=temperature,
        num_trials=num_trials,
    )
    if not has_draft_logits:
        inputs["draft_logits"] = None

    sampled, num_sampled = rejection_sample(
        **inputs,
        num_speculative_steps=num_speculative_steps,
        use_block_verification=True,
    )

    target_probs = torch.softmax(target_logits_1d, dim=0)
    for pos in range(num_speculative_steps + 1):
        accepted_mask = num_sampled >= pos + 1
        _assert_distribution_match(
            sampled[accepted_mask, pos], target_probs, device, label=f"position {pos}"
        )


@pytest.mark.parametrize("num_speculative_steps", [3, 5])
def test_block_verification_accepts_at_least_as_many(num_speculative_steps: int):
    """
    Block verification is designed to accept at least as long a prefix as
    token verification in expectation. Verify the mean accepted length is no
    worse than the standard method on the same inputs.
    """

    torch.manual_seed(0)
    device = "cuda"
    num_trials = 20 * VOCAB_SIZE
    temperature = 1.0

    target_logits_1d = torch.randn(VOCAB_SIZE, device=device, dtype=torch.float32)
    # A draft close to the target gives block verification room to recover
    # prefixes that token verification would have truncated.
    draft_logits_1d = target_logits_1d + 0.5 * torch.randn(
        VOCAB_SIZE, device=device, dtype=torch.float32
    )

    inputs = _build_rejection_sample_inputs(
        target_logits_1d,
        draft_logits_1d,
        num_speculative_steps,
        temperature=temperature,
        num_trials=num_trials,
    )

    _, num_sampled_standard = rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps
    )
    _, num_sampled_block = rejection_sample(
        **inputs,
        num_speculative_steps=num_speculative_steps,
        use_block_verification=True,
    )

    mean_standard = (num_sampled_standard - 1).float().mean().item()
    mean_block = (num_sampled_block - 1).float().mean().item()
    # Allow a small slack for sampling noise.
    assert mean_block >= mean_standard - 1e-2, (
        f"Block verification mean accepted length {mean_block:.4f} is worse "
        f"than standard {mean_standard:.4f}."
    )


@pytest.mark.parametrize("has_draft_logits", [True, False])
def test_chunked_requests_match_full_batch(has_draft_logits: bool):
    torch.manual_seed(7)
    device = "cuda"
    num_reqs = 5
    num_speculative_steps = 3
    vocab_size = 257

    target_logits = torch.randn(vocab_size, device=device)
    draft_logits = torch.randn(vocab_size, device=device)
    inputs = _build_rejection_sample_inputs(
        target_logits,
        draft_logits,
        num_speculative_steps,
        temperature=0.6,
        num_trials=num_reqs,
    )
    padded_target_logits = torch.empty(
        inputs["target_logits"].shape[0], vocab_size + 3, device=device
    )
    padded_target_logits[:, :vocab_size].copy_(inputs["target_logits"])
    inputs["target_logits"] = padded_target_logits[:, :vocab_size]
    assert inputs["target_logits"].stride(-1) == 1
    assert not inputs["target_logits"].is_contiguous()
    if not has_draft_logits:
        inputs["draft_logits"] = None

    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=num_speculative_steps
    )

    sampled_chunks = []
    num_sampled_chunks = []
    for start, end in ((0, 2), (2, 5)):
        lo = start * (num_speculative_steps + 1)
        hi = end * (num_speculative_steps + 1)
        chunk_inputs = dict(inputs)
        for name in (
            "target_logits",
            "draft_sampled",
            "pos",
            "expanded_idx_mapping",
            "expanded_local_pos",
        ):
            chunk_inputs[name] = inputs[name][lo:hi]
        chunk_inputs["cu_num_logits"] = inputs["cu_num_logits"][start : end + 1] - lo
        chunk_inputs["idx_mapping"] = inputs["idx_mapping"][start:end]

        chunk_sampled, chunk_num_sampled = rejection_sample(
            **chunk_inputs, num_speculative_steps=num_speculative_steps
        )
        sampled_chunks.append(chunk_sampled)
        num_sampled_chunks.append(chunk_num_sampled)

    chunked_sampled = torch.cat(sampled_chunks)
    chunked_num_sampled = torch.cat(num_sampled_chunks)
    assert torch.equal(chunked_num_sampled, num_sampled)
    steps = torch.arange(num_speculative_steps + 1, device=device)
    valid = steps.unsqueeze(0) < num_sampled.unsqueeze(1)
    assert torch.equal(chunked_sampled[valid], sampled[valid])
