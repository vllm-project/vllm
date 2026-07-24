# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Model Runner V2 Gumbel-max sampling kernel.

Accuracy: define a target categorical distribution as a non-negative int64
count tensor summing to N, turn it into logits (= log(count)), sample many
times with `gumbel_sample`, and check the empirical distribution matches.

The count tensor is deliberately heavy-tailed (one dominant token, the rest
~18 logits below). That tail is the sensitive part: the fp32 Gumbel noise must
reach ~18 to ever sample it. A flat distribution would keep every token within
a few logits of the top and would not exercise the noise tail at all.
"""

import math

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for Gumbel sampler tests", allow_module_level=True)

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample, tl_rand32, tl_rand64

DEVICE = "cuda"
VOCAB_SIZE = 200_000
NUM_SAMPLES = 500_000
# Dominant token is exp(HEAD_LOG_GAP)x larger than the unit-count tail, so the
# tail sits ~HEAD_LOG_GAP logits below the top.
HEAD_LOG_GAP = 18.0
# 10-sigma band: a correct sampler effectively never trips it.
Z_TOLERANCE = 10.0


def _make_heavy_tailed_counts(seed: int = 1234) -> torch.Tensor:
    """Non-negative int64 counts of shape [VOCAB_SIZE]; target prob = counts/N."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    counts = torch.randint(
        1, 4, (VOCAB_SIZE,), generator=gen, dtype=torch.int64, device=DEVICE
    )
    counts[0] = round(math.exp(HEAD_LOG_GAP))  # dominant token
    return counts


def _counts_to_logits(counts: torch.Tensor) -> torch.Tensor:
    # softmax(log(count)) == count / sum(count); count 0 -> logit -inf -> prob 0.
    return counts.double().log().to(torch.float32)


def _sample(
    logits_1d: torch.Tensor,
    num_samples: int,
    *,
    use_fp64: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample `num_samples` tokens from one logit vector.

    Fixed seed with a distinct `pos` per sample gives independent draws; the
    logits are broadcast with a 0-stride view to avoid materializing
    [num_samples, vocab_size].
    """
    vocab_size = logits_1d.shape[0]
    logits = logits_1d.unsqueeze(0).expand(num_samples, vocab_size)
    idx_mapping = torch.zeros(num_samples, dtype=torch.int32, device=DEVICE)
    temp = torch.tensor([temperature], dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([0xABCD], dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_samples, dtype=torch.int64, device=DEVICE)
    return gumbel_sample(
        logits,
        idx_mapping,
        temp,
        seed,
        pos,
        apply_temperature=True,
        use_fp64=use_fp64,
    )


def _z_score(observed: int, expected: float, num_trials: int) -> float:
    p = expected / num_trials
    return (observed - expected) / math.sqrt(num_trials * p * (1 - p))


def _sample_histogram(
    logits_1d: torch.Tensor, num_samples: int, *, chunk: int = 1_000_000
) -> torch.Tensor:
    """Histogram of `num_samples` draws, accumulated in chunks.

    Chunking keeps the kernel's per-sample scratch ([chunk, num_blocks]) bounded
    so a large sample count does not blow up memory.
    """
    vocab_size = logits_1d.shape[0]
    hist = torch.zeros(vocab_size, dtype=torch.float64, device=DEVICE)
    for start in range(0, num_samples, chunk):
        size = min(chunk, num_samples - start)
        logits = logits_1d.unsqueeze(0).expand(size, vocab_size)
        idx_mapping = torch.zeros(size, dtype=torch.int32, device=DEVICE)
        temp = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
        seed = torch.tensor([0xABCD], dtype=torch.int64, device=DEVICE)
        pos = torch.arange(start, start + size, dtype=torch.int64, device=DEVICE)
        out = gumbel_sample(
            logits, idx_mapping, temp, seed, pos, apply_temperature=True
        )
        hist += torch.bincount(out, minlength=vocab_size).double()
    return hist


# ----------------------------- Accuracy ------------------------------------


@pytest.mark.parametrize("use_fp64", [False, True])
def test_sampling_matches_target_distribution(use_fp64: bool):
    counts = _make_heavy_tailed_counts()
    total = counts.sum().item()
    logits = _counts_to_logits(counts)

    sampled = _sample(logits, NUM_SAMPLES, use_fp64=use_fp64)
    assert sampled.min() >= 0 and sampled.max() < VOCAB_SIZE

    # The dominant token (index 0) and the aggregate tail are the two
    # statistically resolvable bins (individual tail tokens are far below the
    # ~5/N detectability floor). The tail mass is small but well above noise,
    # and it lives beyond the fp32 Gumbel cap -- the regime sensitive to noise
    # precision -- so matching it is the meaningful check.
    tail_prob = (total - counts[0].item()) / total
    tail_count = (sampled != 0).sum().item()
    z = _z_score(tail_count, NUM_SAMPLES * tail_prob, NUM_SAMPLES)
    assert abs(z) < Z_TOLERANCE, (
        f"sampled tail mass {tail_count / NUM_SAMPLES:.3e} != target "
        f"{tail_prob:.3e} (z={z:.2f})"
    )


def test_full_vocab_distribution_fidelity():
    """The sampled distribution matches the target across the WHOLE vocab.

    A near-flat count tensor makes every one of the 200K bins individually
    measurable. With ~20 samples/bin, a goodness-of-fit over all bins checks
    that no part of the vocab is over- or under-represented (the heavy-tailed
    test above only resolves head vs aggregate tail). Empirically the fp32
    sampler is as faithful here as torch.multinomial; the residual error is the
    multinomial sampling-noise floor, not the kernel.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(2024)
    counts = torch.randint(
        500, 1500, (VOCAB_SIZE,), generator=gen, dtype=torch.int64, device=DEVICE
    )
    total = counts.sum().item()
    logits = _counts_to_logits(counts)

    num_samples = 4_000_000
    hist = _sample_histogram(logits, num_samples)

    # Diversity: essentially every token must be reachable (no starved region).
    coverage = (hist > 0).sum().item() / VOCAB_SIZE
    assert coverage > 0.99, f"only {coverage:.4f} of the vocab was ever sampled"

    # Goodness-of-fit across all bins (each has expected count >= ~10).
    expected = (counts.double() / total) * num_samples
    chi2 = (((hist - expected) ** 2) / expected).sum().item()
    df = VOCAB_SIZE - 1
    assert chi2 < df + 10 * math.sqrt(2 * df), f"chi2={chi2:.0f}, df={df}"


# ----------------------------- RNG precision --------------------------------


@triton.jit
def _draw_uniform_kernel(offset_ptr, out32_ptr, out64_ptr, seed, N: tl.constexpr):
    idx = tl.arange(0, N)
    offs = tl.load(offset_ptr + idx)
    u32 = tl_rand32(seed, offs, includes_zero=False)
    u64 = tl_rand64(seed, offs, includes_zero=False)
    tl.store(out32_ptr + idx, u32)
    tl.store(out64_ptr + idx, u64)


def test_rand32_resolves_below_tl_rand_floor():
    """`tl_rand32` draws 64 random bits, so its u -> 0 tail resolves below
    `tl.rand`'s 2**-31 floor, and it must agree with `tl_rand64` up to fp32
    rounding. The offsets are draws from the seed=12345 Philox stream found
    (by scan) to fall below 2**-31; they are impossible for a 31-bit uniform.
    """
    seed = 12345
    offsets = torch.tensor(
        [4982566788, 5277073014, 5357046532, 12285768576],
        dtype=torch.int64,
        device=DEVICE,
    )
    u32 = torch.empty(4, dtype=torch.float32, device=DEVICE)
    u64 = torch.empty(4, dtype=torch.float64, device=DEVICE)
    _draw_uniform_kernel[(1,)](offsets, u32, u64, seed, N=4)

    assert (u32 > 0).all()
    assert (u32 < 2.0**-31).all(), f"u32={u32.tolist()}"
    assert torch.equal(u32, u64.float()), f"u32={u32.tolist()}, u64={u64.tolist()}"


# ----------------------------- Edge cases ----------------------------------


def test_greedy_temperature_zero_returns_argmax():
    """temperature == 0 skips Gumbel noise and returns the exact argmax."""
    torch.manual_seed(0)
    num_reqs = 128
    logits = torch.randn(num_reqs, VOCAB_SIZE, device=DEVICE, dtype=torch.float32)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    temp = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)

    sampled = gumbel_sample(
        logits, idx_mapping, temp, seed, pos, apply_temperature=True
    )
    assert torch.equal(sampled, logits.argmax(dim=-1))


def test_zero_count_tokens_are_never_sampled():
    """Count 0 -> -inf logit -> probability 0; must never be selected."""
    counts = _make_heavy_tailed_counts(seed=7)
    zeroed = torch.arange(1, VOCAB_SIZE, 2, device=DEVICE)  # odd indices (not head)
    counts[zeroed] = 0
    logits = _counts_to_logits(counts)

    sampled = _sample(logits, NUM_SAMPLES)
    assert sampled.min() >= 0 and sampled.max() < VOCAB_SIZE
    assert not torch.isin(sampled, zeroed).any(), "sampled a zero-probability token"


def test_single_nonzero_token_is_always_sampled():
    """A lone finite logit must win every draw, regardless of its index."""
    counts = torch.zeros(VOCAB_SIZE, dtype=torch.int64, device=DEVICE)
    counts[123_456] = 1000
    logits = _counts_to_logits(counts)

    sampled = _sample(logits, 10_000)
    assert (sampled == 123_456).all()


@pytest.mark.parametrize("vocab_size", [1, 999, 1024, 4097])
def test_vocab_size_not_multiple_of_block(vocab_size: int):
    """Per-block tail masking for non-block-aligned vocab; all bins measurable."""
    gen = torch.Generator(device=DEVICE).manual_seed(vocab_size)
    counts = torch.randint(
        20, 200, (vocab_size,), generator=gen, dtype=torch.int64, device=DEVICE
    )
    total = counts.sum().item()
    logits = _counts_to_logits(counts)
    num_samples = max(40 * vocab_size, 50_000)

    sampled = _sample(logits, num_samples)
    assert sampled.min() >= 0 and sampled.max() < vocab_size

    observed = torch.bincount(sampled, minlength=vocab_size).double()
    expected = (counts.double() / total) * num_samples
    chi2 = (((observed - expected) ** 2) / expected).sum().item()
    df = vocab_size - 1
    if df >= 1:
        assert chi2 < df + 10 * math.sqrt(2 * df), f"chi2={chi2:.1f}, df={df}"
