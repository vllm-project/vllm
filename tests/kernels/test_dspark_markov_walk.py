# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dspark_markov_walk against an exact replay of its int8 arithmetic.

The kernel's dot products are exact int32 sums, so a float64 replay of the
same quantized operands and fp32 roundings reproduces its bf16 logits bit for
bit (up to double rounding in the final fma, which random data never hits).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dspark.markov_walk import (
    STATE_NUMEL,
    DSparkMarkovWalker,
    pick_tiles,
    quantize_w1,
    quantize_w2,
    unsupported_reason,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability(90)
    or not hasattr(torch.ops._C, "dspark_markov_walk"),
    reason="dspark_markov_walk needs an sm_90 build",
)

DEVICE = "cuda"
RANK = 256


def make_head(vocab: int, seed: int = 0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w1 = (torch.randn(vocab, RANK, generator=g, device=DEVICE) * 0.05).bfloat16()
    w2 = (torch.randn(vocab, RANK, generator=g, device=DEVICE) * 0.02).bfloat16()
    return w1, w2


class Reference:
    """Float64 replay of the kernel's numerics."""

    def __init__(self, w1, w2, logit_scale=1.0):
        q_w2, self.row_scale = quantize_w2(w2, logit_scale)
        self.q_w2 = q_w2.double()
        q_hi, q_lo, _, self.s_lo = quantize_w1(w1)
        self.q_hi, self.q_lo = q_hi.double(), q_lo.double()

    def logits(self, base_k: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """bf16 [n, V] logits of one step from base [n, V] and prev [n]."""
        d_hi = self.q_hi[prev] @ self.q_w2.T
        d_lo = self.q_lo[prev] @ self.q_w2.T
        combined = (d_hi * 256 + d_lo).long().float().double()
        bias = (self.s_lo[prev].double()[:, None] * combined).float().double()
        logit = self.row_scale.double()[None] * bias + base_k.double()
        return logit.float().bfloat16()


def run_walk(
    walker,
    base,
    anchor,
    *,
    temps=None,
    seeds=None,
    pos=None,
    idx=None,
    draft_logits=None,
    tokens=None,
):
    nb, steps, _ = base.shape
    if idx is None:
        idx = torch.arange(nb, device=DEVICE).repeat_interleave(steps)
    if seeds is None:
        seeds = torch.arange(nb, dtype=torch.int64, device=DEVICE) + 1234
    if pos is None:
        pos = torch.arange(nb * steps, dtype=torch.int64, device=DEVICE)
    if tokens is None:
        tokens = torch.full((nb, steps), -1, dtype=torch.int64, device=DEVICE)
    walker.walk(base, anchor.int(), 1, idx, temps, seeds, pos, tokens, draft_logits)
    return tokens


@pytest.mark.parametrize("vocab", [151936, 129280, 248320, 4096])
@pytest.mark.parametrize("num_reqs", [1, 3, 32, 33, 64])
@pytest.mark.parametrize("steps", [1, 5, 16])
def test_greedy_matches_reference(vocab, num_reqs, steps):
    if pick_tiles(vocab, current_platform.num_compute_units()) is None:
        pytest.skip("vocabulary too large for this GPU")
    w1, w2 = make_head(vocab)
    walker = DSparkMarkovWalker(w1, w2, steps)
    ref = Reference(w1, w2)
    g = torch.Generator(device=DEVICE).manual_seed(num_reqs * 100 + steps)
    base = (
        torch.randn(num_reqs, steps, vocab, generator=g, device=DEVICE) * 3
    ).bfloat16()
    anchor = torch.randint(0, vocab, (num_reqs,), generator=g, device=DEVICE)
    tokens = run_walk(walker, base, anchor)

    prev = anchor
    for k in range(steps):
        expected = ref.logits(base[:, k], prev).argmax(-1)
        torch.testing.assert_close(tokens[:, k], expected, rtol=0, atol=0)
        prev = tokens[:, k]


def test_logit_scale_folds_into_the_bias():
    vocab, steps = 4096, 4
    w1, w2 = make_head(vocab, seed=1)
    walker = DSparkMarkovWalker(w1, w2, steps, logit_scale=0.75)
    ref = Reference(w1, w2, logit_scale=0.75)
    base = (torch.randn(2, steps, vocab, device=DEVICE) * 0.5).bfloat16()
    anchor = torch.tensor([3, 4000], device=DEVICE)
    tokens = run_walk(walker, base, anchor)
    prev = anchor
    for k in range(steps):
        assert torch.equal(tokens[:, k], ref.logits(base[:, k], prev).argmax(-1))
        prev = tokens[:, k]


@pytest.mark.parametrize("num_reqs", [1, 40])
def test_draft_logits_cache(num_reqs):
    """Sampled requests cache the exact logits they drew from; greedy and
    padding requests leave their rows alone."""
    vocab, steps, max_reqs = 151936, 7, 128
    w1, w2 = make_head(vocab, seed=2)
    walker = DSparkMarkovWalker(w1, w2, steps)
    ref = Reference(w1, w2)
    base = (torch.randn(num_reqs, steps, vocab, device=DEVICE) * 3).bfloat16()
    anchor = torch.randint(0, vocab, (num_reqs,), device=DEVICE)
    # request b lives in state slot 2b + 1; every third one is greedy, the
    # last one is a padding row
    slots = torch.arange(num_reqs, device=DEVICE) * 2 + 1
    slots[-1] = -1
    idx = slots.repeat_interleave(steps)
    temperature = torch.zeros(max_reqs, device=DEVICE)
    temperature[slots[slots >= 0]] = 0.8
    temperature[slots[::3][slots[::3] >= 0]] = 0.0
    sentinel = 7.0
    cache = torch.full(
        (max_reqs, steps, vocab), sentinel, dtype=torch.bfloat16, device=DEVICE
    )
    seeds = torch.randint(0, 2**62, (max_reqs,), device=DEVICE)
    tokens = run_walk(
        walker,
        base,
        anchor,
        temps=temperature,
        seeds=seeds,
        idx=idx,
        draft_logits=cache,
    )

    assert ((tokens >= 0) & (tokens < vocab)).all()
    prev = anchor
    for k in range(steps):
        expected = ref.logits(base[:, k], prev)
        for b in range(num_reqs):
            slot = int(slots[b])
            sampled = slot >= 0 and float(temperature[slot]) > 0
            if sampled:
                assert torch.equal(cache[slot, k], expected[b])
            elif slot >= 0:
                assert (cache[slot, k] == sentinel).all()
            if not sampled:  # greedy and padding rows take the argmax
                assert int(tokens[b, k]) == int(expected[b].argmax())
        prev = tokens[:, k]
    untouched = torch.ones(max_reqs, dtype=torch.bool, device=DEVICE)
    untouched[slots[slots >= 0]] = False
    assert (cache[untouched] == sentinel).all()


def test_sampling_is_seeded_and_distributed():
    vocab, steps = 4096, 1
    w1 = torch.randn(vocab, RANK, device=DEVICE).bfloat16()
    w2 = torch.zeros(vocab, RANK, dtype=torch.bfloat16, device=DEVICE)  # no bias
    walker = DSparkMarkovWalker(w1, w2, steps)
    logits = torch.full((vocab,), -30.0, device=DEVICE)
    hot = torch.tensor([5, 77, 1000, 1001, 2047, 4095], device=DEVICE)
    logits[hot] = torch.tensor([1.0, 0.5, 0.0, -0.5, 0.25, 0.75], device=DEVICE)
    base = logits.bfloat16().view(1, 1, vocab)
    temperature = torch.tensor([0.9], device=DEVICE)
    probs = torch.softmax(base.float().view(-1) / 0.9, -1)[hot].double()

    n = 6000
    counts = torch.zeros(vocab, dtype=torch.int64, device=DEVICE)
    first = None
    for i in range(n):
        tokens = run_walk(
            walker,
            base,
            torch.zeros(1, device=DEVICE, dtype=torch.long),
            temps=temperature,
            seeds=torch.tensor([42], device=DEVICE),
            pos=torch.tensor([i], device=DEVICE),
        )
        counts[tokens.view(-1)] += 1
        if i == 0:
            first = tokens.clone()
    assert int(counts[hot].sum()) == n
    # the same (seed, position) replays the same draw
    again = run_walk(
        walker,
        base,
        torch.zeros(1, device=DEVICE, dtype=torch.long),
        temps=temperature,
        seeds=torch.tensor([42], device=DEVICE),
        pos=torch.tensor([0], device=DEVICE),
    )
    assert torch.equal(first, again)
    expected = probs * n
    chi2 = float((((counts[hot].double() - expected) ** 2) / expected).sum())
    assert chi2 < 30.0, (chi2, counts[hot].tolist(), expected.tolist())  # 5 dof


def test_non_finite_rows_stay_in_range():
    vocab, steps = 151936, 3
    w1, w2 = make_head(vocab, seed=3)
    walker = DSparkMarkovWalker(w1, w2, steps)
    base = torch.randn(9, steps, vocab, device=DEVICE).bfloat16()
    base[4] = float("nan")
    base[7] = float("-inf")
    tokens = run_walk(walker, base, torch.randint(0, vocab, (9,), device=DEVICE))
    assert ((tokens >= 0) & (tokens < vocab)).all()
    assert (tokens[7] == 0).all()


def test_rejects_unsupported_heads():
    w1, w2 = make_head(4096)
    assert unsupported_reason(w1, w2, 7) is None
    assert unsupported_reason(w1, w2[:, :128], 7) is not None
    assert unsupported_reason(w1[:2048], w2, 7) is not None
    assert unsupported_reason(w1, w2, 17) is not None
    assert STATE_NUMEL == 16 + 2 * 16 * 64 * 16
