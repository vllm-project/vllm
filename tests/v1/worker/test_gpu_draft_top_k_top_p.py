# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probabilistic drafting restricted to the requests' top-k / top-p support.

The target samples from top-k / top-p truncated probabilities, so a draft token
outside that support is always rejected. The V2 drafter now applies the same
truncation before sampling. These tests check that the mask matches the target's
support (and is left off for rows without top-k), that draft sampling and the
draft-logits cache both see it, and that the rejection sampler's output
distribution is unchanged while acceptance rises.
"""

import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.draft_support import (
    draft_support_max_top_k,
    draft_top_k_top_p_threshold,
    mask_below_threshold,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires CUDA and Triton"
)

DEVICE = "cuda"


def _speculator(top_k=None, top_p=None) -> SimpleNamespace:
    speculator = SimpleNamespace(
        draft_top_k=top_k,
        draft_top_p=top_p,
        draft_watermarker=None,
        use_fp64_gumbel=False,
        acceptance_estimator=None,
        model=SimpleNamespace(compute_logits=lambda hidden_states: hidden_states),
    )
    speculator._maybe_predict_acceptance = partial(
        DraftModelSpeculator._maybe_predict_acceptance, speculator
    )
    speculator._draft_support_threshold = partial(
        DraftModelSpeculator._draft_support_threshold, speculator
    )
    return speculator


def _apply_support(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """The logits with tokens outside the top-k / top-p support at -inf."""
    threshold = draft_top_k_top_p_threshold(
        logits, idx_mapping, top_k, top_p, temperature
    )
    return mask_below_threshold(logits, threshold)


def _no_top_p(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.float32, device=DEVICE)


def _no_top_k(n: int, vocab: int) -> torch.Tensor:
    return torch.full((n,), vocab, dtype=torch.int32, device=DEVICE)


def _reference_support(
    logits: torch.Tensor, top_k: int, top_p: float, temperature: float
) -> torch.Tensor:
    """Tokens kept by top-k then top-p over the temperature-scaled distribution,
    written independently of the kernels under test."""
    scaled = logits.double() / (temperature if temperature > 0 else 1.0)
    order = torch.argsort(scaled, descending=True)
    keep = torch.zeros_like(scaled, dtype=torch.bool)
    kept = order[:top_k]
    probs = torch.softmax(scaled[kept], dim=-1)
    # Smallest prefix whose mass reaches top_p; the top token always stays.
    cum = torch.cumsum(probs, dim=-1)
    n = int((cum < top_p).sum().item()) + 1
    keep[kept[: min(n, top_k)]] = True
    return keep


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mask_matches_top_k_top_p_support(dtype: torch.dtype):
    torch.manual_seed(0)
    vocab, max_num_reqs = 1000, 8
    logits = (torch.randn(4, vocab, device=DEVICE) * 3).to(dtype)
    idx_mapping = torch.tensor([5, 0, 3, 7], dtype=torch.int32, device=DEVICE)
    top_k = torch.full((max_num_reqs,), vocab, dtype=torch.int32, device=DEVICE)
    top_p = torch.ones(max_num_reqs, dtype=torch.float32, device=DEVICE)
    temperature = torch.ones(max_num_reqs, dtype=torch.float32, device=DEVICE)
    # (req_state_idx, top_k, top_p, temperature); one row keeps the full vocab,
    # one is greedy (temperature 0).
    rows = [
        (5, 20, 0.9, 0.6),
        (0, vocab, 1.0, 1.0),
        (3, 50, 0.5, 1.5),
        (7, 10, 0.95, 0.0),
    ]
    for req, k, p, t in rows:
        top_k[req], top_p[req], temperature[req] = k, p, t

    out = _apply_support(logits, idx_mapping, top_k, top_p, temperature)

    assert out.dtype == dtype and out.shape == logits.shape
    for i, (_, k, p, t) in enumerate(rows):
        expected = _reference_support(logits[i].float(), k, p, t)
        kept = ~out[i].isneginf()
        assert kept[expected].all(), f"row {i}: a support token was masked"
        if dtype == torch.float32:
            assert torch.equal(kept, expected), f"row {i}: support differs"
        else:
            # bfloat16 has ties: the reference breaks them by position, the
            # kernel keeps every tied token, and tokens tied at the top-k cut
            # also widen the set top-p is normalised over.
            assert kept.sum() <= 2 * expected.sum() + 8, f"row {i}: support differs"
        # Kept logits are passed through untouched.
        assert torch.equal(out[i][kept], logits[i][kept])


def test_top_p_is_taken_after_temperature():
    torch.manual_seed(1)
    vocab, k = 500, 400
    logits = torch.randn(2, vocab, device=DEVICE) * 2
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    top_k = torch.tensor([k, k], dtype=torch.int32, device=DEVICE)
    top_p = torch.tensor([0.9, 0.9], dtype=torch.float32, device=DEVICE)
    temperature = torch.tensor([0.5, 2.0], dtype=torch.float32, device=DEVICE)
    out = _apply_support(
        torch.cat([logits[:1], logits[:1]]), idx_mapping, top_k, top_p, temperature
    )
    cold, hot = (~out.isneginf()).sum(dim=-1).tolist()
    # A flatter (hotter) distribution needs more tokens to reach the same mass.
    assert cold < hot < k
    for i, t in enumerate((0.5, 2.0)):
        assert torch.equal(~out[i].isneginf(), _reference_support(logits[0], k, 0.9, t))


def test_padding_rows_are_left_alone():
    torch.manual_seed(2)
    vocab = 300
    logits = torch.randn(3, vocab, device=DEVICE)
    idx_mapping = torch.tensor([0, -1, 1], dtype=torch.int32, device=DEVICE)
    top_k = torch.tensor([5, 5], dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(2, dtype=torch.float32, device=DEVICE)
    out = _apply_support(logits, idx_mapping, top_k, _no_top_p(2), temperature)
    assert (~out[0].isneginf()).sum() == 5
    assert torch.equal(out[1], logits[1])
    assert (~out[2].isneginf()).sum() == 5


@pytest.mark.parametrize("vocab", [16_384, 151_936])
def test_top_k_above_the_limit_is_left_unmasked(vocab: int):
    """Top-k up to 1024 is masked whatever the vocabulary; a wider top-k is
    left alone, like no top-k."""
    torch.manual_seed(5)
    max_top_k = draft_support_max_top_k(vocab)
    assert max_top_k == 1024
    logits = torch.randn(3, vocab, device=DEVICE)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    top_k = torch.tensor(
        [10, max_top_k, max_top_k + 1], dtype=torch.int32, device=DEVICE
    )
    temperature = torch.ones(3, dtype=torch.float32, device=DEVICE)
    out = _apply_support(logits, idx_mapping, top_k, _no_top_p(3), temperature)
    for row, k in enumerate((10, max_top_k)):
        kept = ~out[row].isneginf()
        assert kept[_reference_support(logits[row], k, 1.0, 1.0)].all()
        assert k <= kept.sum() < vocab // 2, int(kept.sum())
    assert torch.equal(out[2], logits[2])


def test_top_p_without_top_k_is_left_unmasked():
    """Top-p alone removes little and is not applied; with top-k it is exact
    over a small vocabulary, even when top-k is as wide as the vocabulary - 1."""
    torch.manual_seed(6)
    vocab = 1024
    # A peaked row with top-k and top-p, a flat row with top-p only, and a
    # geometric row with top-k = vocab - 1 and top-p.
    logits = torch.randn(3, vocab, device=DEVICE)
    logits[0] *= 8
    logits[2] = -0.02 * torch.randperm(vocab, device=DEVICE).float()
    idx_mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device=DEVICE)
    top_k = torch.tensor([vocab, vocab - 1, 30], dtype=torch.int32, device=DEVICE)
    top_p = torch.tensor([0.9, 0.9, 0.8], dtype=torch.float32, device=DEVICE)
    temperature = torch.tensor([1.0, 0.7, 1.3], dtype=torch.float32, device=DEVICE)
    out = _apply_support(logits, idx_mapping, top_k, top_p, temperature)
    kept = ~out.isneginf()
    assert torch.equal(kept[0], _reference_support(logits[0], 30, 0.8, 1.3))
    assert torch.equal(out[1], logits[1])
    assert torch.equal(kept[2], _reference_support(logits[2], vocab - 1, 0.9, 0.7))


def test_reduced_draft_vocab_with_neg_inf_logits():
    """EAGLE3 heads with a reduced draft vocabulary hand over full-vocabulary
    logits that are -inf outside it. The threshold must still be placed."""
    torch.manual_seed(10)
    vocab, draft_vocab = 128_256, 32_000
    logits = torch.full((2, vocab), float("-inf"), device=DEVICE)
    draft_ids = torch.randperm(vocab, device=DEVICE)[:draft_vocab]
    logits[:, draft_ids] = torch.randn(2, draft_vocab, device=DEVICE) * 3
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    top_k = torch.tensor([20, 1000], dtype=torch.int32, device=DEVICE)
    top_p = torch.tensor([0.9, 0.9], dtype=torch.float32, device=DEVICE)
    temperature = torch.ones(2, dtype=torch.float32, device=DEVICE)
    out = _apply_support(logits, idx_mapping, top_k, top_p, temperature)
    kept = ~out[0].isneginf()
    assert kept[_reference_support(logits[0], 20, 0.9, 1.0)].all()
    assert kept.sum() < 200, int(kept.sum())
    # A wide top-k: the top-p cut still applies.
    kept = ~out[1].isneginf()
    assert kept[_reference_support(logits[1], 1000, 0.9, 1.0)].all()
    assert kept.sum() < draft_vocab, int(kept.sum())


def test_sample_draft_samples_and_caches_inside_support():
    torch.manual_seed(3)
    vocab, num_reqs, k = 256, 512, 4
    base = torch.randn(vocab, device=DEVICE)
    hidden = base.expand(num_reqs, vocab).contiguous()
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    top_k = torch.full((num_reqs,), k, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
    seeds = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
    positions = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
    cache = torch.zeros(num_reqs, 1, vocab, dtype=torch.float32, device=DEVICE)

    spec = _speculator(top_k, _no_top_p(num_reqs))
    sampled = DraftModelSpeculator.sample_draft(
        spec,
        hidden,
        positions,
        idx_mapping,
        temperature,
        seeds,
        torch.zeros((), dtype=torch.int64, device=DEVICE),
        cache,
    )

    support = torch.zeros(vocab, dtype=torch.bool, device=DEVICE)
    support[torch.topk(base, k).indices] = True
    assert support[sampled].all(), "a draft token was sampled outside top-k"
    # Every support token is reachable with 512 independent draws.
    assert set(sampled.unique().tolist()) == set(support.nonzero().flatten().tolist())
    assert (cache[:, 0, ~support] == float("-inf")).all()
    assert torch.equal(cache[:, 0, support], hidden[:, support])


def _first_tokens_after_verification(
    target_row: torch.Tensor, draft_row: torch.Tensor, target_k: int, mask_draft: bool
) -> tuple[torch.Tensor, float]:
    """Draft one token per request, verify it, and return the first emitted
    token of every request plus the acceptance rate."""
    vocab = target_row.numel()
    num_reqs = 200_000
    target_support = torch.zeros(vocab, dtype=torch.bool, device=DEVICE)
    target_support[torch.topk(target_row, target_k).indices] = True
    # The target sampler hands the rejection sampler top-k-processed logits.
    processed = target_row.masked_fill(~target_support, float("-inf"))

    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    top_k = torch.full((num_reqs,), target_k, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
    seeds = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE) * 7919 + 13
    draft_cache = torch.zeros(num_reqs, 1, vocab, dtype=torch.float32, device=DEVICE)

    spec = _speculator(top_k, _no_top_p(num_reqs)) if mask_draft else _speculator()
    draft_tokens = DraftModelSpeculator.sample_draft(
        spec,
        draft_row.expand(num_reqs, vocab).contiguous(),
        torch.full((num_reqs,), 1000, dtype=torch.int64, device=DEVICE),
        idx_mapping,
        temperature,
        seeds,
        torch.zeros((), dtype=torch.int64, device=DEVICE),
        draft_cache,
    )

    # Two logits per request: verify the draft token, then the bonus position.
    target_logits = processed.expand(2 * num_reqs, vocab).contiguous()
    draft_sampled = torch.zeros(2 * num_reqs, dtype=torch.int64, device=DEVICE)
    draft_sampled[1::2] = draft_tokens
    sampled, num_sampled = rejection_sample(
        target_logits=target_logits,
        draft_logits=draft_cache,
        draft_sampled=draft_sampled,
        cu_num_logits=torch.arange(
            0, 2 * num_reqs + 1, 2, dtype=torch.int32, device=DEVICE
        ),
        pos=torch.tensor([1000, 1001], dtype=torch.int64, device=DEVICE).repeat(
            num_reqs
        ),
        idx_mapping=idx_mapping,
        expanded_idx_mapping=idx_mapping.repeat_interleave(2),
        expanded_local_pos=torch.tensor(
            [0, 1], dtype=torch.int32, device=DEVICE
        ).repeat(num_reqs),
        temperature=temperature,
        seed=seeds,
        num_speculative_steps=1,
    )
    acceptance = (num_sampled == 2).float().mean().item()
    return sampled[:, 0], acceptance


def test_rejection_output_distribution_unchanged_and_acceptance_rises():
    torch.manual_seed(4)
    vocab, target_k = 64, 6
    target_row = torch.randn(vocab, device=DEVICE) * 1.5
    # A draft that agrees on the ranking at the top but spreads mass into the tail.
    draft_row = target_row * 0.6 + torch.randn(vocab, device=DEVICE) * 0.8

    support = torch.zeros(vocab, dtype=torch.bool, device=DEVICE)
    support[torch.topk(target_row, target_k).indices] = True
    expected = torch.softmax(target_row.masked_fill(~support, float("-inf")), dim=-1)

    results = {}
    for mask_draft in (False, True):
        first, acceptance = _first_tokens_after_verification(
            target_row, draft_row, target_k, mask_draft
        )
        assert support[first].all(), "emitted a token outside the target support"
        counts = torch.bincount(first, minlength=vocab).double()
        n = counts.sum()
        exp_counts = expected.double() * n
        keep = exp_counts > 0
        chi2 = (((counts - exp_counts) ** 2)[keep] / exp_counts[keep]).sum().item()
        dof = int(keep.sum().item()) - 1
        # Generous bound (~6 sigma for chi-square with this many degrees of freedom).
        assert chi2 < dof + 6 * math.sqrt(2 * dof), (
            f"mask_draft={mask_draft}: output distribution differs, chi2={chi2:.1f}"
        )
        results[mask_draft] = acceptance

    assert results[True] > results[False] + 0.02, results


def test_rows_without_top_k_are_left_alone():
    torch.manual_seed(7)
    vocab = 3072
    logits = torch.randn(3, vocab, device=DEVICE)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    top_k = torch.tensor([vocab, 7, vocab], dtype=torch.int32, device=DEVICE)
    top_p = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32, device=DEVICE)
    temperature = torch.ones(3, dtype=torch.float32, device=DEVICE)
    out = _apply_support(logits, idx_mapping, top_k, top_p, temperature)
    assert torch.equal(out[0], logits[0])
    assert (~out[1].isneginf()).sum() == 7
    assert torch.equal(out[2], logits[2])


def test_draft_params_follow_the_samplers_buffer_ring():
    """The sampler moves `top_k.gpu` to another buffer on every
    `apply_staged_writes()`. The drafter must see fresh values on every step,
    in tensors whose address does not change (CUDA graphs capture it)."""
    from vllm import SamplingParams
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    vocab, max_num_reqs, slot = 512, 8, 2
    states = SamplingStates(max_num_reqs, vocab)
    spec = SimpleNamespace(
        draft_logits=torch.empty(0),
        draft_top_k=None,
        draft_top_p=None,
        max_num_reqs=max_num_reqs,
        vocab_size=vocab,
        device=torch.device(DEVICE),
    )
    DraftModelSpeculator.set_draft_sampling_params(spec, states.top_k, states.top_p)
    captured_top_k, captured_top_p = spec.draft_top_k, spec.draft_top_p

    # A new request takes the slot on every step.
    for k, p in [(3, 0.9), (7, 0.8), (-1, 1.0), (5, 0.7), (11, 0.95)]:
        states.add_request(slot, SamplingParams(top_k=k, top_p=p))
        states.apply_staged_writes()
        DraftModelSpeculator._copy_draft_sampling_params(spec)
        torch.accelerator.synchronize()
        assert spec.draft_top_k is captured_top_k
        assert spec.draft_top_p is captured_top_p
        assert spec.draft_top_k[slot].item() == (k if k > 0 else vocab)
        assert spec.draft_top_p[slot].item() == pytest.approx(p)


def test_sample_draft_in_a_cuda_graph_follows_later_top_k_changes():
    """The drafter records sample_draft in CUDA graphs before any request
    arrives; the mask must still follow the requests' top-k / top-p when the
    graph is replayed."""
    torch.manual_seed(8)
    vocab, num_reqs, k = 512, 256, 3
    base = torch.randn(vocab, device=DEVICE)
    hidden = base.expand(num_reqs, vocab).contiguous()
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    top_k = _no_top_k(num_reqs, vocab)
    top_p = _no_top_p(num_reqs)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
    seeds = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
    positions = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
    step = torch.zeros((), dtype=torch.int64, device=DEVICE)
    cache = torch.zeros(num_reqs, 1, vocab, dtype=torch.float32, device=DEVICE)
    spec = _speculator(top_k, top_p)

    def run() -> torch.Tensor:
        return DraftModelSpeculator.sample_draft(
            spec, hidden, positions, idx_mapping, temperature, seeds, step, cache
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampled = run()

    # Requests arrive after capture: the first half uses top-k, the rest not.
    top_k[: num_reqs // 2] = k
    positions += 1
    graph.replay()
    torch.accelerator.synchronize()

    support = torch.zeros(vocab, dtype=torch.bool, device=DEVICE)
    support[torch.topk(base, k).indices] = True
    half = num_reqs // 2
    assert support[sampled[:half]].all(), "the replayed graph ignored top-k"
    assert (cache[:half, 0, ~support] == float("-inf")).all()
    assert not (cache[half:, 0] == float("-inf")).any(), "unmasked rows were masked"


@pytest.mark.parametrize("top_k, top_p", [(20, 1.0), (50, 0.9), (1024, 0.95)])
def test_large_vocab_keeps_at_least_the_support(top_k: int, top_p: float):
    """Past 4096 tokens the threshold comes from block maxima: it may keep more
    than the support, never less, and kept logits pass through untouched."""
    torch.manual_seed(9)
    vocab, num_rows = 151_936, 4
    # Peaked like a language model: nearly all the mass on a dozen tokens.
    logits = torch.randn(num_rows, vocab, device=DEVICE)
    logits[:, torch.randperm(vocab, device=DEVICE)[:12]] += 14
    idx_mapping = torch.arange(num_rows, dtype=torch.int32, device=DEVICE)
    top_k_t = torch.full((num_rows,), top_k, dtype=torch.int32, device=DEVICE)
    top_p_t = torch.full((num_rows,), top_p, dtype=torch.float32, device=DEVICE)
    temperature = torch.full((num_rows,), 0.8, dtype=torch.float32, device=DEVICE)
    out = _apply_support(logits, idx_mapping, top_k_t, top_p_t, temperature)
    for i in range(num_rows):
        expected = _reference_support(logits[i], min(top_k, vocab), top_p, 0.8)
        kept = ~out[i].isneginf()
        assert kept[expected].all(), f"row {i}: a support token was masked"
        assert torch.equal(out[i][kept], logits[i][kept])
        # Masking still happens: few tokens survive.
        assert kept.sum() < vocab // 100, int(kept.sum())


class _Recorder:
    """Stands in for the draft watermarker and the acceptance estimator and
    keeps the logits it was given."""

    def __init__(self) -> None:
        self.logits: torch.Tensor | None = None

    def sample(self, logits, sampled, idx_mapping, temperature):
        self.logits = logits.clone()
        return sampled

    def predict(self, logits, *args):
        self.logits = logits.clone()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_watermarker_gets_the_masked_logits_and_the_estimator_the_raw_ones(
    dtype: torch.dtype,
):
    """The watermarker resamples from the logits it is handed, so they carry the
    same support as the sampled and cached drafts. The acceptance estimator
    sees the logits before the mask, with or without a watermarker."""
    torch.manual_seed(11)
    vocab, num_reqs, k = 256, 16, 4
    hidden = (torch.randn(num_reqs, vocab, device=DEVICE) * 3).to(dtype)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    top_k = torch.full((num_reqs,), k, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
    cache = torch.zeros(num_reqs, 1, vocab, dtype=torch.float32, device=DEVICE)
    spec = _speculator(top_k, _no_top_p(num_reqs))
    spec.draft_watermarker = _Recorder()
    spec.acceptance_estimator = _Recorder()
    spec.draft_token_confidence_probs = None
    spec.temperature = temperature

    DraftModelSpeculator.sample_draft(
        spec,
        hidden,
        torch.arange(num_reqs, dtype=torch.int64, device=DEVICE),
        idx_mapping,
        temperature,
        torch.arange(num_reqs, dtype=torch.int64, device=DEVICE),
        torch.zeros((), dtype=torch.int64, device=DEVICE),
        cache,
    )

    masked = spec.draft_watermarker.logits
    kept = ~masked.isneginf()
    assert torch.equal(kept, ~cache[:, 0].isneginf())
    assert torch.equal(masked[kept], hidden[kept])
    assert (kept.sum(dim=-1) >= k).all()
    if dtype == torch.float32:
        assert (kept.sum(dim=-1) == k).all()
    assert torch.equal(spec.acceptance_estimator.logits, hidden)


def test_dspark_samples_and_caches_inside_support():
    """DSpark samples in its own _sample_logits, not in sample_draft."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    torch.manual_seed(12)
    vocab, num_reqs, k, num_steps = 256, 512, 4, 2
    base = torch.randn(vocab, device=DEVICE)
    logits = base.expand(num_reqs, vocab).contiguous()
    idx_map = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    top_k = torch.full((num_reqs,), k, dtype=torch.int32, device=DEVICE)
    spec = _speculator(top_k, _no_top_p(num_reqs))
    spec.draft_logits = torch.zeros(
        num_reqs, num_steps, vocab, dtype=torch.float32, device=DEVICE
    )
    spec._d2t_scatter_index = None
    spec._draft_scatter_buf = None
    spec.temperature = torch.ones(num_reqs, dtype=torch.float32, device=DEVICE)
    spec.seeds = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
    spec._step_cols = torch.arange(num_steps, dtype=torch.int64, device=DEVICE)
    sample_pos = torch.full((num_reqs,), 100, dtype=torch.int64, device=DEVICE)

    sampled = DSparkSpeculator._sample_logits(spec, logits, idx_map, sample_pos, 1)

    support = torch.zeros(vocab, dtype=torch.bool, device=DEVICE)
    support[torch.topk(base, k).indices] = True
    assert support[sampled].all(), "a draft token was sampled outside top-k"
    assert set(sampled.unique().tolist()) == set(support.nonzero().flatten().tolist())
    assert (spec.draft_logits[:, 1, ~support] == float("-inf")).all()
    assert torch.equal(spec.draft_logits[:, 1, support], logits[:, support])
