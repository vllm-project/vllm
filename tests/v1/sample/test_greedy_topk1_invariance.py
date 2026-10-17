# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for vllm-project/vllm#5404 — V1 sampler determinism.

`top_k=1` (any temperature) must pick the same token as `temperature=0` on
identical logits. Before the fix, `apply_top_k_only` kept all top-tied tokens
and the downstream `probs.div(q).argmax` (Gumbel-max) could pick a different
tied token from the row's argmax.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

VOCAB_SIZE = 32


def _make_metadata(
    *,
    temperature: list[float],
    top_k: list[int] | None,
    top_p: list[float] | None,
) -> SamplingMetadata:
    batch_size = len(temperature)
    temp_t = torch.tensor(temperature, dtype=torch.float32)
    eps = 1e-5
    all_greedy = bool((temp_t < eps).all().item())
    all_random = bool((temp_t >= eps).all().item())
    top_k_t = torch.tensor(top_k, dtype=torch.int64) if top_k is not None else None
    top_p_t = torch.tensor(top_p, dtype=torch.float32) if top_p is not None else None
    return SamplingMetadata(
        temperature=temp_t,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=top_p_t,
        top_k=top_k_t,
        has_top_k_one=bool(top_k is not None and 1 in top_k),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch_size),
        presence_penalties=torch.zeros(batch_size),
        repetition_penalties=torch.ones(batch_size),
        output_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _tied_top_logits(dtype: torch.dtype) -> torch.Tensor:
    """One row with two tokens tied at the maximum, others strictly lower."""
    row = [1.0, 1.0, 0.5, 0.0, -1.0, -2.0] + [-3.0] * (VOCAB_SIZE - 6)
    return torch.tensor([row], dtype=dtype)


@pytest.mark.parametrize("temperature", [0.1, 0.7, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_topk1_matches_greedy_on_tied_logits(temperature: float, dtype: torch.dtype):
    logits_fp = _tied_top_logits(dtype).to(torch.float32)
    greedy_token = int(logits_fp.argmax(dim=-1).item())

    sampler = Sampler()
    md = _make_metadata(temperature=[temperature], top_k=[1], top_p=None)

    seen: set[int] = set()
    for seed in range(32):
        torch.manual_seed(seed)
        sampled, _ = sampler.sample(logits_fp.clone(), md)
        seen.add(int(sampled[0].item()))
    assert seen == {greedy_token}, (
        f"top_k=1 (T={temperature}, dtype={dtype}) sampled {seen}, "
        f"expected only {greedy_token} (= argmax)"
    )


def test_topk1_in_mixed_batch_matches_temperature_zero():
    """A single row's logits processed via temperature=0 vs top_k=1 in the
    same mixed batch must pick the same token across many seeds."""
    row = _tied_top_logits(torch.float32)[0].tolist()
    # Two rows with identical logits: row 0 greedy via temp=0, row 1 via top_k=1.
    logits = torch.tensor([row, row], dtype=torch.float32)
    md = _make_metadata(
        temperature=[0.0, 0.7],
        top_k=[VOCAB_SIZE, 1],
        top_p=None,
    )
    sampler = Sampler()
    for seed in range(32):
        torch.manual_seed(seed)
        sampled, _ = sampler.sample(logits.clone(), md)
        assert sampled[0].item() == sampled[1].item(), (
            f"seed={seed}: row 0 (T=0) picked {sampled[0].item()} but "
            f"row 1 (top_k=1) picked {sampled[1].item()}"
        )


@pytest.mark.parametrize("top_p", [0.5, 0.9, 0.99])
def test_topk1_with_topp_still_matches_greedy(top_p: float):
    logits = _tied_top_logits(torch.float32)
    greedy_token = int(logits.argmax(dim=-1).item())

    md = _make_metadata(
        temperature=[0.7],
        top_k=[1],
        top_p=[top_p],
    )
    sampler = Sampler()
    seen: set[int] = set()
    for seed in range(16):
        torch.manual_seed(seed)
        sampled, _ = sampler.sample(logits.clone(), md)
        seen.add(int(sampled[0].item()))
    assert seen == {greedy_token}


def test_pure_random_batch_with_topk1_row():
    """`all_random=True` (no temperature=0 rows) but a top_k=1 row is present.
    Without the fix, `greedy_sampled` would not be computed and the top_k=1
    row would go through the Gumbel-max path; with the fix it picks argmax.
    """
    row = _tied_top_logits(torch.float32)[0].tolist()
    logits = torch.tensor([row, row], dtype=torch.float32)
    md = _make_metadata(
        temperature=[0.7, 0.7],
        top_k=[VOCAB_SIZE, 1],
        top_p=None,
    )
    assert md.all_random is True
    greedy_token = int(logits[0].argmax().item())

    sampler = Sampler()
    topk1_tokens: set[int] = set()
    for seed in range(32):
        torch.manual_seed(seed)
        sampled, _ = sampler.sample(logits.clone(), md)
        topk1_tokens.add(int(sampled[1].item()))
    assert topk1_tokens == {greedy_token}


def test_topk_greater_than_one_is_unaffected():
    """top_k>1 rows must still go through the random path. Use untied logits
    so there's a single argmax, but place a near-second close enough that the
    Gumbel-max of the (filtered) softmax has positive probability of picking
    it on at least one seed."""
    row = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5] + [-3.0] * (VOCAB_SIZE - 6)
    logits = torch.tensor([row], dtype=torch.float32)
    md = _make_metadata(
        temperature=[1.0],
        top_k=[3],
        top_p=None,
    )
    sampler = Sampler()
    seen: set[int] = set()
    for seed in range(64):
        torch.manual_seed(seed)
        sampled, _ = sampler.sample(logits.clone(), md)
        seen.add(int(sampled[0].item()))
    # top_k=3 should produce more than one distinct outcome over 64 seeds —
    # otherwise we accidentally turned everything into greedy.
    assert len(seen) > 1, (
        f"top_k=3 produced only {seen} across 64 seeds — random path broken"
    )
    # And the picked tokens must lie within the top-3 set {0, 1, 2}.
    assert seen <= {0, 1, 2}
