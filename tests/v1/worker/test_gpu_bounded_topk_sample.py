# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the bounded top-k sampler (aiter topk_select + survivors kernel).

The bounded path must return the token the stock path (full-vocab top-k/top-p
mask + `gumbel_sample`) returns for the same seed and position, except where
the k-th logit is tied and the two selectors keep different tied tokens.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("triton")
if not current_platform.is_rocm():
    pytest.skip("ROCm aiter topk_select required", allow_module_level=True)

from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.topk_sample import (
    bounded_topk_available,
    bounded_topk_sample,
)

if not bounded_topk_available():
    pytest.skip("aiter topk_select unavailable", allow_module_level=True)

DEVICE = current_platform.device_type
VOCAB = 32768


def _inputs(num_tokens: int, top_k: int, top_p: float, temperature: float):
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    # Distinct fp32 values so the k-th logit is never tied.
    logits = torch.randn(num_tokens, VOCAB, device=DEVICE, generator=gen) * 3
    mapping = torch.arange(num_tokens, device=DEVICE, dtype=torch.int32)
    k = torch.full((num_tokens,), top_k, device=DEVICE, dtype=torch.int32)
    p = torch.full((num_tokens,), top_p, device=DEVICE, dtype=torch.float32)
    t = torch.full((num_tokens,), temperature, device=DEVICE, dtype=torch.float32)
    seeds = torch.randint(-(2**60), 2**60, (num_tokens,), device=DEVICE)
    pos = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64) + 100
    return logits, mapping, k, p, t, seeds, pos


@pytest.mark.parametrize("top_k,top_p", [(64, 0.95), (64, 1.0), (8, 0.5), (128, 0.9)])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_matches_stock_path(top_k, top_p, temperature):
    logits, mapping, k, p, t, seeds, pos = _inputs(512, top_k, top_p, temperature)
    scaled = logits / temperature
    masked = apply_top_k_top_p_triton(scaled.clone(), k, p if top_p < 1.0 else None)
    expected = gumbel_sample(
        masked, mapping, t, seeds, pos, apply_temperature=False, is_drafting=False
    )
    actual = bounded_topk_sample(
        logits, top_k, mapping, k, p if top_p < 1.0 else None, t, seeds, pos
    )
    assert (actual == expected).float().mean().item() >= 0.99


def test_greedy_rows_take_the_argmax():
    logits, mapping, k, p, t, seeds, pos = _inputs(64, 64, 0.95, 0.0)
    actual = bounded_topk_sample(logits, 64, mapping, k, p, t, seeds, pos)
    assert torch.equal(actual, logits.argmax(dim=-1))


def test_mixed_per_request_top_k_stays_in_support():
    logits, mapping, _, p, t, seeds, pos = _inputs(256, 64, 0.95, 1.0)
    k = torch.where(
        torch.arange(256, device=DEVICE) % 2 == 0,
        torch.tensor(4, device=DEVICE),
        torch.tensor(64, device=DEVICE),
    ).to(torch.int32)
    sampled = bounded_topk_sample(logits, 64, mapping, k, p, t, seeds, pos)
    rank = (logits > logits.gather(1, sampled[:, None])).sum(dim=1)
    assert torch.all(rank < k)


def test_greedy_ties_take_the_lowest_token_id():
    logits, mapping, k, p, t, seeds, pos = _inputs(64, 64, 0.95, 0.0)
    logits[:, [4000, 100, 20000]] = 1000.0
    for _ in range(3):
        actual = bounded_topk_sample(logits, 64, mapping, k, p, t, seeds, pos)
        assert torch.all(actual == 100)
