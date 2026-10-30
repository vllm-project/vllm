# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's fused DSpark Markov sampling against DSparkSpeculator's loop:
greedy tokens and the cached logits match, sampled tokens follow the softmax."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")

VOCAB, RANK, N_SPEC, REQS = 129280, 256, 5, 64


def _sample(fused: bool, temps: torch.Tensor, base: torch.Tensor, seed: int):
    from vllm.models.deepseek_v41.nvidia.ops.fused_markov import FusedMarkovSampler
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    torch.manual_seed(seed)
    g = torch.Generator(device="cuda").manual_seed(0)
    w1 = torch.randn(VOCAB, RANK, generator=g, device="cuda").to(base.dtype)
    w2 = (torch.randn(VOCAB, RANK, generator=g, device="cuda") * 0.05).to(base.dtype)
    s = DSparkSpeculator.__new__(DSparkSpeculator)
    s.model = SimpleNamespace(
        model=SimpleNamespace(
            markov_head=SimpleNamespace(markov_w1=SimpleNamespace(weight=w1))
        ),
        compute_draft_logits=lambda h: base,
        markov_embed=lambda ids: F.embedding(ids, w1),
        markov_bias=lambda e: F.linear(e, w2),
    )
    s.num_speculative_steps, s.temperature = N_SPEC, temps
    s._draft_topk = s.acceptance_estimator = s.draft_watermarker = None
    s._d2t_scatter_index, s.use_confidence_head, s.use_fp64_gumbel = None, False, False
    cache = torch.zeros(REQS, N_SPEC, VOCAB, dtype=base.dtype, device="cuda")
    s.draft_logits = SimpleNamespace(values=cache, row_mapping=None)
    s.seeds = torch.randint(0, 2**62, (REQS,), device="cuda")
    s._step_cols = torch.arange(N_SPEC, dtype=torch.int32, device="cuda")
    s.sample_indices = torch.arange(REQS * N_SPEC, device="cuda")
    s.sample_idx_mapping = torch.randperm(REQS, generator=g, device="cuda")
    s.sample_idx_mapping = s.sample_idx_mapping.repeat_interleave(N_SPEC)
    s.sample_pos = torch.arange(REQS * N_SPEC, device="cuda") + 1000
    ids = torch.randint(0, VOCAB, (REQS * N_SPEC,), generator=g, device="cuda")
    s.input_buffers = SimpleNamespace(input_ids=ids)
    s._anchor_idx = torch.arange(REQS, device="cuda") * N_SPEC
    s.draft_tokens = torch.zeros(REQS, N_SPEC, dtype=torch.int64, device="cuda")
    hidden = torch.empty(REQS * N_SPEC, 1, device="cuda")
    if fused:
        with torch.device("cuda"):
            sampler = FusedMarkovSampler(REQS, VOCAB, RANK, base.dtype)
        assert sampler(s, REQS, hidden)
    else:
        DSparkSpeculator._sample_sequential(s, REQS, hidden)
    return s.draft_tokens, cache, s.sample_idx_mapping[::N_SPEC]


def test_greedy_and_cache_match_loop():
    base = (torch.randn(REQS * N_SPEC, VOCAB, device="cuda") * 3).to(torch.bfloat16)
    temps = torch.tensor([0.0, 1.0], device="cuda").repeat(REQS // 2)
    ref_tok, ref_cache, slots = _sample(False, temps, base, 1)
    tok, cache, _ = _sample(True, temps, base, 1)
    greedy = temps[slots] == 0
    assert torch.equal(tok[greedy], ref_tok[greedy])
    assert torch.equal(cache[temps == 0], ref_cache[temps == 0])
    # Sampled requests draw other tokens than Gumbel-max, so only step 0 agrees.
    assert torch.equal(cache[:, 0], ref_cache[:, 0])


@pytest.mark.parametrize("temp", [0.6, 1.0])
def test_sampled_tokens_follow_softmax(temp):
    row = (torch.randn(VOCAB, device="cuda") * 3).to(torch.bfloat16)
    base = row.expand(REQS * N_SPEC, VOCAB).contiguous()
    temps = torch.full((REQS,), temp, device="cuda")
    counts = torch.zeros(VOCAB, device="cuda")
    expected = torch.zeros(VOCAB, device="cuda")
    for seed in range(40):
        tok, cache, slots = _sample(True, temps, base, seed)
        counts += torch.bincount(tok[:, 0], minlength=VOCAB)
        expected += torch.softmax(cache[slots, 0].float() / temp, -1).sum(0)
    n = counts.sum()
    top = expected.topk(10).indices
    p = expected[top] / n
    z = (counts[top] / n - p) / (p * (1 - p) / n).sqrt()
    assert z.abs().max() < 5, z
