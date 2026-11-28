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
    s._fuse_markov_bias = False
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


def test_sampled_tokens_follow_softmax():
    temp = 0.6
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


@pytest.mark.parametrize("rows", [1, 4, 17])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("mixed_temperature", [False, True])
@torch.inference_mode()
def test_confidence_uses_proposal_input_after_request_reorder(
    rows, use_graph, mixed_temperature, dist_init
):
    """AV saves the previous embedding, not the sampled token or final scratch.

    Replaying after slot/anchor changes must update request-major confidence
    inputs while preserving unmapped cache slots and AV-off CDF proposals.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead
    from vllm.models.deepseek_v41.nvidia.ops.fused_markov import FusedMarkovSampler

    vocab, rank, steps, hidden_size = 4097, 32, 5, 16
    capacity = rows + 3
    torch.manual_seed(117)
    with torch.device("cuda"), set_current_vllm_config(VllmConfig()):
        embedding = torch.nn.Embedding(vocab, rank, dtype=torch.bfloat16)
        bias = torch.nn.Linear(rank, vocab, bias=False, dtype=torch.bfloat16)
        confidence_head = DSparkConfidenceHead(hidden_size + rank, "confidence_head")
        confidence_head.proj.weight.normal_(0, 0.03)
        samplers = [
            FusedMarkovSampler(
                capacity,
                vocab,
                rank,
                torch.bfloat16,
                confidence_steps=steps if av else 0,
            )
            for av in [False, True]
        ]
        base = torch.randn(rows * steps, vocab, dtype=torch.bfloat16)
        hidden = torch.randn(rows * steps, hidden_size, dtype=torch.bfloat16)
        anchors = torch.randint(vocab, (rows,))
        anchor_embeds = embedding(anchors)
        slots = torch.randperm(capacity)[:rows].clone()
        if rows > 1:
            slots[-1] = -1
        mapping = slots.repeat_interleave(steps)
        positions = torch.arange(rows * steps, dtype=torch.int64) + 37
        temps = torch.zeros(capacity)
        if mixed_temperature:
            temps[::2] = 0.6
            temps[1::2] = 1.0
        seeds = torch.arange(capacity, dtype=torch.int64) * 19 + 73
        caches = [
            torch.full((capacity, steps, vocab), 123, dtype=base.dtype)
            for _ in samplers
        ]
        outputs = [
            torch.full((capacity, steps), -99, dtype=torch.int64) for _ in samplers
        ]
        confidence = torch.empty(rows * steps, dtype=torch.float32)

    def run(i):
        embeds = samplers[i].sample(
            base,
            anchor_embeds,
            bias,
            embedding.weight,
            mapping,
            positions,
            temps,
            seeds,
            caches[i],
            outputs[i][:rows],
        )
        if embeds is not None:
            confidence.copy_(
                torch.sigmoid(confidence_head(hidden, embeds.flatten(0, 1)))
            )

    graphs = []
    for i in range(2):
        for _ in range(3):
            run(i)
        if use_graph:
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(i)
            graphs.append(graph)
    history = samplers[1].confidence_embeds
    assert history is not None and samplers[0].confidence_embeds is None
    address = history.data_ptr()
    for replay in range(3):
        anchors.copy_((anchors + 13 + replay) % vocab)
        anchor_embeds.copy_(embedding(anchors))
        slots.copy_(torch.where(slots >= 0, (slots + 1) % capacity, slots))
        mapping.copy_(slots.repeat_interleave(steps))
        positions.add_(1)
        for cache, output in zip(caches, outputs):
            cache.fill_(123)
            output.fill_(-99)
        history.fill_(float("nan"))
        confidence.fill_(float("nan"))
        for i in range(2):
            if use_graph:
                graphs[i].replay()
            else:
                run(i)
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(caches[0], caches[1])
        prior_tokens = torch.cat((anchors[:, None], outputs[1][:rows, :-1]), dim=1)
        expected_embeds = embedding(prior_tokens)
        assert torch.equal(history[:rows], expected_embeds)
        assert history.data_ptr() == address
        assert history[rows:].isnan().all()
        expected = torch.sigmoid(confidence_head(hidden, expected_embeds.flatten(0, 1)))
        assert torch.equal(confidence, expected)
        valid = slots >= 0
        for step in range(steps):
            expected_logits = base.view(rows, steps, vocab)[:, step] + bias(
                expected_embeds[:, step]
            )
            assert torch.equal(caches[1][slots[valid], step], expected_logits[valid])
        untouched = torch.ones(capacity, dtype=torch.bool, device="cuda")
        untouched[slots[valid]] = False
        assert (caches[1][untouched] == 123).all()
        assert (outputs[1][rows:] == -99).all()
