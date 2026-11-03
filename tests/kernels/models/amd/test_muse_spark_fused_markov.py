# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark fused_markov.sample_sequential against vLLM's
DSparkSpeculator._sample_sequential.

Both paths run on the same Markov head (random, or the released draft's when
MUSE_DSPARK_DRAFT names its directory), base logits and sampling state, for 1
to 16 requests: probabilistic drafts (the recipe) at temperatures 0, 0.6 and
1.0 with fp32 and fp64 Gumbel noise, and greedy drafts. vLLM's loop runs
Muse's output transform and its d2t scatter buffer as in the server. Its
bias is the server's GEMV. The fused launch adds the bias's 512 products in
another order, so a bias value can differ in its last bf16 bit, and then a
cached logit or a draft token can differ. The rejection sampler checks every
draft token against the target, so this changes only what the draft
proposes. Against vLLM's loop the test allows at most 1 in 1000 cached logits
of each request and round to differ, and a draft token may differ only when a
cached logit of its own round differed.

The sampling itself is checked exactly: each fused draft token must be the
token that vLLM's sampler (DSparkSpeculator._sample_logits) draws from the
logits that the fused launch cached for that request and round.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the fused launch is for ROCm"
)

VOCAB, RANK, N_SPEC, MAX_REQS = 202048, 512, 7, 16
NUM_REQS = [1, 2, 3, 4, 5, 8, 16]
DRAFT = os.environ.get("MUSE_DSPARK_DRAFT", "")


class _MarkovW2(torch.nn.Module):
    def __init__(self, w):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()


class _Model:
    """The parts of the Muse DSpark draft model that the sampling uses. The
    base logits are fixed, so both paths sample from the same ones."""

    def __init__(self, w1, w2, base):
        from vllm.models.muse_spark.common.dspark import MuseSparkDSparkForCausalLM

        head = torch.nn.Module()
        head.markov_w1 = torch.nn.Embedding(VOCAB, RANK, _weight=w1)
        head.markov_w2 = _MarkovW2(w2)
        self.model = SimpleNamespace(markov_head=head)
        self.lm_head = SimpleNamespace(weight=w2[:1])
        self.logits_processor = SimpleNamespace(head_dtype=torch.bfloat16)
        self.base = base
        # The released draft's: metap "sp" at width 8192 / 256, soft cap 20.
        self.output_multiplier = (8192 / 256) ** -0.5
        self.output_soft_cap_temp = 20.0
        self.d2t_is_identity = True
        self._transform = MuseSparkDSparkForCausalLM.apply_output_logit_transform

    def compute_draft_logits(self, hidden):
        return self.base[: hidden.shape[0]]

    def markov_embed(self, ids):
        return self.model.markov_head.markov_w1(ids)

    def markov_bias(self, e):
        w2 = self.model.markov_head.markov_w2
        return w2.quant_method.apply(w2, e)

    def apply_output_logit_transform(self, logits):
        return self._transform(self, logits)

    def map_draft_to_target(self, ids):
        return ids


def _weights(g, dev, exact=False):
    if exact:
        # k / 16 with k in -2..2: every partial sum of the bias is exact in
        # fp32, so the bias does not depend on the order of the sum.
        w = torch.randint(-2, 3, (2, VOCAB, RANK), generator=g, device=dev)
        return (w / 16).to(torch.bfloat16).unbind(0)
    if DRAFT:
        from safetensors import safe_open

        with safe_open(os.path.join(DRAFT, "model.safetensors"), "pt") as f:
            w1 = f.get_tensor("markov_head.markov_w1.weight")
            w2 = f.get_tensor("markov_head.markov_w2.weight")
        return w1.to(dev), w2.to(dev)
    w1 = torch.randn(VOCAB, RANK, generator=g, device=dev).to(torch.bfloat16)
    w2 = (torch.randn(VOCAB, RANK, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    return w1, w2


def _speculator(temp, seed, greedy=False, fp64=False, exact=False):
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    dev = torch.device("cuda")
    bf16 = torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(seed)
    w1, w2 = _weights(g, dev, exact)
    base = torch.randn(MAX_REQS * N_SPEC, VOCAB, generator=g, device=dev) * 16
    s = DSparkSpeculator.__new__(DSparkSpeculator)
    s.model = _Model(w1, w2, base.to(bf16))
    s.num_speculative_steps = N_SPEC
    s._draft_topk = None
    s.use_confidence_head = False
    s.acceptance_estimator = None
    s.draft_watermarker = None
    s.use_fp64_gumbel = fp64
    s.draft_logits = None
    s._d2t_scatter_index = None
    if not greedy:
        s.draft_logits = DraftLogitsCache(
            MAX_REQS, N_SPEC, VOCAB, dtype=bf16, fill=0.0, device=dev
        )
        # The server's d2t scatter (d2t = 0) into the cache's dtype.
        s._d2t_scatter_index = torch.arange(VOCAB, device=dev)
        s._draft_scatter_buf = torch.full(
            (MAX_REQS, VOCAB), float("-inf"), dtype=bf16, device=dev
        )
    s.temperature = torch.full((MAX_REQS,), temp, dtype=torch.float32, device=dev)
    s.seeds = torch.randint(0, 2**62, (MAX_REQS,), generator=g, device=dev)
    s._step_cols = torch.arange(N_SPEC, dtype=torch.int32, device=dev)
    s.sample_indices = torch.arange(MAX_REQS * N_SPEC, dtype=torch.int64, device=dev)
    # Request slots in another order than the rows, and positions past a
    # long context, as in the server.
    slots = torch.randperm(MAX_REQS, generator=g, device=dev).int()
    s.sample_idx_mapping = slots.repeat_interleave(N_SPEC)
    ctx = torch.randint(7, 1 << 18, (MAX_REQS,), generator=g, device=dev)
    steps = torch.arange(N_SPEC, device=dev)
    s.sample_pos = (ctx[:, None] + 1 + steps[None, :]).flatten()
    s.input_buffers = SimpleNamespace(
        input_ids=torch.randint(0, VOCAB, (MAX_REQS * N_SPEC,), generator=g, device=dev)
    )
    s._anchor_idx = torch.arange(MAX_REQS, dtype=torch.int64, device=dev) * N_SPEC
    s.draft_tokens = torch.zeros(MAX_REQS, N_SPEC, dtype=torch.int64, device=dev)
    return s


def _run_both(s, num_reqs):
    from vllm.models.muse_spark.amd.ops import fused_markov
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    hidden = torch.zeros(MAX_REQS * N_SPEC, 8, device="cuda")
    DSparkSpeculator._sample_sequential(s, num_reqs, hidden)
    torch.accelerator.synchronize()
    ref_tok = s.draft_tokens.clone()
    ref_cache = s.draft_logits.values.clone() if s.draft_logits is not None else None
    s.draft_tokens.zero_()
    if s.draft_logits is not None:
        s.draft_logits.values.zero_()
    assert fused_markov.sample_sequential(s, num_reqs, hidden)
    torch.accelerator.synchronize()
    got_cache = s.draft_logits.values.clone() if s.draft_logits is not None else None
    return ref_tok, s.draft_tokens.clone(), ref_cache, got_cache


@pytest.mark.parametrize("fp64", [False, True])
@pytest.mark.parametrize("temp", [0.0, 0.6, 1.0])
@pytest.mark.parametrize("num_reqs", NUM_REQS)
def test_fused_markov_matches_vllm_loop(num_reqs, temp, fp64):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    s = _speculator(temp, seed=int(temp * 10) + num_reqs, fp64=fp64)
    ref_tok, got_tok, ref_cache, got_cache = _run_both(s, num_reqs)

    # rc and gc are [num_reqs, N_SPEC, VOCAB]: the cached logits of each
    # request and round. The differences are counted for each request and
    # round, so that a rounding difference in one round cannot excuse a wrong
    # token in another.
    slots = s.sample_idx_mapping[: num_reqs * N_SPEC : N_SPEC].long()
    rc, gc = ref_cache[slots], got_cache[slots]
    step_diff = (rc.view(torch.int16) != gc.view(torch.int16)).sum(dim=-1)
    assert int(step_diff.max()) <= VOCAB // 1000, (
        f"cached logits that differ, by request and round: {step_diff.tolist()}"
    )
    tok_diff = ref_tok[:num_reqs] != got_tok[:num_reqs]
    unexplained = (tok_diff & (step_diff == 0)).nonzero().tolist()
    assert not unexplained, (
        f"draft tokens differ at (request, round) {unexplained}, "
        "while every cached logit of that round is equal"
    )

    # Each fused token must be exactly the token that vLLM's sampler draws
    # from the logits that the fused launch cached for its request and round,
    # with the same request slot, seed, position, temperature and cache
    # column. _sample_logits writes the logits it gets back into the same
    # cache entries, so the checks above use copies.
    idx_map = s.sample_idx_mapping[: num_reqs * N_SPEC].view(num_reqs, N_SPEC)
    pos = s.sample_pos[: num_reqs * N_SPEC].view(num_reqs, N_SPEC)
    for k in range(N_SPEC):
        drawn = DSparkSpeculator._sample_logits(
            s, gc[:, k], idx_map[:, k], pos[:, k], k
        )
        fused = got_tok[:num_reqs, k]
        assert torch.equal(drawn, fused), (
            f"round {k}: the fused launch drew {fused.tolist()}, "
            f"vLLM's sampler draws {drawn.tolist()} from the same logits"
        )


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("num_reqs", [1, 2, 4, 8])
def test_fused_markov_bit_exact_with_exact_bias(num_reqs, greedy):
    # With a bias that does not depend on the order of its sum, the fused
    # launches give the same draft tokens and cache bits as vLLM's loop.
    s = _speculator(1.0, seed=30 + num_reqs, greedy=greedy, exact=True)
    ref_tok, got_tok, ref_cache, got_cache = _run_both(s, num_reqs)
    assert torch.equal(ref_tok, got_tok)
    if not greedy:
        assert torch.equal(ref_cache.view(torch.int16), got_cache.view(torch.int16))


@pytest.mark.parametrize("num_reqs", NUM_REQS)
def test_fused_markov_greedy_drafts(num_reqs):
    s = _speculator(0.0, seed=20 + num_reqs, greedy=True)
    ref_tok, got_tok, _, _ = _run_both(s, num_reqs)
    assert torch.equal(ref_tok, got_tok), (
        f"greedy drafts: vLLM {ref_tok[:num_reqs].tolist()}, "
        f"fused {got_tok[:num_reqs].tolist()}"
    )


def test_fused_markov_falls_back():
    from vllm.models.muse_spark.amd.ops import fused_markov

    s = _speculator(1.0, seed=3)
    hidden = torch.zeros(MAX_REQS * N_SPEC, 8, device="cuda")
    assert not fused_markov.sample_sequential(s, fused_markov.MAX_ROWS + 1, hidden)
    s.model.d2t_is_identity = False
    assert not fused_markov.sample_sequential(s, 1, hidden)
    assert not s.draft_tokens.any()
