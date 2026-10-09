# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fused_markov.sample_sequential against vLLM's
DSparkSpeculator._sample_sequential.

Both paths run on the same random Markov head, base logits and sampling
state, for 1 and 2 requests at temperatures 0, 0.6 and 1.0. vLLM's bias is
the server's GEMV (rocm_unquantized_gemm). The fused launch adds the bias's
256 products in another order, so a bias value can differ in its last bf16
bit, and then a draft token can differ. That changes only what the draft
proposes, because the rejection sampler checks every draft token against the
target. So the test allows at most 1 in 1000 cached logits to differ, and a
draft token may differ only when a cached logit differed.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the fused launch is for ROCm"
)

VOCAB, RANK, N_SPEC, MAX_REQS = 129280, 256, 5, 4


def _set_gate(monkeypatch, on: bool):
    from vllm.model_executor.layers.dsv41_gfx942 import enabled

    if on:
        monkeypatch.setenv("VLLM_ROCM_MONO_DECODE", "1")
    else:
        monkeypatch.delenv("VLLM_ROCM_MONO_DECODE", raising=False)
    enabled.cache_clear()


class _MarkovW2(torch.nn.Module):
    def __init__(self, w):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()


class _Model:
    """The parts of the DSpark draft model that the sampling uses. The base
    logits are fixed, so both paths sample from the same ones."""

    def __init__(self, w1, w2, base):
        head = torch.nn.Module()
        head.markov_w1 = torch.nn.Embedding(VOCAB, RANK, _weight=w1)
        head.markov_w2 = _MarkovW2(w2)
        self.model = SimpleNamespace(markov_head=head)
        self.base = base

    def compute_draft_logits(self, hidden):
        return self.base[: hidden.shape[0]]

    def markov_embed(self, ids):
        return self.model.markov_head.markov_w1(ids)

    def markov_bias(self, e):
        from vllm.model_executor.layers.utils import rocm_unquantized_gemm

        w2 = self.model.markov_head.markov_w2
        return rocm_unquantized_gemm(w2, e, w2.weight, None)

    def map_draft_to_target(self, ids):
        return ids


def _speculator(temp: float, seed: int):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    dev = torch.device("cuda")
    bf16 = torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(seed)
    w1 = torch.randn(VOCAB, RANK, generator=g, device=dev).to(bf16)
    w2 = (torch.randn(VOCAB, RANK, generator=g, device=dev) * 0.05).to(bf16)
    base = torch.randn(MAX_REQS * N_SPEC, VOCAB, generator=g, device=dev) * 3
    s = DSparkSpeculator.__new__(DSparkSpeculator)
    s.model = _Model(w1, w2, base.to(bf16))
    s.num_speculative_steps = N_SPEC
    s._draft_topk = None
    s.use_confidence_head = False
    s.acceptance_estimator = None
    s.draft_watermarker = None
    s._d2t_scatter_index = None
    s._draft_scatter_buf = None
    s.use_fp64_gumbel = False
    s.draft_logits = torch.zeros(MAX_REQS, N_SPEC, VOCAB, dtype=bf16, device=dev)
    s.temperature = torch.full((MAX_REQS,), temp, dtype=torch.float32, device=dev)
    s.seeds = torch.randint(0, 2**62, (MAX_REQS,), generator=g, device=dev)
    s._step_cols = torch.arange(N_SPEC, dtype=torch.int32, device=dev)
    s.sample_indices = torch.arange(MAX_REQS * N_SPEC, dtype=torch.int64, device=dev)
    # Request slots in another order than the rows, and positions past a
    # long context, as in the server.
    slots = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=dev)
    s.sample_idx_mapping = slots.repeat_interleave(N_SPEC)
    ctx = torch.tensor([131072, 40000, 7, 99999], dtype=torch.int64, device=dev)
    steps = torch.arange(N_SPEC, device=dev)
    s.sample_pos = (ctx[:, None] + 1 + steps[None, :]).flatten()
    s.input_buffers = SimpleNamespace(
        input_ids=torch.randint(0, VOCAB, (MAX_REQS * N_SPEC,), generator=g, device=dev)
    )
    s._anchor_idx = torch.arange(MAX_REQS, dtype=torch.int64, device=dev) * N_SPEC
    s.draft_tokens = torch.zeros(MAX_REQS, N_SPEC, dtype=torch.int64, device=dev)
    return s


@pytest.mark.parametrize("temp", [0.0, 0.6, 1.0])
@pytest.mark.parametrize("num_reqs", [1, 2])
def test_fused_markov_matches_vllm_loop(monkeypatch, num_reqs, temp):
    from vllm.v1.worker.gpu.spec_decode.dspark import fused_markov
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    s = _speculator(temp, seed=int(temp * 10) + num_reqs)
    hidden = torch.zeros(MAX_REQS * N_SPEC, 8, device="cuda")

    _set_gate(monkeypatch, on=False)
    DSparkSpeculator._sample_sequential(s, num_reqs, hidden)
    torch.accelerator.synchronize()
    ref_tok, ref_cache = s.draft_tokens.clone(), s.draft_logits.clone()

    s.draft_tokens.zero_()
    s.draft_logits.zero_()
    _set_gate(monkeypatch, on=True)
    try:
        assert fused_markov.sample_sequential(s, num_reqs, hidden)
        torch.accelerator.synchronize()
    finally:
        _set_gate(monkeypatch, on=False)
    got_tok, got_cache = s.draft_tokens, s.draft_logits

    slots = s.sample_idx_mapping[: num_reqs * N_SPEC : N_SPEC].long()
    rc, gc = ref_cache[slots], got_cache[slots]
    cache_diff = int((rc.view(torch.int16) != gc.view(torch.int16)).sum())
    tok_diff = int((ref_tok[:num_reqs] != got_tok[:num_reqs]).sum())
    assert cache_diff <= rc.numel() // 1000, (
        f"{cache_diff} of {rc.numel()} cached logits differ"
    )
    assert tok_diff == 0 or cache_diff > 0, (
        f"{tok_diff} draft tokens differ while every cached logit is equal"
    )
