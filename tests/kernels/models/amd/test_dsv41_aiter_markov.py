# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AITER FlyDSL DSpark Markov sampler call site of the DeepSeek-V4.1 draft
model (ops/aiter_markov.py), driven through
DSparkSpeculator._sample_sequential.

The gating tests need no AITER op: one replaces the op module with a stand-in
that records what it is asked, the other removes it. The GPU test needs an
AITER with `aiter.ops.flydsl.markov_chain_sample` on gfx950 and compares the
op's draft tokens and draft logits cache with vLLM's Markov loop.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_rocm(), reason="the AITER call site is for ROCm"
    ),
    # The Markov head's LogitsProcessor reads the current vLLM config.
    pytest.mark.usefixtures("default_vllm_config"),
]

VOCAB, RANK, N_SPEC = 129280, 256, 5
OP_MODULE = "aiter.ops.flydsl.markov_chain_sample"
# Fill of the draft logits cache, so rows the sampler must not write show.
SENTINEL = -7.0


@pytest.fixture
def markov_flag(monkeypatch):
    """Sets VLLM_ROCM_USE_AITER_FLYDSL_MARKOV_SAMPLER (with VLLM_ROCM_USE_AITER)
    and re-reads rocm_aiter_ops's cached env values, also after the test."""
    from vllm._aiter_ops import rocm_aiter_ops

    def set_flag(on: bool) -> None:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv(
            "VLLM_ROCM_USE_AITER_FLYDSL_MARKOV_SAMPLER", "1" if on else "0"
        )
        rocm_aiter_ops.refresh_env_variables()

    yield set_flag
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


def _model(vocab: int, device: str, seed: int, base: torch.Tensor | None = None):
    """The DeepSeek-V4.1 draft model with a random Markov head. Its base logits
    are `base`, so that every path samples from the same ones."""
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
    from vllm.models.deepseek_v41.amd.dspark import DSparkDeepseekV4ForCausalLM

    g = torch.Generator(device=device).manual_seed(seed)
    head = DSparkMarkovHead(vocab, vocab, RANK, prefix="markov_head")
    head = head.to(device=device, dtype=torch.bfloat16)
    head.markov_w1.weight.data.copy_(
        torch.randn(vocab, RANK, generator=g, device=device)
    )
    head.markov_w2.weight.data.copy_(
        torch.randn(vocab, RANK, generator=g, device=device) * 0.05
    )
    model = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace(markov_head=head)
    model.logits_processor = LogitsProcessor(vocab)
    if base is None:
        base = torch.zeros(0, vocab, dtype=torch.bfloat16, device=device)
    model.compute_draft_logits = lambda hidden: base[: hidden.shape[0]]  # type: ignore[method-assign]
    return model


def _speculator(model, num_reqs: int, vocab: int, device: str, seed: int):
    """A DSparkSpeculator of DeepSeek-V4.1's probabilistic drafting (5 steps,
    full vocabulary, replicated bf16 draft logits cache), as `load_model`
    leaves it. Requests take shuffled slots of a larger cache, mixed
    temperatures and positions past a long context. With more than one
    request, the last one is an inert padding row (slot -1)."""
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    g = torch.Generator(device=device).manual_seed(seed)
    num_slots = num_reqs + 3
    s = DSparkSpeculator.__new__(DSparkSpeculator)
    s.model = model
    s.device = torch.device(device)
    s.num_speculative_steps = N_SPEC
    s.vocab_size = vocab
    s.sample_from_anchor = True
    s._draft_topk = None
    s.use_confidence_head = False
    s.acceptance_estimator = None
    s.draft_watermarker = None
    s._d2t_scatter_index = None
    s._draft_scatter_buf = None
    s._fuse_markov_bias = True
    s.use_fp64_gumbel = False
    s.draft_logits = DraftLogitsCache(
        num_slots, N_SPEC, vocab, dtype=torch.bfloat16, fill=SENTINEL, device=device
    )
    temps = torch.tensor([0.0, 0.6, 1.0, 1.5], device=device)
    s.temperature = temps[torch.arange(num_slots, device=device) % 4]
    s.seeds = torch.randint(0, 2**62, (num_slots,), generator=g, device=device)
    s._step_cols = torch.arange(N_SPEC, dtype=torch.int32, device=device)
    num_sample = num_reqs * N_SPEC
    s.sample_indices = torch.arange(num_sample, dtype=torch.int64, device=device)
    slots = torch.randperm(num_slots, generator=g, device=device)[:num_reqs]
    if num_reqs > 1:
        slots[-1] = -1
    s.sample_idx_mapping = slots.repeat_interleave(N_SPEC)
    ctx = torch.randint(0, 1 << 17, (num_reqs,), generator=g, device=device)
    steps = torch.arange(N_SPEC, device=device)
    s.sample_pos = (ctx[:, None] + 1 + steps[None, :]).flatten()
    s.input_buffers = SimpleNamespace(
        input_ids=torch.randint(
            0, vocab, (num_sample,), generator=g, device=device, dtype=torch.int32
        )
    )
    s._anchor_idx = torch.arange(num_reqs, dtype=torch.int64, device=device) * N_SPEC
    s.draft_tokens = torch.zeros(num_reqs, N_SPEC, dtype=torch.int64, device=device)
    return s


class _FakeOp(ModuleType):
    """Stands in for `aiter.ops.flydsl.markov_chain_sample`: declines what the
    real helper declines and records the warm-ups and launches."""

    def __init__(self):
        super().__init__(OP_MODULE)
        self.warmups: list[torch.device] = []
        self.launches: list[tuple] = []

    def markov_chain_sample_unsupported_reason(
        self,
        markov_w1,
        markov_w2,
        draft_logits_cache,
        *,
        num_speculative_steps,
        use_fp64_gumbel,
        draft_vocab_size,
        has_confidence_head,
        has_draft_watermarker,
        has_acceptance_estimator,
    ):
        if num_speculative_steps != N_SPEC or draft_vocab_size != markov_w2.shape[0]:
            return "shape"
        if use_fp64_gumbel or has_confidence_head:
            return "feature"
        if has_draft_watermarker or has_acceptance_estimator:
            return "feature"
        return None

    def warmup_markov_chain_sample(self, device, idx_dtype, ids_dtype):
        self.warmups.append(device)

    def flydsl_markov_chain_sample(self, *args):
        self.launches.append(args)


def _set_watermarker(s):
    s.draft_watermarker = object()


def _set_estimator(s):
    s.acceptance_estimator = object()


def _set_sharded(s):
    s.draft_logits.row_mapping = torch.zeros(1, dtype=torch.int32)


def _set_reduced_vocab(s):
    s._d2t_scatter_index = torch.arange(s.vocab_size // 2)


def _set_soft_cap(s):
    s.model.logits_processor.soft_cap = 30.0


def _set_fp32_head(s):
    s.model.logits_processor.head_dtype = torch.float32


@pytest.mark.parametrize(
    "change",
    [
        lambda s: setattr(s, "draft_logits", None),
        _set_sharded,
        lambda s: setattr(s, "_draft_topk", 64),
        lambda s: setattr(s, "sample_from_anchor", False),
        lambda s: setattr(s, "use_confidence_head", True),
        _set_watermarker,
        _set_estimator,
        lambda s: setattr(s, "use_fp64_gumbel", True),
        lambda s: setattr(s, "num_speculative_steps", 3),
        _set_reduced_vocab,
        _set_soft_cap,
        _set_fp32_head,
    ],
    ids=[
        "greedy",
        "sharded-cache",
        "top-k",
        "fill-in",
        "confidence-head",
        "watermarker",
        "estimator",
        "fp64-gumbel",
        "3-steps",
        "reduced-vocab",
        "soft-cap",
        "fp32-head",
    ],
)
def test_aiter_markov_declines_speculator(monkeypatch, change):
    """A speculator feature the kernel does not compute keeps vLLM's loop and
    compiles nothing; the unchanged speculator takes the kernel."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.deepseek_v41.amd.ops import aiter_markov

    op = _FakeOp()
    monkeypatch.setattr(rocm_aiter_ops, "flydsl_markov_sampler", lambda: op)
    vocab = 1024
    s = _speculator(_model(vocab, "cpu", seed=0), 2, vocab, "cpu", seed=0)
    assert aiter_markov.resolve(s, s.model) is op
    assert op.warmups == [s.device]

    change(s)
    assert aiter_markov.resolve(s, s.model) is None
    assert op.warmups == [s.device]


def test_aiter_markov_per_call_fallback(monkeypatch):
    """Per call, only bf16 base logits with unit column stride and the anchor
    indices take the kernel, and it gets the speculator's own buffers. The
    first draft step decides; a CUDA graph capture before it keeps vLLM's loop
    and decides nothing."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.deepseek_v41.amd.ops import fused_markov

    op = _FakeOp()
    monkeypatch.setattr(rocm_aiter_ops, "flydsl_markov_sampler", lambda: op)
    vocab, n = 1024, 3
    model = _model(vocab, "cpu", seed=1)
    s = _speculator(model, n, vocab, "cpu", seed=1)
    base = torch.zeros(n, N_SPEC, vocab, dtype=torch.bfloat16)
    idx_map = s.sample_idx_mapping.view(n, N_SPEC)
    pos = s.sample_pos.view(n, N_SPEC)
    prev = s.input_buffers.input_ids
    anchor = s._anchor_idx

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert model._aiter_markov_sampler(s) is None
    assert not model._aiter_markov_resolved and not op.warmups
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    # The AITER op takes the chain, so the fused per-step launches stand aside.
    monkeypatch.setattr(
        fused_markov, "sample_sequential", lambda *a: pytest.fail("fused_markov ran")
    )
    assert model.sample_sequential(s, n, torch.zeros(n * N_SPEC, 8)) is False
    assert op.warmups == [s.device]

    for args, kwargs in (
        ((base.float(), prev, idx_map, pos, s.draft_tokens), {"prev_indices": anchor}),
        ((base, prev[anchor], idx_map, pos, s.draft_tokens), {}),
        (
            (base[..., ::2], prev, idx_map, pos, s.draft_tokens),
            {"prev_indices": anchor},
        ),
    ):
        assert model.sample_markov_chain(s, *args, **kwargs) is None
    assert not op.launches

    embeds = model.sample_markov_chain(
        s, base, prev, idx_map, pos, s.draft_tokens, prev_indices=anchor
    )
    assert embeds is not None and embeds.shape == (n, N_SPEC, 0)
    (launch,) = op.launches
    head = model.model.markov_head
    expected = (
        base.view(n * N_SPEC, vocab),
        head.markov_w1.weight,
        head.markov_w2.weight,
        prev,
        anchor,
        idx_map,
        pos,
        s.temperature,
        s.seeds,
        s.draft_logits.values,
        s.draft_tokens,
    )
    assert len(launch) == len(expected)
    for got, want in zip(launch, expected):
        assert got.shape == want.shape and got.data_ptr() == want.data_ptr()


def test_aiter_markov_falls_back_without_op(markov_flag, monkeypatch):
    """With the flag on and an AITER that lacks the op (every AITER release
    today), the draft step runs vLLM's loop and gives its tokens."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    vocab, n = 1024, 4
    hidden = torch.zeros(n * N_SPEC, 8, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(2)
    base = torch.randn(n * N_SPEC, vocab, generator=g, device="cuda") * 3
    base = base.to(torch.bfloat16)
    tokens = []
    for on in (False, True):
        markov_flag(on)
        monkeypatch.setitem(sys.modules, OP_MODULE, None)
        rocm_aiter_ops._flydsl_markov_sampler_module.cache_clear()
        try:
            model = _model(vocab, "cuda", 2, base)
            s = _speculator(model, n, vocab, "cuda", seed=2)
            DSparkSpeculator._sample_sequential(s, n, hidden)
        finally:
            rocm_aiter_ops._flydsl_markov_sampler_module.cache_clear()
        assert s.model._aiter_markov_resolved and s.model._aiter_markov is None
        tokens.append(s.draft_tokens.clone())
    torch.testing.assert_close(tokens[1], tokens[0], rtol=0, atol=0)


@pytest.mark.parametrize("num_reqs", [1, 5, 37, 200])
def test_aiter_markov_matches_vllm_loop(markov_flag, num_reqs):
    """The draft step through the AITER op against vLLM's Markov loop.

    The op sums the Markov bias in another order than vLLM's GEMV, so a cached
    logit can differ in its last bf16 bit. So each step is checked against a
    reference built from the op's own previous tokens (at most 1 in 1000
    cached logits of a request and step may differ), and each draft token must
    be exactly the one vLLM's sampler draws from the logits the op cached.
    Against the loop run end to end, a draft token may differ only from a step
    on whose cached logits the two paths differ. 200 requests take two
    launches.
    """
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    markov_flag(True)
    if rocm_aiter_ops.flydsl_markov_sampler() is None:
        pytest.skip(f"needs AITER's {OP_MODULE} on gfx950")

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(num_reqs)
    base = (torch.randn(num_reqs * N_SPEC, VOCAB, generator=g, device=dev) * 3).to(
        torch.bfloat16
    )
    hidden = torch.zeros(num_reqs * N_SPEC, 8, device=dev)
    runs = []
    for on in (True, False):
        markov_flag(on)
        s = _speculator(_model(VOCAB, dev, 3, base), num_reqs, VOCAB, dev, 3)
        DSparkSpeculator._sample_sequential(s, num_reqs, hidden)
        torch.accelerator.synchronize()
        assert (s.model._aiter_markov is not None) == on
        runs.append((s, s.draft_tokens.clone(), s.draft_logits.values.clone()))
    (s, tok, cache), (_, loop_tok, loop_cache) = runs

    idx_map = s.sample_idx_mapping.view(num_reqs, N_SPEC)
    pos = s.sample_pos.view(num_reqs, N_SPEC)
    active = idx_map[:, 0] >= 0
    slots = idx_map[active, 0]
    unused = torch.ones(cache.shape[0], dtype=torch.bool, device=dev)
    unused[slots] = False
    assert bool((cache[unused] == SENTINEL).all()), "the op wrote an unused slot"

    # Teacher forcing: the reference of step k uses the op's token of step
    # k - 1 (the anchor token at step 0).
    prev = s.input_buffers.input_ids[s._anchor_idx].long()
    base3 = base.view(num_reqs, N_SPEC, VOCAB)
    step_diff = torch.zeros(num_reqs, N_SPEC, dtype=torch.int64, device=dev)
    for k in range(N_SPEC):
        bias = s.model.markov_bias(s.model.markov_embed(prev))
        ref = (base3[:, k] + bias)[active]
        got = cache[slots, k]
        step_diff[active, k] = (ref.view(torch.int16) != got.view(torch.int16)).sum(-1)
        prev = tok[:, k]
    assert int(step_diff.max()) <= VOCAB // 1000, (
        f"cached logits that differ, by request and draft step: {step_diff.tolist()}"
    )

    for k in range(N_SPEC):
        drawn = DSparkSpeculator._sample_logits(
            s, cache[slots, k].clone(), idx_map[active, k], pos[active, k], k
        )
        assert torch.equal(drawn, tok[active, k]), f"draft step {k}"

    loop_diff = cache[slots].view(torch.int16) != loop_cache[slots].view(torch.int16)
    loop_diff = loop_diff.sum(-1)
    first = (tok[active] != loop_tok[active]).int().argmax(dim=1)
    differs = (tok[active] != loop_tok[active]).any(dim=1)
    rows = torch.arange(int(active.sum()), device=dev)
    unexplained = differs & (loop_diff[rows, first] == 0)
    assert not bool(unexplained.any()), (
        f"draft tokens differ from vLLM's loop at requests "
        f"{unexplained.nonzero().flatten().tolist()} on a step whose cached "
        "logits are equal"
    )
