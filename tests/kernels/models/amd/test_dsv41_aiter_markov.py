# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AITER FlyDSL DSpark Markov sampler call site of the DeepSeek-V4.1 draft
model (ops/aiter_markov.py), driven through DSparkSpeculator._sample_sequential.

The speculator is stubbed as in test_dsv41_gfx942_fused_markov.py. The gating
tests replace the AITER op module with a stand-in that records what it is
asked; the GPU test needs AITER's op on gfx950 and compares it with vLLM's
Markov loop.
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
    rocm_aiter_ops._flydsl_markov_sampler_module.cache_clear()


class _StandIn(ModuleType):
    """Stands in for the AITER op module and records what it is asked."""

    def __init__(self):
        super().__init__(OP_MODULE)
        self.reason: str | None = None
        self.asked: list[tuple] = []
        self.warmups: list[torch.device] = []
        self.launches: list[tuple] = []

    def markov_chain_sample_unsupported_reason(self, *args, **kwargs):
        self.asked.append((args, kwargs))
        return self.reason

    def warmup_markov_chain_sample(self, device, idx_dtype, ids_dtype):
        self.warmups.append(device)

    def flydsl_markov_chain_sample(self, *args):
        self.launches.append(args)


@pytest.fixture
def stand_in(markov_flag, monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the AITER Markov sampler is gfx950 only")
    op = _StandIn()
    monkeypatch.setitem(sys.modules, OP_MODULE, op)
    markov_flag(True)
    rocm_aiter_ops._flydsl_markov_sampler_module.cache_clear()
    return op


def _model(vocab: int, seed: int, base: torch.Tensor | None = None):
    """The DeepSeek-V4.1 draft model with a random Markov head, whose base
    logits are `base`, so that every path samples from the same ones."""
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
    from vllm.models.deepseek_v41.amd.dspark import DSparkDeepseekV4ForCausalLM

    g = torch.Generator(device="cuda").manual_seed(seed)
    head = DSparkMarkovHead(vocab, vocab, RANK, prefix="markov_head")
    head = head.to(device="cuda", dtype=torch.bfloat16)
    for w, scale in ((head.markov_w1.weight, 1.0), (head.markov_w2.weight, 0.05)):
        w.data.copy_(torch.randn(vocab, RANK, generator=g, device="cuda") * scale)
    model = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace(markov_head=head)
    model.logits_processor = LogitsProcessor(vocab)
    if base is None:
        base = torch.zeros(0, vocab, dtype=torch.bfloat16, device="cuda")
    model.compute_draft_logits = lambda hidden: base[: hidden.shape[0]]  # type: ignore[method-assign]
    return model


def _speculator(model, num_reqs: int, vocab: int, seed: int):
    """A DSparkSpeculator of DeepSeek-V4.1's probabilistic drafting, as
    `load_model` leaves it. Requests take shuffled slots of a larger cache,
    mixed temperatures and positions past a long context. With more than one
    request, the last one is an inert padding row (slot -1)."""
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    num_slots, num_sample = num_reqs + 3, num_reqs * N_SPEC
    s = DSparkSpeculator.__new__(DSparkSpeculator)
    s.model, s.device, s.vocab_size = model, dev, vocab
    s.num_speculative_steps = N_SPEC
    s.sample_from_anchor = True
    s._draft_topk = s.acceptance_estimator = s.draft_watermarker = None
    s._d2t_scatter_index = s._draft_scatter_buf = None
    s.use_confidence_head = s.use_fp64_gumbel = False
    s._fuse_markov_bias = True
    s.draft_logits = DraftLogitsCache(
        num_slots, N_SPEC, vocab, dtype=torch.bfloat16, fill=SENTINEL, device=dev
    )
    temps = torch.tensor([0.0, 0.6, 1.0, 1.5], device=dev)
    s.temperature = temps[torch.arange(num_slots, device=dev) % 4]
    s.seeds = torch.randint(0, 2**62, (num_slots,), generator=g, device=dev)
    s._step_cols = torch.arange(N_SPEC, dtype=torch.int32, device=dev)
    s.sample_indices = torch.arange(num_sample, dtype=torch.int64, device=dev)
    slots = torch.randperm(num_slots, generator=g, device=dev)[:num_reqs]
    if num_reqs > 1:
        slots[-1] = -1
    s.sample_idx_mapping = slots.repeat_interleave(N_SPEC)
    ctx = torch.randint(0, 1 << 17, (num_reqs, 1), generator=g, device=dev)
    s.sample_pos = (ctx + 1 + torch.arange(N_SPEC, device=dev)).flatten()
    ids = torch.randint(0, vocab, (num_sample,), generator=g, device=dev)
    s.input_buffers = SimpleNamespace(input_ids=ids.int())
    s._anchor_idx = torch.arange(num_reqs, dtype=torch.int64, device=dev) * N_SPEC
    s.draft_tokens = torch.zeros(num_reqs, N_SPEC, dtype=torch.int64, device=dev)
    return s


def test_aiter_markov_forwards_speculator_features(stand_in: _StandIn):
    """The features only the op can judge reach its helper as stated; the op's
    reason keeps vLLM's loop, compiling nothing."""
    from vllm.models.deepseek_v41.amd.ops import aiter_markov

    vocab = 1024
    s = _speculator(_model(vocab, seed=0), 2, vocab, seed=0)
    s.num_speculative_steps, s.use_fp64_gumbel, s.use_confidence_head = 3, True, True
    s.draft_watermarker = s.acceptance_estimator = object()
    s._d2t_scatter_index = torch.arange(vocab // 2)
    stand_in.reason = "unsupported"
    assert aiter_markov.resolve(s, s.model) is None and not stand_in.warmups

    ((args, kwargs),) = stand_in.asked
    head = s.model.model.markov_head
    want = (head.markov_w1.weight, head.markov_w2.weight, s.draft_logits.values)
    assert [a.data_ptr() for a in args] == [w.data_ptr() for w in want]
    assert kwargs == dict(
        num_speculative_steps=3,
        use_fp64_gumbel=True,
        draft_vocab_size=vocab // 2,
        has_confidence_head=True,
        has_draft_watermarker=True,
        has_acceptance_estimator=True,
    )
    stand_in.reason = None
    assert aiter_markov.resolve(s, s.model) is stand_in
    assert stand_in.warmups == [s.device]


def _set_sharded(s):
    s.draft_logits.row_mapping = torch.zeros(1, dtype=torch.int32)


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
        _set_soft_cap,
        _set_fp32_head,
    ],
    ids=["greedy", "sharded-cache", "top-k", "fill-in", "soft-cap", "fp32-head"],
)
def test_aiter_markov_declines_speculator(stand_in: _StandIn, change):
    """What vLLM decides itself keeps its loop without asking the op."""
    from vllm.models.deepseek_v41.amd.ops import aiter_markov

    vocab = 1024
    s = _speculator(_model(vocab, seed=0), 2, vocab, seed=0)
    change(s)
    assert aiter_markov.resolve(s, s.model) is None
    assert not stand_in.asked and not stand_in.warmups


def test_aiter_markov_per_call_fallback(stand_in: _StandIn, monkeypatch):
    """A capture before the first draft step decides nothing; the op takes
    precedence over the --monokernel fused launches; per call, only aligned
    bf16 base logits with unit column stride and anchor indices take the op,
    which gets the speculator's own buffers."""
    from vllm.models.deepseek_v41.amd import ops
    from vllm.models.deepseek_v41.amd.ops import fused_markov

    vocab, n = 1024, 3
    base = torch.randn(n * N_SPEC, vocab, device="cuda").to(torch.bfloat16)
    model = _model(vocab, seed=1, base=base)
    s = _speculator(model, n, vocab, seed=1)
    hidden = torch.zeros(n * N_SPEC, 8, device="cuda")

    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        assert model._aiter_markov_sampler(s) is None
    assert not stand_in.asked and not stand_in.warmups

    monkeypatch.setattr(ops, "_monokernel", True)
    ops.enabled.cache_clear()
    ops.monokernel_cdna3_or_4.cache_clear()
    assert fused_markov.sample_sequential(s, n, hidden)
    assert model.sample_sequential(s, n, hidden) is False
    assert stand_in.warmups == [s.device]

    base3 = base.view(n, N_SPEC, vocab)
    flat = torch.zeros(base.numel() + 4, dtype=torch.bfloat16, device="cuda")
    unaligned = flat[1 : 1 + base.numel()].view(n, N_SPEC, vocab)
    idx_map = s.sample_idx_mapping.view(n, N_SPEC)
    pos = s.sample_pos.view(n, N_SPEC)
    prev, anchor = s.input_buffers.input_ids, s._anchor_idx
    for b, p, a in (
        (base3.float(), prev, anchor),
        (base3[..., ::2], prev, anchor),
        (unaligned, prev, anchor),
        (base3, prev[anchor], None),
        (base3, prev, anchor.int()),
    ):
        assert (
            model.sample_markov_chain(
                s, b, p, idx_map, pos, s.draft_tokens, prev_indices=a
            )
            is None
        )
    assert not stand_in.launches

    embeds = model.sample_markov_chain(
        s, base3, prev, idx_map, pos, s.draft_tokens, prev_indices=anchor
    )
    assert embeds is not None and embeds.shape == (n, N_SPEC, 0)
    (launch,) = stand_in.launches
    head = model.model.markov_head
    want = (base, head.markov_w1.weight, head.markov_w2.weight, prev, anchor,
            idx_map, pos, s.temperature, s.seeds, s.draft_logits.values,
            s.draft_tokens)  # fmt: skip
    assert [(t.shape, t.data_ptr()) for t in launch] == [
        (t.shape, t.data_ptr()) for t in want
    ]


@pytest.mark.parametrize(
    "module", [None, ModuleType(OP_MODULE)], ids=["no-op", "old-api"]
)
def test_aiter_markov_falls_back_without_op(markov_flag, monkeypatch, module):
    """With the flag on and an AITER without the op or with another API, the
    speculator keeps vLLM's loop instead of failing in the draft step."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.deepseek_v41.amd.ops import aiter_markov

    monkeypatch.setitem(sys.modules, OP_MODULE, module)
    markov_flag(True)
    rocm_aiter_ops._flydsl_markov_sampler_module.cache_clear()
    s = _speculator(_model(1024, seed=2), 2, 1024, seed=2)
    assert aiter_markov.resolve(s, s.model) is None


@pytest.mark.parametrize("num_reqs", [1, 5, 37, 200])
def test_aiter_markov_matches_vllm_loop(markov_flag, num_reqs):
    """The draft step through the AITER op against vLLM's Markov loop.

    The op sums the Markov bias in another order than vLLM's GEMV, so a cached
    logit can differ in its last bf16 bit. Each step is checked against a
    reference built from the op's own previous tokens (at most 1 in 1000
    cached logits of a request and step may differ), and each draft token must
    be exactly the one vLLM's sampler draws from the logits the op cached.
    Against the loop run end to end, a draft token may differ only from a step
    whose cached logits differ. 200 requests take two launches.
    """
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    markov_flag(True)
    if rocm_aiter_ops.flydsl_markov_sampler() is None:
        pytest.skip(f"needs AITER's {OP_MODULE} on gfx950")

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(num_reqs)
    base = torch.randn(num_reqs * N_SPEC, VOCAB, generator=g, device=dev) * 3
    base = base.bfloat16()
    hidden = torch.zeros(num_reqs * N_SPEC, 8, device=dev)
    runs = []
    for on in (True, False):
        markov_flag(on)
        s = _speculator(_model(VOCAB, 3, base), num_reqs, VOCAB, 3)
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

    # Teacher forcing: step k's reference uses the op's token of step k - 1.
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
    differs = tok[active] != loop_tok[active]
    first = differs.int().argmax(dim=1)
    rows = torch.arange(int(active.sum()), device=dev)
    unexplained = differs.any(dim=1) & (loop_diff[rows, first] == 0)
    assert not bool(unexplained.any()), (
        f"draft tokens differ from vLLM's loop at requests "
        f"{unexplained.nonzero().flatten().tolist()} on a step whose cached "
        "logits are equal"
    )
