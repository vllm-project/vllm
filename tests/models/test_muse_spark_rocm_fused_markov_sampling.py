# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's fused ROCm DSpark Markov sampling against the stock loop.

The stock loop (`DSparkSpeculator._sample_sequential` without a
`sample_sequential` hook) and the fused launches run on the same random Markov
head, base logits and sampling state. They share the bias GEMM, so the drafted
tokens must be equal and the cached draft logits within one ulp of the cache
dtype (the fp32 tanh is not torch's instruction sequence). Greedy drafts (no
cache) must draw the loop's tokens.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

VOCAB, RANK, N_SPEC, MAX_REQS = 4099, 16, 5, 48
# The released draft's output transform: metap "sp" at width 8192 / 256, cap 20.
MULT, SOFT_CAP = (8192 / 256) ** -0.5, 20.0


class _MarkovHead(nn.Module):
    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        quant_method: QuantizeMethodBase | None = None,
        padding_size: int = 1,
    ) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(VOCAB, RANK, _weight=w1)
        # The draft's vocabulary head, laid out by its quantization method.
        # Quantized GEMMs need a multiple of 16 rows. The padding sets the
        # row stride of the bias.
        with set_default_torch_dtype(w2.dtype):
            w2_head = ParallelLMHead(
                VOCAB,
                RANK,
                padding_size=padding_size if quant_method is None else 16,
                disable_tp=True,
                quant_method=quant_method,
            )
        padded = w2.new_zeros(w2_head.num_embeddings_per_partition, RANK)
        padded[:VOCAB] = w2
        w2_head.weight = nn.Parameter(padded, requires_grad=False)
        w2_head.quant_method.process_weights_after_loading(w2_head)
        self.markov_w2 = w2_head


class _DraftModel(nn.Module):
    """The parts of the Muse DSpark draft that Markov sampling uses: the same
    head, bias and output transform, and fixed base logits."""

    # The draft model sets this after loading its d2t map.
    d2t_is_identity = True

    def __init__(self, w1, w2, base, quant_method=None, padding_size=1) -> None:
        super().__init__()
        self.output_multiplier = MULT
        self.output_soft_cap_temp = SOFT_CAP
        self.model = nn.Module()
        self.model.markov_head = _MarkovHead(w1, w2, quant_method, padding_size)
        self.base = base

    def compute_draft_logits(self, hidden):
        return self.base[: hidden.shape[0]]

    def markov_embed(self, ids):
        return self.model.markov_head.markov_w1(ids)

    def markov_bias(self, embed):
        w2 = self.model.markov_head.markov_w2
        return w2.quant_method.apply(w2, embed)[:, :VOCAB]

    def map_draft_to_target(self, ids):
        return ids

    def apply_output_logit_transform(self, logits):
        logits = logits.float() * self.output_multiplier
        return self.output_soft_cap_temp * torch.tanh(
            logits / self.output_soft_cap_temp
        )


def _speculator(
    device: torch.device,
    temp: float,
    seed: int,
    cache_dtype: torch.dtype = torch.float32,
    greedy: bool = False,
    fp64: bool = False,
    quant_method: QuantizeMethodBase | None = None,
    padding_size: int = 1,
) -> DSparkSpeculator:
    bf16 = torch.bfloat16
    g = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.randn(VOCAB, RANK, generator=g, device=device).to(bf16)
    w2 = (torch.randn(VOCAB, RANK, generator=g, device=device) * 0.05).to(bf16)
    base = torch.randn(MAX_REQS * N_SPEC, VOCAB, generator=g, device=device) * 3
    s = DSparkSpeculator.__new__(DSparkSpeculator)
    s.model = _DraftModel(w1, w2, base.to(bf16), quant_method, padding_size)
    s.num_speculative_steps = N_SPEC
    s.max_num_reqs = MAX_REQS
    s._draft_topk = None
    s.use_confidence_head = False
    s.use_acceptance_estimator = False
    s.acceptance_estimator = None
    s.draft_watermarker = None
    s._fuse_markov_bias = False
    # As in the server (d2t = 0): the loop scatters the fp32 transformed
    # logits through an identity map into a buffer of the cache dtype.
    s._d2t_scatter_index = None if greedy else torch.arange(VOCAB, device=device)
    s._draft_scatter_buf = (
        None
        if greedy
        else torch.full((MAX_REQS, VOCAB), float("-inf"), dtype=cache_dtype).to(device)
    )
    s.use_fp64_gumbel = fp64
    s.draft_logits = (
        None
        if greedy
        else DraftLogitsCache(
            MAX_REQS, N_SPEC, VOCAB, dtype=cache_dtype, fill=0.0, device=device
        )
    )
    s.temperature = torch.full((MAX_REQS,), temp, dtype=torch.float32, device=device)
    s.seeds = torch.randint(0, 2**62, (MAX_REQS,), generator=g, device=device)
    s._step_cols = torch.arange(N_SPEC, dtype=torch.int32, device=device)
    s.sample_indices = torch.arange(MAX_REQS * N_SPEC, dtype=torch.int64, device=device)
    # Request slots in another order than the rows, and positions past a long
    # context, as in the server.
    slots = torch.randperm(MAX_REQS, generator=g, device=device).to(torch.int32)
    s.sample_idx_mapping = slots.repeat_interleave(N_SPEC)
    ctx = torch.randint(7, 131072, (MAX_REQS,), generator=g, device=device)
    steps = torch.arange(N_SPEC, device=device)
    s.sample_pos = (ctx[:, None] + 1 + steps[None, :]).flatten()
    s.input_buffers = type(
        "Buffers",
        (),
        {
            "input_ids": torch.randint(
                0, VOCAB, (MAX_REQS * N_SPEC,), generator=g, device=device
            )
        },
    )()
    s._anchor_idx = torch.arange(MAX_REQS, dtype=torch.int64, device=device) * N_SPEC
    s.draft_tokens = torch.zeros(MAX_REQS, N_SPEC, dtype=torch.int64, device=device)
    return s


def _fused(spec: DSparkSpeculator):
    from vllm.models.muse_spark.amd.ops.fused_markov_sampling import (
        MuseSparkFusedMarkov,
    )

    return MuseSparkFusedMarkov(spec.model)


@pytest.mark.parametrize(
    "gfx950, monokernel, batch_sharded, expected",
    [
        (True, False, False, True),
        (False, False, False, False),
        # --monokernel takes the fused Markov rounds of ops.fused_markov.
        (True, True, False, False),
        (True, False, True, False),
    ],
)
def test_fused_markov_gating(monkeypatch, gfx950, monokernel, batch_sharded, expected):
    """This module on gfx950, except with --monokernel (ops.fused_markov) or
    batch-sharded sampling (the speculator's sharded loop)."""
    import vllm.platforms.rocm as rocm_platform
    from vllm.models.muse_spark.amd.ops.fused_markov_sampling import (
        use_fused_markov_sampling,
    )

    monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: gfx950)
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=monokernel),
        parallel_config=SimpleNamespace(enable_batch_sharded_sampling=batch_sharded),
    )
    assert use_fused_markov_sampling(config) == expected


@pytest.mark.parametrize("d2t_offset, identity", [(0, True), (7, False)])
def test_loading_sets_the_draft_map_flag(monkeypatch, d2t_offset, identity):
    """The draft's process_weights_after_loading sets the flag the hook reads."""
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
    from vllm.models.muse_spark.amd.dspark import MuseSparkDSparkForCausalLM

    # Only the DFlash base step needs real attention layers.
    monkeypatch.setattr(
        DFlashQwen3ForCausalLM, "process_weights_after_loading", lambda self: None
    )
    spec = _speculator(torch.device("cpu"), 1.0, seed=0)
    draft = MuseSparkDSparkForCausalLM.__new__(MuseSparkDSparkForCausalLM)
    nn.Module.__init__(draft)
    draft.model = spec.model.model
    draft.draft_id_to_target_id = nn.Parameter(
        torch.full((VOCAB,), d2t_offset), requires_grad=False
    )
    fused = _fused(spec)
    fused.model = draft
    draft.sample_sequential = fused.sample
    draft.process_weights_after_loading()
    assert (fused.why_not(spec) is None) == identity


@pytest.mark.parametrize(
    "attr, value, reason",
    [
        ("draft_watermarker", object(), "watermarker"),
        ("use_confidence_head", True, "adaptive verification"),
        ("use_acceptance_estimator", True, "adaptive verification"),
        ("_draft_topk", 64, "top-k"),
    ],
)
def test_fused_markov_declines_unsupported_speculator_state(attr, value, reason):
    """Each unsupported speculator state sends the step to the stock loop."""
    from vllm.models.muse_spark.amd.ops.fused_markov_sampling import (
        MuseSparkFusedMarkov,
    )

    spec = _speculator(torch.device("cpu"), 1.0, seed=0)
    setattr(spec, attr, value)
    got = MuseSparkFusedMarkov(spec.model).why_not(spec)
    assert got is not None and reason in got
    assert (
        MuseSparkFusedMarkov(spec.model).why_not(
            _speculator(torch.device("cpu"), 1.0, seed=0)
        )
        is None
    )


def test_fused_markov_declines_non_identity_draft_map():
    spec = _speculator(torch.device("cpu"), 1.0, seed=0)
    spec.model.d2t_is_identity = False
    assert "draft->target" in _fused(spec).why_not(spec)


def test_fused_markov_declines_a_narrower_draft_vocab():
    """A cache wider than the draft vocabulary keeps columns the kernel would
    not write."""
    spec = _speculator(torch.device("cpu"), 1.0, seed=0)
    spec.draft_logits = DraftLogitsCache(
        MAX_REQS, N_SPEC, VOCAB + 1, dtype=torch.float32, fill=0.0, device="cpu"
    )
    assert "cache width" in _fused(spec).why_not(spec)


def test_fused_markov_takes_greedy_and_fp64_drafts():
    """Greedy drafts and fp64 Gumbel noise run the fused launches."""
    cpu = torch.device("cpu")
    for kwargs in ({"greedy": True}, {"fp64": True}, {"greedy": True, "fp64": True}):
        spec = _speculator(cpu, 1.0, seed=0, **kwargs)
        assert _fused(spec).why_not(spec) is None


def _hidden(spec: DSparkSpeculator) -> torch.Tensor:
    return torch.zeros(MAX_REQS * N_SPEC, 8, device=spec.draft_tokens.device)


def _run_both(spec: DSparkSpeculator, num_reqs: int, fused_run=None):
    """(loop tokens, loop cache, fused tokens, fused cache) on one state."""
    hidden = _hidden(spec)
    DSparkSpeculator._sample_sequential(spec, num_reqs, hidden)
    torch.accelerator.synchronize()
    ref_tokens = spec.draft_tokens[:num_reqs].clone()
    cache = spec.draft_logits
    ref_cache = cache.values.clone() if cache is not None else None

    spec.draft_tokens.fill_(-1)
    if cache is not None:
        cache.values.zero_()
    if fused_run is None:
        assert _fused(spec).sample(spec, num_reqs, hidden)
    else:
        fused_run()
    torch.accelerator.synchronize()
    tokens = spec.draft_tokens[:num_reqs]
    if cache is None or ref_cache is None:
        return ref_tokens, None, tokens, None
    slots = spec.sample_idx_mapping[: num_reqs * N_SPEC : N_SPEC].long()
    return ref_tokens, ref_cache[slots], tokens, cache.values[slots]


_rocm_only = pytest.mark.skipif(
    not (current_platform.is_rocm()), reason="the fused launches are for ROCm"
)


def _assert_within_one_ulp(cache: torch.Tensor, ref: torch.Tensor) -> None:
    """bf16 caches: same NaN positions, elsewhere at most one ulp apart."""
    assert cache.dtype == ref.dtype == torch.bfloat16
    nan = ref.isnan()
    assert torch.equal(cache.isnan(), nan)

    def ordered(x: torch.Tensor) -> torch.Tensor:
        bits = x.view(torch.int16).long()
        return torch.where(bits < 0, -32768 - bits, bits)

    assert (ordered(cache) - ordered(ref))[~nan].abs().max() <= 1


@_rocm_only
@pytest.mark.parametrize("fp64", [False, True])
@pytest.mark.parametrize("temp", [0.0, 0.6, 1.0])
@pytest.mark.parametrize("num_reqs", [1, 3, MAX_REQS])
def test_fused_markov_matches_the_stock_loop(num_reqs, temp, fp64):
    """bf16 cache (the server's): equal tokens, cached logits within an ulp."""
    spec = _speculator(
        torch.device("cuda"),
        temp,
        seed=int(temp * 10) + num_reqs,
        cache_dtype=torch.bfloat16,
        fp64=fp64,
    )
    ref_tokens, ref_cache, tokens, cache = _run_both(spec, num_reqs)
    assert torch.equal(tokens, ref_tokens)
    _assert_within_one_ulp(cache, ref_cache)


@_rocm_only
@pytest.mark.parametrize("temp", [0.0, 1.0])
@pytest.mark.parametrize("num_reqs", [1, MAX_REQS])
def test_fused_markov_fp32_cache(num_reqs, temp):
    """fp32 cache: same tokens; cached logits within a few fp32 ulps."""
    spec = _speculator(torch.device("cuda"), temp, seed=7 + num_reqs)
    ref_tokens, ref_cache, tokens, cache = _run_both(spec, num_reqs)
    assert torch.equal(tokens, ref_tokens)
    torch.testing.assert_close(cache, ref_cache, rtol=0, atol=1e-5)


@_rocm_only
@pytest.mark.parametrize("fp64", [False, True])
@pytest.mark.parametrize("num_reqs", [1, 3, MAX_REQS])
def test_fused_markov_greedy_drafts_match_the_stock_loop(num_reqs, fp64):
    """Greedy drafts (no draft logits cache): the loop's greedy_argmax tokens."""
    spec = _speculator(
        torch.device("cuda"), 1.0, seed=20 + num_reqs, greedy=True, fp64=fp64
    )
    ref_tokens, _, tokens, _ = _run_both(spec, num_reqs)
    assert torch.equal(tokens, ref_tokens)


@_rocm_only
@pytest.mark.parametrize("fp64", [False, True])
def test_fused_markov_ties_take_the_lowest_token(fp64):
    """The reduction across blocks keeps the first of equal maxima, as the
    loop's argmax over block maxima: a temperature-0 row whose logits are all
    equal must draw token 0 at every step."""
    spec = _speculator(
        torch.device("cuda"), 0.0, seed=3, cache_dtype=torch.bfloat16, fp64=fp64
    )
    spec.model.base.zero_()
    spec.model.model.markov_head.markov_w2.weight.zero_()
    ref_tokens, ref_cache, tokens, cache = _run_both(spec, MAX_REQS)
    assert torch.equal(tokens, ref_tokens)
    _assert_within_one_ulp(cache, ref_cache)
    assert not tokens.any()


@_rocm_only
@pytest.mark.parametrize("fp64", [False, True])
@pytest.mark.parametrize(
    "vocab, padding_size",
    [
        (VOCAB, 1),
        (VOCAB, DEFAULT_VOCAB_PADDING_SIZE),
        (202048, DEFAULT_VOCAB_PADDING_SIZE),
    ],
)
def test_fused_markov_nan_rows(monkeypatch, vocab, padding_size, fp64):
    """NaN logits are cached as is and sample as -inf, whatever the row
    strides: a row of negative NaNs draws token 0; a row with NaNs of both
    signs, one with a NaN in the last, partly masked block and one with NaNs
    next to a clear winner draw the tokens of the same rows with -inf instead
    of NaN. (The loop's tl.max over NaN depends on its reduction order, so
    its tokens are only checked to be in the vocabulary.)"""
    monkeypatch.setitem(globals(), "VOCAB", vocab)
    num_reqs = 4

    def speculator(neg: float, pos: float) -> DSparkSpeculator:
        spec = _speculator(
            torch.device("cuda"),
            1.0,
            seed=11,
            cache_dtype=torch.bfloat16,
            fp64=fp64,
            padding_size=padding_size,
        )
        base = spec.model.base.view(MAX_REQS, N_SPEC, vocab)
        base[0] = neg
        base[1, :, 2000] = neg
        base[1, :, 3000] = pos
        base[2, :, vocab - 2] = neg
        base[3, :, 0] = 200.0
        base[3, :, [1, 256]] = pos
        return spec

    nan = torch.tensor(float("nan"), dtype=torch.bfloat16)
    spec = speculator(-nan, nan)
    ref_tokens, ref_cache, tokens, cache = _run_both(spec, num_reqs)
    assert ((ref_tokens >= 0) & (ref_tokens < vocab)).all()
    # The cached logits match at each step that follows the same tokens.
    same = (tokens == ref_tokens).int().cumprod(dim=1)
    same = torch.cat([torch.ones_like(same[:, :1]), same[:, :-1]], dim=1).bool()
    _assert_within_one_ulp(cache[same], ref_cache[same])
    assert not tokens[0].any()
    assert not tokens[3].any()
    # The soft cap maps -inf to -cap, which none of these rows draws.
    finite = speculator(float("-inf"), float("-inf"))
    DSparkSpeculator._sample_sequential(finite, num_reqs, _hidden(finite))
    assert torch.equal(tokens[1:], finite.draft_tokens[1:num_reqs])


@_rocm_only
@pytest.mark.parametrize("fp64", [False, True])
def test_fused_markov_graph_replays(fp64):
    """One captured graph replayed with new seeds draws the loop's tokens each
    time: the per-row counters and slots reset themselves."""
    num_reqs = 5
    spec = _speculator(
        torch.device("cuda"), 1.0, seed=5, cache_dtype=torch.bfloat16, fp64=fp64
    )
    fused = _fused(spec)
    hidden = _hidden(spec)
    # The first call checks the state and allocates the scratch, outside capture.
    assert fused.sample(spec, num_reqs, hidden)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused.sample(spec, num_reqs, hidden)
    g = torch.Generator(device="cuda").manual_seed(0)
    for _ in range(3):
        spec.seeds.copy_(
            torch.randint(0, 2**62, (MAX_REQS,), generator=g, device="cuda")
        )
        ref_tokens, ref_cache, tokens, cache = _run_both(spec, num_reqs, graph.replay)
        assert torch.equal(tokens, ref_tokens)
        _assert_within_one_ulp(cache, ref_cache)


@_rocm_only
@pytest.mark.parametrize("fp64", [False, True])
def test_fused_markov_is_deterministic(fp64):
    """The same inputs draw the same tokens and cached logits, whatever order
    the programs of a row finish in."""
    spec = _speculator(
        torch.device("cuda"), 1.0, seed=9, cache_dtype=torch.bfloat16, fp64=fp64
    )
    fused = _fused(spec)
    hidden = _hidden(spec)
    runs = []
    for _ in range(2):
        spec.draft_tokens.fill_(-1)
        spec.draft_logits.values.zero_()
        assert fused.sample(spec, MAX_REQS, hidden)
        runs.append((spec.draft_tokens.clone(), spec.draft_logits.values.clone()))
    assert torch.equal(runs[0][0], runs[1][0])
    assert torch.equal(runs[0][1], runs[1][1])


@_rocm_only
@pytest.mark.parametrize("num_reqs", [1, MAX_REQS])
def test_fused_markov_takes_an_fp8_markov_head(dist_init, monkeypatch, num_reqs):
    """A Markov head on the online FP8 lm_head method, whose kernel may keep
    the weight transposed, takes the fused launches: equal tokens, cached
    logits within an ulp."""
    # The online method reads the model dtype from the current config.
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    with set_default_torch_dtype(torch.bfloat16):
        quant_method = Fp8PtpcOnlineLinearMethod()
    spec = _speculator(
        torch.device("cuda"),
        1.0,
        seed=30 + num_reqs,
        cache_dtype=torch.bfloat16,
        quant_method=quant_method,
    )
    assert spec.model.model.markov_head.markov_w2.weight.dtype != torch.bfloat16
    assert _fused(spec).why_not(spec) is None
    ref_tokens, ref_cache, tokens, cache = _run_both(spec, num_reqs)
    assert torch.equal(tokens, ref_tokens)
    _assert_within_one_ulp(cache, ref_cache)
