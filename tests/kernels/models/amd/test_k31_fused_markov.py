# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_K31_BEAT_MARKOV's fused Markov launches (kimi_k3/amd/ops/markov_sample,
taken through specialized_decode/levers/dspark under latency_specialized)
against vLLM's DSparkSpeculator._sample_sequential loop, at Kimi-K3.1 DSpark's
shapes (vocabulary 163840, Markov rank 512, 7 draft steps).

Both paths run on the same random Markov head, base logits and sampling state,
for 1, 2 and 5 requests at temperatures 0, 0.6 and 1.0, with a replicated and
a row-mapped (batch-sharded) draft logits cache. vLLM's bias is the server's
GEMV (rocm_unquantized_gemm). The fused launch adds the bias's 512 products in
another order, so a bias value can differ in its last bf16 bit and then a draft
token can differ. Against vLLM's loop the test allows at most 1 in 1000 cached
logits of each request and step to differ, and a draft token may differ only
when a cached logit of its own step, or an earlier token of its request,
differed.

The sampling itself is checked exactly: each fused draft token must be the token
that vLLM's sampler (DSparkSpeculator._sample_logits) draws from the logits the
fused launch cached for that request and step, and those cached logits must be
the base logits plus the reference GEMV's bias of the fused previous token
(within the same last-bit tolerance).

The other tests cover the speculator's fallback cases (why_not), the padded
sample rows of graph capture (no request, position 0) and 200 replays of a
captured draft step against eager calls.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the fused launch is for ROCm"
)

VOCAB, RANK, N_SPEC, MAX_REQS = 163840, 512, 7, 8
# The Kimi-K3.1 DSpark drafts the launches take: the preview's MLA draft and the
# release's GQA sliding-window draft (the same DSparkMarkovHead).
DRAFTS = ("preview", "release")


class _MarkovW2(torch.nn.Module):
    def __init__(self, w):
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()


def _draft_cls(draft: str):
    if draft == "release":
        from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

        return Qwen3DSparkForCausalLM
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM

    return K3DSparkForCausalLM


def _model(w1, w2, base, draft: str = "preview"):
    class _Model(_draft_cls(draft)):
        """The Kimi-K3.1 draft's Markov interface on a random head (the fused
        launches take Kimi-K3.1 DSpark drafts only)."""

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.draft_id_to_target_id = None
            head = torch.nn.Module()
            head.markov_w1 = torch.nn.Embedding(VOCAB, RANK, _weight=w1)
            head.markov_w2 = _MarkovW2(w2)
            self.model = SimpleNamespace(markov_head=head)
            self.logits_processor = SimpleNamespace(
                scale=1.0, soft_cap=None, head_dtype=torch.bfloat16
            )
            self.base = base

        def compute_draft_logits(self, hidden):
            return self.base[: hidden.shape[0]].clone()

        def markov_embed(self, ids):
            return self.model.markov_head.markov_w1(ids)

        def markov_bias(self, e):
            from vllm.model_executor.layers.utils import rocm_unquantized_gemm

            w2 = self.model.markov_head.markov_w2
            return rocm_unquantized_gemm(w2, e, w2.weight, None)

        def map_draft_to_target(self, ids):
            return ids

        def compute_confidence(self, hidden, embed):
            # Any function of both: the fused path must hand it the same rows.
            return torch.sigmoid(
                hidden.float().sum(-1) * 1e-2 + embed.float().sum(-1) * 1e-3
            )

    return _Model()


def _speculator(
    temp: float,
    seed: int,
    row_mapped: bool,
    draft: str = "preview",
    greedy: bool = False,
):
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    dev = torch.device("cuda")
    bf16 = torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(seed)
    w1 = (torch.randn(VOCAB, RANK, generator=g, device=dev) * 0.5).to(bf16)
    w2 = (torch.randn(VOCAB, RANK, generator=g, device=dev) * 0.05).to(bf16)
    base = torch.randn(MAX_REQS * N_SPEC, VOCAB, generator=g, device=dev) * 3
    s = DSparkSpeculator.__new__(DSparkSpeculator)
    s.model = _model(w1, w2, base.to(bf16), draft)
    s.num_speculative_steps = N_SPEC
    s.max_num_reqs = MAX_REQS
    s._draft_topk = None
    s.use_confidence_head = False
    s.acceptance_estimator = None
    s.draft_watermarker = None
    s._d2t_scatter_index = None
    s._draft_scatter_buf = None
    s.use_fp64_gumbel = False
    s.draft_logits = DraftLogitsCache(
        MAX_REQS, N_SPEC, VOCAB, dtype=bf16, fill=0.0, device=dev
    )
    s.draft_token_confidence_probs = torch.zeros(
        MAX_REQS, N_SPEC, dtype=torch.float32, device=dev
    )
    if row_mapped:
        # A batch-sharded cache: half the request slots cached at other rows,
        # the other half on another rank (-1).
        s.draft_logits.row_mapping = torch.tensor(
            [3, -1, 0, -1, 2, -1, 1, -1], dtype=torch.int32, device=dev
        )
    s.temperature = torch.full((MAX_REQS,), temp, dtype=torch.float32, device=dev)
    s.seeds = torch.randint(0, 2**62, (MAX_REQS,), generator=g, device=dev)
    s._step_cols = torch.arange(N_SPEC, dtype=torch.int32, device=dev)
    s.sample_indices = torch.arange(MAX_REQS * N_SPEC, dtype=torch.int64, device=dev)
    slots = torch.tensor([2, 0, 3, 1, 6, 5, 4, 7], dtype=torch.int32, device=dev)
    s.sample_idx_mapping = slots.repeat_interleave(N_SPEC)
    ctx = torch.tensor(
        [360000, 40000, 7, 99999, 131072, 5, 777777, 12], dtype=torch.int64, device=dev
    )
    steps = torch.arange(N_SPEC, device=dev)
    s.sample_pos = (ctx[:, None] + 1 + steps[None, :]).flatten()
    s.input_buffers = SimpleNamespace(
        input_ids=torch.randint(0, VOCAB, (MAX_REQS * N_SPEC,), generator=g, device=dev)
    )
    s._anchor_idx = torch.arange(MAX_REQS, dtype=torch.int64, device=dev) * N_SPEC
    s.draft_tokens = torch.zeros(MAX_REQS, N_SPEC, dtype=torch.int64, device=dev)
    if greedy:
        s.draft_logits = None
    return s


def _sample_sequential(s, num_reqs, hidden):
    """``DSparkSpeculator._sample_sequential`` as ``levers.dspark`` patches it
    under latency_specialized."""
    from vllm.models.kimi_k3.amd.specialized_decode.levers import dspark
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    dspark._wrap_sample_sequential(DSparkSpeculator._sample_sequential)(
        s, num_reqs, hidden
    )


def _set_markov(monkeypatch, on: bool):
    from vllm.models.kimi_k3.amd.specialized_decode import levers

    monkeypatch.setattr(levers, "_active", True)
    # The environment, not envs' attribute: monkeypatch would restore the value
    # envs' __getattr__ returned as a plain attribute, pinning the flag for the
    # tests after this one.
    monkeypatch.setenv("VLLM_K31_BEAT_MARKOV", "1" if on else "0")


def _run(monkeypatch, s, num_reqs, fused: bool, hidden=None):
    _set_markov(monkeypatch, fused)
    s.draft_tokens.zero_()
    s.draft_token_confidence_probs.zero_()
    if s.draft_logits is not None:
        s.draft_logits.values.zero_()
    if hidden is None:
        hidden = torch.zeros(MAX_REQS * N_SPEC, 8, device="cuda")
    _sample_sequential(s, num_reqs, hidden)
    torch.accelerator.synchronize()
    cache = None if s.draft_logits is None else s.draft_logits.values.clone()
    return s.draft_tokens.clone(), cache


def _spy_run(monkeypatch):
    """Record what each markov_sample.run call returns."""
    from vllm.models.kimi_k3.amd.ops import markov_sample

    ran = []
    real = markov_sample.run

    def run(*args, **kwargs):
        ran.append(real(*args, **kwargs))
        return ran[-1]

    monkeypatch.setattr(markov_sample, "run", run)
    return ran


def _assert_slots_clear():
    from vllm.models.kimi_k3.amd.ops import markov_sample

    assert markov_sample._SLOTS
    for slots in markov_sample._SLOTS.values():
        assert not bool(slots.view(torch.int64).any())


def _cache_rows(s, num_reqs):
    """Cache row of each request (-1 when another rank caches it)."""
    slots = s.sample_idx_mapping[: num_reqs * N_SPEC : N_SPEC].long()
    if s.draft_logits.row_mapping is None:
        return slots
    return s.draft_logits.row_mapping[slots].long()


@pytest.mark.parametrize("draft", DRAFTS)
@pytest.mark.parametrize("row_mapped", [False, True])
@pytest.mark.parametrize("temp", [0.0, 0.6, 1.0])
@pytest.mark.parametrize("num_reqs", [1, 2, 5])
def test_k31_fused_markov_matches_vllm_loop(
    monkeypatch, num_reqs, temp, row_mapped, draft
):
    from vllm.models.kimi_k3.amd.ops import markov_sample
    from vllm.models.kimi_k3.amd.specialized_decode.levers import dspark
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    # The server takes the fused launches for 1 request only (MAX_ROWS); they
    # handle more, so check those too.
    monkeypatch.setattr(markov_sample, "MAX_ROWS", MAX_REQS)
    s = _speculator(
        temp, seed=int(temp * 10) + num_reqs, row_mapped=row_mapped, draft=draft
    )
    assert markov_sample.why_not(s, num_reqs) is None

    ref_tok, ref_cache = _run(monkeypatch, s, num_reqs, fused=False)
    ran = _spy_run(monkeypatch)
    got_tok, got_cache = _run(monkeypatch, s, num_reqs, fused=True)
    # The fused launches ran (no silent fallback to vLLM's loop), and left
    # their slots cleared for the next call (or graph replay).
    assert ran == [True]
    assert dspark.fused_markov_ok(s, num_reqs)
    _assert_slots_clear()

    rows = _cache_rows(s, num_reqs)
    local = rows >= 0
    # Rows of other ranks must stay untouched.
    written = got_cache.view(-1, VOCAB).abs().sum(-1) != 0
    assert int(written.sum()) == int(local.sum()) * N_SPEC

    lr = rows[local]
    rc, gc = ref_cache[lr], got_cache[lr]  # [local reqs, N_SPEC, VOCAB]
    step_diff = (rc.view(torch.int16) != gc.view(torch.int16)).sum(dim=-1)
    if step_diff.numel():
        assert int(step_diff.max()) <= VOCAB // 1000, (
            f"cached logits that differ, by request and step: {step_diff.tolist()}"
        )
    tok_diff = ref_tok[:num_reqs] != got_tok[:num_reqs]
    # A token may differ only after a rounding difference in its own step or
    # an earlier difference in its chain (the next step then has another prev).
    for r_i, r in enumerate(local.nonzero().flatten().tolist()):
        for k in range(N_SPEC):
            if not tok_diff[r, k]:
                continue
            earlier = bool(tok_diff[r, :k].any()) or bool(
                (step_diff[r_i, : k + 1] > 0).any()
            )
            assert earlier, f"draft token ({r}, {k}) differs without a cause"

    # Exact sampling: each fused token is what vLLM's sampler draws from the
    # logits the fused launch cached; the cached logits are the base logits plus
    # the reference bias of the fused previous token.
    idx_map = s.sample_idx_mapping[: num_reqs * N_SPEC].view(num_reqs, N_SPEC)
    pos = s.sample_pos[: num_reqs * N_SPEC].view(num_reqs, N_SPEC)
    base = s.model.base[: num_reqs * N_SPEC].view(num_reqs, N_SPEC, VOCAB)
    anchors = s.input_buffers.input_ids[s._anchor_idx[:num_reqs]]
    for k in range(N_SPEC):
        prev = anchors if k == 0 else got_tok[:num_reqs, k - 1]
        ref_logits = base[:, k] + s.model.markov_bias(s.model.markov_embed(prev))
        if bool(local.any()):
            g_k = got_cache[lr, k]
            nd = (g_k.view(torch.int16) != ref_logits[local].view(torch.int16)).sum(-1)
            assert int(nd.max()) <= VOCAB // 1000, nd.tolist()
            # Sample from the fused cached logits (for local rows) exactly.
            logits_k = ref_logits.clone()
            logits_k[local] = g_k
        else:
            logits_k = ref_logits
        drawn = DSparkSpeculator._sample_logits(
            s, logits_k, idx_map[:, k], pos[:, k], k
        )
        fused = got_tok[:num_reqs, k]
        # Rows of other ranks sample from the reference logits here, which may
        # differ in a last bit from the fused ones; check local rows exactly.
        assert torch.equal(drawn[local], fused[local]), (
            f"draft step {k}: fused {fused.tolist()}, vLLM draws {drawn.tolist()}"
        )


@pytest.mark.parametrize("draft", DRAFTS)
def test_k31_fused_markov_why_not(draft):
    """The cases where the fused launches would not sample as vLLM's loop."""
    from vllm.models.kimi_k3.amd.ops import markov_sample

    s = _speculator(1.0, seed=3, row_mapped=False, draft=draft)
    assert markov_sample.why_not(s, 1) is None
    assert markov_sample.why_not(s, 2) is not None  # MAX_ROWS = 1
    model = s.model
    s.model = SimpleNamespace(**vars(model))  # not a Kimi-K3.1 DSpark draft
    assert markov_sample.why_not(s, 1) is not None
    s.model = model
    s.draft_watermarker = object()
    assert markov_sample.why_not(s, 1) is not None
    s.draft_watermarker = None
    cache, s.draft_logits = s.draft_logits, None  # greedy drafts: served
    assert markov_sample.why_not(s, 1) is None
    s.draft_logits = cache
    s.use_confidence_head = True  # served: computed after the launches
    assert markov_sample.why_not(s, 1) is None
    s.use_confidence_head = False
    s.model.draft_id_to_target_id = torch.zeros(4, dtype=torch.long)
    assert markov_sample.why_not(s, 1) is not None  # a reduced vocabulary
    s.model.draft_id_to_target_id = None
    s.use_fp64_gumbel = True
    assert markov_sample.why_not(s, 1) is not None


@pytest.mark.parametrize("draft", DRAFTS)
@pytest.mark.parametrize("num_reqs", [1, 2, 5])
def test_k31_fused_markov_greedy(monkeypatch, num_reqs, draft):
    """Greedy drafts (no draft logits cache): each fused token is the argmax of
    the base logits plus vLLM's bias of the fused previous token. Only a tie
    within the bias's last-bit rounding may pick another token than vLLM's
    loop."""
    from vllm.models.kimi_k3.amd.ops import markov_sample

    monkeypatch.setattr(markov_sample, "MAX_ROWS", MAX_REQS)
    s = _speculator(0.0, seed=21 + num_reqs, row_mapped=False, draft=draft, greedy=True)
    assert markov_sample.why_not(s, num_reqs) is None
    ref_tok, _ = _run(monkeypatch, s, num_reqs, fused=False)
    ran = _spy_run(monkeypatch)
    got_tok, _ = _run(monkeypatch, s, num_reqs, fused=True)
    assert ran == [True]
    _assert_slots_clear()

    base = s.model.base[: num_reqs * N_SPEC].view(num_reqs, N_SPEC, VOCAB)
    anchors = s.input_buffers.input_ids[s._anchor_idx[:num_reqs]]
    for k in range(N_SPEC):
        prev = anchors if k == 0 else got_tok[:num_reqs, k - 1]
        ref = (base[:, k] + s.model.markov_bias(s.model.markov_embed(prev))).float()
        top = ref.max(-1).values
        picked = ref.gather(1, got_tok[:num_reqs, k : k + 1]).squeeze(1)
        # One bf16 step (2^-7 relative) at the top value.
        assert bool((picked >= top - top.abs() * 2**-7).all()), (k, picked, top)
    tok_diff = ref_tok[:num_reqs] != got_tok[:num_reqs]
    # Nearly every chain matches vLLM's loop token for token.
    assert int(tok_diff.any(-1).sum()) <= max(1, num_reqs // 2), tok_diff.tolist()


@pytest.mark.parametrize("draft", DRAFTS)
@pytest.mark.parametrize("greedy", [False, True])
def test_k31_fused_markov_confidence_head(monkeypatch, draft, greedy):
    """With the speculator's confidence head on, the fused path hands the head
    the same hidden rows and Markov embeddings (of the anchor, then each
    drafted token) as vLLM's loop: equal tokens give equal confidences."""
    from vllm.models.kimi_k3.amd.ops import markov_sample

    num_reqs = 3
    monkeypatch.setattr(markov_sample, "MAX_ROWS", MAX_REQS)
    s = _speculator(1.0, seed=5, row_mapped=False, draft=draft, greedy=greedy)
    s.use_confidence_head = True
    assert markov_sample.why_not(s, num_reqs) is None
    g = torch.Generator(device="cuda").manual_seed(99)
    hidden = torch.randn(MAX_REQS * N_SPEC, 8, generator=g, device="cuda")
    ref_tok, _ = _run(monkeypatch, s, num_reqs, fused=False, hidden=hidden)
    ref_conf = s.draft_token_confidence_probs.clone()
    ran = _spy_run(monkeypatch)
    got_tok, _ = _run(monkeypatch, s, num_reqs, fused=True, hidden=hidden)
    got_conf = s.draft_token_confidence_probs.clone()
    assert ran == [True]
    same = (ref_tok[:num_reqs] == got_tok[:num_reqs]).all(-1)
    assert bool(same.any()), "no request drafted the same tokens"
    assert torch.equal(ref_conf[:num_reqs][same], got_conf[:num_reqs][same])
    assert bool((got_conf[:num_reqs] > 0).all())
    # Rows past the batch stay untouched.
    assert not bool(got_conf[num_reqs:].any())


@pytest.mark.parametrize("row_mapped", [False, True])
def test_k31_fused_markov_padded_rows(monkeypatch, row_mapped):
    """The capture state of DFlashSpeculator.capture(): sample rows of no
    request (-1) at position 0. Nothing is cached and the tokens stay in the
    vocabulary."""
    s = _speculator(1.0, seed=11, row_mapped=row_mapped)
    s.sample_idx_mapping.fill_(-1)
    s.sample_pos.zero_()
    ran = _spy_run(monkeypatch)
    tok, cache = _run(monkeypatch, s, 1, fused=True)
    assert ran == [True]
    _assert_slots_clear()
    assert not bool(cache.view(torch.int16).any())
    assert bool(((tok[0] >= 0) & (tok[0] < VOCAB)).all()), tok[0].tolist()


@pytest.mark.parametrize("row_mapped", [False, True])
def test_k31_fused_markov_graph_replay(monkeypatch, row_mapped):
    """One request's draft step captured in a graph, as in the DFlash FULL
    graph, and replayed with new base logits, seeds and anchors: each replay
    matches an eager call, keeps the tokens in the vocabulary and leaves the
    slots cleared for the next replay (the last launch clears what it read)."""
    s = _speculator(1.0, seed=7, row_mapped=row_mapped)
    _set_markov(monkeypatch, True)
    ran = _spy_run(monkeypatch)
    hidden = torch.zeros(MAX_REQS * N_SPEC, 8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _sample_sequential(s, 1, hidden)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _sample_sequential(s, 1, hidden)
    assert ran == [True] * 3

    g = torch.Generator(device="cuda").manual_seed(1234)
    for i in range(200):
        s.model.base.copy_(
            torch.randn(s.model.base.shape, generator=g, device="cuda") * 3
        )
        s.seeds.copy_(
            torch.randint(0, 2**62, s.seeds.shape, generator=g, device="cuda")
        )
        s.input_buffers.input_ids.copy_(
            torch.randint(
                0, VOCAB, s.input_buffers.input_ids.shape, generator=g, device="cuda"
            )
        )
        graph.replay()
        torch.accelerator.synchronize()
        g_tok = s.draft_tokens[0].clone()
        g_cache = s.draft_logits.values.clone()
        assert bool(((g_tok >= 0) & (g_tok < VOCAB)).all()), (i, g_tok.tolist())
        _assert_slots_clear()
        _sample_sequential(s, 1, hidden)
        torch.accelerator.synchronize()
        assert torch.equal(s.draft_tokens[0], g_tok), (
            i,
            g_tok.tolist(),
            s.draft_tokens[0].tolist(),
        )
        assert torch.equal(s.draft_logits.values, g_cache), i
        _assert_slots_clear()


def _tp_run(monkeypatch, temp, row_mapped, greedy, tp_size, padded=False):
    """VLLM_K31_BEAT_MARKOVTP's launches (``markov_sample.run``'s ``tp``) on
    ``tp_size`` simulated ranks of one GPU: a thread each, in lockstep, with
    their own slots and logits slices, the all-gather concatenating every
    rank's tensor along the last dim. Returns each rank's (tokens, cache) and
    the whole-vocabulary launch's on the same state."""
    import threading

    from vllm.models.kimi_k3.amd.ops import markov_sample

    def make():
        s = _speculator(temp, seed=31, row_mapped=row_mapped, greedy=greedy)
        if padded:
            s.sample_idx_mapping.fill_(-1)
            s.sample_pos.zero_()
        if s.draft_logits is not None:
            s.draft_logits.values.zero_()
        return s

    def base(s):
        return s.model.base[:N_SPEC].view(1, N_SPEC, VOCAB)

    ref = make()
    assert markov_sample.run(ref, 1, base(ref))
    torch.accelerator.synchronize()
    want = (ref.draft_tokens[0].clone(), _cache_of(ref))
    del ref

    # Per-thread slots and slices (the module keeps one set a device).
    local = threading.local()
    real_slots = markov_sample._slots

    def slots(*a):
        if not hasattr(local, "slots"):
            local.slots = real_slots(*a).clone()
        return local.slots

    def logits(device, n_spec, nv):
        if not hasattr(local, "loc"):
            local.loc = torch.empty(n_spec, nv, dtype=torch.bfloat16, device=device)
        return local.loc

    monkeypatch.setattr(markov_sample, "_slots", slots)
    monkeypatch.setattr(markov_sample, "_local_logits", logits)
    specs = [make() for _ in range(tp_size)]
    # Compile once, single-threaded (any gather of the right shape will do).
    warm = make()
    markov_sample.run(
        warm, 1, base(warm), (0, tp_size, lambda x: torch.cat([x] * tp_size, -1))
    )
    torch.accelerator.synchronize()
    del warm
    local.__dict__.clear()

    bar = threading.Barrier(tp_size)
    bufs: list = [None] * tp_size
    errs: list = []

    def rank(r):
        def gather(x):
            bufs[r] = x.clone()
            bar.wait()
            out = torch.cat(bufs, -1)
            bar.wait()
            return out

        try:
            assert markov_sample.run(specs[r], 1, base(specs[r]), (r, tp_size, gather))
            torch.accelerator.synchronize()
            # Every rank leaves its slots cleared.
            assert not bool(local.slots.view(torch.int64).any())
        except BaseException as e:  # noqa: BLE001
            errs.append(e)
            bar.abort()

    threads = [threading.Thread(target=rank, args=(r,)) for r in range(tp_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise errs[0]
    got = [(s.draft_tokens[0].clone(), _cache_of(s)) for s in specs]
    return want, got


def _cache_of(s):
    return None if s.draft_logits is None else s.draft_logits.values.clone()


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize(
    "temp,row_mapped,greedy,padded",
    [
        (1.0, False, False, False),
        (0.6, True, False, False),
        (0.0, False, False, False),
        (0.0, False, True, False),
        (1.0, False, False, True),
    ],
)
def test_k31_fused_markov_tp_split(
    monkeypatch, tp_size, temp, row_mapped, greedy, padded
):
    """The TP-split launches give every rank exactly the whole-vocabulary
    launch's draft tokens and cached logits (the same bias, noise and key per
    token; only which rank computes it changes)."""
    (want_tok, want_cache), got = _tp_run(
        monkeypatch, temp, row_mapped, greedy, tp_size, padded
    )
    for r, (tok, cache) in enumerate(got):
        assert torch.equal(tok, want_tok), (r, tok.tolist(), want_tok.tolist())
        if want_cache is not None:
            assert torch.equal(cache.view(torch.int16), want_cache.view(torch.int16)), r
    if padded:
        assert not bool(got[0][1].view(torch.int16).any())


def test_k31_fused_markov_tp_split_why_not():
    from vllm.models.kimi_k3.amd.ops import markov_sample

    assert markov_sample.tp_split_why_not(1, VOCAB, 8) is None
    assert markov_sample.tp_split_why_not(1, VOCAB, 2) is None
    assert markov_sample.tp_split_why_not(2, VOCAB, 8) is not None
    assert markov_sample.tp_split_why_not(1, VOCAB, 1) is not None
    assert markov_sample.tp_split_why_not(1, VOCAB, 6) is not None
    assert markov_sample.tp_split_why_not(1, VOCAB + 32, 8) is not None
