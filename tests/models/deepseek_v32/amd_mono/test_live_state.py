# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the live-mode step logic (mono/live.py ``MonoLive``, built bare around
fake kernel ops) and the library's mono-layer custom op: the eager health check and its
fail-stop, rank-uniform votes, the op schema and no-alias contract, the step decision
outside the op, KV cache rebinding after CUDA-graph memory profiling, RoPE tables /
metadata lookup, and ``Glm5MonoKernel.forward``'s launch caching (needs FlyDSL)."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.common.mono import MonoRuntime, StepDecision
from vllm.models.common.mono.runtime import NO_STEP
from vllm.models.deepseek_v32.amd.mono import dispatch as D
from vllm.models.deepseek_v32.amd.mono import guards as G
from vllm.models.deepseek_v32.amd.mono import live as LV
from vllm.models.deepseek_v32.amd.mono.kernel.glm.layout import POLL_STAGES
from vllm.models.deepseek_v32.amd.mono.spec import GLM5_MONO

H = 16  # stand-in hidden size


@pytest.fixture(autouse=True)
def _single_rank_boundary_norm(monkeypatch):
    """The first mono layer's boundary norm on one rank (all-reduce = identity), unfused
    path: residual += h in place."""
    from vllm.models.common.ops import fused_allreduce_rms_norm as far

    def _far(h, r, norm):
        r.add_(h)
        return norm(r), r

    monkeypatch.setattr(far, "fused_allreduce_rms_norm", _far)


class ScratchOp:
    """Sticky poll-error words in a scratch buffer, as Glm5MonoKernel.poll_error."""

    def __init__(self):
        n = 4 * len(POLL_STAGES)
        self.scr_layout = {"poll_err": 0, "poll_abort": n}
        self.scratch = torch.zeros(n + 4, dtype=torch.uint8)
        self.step, self.poll_xrank = torch.zeros(1), torch.zeros(8, dtype=torch.int32)

    def words(self):
        return self.scratch.view(torch.int32)

    def poll_error(self, clear=True):
        expired = tuple(n for n, w in zip(POLL_STAGES, self.words().tolist()) if w)
        if clear and expired:
            self.words().zero_()
        return expired


def bare_rt(widths=(1,), vote=True, vote_each_step=False):
    """A MonoRuntime on CPU whose vote is a recording single-rank identity."""
    rt = MonoRuntime.__new__(MonoRuntime)
    rt.spec = replace(GLM5_MONO, widths=widths)
    rt.vllm_config = None
    rt.rank, rt.world, rt.cpu_group = 0, 1, None
    rt.device = torch.device("cpu")
    rt._peer_factory, rt.epoch = None, None
    rt._voting, rt._vote_each_step = vote, vote_each_step
    rt._reserved, rt.enabled, rt._voted = {}, True, None
    rt.step_index, rt.step = 0, NO_STEP
    rt.votes = []
    rt.vote = lambda ok: (rt.votes.append(ok), ok)[1]
    return rt


def bare_live(step_sync=True, check_every=1, widths=(1,)):
    S = widths[0]
    lv = LV.MonoLive.__new__(LV.MonoLive)
    lv.cfg = LV.LiveConfig(ckpt="x", step_sync=step_sync, check_every=check_every)
    lv.rank = 0
    lv.dev_nonfinite = None
    lv.sizes = widths
    lv.ops = {(3, S): ScratchOp()}
    lv._st = dict(T=1, S=S, md=None)
    lv.rt = bare_rt(widths, vote=step_sync, vote_each_step=check_every <= 0)
    lv.rt.reserve((S, False), owner=lv.ops[(3, S)], width=S)
    lv.rt.step_index = 1
    return lv


@pytest.mark.usefixtures("no_device_sync")
@pytest.mark.parametrize("bad", ["expired", "nonfinite"])
def test_health_check_disables(monkeypatch, bad):
    """MONO_LIVE_FAILSTOP=warn: an expired poll or non-finite output disables mono from
    the next step, through the vote."""
    monkeypatch.setenv("MONO_LIVE_FAILSTOP", "warn")
    lv = bare_live()
    out = torch.zeros(1, 4)
    if bad == "expired":
        lv.ops[(3, 1)].words()[0] = 1
    else:
        out[0, 0] = float("nan")
    lv._end_step(out)
    assert lv.rt.enabled is False and lv.rt.votes == [False]


@pytest.mark.usefixtures("no_device_sync")
def test_eager_expiry_reaches_poll_watch(monkeypatch):
    """Eager health check: an expired poll stays visible to the fail-stop watch (the
    check used to clear the words, so the watch never fired)."""
    zeros = torch.zeros
    monkeypatch.setattr(
        torch, "zeros", lambda *a, pin_memory=False, **k: zeros(*a, **k)
    )
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: NS(record=lambda: None, query=lambda: True)
    )
    out = torch.zeros(1, 4)
    for mode in ("1", "warn", "0"):
        monkeypatch.setenv("MONO_LIVE_FAILSTOP", mode)
        lv = bare_live()
        op = lv.ops[(3, 1)]
        watch = G.PollErrorWatch(lv) if mode != "0" else None
        if watch:
            watch.after_step()  # clean snapshot of step N-1
        op.words()[0] = 1  # step N: a wait expires
        if mode == "1":  # raise (default): the eager check fail-stops, words stay set
            assert watch is not None
            with pytest.raises(RuntimeError, match="fail-stop"):
                lv._end_step(out)
            watch.after_step()  # compute_logits of step N: snapshot
            with pytest.raises(RuntimeError, match="expired kernel polls"):
                watch.after_step()  # step N+1: the watch fail-stops too
            continue
        lv._end_step(out)
        assert lv.rt.enabled is False
        if watch:  # warn: the watch still reports the incident, then clears
            assert op.words()[0] == 1
            watch.after_step()
            watch.after_step()
            assert watch.n_incidents == 1
        assert op.words()[0] == 0  # off (no watch): the check clears the words itself


@pytest.mark.usefixtures("no_device_sync")
def test_step_sync_votes():
    """Rank-uniform votes only at the first step, while off and on check steps: the
    runtime holds the state, MonoLive's end-of-step health check folds a vote into the
    sync it already took."""
    out = torch.zeros(1, 4)
    # check_every=1: one vote per mono step, taken with the health check at its end
    lv = bare_live(check_every=1)
    rt = lv.rt
    assert rt._state_reason() == "" and rt.votes == [True]  # first step: vote at begin
    for n in range(1, 4):
        rt.step_index = n
        lv._end_step(out)
        assert rt._state_reason() == ""  # rides on the end-of-step vote
    assert rt.votes == [True] * 4, rt.votes
    # a local disable applies through the next vote, on every rank at the same step
    rt.enabled = False
    assert rt._state_reason() == ""
    rt.step_index += 1
    lv._end_step(out)
    assert rt.votes[-1] is False and rt._state_reason() == "disabled"
    nv = len(rt.votes)
    assert rt._state_reason() == "disabled" and len(rt.votes) == nv + 1  # off: vote
    rt.enabled = True
    assert rt._state_reason() == "" and rt.votes[-1] is True
    rt.vote = lambda ok: False  # a peer is disabled
    rt._voted = None
    assert rt._state_reason() == "peer_no_go"
    # check_every=3: votes only on check steps
    lv = bare_live(check_every=3)
    lv.rt._state_reason()
    for n in range(1, 7):
        lv.rt.step_index = n
        lv._end_step(out)
        assert lv.rt._state_reason() == ""
    assert len(lv.rt.votes) == 1 + 2, lv.rt.votes
    # check_every=0 (no health check): a vote at every candidate step
    rt = bare_live(check_every=0).rt
    assert [rt._state_reason() for _ in range(3)] == [""] * 3 and len(rt.votes) == 3
    # no step_sync (FULL graphs): never a vote, enabled read directly
    rt = bare_live(step_sync=False).rt
    assert rt._state_reason() == ""
    rt.enabled = False
    assert rt._state_reason() == "disabled" and rt.votes == []


class FakeDecode:
    """A registered MonoOp stand-in: the first layer mutates residual, outputs are
    fresh."""

    def __init__(self):
        self.zbuf = [torch.zeros(8, H) for _ in range(4)]

    def forward(self, positions, hidden_states, residual, layer_idx):
        if layer_idx == 3:
            residual.add_(hidden_states)
        T = hidden_states.shape[0]
        return self.zbuf[layer_idx - 3][:T], residual * 2 + positions[:T, None]


_CPU_LIB: list = []
MODEL_ID = 7


def test_op_schema(monkeypatch):
    """vllm::mono_layer declares the hidden_states / residual mutation and passes
    opcheck; compile(aot_eager) keeps the in-place residual update."""
    from vllm.models.common.mono import op as M
    from vllm.platforms import current_platform

    if current_platform.dispatch_key != "CPU" and not _CPU_LIB:
        _CPU_LIB.append(torch.library.Library("vllm", "IMPL"))  # the impl, for CPU
        _CPU_LIB[0].impl("mono_layer", M._mono_layer, "CPU")
    M.register_mono_layer_op(FakeDecode(), MODEL_ID)
    op = torch.ops.vllm.mono_layer.default
    assert "Tensor(a1!) hidden_states" in str(op._schema)
    assert "Tensor(a2!) residual" in str(op._schema)
    T = 4
    p = torch.arange(T, dtype=torch.float32)
    for L in (3, 4):
        torch.library.opcheck(
            op, (p, torch.randn(T, H), torch.randn(T, H), L, MODEL_ID)
        )

    def f(p, h, r):
        a, b = M.mono_layer(p, h, r, 3, MODEL_ID)
        return a + 1, b

    h, r = torch.randn(T, H), torch.randn(T, H)
    r_e, r_c = r.clone(), r.clone()
    out_e = f(p, h, r_e)
    out_c = torch.compile(f, backend="aot_eager", fullgraph=True)(p, h, r_c)
    assert all(torch.equal(x, y) for x, y in zip(out_e, out_c))
    assert torch.equal(r_e, r_c) and torch.equal(r_c, r + h)


class KOp:
    def forward(self, h, *a, **k):
        return torch.ones(h.shape[0], h.shape[1])


def mono_live(layers=(3, 4, 5), T=2, S=2):
    lv = bare_live(check_every=0, widths=(S,))
    lv.order, lv.first, lv.last = list(layers), layers[0], layers[-1]
    lv.layers = {
        L: NS(layer_idx=L, input_layernorm=lambda x: x * 0.5, self_attn=None)
        for L in layers
    }
    lv.has_indexer = {L: False for L in layers}
    lv.fused_layers = frozenset()
    lv.ops = {(L, S): KOp() for L in layers}
    lv._flat = {L: torch.zeros(1) for L in layers}
    lv._zbuf = [torch.zeros(8, H) for _ in layers]
    lv._zviews, lv._zret = {}, {}
    lv._epoch_kw = {L: dict(layer=L - layers[0], advance=False) for L in layers}
    lv.b_x = torch.zeros(8, H)
    lv.b_curpos = lv.cos = lv.sin = lv.b_indices = torch.zeros(1)
    lv._sviews = {S: (torch.zeros(S), torch.zeros(S), torch.zeros(S + 1))}
    lv._st = dict(T=T, S=S, md=None)
    lv.rt.step = StepDecision(width=S)
    lv._convert_topk = lambda layer: None
    return lv


def test_mono_forward_no_alias():
    lv = mono_live()
    T = 2
    h, r = torch.randn(T, H), torch.randn(T, H)
    r0 = r.clone()
    z, out = lv.mono_forward(lv.layers[3], torch.zeros(T), h, r)
    assert torch.equal(r, r0 + h)  # boundary norm: declared in-place residual update
    for L in (4, 5):
        z_in, res_in = z, out
        z, out = lv.mono_forward(lv.layers[L], torch.zeros(T), z_in, res_in)
        ptrs = {t.untyped_storage().data_ptr() for t in (z_in, res_in)}
        assert not ptrs & {t.untyped_storage().data_ptr() for t in (z, out)}, L
        assert torch.count_nonzero(z) == 0
    with pytest.raises(RuntimeError, match="dispatch state corrupted"):
        lv.mono_forward(lv.layers[5], torch.zeros(T), torch.zeros(T, H), out)


def cache_layers(layers, blocks):
    """Layers with vLLM-shaped MLA [blocks, 16, 576] bf16 and shuffled index [blocks,
    16, 132] caches (no cache when blocks == 0)."""

    def caches(shape, dtype):
        return (
            [torch.zeros(blocks, 16, shape, dtype=dtype)]
            if blocks
            else torch.tensor([])
        )

    return {
        L: NS(
            layer_idx=L,
            self_attn=NS(
                kv_cache=caches(576, torch.bfloat16),
                indexer=NS(
                    k_cache=NS(
                        kv_cache=caches(132, torch.uint8), uses_shuffled_layout=True
                    )
                ),
            ),
        )
        for L in layers
    }


class Layer(NS):
    """vLLM's decoder layer stand-in: records its calls."""

    def __call__(self, p, h, r):
        self.calls.append(("vllm", self.layer_idx))
        return h, r


def decode(lv, calls, monkeypatch, guard=None):
    obj = D.Glm5MonoDecode.__new__(D.Glm5MonoDecode)
    obj.lv, obj.rt, obj._guarded = lv, lv.rt, None
    obj.model = obj.cfg = obj.vllm_config = None
    obj.model_id = MODEL_ID
    if guard is not None:
        obj._maybe_guard = guard
    monkeypatch.setattr(
        D, "mono_layer", lambda p, h, r, L, mid: (calls.append(("op", L)), (h, r))[1]
    )
    return obj


def run_step(obj, calls, layers, residual=True):
    calls.clear()
    h = torch.zeros(2, H)
    for lay in layers.values():
        obj.forward_layer(lay, torch.zeros(2), h, h if residual else None)
    return list(calls)


@pytest.mark.usefixtures("no_device_sync")
def test_dispatch_decision_outside_op(monkeypatch):
    """forward_layer: the step decision and vLLM's fallback run outside the custom op;
    the op only sees go steps; no residual -> vLLM's layer, no decision."""
    calls: list = []
    lv = mono_live()
    go = dict(v=False)

    def step_reason(layer, h, r):
        calls.append(("begin", layer.layer_idx))
        return "" if go["v"] else "no_metadata"

    lv.step_reason = step_reason
    lv.prepare_step = lambda p, S: None
    obj = decode(lv, calls, monkeypatch, guard=lambda layer: True)
    layers = {L: Layer(layer_idx=L, calls=calls) for L in (3, 4, 5)}
    for v, kind in ((False, "vllm"), (True, "op")):
        go["v"] = v
        assert run_step(obj, calls, layers) == [("begin", 3)] + [
            (kind, L) for L in (3, 4, 5)
        ]
    assert run_step(obj, calls, {3: layers[3]}, residual=False) == [("vllm", 3)]


@pytest.mark.usefixtures("no_device_sync")
def test_dispatch_guards_follow_cache_binding(monkeypatch):
    """forward_layer: no bound cache -> vLLM's layer and no stale go decision; the KV
    guards run again for every new binding of the first mono layer's cache."""
    runs = []
    monkeypatch.setattr(
        G, "check_before_install", lambda *a, **k: runs.append("before")
    )
    monkeypatch.setattr(G, "check_after_install", lambda *a, **k: runs.append("after"))
    calls: list = []
    lv = mono_live()
    lv.step_reason = lambda layer, h, r: ""
    lv.prepare_step = lambda p, S: None
    obj = decode(lv, calls, monkeypatch)

    def run(blocks):
        layers = {
            L: Layer(layer_idx=L, calls=calls, self_attn=lay.self_attn)
            for L, lay in cache_layers((3, 4, 5), blocks).items()
        }
        return run_step(obj, calls, layers)

    mono = [("op", 3), ("op", 4), ("op", 5)]
    assert run(8) == mono and len(runs) == 2
    # profiling caches freed, real ones not yet bound: vLLM's layers, active reset
    assert run(0) == [("vllm", 3), ("vllm", 4), ("vllm", 5)] and not lv.rt.step
    assert run(64) == mono and len(runs) == 4


class TableOp:
    tables = None

    def set_index_tables(self, ic, bt):
        self.tables = ic.data_ptr()


def test_caches_rebound_after_profiling():
    """Under FULL graphs, vLLM's CUDA-graph memory profiling binds minimal KV caches (8
    blocks), runs mono steps on them, frees them and binds the real caches. The MLA
    cache views and the fused-indexer tables must follow (they used to be resolved once:
    the real slots then addressed the 8-block buffer -> GPU memory access fault)."""
    lv = bare_live()
    lv.first, lv.sizes = 3, (1, 2)
    lv.fused_layers = frozenset({4})
    lv.ops = {(L, s): TableOp() for L in (3, 4) for s in lv.sizes}
    lv.b_bt = torch.zeros(2, 4, dtype=torch.int32)
    lv._flat, lv._index_tables_set, lv._index_ptrs = {}, False, None
    for blocks in (8, 8192):
        lv.layers = cache_layers((3, 4), blocks)
        lv._ensure_caches(16)  # prepare_step's cache handling
        if not lv._index_tables_set:
            assert lv._set_index_tables()
        mla = {L: lay.self_attn.kv_cache[0].data_ptr() for L, lay in lv.layers.items()}
        idx = lv.layers[4].self_attn.indexer.k_cache.kv_cache[0].data_ptr()
        assert {L: v.data_ptr() for L, v in lv._flat.items()} == mla
        assert lv._flat[3].shape[0] == blocks * 16
        assert all(lv.ops[(4, s)].tables == idx for s in lv.sizes)
    # unchanged caches: nothing rebound, tables kept (no H2D copy per step)
    flat = dict(lv._flat)
    lv._ensure_caches(16)
    assert lv._index_tables_set and all(lv._flat[L] is flat[L] for L in (3, 4))


def test_rope_tables_and_layer_metadata(monkeypatch):
    name = "model.layers.3.self_attn.attn"
    c = torch.randn(100, 64)
    lay = NS(self_attn=NS(rotary_emb=NS(cos_sin_cache=c), layer_name=name))
    cos, sin = LV.rope_tables(lay, 40)
    assert cos.shape == sin.shape == (40, 32)
    assert cos.dtype is torch.bfloat16 and cos.is_contiguous() and sin.is_contiguous()
    assert torch.equal(cos, c[:40, :32].bfloat16()) and torch.equal(
        sin, c[:40, 32:].bfloat16()
    )
    assert LV.rope_tables(lay)[0].shape == (100, 32)
    md, sm = NS(x=1), torch.arange(4)
    for attn_md, slots, want in (
        ({name: md}, {name: sm}, (md, sm)),
        ([{name: md}], {name: sm}, (md, sm)),  # ubatching: the first element
        ([], None, (None, None)),
        (None, sm, (None, None)),
    ):
        fc = NS(attn_metadata=attn_md, slot_mapping=slots)
        monkeypatch.setattr(
            "vllm.forward_context.get_forward_context", lambda fc=fc: fc
        )
        assert LV.layer_metadata(name) == want


def test_op_launch_caching(monkeypatch):
    """Glm5MonoKernel.forward validates once per input signature and builds the weight
    pointers once; the layer range is checked on every launch."""
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v32.amd.mono.kernel.config import KvCacheLayout
    from vllm.models.deepseek_v32.amd.mono.kernel.glm import op as OP

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    keys = (
        "g_in g_q g_kv g_post w_qkv_a w_q_b w_uk w_uv w_o w_r bias w_ug s_ug w_dn s_dn"
    )
    launches = []
    op = OP.Glm5MonoKernel.__new__(OP.Glm5MonoKernel)
    for k, v in dict(
        W=NS(t={k: torch.zeros(4) for k in keys.split()}),
        packed={},
        S=2,
        launches_per_step=4,
        with_indexer=False,
        index_paged=False,
        _index_tables=None,
        kv_cache_layout=KvCacheLayout.ATOM,
        kv_cache_dtype="bf16",
        launch=lambda *a, **k: launches.append(a),
        scratch=torch.zeros(4),
        sym=0,
        peers=torch.zeros(4),
        index_params=None,
        timeline=None,
        step=torch.zeros(1),
        rank=0,
        _validated_sig=None,
        _wptrs=None,
    ).items():
        setattr(op, k, v)
    counts = dict(val=0, wp=0)
    real_validate, real_wp = op._validate, op._weight_ptrs

    def validate(*a):
        counts["val"] += 1
        return real_validate(*a)

    def wptrs():
        counts["wp"] += 1
        return real_wp()

    op._validate, op._weight_ptrs = validate, wptrs
    h = torch.zeros(2, 6144, dtype=torch.bfloat16)
    kv = torch.zeros(8, 576, dtype=torch.bfloat16)
    pos, sl = torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64)
    ip, ind, cur = (
        torch.zeros(5, dtype=torch.int32),
        torch.zeros(8, dtype=torch.int32),
        torch.zeros(1),
    )

    def fwd(positions=None, kv_=kv, layer=1):
        return op.forward(
            h,
            cur,
            kv_,
            kv_,
            ind,
            cur,
            cur,
            layer=layer,
            advance=False,
            positions=pos[:2] if positions is None else positions,
            slot_mapping=sl[:2],
            sparse_kv_indptr=ip[:3],
        )

    for L in (0, 1, 2, 3, 1):
        fwd(layer=L)
    assert counts == dict(val=1, wp=1) and len(launches) == 5
    fwd(positions=torch.zeros(4, dtype=torch.int64)[:2])  # another buffer: re-validated
    assert counts["val"] == 2
    for kw, frag in (
        (dict(positions=torch.zeros(2, dtype=torch.int32)), "positions"),
        (dict(kv_=torch.zeros(8, 512, dtype=torch.bfloat16)), "576"),
        (dict(layer=4), "layer"),
    ):
        with pytest.raises(ValueError, match=frag):
            fwd(**kw)
    # same weight pointers, built once
    assert launches[-1][11:31] == launches[0][11:31] and counts["wp"] == 1
