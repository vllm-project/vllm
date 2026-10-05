# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the live-mode per-step state machine (mono/live.py ``MonoLive``, built
bare with ``__new__`` around fake kernel ops) and of the dispatch custom op:

* enable / disable under FULL cudagraphs is refused (False; captured graphs keep their
  decision); a failed health check under FULL graphs fail-stops instead of disabling;
* the ``vllm::glm5_mono_decode_layer`` schema (mutates hidden_states / residual, fresh
  outputs) passes ``torch.library.opcheck`` and ``torch.compile(aot_eager)``; a schema
  without the mutation fails opcheck; ``mono_forward`` never returns an alias of its
  inputs and keeps the identity check; the step decision runs outside the op;
* rank-uniform votes only at the first step / while off / on check steps;
* the MLA cache views, fused-indexer tables and dispatch KV guards follow a new KV
  cache binding (CUDA-graph memory profiling binds minimal caches first);
* ``Glm5MonoKernel.forward`` validates once per input signature (needs FlyDSL).
"""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.deepseek_v32.amd.mono import dispatch as D
from vllm.models.deepseek_v32.amd.mono import live as LV
from vllm.utils.torch_utils import direct_register_custom_op


@pytest.fixture(autouse=True)
def _single_rank_boundary_norm(monkeypatch):
    """The first mono layer's boundary norm on one rank (all-reduce = identity), unfused
    path: residual += h in place."""
    from vllm.models.common.ops import fused_allreduce_rms_norm as far

    def _far(h, r, norm):
        r.add_(h)
        return norm(r), r

    monkeypatch.setattr(far, "fused_allreduce_rms_norm", _far)


def raises(fn, frag):
    try:
        fn()
    except RuntimeError as e:
        assert frag in str(e), (frag, str(e))
        return str(e)
    raise AssertionError(f"expected RuntimeError containing {frag!r}")


class FakeOp:
    def __init__(self):
        self.expired = ()
        self._owns_runtime = True

    def poll_error(self, clear=True):
        e = self.expired
        if clear:
            self.expired = ()
        return e


def bare_live(full_graphs=False, step_sync=True, check_every=1):
    lv = LV.MonoLive.__new__(LV.MonoLive)
    lv.cfg = LV.LiveConfig(ckpt="x", step_sync=step_sync, check_every=check_every)
    lv.full_graphs = full_graphs
    lv.enabled, lv.disabled_reason = True, ""
    lv.dev_nonfinite = None
    lv.sizes = (1,)
    lv.ops = {(3, 1): FakeOp()}
    lv.stats = dict(
        steps_seen=0,
        steps_mono=1,
        steps_fallback_decode=0,
        steps_no_decode=0,
        decode_tokens=0,
        mono_tokens=0,
        mono_padded_rows=0,
        fallback_reasons={},
        mono_steps_by_S={},
        expired=[],
        nonfinite_steps=0,
    )
    lv._st = dict(T=1, S=1, md=None)
    lv._n_mono, lv._voted = 1, None
    lv.votes = []
    lv._go = lambda ok: (lv.votes.append(ok), ok)[1]  # single-rank CPU group
    return lv


def test_enable_under_full_graphs():
    lv = bare_live(full_graphs=True, step_sync=False)
    assert lv.set_enabled(True) is True and lv.enabled  # no change: no-op
    # refused (an RPC raise would kill the engine)
    assert lv.set_enabled(False) is False and lv.enabled is True
    lv = bare_live(full_graphs=False)
    assert lv.set_enabled(False) is True and lv.enabled is False and lv.disabled_reason
    assert (
        lv.set_enabled(True) is True and lv.enabled is True and lv.disabled_reason == ""
    )


@pytest.mark.usefixtures("no_device_sync")
def test_health_failure_failstop_under_graphs(monkeypatch):
    monkeypatch.setenv("MONO_LIVE_FAILSTOP", "warn")
    for full in (False, True):
        lv = bare_live(full_graphs=full, step_sync=True)
        lv.ops[(3, 1)].expired = ("attn",)
        out = torch.zeros(1, 4)
        if full:
            with pytest.raises(RuntimeError, match="fail-stop"):
                lv._end_step(out)
        else:
            lv._end_step(out)
            assert lv.enabled is False and "expired" in lv.disabled_reason
        lv = bare_live(full_graphs=full, step_sync=True)
        bad = torch.full((1, 4), float("nan"))
        if full:
            with pytest.raises(RuntimeError, match="fail-stop"):
                lv._end_step(bad)
        else:
            lv._end_step(bad)
            assert lv.enabled is False and lv.stats["nonfinite_steps"] == 1


class ScratchOp:
    """Sticky poll-error words in a scratch buffer, as Glm5MonoKernel.poll_error."""

    def __init__(self):
        from vllm.models.deepseek_v32.amd.mono.kernel.glm.layout import POLL_STAGES

        self.stages = POLL_STAGES
        self.scr_layout = {"poll_err": 0}
        self.scratch = torch.zeros(4 * len(POLL_STAGES), dtype=torch.uint8)
        self._owns_runtime = True

    def words(self):
        return self.scratch.view(torch.int32)

    def poll_error(self, clear=True):
        expired = tuple(n for n, w in zip(self.stages, self.words().tolist()) if w)
        if clear and expired:
            self.words().zero_()
        return expired


@pytest.mark.usefixtures("no_device_sync")
def test_eager_expiry_reaches_poll_watch(monkeypatch):
    """Eager health check: an expired poll must stay visible to the fail-stop watch
    (the check used to clear the words, so the watch never fired)."""
    from vllm.models.deepseek_v32.amd.mono import guards as G

    zeros = torch.zeros
    monkeypatch.setattr(
        torch, "zeros", lambda *a, pin_memory=False, **k: zeros(*a, **k)
    )
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: NS(record=lambda: None, query=lambda: True)
    )
    out = torch.zeros(1, 4)

    # raise (default): the eager check fail-stops too, words stay set
    lv = bare_live(step_sync=True)
    op = lv.ops[(3, 1)] = ScratchOp()
    op.words()[0] = 1
    with pytest.raises(RuntimeError, match="fail-stop"):
        lv._end_step(out)
    assert op.words()[0] == 1

    # warn: the check disables mono, the watch still reports the incident and clears
    monkeypatch.setenv("MONO_LIVE_FAILSTOP", "warn")
    lv = bare_live(step_sync=True)
    op = lv.ops[(3, 1)] = ScratchOp()
    watch = G.PollErrorWatch(lv)
    watch.after_step()  # clean snapshot of step N-1
    op.words()[0] = 1  # step N: a wait expires
    lv._end_step(out)
    assert lv.enabled is False and op.words()[0] == 1
    watch.after_step()  # compute_logits of step N: snapshot
    watch.after_step()  # step N+1: check
    assert watch.n_incidents == 1 and op.words()[0] == 0

    # off (no watch): the check clears the words itself
    monkeypatch.setenv("MONO_LIVE_FAILSTOP", "0")
    lv = bare_live(step_sync=True)
    op = lv.ops[(3, 1)] = ScratchOp()
    op.words()[0] = 1
    lv._end_step(out)
    assert lv.enabled is False and op.words()[0] == 0


H = 16  # stand-in hidden size for the op-level tests


class FakeDecode:
    """active() stand-in with the mono path's aliasing / mutation behaviour (first layer
    mutates residual)."""

    def __init__(self):
        self.zbuf = [torch.zeros(8, H) for _ in range(4)]

    def forward_layer_impl(self, layer_idx, positions, hidden_states, residual):
        T = hidden_states.shape[0]
        if layer_idx == 3:
            residual.add_(hidden_states)
        return self.zbuf[layer_idx - 3][:T], residual * 2 + positions[:T, None]


def _cpu_impl(M):
    """Register the op's Python implementation for CPU tensors too (on ROCm the real
    registration targets the CUDA key); the schema and fake impl are the real ones."""
    from vllm.platforms import current_platform

    if current_platform.dispatch_key != "CPU" and "lib" not in _CPU_LIB:
        lib = torch.library.Library("vllm", "IMPL")
        lib.impl("glm5_mono_decode_layer", M._glm5_mono_decode_layer, "CPU")
        _CPU_LIB["lib"] = lib


_CPU_LIB: dict = {}


def test_op_schema():
    from vllm.models.deepseek_v32.amd.ops import glm5_mono as M

    _cpu_impl(M)
    D._ACTIVE["obj"] = FakeDecode()
    try:
        op = torch.ops.vllm.glm5_mono_decode_layer.default
        schema = str(op._schema)
        assert (
            "Tensor(a1!) hidden_states" in schema and "Tensor(a2!) residual" in schema
        ), schema
        T = 4
        for L in (3, 4):
            args = (
                torch.arange(T, dtype=torch.float32),
                torch.randn(T, H),
                torch.randn(T, H),
                L,
            )
            torch.library.opcheck(op, args)

        # functionalization / compile: same outputs and the same in-place residual
        # update as eager
        def f(p, h, r):
            a, b = M.glm5_mono_decode_layer(p, h, r, 3)
            return a + 1, b

        p, h, r = (
            torch.arange(T, dtype=torch.float32),
            torch.randn(T, H),
            torch.randn(T, H),
        )
        r_e, r_c = r.clone(), r.clone()
        out_e = f(p, h, r_e)
        out_c = torch.compile(f, backend="aot_eager", fullgraph=True)(p, h, r_c)
        assert all(torch.equal(x, y) for x, y in zip(out_e, out_c)) and torch.equal(
            r_e, r_c
        )
        assert torch.equal(r_c, r + h)  # the mutation survived functionalization
        # the old registration (mutates_args=[], output aliasing an input) is caught by
        # the same check
        lib = torch.library.Library("mono_review_test", "DEF")

        def old(
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
            layer_idx: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual.add_(hidden_states)
            return hidden_states, residual * 2

        def old_fake(positions, hidden_states, residual, layer_idx):
            return hidden_states.new_empty(hidden_states.shape), residual.new_empty(
                residual.shape
            )

        direct_register_custom_op(
            "old_layer",
            old,
            mutates_args=[],
            fake_impl=old_fake,
            target_lib=lib,
            dispatch_key="CPU",
        )
        with pytest.raises(
            (torch.testing._internal.optests.OpCheckError, RuntimeError)
        ):
            torch.library.opcheck(
                torch.ops.mono_review_test.old_layer.default,
                (p, torch.randn(T, H), torch.randn(T, H), 3),
            )
    finally:
        D._ACTIVE["obj"] = None


class KOp:
    def forward(self, h, *a, **k):
        return torch.ones(h.shape[0], h.shape[1])


def mono_live(layers=(3, 4, 5), T=2, S=2):
    lv = bare_live()
    lv.order = list(layers)
    lv.first, lv.last = layers[0], layers[-1]
    lv.layers = {
        L: NS(layer_idx=L, input_layernorm=lambda x: x * 0.5, self_attn=None)
        for L in layers
    }
    lv.has_indexer = {L: False for L in layers}
    lv.fused_layers = frozenset()
    lv.ops = {(L, S): KOp() for L in layers}
    lv._flat = {L: torch.zeros(1) for L in layers}
    lv.max_rows = 8
    lv._zbuf = [torch.zeros(8, H) for _ in layers]
    lv._zviews = {}
    lv._epoch_kw = {L: dict(layer=L - layers[0], advance=False) for L in layers}
    lv._zret = {}
    lv.b_x = torch.zeros(8, H)
    lv.b_curpos = lv.cos = lv.sin = lv.b_indices = torch.zeros(1)
    lv.b_pos, lv.b_slot, lv.b_indptr = torch.zeros(8), torch.zeros(8), torch.zeros(9)
    lv._sviews = {S: (lv.b_pos[:S], lv.b_slot[:S], lv.b_indptr[: S + 1])}
    lv.cfg.check_every = 0
    lv.active = True
    lv._st = dict(T=T, S=S, md=None)
    lv._convert_topk = lambda layer: None
    return lv


def test_mono_forward_no_alias():
    lv = mono_live()
    T = 2
    h, r = torch.randn(T, H), torch.randn(T, H)
    r0 = r.clone()
    z, out = lv.mono_forward(lv.layers[3], torch.zeros(T), h, r)
    # first layer: boundary norm updated residual in place (declared mutation)
    assert torch.equal(r, r0 + h)
    for L in (4, 5):
        z_in, res_in = z, out
        z, out = lv.mono_forward(lv.layers[L], torch.zeros(T), z_in, res_in)
        for o in (z, out):
            for i in (z_in, res_in):
                assert (
                    o.untyped_storage().data_ptr() != i.untyped_storage().data_ptr()
                ), (L, "output aliases input")
        assert torch.count_nonzero(z) == 0
    raises(
        lambda: lv.mono_forward(lv.layers[5], torch.zeros(T), torch.zeros(T, H), out),
        "dispatch state corrupted",
    )


def test_dispatch_decision_outside_op():
    """forward_layer: the step decision and vLLM's fallback run outside the custom op;
    the op only sees go steps."""
    calls = []
    lv = mono_live()
    go = dict(v=False)

    def begin(layer, p, h, r):
        calls.append(("begin", layer.layer_idx))
        lv.active = go["v"]

    lv._begin_step = begin
    obj = D.Glm5MonoDecode.__new__(D.Glm5MonoDecode)
    obj.lv = lv
    obj._maybe_guard = lambda layer: True

    def op(p, h, r, L):
        calls.append(("op", L))
        return h, r

    obj._op = op

    class Layer(NS):
        def __call__(self, p, h, r):
            calls.append(("vllm", self.layer_idx))
            return h, r

    layers = {L: Layer(layer_idx=L) for L in (3, 4, 5)}
    h, r = torch.zeros(2, H), torch.zeros(2, H)
    for v in (False, True):
        go["v"] = v
        calls.clear()
        for L in (3, 4, 5):
            obj.forward_layer(layers[L], torch.zeros(2), h, r)
        kind = "op" if v else "vllm"
        assert calls == [("begin", 3), (kind, 3), (kind, 4), (kind, 5)], calls
    calls.clear()
    # no residual: vLLM's layer, no decision
    obj.forward_layer(layers[3], torch.zeros(2), h, None)
    assert calls == [("vllm", 3)], calls


def _cache_layers(layers, blocks):
    """Fake layers with vLLM-shaped MLA [blocks, 16, 576] bf16 and shuffled index
    [blocks, 16, 132] caches."""
    return {
        L: NS(
            layer_idx=L,
            self_attn=NS(
                kv_cache=[torch.zeros(blocks, 16, 576, dtype=torch.bfloat16)],
                indexer=NS(
                    k_cache=NS(
                        kv_cache=[torch.zeros(blocks, 16, 132, dtype=torch.uint8)],
                        uses_shuffled_layout=True,
                    )
                ),
            ),
        )
        for L in layers
    }


class TableOp:
    def __init__(self):
        self.tables = None

    def set_index_tables(self, ic, bt):
        self.tables = ic.data_ptr()


def test_caches_rebound_after_profiling():
    """Under FULL graphs, vLLM's CUDA-graph memory profiling binds minimal KV caches (8
    blocks), runs mono steps on them, frees them and binds the real caches. The MLA
    cache views and the fused-indexer tables must follow (they used to be resolved once:
    the real slots then addressed the 8-block buffer -> GPU memory access fault)."""
    lv = bare_live()
    layers = (3, 4)
    lv.first, lv.sizes = 3, (1, 2)
    lv.fused_layers = frozenset({4})
    lv.ops = {(L, s): TableOp() for L in layers for s in lv.sizes}
    lv.b_bt = torch.zeros(2, 4, dtype=torch.int32)
    lv._flat, lv._index_tables_set, lv._index_ptrs = {}, False, None

    def bind(blocks):
        new = _cache_layers(layers, blocks)
        lv.layers = new
        return new

    def step():
        # _begin_step's cache handling
        lv._ensure_caches(16)
        if lv.fused_layers and not lv._index_tables_set:
            assert lv._set_index_tables()

    def ptrs(new):
        mla = {L: lay.self_attn.kv_cache[0].data_ptr() for L, lay in new.items()}
        idx = new[4].self_attn.indexer.k_cache.kv_cache[0].data_ptr()
        return mla, idx

    minimal = bind(8)
    step()
    mla, idx = ptrs(minimal)
    assert {L: v.data_ptr() for L, v in lv._flat.items()} == mla
    assert all(lv.ops[(4, s)].tables == idx for s in lv.sizes)
    real = bind(8192)
    step()
    mla, idx = ptrs(real)
    assert {L: v.data_ptr() for L, v in lv._flat.items()} == mla
    assert lv._flat[3].shape[0] == 8192 * 16
    assert all(lv.ops[(4, s)].tables == idx for s in lv.sizes)
    # unchanged caches: nothing rebound, tables kept (no H2D copy per step)
    flat = dict(lv._flat)
    lv._ensure_caches(16)
    assert lv._index_tables_set and all(lv._flat[L] is flat[L] for L in layers)


def test_dispatch_guards_follow_cache_binding(monkeypatch):
    """forward_layer: no bound cache -> vLLM's layer and no stale go decision; the KV
    guards run again for every new binding of the first mono layer's cache."""
    from vllm.models.deepseek_v32.amd.mono import guards as G

    runs = []
    monkeypatch.setattr(
        G, "check_before_install", lambda *a, **k: runs.append("before")
    )
    monkeypatch.setattr(G, "check_after_install", lambda *a, **k: runs.append("after"))
    calls = []
    lv = mono_live()
    lv._begin_step = lambda layer, p, h, r: setattr(lv, "active", True)
    obj = D.Glm5MonoDecode.__new__(D.Glm5MonoDecode)
    obj.lv, obj._guarded = lv, None
    obj.model = obj.cfg = obj.vllm_config = None
    obj._op = lambda p, h, r, L: (calls.append(("op", L)), (h, r))[1]

    class Layer(NS):
        def __call__(self, p, h, r):
            calls.append(("vllm", self.layer_idx))
            return h, r

    h, r = torch.zeros(2, H), torch.zeros(2, H)

    def run(blocks):
        caches = _cache_layers((3, 4, 5), blocks) if blocks else None
        calls.clear()
        for L in (3, 4, 5):
            kv = caches[L].self_attn.kv_cache if caches else torch.tensor([])
            layer = Layer(layer_idx=L, self_attn=NS(kv_cache=kv))
            obj.forward_layer(layer, torch.zeros(2), h, r)
        return list(calls)

    assert run(8) == [("op", 3), ("op", 4), ("op", 5)] and len(runs) == 2
    # profiling caches freed, real ones not yet bound: vLLM's layers, active reset
    assert run(0) == [("vllm", 3), ("vllm", 4), ("vllm", 5)] and not lv.active
    assert run(64) == [("op", 3), ("op", 4), ("op", 5)] and len(runs) == 4


@pytest.mark.usefixtures("no_device_sync")
def test_step_sync_votes():
    # check_every=1 (default): one vote per mono step, taken with the health check at
    # the step's end
    lv = bare_live(check_every=1)
    # first step: no standing vote -> vote at begin
    assert lv._state_reason() == "" and lv.votes == [True]
    out = torch.zeros(1, 4)
    for n in range(1, 4):
        lv._n_mono = n
        lv._end_step(out)  # health ok -> vote
        assert lv._state_reason() == ""  # next step: rides on that vote
    assert lv.votes == [True] * 4, lv.votes
    # a local disable (RPC) is applied through the next vote, on every rank at the same
    # step
    lv.set_enabled(False)
    # the standing vote still says on (rank-uniform until the next vote)
    assert lv._state_reason() == ""
    lv._n_mono += 1
    lv._end_step(out)
    assert lv.votes[-1] is False and lv._state_reason() == "disabled"
    nv = len(lv.votes)
    # while off: vote at each candidate step
    assert lv._state_reason() == "disabled" and len(lv.votes) == nv + 1
    lv.set_enabled(True)
    assert lv._state_reason() == "" and lv.votes[-1] is True
    # peer disabled (MIN vote False while this rank is enabled)
    lv2 = bare_live()
    lv2._go = lambda ok: False
    assert lv2._state_reason() == "peer_no_go"
    # check_every=3: the vote only on check steps
    lv = bare_live(check_every=3)
    lv._state_reason()
    for n in range(1, 7):
        lv._n_mono = n
        lv._end_step(out)
        assert lv._state_reason() == ""
    assert len(lv.votes) == 1 + 2, lv.votes  # begin + steps 3 and 6
    # check_every=0 (no health check): vote at every candidate step
    lv = bare_live(check_every=0)
    for _ in range(3):
        assert lv._state_reason() == ""
    assert len(lv.votes) == 3
    # no step_sync (FULL graphs): never a vote, enabled read directly
    lv = bare_live(step_sync=False)
    assert lv._state_reason() == "" and lv.votes == []
    lv.enabled = False
    assert lv._state_reason() == "disabled" and lv.votes == []


def test_op_launch_caching(monkeypatch):
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v32.amd.mono.kernel.config import KvCacheLayout
    from vllm.models.deepseek_v32.amd.mono.kernel.glm import op as OP

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    keys = [
        "g_in",
        "g_q",
        "g_kv",
        "g_post",
        "w_qkv_a",
        "w_q_b",
        "w_uk",
        "w_uv",
        "w_o",
        "w_r",
        "bias",
        "w_ug",
        "s_ug",
        "w_dn",
        "s_dn",
    ]
    launches = []
    op = OP.Glm5MonoKernel.__new__(OP.Glm5MonoKernel)
    for k, v in dict(
        W=NS(t={k: torch.zeros(4) for k in keys}),
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
    nval, nw = [0], [0]
    real_validate, real_wp = op._validate, op._weight_ptrs

    def validate(*a):
        nval[0] += 1
        return real_validate(*a)

    def wptrs():
        nw[0] += 1
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
    assert nval[0] == 1 and nw[0] == 1 and len(launches) == 5, (nval, nw, len(launches))
    ref = launches[0]
    pos2 = torch.zeros(4, dtype=torch.int64)
    fwd(positions=pos2[:2])  # another buffer -> re-validated
    assert nval[0] == 2
    try:
        # changed dtype -> validated -> refused
        fwd(positions=torch.zeros(2, dtype=torch.int32))
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "positions" in str(e)
    try:
        fwd(kv_=torch.zeros(8, 512, dtype=torch.bfloat16))
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "576" in str(e)
    try:
        fwd(layer=4)  # the layer range is checked on every launch
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "layer" in str(e)
    # same weight pointers, built once
    assert launches[-1][11:31] == ref[11:31] and nw[0] == 1
