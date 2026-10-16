# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the mono layers' step logic (``runtime/mono_layers.MonoLayers``,
built bare around fake kernel ops) and the executor's plan (runtime/decode.py): the
eager health check and its fail-stop, rank-uniform votes, the no-alias contract, the
once-per-step plan and Step.layer, the padded step inputs, KV cache binding at every
prepare_kv (CUDA-graph memory profiling, then the real cache), RoPE tables / metadata
lookup, and ``Glm5MonoKernel.forward``'s launch caching (needs FlyDSL)."""

from functools import partial
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.glm.layout import (
    POLL_STAGES,
)
from vllm.models.deepseek_v32.amd.specialized_decode.runtime import (
    BATCH_SIZES,
    DecodeConfig,
)
from vllm.models.deepseek_v32.amd.specialized_decode.runtime import decode as D
from vllm.models.deepseek_v32.amd.specialized_decode.runtime import guards as G
from vllm.models.deepseek_v32.amd.specialized_decode.runtime import mono_layers as ML

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

    _owns_runtime = True

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


def bare_layers(step_sync=True, check_every=1, failstop="raise"):
    ml = ML.MonoLayers.__new__(ML.MonoLayers)
    ml.cfg = DecodeConfig(
        step_sync=step_sync, check_every=check_every, failstop=failstop
    )
    ml.rank, ml.enabled = 0, True
    ml.fused_layers = frozenset()
    ml.dev_nonfinite = None
    ml.batch_sizes = (1,)
    ml.ops = {(3, 1): ScratchOp()}
    ml._st = dict(T=1, S=1, md=None)
    ml._n_steps, ml._voted = 1, None
    ml.votes = []
    ml._go = lambda ok: (ml.votes.append(ok), ok)[1]  # single-rank CPU group
    return ml


@pytest.mark.usefixtures("no_device_sync")
@pytest.mark.parametrize("bad", ["expired", "nonfinite"])
def test_health_check_disables(bad):
    """failstop="warn": an expired poll or non-finite output disables the kernels from
    the next step, through the vote."""
    ml = bare_layers(failstop="warn")
    out = torch.zeros(1, 4)
    if bad == "expired":
        ml.ops[(3, 1)].words()[0] = 1
    else:
        out[0, 0] = float("nan")
    ml._end_step(out)
    assert ml.enabled is False and ml.votes == [False]


@pytest.mark.usefixtures("no_device_sync")
def test_eager_expiry_reaches_poll_watch(monkeypatch):
    """Eager health check: an expired poll stays visible to the fail-stop watch (the
    check must not clear the words before the watch reads them)."""
    zeros = torch.zeros
    monkeypatch.setattr(
        torch, "zeros", lambda *a, pin_memory=False, **k: zeros(*a, **k)
    )
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: NS(record=lambda: None, query=lambda: True)
    )
    out = torch.zeros(1, 4)
    for mode in ("raise", "warn", "off"):
        ml = bare_layers(failstop=mode)
        op = ml.ops[(3, 1)]
        watch = G.PollErrorWatch(ml) if mode != "off" else None
        if watch:
            watch.after_step()  # clean snapshot of step N-1
        op.words()[0] = 1  # step N: a wait expires
        if mode == "raise":  # the eager check fail-stops, words stay set
            assert watch is not None
            with pytest.raises(RuntimeError, match="fail-stop"):
                ml._end_step(out)
            watch.after_step()  # compute_logits of step N: snapshot
            with pytest.raises(RuntimeError, match="expired kernel polls"):
                watch.after_step()  # step N+1: the watch fail-stops too
            continue
        ml._end_step(out)
        assert ml.enabled is False
        if watch:  # warn: the watch still reports the incident, then clears
            assert op.words()[0] == 1
            watch.after_step()
            watch.after_step()
            assert watch.n_incidents == 1
        assert op.words()[0] == 0  # off (no watch): the check clears the words itself


@pytest.mark.usefixtures("no_device_sync")
def test_step_sync_votes():
    """Rank-uniform votes only at the first step, while off and on check steps."""
    out = torch.zeros(1, 4)
    # check_every=1: one vote per mono step, taken with the health check at its end
    ml = bare_layers(check_every=1)
    assert ml._state_reason() == "" and ml.votes == [True]  # first step: vote at begin
    for n in range(1, 4):
        ml._n_steps = n
        ml._end_step(out)
        assert ml._state_reason() == ""  # rides on the end-of-step vote
    assert ml.votes == [True] * 4, ml.votes
    # a local disable applies through the next vote, on every rank at the same step
    ml.enabled = False
    assert ml._state_reason() == ""
    ml._n_steps += 1
    ml._end_step(out)
    assert ml.votes[-1] is False and ml._state_reason() == "disabled"
    nv = len(ml.votes)
    assert ml._state_reason() == "disabled" and len(ml.votes) == nv + 1  # off: vote
    ml.enabled = True
    assert ml._state_reason() == "" and ml.votes[-1] is True
    ml._go = lambda ok: False  # a peer is disabled
    ml._voted = None
    assert ml._state_reason() == "peer_no_go"
    # check_every=3: votes only on check steps
    ml = bare_layers(check_every=3)
    ml._state_reason()
    for n in range(1, 7):
        ml._n_steps = n
        ml._end_step(out)
        assert ml._state_reason() == ""
    assert len(ml.votes) == 1 + 2, ml.votes
    # check_every=0 (no health check): a vote at every candidate step
    ml = bare_layers(check_every=0)
    assert [ml._state_reason() for _ in range(3)] == [""] * 3 and len(ml.votes) == 3
    # no step_sync (FULL graphs): never a vote, enabled read directly
    ml = bare_layers(step_sync=False)
    assert ml._state_reason() == ""
    ml.enabled = False
    assert ml._state_reason() == "disabled" and ml.votes == []


class KOp:
    def forward(self, h, *a, **k):
        return torch.ones(h.shape[0], h.shape[1])


def mono_layers(layers=(3, 4, 5), T=2, S=2):
    ml = bare_layers(check_every=0)
    ml.order, ml.first, ml.last = list(layers), layers[0], layers[-1]
    ml.layers = {
        L: NS(layer_idx=L, input_layernorm=lambda x: x * 0.5, self_attn=None)
        for L in layers
    }
    ml.has_indexer = {L: False for L in layers}
    ml.ops = {(L, S): KOp() for L in layers}
    ml._flat = {L: torch.zeros(1) for L in layers}
    ml._zbuf = [torch.zeros(8, H) for _ in layers]
    ml._zviews, ml._zret = {}, {}
    ml._epoch_kw = {L: dict(layer=L - layers[0], advance=False) for L in layers}
    ml.b_x = torch.zeros(8, H)
    ml.b_curpos = ml.cos = ml.sin = ml.b_indices = torch.zeros(1)
    ml._sviews = {S: (torch.zeros(S), torch.zeros(S), torch.zeros(S + 1))}
    ml.active, ml._st = True, dict(T=T, S=S, md=None)
    ml._convert_topk = lambda layer: None
    return ml


def test_run_layer_no_alias():
    ml = mono_layers()
    T = 2
    h, r = torch.randn(T, H), torch.randn(T, H)
    r0 = r.clone()
    z, out = ml.run_layer(ml.layers[3], torch.zeros(T), h, r)
    assert torch.equal(r, r0 + h)  # boundary norm: declared in-place residual update
    for L in (4, 5):
        z_in, res_in = z, out
        z, out = ml.run_layer(ml.layers[L], torch.zeros(T), z_in, res_in)
        ptrs = {t.untyped_storage().data_ptr() for t in (z_in, res_in)}
        assert not ptrs & {t.untyped_storage().data_ptr() for t in (z, out)}, L
        assert torch.count_nonzero(z) == 0
    with pytest.raises(RuntimeError, match="step state corrupted"):
        ml.run_layer(ml.layers[5], torch.zeros(T), torch.zeros(T, H), out)


def cache_layers(layers, blocks):
    """Layers with vLLM-shaped MLA [blocks, 16, 576] bf16 caches (no cache when
    blocks == 0)."""
    return {
        L: NS(
            layer_idx=L,
            self_attn=NS(
                kv_cache=(
                    [torch.zeros(blocks, 16, 576, dtype=torch.bfloat16)]
                    if blocks
                    else torch.tensor([])
                )
            ),
        )
        for L in layers
    }


class Layer(NS):
    """vLLM's decoder layer stand-in."""


def bare_decode(ml, kv=(1, 1)):
    obj = D.SpecializedDecode.__new__(D.SpecializedDecode)
    obj.mono, obj._kv, obj.backbone, obj.cfg = ml, kv, None, ml.cfg
    obj.kernels = {L: partial(ml.run_layer, lay) for L, lay in ml.layers.items()}
    return obj


@pytest.fixture
def forward_context(monkeypatch):
    fc = NS(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(D, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(D, "get_forward_context", lambda: fc)
    return fc


def test_plan(monkeypatch, forward_context):
    """One decision per step, before the layer loop: a Step whose layer(i) is the
    kernel of mono layer i (None for vLLM's layers); stage inputs, steps before a KV
    bind, PIECEWISE steps and no-go steps run vLLM's layers; a KV cache re-bound
    without prepare_kv raises."""
    ml = mono_layers()
    kv = torch.zeros(4, 16, 576)
    for L in ml.layers:
        ml.layers[L].self_attn = NS(kv_cache=[kv])
    obj = bare_decode(ml, kv=(kv.data_ptr(), kv.numel()))
    go = dict(v=True)
    begins = []

    def begin(p):
        begins.append(p.shape[0])
        ml.active, ml._st = go["v"], dict(T=2, S=2, md=None)
        return 2 if go["v"] else None

    ml.begin_step = begin
    step = obj.plan(torch.zeros(2), None)
    assert step is not None and (step.T, step.S) == (2, 2) and begins == [2]
    assert step.layer(2) is None and step.layer(78) is None
    z, out = step.layer(3)(torch.zeros(2), torch.randn(2, H), torch.randn(2, H))
    assert out.shape == (2, H) and ml._zret[3] is z  # MonoLayers.run_layer
    go["v"] = False
    assert obj.plan(torch.zeros(2), None) is None and not ml.active
    go["v"] = True
    assert obj.plan(torch.zeros(2), object()) is None  # PP stage input
    forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
    assert obj.plan(torch.zeros(2), None) is None
    forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    assert obj.plan(torch.zeros(2), None) is not None
    monkeypatch.setattr(D, "is_forward_context_available", lambda: False)
    assert obj.plan(torch.zeros(2), None) is None
    monkeypatch.setattr(D, "is_forward_context_available", lambda: True)
    assert len(begins) == 3
    ml.active = True
    obj._kv = None  # no KV cache bound yet (profiling / dummy runs)
    assert obj.plan(torch.zeros(2), None) is None and not ml.active
    obj._kv = (1, 1)
    with pytest.raises(RuntimeError, match="re-bound without prepare_kv"):
        obj.plan(torch.zeros(2), None)


@pytest.mark.parametrize(
    "n_dec, T_md, prefills, qlen, topk, T, want",
    [
        (4, 4, 0, 1, 2048, 4, ""),
        (8, 8, 0, 2, 2048, 8, "query_len"),  # a verify step (speculative decoding)
        (4, 40, 1, 36, 2048, 40, "prefill_or_mixed"),
        (4, 8, 0, 1, 2048, 8, "prefill_or_mixed"),
        (8, 8, 0, 1, 2048, 12, "padded_T"),
        (16, 16, 0, 1, 2048, 16, "too_many_rows"),
        (8, 8, 0, 1, 1024, 8, "topk"),
    ],
)
def test_step_reason(n_dec, T_md, prefills, qlen, topk, T, want):
    """The per-step go / no-go rule on attention metadata."""
    md = NS(
        num_decode_tokens=n_dec,
        num_actual_tokens=T_md,
        num_prefills=prefills,
        max_query_len=qlen,
        topk_tokens=topk,
    )
    fits = ML.padded_batch_size(T, BATCH_SIZES) is not None
    assert ML.step_reason(md, T, fits) == want


class StepOp:
    _owns_runtime = True

    def __init__(self):
        self.steps = 0

    def advance_step(self):
        self.steps += 1


def test_begin_step(monkeypatch, no_device_sync):
    """The metadata decision and the padded per-step inputs (3 rows -> batch 4)."""
    ml = bare_layers(step_sync=False)
    ml.batch_sizes, ml.first = (1, 2, 4), 3
    ml.layers = {3: NS(self_attn=NS(layer_name="l3"))}
    ml.block_size = 16
    ml.ops = {(3, 4): StepOp()}
    ml.b_pos = torch.full((4,), 9, dtype=torch.int64)
    ml.b_slot = torch.zeros(4, dtype=torch.int64)
    ml.b_indptr = torch.zeros(5, dtype=torch.int32)
    ml._zbuf = [torch.ones(4, H)]
    md = NS(
        num_prefills=0,
        num_decode_tokens=3,
        num_actual_tokens=3,
        max_query_len=1,
        topk_tokens=2048,
        block_size=16,
        paged_kv_indptr=torch.tensor([0, 5, 9, 12], dtype=torch.int32),
    )
    sm = torch.tensor([7, 8, 9])
    monkeypatch.setattr(ML, "layer_metadata", lambda name: (md, sm))
    pos = torch.tensor([4, 5, 6])
    assert ml.begin_step(pos) == 4 and ml.active and ml._st["S"] == 4
    assert ml.b_pos.tolist() == [4, 5, 6, 0] and ml.b_slot.tolist() == [7, 8, 9, -1]
    assert ml.b_indptr.tolist() == [0, 5, 9, 12, 12]  # padded row: empty sparse range
    assert torch.count_nonzero(ml._zbuf[-1][:3]) == 0 and ml.ops[(3, 4)].steps == 1
    md.num_prefills = 1
    assert ml.begin_step(pos) is None and not ml.active
    md.num_prefills = 0
    assert ml.begin_step(torch.arange(5)) is None  # 5 rows: above the largest batch
    md.block_size = 64
    with pytest.raises(RuntimeError, match="block size"):
        ml.begin_step(pos)
    monkeypatch.setattr(ML, "layer_metadata", lambda name: (None, None))
    assert ml.begin_step(pos) is None


def test_prepare_kv_follows_cache_binding():
    """prepare_kv binds the kernels to each KV cache vLLM binds; while none is bound,
    plan runs vLLM's layers."""
    ml = mono_layers()
    binds = []
    ml.bind_caches = lambda: binds.append(ml.layers[3].self_attn.kv_cache[0].shape)
    obj = bare_decode(ml, kv=None)
    for blocks in (8, 0, 64):
        for L, lay in cache_layers((3, 4, 5), blocks).items():
            ml.layers[L].self_attn = lay.self_attn
        obj.prepare_kv()
        if blocks:
            kv = ml.layers[3].self_attn.kv_cache[0]
            assert obj._kv == (kv.data_ptr(), kv.numel())
        else:
            assert obj._kv is None
    assert [b[0] for b in binds] == [8, 64]


def test_caches_rebound_after_profiling():
    """Under FULL graphs, vLLM's CUDA-graph memory profiling binds minimal KV caches (8
    blocks), runs kernel steps on them, frees them and binds the real caches. The MLA
    cache views follow every bind (kept from the first, the real slots would address
    the 8-block buffer: a GPU memory access fault)."""
    ml = bare_layers()
    ml.first, ml.batch_sizes = 3, (1, 2)
    ml._flat = {}
    for blocks in (8, 8192):
        ml.layers = cache_layers((3, 4), blocks)
        ml.bind_caches()  # prepare_kv
        mla = {L: lay.self_attn.kv_cache[0].data_ptr() for L, lay in ml.layers.items()}
        assert {L: v.data_ptr() for L, v in ml._flat.items()} == mla
        assert ml._flat[3].shape[0] == blocks * 16 and ml.block_size == 16


def test_bind_caches_checks_the_mla_layout():
    ml = bare_layers()
    ml.first = 3
    good = {L: torch.zeros(4, 16, 576, dtype=torch.bfloat16) for L in (3, 4, 5)}
    for frag, kv in {
        None: None,
        "dtype": torch.zeros(4, 16, 576, dtype=torch.float16),
        "not contiguous": torch.zeros(4, 576, 16, dtype=torch.bfloat16).transpose(1, 2),
        "rows of 576": torch.zeros(4, 16, 512, dtype=torch.bfloat16),
        "the same tensor as layer 3": good[3],
    }.items():
        caches = good if kv is None else {**good, 5: kv}
        ml.layers = {L: NS(self_attn=NS(kv_cache=[c])) for L, c in caches.items()}
        if frag is None:
            ml.bind_caches()
            assert ml.block_size == 16 and ml._flat[5].shape == (64, 576)
        else:
            with pytest.raises(RuntimeError, match=f"layer 5 MLA cache: .*{frag}"):
                ml.bind_caches()


class FakeKV:
    def __init__(self, nbytes):
        self.n = nbytes // 2

    def numel(self):
        return self.n

    def element_size(self):
        return 2


@pytest.mark.parametrize(
    "kv, frag",
    [(FakeKV(5 << 30), "layer 5 KV cache is 5.00 GiB >= 4 GiB"), (FakeKV(0), "no KV")],
)
def test_bind_caches_checks_bound_and_size(kv, frag):
    """Every mono layer's KV cache is bound and < 4 GiB."""
    ml = bare_layers()
    ml.first = 3
    caches = {L: torch.zeros(4, 16, 576, dtype=torch.bfloat16) for L in (3, 4)}
    ml.layers = {
        L: NS(self_attn=NS(kv_cache=[c])) for L, c in {**caches, 5: kv}.items()
    }
    with pytest.raises(RuntimeError, match=frag):
        ml.bind_caches()


def test_rope_tables_and_layer_metadata(monkeypatch):
    name = "model.layers.3.self_attn.attn"
    c = torch.randn(100, 64)
    lay = NS(self_attn=NS(rotary_emb=NS(cos_sin_cache=c), layer_name=name))
    cos, sin = ML.rope_tables(lay, 40)
    assert cos.shape == sin.shape == (40, 32)
    assert cos.dtype is torch.bfloat16 and cos.is_contiguous() and sin.is_contiguous()
    assert torch.equal(cos, c[:40, :32].bfloat16()) and torch.equal(
        sin, c[:40, 32:].bfloat16()
    )
    assert ML.rope_tables(lay)[0].shape == (100, 32)
    md, sm = NS(x=1), torch.arange(4)
    for attn_md, slots, want in (
        ({name: md}, {name: sm}, (md, sm)),
        ({}, None, (None, None)),
        (None, sm, (None, None)),
    ):
        fc = NS(attn_metadata=attn_md, slot_mapping=slots)
        monkeypatch.setattr(
            "vllm.forward_context.get_forward_context", lambda fc=fc: fc
        )
        assert ML.layer_metadata(name) == want


def test_op_launch_caching(monkeypatch, import_kernel):
    """Glm5MonoKernel.forward validates once per input signature and builds the weight
    pointers once; the layer range is checked on every launch."""
    OP = import_kernel("glm.op")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    keys = (
        "g_in g_q g_kv g_post w_qkv_a w_q_b w_uk w_uv w_o w_r bias w_ug s_ug w_dn s_dn "
        "w_ug_sh s_ug_sh w_dn_sh s_dn_sh"
    )
    launches = []
    op = OP.Glm5MonoKernel.__new__(OP.Glm5MonoKernel)
    for k, v in dict(
        W=NS(t={k: torch.zeros(4) for k in keys.split()}),
        packed={},
        S=2,
        launches_per_step=4,
        launch=lambda *a, **k: launches.append(a),
        scratch=torch.zeros(4),
        sym=0,
        peers=torch.zeros(4),
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


def _attn_layers(fp8=(), index_fp8=True, layers=(3, 4, 5, 6, 7, 8), indexer=(4, 6, 8)):
    """Mono layers whose attention linears are FP8 for the kernel matrices in ``fp8``
    ({layer: names} or one tuple for all), the indexer layers' wq_b FP8 if
    ``index_fp8``."""

    def w(is_fp8):
        return NS(weight=torch.zeros(2, dtype=torch.float8_e4m3fn if is_fp8 else bf))

    bf = torch.bfloat16
    out, has = {}, {}
    for L in layers:
        f = fp8.get(L, ()) if isinstance(fp8, dict) else fp8
        attn = NS(**{m: w(k in f) for k, m in ML.ATTENTION_MODULES.items()})
        attn.indexer = NS(wq_b=w(index_fp8)) if L in indexer else None
        out[L], has[L] = NS(self_attn=attn), L in indexer
    return out, has


ALL_FP8 = tuple(ML.ATTENTION_MODULES)


@pytest.mark.parametrize(
    "fp8, index_fp8, mml, layers, kw, want",
    [
        # BF16 checkpoint (GLM-5.2-MXFP4): BF16 attention, vLLM's indexer
        ((), False, 4096, range(3, 9), {}, (False, set(), "BF16 attention")),
        # BF16 attention with an FP8 wq_b: still the BF16 path
        ((), True, 4096, range(3, 9), {}, (False, set(), "BF16 attention")),
        # Quark PTPC FP8 attention + FP8 wq_b, max_model_len <= 4096: fused
        (ALL_FP8, True, 4096, range(3, 9), {}, (True, {4, 6, 8}, "")),
        (ALL_FP8, True, 1000, range(3, 9), {}, (True, {4, 6, 8}, "")),
        # > 4096: FP8 attention, non-fused indexer
        (ALL_FP8, True, 4097, range(3, 9), {}, (True, set(), "max_model_len 4097")),
        (ALL_FP8, True, 8192, range(3, 9), {}, (True, set(), "LDS-resident")),
        # FP8 attention, BF16 wq_b: non-fused
        (ALL_FP8, False, 4096, range(3, 9), {}, (True, set(), "wq_b is not FP8")),
        # the first mono layer is an indexer layer: non-fused
        (ALL_FP8, True, 4096, range(4, 9), {}, (True, set(), "first mono layer")),
        # debug switch: non-fused
        (ALL_FP8, True, 4096, range(3, 9), dict(no_fused=True), (True, set(), "debug")),
    ],
)
def test_attention_path(fp8, index_fp8, mml, layers, kw, want):
    """The loaded weights decide the attention precision and the fused indexer."""
    lays, has = _attn_layers(fp8, index_fp8, layers)
    fp8_attn, fused, why = ML.attention_path(lays, has, mml, **kw)
    assert (fp8_attn, set(fused)) == want[:2]
    assert want[2] in why and (why == "") == bool(want[1])


def test_attention_path_refusals():
    """Partial FP8 attention in a layer, or FP8 / BF16 mixed across mono layers, is
    refused with the reason (the kernel never requantizes)."""
    for part in (("q_b",), ("qkv_a", "q_b", "o")):
        lays, has = _attn_layers({L: ALL_FP8 for L in range(3, 9)} | {5: part})
        with pytest.raises(
            RuntimeError,
            match=r"partial FP8 attention on layers \[5\]",
        ):
            ML.attention_path(lays, has, 4096)
    lays, has = _attn_layers({L: ALL_FP8 for L in (3, 4, 5)})
    with pytest.raises(
        RuntimeError, match="mixed attention precision: FP8 on layers 3..5"
    ):
        ML.attention_path(lays, has, 4096)


def test_index_max_seq():
    assert [ML.index_max_seq(n) for n in (1, 512, 513, 4096, 4097)] == [
        512,
        512,
        1024,
        4096,
        4608,
    ]


def test_fused_indexer_step(monkeypatch, no_device_sync):
    """Fused-indexer layers: begin_step copies the decode block table prefix into the
    persistent table (16-token blocks only); run_layer neither refreshes vLLM's indexer
    nor converts its top-k on them (the kernel writes the CSR)."""
    ml = mono_layers(T=2, S=2)
    ml.cfg = DecodeConfig(step_sync=False, check_every=0)
    ml.batch_sizes, ml.block_size = (2,), 16
    ml.has_indexer = {3: False, 4: True, 5: True}
    ml.fused_layers = frozenset({4})
    ml.indexer_refresh = {}
    calls = []
    ml._convert_topk = lambda layer: calls.append(("convert", layer.layer_idx))
    ml.layers[3].self_attn = NS(layer_name="l3")

    class Attn(NS):
        def __call__(self, positions, hidden_states):
            calls.append(("attn", self.L))

    ml.layers[5].self_attn = Attn(L=5)
    ml.b_pos = torch.zeros(2, dtype=torch.int64)
    ml.b_slot = torch.zeros(2, dtype=torch.int64)
    ml.b_indptr = torch.zeros(3, dtype=torch.int32)
    ml.b_bt = torch.zeros(2, 3, dtype=torch.int32)
    ml.ops[(0, 2)] = StepOp()  # the runtime owner (begin_step advances it)
    bt = torch.arange(10, dtype=torch.int32).view(2, 5)
    md = NS(
        num_prefills=0,
        num_decode_tokens=2,
        num_actual_tokens=2,
        max_query_len=1,
        topk_tokens=2048,
        block_size=16,
        block_table=bt,
        paged_kv_indptr=torch.tensor([0, 5, 9], dtype=torch.int32),
    )
    monkeypatch.setattr(ML, "layer_metadata", lambda name: (md, torch.zeros(2)))
    assert ml.begin_step(torch.zeros(2, dtype=torch.int64)) == 2
    assert torch.equal(ml.b_bt, bt[:, :3])
    z, out = ml.run_layer(
        ml.layers[3], torch.zeros(2), torch.zeros(2, H), torch.zeros(2, H)
    )
    for L in (4, 5):
        z, out = ml.run_layer(ml.layers[L], torch.zeros(2), z, out)
    assert calls == [("convert", 3), ("attn", 5), ("convert", 5)]
    md.block_size = ml.block_size = 64
    with pytest.raises(RuntimeError, match="16-token KV block, got 64"):
        ml.begin_step(torch.zeros(2, dtype=torch.int64))
