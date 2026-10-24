# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2(split) (``K2Build(core="split")``, the in-kernel split-KV attention
core, DESIGN_BC 7, WP-C3) at TP8 against the decode vLLM runs, real shapes,
random weights in the post-load formats.

The front is the stock one (``fused_norm_residual_norm_fwd``, the qkv GEMM as
served, the fused QK-norm / RoPE / KV insert into the layer's cache) or K1
(``mono/k1.py``, Phase B1). Every step runs
the same front twice on two copies of the cache: once into the stock core
(``paged_attention_v1``, a sliding layer on ``_build_swa_decode_tables``'
window-trimmed table) and K2(gated) (B6: bit for bit K2(tensor) on vLLM's
gated core), once into K2(split). The metadata is MRV2's persistent buffers
(``reference.FakeMrv2``: one full and one sliding KV group, each its own
block table and slot mapping, a sliding table nulled below the window).

[TP8] ``test_layer_chain_split``: on the live rows the split core's gated rows
(``xo``) agree with ``rms_norm_gated`` of the stock attention (cos) and are
never further from the fp64 gated attention than stock's (E9);
``stream_out`` agrees with the K2(gated) step and with vLLM's stock span from
the stock gated core (cos 0.99999); the tail is Phase A's (``stream_out`` and
``branch_out`` bit for bit K2(tensor) on the split core's own gated rows) and
``branch_out`` agrees with vLLM's experts on its own ffn_in rows and picks
(cos 0.9999, Phase A's reference, DESIGN_BC 9.3; the whole chain's branch is
printed: the stock span itself moves to cos ~0.99 between the two cores);
the pad rows are exact zeros, and NaN in their qkv rows changes nothing; the
ranks agree bit for bit; the cache bytes after the split step equal the gated
step's (the split core never writes the cache); no wait gave up.

[TP8] ``test_graph_replay_follows_metadata``: one CUDA graph of ``step_begin;
front; K2(split)`` for a full and a sliding layer chained (and one of the
same two layers' gated chain, the stock core's max length the table's: a
sliding layer's trimmed table caps it at the window on the device, as a
captured decode does), captured on the persistent buffers; between
replays the host advances ``seq_lens``, positions, slots and the block tables
as MRV2 does (crossing pages and the sliding window); every replay equals an
eager run of the same step on the same contents, 300 replays, health zero.
"""

import math
from types import SimpleNamespace

import pytest
import ray
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    reference as ref,
)
from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_numerics as tn,
)
from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_numerics_front as tnf,
)
from tests.utils import multi_gpu_test, multi_process_parallel  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (  # noqa: E402
    STARTUP_WAIT_SECONDS,
)

bf = torch.bfloat16
f32 = torch.float32
H, CORE, QKV = spec.HIDDEN, spec.CORE, spec.QKV_ROWS
FULL, SLIDING = "full", "sliding"
KINDS = (FULL, SLIDING)
# the kernel layers of the fake model: layer 1 full (NoPE), layer 2 sliding
LAYER = {FULL: 1, SLIDING: 2}
NAME = {FULL: "layers.1.self_attn.attn", SLIDING: "layers.2.self_attn.attn"}
MAX_POS = 1 << 16  # the RoPE table's rows: every position below
MAX_BLOCKS = 2112  # a block table's row: 33,792 keys (> the 32,773 case)
COS_CORE = 0.99999
REPLAYS = 300
# (requests' context lengths, width): rows past the requests are MRV2's pad
# rows (seq_lens 0, slot -1). 2065: a window starting mid-page; 2064: at a
# page start; 2047 .. 2050 around the window; 32773 past 32k; 1, 16, 17 the
# first page's edges
CASES = (
    ((2065,), 1),
    ((32773,), 1),
    ((2049, 17, 2064), 3),
    ((1, 16, 257, 2047, 2050, 4103), 8),
)
# the replay graph's two requests: request 0 crosses the window (and a page
# every 16 steps), request 1 is past it from the start
REPLAY_START = (2030, 4090)
# an fp8 cache's K / V scales: not 1, and e4m3 times either is exact in bf16
KV8_SCALES = (0.75, 1.25)


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the Muse mono kernels run on gfx950 (MI355X)"
)


# ---------------------------------------------------------------- helpers


def gate64(attn, og) -> torch.Tensor:
    """The output norm and gate in fp64 (no rounding): per (token, head) row
    of 64, ``a / sqrt(mean a^2 + eps) sigmoid(g)``. [S, HEADS, 64]."""
    a = attn.double().reshape(attn.shape[0], spec.HEADS, spec.HEAD_DIM)
    g = og.double().reshape(a.shape)
    r = torch.rsqrt(a.square().mean(-1, keepdim=True) + spec.RMS_EPS)
    return a * r * torch.sigmoid(g)


def _fake_mrv2(device) -> ref.FakeMrv2:
    """MRV2's persistent buffers of the two kernel layers: each its own KV
    group, the sliding one's table nulled below the window."""
    return ref.FakeMrv2(
        {FULL: [NAME[FULL]], SLIDING: [NAME[SLIDING]]},
        {SLIDING},
        max_blocks=MAX_BLOCKS,
        device=device,
    )


def _schedule(steps: int = REPLAYS) -> list[tuple[int, ...]]:
    """The replay graph's per-step context lengths."""
    return [tuple(L + i for L in REPLAY_START) for i in range(steps)]


def _scales(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The layer's K / V scales as vLLM holds them (fp32 0-d, the cache's
    device): ``KV8_SCALES`` for an fp8 cache, else 1."""
    ks, vs = KV8_SCALES if kv.dtype == torch.uint8 else (1.0, 1.0)
    return (
        torch.tensor(ks, dtype=f32, device=kv.device),
        torch.tensor(vs, dtype=f32, device=kv.device),
    )


def _pages(lens, nblocks: int, seed: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return ref.assign_pages(lens, nblocks, generator=g)


def _front_weights(rank: int, device) -> dict:
    """One rank's K1 weights (the previous layer's FFN boundary, this layer's
    pre-qkv norm, its qkv_proj shard [2304, 8192], the RoPE table):
    replicated tensors from seed 17, the shard from 300 + rank."""
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )

    gc = torch.Generator(device=device).manual_seed(17)
    gr = torch.Generator(device=device).manual_seed(300 + rank)

    def rn(g, *shape, s=1.0, dtype=bf):
        return (torch.randn(*shape, generator=g, device=device) * s).to(dtype)

    conn = MuseSpark1xNormPreservingResidualConnection(H).to(device)
    with torch.no_grad():
        conn.gate.copy_(rn(gc, H, dtype=f32))
    conn.materialize_coefficients()
    return {
        "w_postffn": rn(gc, H, s=0.1),
        "ffn_alpha": conn.alpha.contiguous(),
        "ffn_beta": conn.beta.contiguous(),
        "w_preqkv": rn(gc, H, s=0.1),
        "w_qkv": rn(gr, QKV, H, s=H**-0.5),
        "cos_sin": ref.rope_table(MAX_POS, device=device),
    }


def _boundary_inputs(device, S: int, seed: int):
    """(branch bf16, stream fp32) [S, H], replicated: the previous layer's
    deferred boundary."""
    g = torch.Generator(device=device).manual_seed(seed)
    branch = torch.randn(S, H, generator=g, device=device).to(bf)
    stream = torch.randn(S, H, generator=g, device=device)
    return branch, stream


def _stock_front(fw, kind, branch, stream, positions, slots, kv):
    """The stock front of a kernel layer: the deferred boundary, the qkv GEMM
    (as served), the fused QK-norm / RoPE / KV insert in place. ->
    (stream_out, qkvo)."""
    from vllm import _custom_ops as ops
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
        fused_norm_residual_norm_fwd,
    )

    attn_in, stream_out = fused_norm_residual_norm_fwd(
        branch,
        stream,
        fw["w_postffn"],
        fw["w_preqkv"],
        spec.POST_FFN_EPS,
        spec.RMS_EPS,
        alpha=fw["ffn_alpha"],
        beta=fw["ffn_beta"],
        post_gain_center=spec.GAIN,
        pre_gain_center=spec.GAIN,
        post_input_scale=True,
        out_dtype=bf,
    )
    qkvo = tn._gemm(attn_in, fw["w_qkv"])
    rope = kind == SLIDING
    ops.fused_muse_spark_qknorm_rope_kv_insert(
        qkvo,
        positions,
        spec.HEADS,
        spec.KV_HEADS,
        spec.HEAD_DIM,
        spec.RMS_EPS,
        fw["cos_sin"] if rope else None,
        spec.HEAD_DIM if rope else 0,
        slots,
        kv,
        spec.KV_BLOCK,
        "fp8" if kv.dtype == torch.uint8 else "auto",
        *_scales(kv),
    )
    return stream_out, qkvo


def _k1_front(runner, fw, kind, branch, stream, positions, slots, pad, kv):
    """K1's front (Phase B1) with the runtime's arguments. -> (stream_out,
    qkvo)."""
    from vllm.models.muse_spark.amd.specialized_decode.runtime import layers

    S = branch.shape[0]
    rope = kind == SLIDING
    stream_out = torch.empty_like(stream)
    qkvo = torch.empty(S, QKV, dtype=bf, device=stream.device)
    layers.k1_launch(
        spec.K1Build(tokens=S, rope=rope, kv_fp8=kv.dtype == torch.uint8),
        {
            **{n: fw[n] for n in ("w_postffn", "ffn_alpha", "ffn_beta", "w_preqkv")},
            "w_qkv": fw["w_qkv"],
            "cos_sin": fw["cos_sin"] if rope else 0,
            "max_pos": MAX_POS if rope else 0,
            **layers.kv_args(kv, *_scales(kv)),
            "branch": branch,
            "stream": stream,
            "pad": pad,
            "positions": positions,
            "slots": slots,
            "stream_out": stream_out,
            "qkvo": qkvo,
            "slot": spec.k1_slot(LAYER[kind]),
            **runner.launch_args(),
            "rank": runner.rank,
        },
    )
    return stream_out, qkvo


def _front(front, runner, fw, kind, branch, stream, step, kv):
    name = NAME[kind]
    if front == "k1":
        return _k1_front(
            runner,
            fw,
            kind,
            branch,
            stream,
            step.positions,
            step.slot_mapping[name],
            step.pad,
            kv,
        )
    return _stock_front(
        fw, kind, branch, stream, step.positions, step.slot_mapping[name], kv
    )


def _stock_attention(qkvo, kv, md, kind, max_len=None):
    """The stock core's attention output bf16 [S, CORE] (``paged_attention_v1``
    as decode runs it, ``reference.stock_attention``); ``max_len``
    defaults to the step's longest context (a graph passes the table's)."""
    S = qkvo.shape[0]
    lens = md.seq_lens.tolist() if max_len is None else [max_len]
    ns = SimpleNamespace(
        case=SimpleNamespace(S=S, window=kind == SLIDING, lens=tuple(lens)),
        kv=kv,
        bt=md.block_table,
        seq_lens=md.seq_lens,
        qkvo=qkvo,
        scales=_scales(kv),
    )
    return ref.stock_attention(ns).reshape(S, CORE)


def _stock_gated(attn, qkvo):
    """VLLM's gated core (o_proj's input) from the attention output."""
    from vllm.third_party.flash_linear_attention.ops.kda import (
        rms_norm_gated,
    )

    S = attn.shape[0]
    return rms_norm_gated(
        attn.view(S, spec.HEADS, 64),
        qkvo[:, spec.ROW_G :].reshape(S, spec.HEADS, 64),
        None,
        None,
        "sigmoid",
        eps=spec.RMS_EPS,
    ).reshape(S, CORE)


def _split_args(qkvo, kv, md) -> dict:
    """K2(split)'s core arguments as ``KernelLayer`` passes them."""
    from vllm.models.muse_spark.amd.specialized_decode.runtime import layers

    bt = md.block_table
    return {
        "qkvo": qkvo,
        **layers.kv_args(kv, *_scales(kv)),
        "block_table": bt,
        "bt_rows": layers.table_rows(bt),
        "bt_stride": bt.stride(0),
        "seq_lens": md.seq_lens,
    }


def _k2(runner, w, key, core_args, stream, pad, kind):
    """One K2 launch of build ``key`` (no epoch bump): (stream_out,
    branch_out)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer

    S = key.tokens
    stream_out = torch.empty_like(stream)
    branch_out = torch.empty(S, H, dtype=bf, device=stream.device)
    layer.k2_launch(
        key,
        {
            **{n: w[n] for n in tn.KERNEL_WEIGHTS},
            **core_args,
            "stream": stream,
            "pad": pad,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "slot": spec.k2_slot(LAYER[kind]),
            **runner.launch_args(),
            **runner.peer_args(),
        },
    )
    return stream_out, branch_out


def _split_key(S: int, kind: str, kv8: bool = False) -> spec.K2Build:
    """The split build of width S and ``kind`` (``kv8``: an fp8 cache)."""
    return spec.K2Build(tokens=S, core="split", window=kind == SLIDING, kv_fp8=kv8)


def _split_step(
    front,
    runner,
    w,
    fw,
    kind,
    branch,
    stream,
    step,
    kv,
    *,
    begin=True,
):
    """``step_begin; front; K2(split)`` of one layer: (stream_out, branch_out,
    qkvo)."""
    S = branch.shape[0]
    if begin:
        runner.step_begin()
    s1, qkvo = _front(front, runner, fw, kind, branch, stream, step, kv)
    md = step.attn_metadata[NAME[kind]]
    key = _split_key(S, kind, kv.dtype == torch.uint8)
    out = _k2(runner, w, key, _split_args(qkvo, kv, md), s1, step.pad, kind)
    return (*out, qkvo)


def _gated_step(
    front, runner, w, fw, kind, branch, stream, step, kv, max_len=None, *, begin=True
):
    """``step_begin; front; stock core; K2(gated)`` of one layer:
    (stream_out, branch_out, qkvo, attn, front's stream_out)."""
    S = branch.shape[0]
    if begin:
        runner.step_begin()
    s1, qkvo = _front(front, runner, fw, kind, branch, stream, step, kv)
    attn = _stock_attention(qkvo, kv, step.attn_metadata[NAME[kind]], kind, max_len)
    key = spec.K2Build(tokens=S, core="gated")
    out = _k2(runner, w, key, {"attn": attn, "qkvo": qkvo}, s1, step.pad, kind)
    return (*out, qkvo, attn, s1)


def _xo(runner, S: int) -> torch.Tensor:
    """The split core's gated rows (the combine's ``xo``) of the last launch."""
    off, nbytes = spec.scratch_layout()[spec.XO]
    return runner.scratch[off : off + nbytes].view(bf).view(-1, CORE)[:S].clone()


def _bytes(kv: torch.Tensor) -> torch.Tensor:
    return kv.view(torch.int16)


def _check_fp8_insert(qkvo, kv, slots, pad) -> None:
    """Every live row's K / V heads in an fp8 cache are the e4m3 of the bf16
    rows the front left in ``qkvo`` times RN(1 / scale), bit for bit (the
    stock insert's and K1's rule alike)."""
    ks, vs = KV8_SCALES
    want = ref.kv_to_fp8(
        torch.cat(
            [
                qkvo[:, spec.ROW_K : spec.ROW_V].reshape(-1, spec.KV_HEADS, 64),
                qkvo[:, spec.ROW_V : spec.ROW_G].reshape(-1, spec.KV_HEADS, 64),
            ],
            -1,
        ),
        ks,
        vs,
    ).cpu()
    for t, sl in enumerate(slots.tolist()):
        if bool(pad[t]) or sl < 0:
            continue
        got = kv[sl // spec.KV_BLOCK, :, sl % spec.KV_BLOCK].cpu()
        assert torch.equal(got, want[t]), (t, sl)


# ---------------------------------------------------------------- [TP8]


def _check_chain(runner, w, kind, case, gated, split, s_front, step, kvs):
    """The 9.3 checks of one step (``gated``/``split``: the two runs'
    outputs; ``kvs``: their caches)."""
    lens, S = case
    dev = s_front.device
    sg, bg, qkvo_g, attn, _ = gated
    ss, bs, qkvo_s, xo = split
    pad = step.pad.cpu()
    live = ~pad
    lv = live.to(dev)
    md = step.attn_metadata[NAME[kind]]
    where = f"{kind} {lens} S={S}"
    tn._assert_healthy(runner)
    # the branch vLLM's experts give on the split launch's own ffn_in rows and
    # picks (read back before any later launch; collective)
    own = tnf._own_branch(runner, w, S)
    # the stock span (collective) from vLLM's gated core on the same rows, and
    # from the split core's own gated rows (diagnostic: stock's sensitivity)
    stock_core = _stock_gated(attn.clone(), qkvo_g)
    span: dict = {}
    tn._stock_span(w, stock_core, s_front, step.pad, span, None)
    span_xo: dict = {}
    tn._stock_span(w, xo, s_front, step.pad, span_xo, None)
    # Phase A's K2(tensor) on the split core's own gated rows: the same tail
    runner.step_begin()
    tensor = spec.K2Build(tokens=S)
    st, bt = _k2(runner, w, tensor, {"core": xo}, s_front, step.pad, kind)
    torch.accelerator.synchronize()
    tn._assert_healthy(runner)
    with tn._tp_checks(f"{where}: the split core"):
        assert torch.equal(qkvo_s[lv], qkvo_g[lv]), "the front is not repeatable"
        assert torch.equal(_bytes(kvs[0]), _bytes(kvs[1])), "the KV cache differs"
        # the gated rows: vs vLLM's gated core on the stock attention, and
        # never further from the fp64 gated attention than stock (E9)
        q = qkvo_g[:, :CORE].reshape(S, spec.HEADS, 64)
        kv_ref = kvs[1]
        if kv_ref.dtype == torch.uint8:
            kv_ref = ref.kv_from_fp8(kv_ref, *KV8_SCALES)
            _check_fp8_insert(qkvo_g, kvs[1], step.slot_mapping[NAME[kind]], pad)
        a64 = ref.gqa_decode(
            q, kv_ref, md.block_table, md.seq_lens, window=kind == SLIDING, pad=pad
        )
        g64 = gate64(a64, qkvo_g[:, spec.ROW_G :].cpu())[live.cpu()]
        got = xo.cpu().reshape(S, spec.HEADS, 64)[live]
        stock = stock_core.cpu().reshape(S, spec.HEADS, 64)[live]
        assert ref.cos(got, stock) > COS_CORE, (where, ref.cos(got, stock))
        e_got = (got.double() - g64).abs().max().item()
        e_stock = (stock.double() - g64).abs().max().item()
        ulp = 2.0 ** (math.floor(math.log2(g64.abs().max().item())) - 7)
        assert e_got <= e_stock + 2 * ulp, (where, e_got, e_stock)
        u = ref.row_ulps(got, stock)
        print(
            f"{where}: xo vs stock gate: max {u.max().item():.2f} row ulps, "
            f"<= 1 on {(u <= 1).float().mean().item():.5f}; fp64 err "
            f"{e_got:.3g} (stock {e_stock:.3g})",
            flush=True,
        )
        assert (xo[pad.to(xo.device)].float() == 0).all()
        # the layer's outputs. stream_out: vs the K2(gated) step and the stock
        # span. The tail after the core is Phase A's: bit for bit K2(tensor)
        # on the split core's own gated rows. branch_out: vs vLLM's experts on
        # its own ffn_in rows and picks (Phase A's reference, DESIGN_BC 9.3),
        # not vs the gated / stock chain: two cores that differ by an ulp on
        # ~40 % of the rows' elements (split vs paged_attention_v1, both held
        # above) take the stock span itself to branch cos 0.973 .. 0.995
        # (``stock span`` printed)
        assert ref.cos(ss[lv], sg[lv]) > tn.COS_STAGE, where
        assert ref.cos(ss[lv], span["stream"][lv]) > tn.COS_STAGE, where
        assert torch.equal(st, ss) and torch.equal(bt, bs), (where, "K2(tensor)")
        assert ref.cos(bs[lv], own[lv]) > tn.COS_CHAIN, where
        assert (ss[~lv] == 0).all() and (bs[~lv].float() == 0).all(), where
        sb = span["branch"][lv]
        print(
            f"{where}: branch cos vs own experts {ref.cos(bs[lv], own[lv]):.7f}; "
            f"whole chain vs K2(gated) {ref.cos(bs[lv], bg[lv]):.5f}, vs stock "
            f"span {ref.cos(bs[lv], span['branch'][lv]):.5f}; stock span on "
            f"xo vs on stock's core {ref.cos(span_xo['branch'][lv], sb):.5f}",
            flush=True,
        )
    tn._all_ranks_equal(ss, f"stream_out ({where})")
    tn._all_ranks_equal(bs, f"branch_out ({where})")


def _chain(
    front,
    kv8,
    monkeypatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # the first launch of each build compiles it: the startup budget
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    w = tn._weights(rank, device)
    fw = _front_weights(rank, device)
    mrv2 = _fake_mrv2(device)
    for ci, (lens, S) in enumerate(CASES):
        need = sum(-(-L // spec.KV_BLOCK) for L in lens)
        nblocks = need + 64
        pages = _pages(lens, nblocks, 100 + ci)
        step = mrv2.step(list(lens), {FULL: pages, SLIDING: pages}, S)
        branch, stream = _boundary_inputs(device, S, 50 + ci)
        for kind in KINDS:
            g = torch.Generator().manual_seed(1000 * ci + 7 * rank + LAYER[kind])
            kv = ref.kv_cache(nblocks, device=device, generator=g)
            if kv8:
                kv = ref.kv_to_fp8(kv, *KV8_SCALES)
            kvs = (kv, kv.clone())
            gated = _gated_step(
                front, runner, w, fw, kind, branch, stream, step, kvs[0]
            )
            ss, bs, qkvo_s = _split_step(
                front,
                runner,
                w,
                fw,
                kind,
                branch,
                stream,
                step,
                kvs[1],
            )
            torch.accelerator.synchronize()
            split = (ss, bs, qkvo_s, _xo(runner, S))
            _check_chain(runner, w, kind, (lens, S), gated, split, gated[4], step, kvs)
            md = step.attn_metadata[NAME[kind]]
            if len(lens) == S:
                continue
            # NaN in the pad rows' qkv rows: nothing else moves
            dirty = qkvo_s.clone()
            dirty[step.pad] = float("nan")
            key = _split_key(S, kind, kv8)
            runner.step_begin()
            again = _k2(
                runner, w, key, _split_args(dirty, kvs[1], md), gated[4], step.pad, kind
            )
            torch.accelerator.synchronize()
            tn._assert_healthy(runner)
            with tn._tp_checks(f"{kind} {lens}: NaN pad rows"):
                assert torch.equal(again[0], ss) and torch.equal(again[1], bs)


@ray.remote(num_gpus=1, max_calls=1)
def _chain_stock_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _chain("stock", False, monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@ray.remote(num_gpus=1, max_calls=1)
def _chain_k1_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _chain("k1", False, monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@ray.remote(num_gpus=1, max_calls=1)
def _chain_stock_kv8_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _chain("stock", True, monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@ray.remote(num_gpus=1, max_calls=1)
def _chain_k1_kv8_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _chain("k1", True, monkeypatch, tp_size, pp_size, rank, distributed_init_port)


_CHAIN_WORKERS = {
    ("stock", "bf16"): _chain_stock_worker,
    ("k1", "bf16"): _chain_k1_worker,
    ("stock", "fp8"): _chain_stock_kv8_worker,
    ("k1", "fp8"): _chain_k1_kv8_worker,
}


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize("kv", ["bf16", "fp8"])
@pytest.mark.parametrize("front", ["stock", "k1"])
def test_layer_chain_split(
    front: str, kv: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _CHAIN_WORKERS[front, kv])


def _replay(
    front,
    monkeypatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    w = tn._weights(rank, device)
    fw = _front_weights(rank, device)
    sched = _schedule()
    S = len(REPLAY_START)
    need = sum(-(-L // spec.KV_BLOCK) for L in sched[-1])
    nblocks = need + 64
    pages = _pages(sched[-1], nblocks, 7)
    mrv2 = _fake_mrv2(device)
    step = mrv2.step(list(sched[0]), {FULL: pages, SLIDING: pages}, S)
    g = torch.Generator().manual_seed(77 + rank)
    kv = {k: ref.kv_cache(nblocks, device=device, generator=g) for k in KINDS}
    kv_g = {k: t.clone() for k, t in kv.items()}
    branch, stream = _boundary_inputs(device, S, 9)
    cap = MAX_BLOCKS * spec.KV_BLOCK

    def chain():
        """Two kernel layers: the full one, then the sliding one on its
        deferred boundary (one step_begin)."""
        s1, b1, _ = _split_step(
            front,
            runner,
            w,
            fw,
            FULL,
            branch,
            stream,
            step,
            kv[FULL],
        )
        s2, b2, _ = _split_step(
            front,
            runner,
            w,
            fw,
            SLIDING,
            b1,
            s1,
            step,
            kv[SLIDING],
            begin=False,
        )
        return [s1, b1, s2, b2]

    def gated():
        """The same two layers on the stock core and K2(gated). The stock
        core's max length is the table's (``cap``), a host constant: a sliding
        layer's ``_build_swa_decode_tables`` trims the table and the lengths
        on the device and caps the max length at the window."""
        s1, b1 = _gated_step(
            front, runner, w, fw, FULL, branch, stream, step, kv_g[FULL], cap
        )[:2]
        s2, b2 = _gated_step(
            front,
            runner,
            w,
            fw,
            SLIDING,
            b1,
            s1,
            step,
            kv_g[SLIDING],
            cap,
            begin=False,
        )[:2]
        return [s1, b1, s2, b2]

    chain()  # warm-ups: compile, load, JIT
    gated()
    torch.accelerator.synchronize()
    graphs, outs = {}, {}
    for name, fn in (("split", chain), ("gated", gated)):
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            outs[name] = fn()
    torch.accelerator.synchronize()
    for i, lens in enumerate(sched):
        mrv2.step(list(lens), {FULL: pages, SLIDING: pages}, S)  # in place
        for name, fn in (("split", chain), ("gated", gated)):
            graphs[name].replay()
            torch.accelerator.synchronize()
            got = [t.clone() for t in outs[name]]
            want = fn()  # eager: the same step's K / V rewritten in place
            torch.accelerator.synchronize()
            with tn._tp_checks(f"replay {i} ({name}, lens {lens})"):
                for a, b in zip(got, want):
                    assert torch.equal(a, b), (name, i, lens)
                    assert torch.isfinite(a.float()).all(), (name, i)
    tn._assert_healthy(runner)


@ray.remote(num_gpus=1, max_calls=1)
def _replay_stock_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _replay("stock", monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@ray.remote(num_gpus=1, max_calls=1)
def _replay_k1_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _replay("k1", monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize("front", ["stock", "k1"])
def test_graph_replay_follows_metadata(
    front: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _replay_k1_worker if front == "k1" else _replay_stock_worker
    multi_process_parallel(monkeypatch, 8, 1, worker)
