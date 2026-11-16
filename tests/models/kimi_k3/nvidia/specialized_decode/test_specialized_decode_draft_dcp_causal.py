# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The causal DSpark draft block on the DCP mailbox path (`DraftDcp`) decoded as the
stock causal MLA DCP decode (`drafter.attention.dcp_causal_native_decode`,
"native") against the context fold it replaces at DCP 16 (`folded_decode` over the
context keys, then `dcp_causal_block_merge`, "fold"), for every rank of a
simulated interleave-1 DCP 16 world on one GPU:

- each rank against an exact FP64 reference over the keys it holds that a block
  token may see (global position <= L + j): output, natural-log LSE and the rows
  with no visible key (LSE -inf);
- the ranks' partials merged by LSE (exp(lse - max) weights, as `DcpValue`)
  against the exact causal attention over all keys.

The native output error stays within 1.05x the fold path's for contexts of 32K keys
or more. With only a few context keys the block keys dominate, and the stock decode
rounds their P to FP8 where the fold's merge keeps FP32, so those cases check the
LSE, the empty rows and finiteness only.

The decode policy and `DraftDcp._forward`'s wiring of each decode run without a
GPU."""

import math
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.platforms import current_platform

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="cute-dsl MLA DCP decode needs SM100",
)

KV_LORA, ROPE, NOPE = 512, 64, 128
D = KV_LORA + ROPE
PAGE = 64
SCALE = 192**-0.5
WORLD = 16
HEADS = 96  # the gathered draft heads at DCP 16 (6 per rank)


def _cdiv(a, b):
    return -(-a // b)


CAP_LOCAL = _cdiv(600_000, WORLD)  # a captured graph's local max_seq_len
WIDTH = _cdiv(_cdiv(CAP_LOCAL, PAGE), 2) * 2  # its block-table width (even)
LONG = 32768
RATIO = 1.05
LSE_TOL = 2e-6
PAD = -1  # a zero-length padded request (as in a graph padded past the live ones)


def _attention():
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter import attention

    return attention


def _global(ctxs, k, seed):
    """Every request's keys (context + block, FP8) and the block's queries."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = [
        (torch.randn(c + k, D, device="cuda", generator=g) * 0.5).to(
            torch.float8_e4m3fn
        )
        if c != PAD
        else None
        for c in ctxs
    ]
    q = (torch.randn(len(ctxs) * k, HEADS, D, device="cuda", generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    return keys, q


def _shard(keys, rank, seed):
    """Rank `rank`'s interleave-1 paged KV (global key p on rank p % WORLD at local
    p // WORLD) on randomly permuted pages, unused pages filled with junk. A padded
    request has length 0 and request 0's block-table row. Returns (kv, block table,
    local lengths, global lengths)."""
    g = torch.Generator(device="cpu").manual_seed(seed * 31 + rank)
    local = [x[rank::WORLD] if x is not None else None for x in keys]
    pages = [_cdiv(x.shape[0], PAGE) if x is not None else 0 for x in local]
    total = sum(pages) + 8
    perm = torch.randperm(total, generator=g).tolist()
    kv = (torch.randn(total, PAGE, D, device="cuda") * 3).to(torch.float8_e4m3fn)
    bt = torch.zeros(len(local), WIDTH, dtype=torch.int32)
    lens, tots, used = [], [], 0
    for r, (x, n) in enumerate(zip(local, pages)):
        if x is None:
            bt[r] = bt[0]
            lens.append(0)
            tots.append(0)
            continue
        ids = torch.tensor(perm[used : used + n], dtype=torch.long)
        used += n
        if n:  # a rank may hold none of a short request's keys
            flat = torch.zeros(n * PAGE, D, device="cuda", dtype=torch.float8_e4m3fn)
            flat[: x.shape[0]] = x
            kv.view(torch.uint8)[ids.cuda()] = flat.view(torch.uint8).view(n, PAGE, D)
            bt[r, :n] = ids.to(torch.int32)
        lens.append(x.shape[0])
        tots.append(keys[r].shape[0])

    def i32(v):
        return torch.tensor(v, dtype=torch.int32, device="cuda")

    return kv, bt.cuda(), i32(lens), i32(tots)


def _empty_rows():
    return (
        torch.zeros(HEADS, KV_LORA, device="cuda", dtype=torch.float64),
        torch.full((HEADS,), -math.inf, device="cuda", dtype=torch.float64),
    )


def _reference_rank(keys, q, k, rank):
    """Exact FP64 output [R * k, H, 512] and LSE [R * k, H] over this rank's keys."""
    outs, lses = [], []
    for r, kk in enumerate(keys):
        if kk is None:
            for o, lse in [_empty_rows()] * k:
                outs.append(o)
                lses.append(lse)
            continue
        ctx = kk.shape[0] - k
        # Global positions of this rank's keys (none when rank >= the key count).
        pos = torch.arange(rank, max(rank, kk.shape[0]), WORLD, device="cuda")
        rows = kk[rank::WORLD].double()
        for j in range(k):
            sel = pos <= ctx + j
            if not bool(sel.any()):
                o, lse = _empty_rows()
                outs.append(o)
                lses.append(lse)
                continue
            kj = rows[sel]
            s = (q[r * k + j].double() @ kj.T) * SCALE
            lse = torch.logsumexp(s, -1)
            outs.append(torch.exp(s - lse[:, None]) @ kj[:, :KV_LORA])
            lses.append(lse)
    return torch.stack(outs), torch.stack(lses)


def _reference_full(kk, q, k, r):
    """Exact FP64 causal attention of request `r`'s block over all its keys."""
    ctx = kk.shape[0] - k
    rows = kk.double()
    outs = []
    for j in range(k):
        s = (q[r * k + j].double() @ rows[: ctx + j + 1].T) * SCALE
        outs.append(torch.softmax(s, -1) @ rows[: ctx + j + 1, :KV_LORA])
    return torch.stack(outs)


def _fold(mod, q, kv, bt, tot, k, rank, ws):
    """The fold path as `DraftDcp` runs it for a causal block."""
    reqs = q.shape[0] // k
    ctx_tot = (tot[:reqs] - k).to(torch.int32)
    ctx_local = torch.clamp((ctx_tot - rank + WORLD - 1) // WORLD, min=0).to(
        torch.int32
    )
    o, lse = mod.folded_decode(
        q,
        kv,
        bt,
        ctx_local,
        CAP_LOCAL,
        k,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        workspace=ws,
        return_lse=True,
        dcp_world_size=WORLD,
        dcp_rank=rank,
        dcp_tot_seq_lens=ctx_tot,
        table_cache=None,
    )
    return mod.dcp_causal_block_merge(
        q, kv, bt, ctx_tot, o, lse, k, WORLD, rank, SCALE, 1.0
    )


def _native(mod, q, kv, bt, lens, tot, k, rank, ws):
    return mod.dcp_causal_native_decode(
        q,
        kv,
        bt,
        lens,
        tot,
        CAP_LOCAL,
        k,
        WORLD,
        rank,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        workspace=ws,
    )


def _rel(a, b):
    den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else 0.0


def _merge(parts_o, parts_lse):
    """The ranks' partials merged by LSE, as `DcpValue` weights them."""
    lse = torch.stack(parts_lse)  # [W, rows, H]
    m = lse.max(0).values
    w = torch.where(torch.isfinite(lse), torch.exp(lse - m), torch.zeros_like(lse))
    o = torch.stack(parts_o)
    o = torch.where(w[..., None] > 0, o, torch.zeros_like(o))
    return (o * w[..., None]).sum(0) / w.sum(0)[..., None]


CASES = [
    # One request. 32780 and 524300: the block wraps from rank 15 to rank 0.
    *[(f"one_{c}", k, [c]) for k in (7, 4) for c in (32805, 32780, 524300)],
    # The c = 1 draft graph: one live request and a zero-length padded one.
    *[("padded", k, [32780, PAD]) for k in (7, 4)],
    # A uniform block over mixed contexts; the 5-key request's rows are empty on
    # most ranks.
    *[("mixed", k, [70012, 32805, 5]) for k in (7, 4)],
    # Few context keys: empty rows on most ranks (LSE / empty rows / finite only).
    ("short", 7, [1, 3, 8, 15, 16, 21]),
]


@requires_sm100
@pytest.mark.parametrize("name,k,ctxs", CASES, ids=[f"{c[0]}-k{c[1]}" for c in CASES])
def test_native_matches_fold_and_exact(name, k, ctxs):
    mod = _attention()
    seed = 1000 * k + sum(ctxs) % 997
    keys, q = _global(ctxs, k, seed)
    ws = torch.zeros(256 << 20, dtype=torch.int8, device="cuda")
    reqs = len(ctxs)
    long_reqs = [r for r, c in enumerate(ctxs) if c >= LONG]
    parts: dict[str, tuple[list, list]] = {"fold": ([], []), "native": ([], [])}
    worst_ratio, worst_lse = 0.0, 0.0
    for rank in range(WORLD):
        kv, bt, lens, tot = _shard(keys, rank, seed)
        ref_o, ref_lse = _reference_rank(keys, q, k, rank)
        ref_empty = ~torch.isfinite(ref_lse)
        live = ~ref_empty
        got = {
            "fold": _fold(mod, q, kv, bt, tot, k, rank, ws),
            "native": _native(mod, q, kv, bt, lens, tot, k, rank, ws),
        }
        rel: dict[str, list[float]] = {}
        for path, (o, lse) in got.items():
            o64 = o.double().reshape(ref_o.shape)
            l64 = lse.double().reshape(ref_lse.shape)
            empty = ~torch.isfinite(l64)
            if path == "native":
                assert torch.equal(empty, ref_empty), (name, rank)
                assert bool(torch.isfinite(o64).all()), (name, rank)
                if bool(live.any()):
                    err = (l64 - ref_lse)[live].abs().max().item()
                    worst_lse = max(worst_lse, err)
            rel[path] = []
            for r in range(reqs):
                rows = slice(r * k, (r + 1) * k)
                lv = live[rows]
                rel[path].append(
                    _rel(o64[rows][lv], ref_o[rows][lv]) if bool(lv.any()) else 0.0
                )
            parts[path][0].append(torch.where(empty[..., None], 0.0, o64))
            parts[path][1].append(l64)
        for r in long_reqs:
            if rel["fold"][r] > 0:
                worst_ratio = max(worst_ratio, rel["native"][r] / rel["fold"][r])
        del kv, bt
    assert worst_lse <= LSE_TOL, (name, worst_lse)
    assert worst_ratio <= RATIO, (name, worst_ratio)
    merged = {path: _merge(*parts[path]) for path in parts}
    for r in long_reqs:
        rows = slice(r * k, (r + 1) * k)
        ref = _reference_full(keys[r], q, k, r)
        e_fold, e_native = (_rel(merged[p][rows], ref) for p in ("fold", "native"))
        assert e_native <= RATIO * e_fold, (name, r, e_native, e_fold)
    for r, kk in enumerate(keys):
        if kk is not None:
            rows = slice(r * k, (r + 1) * k)
            assert bool(torch.isfinite(merged["native"][rows]).all()), (name, r)


@requires_sm100
@pytest.mark.parametrize("k,reqs", [(7, 2), (4, 2), (7, 5)])
def test_native_graph_replays_new_lengths(k, reqs):
    """Captured at one context and replayed at another (both ways), with one live
    request plus padding or all live: bit-identical to eager at the replay lengths,
    finite live rows, LSE -inf on every padded row."""
    mod = _attention()
    rank = 3
    ctx_a, ctx_b = 32805, 524325
    g = torch.Generator(device="cuda").manual_seed(11 * k + reqs)
    pages = _cdiv(_cdiv(ctx_b + k - rank, WORLD), PAGE)
    kv = (torch.randn(reqs * pages + 1, PAGE, D, device="cuda", generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    bt = torch.zeros(8, WIDTH, dtype=torch.int32, device="cuda")
    for r in range(reqs):
        bt[r, :pages] = torch.arange(
            1 + r * pages, 1 + (r + 1) * pages, dtype=torch.int32, device="cuda"
        )
    q = (torch.randn(reqs * k, HEADS, D, device="cuda", generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    lens = torch.zeros(8, dtype=torch.int32, device="cuda")
    tot = torch.zeros(8, dtype=torch.int32, device="cuda")
    ws = torch.zeros(256 << 20, dtype=torch.int8, device="cuda")

    def set_lens(ctx, live):
        lens.zero_()
        tot.zero_()
        lens[:live] = _cdiv(ctx + k - rank, WORLD)
        tot[:live] = ctx + k

    def run():
        return _native(mod, q, kv, bt, lens, tot, k, rank, ws)

    for capture_ctx, replay_ctx in ((ctx_a, ctx_b), (ctx_b, ctx_a)):
        for live in sorted({1, reqs}):
            set_lens(capture_ctx, live)
            run()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                go, gl = run()
            set_lens(replay_ctx, live)
            graph.replay()
            eo, el = run()
            torch.accelerator.synchronize()
            assert torch.equal(go.view(torch.int16), eo.view(torch.int16))
            assert torch.equal(gl.view(torch.int32), el.view(torch.int32))
            assert bool(torch.isfinite(gl[: live * k]).all())
            assert bool((gl[live * k :] == -math.inf).all())
            del graph


def test_causal_dcp_decode_policy():
    """Every row count has one decode; the c = 1 draft graphs at DCP 16 (two
    requests, one of them padding: 14 rows at k = 7, 8 at k = 4) take the stock
    decode, and DCP sizes without an entry keep the fold."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import (
        batch_table,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.attention import (
        CAUSAL_DCP_DECODE,
        causal_dcp_decode,
    )

    for world, spec in CAUSAL_DCP_DECODE.items():
        assert set(batch_table(spec).values()) <= {"native", "fold"}, world
    for rows in (7, 14, 4, 8):
        assert causal_dcp_decode(16, rows) == "native"
    for world in (1, 2, 4, 8, 32):
        if world not in CAUSAL_DCP_DECODE:
            assert all(causal_dcp_decode(world, n) == "fold" for n in range(1, 321))
    assert causal_dcp_decode(16, 320) == "fold"


class _Decoded(Exception):
    """Raised by the decode spies: `_forward` stops at its decode call."""

    def __init__(self, path, args, kwargs):
        super().__init__(path)
        self.path, self.args, self.kwargs = path, args, kwargs


def _spy(path):
    def run(*args, **kwargs):
        raise _Decoded(path, args, kwargs)

    return run


@pytest.mark.parametrize("policy", ["native", "fold", "non_causal"])
def test_forward_wires_the_block_decode(policy, monkeypatch):
    """`DraftDcp._forward` up to its decode, on CPU with stand-ins for the query
    projection, the q-mailbox gather and the kernels: a causal block at a DCP size
    whose `CAUSAL_DCP_DECODE` picks "native" calls `dcp_causal_native_decode`
    with the gathered query, this rank's local lengths, the global lengths (block
    included) and the step's max_seq_len; "fold" (no entry) folds the context
    (global L = total - k, this rank's share of it) as `_fold` above; a non-causal
    block keeps the folded decode over the local and global lengths."""
    mod = _attention()
    world, rank, heads, k, reqs, cap = 2, 1, 2, 3, 2, 8
    n = reqs * k
    q_lora, kv_lora, rope, nope = 8, 4, 2, 3
    monkeypatch.setitem(
        mod.CAUSAL_DCP_DECODE, world, "1-35:native" if policy == "native" else ""
    )
    for name in ("dcp_causal_native_decode", "folded_decode"):
        monkeypatch.setattr(mod, name, _spy(name))
    monkeypatch.setattr(mod, "_scales", lambda impl, layer: (0.25, 2.0))
    workspace = torch.zeros(4, dtype=torch.int8)
    gathers = []
    stand_ins = {
        "vllm.models.common.ops": NS(fused_q_kv_rmsnorm=lambda q, kv, *a: (q, kv)),
        mod.__name__.rsplit(".", 2)[0] + ".cutedsl.common.host": NS(
            stream=None, to_cute=None
        ),
        mod.__name__.rsplit(".", 1)[0] + ".comm": NS(
            Q_SENT=0,
            compile_gather=lambda *a: lambda *args: gathers.append(args),
            gather_ctas=lambda *a: 1,
        ),
        "vllm.v1.attention.backends.mla.flashinfer_mla": NS(
            _get_workspace_buffer=lambda return_lse: workspace
        ),
    }
    for name, module in stand_ins.items():
        monkeypatch.setitem(sys.modules, name, module)

    def proj(width):
        return lambda x: (torch.zeros(x.shape[0], width),)

    kv_cache = torch.zeros(3, 64, kv_lora + rope, dtype=torch.uint8)
    attn = NS(
        layer_name="draft.attn",
        fused_qkv_a_proj=proj(q_lora + kv_lora + rope),
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        qk_rope_head_dim=rope,
        qk_nope_head_dim=nope,
        qk_head_dim=nope + rope,
        v_head_dim=4,
        num_local_heads=heads,
        q_a_layernorm=NS(weight=torch.ones(1)),
        kv_a_layernorm=NS(weight=torch.ones(1)),
        rms_norm_eps=1e-6,
        q_b_proj=proj(heads * (nope + rope)),
        rotary_emb=None,
        W_UK_T=torch.zeros(heads, nope, kv_lora),
        _decode_concat_cache=lambda *a: torch.zeros(n, heads, 576, dtype=torch.uint8),
        _attn_read_kv_cache=lambda: kv_cache,
        impl=NS(),
    )
    boxes = [torch.zeros(cap, world * heads, 576, dtype=torch.uint8) for _ in range(2)]
    dcp = object.__new__(mod.DraftDcp)
    dcp.size, dcp.rank, dcp.cap, dcp.device = world, rank, cap, "cpu"
    dcp.box = NS(q=boxes, q_mc=0)
    tot = torch.tensor([70012, 32805], dtype=torch.int32)
    decode = NS(
        block_table=torch.zeros(reqs, 4, dtype=torch.int32),
        seq_lens=(tot + world - 1 - rank) // world,
        dcp_tot_seq_lens=tot,
    )
    metadata = NS(causal=policy != "non_causal", decode=decode, max_seq_len=35006)
    context = NS(slot_mapping={attn.layer_name: torch.zeros(n, dtype=torch.int64)})
    parity = 1
    with pytest.raises(_Decoded) as called:
        dcp._forward(
            attn, torch.arange(n), torch.zeros(n, 16), metadata, context, n, k, parity
        )
    got = called.value
    assert len(gathers) == 1  # the query gathered into the parity's mailbox first
    query = got.args[0]
    assert query.dtype == torch.float8_e4m3fn
    assert query.shape == (n, world * heads, 576)
    assert query.data_ptr() == boxes[parity].data_ptr()
    assert got.args[1] is kv_cache and got.args[2] is decode.block_table
    common = dict(
        kv_lora_rank=kv_lora,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        bmm1_scale=0.25,
        bmm2_scale=2.0,
        workspace=workspace,
    )
    for key, value in common.items():
        assert got.kwargs[key] is value or got.kwargs[key] == value, key
    if policy == "native":
        assert got.path == "dcp_causal_native_decode"
        assert got.args[3] is decode.seq_lens
        assert got.args[4] is decode.dcp_tot_seq_lens
        assert got.args[5:] == (metadata.max_seq_len, k, world, rank)
        return
    assert got.path == "folded_decode"
    assert got.args[4:] == (metadata.max_seq_len, k)
    assert got.kwargs["return_lse"] is True
    assert (got.kwargs["dcp_world_size"], got.kwargs["dcp_rank"]) == (world, rank)
    if policy == "fold":
        ctx_tot = tot - k
        ctx_local = torch.clamp((ctx_tot - rank + world - 1) // world, min=0)
        assert torch.equal(got.args[3], ctx_local.int())
        assert torch.equal(got.kwargs["dcp_tot_seq_lens"], ctx_tot.int())
        assert got.kwargs["table_cache"] is None
    else:
        assert got.args[3] is decode.seq_lens
        assert got.kwargs["dcp_tot_seq_lens"] is decode.dcp_tot_seq_lens
