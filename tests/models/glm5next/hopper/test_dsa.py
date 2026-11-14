# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Hopper DSA layer (`hopper/dsa.py`) on synthetic TP4 rank-0 shapes:

- the MLA KV insert and the q a-norm are the bits of stock's
  `fused_q_kv_rmsnorm` + `concat_and_cache_mla` on the same a-projection;
- the tail ring and indexer cache entries are the bytes of stock's
  `kpool_decode_update_and_maybe_write_cache_batched` on the same key and gate;
- the projections are no less accurate than an FP32 GEMM against FP64;
- the pool logits match an FP64 reference built on stock's query quant, and the
  top-512 pools match its exact top-k except at near-ties, for short contexts
  (every pool), contexts above the top-k, and long ones (pass-based select);
- the sparse MLA, W_UV and o_proj match FP64 on the selected rows;
- padded rows write nothing; CUDA graph replays follow new inputs; draft steps
  that reuse the logical top-k attend to the same rows.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(90), reason="Requires SM90"
)

HEADS, NOPE, LATENT, VALUE, HIDDEN, Q_LORA = 16, 256, 512, 256, 4096, 1536
IHEADS, IDIM, KPOOL, RING = 32, 128, 4, 16
BLOCK, PAGE = 256, 64


def _bf16(t):
    return t.to(torch.bfloat16)


def _weights(g, max_pools):
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.dsa import DsaWeights

    def proj(n, k):
        return _bf16(torch.randn(n, k, generator=g, device="cuda") * k**-0.5)

    kv_b = proj(HEADS * (NOPE + VALUE), LATENT)
    w_uk, w_uv = kv_b.T.view(LATENT, HEADS, NOPE + VALUE).split([NOPE, VALUE], -1)
    return DsaWeights(
        qkv_a=proj(Q_LORA + LATENT, HIDDEN),
        q_b=proj(HEADS * NOPE, Q_LORA),
        o_proj=proj(HIDDEN, HEADS * VALUE),
        wq_b=proj(IHEADS * IDIM, Q_LORA),
        wk_weights=proj(IDIM + IHEADS, HIDDEN),
        gate=proj(IDIM, HIDDEN),
        ape=torch.randn(KPOOL, IDIM, generator=g, device="cuda") * 0.5,
        k_norm_weight=1 + 0.1 * torch.randn(IDIM, generator=g, device="cuda"),
        k_norm_bias=0.1 * torch.randn(IDIM, generator=g, device="cuda"),
        k_norm_eps=1e-6,
        q_norm=_bf16(1 + 0.1 * torch.randn(Q_LORA, generator=g, device="cuda")),
        kv_norm=_bf16(1 + 0.1 * torch.randn(LATENT, generator=g, device="cuda")),
        rms_eps=1e-5,
        w_uk_t=w_uk.permute(1, 2, 0),  # [N, P, L], L contiguous
        w_uv=w_uv.transpose(0, 1),  # [N, L, V], L contiguous
        softmax_scale=NOPE**-0.5,
        index_scale=IDIM**-0.5 * IHEADS**-0.5,
        max_pools=max_pools,
    )


class Step:
    """Synthetic caches and metadata of one decode step: request r has
    `contexts[r]` earlier tokens and verifies q more."""

    def __init__(self, g, contexts, q, padded=0):
        from vllm.models.glm5next.nvidia.specialized_decode.hopper.dsa import (
            DsaCaches,
            DsaInputs,
        )

        dev = "cuda"
        reqs = len(contexts)
        self.q, self.contexts = q, contexts
        longest = max(contexts) + q
        blocks = math.ceil(longest / BLOCK)
        pages = math.ceil(longest / KPOOL / PAGE)
        # Padded block stride, as the hybrid allocator lays blocks out.
        kv = torch.randn(reqs * blocks + 1, BLOCK + 8, LATENT, generator=g, device=dev)
        kv = _bf16(kv)[:, :BLOCK]
        values = torch.randn(reqs * pages + 1, PAGE * IDIM, generator=g, device=dev)
        values = (values * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
        exps = torch.randint(-12, -4, (reqs * pages + 1, PAGE), generator=g, device=dev)
        scales = torch.exp2(exps.float())
        index = torch.cat([values.view(torch.uint8), scales.view(torch.uint8)], 1)
        index = index.view(reqs * pages + 1, PAGE, IDIM + 4).contiguous()
        tail = _bf16(torch.randn(reqs + 1, 2, RING, IDIM, generator=g, device=dev))
        self.caches = DsaCaches(kv, index, tail)
        # Block 0 / page 0 stay unused so a wrong table read shows.
        cpu = torch.Generator().manual_seed(int(contexts[0]) + q)
        bt = 1 + torch.randperm(reqs * blocks, generator=cpu).view(reqs, blocks)
        it = 1 + torch.randperm(reqs * pages, generator=cpu).view(reqs, pages)
        positions, slots, islots, tslots, req_ids = [], [], [], [], []
        for r, ctx in enumerate(contexts):
            for t in range(q):
                p = ctx + t
                live = r < reqs - padded
                positions.append(p if live else 0)
                req_ids.append(r)
                slots.append(int(bt[r, p // BLOCK]) * BLOCK + p % BLOCK if live else -1)
                pool = p // KPOOL
                full = p % KPOOL == KPOOL - 1
                islots.append(
                    int(it[r, pool // PAGE]) * PAGE + pool % PAGE
                    if live and full
                    else -1
                )
                tslots.append((r + 1) * RING + p % RING if live else -1)

        def i64(v):
            return torch.tensor(v, dtype=torch.int64, device=dev)

        self.inputs = DsaInputs(
            positions=i64(positions),
            kv_slots=i64(slots),
            req_ids=torch.tensor(req_ids, dtype=torch.int32, device=dev),
            block_table=bt.to(dev, torch.int32),
            block_size=BLOCK,
            index_slots=i64(islots),
            index_block_table=it.to(dev, torch.int32),
            tail_slots=i64(tslots),
        )
        self.tokens = reqs * q
        self.x = _bf16(torch.randn(self.tokens, HIDDEN, generator=g, device=dev))

    def clone_caches(self):
        from vllm.models.glm5next.nvidia.specialized_decode.hopper.dsa import DsaCaches

        c = self.caches
        return DsaCaches(c.kv.clone(), c.index.clone(), c.tail.clone())


def _layer(weights, tokens, q, topk_buffer=None):
    from vllm.models.glm5next.nvidia.specialized_decode.hopper.dsa import (
        HopperDsaLayer,
    )

    return HopperDsaLayer.from_weights(weights, [tokens], q, topk_buffer=topk_buffer)


def _ws(layer, m):
    from vllm.models.glm5next.nvidia.specialized_decode.hopper import dsa

    ws = layer.ws
    return dict(
        y=ws.y[:m].clone(), qc=ws.qc[:m].clone(), q=ws.q[:m].clone(),
        qabs=ws.qabs[:m].clone(), ob=ws.ob[:m].clone(), ao=ws.ao[:m].clone(),
        rows=ws.rows[:m].clone(), counts=ws.counts[:m].clone(),
        logits=ws.logits[:m].clone(), gate=dsa.GATE, key=dsa.KEY, wproj=dsa.WPROJ,
    )  # fmt: skip


def _run(contexts, q, seed=0, padded=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    max_pools = (max(contexts) + q) // KPOOL + 64
    weights = _weights(g, max_pools)
    step = Step(g, contexts, q, padded)
    before = step.clone_caches()
    layer = _layer(weights, step.tokens, q)
    out = layer.forward(step.x, step.inputs, step.caches, q)
    torch.cuda.synchronize()
    return weights, step, before, out, _ws(layer, step.tokens)


CASES = [
    ([1000], 6),  # short: every pool
    ([2045, 2046], 6),  # around the top-k
    ([2400, 3000, 2200], 6),
    ([2048, 2600, 3100, 4000], 6),
    ([2400] * 4, 1),
    ([9000, 33000], 6),  # pass-based select
    ([33000], 1),
]


@pytest.mark.parametrize("contexts,q", CASES[:4])
def test_kv_insert_and_q_norm_match_stock(contexts, q):
    from vllm import _custom_ops as ops
    from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm

    w, step, before, _, ws = _run(contexts, q)
    y = _bf16(ws["y"])
    qc, kvn = fused_q_kv_rmsnorm(
        y[:, :Q_LORA], y[:, Q_LORA : Q_LORA + LATENT], w.q_norm, w.kv_norm, w.rms_eps
    )
    assert torch.equal(ws["qc"], qc)
    kv = before.kv.clone()
    ops.concat_and_cache_mla(
        kvn, kvn.new_empty(kvn.shape[0], 0), kv, step.inputs.kv_slots, "auto",
        torch.ones(1, device="cuda"),
    )  # fmt: skip
    assert torch.equal(step.caches.kv.view(torch.int16), kv.view(torch.int16))


@pytest.mark.parametrize("contexts,q", CASES[:5])
def test_kpool_matches_stock(contexts, q):
    """Tail stash and pool compression: stock's bytes on the same key and gate
    (the key LayerNorm of both sides is eager PyTorch's; where its BF16 output
    rounds differently from the fused one the entry may differ)."""
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        kpool_decode_update_and_maybe_write_cache_batched,
    )

    w, step, before, _, ws = _run(contexts, q)
    y = _bf16(ws["y"])
    key_ref = F.layer_norm(
        y[:, ws["key"]:ws["key"] + IDIM].float(), (IDIM,), w.k_norm_weight,
        w.k_norm_bias, w.k_norm_eps,
    ).bfloat16()  # fmt: skip
    gate = y[:, ws["gate"] : ws["gate"] + IDIM]
    tail, index = before.tail.clone(), before.index.clone()
    shape = (len(contexts), q)
    inp = step.inputs
    kpool_decode_update_and_maybe_write_cache_batched(
        index, tail, inp.tail_slots.view(shape), key_ref.view(*shape, IDIM),
        gate.reshape(*shape, IDIM), w.ape, inp.index_slots.view(shape),
        inp.positions.view(shape), KPOOL,
    )  # fmt: skip
    ring = step.caches.tail.view(torch.int16)
    same_ring = (ring == tail.view(torch.int16)).float().mean().item()
    same_index = (step.caches.index == index).float().mean().item()
    assert same_ring > 0.999 and same_index > 0.9999, (same_ring, same_index)


def _gemm_error_ok(ours, x, weight):
    exact = x.double() @ weight.double().T
    fp32 = x.float() @ weight.float().T
    mine = (ours.double() - exact).abs().max().item()
    ref = (fp32.double() - exact).abs().max().item()
    return mine <= 4 * ref + 1e-5, (mine, ref)


def test_projections_vs_fp64():
    torch.backends.cuda.matmul.allow_tf32 = False
    w, step, _, _, ws = _run([2400, 2600], 6)
    w1 = torch.cat([w.qkv_a, w.gate, w.wk_weights])
    ok, errs = _gemm_error_ok(ws["y"], step.x, w1)
    assert ok, errs
    w2 = torch.cat([w.q_b, w.wq_b])
    exact = ws["qc"].double() @ w2.double().T
    err = (ws["q"].double() - exact).abs() / exact.abs().clamp_min(1e-3)
    assert err.max().item() < 2**-7, err.max().item()  # one BF16 rounding


@pytest.mark.parametrize("contexts", [[2400], [2400, 2600]])
def test_absorb_vs_fp64(contexts):
    """The W_UK absorb (main's CuTe head GEMV at up to 6 tokens, Triton above)."""
    w, step, _, _, ws = _run(contexts, 6)
    qn = ws["q"][:, : HEADS * NOPE].view(-1, HEADS, NOPE).double()
    exact = torch.einsum("thp,hpl->thl", qn, w.w_uk_t.double())
    err = (ws["qabs"].double() - exact).abs() / exact.abs().clamp_min(1e-2)
    assert err.max().item() < 2**-7, err.max().item()


def _index_logits(w, step, ws):
    """FP64 pool logits per row from stock's query quant of our indexer query."""
    from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8

    iq = ws["q"][:, HEADS * NOPE :].contiguous()
    q_fp8, q_scale = fwht128_quant_fp8(iq.view(-1, IDIM))
    q8 = q_fp8.view(-1, IHEADS, IDIM).double()
    weight = ws["y"][:, ws["wproj"] : ws["wproj"] + IHEADS] * q_scale.view(-1, IHEADS)
    weight = (weight * w.index_scale).double()
    index = step.caches.index
    pages = index.shape[0]
    flat = index.view(pages, -1)
    values = flat[:, : PAGE * IDIM].contiguous().view(torch.float8_e4m3fn)
    values = values.view(pages, PAGE, IDIM).double()
    scales = flat[:, PAGE * IDIM :].contiguous().view(torch.float32)
    out = []
    inp = step.inputs
    for row in range(step.tokens):
        if int(inp.kv_slots[row]) < 0:
            out.append(None)
            continue
        r = int(inp.req_ids[row])
        npool = (int(inp.positions[row]) + 1) // KPOOL
        pool = torch.arange(npool, device="cuda")
        page = inp.index_block_table[r, pool // PAGE].long()
        k = values[page, pool % PAGE]
        s = (q8[row] @ k.T).clamp_min(0) * scales[page, pool % PAGE].double()
        out.append((weight[row] @ s, npool))
    return out


@pytest.mark.parametrize("contexts,q", CASES)
def test_select_matches_exact_topk(contexts, q):
    w, step, _, _, ws = _run(contexts, q)
    inp = step.inputs
    stride = step.caches.kv.stride(0) // LATENT
    for row, ref in enumerate(_index_logits(w, step, ws)):
        count = int(ws["counts"][row])
        if ref is None:
            assert count == 0
            continue
        logits, npool = ref
        r = int(inp.req_ids[row])
        ctx = int(inp.positions[row]) + 1
        if npool > 512:
            got = ws["logits"][row, :npool].double()
            atol = 1e-6 * float(logits.abs().max())
            assert torch.allclose(got, logits, rtol=1e-4, atol=atol), row
            order = torch.argsort(-logits, stable=True)
            pools = order[:512]
            threshold = logits[order[511]]
            near = (logits - threshold).abs() <= 1e-4 * threshold.abs() + 1e-9
        else:
            pools = torch.arange(npool, device="cuda")
            near = torch.zeros(npool, dtype=torch.bool, device="cuda")
        tokens = (pools[:, None] * KPOOL + torch.arange(KPOOL, device="cuda")).view(-1)
        tokens = torch.cat([tokens, torch.arange(npool * KPOOL, ctx, device="cuda")])
        assert count == tokens.numel()
        bt = inp.block_table[r].long()
        want = bt[tokens // BLOCK] * stride + tokens % BLOCK
        got_rows = ws["rows"][row, :count].long()
        diff = set(want.tolist()) ^ set(got_rows.tolist())
        flips = {int(t) // KPOOL for t in diff}  # rough: pools of differing rows
        assert len(diff) <= 4 * 2 * int(near.sum()), (row, len(diff), flips)


def test_select_hadamard_mma_matches_butterflies(monkeypatch):
    """The MMA form of the indexer query's Hadamard gives the butterflies' bits."""
    from vllm.models.glm5next.nvidia.specialized_decode.hopper import dsa

    base = _run([2400, 9000], 6, seed=7)[4]
    monkeypatch.setattr(dsa, "SELECT_HADAMARD_MMA", True)
    mma = _run([2400, 9000], 6, seed=7)[4]
    assert torch.equal(base["counts"], mma["counts"])
    pools = (
        torch.tensor([2400, 9000]).repeat_interleave(6) + torch.arange(6).repeat(2)
    ) // KPOOL
    for row, n in enumerate(pools.tolist()):
        assert torch.equal(base["logits"][row, : n + 1], mma["logits"][row, : n + 1])
        assert torch.equal(base["rows"][row], mma["rows"][row])


def _attention_ref(w, step, ws):
    stride = step.caches.kv.stride(0) // LATENT
    kv = step.caches.kv
    flat = kv.as_strided(((kv.shape[0] - 1) * stride + BLOCK, LATENT), (LATENT, 1))
    out = torch.zeros(step.tokens, HEADS, LATENT, dtype=torch.float64, device="cuda")
    for row in range(step.tokens):
        count = int(ws["counts"][row])
        if count == 0:
            continue
        k = flat[ws["rows"][row, :count].long()].double()
        s = ws["qabs"][row].double() @ k.T * w.softmax_scale
        out[row] = torch.softmax(s, -1) @ k
    return out


@pytest.mark.parametrize("contexts,q", [CASES[2], CASES[5]])
def test_attention_and_output_vs_fp64(contexts, q):
    w, step, _, out, ws = _run(contexts, q)
    ref = _attention_ref(w, step, ws)
    err = (ws["ob"].double() - ref).norm() / ref.norm()
    assert err < 8e-3, err
    ao = torch.einsum("thl,hlv->thv", ws["ob"].double(), w.w_uv.double())
    assert ((ws["ao"].double() - ao.reshape(step.tokens, -1)).norm() / ao.norm()) < 4e-3
    final = ws["ao"].double() @ w.o_proj.double().T
    assert ((out.double() - final).norm() / final.norm()) < 4e-3


def test_padded_rows_write_nothing():
    w, step, before, out, ws = _run([2400, 2600, 0], 6, padded=1)
    assert torch.isfinite(out.float()).all()
    assert (ws["counts"][12:] == 0).all()
    # The padded request's blocks are untouched.
    bt = step.inputs.block_table[2].long()
    assert torch.equal(step.caches.kv[bt], before.kv[bt])
    assert torch.equal(step.caches.tail[3], before.tail[3])


@pytest.mark.parametrize("q", [1, 6])
def test_cuda_graph_replay_follows_inputs(q):
    """A captured step replayed on new contents of its buffers (inputs, metadata,
    caches) gives the eager result on those contents, bit for bit."""
    g = torch.Generator(device="cuda").manual_seed(3)
    weights = _weights(g, 2048)
    a = Step(g, [2400, 6000], q)
    b = Step(g, [2400, 6000], q)
    layer = _layer(weights, a.tokens, q)
    layer.forward(a.x, a.inputs, a.clone_caches(), q)  # compile
    caches = a.clone_caches()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = layer.forward(a.x, a.inputs, caches, q)
    a.x.copy_(b.x)
    for name in ("positions", "kv_slots", "req_ids", "block_table", "index_slots",
                 "index_block_table", "tail_slots"):  # fmt: skip
        getattr(a.inputs, name).copy_(getattr(b.inputs, name))
    for name in ("kv", "index", "tail"):
        getattr(caches, name).copy_(getattr(b.caches, name))
    eager_caches = b.clone_caches()
    graph.replay()
    eager = layer.forward(b.x, b.inputs, eager_caches, q)
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    for name in ("kv", "index", "tail"):
        assert torch.equal(getattr(caches, name), getattr(eager_caches, name))


def test_draft_reuses_logical_topk():
    g = torch.Generator(device="cuda").manual_seed(5)
    weights = _weights(g, 4096)
    step = Step(g, [2400, 9000], 6)
    topk = torch.full((32, 2176), -7, dtype=torch.int32, device="cuda")
    layer = _layer(weights, step.tokens, 6, topk_buffer=topk)
    layer.forward(step.x, step.inputs, step.caches, 6)
    rows = layer.ws.rows[: step.tokens].clone()
    counts = layer.ws.counts[: step.tokens].clone()
    layer.forward(step.x, step.inputs, step.caches, 6, skip_topk=True)
    torch.cuda.synchronize()
    assert torch.equal(layer.ws.counts[: step.tokens], counts)
    for row in range(step.tokens):
        n = int(counts[row])
        assert set(layer.ws.rows[row, :n].tolist()) == set(rows[row, :n].tolist())
        assert (topk[row, n:] == -1).all()
