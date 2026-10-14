# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash whole-layer sparse-attention MonoKernel on CDNA4, a process a rank.

Every stage is checked on the kernel's own inputs to it: the attention-side mHC,
the q_a / kv_a / indexer projections, the norms and the MLA cache row, the kpool
compression and tail ring (against vLLM's kpool kernel), q_b, the index query
(against vLLM's FWHT FP8 quant), W_UK, the pool scores, the top-512 selection with
its tail fit, the sparse attention and W_UV, then the FFN seam on the TP sum of the
o_proj partials and the MoE. Rows mix short (every token), long (scored), pool
completing and padding rows."""

import pytest
import torch

from .test_flash_ffn_kernel import (
    REPLAYS,
    _Layer,
    _moe_reference,
    _on_cdna4,
    _rel,
    _vllm_seam,
    _worker,
)

ROWS = (2, 4, 8)
BLOCK = 64  # MLA tokens a cache block: 16 kpool entries a block (preshuffled)
MAX_LEN = 8192
# tokens of each row including the new one (0: padding); 4100, 1500, 7000, 64 and
# 6200 complete a pool, 2050 / 2049 / 3333 keep a 2 / 1 / 1-token tail
LENS = (4100, 1500, 2050, 7000, 2049, 64, 3333)


def _bf16(w):
    wb = w.bfloat16().contiguous()
    return wb, None, wb.float()


def _fp8_block(w):
    """FP8 E4M3 with f32 128 x 128 block scales: (q, scales, dequantized)."""
    R, K = w.shape
    wb = w.float().view(R // 128, 128, K // 128, 128)
    s = wb.abs().amax((1, 3)) / 448.0 + 1e-12
    q = (wb / s[:, None, :, None]).to(torch.float8_e4m3fn)
    deq = (q.float() * s[:, None, :, None]).view(R, K)
    return q.view(R, K), s.contiguous(), deq


def _icache_perm(ipb: int, device) -> torch.Tensor:
    """Byte of (pool offset, dim) in an index-cache block (16 x 16-byte tiles)."""
    off = torch.arange(ipb, device=device)[:, None]
    d = torch.arange(128, device=device)[None, :]
    if ipb == 1:
        return (off * 128 + d).reshape(-1)
    return ((off // 16) * 2048 + (d // 16) * 256 + (off % 16) * 16 + d % 16).reshape(-1)


def _icache_read(cache):
    """(values [NB, IPB, 128] f32, scales [NB, IPB]) of an index cache."""
    nb, ipb = cache.shape[:2]
    flat = cache.view(nb, -1)
    vals = flat[:, _icache_perm(ipb, cache.device)].contiguous()
    vals = vals.view(torch.float8_e4m3fn).float().view(nb, ipb, 128)
    scales = flat[:, ipb * 128 :].contiguous().view(torch.float32)
    return vals, scales


class _Dsa:
    """One sparse-attention layer's random weights (every rank builds the same) and
    this rank's shard, in vLLM's post-load layouts."""

    def __init__(self, rank: int, tp: int, device, fp8=(True, True, True)) -> None:
        from vllm.models.glm5next.amd.ops.mono.config import (
            HC_MULT,
            HIDDEN,
            INDEX_DIM,
            INDEX_HEADS,
            KV_LORA,
            MLA_HEADS,
            NOPE_DIM,
            Q_LORA,
            V_DIM,
        )

        g = torch.Generator(device=device).manual_seed(2)
        H = MLA_HEADS // tp
        self.H, self.tp, self.rank = H, tp, rank

        def rnd(*shape, scale=1.0):
            return torch.randn(*shape, generator=g, device=device) * scale

        nmix = HC_MULT * (2 + HC_MULT)
        self.hc_fn = rnd(nmix, HC_MULT * HIDDEN, scale=0.01)
        self.hc_scale = torch.rand(3, generator=g, device=device) + 0.5
        self.hc_base = rnd(nmix, scale=0.1)
        self.g_in = (1 + 0.1 * rnd(HIDDEN)).bfloat16()
        quant = [_fp8_block if f else _bf16 for f in fp8]
        self.w_qkv, self.s_qkv, self.d_qkv = quant[0](
            rnd(Q_LORA + KV_LORA, HIDDEN, scale=HIDDEN**-0.5)
        )
        self.w_idx = rnd(INDEX_DIM + INDEX_HEADS, HIDDEN, scale=HIDDEN**-0.5).bfloat16()
        self.w_gate = rnd(INDEX_DIM, HIDDEN, scale=HIDDEN**-0.5).bfloat16()
        self.g_qa = (1 + 0.1 * rnd(Q_LORA)).bfloat16()
        self.g_kv = (1 + 0.1 * rnd(KV_LORA)).bfloat16()
        self.ln_w = 1 + 0.1 * rnd(INDEX_DIM)
        self.ln_b = 0.1 * rnd(INDEX_DIM)
        self.ape = 0.5 * rnd(4, INDEX_DIM)
        q_b, s_qb, d_qb = quant[1](
            rnd(MLA_HEADS * NOPE_DIM, Q_LORA, scale=Q_LORA**-0.5)
        )
        rows = slice(rank * H * NOPE_DIM, (rank + 1) * H * NOPE_DIM)
        self.w_qb, self.d_qb = q_b[rows].contiguous(), d_qb[rows]
        self.s_qb = (
            None
            if s_qb is None
            else s_qb[rank * H * 2 : (rank + 1) * H * 2].contiguous()
        )
        self.w_iq = rnd(INDEX_HEADS * INDEX_DIM, Q_LORA, scale=Q_LORA**-0.5).bfloat16()
        kv_b = rnd(MLA_HEADS * (NOPE_DIM + V_DIM), KV_LORA, scale=KV_LORA**-0.5)
        per = NOPE_DIM + V_DIM
        self.w_kvb = kv_b[rank * H * per : (rank + 1) * H * per].bfloat16().contiguous()
        heads = self.w_kvb.view(H, per, KV_LORA)
        self.w_uk = heads[:, :NOPE_DIM].transpose(1, 2).contiguous()  # [H, 512, 256]
        self.w_uv = heads[:, NOPE_DIM:]  # [H, 256, 512]
        w_o, s_o, d_o = quant[2](
            rnd(HIDDEN, MLA_HEADS * V_DIM, scale=(MLA_HEADS * V_DIM) ** -0.5)
        )
        self.d_o_full = d_o
        cols = slice(rank * H * V_DIM, (rank + 1) * H * V_DIM)
        self.w_o = w_o[:, cols].contiguous()
        self.s_o = (
            None
            if s_o is None
            else s_o[:, rank * H * 2 : (rank + 1) * H * 2].contiguous()
        )

    def weights(self):
        from vllm.models.glm5next.amd.ops.mono.ffn.op import FlashDsaWeights

        return FlashDsaWeights(
            hc_fn=self.hc_fn,
            hc_scale=self.hc_scale,
            hc_base=self.hc_base,
            g_norm=self.g_in,
            w_qkv=self.w_qkv,
            s_qkv=self.s_qkv,
            w_idx=self.w_idx,
            w_gate=self.w_gate,
            g_qa=self.g_qa,
            g_kv=self.g_kv,
            ln_w=self.ln_w,
            ln_b=self.ln_b,
            ape=self.ape,
            w_qb=self.w_qb,
            s_qb=self.s_qb,
            w_iq=self.w_iq,
            w_uk=self.w_uk,
            w_kvb=self.w_kvb,
            w_o=self.w_o,
            s_o=self.s_o,
        )


class _Caches:
    """Random caches for S requests of up to MAX_LEN tokens, the block table, and
    the step's row table."""

    def __init__(self, S: int, device, seed: int) -> None:
        from vllm.models.glm5next.amd.ops.mono.config import INDEX_DIM, KV_LORA
        from vllm.models.glm5next.amd.ops.mono.ffn.layout import ROW_WORDS

        g = torch.Generator(device=device).manual_seed(seed)
        per = MAX_LEN // BLOCK
        nb = 1 + S * per
        ipb = BLOCK // 4
        self.ipb = ipb
        self.kv = torch.randn(nb, BLOCK, KV_LORA, generator=g, device=device).bfloat16()
        vals = (torch.randn(nb, ipb, INDEX_DIM, generator=g, device=device) * 64).clamp(
            -448, 448
        )
        scales = torch.exp2(
            -torch.randint(6, 10, (nb, ipb), generator=g, device=device).float()
        )
        idx = torch.zeros(nb, ipb, INDEX_DIM + 4, dtype=torch.uint8, device=device)
        flat = idx.view(nb, -1)
        flat[:, _icache_perm(ipb, device)] = (
            vals.to(torch.float8_e4m3fn).view(torch.uint8).view(nb, -1)
        )
        flat[:, ipb * INDEX_DIM :] = scales.view(torch.uint8).view(nb, -1)
        self.idx = idx
        self.tail = torch.randn(
            S + 1, 2, 4, INDEX_DIM, generator=g, device=device
        ).bfloat16()
        perm = torch.randperm(nb - 1, generator=g, device=device) + 1
        self.bt = perm[: S * per].view(S, per).int().contiguous()
        lens = list(LENS[: S - 1]) + [0]
        self.lens = lens
        rows = torch.full((S, ROW_WORDS), -1, dtype=torch.int32, device=device)
        for r, n in enumerate(lens):
            rows[r, 4] = r
            if n == 0:
                continue
            pos = n - 1
            blk = int(self.bt[r, pos // BLOCK])
            rows[r, 0] = pos
            rows[r, 1] = blk * BLOCK + pos % BLOCK
            rows[r, 2] = blk * ipb + (pos // 4) % ipb if pos % 4 == 3 else -1
            rows[r, 3] = (r + 1) * 4 + pos % 4
        self.rows = rows

    def caches(self):
        from vllm.models.glm5next.amd.ops.mono.ffn.op import FlashDsaCaches

        return FlashDsaCaches(self.kv, self.idx, self.tail)

    def clone(self):
        c = object.__new__(_Caches)
        c.__dict__.update(self.__dict__)
        c.kv, c.idx, c.tail = self.kv.clone(), self.idx.clone(), self.tail.clone()
        return c


def _box(rt, name, n, packed=False):
    """The first ``n`` values of a mailbox: f32 pairs, or packed bf16 pairs."""
    words = rt.scratch[rt.sc[name] :].view(torch.int32)
    if packed:
        return words[:n:2].contiguous().view(torch.bfloat16)
    return words[: 2 * n : 2].contiguous().view(torch.float32)


def _rms(x, w, eps=1e-5):
    x = x.float()
    return (
        x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    ).bfloat16()


def _check(rank: int, tp: int, device, fp8) -> None:
    import torch.distributed as dist
    import torch.nn.functional as F

    from vllm.distributed import get_tp_group
    from vllm.models.glm5next.amd.ops.kpool_compress import (
        kpool_decode_update_and_maybe_write_cache_batched,
    )
    from vllm.models.glm5next.amd.ops.mono.config import (
        HC_MULT,
        HIDDEN,
        INDEX_DIM,
        INDEX_EPS,
        INDEX_HEADS,
        KV_LORA,
        MLA_SCALE,
        MOE_SLOTS,
        NOPE_DIM,
        Q_LORA,
        TOPK_POOLS,
        TOPK_TOKENS,
        V_DIM,
    )
    from vllm.models.glm5next.amd.ops.mono.ffn.op import FlashFfnOp, FlashFfnRuntime
    from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8

    lay, dsa = _Layer(rank, tp, device), _Dsa(rank, tp, device, fp8)
    H = dsa.H
    group = get_tp_group().cpu_group
    max_pools = MAX_LEN // 4
    for S in ROWS if tp > 2 else ROWS[:2]:
        rt = FlashFfnRuntime(
            S,
            rank,
            tp,
            group,
            2048 // tp,
            1,
            20_000_000,
            False,
            dsa=True,
            max_pools=max_pools,
            dsa_fp8=fp8,
        )
        op = FlashFfnOp(rt, lay.weights(), 0, dsa=dsa.weights())
        g = lay.g
        x = torch.randn(S, HIDDEN, generator=g, device=device).bfloat16()
        res = torch.randn(S, HC_MULT, HIDDEN, generator=g, device=device).bfloat16()
        post = torch.rand(S, HC_MULT, 1, generator=g, device=device) * 2
        comb = torch.softmax(
            torch.randn(S, HC_MULT, HC_MULT, generator=g, device=device), -1
        )
        cache = _Caches(S, device, seed=10 + S)
        before = cache.clone()
        rows, live = cache.rows, S - 1
        pos = rows[:, 0].long()
        dist.barrier(group)
        outs = [
            t.clone()
            for t in op.launch(
                x,
                res,
                post,
                comb,
                caches=cache.caches(),
                rows=rows,
                block_table=cache.bt,
            )
        ]
        torch.accelerator.synchronize()
        assert not rt.words.any(), f"S={S}: expired waits {rt.words.tolist()}"
        y, r2, p2, c2 = outs
        gathered = [torch.empty_like(y).cpu() for _ in range(tp)]
        dist.all_gather(gathered, y.cpu(), group=group)
        assert all(torch.equal(gathered[0], o) for o in gathered), f"S={S}: ranks"

        # attention-side mHC
        from types import SimpleNamespace

        alay = SimpleNamespace(
            fn=dsa.hc_fn, scale=dsa.hc_scale, base=dsa.hc_base, gamma=dsa.g_in
        )
        ra, pa, ca, a_ref = _vllm_seam(alay, x, res, post, comb)
        a_in = _box(rt, "a_in", S * HIDDEN, packed=True).view(S, HIDDEN)
        assert _rel(a_in, a_ref) < 1e-2, f"S={S}: attention input {_rel(a_in, a_ref)}"
        a = a_in.float()

        # q_a | kv_a (FP8 weights on bf16 activations), wk | weights_proj | gate
        qa = _box(rt, "qa", S * Q_LORA).view(S, Q_LORA)
        kva = _box(rt, "kva", S * KV_LORA).view(S, KV_LORA)
        want = (a @ dsa.d_qkv.T).bfloat16().float()
        got = torch.cat([qa, kva], 1)
        assert _rel(got, want) < 1e-2, f"S={S}: q_a | kv_a {_rel(got, want)}"
        ik = _box(rt, "ik", S * INDEX_DIM).view(S, INDEX_DIM)
        iw = _box(rt, "iw", S * INDEX_HEADS).view(S, INDEX_HEADS)
        ig = _box(rt, "ig", S * INDEX_DIM).view(S, INDEX_DIM)
        kw = a @ dsa.w_idx.float().T
        assert _rel(ik, kw[:, :INDEX_DIM].bfloat16()) < 1e-2, f"S={S}: index k"
        assert _rel(iw, kw[:, INDEX_DIM:]) < 1e-4, f"S={S}: index head weights"
        assert _rel(ig, (a @ dsa.w_gate.float().T).bfloat16()) < 1e-2, f"S={S}: gate"

        # q_a RMSNorm; KV RMSNorm into the MLA cache at the row's slot
        q_an = _box(rt, "q_an", S * Q_LORA, packed=True).view(S, Q_LORA)
        assert _rel(q_an, _rms(qa, dsa.g_qa)) < 1e-2, f"S={S}: q_a norm"
        kv_want = _rms(kva, dsa.g_kv)
        flat_kv = cache.kv.view(-1, KV_LORA)
        for r in range(live):
            got_row = flat_kv[int(rows[r, 1])]
            assert _rel(got_row, kv_want[r]) < 1e-2, f"S={S} row {r}: KV cache row"
        changed = before.kv.view(-1, KV_LORA).ne(flat_kv).any(-1)
        assert int(changed.sum()) <= live, f"S={S}: stray MLA cache writes"

        # index k LayerNorm + kpool: vLLM's decode kernel on the kernel's inputs
        k_ref = F.layer_norm(ik, (INDEX_DIM,), dsa.ln_w, dsa.ln_b, INDEX_EPS).bfloat16()
        ref = before.clone()
        n = live
        pool_slot = torch.tensor(
            [
                int(cache.bt[r, int(pos[r]) // BLOCK]) * cache.ipb
                + (int(pos[r]) // 4) % cache.ipb
                for r in range(n)
            ],
            dtype=torch.int32,
            device=device,
        )
        kpool_decode_update_and_maybe_write_cache_batched(
            ref.idx,
            ref.tail,
            rows[:n, 3:4].contiguous(),
            k_ref[:n].view(n, 1, INDEX_DIM),
            ig[:n].bfloat16().view(n, 1, INDEX_DIM),
            dsa.ape,
            pool_slot.view(n, 1),
            rows[:n, 0:1].contiguous(),
            4,
            INDEX_DIM,
            round_scale=True,
        )
        assert _rel(cache.tail, ref.tail) < 1e-2, f"S={S}: kpool tail ring"
        v_got, s_got = _icache_read(cache.idx)
        v_ref, s_ref = _icache_read(ref.idx)
        deq_got = v_got * s_got[..., None]
        deq_ref = v_ref * s_ref[..., None]
        assert _rel(deq_got, deq_ref) < 2e-2, (
            f"S={S}: index cache {_rel(deq_got, deq_ref)}"
        )
        for r in range(n):
            if int(pos[r]) % 4 == 3:
                b, o = int(pool_slot[r]) // cache.ipb, int(pool_slot[r]) % cache.ipb
                e = _rel(deq_got[b, o], deq_ref[b, o])
                assert e < 3e-2, f"S={S} row {r}: compressed pool {e}"

        # q_b (FP8) and the index query (FWHT + ue8m0 FP8) and head weights
        q_nope = _box(rt, "q_nope", S * H * NOPE_DIM, packed=True).view(S, H, NOPE_DIM)
        want = (q_an.float() @ dsa.d_qb.T).bfloat16().view(S, H, NOPE_DIM)
        assert _rel(q_nope, want) < 1e-2, f"S={S}: q_b {_rel(q_nope, want)}"
        iq = (q_an.float() @ dsa.w_iq.float().T).bfloat16().view(-1, INDEX_DIM)
        q8_ref, qs_ref = fwht128_quant_fp8(iq.contiguous())
        iq8 = rt.scratch[rt.sc["iq8"] :].view(torch.int32)[
            : 2 * S * INDEX_HEADS * 32 : 2
        ]
        q8 = (
            iq8.contiguous().view(torch.float8_e4m3fn).float().view(S, INDEX_HEADS, 128)
        )
        iqw = _box(rt, "iqw", S * INDEX_HEADS).view(S, INDEX_HEADS)
        w_ref = iw * qs_ref.view(S, INDEX_HEADS) * (INDEX_DIM**-0.5 * INDEX_HEADS**-0.5)
        qs_got = iqw / (iw * (INDEX_DIM**-0.5 * INDEX_HEADS**-0.5))
        deq_q = q8 * qs_got[..., None]
        deq_ref = q8_ref.float().view(S, INDEX_HEADS, 128) * qs_ref.view(
            S, INDEX_HEADS, 1
        )
        assert _rel(deq_q, deq_ref) < 3e-2, f"S={S}: index q {_rel(deq_q, deq_ref)}"
        assert _rel(iqw, w_ref) < 1e-2, f"S={S}: index weights"

        # absorbed query
        q_lat = _box(rt, "q_lat", S * H * KV_LORA, packed=True).view(S, H, KV_LORA)
        want = torch.einsum("shn,hln->shl", q_nope.float(), dsa.w_uk.float())
        assert _rel(q_lat, want.bfloat16()) < 1e-2, f"S={S}: q_lat {_rel(q_lat, want)}"

        # pool scores of long rows, on the kernel's own query and updated cache
        maxp = max_pools
        scores = _box(rt, "iscores", S * maxp).view(S, maxp)
        sel = rt.scratch[rt.sc["tok_sel"] :].view(torch.int32)[: S * TOPK_TOKENS]
        sel = sel.view(S, TOPK_TOKENS)
        vals, kscale = _icache_read(cache.idx)
        for r in range(live):
            L = int(pos[r]) + 1
            if L <= TOPK_TOKENS:
                continue
            npool = L // 4
            p = torch.arange(npool, device=device)
            blk = cache.bt[r, (p * 4) // BLOCK].long()
            off = p % cache.ipb
            k8 = vals[blk, off]  # [P, 128]
            dots = torch.relu(k8 @ q8[r].T)  # [P, 32]
            want = (dots * iqw[r]).sum(-1) * kscale[blk, off]
            got_s = scores[r, :npool]
            assert _rel(got_s, want) < 1e-4, (
                f"S={S} row {r}: scores {_rel(got_s, want)}"
            )
            # selection: 512 distinct pools at or above the 512th score, then the tail
            tail = L - 4 * npool
            keep = TOPK_TOKENS - tail
            toks = sel[r].long()
            assert torch.equal(
                toks[keep:], torch.arange(4 * npool, L, device=device)
            ), f"S={S} row {r}: tail tokens"
            pools = torch.unique(toks[:keep] // 4)
            assert pools.numel() == TOPK_POOLS, f"S={S} row {r}: {pools.numel()} pools"
            thresh = got_s.topk(TOPK_POOLS).values[-1]
            assert bool((got_s[pools] >= thresh).all()), f"S={S} row {r}: selection"
            if tail:
                last = toks[keep - 1] // 4
                assert got_s[last] == got_s[pools].min(), f"S={S} row {r}: fit drop"

        # sparse attention + W_UV on the kernel's q_lat and selection
        o_in = _box(rt, "o_in", S * H * V_DIM, packed=True).view(S, H, V_DIM)
        for r in range(S):
            L = int(pos[r]) + 1
            if L == 0:
                assert not o_in[r].float().abs().max() > 0, f"S={S}: padding row output"
                continue
            toks = torch.arange(L, device=device) if L <= TOPK_TOKENS else sel[r].long()
            blk = cache.bt[r, toks // BLOCK].long()
            kvr = cache.kv[blk, toks % BLOCK].float()  # [T, 512]
            sc = torch.einsum("hl,tl->ht", q_lat[r].float(), kvr) * MLA_SCALE
            o_lat = (torch.softmax(sc, -1) @ kvr).bfloat16().float()
            want = torch.einsum("hl,hvl->hv", o_lat, dsa.w_uv.float())
            e = _rel(o_in[r], want)
            assert e < 2e-2, f"S={S} row {r}: attention {e}"

        # FFN seam on the TP sum of every rank's o_proj partial
        o_flat = o_in.view(S, H * V_DIM).contiguous()
        o_all = [torch.empty_like(o_flat) for _ in range(tp)]
        dist.all_gather(o_all, o_flat)
        x_attn = torch.zeros(S, HIDDEN, device=device)
        for r in range(tp):
            w_o = dsa.d_o_full[:, r * H * V_DIM : (r + 1) * H * V_DIM]
            x_attn += (o_all[r].float() @ w_o.T).bfloat16().float()
        v_r, v_p, v_c, _ = _vllm_seam(lay, x_attn.bfloat16(), ra, pa, ca)
        assert _rel(r2[:live], v_r[:live]) < 2e-3, f"S={S}: R'' {_rel(r2, v_r)}"
        assert _rel(p2[:live], v_p[:live]) < 1e-3 and _rel(c2[:live], v_c[:live]) < 1e-3

        a_ffn = rt.scratch[rt.sc["a"] :].view(torch.int32)[: S * HIDDEN : 2]
        k_in = a_ffn.contiguous().view(torch.bfloat16).view(S, HIDDEN)
        ref_y, order = _moe_reference(k_in, lay.w_r, lay.bias, lay.w13, lay.w2, 2048)
        experts = rt.scratch[rt.sc["sel"] :].view(torch.int32)[: 2 * S * MOE_SLOTS : 2]
        routed = experts.view(S, MOE_SLOTS)[:, 1:].sort(-1).values
        assert torch.equal(routed[:live], order[:live].int().sort(-1).values)
        assert _rel(y[:live], ref_y[:live]) < 5e-3, f"S={S}: MoE {_rel(y, ref_y)}"

        # graph replays: the cache writes are idempotent, every replay bit-exact
        dist.barrier(group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rt.advance_step()
            res_g = op.launch(
                x,
                res,
                post,
                comb,
                caches=cache.caches(),
                rows=rows,
                block_table=cache.bt,
            )
        for i in range(REPLAYS):
            graph.replay()
            if i % 100 == 99 or i == REPLAYS - 1:
                torch.accelerator.synchronize()
                assert all(torch.equal(o, w) for o, w in zip(res_g, outs)), (
                    f"S={S}: replay {i} differs"
                )
        assert not rt.words.any()
        dist.barrier(group)
        rt.close()


# q_a | kv_a, q_b, o_proj: FP8 block-128, or bf16 (vLLM dequantizes some on load)
FORMATS = {
    "fp8": (True, True, True),
    "bf16": (False, False, False),
    "mixed": (False, True, True),
}


@pytest.mark.skipif(not _on_cdna4(), reason="the Flash MonoKernel targets CDNA4")
@pytest.mark.parametrize(
    "tp, fmt", [(2, "mixed"), (4, "fp8"), (4, "bf16"), (8, "mixed")]
)
def test_dsa_layer_matches_vllm_ops(monkeypatch: pytest.MonkeyPatch, tp: int, fmt: str):
    from ....utils import multi_process_parallel

    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    pytest.importorskip("flydsl")
    multi_process_parallel(monkeypatch, tp, 1, _worker(_check, FORMATS[fmt]))
