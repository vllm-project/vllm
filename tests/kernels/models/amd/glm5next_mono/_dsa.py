# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash whole-layer sparse-attention MonoKernel on CDNA4, a process a rank.

Every stage is checked on the kernel's own inputs to it, against the fp64 math of
`reference.py`: the attention-side mHC (and vLLM's mHC op beside it), the q_a /
kv_a / indexer projections, the norms and the MLA cache row, q_b, W_UK, the pool
scores, the sparse attention and W_UV, R'' on the TP sum of the o_proj partials
and the FFN half. Three stages have no fp64 model here and are checked against
the vLLM kernel the kernel replaces, on the same inputs (stage names end in "vs
vLLM"): the kpool compression and tail ring, and the FWHT FP8 index query. The
top-512 selection and its tail fit are checked exactly. Rows mix short (every
token), long (scored), pool completing and padding rows."""

import torch

from ._ffn import (
    POLL_TIMEOUT_US,
    REPLAYS,
    _Layer,
    _vllm_seam,
    add_ffn_half,
)
from .reference import F64, Bound, Numerics, _rms_norm, mhc_post, mhc_pre, rel

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
        from vllm.models.glm5next.amd.specialized_decode.mono.config import (
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
        from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import (
            FlashDsaWeights,
        )

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

    def __init__(
        self,
        S: int,
        device,
        seed: int,
        kv_fp8: bool = False,
        max_len: int = MAX_LEN,
        lens: tuple[int, ...] = LENS,
    ) -> None:
        from vllm.models.glm5next.amd.specialized_decode.mono.config import (
            INDEX_DIM,
            KV_LORA,
        )
        from vllm.models.glm5next.amd.specialized_decode.mono.ffn.layout import (
            ROW_WORDS,
        )

        g = torch.Generator(device=device).manual_seed(seed)
        per = max_len // BLOCK
        nb = 1 + S * per
        ipb = BLOCK // 4
        self.ipb = ipb
        self.kv = torch.randn(nb, BLOCK, KV_LORA, generator=g, device=device).bfloat16()
        # an e4m3 cache over a per-tensor scale (vLLM's _k_scale; not 1 here so
        # the test sees it applied)
        self.k_scale = None
        if kv_fp8:
            self.k_scale = torch.tensor([0.5], device=device)
            self.kv = _e4m3(self.kv, self.k_scale)
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
        lens = list(lens[: S - 1]) + [0]
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
        from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import (
            FlashDsaCaches,
        )

        return FlashDsaCaches(self.kv, self.idx, self.tail, self.k_scale)

    def deq(self, rows: torch.Tensor) -> torch.Tensor:
        """Cache rows as fp32 values."""
        if self.k_scale is None:
            return rows.float()
        return rows.view(torch.float8_e4m3fn).float() * self.k_scale

    def clone(self):
        c = object.__new__(_Caches)
        c.__dict__.update(self.__dict__)
        c.kv, c.idx, c.tail = self.kv.clone(), self.idx.clone(), self.tail.clone()
        return c


def _box(rt, name, n, packed=False):
    """The first `n` values of a mailbox: f32 pairs, or packed bf16 pairs."""
    words = rt.scratch[rt.sc[name] :].view(torch.int32)
    if packed:
        return words[:n:2].contiguous().view(torch.bfloat16)
    return words[: 2 * n : 2].contiguous().view(torch.float32)


def _rms(x, w, eps=1e-5):
    x = x.float()
    return (
        x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    ).bfloat16()


def _e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The fp8 MLA cache write of vLLM (`concat_and_cache_mla`): x / scale,
    saturated to the e4m3 range, as raw bytes."""
    e4m3 = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return e4m3.view(torch.uint8)


def _check(
    rank: int,
    tp: int,
    device,
    fp8,
    kv_fp8: bool,
    widths: tuple[int, ...],
    bounds: dict[str, Bound],
) -> None:
    import torch.distributed as dist
    import torch.nn.functional as F

    from vllm.distributed import get_tp_group
    from vllm.models.glm5next.amd.ops.kpool_compress import (
        kpool_decode_update_and_maybe_write_cache_batched,
    )
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        HC_MULT,
        HIDDEN,
        INDEX_DIM,
        INDEX_EPS,
        INDEX_HEADS,
        KV_LORA,
        MLA_SCALE,
        NOPE_DIM,
        Q_LORA,
        TOPK_POOLS,
        TOPK_TOKENS,
        V_DIM,
    )
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import (
        FlashFfnOp,
        FlashFfnRuntime,
    )
    from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8

    lay, dsa = _Layer(rank, tp, device), _Dsa(rank, tp, device, fp8)
    H = dsa.H
    group = get_tp_group().cpu_group
    max_pools = MAX_LEN // 4
    wfmt = "/".join("fp8" if f else "bf16" for f in fp8)
    title = f"DSA layer, TP{tp}, q_a|kv_a/q_b/o_proj {wfmt}, KV cache "
    num = Numerics(title + ("fp8" if kv_fp8 else "bf16"), bounds, rank)
    for S in widths:
        rt = FlashFfnRuntime(
            S,
            rank,
            tp,
            group,
            2048 // tp,
            1,
            POLL_TIMEOUT_US,
            False,
            dsa=True,
            max_pools=max_pools,
            dsa_fp8=fp8,
            kv_fp8=kv_fp8,
        )
        op = FlashFfnOp(rt, lay.weights(), 0, dsa=dsa.weights())
        g = lay.g
        x = torch.randn(S, HIDDEN, generator=g, device=device).bfloat16()
        res = torch.randn(S, HC_MULT, HIDDEN, generator=g, device=device).bfloat16()
        post = torch.rand(S, HC_MULT, 1, generator=g, device=device) * 2
        comb = torch.softmax(
            torch.randn(S, HC_MULT, HC_MULT, generator=g, device=device), -1
        )
        cache = _Caches(S, device, seed=10 + S, kv_fp8=kv_fp8)
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

        # attention-side mHC (post of the previous layer, pre of this one)
        from types import SimpleNamespace

        alay = SimpleNamespace(
            fn=dsa.hc_fn, scale=dsa.hc_scale, base=dsa.hc_base, gamma=dsa.g_in
        )
        ra, pa, ca, a_vllm = _vllm_seam(alay, x, res, post, comb)
        ra64 = mhc_post(x, res, post, comb)
        pa64, ca64, a64 = mhc_pre(ra64, dsa.hc_fn, dsa.hc_scale, dsa.hc_base, dsa.g_in)
        a_in = _box(rt, "a_in", S * HIDDEN, packed=True).view(S, HIDDEN)
        num.add("attention input", S, a_in[:live], a64[:live], a_vllm[:live])
        a = a_in.to(F64)

        # q_a | kv_a (FP8 or bf16 weights on bf16 activations), wk | weights_proj
        # | gate: fp64 GEMMs on the kernel's own input and the dequantized weights
        qa = _box(rt, "qa", S * Q_LORA).view(S, Q_LORA)
        kva = _box(rt, "kva", S * KV_LORA).view(S, KV_LORA)
        got = torch.cat([qa, kva], 1)
        num.add("q_a | kv_a", S, got[:live], (a @ dsa.d_qkv.to(F64).T)[:live])
        ik = _box(rt, "ik", S * INDEX_DIM).view(S, INDEX_DIM)
        iw = _box(rt, "iw", S * INDEX_HEADS).view(S, INDEX_HEADS)
        ig = _box(rt, "ig", S * INDEX_DIM).view(S, INDEX_DIM)
        kw = a @ dsa.w_idx.to(F64).T
        num.add("index k", S, ik[:live], kw[:live, :INDEX_DIM])
        num.add("index head weights", S, iw[:live], kw[:live, INDEX_DIM:])
        num.add("index gate", S, ig[:live], (a @ dsa.w_gate.to(F64).T)[:live])

        # q_a RMSNorm; KV RMSNorm into the MLA cache at the row's slot (worst row)
        q_an = _box(rt, "q_an", S * Q_LORA, packed=True).view(S, Q_LORA)
        num.add("q_a norm", S, q_an[:live], _rms_norm(qa, dsa.g_qa, 1e-5)[:live])
        kv64 = _rms_norm(kva, dsa.g_kv, 1e-5)
        kv_want = _rms(kva, dsa.g_kv)
        flat_kv = cache.kv.view(-1, KV_LORA)
        errs = []
        for r in range(live):
            errs.append(rel(cache.deq(flat_kv[int(rows[r, 1])]), kv64[r]))
            if kv_fp8:
                # the bytes vLLM's write makes of the same normed row; a bf16 ulp
                # in the norm can move a value across an e4m3 rounding boundary
                want8 = _e4m3(kv_want[r], cache.k_scale)
                diff = (flat_kv[int(rows[r, 1])] != want8).float().mean()
                assert diff < 0.02, f"S={S} row {r}: e4m3 bytes differ at {diff}"
        num.add_errors("KV cache row (worst)", S, max(errs))
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
        num.add("kpool tail vs vLLM", S, cache.tail, ref.tail)
        v_got, s_got = _icache_read(cache.idx)
        v_ref, s_ref = _icache_read(ref.idx)
        deq_got = v_got * s_got[..., None]
        deq_ref = v_ref * s_ref[..., None]
        num.add("index cache vs vLLM", S, deq_got, deq_ref)
        errs = []
        for r in range(n):
            if int(pos[r]) % 4 == 3:
                b, o = int(pool_slot[r]) // cache.ipb, int(pool_slot[r]) % cache.ipb
                errs.append(rel(deq_got[b, o], deq_ref[b, o]))
        if errs:
            num.add_errors("new pool vs vLLM (worst)", S, max(errs))

        # q_b (FP8 or bf16) and the index query (FWHT + ue8m0 FP8) and head weights
        q_nope = _box(rt, "q_nope", S * H * NOPE_DIM, packed=True).view(S, H, NOPE_DIM)
        want = (q_an.to(F64) @ dsa.d_qb.to(F64).T).view(S, H, NOPE_DIM)
        num.add("q_b", S, q_nope[:live], want[:live])
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
        num.add("index q vs vLLM", S, deq_q[:live], deq_ref[:live])
        num.add("index weights vs vLLM", S, iqw[:live], w_ref[:live])

        # absorbed query
        q_lat = _box(rt, "q_lat", S * H * KV_LORA, packed=True).view(S, H, KV_LORA)
        want = torch.einsum("shn,hln->shl", q_nope.to(F64), dsa.w_uk.to(F64))
        num.add("q_lat (W_UK)", S, q_lat[:live], want[:live])

        # pool scores of long rows, on the kernel's own query and updated cache
        scores = _box(rt, "iscores", S * max_pools).view(S, max_pools)
        sel = rt.scratch[rt.sc["tok_sel"] :].view(torch.int32)[: S * TOPK_TOKENS]
        sel = sel.view(S, TOPK_TOKENS)
        vals, kscale = _icache_read(cache.idx)
        errs = []
        for r in range(live):
            L = int(pos[r]) + 1
            if L <= TOPK_TOKENS:
                continue
            npool = L // 4
            p = torch.arange(npool, device=device)
            blk = cache.bt[r, (p * 4) // BLOCK].long()
            off = p % cache.ipb
            k8 = vals[blk, off].to(F64)  # [P, 128]
            dots = torch.relu(k8 @ q8[r].to(F64).T)  # [P, 32]
            want = (dots * iqw[r].to(F64)).sum(-1) * kscale[blk, off].to(F64)
            got_s = scores[r, :npool]
            errs.append(rel(got_s, want))
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
        if errs:
            num.add_errors("pool scores (worst row)", S, max(errs))

        # sparse attention + W_UV on the kernel's q_lat and selection
        o_in = _box(rt, "o_in", S * H * V_DIM, packed=True).view(S, H, V_DIM)
        errs = []
        for r in range(S):
            L = int(pos[r]) + 1
            if L == 0:
                assert not o_in[r].float().abs().max() > 0, f"S={S}: padding row output"
                continue
            toks = torch.arange(L, device=device) if L <= TOPK_TOKENS else sel[r].long()
            blk = cache.bt[r, toks // BLOCK].long()
            kvr = cache.deq(cache.kv[blk, toks % BLOCK]).to(F64)  # [T, 512]
            sc = torch.einsum("hl,tl->ht", q_lat[r].to(F64), kvr) * MLA_SCALE
            o_lat = torch.softmax(sc, -1) @ kvr
            want = torch.einsum("hl,hvl->hv", o_lat, dsa.w_uv.to(F64))
            errs.append(rel(o_in[r], want))
        num.add_errors("attention (worst row)", S, max(errs))

        # R'' on the TP sum of every rank's o_proj partial: fp64 over the fp64
        # attention-side streams, vLLM's sum of bf16 partials (its all-reduce) over
        # its own; then the FFN half on the kernel's own R''
        o_flat = o_in.view(S, H * V_DIM).contiguous()
        o_all = [torch.empty_like(o_flat) for _ in range(tp)]
        dist.all_gather(o_all, o_flat)
        x_attn = torch.zeros(S, HIDDEN, device=device)
        x_attn64 = torch.zeros(S, HIDDEN, device=device, dtype=F64)
        for r in range(tp):
            w_o = dsa.d_o_full[:, r * H * V_DIM : (r + 1) * H * V_DIM]
            x_attn += (o_all[r].float() @ w_o.T).bfloat16().float()
            x_attn64 += o_all[r].to(F64) @ w_o.to(F64).T
        v_r = _vllm_seam(lay, x_attn.bfloat16(), ra, pa, ca)[0]
        r64 = mhc_post(x_attn64, ra64, pa64, ca64)
        num.add("residual", S, r2[:live], r64[:live], v_r[:live])
        add_ffn_half(num, S, lay, r2, p2, c2, y, rt, live)

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
    num.check()


# q_a | kv_a, q_b, o_proj: FP8 block-128, or bf16 (vLLM dequantizes some on load)
def _check_fault(rank: int, tp: int, device) -> None:
    """A stalled rank in a whole sparse-attention launch at the serving settings
    (early-out polls, rows past the indexer's 2048 tokens): the waiting ranks
    expire, report it, and finish the launch without a memory fault."""
    import os
    import tempfile

    import torch.distributed as dist

    # a private cache keeps the stalling build apart
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = tempfile.mkdtemp(prefix="mono_fault_")

    from vllm.distributed import get_tp_group
    from vllm.models.glm5next.amd.specialized_decode.mono.abi import FaultSpec
    from vllm.models.glm5next.amd.specialized_decode.mono.config import HC_MULT, HIDDEN
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import (
        FlashFfnOp,
        FlashFfnRuntime,
    )

    from ._ffn import FAULT_POLL_TIMEOUT_US

    S = 8
    lay, dsa = _Layer(rank, tp, device), _Dsa(rank, tp, device, FORMATS["mixed"])
    group = get_tp_group().cpu_group
    rt = FlashFfnRuntime(
        S,
        rank,
        tp,
        group,
        2048 // tp,
        1,
        FAULT_POLL_TIMEOUT_US,
        True,
        dsa=True,
        max_pools=MAX_LEN // 4,
        dsa_fp8=FORMATS["mixed"],
        fault=FaultSpec(rank=1, step=0, us=2_000_000),
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
    dist.barrier(group)
    op.launch(
        x, res, post, comb, caches=cache.caches(), rows=cache.rows, block_table=cache.bt
    )
    torch.accelerator.synchronize()
    dist.barrier(group)
    report = [None] * tp
    dist.all_gather_object(report, rt.words.tolist(), group=group)
    if rank == 0:
        waiting = [r for r in range(tp) if r != 1]
        assert rt.flags is not None
        flags = rt.flags.tolist()
        assert all(any(report[r]) for r in waiting), report
        assert all(flags[r] for r in waiting), flags
    rt.close()


TIE_LENS = (62593, 65536, 40000, 2050, 64, 16001, 62592)


def _check_ties(rank: int, tp: int, device, live: float) -> None:
    """Rows up to the model length on an index cache whose pools mostly score
    exactly 0 (`live`: the share of pools with nonzero keys), so the top-512
    threshold falls in a bin of thousands of tied pools: every row still
    attends 2048 distinct tokens of its own, and the launch completes."""
    import torch.distributed as dist

    from vllm.distributed import get_tp_group
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        HC_MULT,
        HIDDEN,
        INDEX_DIM,
        TOPK_TOKENS,
    )
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import (
        FlashFfnOp,
        FlashFfnRuntime,
    )

    S, max_len = 8, 65536
    lay, dsa = _Layer(rank, tp, device), _Dsa(rank, tp, device, FORMATS["mixed"])
    group = get_tp_group().cpu_group
    rt = FlashFfnRuntime(
        S,
        rank,
        tp,
        group,
        2048 // tp,
        1,
        POLL_TIMEOUT_US,
        True,
        dsa=True,
        max_pools=max_len // 4,
        dsa_fp8=FORMATS["mixed"],
    )
    op = FlashFfnOp(rt, lay.weights(), 0, dsa=dsa.weights())
    g = lay.g
    x = torch.randn(S, HIDDEN, generator=g, device=device).bfloat16()
    res = torch.randn(S, HC_MULT, HIDDEN, generator=g, device=device).bfloat16()
    post = torch.rand(S, HC_MULT, 1, generator=g, device=device) * 2
    comb = torch.softmax(
        torch.randn(S, HC_MULT, HC_MULT, generator=g, device=device), -1
    )
    cache = _Caches(S, device, seed=10 + S, max_len=max_len, lens=TIE_LENS)
    # zero the FP8 keys of all but a ``live`` share of the pools (alike on every
    # rank: the index is replicated)
    nb, ipb = cache.idx.shape[:2]
    keep = torch.rand(nb, ipb, generator=torch.Generator().manual_seed(7)) < live
    vals = cache.idx.view(nb, -1)[:, _icache_perm(ipb, device)].view(nb, ipb, INDEX_DIM)
    vals[~keep.to(device)] = 0
    cache.idx.view(nb, -1)[:, _icache_perm(ipb, device)] = vals.view(nb, -1)
    dist.barrier(group)
    outs = [
        t.clone()
        for t in op.launch(
            x,
            res,
            post,
            comb,
            caches=cache.caches(),
            rows=cache.rows,
            block_table=cache.bt,
        )
    ]
    torch.accelerator.synchronize()
    assert not rt.words.any(), f"expired waits {rt.words.tolist()}"
    assert all(torch.isfinite(o.float()).all() for o in outs)
    sel = rt.scratch[rt.sc["tok_sel"] :].view(torch.int32)[: S * TOPK_TOKENS]
    sel = sel.view(S, TOPK_TOKENS).cpu()
    for r, n in enumerate(cache.lens):
        if n <= TOPK_TOKENS:
            continue
        t = sel[r]
        assert int(t.min()) >= 0 and int(t.max()) < n, (r, n, t.min(), t.max())
        assert t.unique().numel() == TOPK_TOKENS, (r, n, t.unique().numel())
        tail = n % 4
        assert set(range(n - tail, n)) <= set(t.tolist()), (r, n, "tail")
    rt.close()


FORMATS = {
    "fp8": (True, True, True),
    "bf16": (False, False, False),
    "mixed": (False, True, True),
}
