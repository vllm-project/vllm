# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The references and helpers the Muse Spark numerics tests share: the
comparison metrics (cos, relative L2, bf16 ulps, the GEMV / ulp checks), the
rank-order sums of K2's all-reduce and owner reduce, the K1 boundary's
tolerance, the RoPE table, fp64 GQA decode through a block table, the stock
attention core as decode runs it (``stock_attention``), KV cache fixtures and
page assignment, and ``FakeMrv2`` (MRV2's persistent block tables, slot
mappings, sequence lengths, positions and pad flags, written in place a step).
Every function takes plain tensors (CPU or GPU) and reads its sizes from
them.
"""

import math
from collections.abc import Collection, Mapping, Sequence
from types import SimpleNamespace

import torch

from vllm.models.muse_spark.amd.specialized_decode.mono import spec

BF16 = torch.bfloat16
F32 = torch.float32
F64 = torch.float64


def bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 ulps between a and b (bf16 or rounded to it; +0 and -0 coincide)."""

    def ordered(x):
        i = x.to(BF16).contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (ordered(a) - ordered(b)).abs()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def check_ulps(got, ref, *, frac: float = 0.999, max_ulps: int = 2, what="") -> None:
    """<= 1 bf16 ulp on >= ``frac`` of the elements, never more than
    ``max_ulps``."""
    u = bf16_ulps(got, ref)
    assert u.max().item() <= max_ulps, (what, u.max().item())
    assert (u <= 1).float().mean().item() >= frac, (what, (u > 1).sum().item())


def check_gemv(got, ref, x, w, *, frac: float = 0.999, what="") -> None:
    """bf16 GEMV rows ``got`` vs ``ref`` of ``x @ w.T``: <= 1 bf16 ulp on >=
    ``frac``, and past 1 ulp only within fp32 accumulation noise
    (``|got - ref| <= 2^-20 sum_k |x w|``): two fp32 dots summed in different
    orders can land 2 ulps apart on a cancelling sum, while a dropped K block
    still fails."""
    got, ref = got.cpu(), ref.cpu()
    u = bf16_ulps(got, ref)
    absdot = (x.double().abs() @ w.double().abs().T).cpu()
    noise = (got.double() - ref.double()).abs() <= 2**-20 * absdot
    assert ((u <= 1) | noise).all(), (what, u.max().item())
    assert (u <= 1).float().mean().item() >= frac, (what, (u > 1).sum().item())


# ---------------------------------------------------------------- the stages


def _pad_mask(pad, rows: int, device) -> torch.Tensor:
    """The pad flags as bool [rows] (none when ``pad`` is None)."""
    if pad is None:
        return torch.zeros(rows, dtype=torch.bool, device=device)
    return pad[:rows].to(device=device, dtype=torch.bool)


def rank_order_sum(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """fp32 sum of ``parts`` in rank (list) order."""
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc


def attn_out(partials: Sequence[torch.Tensor]) -> torch.Tensor:
    """The seam's all-reduce: the bf16 partials summed in fp32 in rank order,
    rounded once (the custom-AR rule)."""
    return rank_order_sum(partials).to(BF16)


def _rsq(x: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 ``rsqrt(mean(x^2) + eps)`` over the last dim."""
    x = x.float()
    return torch.rsqrt((x * x).sum(-1, keepdim=True) / x.shape[-1] + eps)


def owner_sum(rows_by_rank: Sequence[torch.Tensor]) -> torch.Tensor:
    """R: the ranks' bf16 rows summed in fp32 in rank order, never rounded
    (README, Numerics: the deviation)."""
    return rank_order_sum(rows_by_rank)


def owner_sumsq(R, w_postexp) -> torch.Tensor:
    """A pick's sum of ``(R (w + 1))^2`` over the pooled width, fp32 [S, 8]
    (fp64 accumulation: the owners' fixed order is within fp32 rounding)."""
    x = R.double() * (w_postexp.double() + spec.GAIN)
    return x.square().sum(-1).float()


def stream_tolerance(branch, stream, w_postffn, alpha, beta) -> torch.Tensor:
    """The ``stream_out`` bound (DESIGN_BC 4): ``1e-5 (|alpha s| + |beta y|)``,
    elementwise; a relative bound fails where ``alpha s + beta y`` cancels."""
    x = branch.float() * (w_postffn.float() + spec.GAIN)
    y = x * _rsq(x, spec.POST_FFN_EPS)
    return 1e-5 * ((alpha.float() * stream.float()).abs() + (beta.float() * y).abs())


def rope_table(max_pos: int, theta: float = spec.ROPE_THETA, device="cpu"):
    """A ``cos_sin_cache`` as ``RotaryEmbedding(64, 64, ..., dtype=float)``
    builds it: fp32 [max_pos, 64], cos in columns 0 .. 31, sin in 32 .. 63."""
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=F32) / 64))
    freqs = torch.outer(torch.arange(max_pos, dtype=F32), inv)
    return torch.cat([freqs.cos(), freqs.sin()], -1).to(device)


def gqa_decode(
    q, kv, block_table, seq_lens, *, window: bool, scale=spec.QK_SCALE, pad=None
) -> torch.Tensor:
    """fp64 decode attention of each row's q heads (bf16 [S, HEADS * 64] or
    [S, HEADS, 64], as K1 leaves them) over its keys ``lo <= k < L`` through
    the block table: ``L = clamp(seq_lens[t], 0, 16 * table row)``, ``lo = L -
    WINDOW`` for a sliding layer (else 0), key k on page ``block_table[t, k //
    16]`` row ``k % 16``; q head h reads KV head ``h // 8``. ``kv`` is a
    layer's cache [blocks, KV_HEADS, KV_BLOCK, KV_C], any strides. Pad rows
    and ``L == 0``: zeros. fp64 [S, HEADS, 64] (exact softmax)."""
    S = q.shape[0]
    pad = _pad_mask(pad, S, "cpu")
    out = torch.zeros(S, spec.HEADS, spec.HEAD_DIM, dtype=F64)
    cap = block_table.shape[1] * spec.KV_BLOCK
    for t in range(S):
        L = min(max(int(seq_lens[t]), 0), cap)
        if bool(pad[t]) or L == 0:
            continue
        lo = max(0, L - spec.WINDOW) if window else 0
        k = torch.arange(lo, L)
        blocks = block_table[t].cpu().long()[k // spec.KV_BLOCK]
        rows = kv[blocks.to(kv.device), :, (k % spec.KV_BLOCK).to(kv.device)]
        rows = rows.cpu().double()  # [keys, KV_HEADS, KV_C]
        K, V = rows[..., : spec.V_OFF], rows[..., spec.V_OFF :]
        qh = q[t].cpu().double().reshape(spec.KV_HEADS, spec.Q_PER_KV, -1)
        p = torch.softmax(torch.einsum("gjd,ngd->gjn", qh, K) * scale, -1)
        out[t] = torch.einsum("gjn,ngd->gjd", p, V).reshape(spec.HEADS, -1)
    return out


# ---------------------------------------------------------------- KV fixtures

PAD_SLOT = -1  # MRV2's PAD_SLOT_ID: no insert


def kv_cache(
    num_blocks: int,
    *,
    layout: str = "LBHNC",
    null: float = math.nan,
    device="cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A layer's bf16 cache [num_blocks, KV_HEADS, KV_BLOCK, KV_C] as AITER FA
    may lay it out: "LBHNC" (a block's heads adjacent, contiguous) or "LHBNC"
    (head-major storage: ``stride(1) > stride(0)``). Random normal values;
    block 0, the null block, holds ``null`` (NaN: a read of it shows)."""
    dims = (num_blocks, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C)
    if layout == "LBHNC":
        kv = torch.randn(*dims, generator=generator)
    elif layout == "LHBNC":
        kv = torch.randn(dims[1], dims[0], *dims[2:], generator=generator)
        kv = kv.transpose(0, 1)
    else:
        raise ValueError(f"KV layout {layout!r} (LBHNC or LHBNC)")
    kv = kv.to(device=device, dtype=BF16)
    kv[0] = null
    return kv


def assign_pages(
    seq_lens: Sequence[int],
    num_blocks: int,
    *,
    extra: int = 0,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Distinct random physical blocks (never 0, the null block) for each
    request's ``cdiv(L, 16) + extra`` pages, in page order."""
    need = [-(-L // spec.KV_BLOCK) + extra for L in seq_lens]
    perm = torch.randperm(num_blocks - 1, generator=generator) + 1
    if sum(need) > perm.numel():
        raise ValueError(f"{sum(need)} pages > {num_blocks - 1} blocks")
    out, at = [], 0
    for n in need:
        out.append(perm[at : at + n].tolist())
        at += n
    return out


class FakeMrv2:
    """Model Runner V2's persistent per-step inputs of the kernel layers
    (DESIGN_BC 1.5), written in place a step as MRV2 writes them, so a captured
    graph's addresses stay valid: per KV group a block table int32 [max_reqs,
    max_blocks] (``BlockTables.input_block_tables[g]``) and a slot-mapping row
    (``slot_mappings`` int64 [G, max_tokens]); ``seq_lens`` int32 [max_reqs];
    ``positions`` int64 and ``is_padding`` bool [max_tokens].

    ``groups``: KV group -> its layers' names; ``sliding``: the sliding
    groups. A sliding group's entries wholly below the window of the
    request's first scheduled token are nulled (block 0); entries past a
    request's last page keep what an earlier step left (``stale`` fills them
    instead); padded rows are zeroed."""

    def __init__(
        self,
        groups: Mapping[str, Sequence[str]],
        sliding: Collection[str],
        *,
        max_reqs: int = spec.MAX_ROWS,
        max_blocks: int = 256,
        max_tokens: int = spec.MAX_ROWS,
        device="cpu",
    ):
        self.groups = {g: tuple(names) for g, names in groups.items()}
        self.sliding = frozenset(sliding)
        self.block_tables = {
            g: torch.zeros(max_reqs, max_blocks, dtype=torch.int32, device=device)
            for g in self.groups
        }
        self.slot_mappings = torch.full(
            (len(self.groups), max_tokens), PAD_SLOT, dtype=torch.int64, device=device
        )
        self.seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.ones(max_tokens, dtype=torch.bool, device=device)

    def _view(self, width: int, q: int) -> SimpleNamespace:
        reqs = width // q
        md, sm = {}, {}
        for gi, (g, names) in enumerate(self.groups.items()):
            for name in names:
                sm[name] = self.slot_mappings[gi, :width]
                md[name] = SimpleNamespace(
                    block_table=self.block_tables[g][:reqs],
                    seq_lens=self.seq_lens[:reqs],
                    slot_mapping=sm[name],
                    decode_metadata=SimpleNamespace(max_query_len=q),
                    num_prefills=0,
                    num_extends=0,
                    num_decode_tokens=width,
                    num_decodes=reqs,
                )
        return SimpleNamespace(
            width=width,
            attn_metadata=md,
            slot_mapping=sm,
            positions=self.positions[:width],
            pad=self.is_padding[:width],
        )

    def capture(self, width: int, q: int = 1) -> SimpleNamespace:
        """A capture / dummy run: rows zeroed, ``seq_lens`` q (each dummy
        request's query), slots ``PAD_SLOT``, positions 0, every row padding."""
        reqs = width // q
        for bt in self.block_tables.values():
            bt[:reqs].zero_()
        self.slot_mappings.fill_(PAD_SLOT)
        self.seq_lens.zero_()
        self.seq_lens[:reqs] = q
        self.positions[:width].zero_()
        self.is_padding[:width].fill_(True)
        return self._view(width, q)

    def step(
        self,
        seq_lens: Sequence[int],
        pages: Mapping[str, Sequence[Sequence[int]]],
        width: int,
        q: int = 1,
        *,
        stale: int | None = None,
    ) -> SimpleNamespace:
        """One decode (q = 1) or uniform verify step of ``len(seq_lens)``
        requests padded to ``width`` rows. ``seq_lens[r]``: request r's context
        including this step's q tokens; ``pages[g][r]``: its physical blocks in
        group g, page order. Returns the step's view: ``attn_metadata`` and
        ``slot_mapping`` by layer name (slices of the persistent buffers),
        ``positions``, ``pad``."""
        reqs, padded = len(seq_lens), width // q
        if padded * q != width or reqs > padded:
            raise ValueError(f"{reqs} requests of {q} rows in a width {width}")
        for gi, (g, bt) in enumerate(self.block_tables.items()):
            self.slot_mappings[gi].fill_(PAD_SLOT)
            for r in range(padded):
                if r >= reqs:
                    bt[r].zero_()
                    continue
                L = seq_lens[r]
                n = -(-L // spec.KV_BLOCK)
                ids = list(pages[g][r][:n])
                if g in self.sliding:  # block_table.py: num_computed_tokens - reach
                    null = max(L - q - (spec.WINDOW - 1), 0) // spec.KV_BLOCK
                    ids[:null] = [0] * null
                bt[r, :n] = torch.tensor(ids, dtype=torch.int32)
                if stale is not None:
                    bt[r, n:] = stale
                for i in range(q):
                    pos = L - q + i
                    blk = pages[g][r][pos // spec.KV_BLOCK]
                    self.slot_mappings[gi, r * q + i] = (
                        blk * spec.KV_BLOCK + pos % spec.KV_BLOCK
                    )
        self.seq_lens.zero_()
        self.seq_lens[:reqs] = torch.tensor(seq_lens, dtype=torch.int32)
        self.positions[:width].zero_()
        self.is_padding[:width].fill_(True)
        for r, L in enumerate(seq_lens):
            self.positions[r * q : (r + 1) * q] = torch.arange(L - q, L)
            self.is_padding[r * q : (r + 1) * q] = False
        return self._view(width, q)


def stock_attention(ns) -> torch.Tensor:
    """What stock decode computes for ``ns`` (``case``: S, window, lens;
    ``kv``, ``bt``, ``seq_lens``, ``qkvo``): ``paged_attention_v1`` on the
    layer's cache views (``_split_kv_cache``), a sliding layer on
    ``_build_swa_decode_tables``' window-trimmed table with the window 2048,
    a full layer on the whole table. bf16 [S, 16, 64]."""
    import aiter  # noqa: F401

    from vllm.v1.attention.backends.rocm_aiter_fa import (
        _PARTITION_SIZE_ROCM,
        AiterFlashAttentionMetadataBuilder,
    )

    case, dev = ns.case, ns.kv.device
    S = case.S
    key_cache, value_cache = ns.kv.transpose(1, 2).split(64, dim=-1)
    bt, lens = ns.bt[:S], ns.seq_lens[:S]
    max_len = max(case.lens)
    if case.window:
        nwb = spec.WINDOW // spec.KV_BLOCK + 1
        builder = SimpleNamespace(
            aot_sliding_window=(spec.WINDOW - 1, 0),
            block_size=spec.KV_BLOCK,
            _swa_block_table=torch.zeros(S, nwb, dtype=torch.int32, device=dev),
            _swa_seq_lens=torch.zeros(S, dtype=torch.int32, device=dev),
            _swa_block_offsets=torch.arange(nwb, dtype=torch.int64, device=dev),
        )
        table, lens, max_len = (
            AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(  # noqa: E501
                builder, bt, lens, S, max_len
            )
        )
        sliding = spec.WINDOW
    else:
        table, sliding = bt, 0
    parts = -(-max_len // _PARTITION_SIZE_ROCM)
    work = torch.empty(
        S * spec.HEADS * parts * 64 * 2 + 2 * S * spec.HEADS * parts * 4,
        dtype=torch.uint8,
        device=dev,
    )
    q = ns.qkvo[:, : spec.CORE].reshape(S, spec.HEADS, 64).contiguous()
    out = torch.empty_like(q)
    one = torch.ones(1, dtype=F32, device=dev)
    # the kernel reads cu_query_lens[t + 1] (query_len = its difference): decode
    # passes ``query_start_loc[:num_decodes]``, a view of the [S + 1] offsets,
    # so the row past the end is real; a bare arange(S) reads past its end and
    # a stale value > 1 there skips the row (output never written)
    qsl = torch.arange(S + 1, dtype=torch.int32, device=dev)[:S]
    torch.ops.aiter.paged_attention_v1(
        out, work, q, key_cache, value_cache, spec.QK_SCALE, table,
        qsl, lens, max_len, None,
        "auto", "NHD", 0.0, one, one, None, _PARTITION_SIZE_ROCM, 1, sliding,
    )  # fmt: skip
    return out


def row_ulps(got, ref) -> torch.Tensor:
    """|got - ref| in bf16 ulps at the scale of each (token, head) row of
    ``ref`` (its largest element's ulp). Element ulps are no measure here:
    an output near zero (V cancelling) has a tiny ulp, and one bf16 rounding
    of P -- stock's and the split's alike -- moves it by many."""
    r = ref.double()
    top = r.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    return (got.double() - r).abs() / torch.exp2(torch.floor(torch.log2(top)) - 7)
