# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MLA query at many local heads (TP1 / DEP: all 96 heads on one GPU), and at
TP8 / DCP8 (12 heads) from SPLIT_FROM rows, as three kernels instead of MlaQuery's
one:

  QueryNorm   the q_a RMSNorm -> the q_c rows [B, 1536] (dsa.overlap.QueryNorm,
              bit-identical to MlaQuery's normalization)
  Q_B         q_c @ Q_B^T -> the BF16 query rows [B, H * 192] (split-K projection:
              one 128-row weight tile per SM at H = 96, the weight streamed once)
  MlaAbsorb   the W_UK absorption of each head's 128 nope columns, the 64 rope
              columns and the MLA KV insert -> the query rows [B, H, 576]

MlaQuery gives every (head, Q_B slice) its own CTA of a cluster that stages and
normalizes every token row itself: at 96 heads that is 768-1536 CTAs, each
re-reading the input rows and running the RMSNorm, several waves of a
latency-bound chain (75 us at 5 rows, 306 us at 80 rows on GB300). Split, the Q_B
weight (56.6 MB at 96 heads) streams through one GEMM wave and the absorption reads
W_UK once per 16 tokens.

Numerics are MlaQuery's: Q_B accumulates in FP32 and rounds to BF16, the absorption
takes the BF16 nope rows, accumulates in FP32 (BF16 MMA) and rounds to BF16 before
the per-tensor FP8 query scale (fp8_ds_mla: the BF16 query); the KV insert is
MlaQuery's own (insert / insert_ds / insert_nvfp4)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Uint16, Uint32

from ..common.primitives import (
    alloc,
    at,
    bf16x2,
    copy_evict,
    copy_g2s,
    launch,
    ldmatrix,
    load_words,
    mma_into,
    multicast_words,
    pack_fp8,
    store_words,
    unpack_bf16,
    view,
    zeros,
)
from .query import MlaQuery

# Heads per GPU from which the MLA query runs split at every batch (TP1 / DEP).
WIDE_HEADS = 64
PARTS = 4  # latent column parts per head (128 columns per CTA)
# Fewer heads (TP8 / DCP8: 12): the smallest batch of rows that runs split (under
# DCP MlaAbsorb publishes into the q mailbox).
# MlaQuery gives each (head, 24 Q_B rows) a CTA of an 8-CTA cluster that stages and
# RMSNorms every token row of its tile itself and exchanges the Q_B rows over DSMEM:
# at 12 heads and 80 / 320 rows its 96-480 CTAs are a latency-bound chain.
SPLIT_FROM = 21
# MlaAbsorb CTAs to aim for (one wave of 128-thread CTAs, about two per SM): small
# head counts split the batch's token tiles over this many CTAs.
ABSORB_CTAS = 304


def wide_query(heads: int) -> bool:
    """Whether the MLA query of `heads` local heads runs split at every batch."""
    return heads >= WIDE_HEADS


def split_query(heads: int, batch: int, dcp: int = 0) -> bool:
    """Whether the MLA query of `batch` rows at `heads` local heads runs split
    (QueryNorm, the Q_B projection, MlaAbsorb): always from WIDE_HEADS heads, else
    from SPLIT_FROM rows (under DCP MlaAbsorb publishes into the q mailbox)."""
    return wide_query(heads) or batch >= SPLIT_FROM


def fi_heads(heads: int) -> int:
    """The FlashInfer sparse MLA query heads of `heads` local heads: 64 < H < 128
    zero-padded to 128 (trtllm-gen takes <= 64 or exactly 128 heads; prims_ts
    before flashinfer-ai/flashinfer#7984 runs other counts as two M64 tiles that
    load every KV row twice: at 96 heads 2.0-2.1x slower from 80 rows, GB300 128K,
    101.9 / 149.3 / 244.7 us at 80 / 160 / 320 rows against 50.7 / 72.4 / 112.6
    padded). The stock decode pads only on such builds
    (`FlashInferMLASparseImpl._prims_ts_query_heads`); the pad heads are zero, so
    both head counts compute the same rows."""
    return 128 if 64 < heads < 128 else heads


def absorb_tokens(batch: int, heads: int = WIDE_HEADS) -> int:
    """Tokens per MlaAbsorb tile: 16 up to 16 rows, 32 up to 32, else 64; below
    WIDE_HEADS heads (the tiles then spread over CTAs) at most 32."""
    tokens = 16 if batch <= 16 else 32 if batch <= 32 else 64
    return tokens if heads >= WIDE_HEADS else min(tokens, 32)


def absorb_splits(batch: int, heads: int) -> int:
    """CTAs per (part, head) of MlaAbsorb: each takes every splits-th token tile."""
    tiles = -(-batch // absorb_tokens(batch, heads))
    return max(1, min(tiles, ABSORB_CTAS // (PARTS * heads)))


class MlaAbsorb(MlaQuery):
    """W_UK absorption, rope columns and MLA KV insert from the Q_B rows. One
    128-thread CTA per (128 latent columns, head): 384 CTAs at 96 heads, one wave
    at three per SM. It stages its 32 KB W_UK slice once, then walks the batch in
    tiles of `absorb_tokens` tokens (the next tile's rows load while the current
    one's output stores); warp w owns latent columns 32w..32w+31 of the part for
    every 16-token MMA tile (BF16 MMA, FP32 accumulation, K = 128). The four parts
    of a head share its rope columns, every CTA one KV insert per 4H tokens.

    One CTA per (part, head, tile) instead ran 1920 CTAs at 320 rows, each a
    latency-bound chain (W_UK, PDL wait, rows, MMAs, stores): 36 us against 1.5
    us of W_UK bytes.

    Small head counts (TP8 / DCP8: 12) split each (part, head)'s token tiles over
    `absorb_splits` CTAs (grid z), each staging the W_UK slice.

    `dcp` (with `q_heads`): each CTA also multicasts the query bytes it stores
    (latent columns, rope columns) into head block `rank` of every rank's q
    mailbox, as MlaQuery.publish; QueryWait polls it (no poll here).

    Arguments as MlaQuery's with the Q_B rows [B, H * 192] (BF16) in place of the
    Q_B weight: (packed rows, q_a norm, kv_a norm, Q_B rows, W_UK, MLA cache, slots,
    1 / q_scale, k_scale, eps, query rows[, q mailbox, its multicast address,
    rank], stream)."""

    def __init__(
        self,
        heads: int,
        batch: int,
        *,
        dsa: bool = True,
        fp8: bool = False,
        nvfp4: bool = False,
        release: bool = True,
        dcp: int = 0,
        q_heads: int = 0,
    ):
        super().__init__(
            heads,
            batch,
            dsa=dsa,
            fp8=fp8,
            nvfp4=nvfp4,
            dcp=dcp,
            q_heads=q_heads,
            store_qc=False,
            poll=False,
            release=release,
        )
        self.box_arg = 11  # the q mailbox follows the query rows
        self.wide, self.tokens = True, absorb_tokens(batch, heads)
        self.latent, self.tiles = 512 // PARTS, -(-batch // self.tokens)
        self.splits = absorb_splits(batch, heads)
        self.grid = (PARTS, heads, self.splits)

    @cute.jit
    def __call__(self, *args):
        launch(self.absorb, args, self.grid, 128)

    @cute.jit
    def stage_rows(self, q_rows, qn, tok0, head, tid):
        """cp.async the nope columns of tokens tok0 .. tok0 + T into `qn`."""
        T: cutlass.Constexpr = self.tokens
        for c in range(tid, T * 16, 128):
            t, j = c // 16, c % 16
            if self.live(tok0 + t):
                dst = qn.iterator + t * (128 + 8) + j * 8
                copy_g2s(dst, at(q_rows, tok0 + t, head * 192 + j * 8))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def publish_words(self, args, token, head, col, words):
        """DCP: `words` (the query bytes of `token`'s head `head` from column
        `col`) into head block `rank` of every rank's q mailbox."""
        if cutlass.const_expr(self.dcp > 0):
            box_mc, rank = args[self.box_arg + 1 : self.box_arg + 3]
            E: cutlass.Constexpr = self.q_bytes
            row = token * self.q_heads + rank * self.heads + head
            address = box_mc + cutlass.Int64(row) * (576 * E) + col * E
            multicast_words(address, words)

    @cute.jit
    def rope(self, q_rows, output, qsi, tok0, part, head, tid, args=()):
        """The rope columns of the tile's tokens (8 per 16 B chunk, the PARTS CTAs
        of a head take every PARTS-th chunk): every load issued before the
        stores."""
        T: cutlass.Constexpr = self.tokens
        per: cutlass.Constexpr = -(-T * 8 // (PARTS * 128))  # chunks per thread
        chunks = []
        for i in cutlass.range_constexpr(per):
            c = (i * 128 + tid) * PARTS + part
            t, k = c // 8, c % 8
            # Past the tile or the batch: a valid row, loaded but not stored.
            row = min(tok0 + t, self.batch - 1)
            source = at(q_rows, row, head * 192 + 128 + 8 * k)
            chunks.append((t, k, load_words(source, count=4, mutable=True)))
        for t, k, words in chunks:
            if (t < T) & self.live(tok0 + t):
                target = at(output, tok0 + t, head, 512 + 8 * k)
                if cutlass.const_expr(self.ds):
                    store_words(target, words)
                    self.publish_words(args, tok0 + t, head, 512 + 8 * k, words)
                else:
                    values: list = []
                    for word in words:
                        values += [v * qsi for v in unpack_bf16(word)]
                    fp8 = (pack_fp8(values[0:4]), pack_fp8(values[4:8]))
                    store_words(target, fp8)
                    self.publish_words(args, tok0 + t, head, 512 + 8 * k, fp8)

    @cute.kernel
    def absorb(self, args):
        q_rows, w_uk, q_scale_inv, output = args[3], args[4], args[7], args[10]
        tid, _, _ = cute.arch.thread_idx()
        part, head, split = cute.arch.block_idx()
        T: cutlass.Constexpr = self.tokens
        MT: cutlass.Constexpr = T // 16
        S: cutlass.Constexpr = self.splits
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        L: cutlass.Constexpr = self.latent
        smem = cutlass.utils.SmemAllocator()
        # 16 B row padding: conflict-free ldmatrix rows.
        uk = alloc(smem, BFloat16, (128, L + 8), 128)
        qn = alloc(smem, BFloat16, (T, 128 + 8), 128)
        ss = alloc(smem, Float32, (8, 2))
        # The CTA's output columns (BF16 for fp8_ds_mla's query, else FP8 bytes),
        # rows padded by 16 B: the pair stores of a warp's 8 row groups hit 8
        # different banks (unpadded, its 128 B rows put them all on one).
        E: cutlass.Constexpr = 2 if self.ds else 1  # bytes per element
        P: cutlass.Constexpr = L * E + 16  # staged row pitch (bytes)
        staged = alloc(smem, cutlass.Uint8, T * P, 16)
        per_row: cutlass.Constexpr = P // (2 * E)  # pairs per staged row
        pair = Uint32 if self.ds else Uint16
        pairs = view(staged.iterator, 0, pair, (T, per_row), (per_row, 1))
        # The static W_UK slice loads before the PDL wait.
        for c in range(tid, 128 * L // 8, 128):
            p, j = c // (L // 8), c % (L // 8)
            dst = uk.iterator + p * (L + 8) + j * 8
            copy_evict(dst, at(w_uk, head, p, part * L + j * 8))
        cute.arch.cp_async_commit_group()
        self.pdl_start()
        self.stage_rows(q_rows, qn, split * T, head, tid)
        qsi = q_scale_inv[0]
        # The KV inserts overlap the staging: CTA (part, head, split) inserts tokens
        # i, i + 4HS, ... (i = 4 * (H * split + head) + part: one each up to 4HS).
        first = (split * self.heads + head) * PARTS + part
        for j in range(first, self.batch, self.heads * PARTS * S):
            if cutlass.const_expr(self.ds):
                self.insert_ds(args, ss, 0, j, True, tid)
            elif cutlass.const_expr(self.nvfp4):
                self.insert_nvfp4(args, ss, 0, j, True, tid)
            else:
                self.insert(args, ss, 0, j, True, tid)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # B fragments: 4 latent n-tiles of 8 per warp, 4 k-chunks of 32 nope rows.
        b_frags = [
            [
                ldmatrix(
                    uk.iterator + (32 * kp + lane) * (L + 8) + 32 * warp + 8 * n,
                    trans=True,
                )
                for kp in range(4)
            ]
            for n in range(4)
        ]
        chunks: cutlass.Constexpr = L * E // 16  # 16 B chunks per row
        for tile in cutlass.range(split, self.tiles, S, unroll=1):
            tok0 = tile * T
            if tile > split:
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
            for m in cutlass.range_constexpr(MT):
                out = zeros((4, 4), Float32)
                t = 16 * m + lane % 8 + 8 * ((lane // 8) % 2)
                for ks in cutlass.range_constexpr(8):
                    p = 16 * ks + 8 * (lane // 16)
                    a = ldmatrix(qn.iterator + t * (128 + 8) + p)
                    for n in cutlass.range_constexpr(4):
                        b = b_frags[n][ks // 2][2 * (ks % 2) : 2 * (ks % 2) + 2]
                        mma_into(out[n, None], a, b, bf16=True)
                # Staged in shared memory (one pair per store): the rows leave as
                # 16 B stores.
                for n in cutlass.range_constexpr(4):
                    for h in cutlass.range_constexpr(2):  # rows group, group + 8
                        r = 16 * m + group + 8 * h
                        c = 32 * warp + 8 * n + 2 * quarter
                        lo, hi = out[n, 2 * h], out[n, 2 * h + 1]
                        if cutlass.const_expr(self.ds):
                            pairs[r, c // 2] = bf16x2(lo, hi)
                        else:
                            scaled = (Float32(BFloat16(v)) * qsi for v in (lo, hi))
                            word = pack_fp8((*scaled, Float32(0), Float32(0)))
                            pairs[r, c // 2] = Uint16(word & 0xFFFF)
            cute.arch.sync_threads()  # qn consumed, staged complete
            if tile + S < self.tiles:  # the next tile's rows load during the stores
                self.stage_rows(q_rows, qn, tok0 + S * T, head, tid)
            self.rope(q_rows, output, qsi, tok0, part, head, tid, args)
            for c in range(tid, T * chunks, 128):
                t, k = c // chunks, c % chunks
                if self.live(tok0 + t):
                    words = load_words(
                        staged.iterator + t * P + k * 16, count=4, shared=True
                    )
                    col = L * part + k * (16 // E)
                    store_words(at(output, tok0 + t, head, col), words)
                    self.publish_words(args, tok0 + t, head, col, words)
            cute.arch.sync_threads()  # staged consumed


# Rows from which the absorption of many local heads (TP1 / DEP: 96) runs as the
# stock decode's W_UK bmm over the heads (cuBLAS: FP32 accumulation, one BF16
# rounding) and MlaQuant (the per-tensor FP8 query, the rope columns and the KV
# insert), instead of MlaAbsorb's mma.sync tiles. GB300 TP1 whole split query
# (QueryNorm, Q_B, absorption), us: 40 rows 21.3 / 21.3 (bmm / MlaAbsorb), 80
# rows 25.8 / 25.5, 320 rows 51.5 / 53.2 (MlaQuant 12.1 + the bmm, against
# MlaAbsorb 21.1).
QUANT_FROM = 160
QUANT_HEADS = 64


def absorb_bmm(heads: int, batch: int) -> bool:
    """Whether `batch` rows of `heads` local heads absorb by the W_UK bmm."""
    return heads >= QUANT_HEADS and batch >= QUANT_FROM


class MlaQuant(MlaAbsorb):
    """The query rows from the W_UK bmm's BF16 latent [H, B, 512] (in MlaAbsorb's
    W_UK argument): the per-tensor FP8 query (latent and the Q_B rows' rope
    columns times 1 / q_scale, as MlaAbsorb), the MLA KV insert and, under DCP, the
    q mailbox publish. One 128-thread CTA per (head, split): thread i of the CTA
    converts 16 B chunks i, i + 128, .. of the head's B x 72 chunks (64 latent,
    8 rope); CTA (head, split) inserts tokens split * H + head, + H * S, .."""

    def __init__(self, heads: int, batch: int, **kwargs):
        super().__init__(heads, batch, **kwargs)
        assert not self.ds, "MlaQuant writes the FlashInfer (per-tensor FP8) query"
        # One round of four 16 B chunks per thread: B * 72 / 512 CTAs per head.
        self.splits = max(1, min(-(-batch * 72 // 512), 64))
        self.grid = (heads, self.splits, 1)

    @cute.jit
    def __call__(self, *args):
        launch(self.quant, args, self.grid, 128)

    @cute.kernel
    def quant(self, args):
        q_rows, latent, q_scale_inv, output = args[3], args[4], args[7], args[10]
        tid, _, _ = cute.arch.thread_idx()
        head, split, _ = cute.arch.block_idx()
        S: cutlass.Constexpr = self.splits
        smem = cutlass.utils.SmemAllocator()
        ss = alloc(smem, Float32, (8, 2))
        self.pdl_start()
        qsi = q_scale_inv[0]
        for j in range(split * self.heads + head, self.batch, self.heads * S):
            if cutlass.const_expr(self.nvfp4):
                self.insert_nvfp4(args, ss, 0, j, True, tid)
            else:
                self.insert(args, ss, 0, j, True, tid)
        per: cutlass.Constexpr = 4  # chunks in flight per thread
        total = self.batch * 72
        for c0 in range(split * 128 + tid, total, 128 * S * per):
            loads = []
            for i in cutlass.range_constexpr(per):
                c = min(c0 + i * 128 * S, total - 1)
                t, k = c // 72, c % 72
                words = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                if k < 64:
                    source = at(latent, head, t, 8 * k)
                    words = load_words(source, count=4, mutable=True)
                else:
                    rope = head * 192 + 128 + 8 * (k - 64)
                    words = load_words(at(q_rows, t, rope), count=4, mutable=True)
                loads.append(words)
            for i in cutlass.range_constexpr(per):
                c = c0 + i * 128 * S
                if c < total:
                    t, k = c // 72, c % 72
                    col = 8 * k  # latent 0..511, then the rope columns 512..575
                    values: list = []
                    for word in loads[i]:
                        values += [v * qsi for v in unpack_bf16(word)]
                    fp8 = (pack_fp8(values[0:4]), pack_fp8(values[4:8]))
                    store_words(at(output, t, head, col), fp8)
                    self.publish_words(args, t, head, col, fp8)
