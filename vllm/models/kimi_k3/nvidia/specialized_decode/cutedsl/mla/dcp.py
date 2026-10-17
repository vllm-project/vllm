# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA decode under decode context parallelism (DCP = TP, rank d owns head block d)
without collectives, three kernels per layer. Each rank's KV cache holds 1 / DCP of
every context.

  MlaQuery     FP8 query rows [B, H, 576] (+ the KV insert where this rank owns the
               slot); each CTA multicasts its bytes into every rank's Lamport q
               mailbox [B, DCP*H, 576], then polls its share until all landed
  tokenspeed   all DCP*H heads over the local KV shard -> BF16 rows + LSE (log2)
  DcpValue     re-arms the q mailbox; rows pushed to their head owner's Lamport
               mailbox (peer stores); the owner LSE-merges its H heads over the DCP
               sources, then W_UV and the gate (padding from the global lengths)
"""

import math
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake
from ..common.primitives import (
    alloc,
    at,
    bf16x2,
    bits_float,
    float_bits,
    launch,
    load_volatile,
    load_words,
    mma_into,
    no_neg_zero,
    pack_pair,
    published,
    sigmoid,
    store_relaxed,
    store_shared_words,
    store_words,
    unpack_bf16,
    warp_sum,
    zeros,
)

LSE_SENT = 0xFFFFFFFF  # a NaN pattern the merge never sends
# The LSE of an empty or invalid source: finite, so the merge never forms inf - inf.
EMPTY_LSE = -3.0e38
Q_SENT = 0xFFFFFFFF  # four FP8 0xFF bytes: never written (satfinite, canonical NaN)
PAIR_SENT = 0x80000000  # BF16 pair (+0, -0); sent pairs have no negative zero


class DcpBox:
    """Model-owned DCP state over ``group``, Lamport mailboxes (sentinel-filled,
    re-armed by their reader): q [2, CAP, DCP*H, 576] FP8 bytes (multicast stores),
    partials [2, DCP, CAP, H, 512] BF16 and LSE [2, DCP, CAP, H] FP32 (peer stores),
    and the local tokenspeed rows / LSE. Parity alternates by layer. ``interleave``:
    the KV cache's DCP interleave (global masking of spec verify rows)."""

    def __init__(self, group, device, size, rank, heads, cap=64, interleave=1):
        self.size, self.rank, self.heads, self.cap = size, rank, heads, cap
        self.interleave = interleave
        empty = symm_mem.empty
        self.q = empty((2, cap, size * heads, 576), dtype=torch.uint8, device=device)
        shape = (2, size, cap, heads)
        self.partial = empty((*shape, 512), dtype=torch.bfloat16, device=device)
        self.lse = empty(shape, dtype=torch.float32, device=device)
        tensors = (self.q, self.partial, self.lse)
        handles = [symm_mem.rendezvous(t, group) for t in tensors]
        self.q_mc = int(handles[0].multicast_ptr or 0)
        if not self.q_mc:
            raise RuntimeError("Kimi K3 specialized MLA DCP needs multicast memory")
        i64 = dict(dtype=torch.int64, device=device)
        self.peers = torch.tensor(handles[1].buffer_ptrs, **i64)
        self.lse_peers = torch.tensor(handles[2].buffer_ptrs, **i64)
        self.q.fill_(0xFF)
        self.partial.view(torch.int32).fill_(-0x80000000)
        self.lse.view(torch.int32).fill_(-1)
        rows = (cap, 1, size * heads)
        self.rows = torch.empty((*rows, 512), dtype=torch.bfloat16, device=device)
        self.rows_lse = torch.empty(rows, dtype=torch.float32, device=device)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)

    def mc(self, parity):
        """The q mailbox multicast address of ``parity``."""
        return Int64(self.q_mc + parity * self.q[0].numel())


class DcpValue:
    """The DCP tail of one MLA layer, one 512-thread CTA per (head, 16 requests).
    Every rank re-arms this parity's q mailbox (tokenspeed has read it) and pushes all
    its local tokenspeed rows (+ LSE) to their head owner's Lamport mailbox (empty
    local shards: zeros, LSE -inf). Warp r then LSE-weights request t0 + r of this
    head over the DCP sources (exp2(lse - max) / sum in FP32), the CTA merges the
    polled rows in FP32
    (one BF16 rounding, re-arming what it read) into shared memory, and W_UV (BF16
    MMA, FP32 accumulation) and the sigmoid gate follow as in MlaValue: warp w owns
    outputs 32 (w // 4) + 8 (w % 4) .. + 7. Rows are tokens, ``q`` per request."""

    def __init__(self, heads, size, batch, sms=148, q=1):
        self.heads, self.size, self.batch = heads, size, batch
        self.q = q
        # Push-only CTAs (z > 0) spread the NVLink stores over more SMs: up to one
        # wave, at least 4 16 B chunks per thread.
        merge = heads * ((batch + 15) // 16)
        chunks = batch * size * heads * 64
        pushers = min(max(1, sms // merge), -(-chunks // (merge * 512 * 4)))
        self.grid = (heads, (batch + 15) // 16, pushers)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 512)

    @cute.kernel
    def kernel(self, args):
        rows, rows_lse, lens, box, lbox, peers, lpeers, mailbox = args[0:8]
        w_uv, gate, totals, output, parity, rank = args[8:14]
        H, D, B, cap = self.heads, self.size, self.batch, box.shape[1]
        smem = cutlass.utils.SmemAllocator()
        latent = alloc(smem, BFloat16, (16, 544), 128)
        scales = alloc(smem, Float32, (16, D))
        tid, _, _ = cute.arch.thread_idx()
        head, tile, pusher = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        v = 32 * (warp // 4) + 8 * (warp % 4) + group
        # B fragments: 8 latent values per 32-wide chunk; A uses the same permutation.
        wb = [
            load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
            for c in range(16)
        ]
        cute.arch.griddepcontrol_wait()
        threads = H * self.grid[1] * self.grid[2] * 512
        me = ((pusher * self.grid[1] + tile) * H + head) * 512 + tid
        for c in range(me, B * D * H * 36, threads):  # 16 B q mailbox chunks
            store_words(mailbox.iterator + c * 16, (Uint32(Q_SENT),) * 4)
        for i in range(me, B * D * H * 64, threads):
            r, k = i // 64, i % 64  # r = b * D * H + global head
            b, gh = r // (D * H), r % (D * H)
            words = (Uint32(0),) * 4
            if lens[b // self.q] > 0:
                words = load_words(at(rows, b, 0, gh, 8 * k), count=4, mutable=True)
            words = tuple(no_neg_zero(w) for w in words)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            store_relaxed(peers[gh // H] + (Int64(slot) * 512 + 8 * k) * 2, words)
        for i in range(me, B * D * H, threads):
            b, gh = i // (D * H), i % (D * H)
            value, n = rows_lse[b, 0, gh], lens[b // self.q]
            bad = (value != value) | (value <= Float32(EMPTY_LSE))  # NaN, -inf
            if (n <= 0) | bad | (value >= Float32(-EMPTY_LSE)):  # or +inf
                value = Float32(EMPTY_LSE)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            store_relaxed(lpeers[gh // H] + Int64(slot) * 4, (float_bits(value),))
        cute.arch.griddepcontrol_launch_dependents()
        if pusher == 0:
            self.merge(args, latent, scales, wb, tid)

    @cute.jit
    def merge(self, args, latent, scales, wb, tid):
        """The owner's merge of request rows t0.. of this head, then W_UV + gate."""
        box, lbox, gate, totals, output = args[3], args[4], args[9], args[10], args[11]
        D, B = self.size, self.batch
        head, tile, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        t0 = tile * 16
        col = head * 128 + 32 * (warp // 4) + 8 * (warp % 4) + 2 * quarter
        gates = []
        for r in cutlass.range_constexpr(2):
            word = Uint32(0)
            if t0 + group + 8 * r < B:
                word = load_words(
                    at(gate, t0 + group + 8 * r, col), count=1, mutable=True
                )[0]
            gates.append(unpack_bf16(word))
        if t0 + warp < B:  # warp r: the source weights of request t0 + r
            lse = Float32(EMPTY_LSE)
            if lane < D:
                source = at(lbox, lane, t0 + warp, head)
                word = load_volatile(source)[0]
                while word == Uint32(LSE_SENT):
                    word = load_volatile(source)[0]
                store_words(source, (Uint32(LSE_SENT),))
                lse = bits_float(word)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = warp_sum(weight)
            scale = Float32(0)
            if total > 0:
                scale = weight / total
            if lane < D:
                scales[warp, lane] = scale
        cute.arch.sync_threads()
        chunk = tid % 64
        for half in cutlass.range_constexpr(2):  # rows r and r + 8, 16 B chunk
            r = tid // 64 + 8 * half
            merged = zeros(8, Float32)
            if t0 + r < B:
                for s0 in cutlass.range_constexpr(0, D, 8):  # 8 polls in flight
                    sources = [
                        at(box, s, t0 + r, head, 8 * chunk)
                        for s in range(s0, min(D, s0 + 8))
                    ]
                    loads = [load_volatile(source, count=4) for source in sources]
                    for i in cutlass.range_constexpr(len(sources)):
                        words = loads[i]
                        while not published(words):
                            words = load_volatile(sources[i], count=4)
                        store_words(sources[i], (Uint32(PAIR_SENT),) * 4)
                        w = scales[r, s0 + i]
                        for e in cutlass.range_constexpr(4):
                            lo, hi = unpack_bf16(words[e])
                            merged[2 * e] += w * lo
                            merged[2 * e + 1] += w * hi
            pairs = tuple(
                pack_pair(merged[2 * e], merged[2 * e + 1], policy=None)
                for e in range(4)
            )
            store_shared_words(latent.iterator + r * 544 + 8 * chunk, pairs)
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    latent.iterator + r * 544 + 32 * c + 8 * quarter,
                    count=4,
                    shared=True,
                )
                for r in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        acc = [parts[0, k] + parts[1, k] for k in range(4)]
        for r in cutlass.range_constexpr(2):
            t = t0 + group + 8 * r
            if t < B:
                # Padded rows may hold a NaN gate; keep inactive rows zero.
                lo, hi = Float32(0), Float32(0)
                if totals[t // self.q] > 0:
                    lo, hi = (
                        Float32(BFloat16(acc[2 * r + e])) * sigmoid(gates[r][e], False)
                        for e in range(2)
                    )
                store_words(at(output, t, col), (bf16x2(lo, hi),))


@cache
def compile_value(heads, size, batch, cap, uv, pitch, device, sms=148, q=1):
    """DcpValue of one batch (``q`` rows per request); ``uv``: the (shape, stride,
    dtype) of W_UV, ``pitch``: the projected rows' (the gate follows the 2112 query /
    KV columns)."""
    f32, i32 = torch.float32, torch.int32
    w_uv, gate = fake(*uv), fake((batch, heads * 128), (pitch, 1))
    rows = fake((batch, 1, size * heads, 512))
    rows_lse = fake((batch, 1, size * heads), None, f32, 4)
    lens = fake((batch // q,), None, i32, 4)
    box = fake((size, cap, heads, 512))
    lbox = fake((size, cap, heads), None, f32, 4)
    peers = fake((size,), None, torch.int64, 8)
    mailbox = fake((cap, size * heads, 576), None, torch.uint8)
    output = fake((batch, heads * 128), (heads * 128, 1))
    args: tuple = (rows, rows_lse, lens, box, lbox, peers, peers, mailbox, w_uv)
    args += (gate, lens, output, Int32(0), Int32(0), make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(DcpValue(heads, size, batch, sms, q), *args)


_WORKSPACE: dict[tuple[torch.device, int], torch.Tensor] = {}


def stock_decode(
    query,
    cache,
    block_table,
    lengths,
    max_seq,
    scales,
    out,
    lse=None,
    min_split=1,
    pdl=True,
    cp=(1, 0, 1),
    totals=None,
    pack=False,
):
    """tokenspeed_mla_decode's FP8 path with caller-owned ``out`` and ``lse`` (log2)
    buffers and no per-call allocation: the split count follows from (B, max_seq,
    min_split); ``lengths`` bound the work, empty splits are skipped by its reducer.
    The q rows of a request mask causally; under DCP in global positions (``cp``:
    world, rank, interleave; ``totals``: the global lengths). ``pack``: M128 tiles hold
    consecutive (query row, head) rows across the q rows."""
    import tvm_ffi
    from tokenspeed_mla import mla_decode as md

    batch, q_len, heads, _ = query.shape
    fp8, device = torch.float8_e4m3fn, query.device
    arch = torch.cuda.get_device_capability(device)
    tiler, _ = md.select_mla_decode_tilers(
        heads, q_len, is_fp8=True, compute_capability=arch
    )
    pack = pack and tiler[0] == 128
    fold = 1 if pack else md.get_mla_decode_fold_sq_factor(heads, q_len, tiler[0])
    rows = (q_len // fold, heads * fold)
    if pack:
        rows = (md.compute_q_tile_layout(heads, q_len, 128)[1], 128)
    sms = md.get_num_sm(device)
    split, size = md._get_split_kv_and_workspace_size(
        batch, *rows, 512, sms, max_seq, fp8, tiler, min_split
    )
    # One split-KV workspace per (device, heads), shared by every MLA layer: the calls
    # are stream ordered. PreparedMla.prepare_cache runs each prepared (batch, q) before
    # any capture, so the buffer reaches its final size first; growing it during a
    # capture would leave the graphs captured before with the freed buffer.
    key = device, heads
    if key not in _WORKSPACE or _WORKSPACE[key].numel() < size:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the MLA split-KV workspace must be sized before capture"
            )
        bound = max(size, sms * heads * 513 * 4)
        _WORKSPACE[key] = torch.empty(bound, dtype=torch.int8, device=device)
    kernel = md._get_compiled_mla_kernel(
        torch_dtype=fp8,
        page_size=cache.shape[1],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
        compute_capability=arch,
        has_local_visible_lens=False,
        skip_correction_threshold=8.0,
        is_workspace_size_zero=size == 0,
        fold_sq_factor=fold,
        causal_mask=True,
        num_heads=heads,
        seq_len_q=q_len,
        window_left=-1,
        cp_world=cp[0],
        cp_rank=cp[1],
        cp_interleave_size=cp[2],
        use_pdl=pdl,
        return_lse=lse is not None,
        reducer_d_tiles=md._get_reducer_d_tiles(batch, q_len, heads, sms, split),
        reducer_max_splits=md._get_reducer_max_splits(split),
        pack_q=pack,
        partial_fp16=False,
    )
    cache = cache.view(fp8)
    softmax, output_scale = scales
    args = [query[..., :512], query[..., 512:], cache[..., :512], cache[..., 512:]]
    args += [block_table, out, lse, _WORKSPACE[key][:size] if size else None]
    args += [Int32(split), lengths, lengths if totals is None else totals, None]
    args += [Float32(softmax), Float32(output_scale)]
    with tvm_ffi.use_torch_stream():
        kernel(*args)


def fill_splits(batch, device):
    """Splits filling the SMs (one CTA pair each); tokenspeed caps itself at 32."""
    from tokenspeed_mla import get_num_sm

    return min(256, max(1, get_num_sm(device) // (2 * batch)))


def local_max(max_context, size):
    """Longest local shard of a ``max_context`` sequence (block-interleaved)."""
    return math.ceil(max_context / (size * 64)) * 64
