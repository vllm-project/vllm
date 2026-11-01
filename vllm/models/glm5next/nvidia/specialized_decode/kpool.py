# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kpool indexer's decode kernels in CuTe DSL, bit for bit with the stock
Triton kernels of ``ops/kpool_compress.py`` and ``as_fp32_logits``.

``kpool_update`` is the decode tail stash plus pool compression: one warp per
request walks its tokens in position order (a completing token reads the
stashes of the earlier ones), four dimensions per lane. The softmax, the
Hadamard butterflies (strides 1 and 2 in a lane, 4..64 across lanes, as in
``indexer_query``), the BF16 roundings, the accurate log2 and the UE8M0 scale
follow the stock kernel's PTX: ``ex2.approx`` of ``(s - max) * log2(e)``, FMA
accumulation in pool order and ``div.full`` divisions.

``upcast_logits`` is ``as_fp32_logits`` with 16-byte accesses.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from vllm.forward_context import in_piecewise_cudagraph
from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    expand_pools_and_append_tail,
    kpool_decode_update_and_maybe_write_cache_batched,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    asm,
    bits_float,
    float_bits,
    load_words,
    pack_fp8,
    store_words,
    unpack_bf16,
)
from vllm.utils.deep_gemm import mqa_logits_use_mx_scales
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroup

from .linear import planned
from .shared_expert import _round_bf16

DIM = 128
FP8_MAX = 448.0
INV_SQRT_DIM = 0.08838834764831845  # 1 / sqrt(128), exact in FP32
LOG2E = 1.4426950408889634


@asm(Float32, pure=True)
def _exp(score, peak):
    """Triton's ``tl.exp(score - peak)``: ``ex2.approx`` of the product with
    log2(e), each step rounded on its own."""
    code = "{ .reg .f32 d; sub.f32 d,$1,$2; mul.f32 d,d,0f3FB8AA3B; "
    return code + "ex2.approx.f32 $0,d; }", "=f,f,f", (score, peak)


@asm(Float32, pure=True)
def _fma(a, b, c):
    return "fma.rn.f32 $0,$1,$2,$3;", "=f,f,f,f", (a, b, c)


@asm(Float32, pure=True)
def _add(a, b):
    return "add.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def _div(a, b):
    """Triton's FP32 ``/``."""
    return "div.full.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def _ceil(value):
    return "cvt.rpi.ftz.f32.f32 $0, $1;", "=f,f", (value,)


def _load_bf16x4(pointer):
    return load_words(pointer, count=2, word_type="u32", mutable=True)


def _bf16x4(words):
    lo, hi = unpack_bf16(words[0]), unpack_bf16(words[1])
    return [lo[0], lo[1], hi[0], hi[1]]


class _UpdateKernel:
    """One warp per request; lane ``l`` owns dimensions ``4l .. 4l + 3``."""

    def __init__(self, pool, ring, page, page_bytes, tail_block, tail_half):
        self.pool, self.ring, self.page = pool, ring, page
        self.page_bytes, self.tail_block, self.tail_half = (
            page_bytes,
            tail_block,
            tail_half,
        )

    @cute.jit
    def __call__(
        self, cache: cute.Tensor, tail: cute.Tensor, tail_slots: cute.Tensor,
        key: cute.Tensor, score: cute.Tensor, ape: cute.Tensor, slots: cute.Tensor,
        positions: cute.Tensor, requests: Int32, tokens: Int32, key_req: Int32,
        key_tok: Int32, score_req: Int32, score_tok: Int32, st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            cache,
            tail,
            tail_slots,
            key,
            score,
            ape,
            slots,
            positions,
            tokens,
            key_req,
            key_tok,
            score_req,
            score_tok,
        ).launch(grid=(requests, 1, 1), block=(32, 1, 1), stream=st, use_pdl=True)

    @cute.kernel
    def kernel(
        self, cache: cute.Tensor, tail: cute.Tensor, tail_slots: cute.Tensor,
        key: cute.Tensor, score: cute.Tensor, ape: cute.Tensor, slots: cute.Tensor,
        positions: cute.Tensor, tokens: Int32, key_req: Int32, key_tok: Int32,
        score_req: Int32, score_tok: Int32,
    ):  # fmt: skip
        req, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        dim = lane * 4
        # APE is a weight: read it before waiting on the producers.
        bias = [[ape[p * DIM + dim + e] for e in range(4)] for p in range(self.pool)]
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for t in cutlass.range(tokens, unroll=1):
            idx = req * tokens + t
            loc = slots[idx]
            pos = Int32(positions[idx])
            tail_slot = tail_slots[idx]
            safe_pos = max(pos, Int32(0))
            slot = safe_pos % self.pool
            phys_slot = safe_pos % self.ring
            block = Int64(max(tail_slot, Int32(0)) // self.ring)
            base = tail.iterator + block * self.tail_block
            key_words = _load_bf16x4(key.iterator + req * key_req + t * key_tok + dim)
            score_words = _load_bf16x4(
                score.iterator + req * score_req + t * score_tok + dim
            )
            if loc >= 0 and pos >= 0 and slot == self.pool - 1:
                self.compress(
                    cache, base, bias, _bf16x4(key_words), _bf16x4(score_words),
                    safe_pos - slot, loc, lane, dim,
                )  # fmt: skip
            if pos >= 0 and tail_slot >= 0:
                row = base + phys_slot * DIM + dim
                store_words(row, key_words)
                store_words(row + self.tail_half, score_words)

    @cute.jit
    def compress(
        self, cache, base, bias, current_key, current_score, start, loc, lane, dim
    ):
        scores = []
        keys = []
        for p in cutlass.range_constexpr(self.pool):
            if cutlass.const_expr(p == self.pool - 1):
                s, k = current_score, current_key
            else:
                row = base + ((start + p) % self.ring) * DIM + dim
                s = _bf16x4(_load_bf16x4(row + self.tail_half))
                k = _bf16x4(_load_bf16x4(row))
            scores.append([_add(s[e], bias[p][e]) for e in range(4)])
            keys.append(k)
        x = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        for e in cutlass.range_constexpr(4):
            peak = Float32(-float("inf"))
            for p in cutlass.range_constexpr(self.pool):
                peak = cute.arch.fmax(peak, scores[p][e])
            denom = Float32(0.0)
            acc = Float32(0.0)
            for p in cutlass.range_constexpr(self.pool):
                prob = _exp(scores[p][e], peak)
                denom = _add(denom, prob)
                acc = _fma(prob, keys[p][e], acc)
            x[e] = _round_bf16(_div(acc, denom))
        # Strides 1 and 2 inside the lane: (a, b) -> (a + b, a - b).
        for stride in (1, 2):
            for i in cutlass.range_constexpr(4):
                if cutlass.const_expr(i % (2 * stride) < stride):
                    a, b = x[i], x[i + stride]
                    x[i], x[i + stride] = a + b, a - b
        # Strides 4..64 pair lanes l and l ^ (stride / 4); the lower lane holds a.
        for shift in cutlass.range_constexpr(5):
            upper = (lane >> shift) & 1
            for i in cutlass.range_constexpr(4):
                other = cute.arch.shuffle_sync_bfly(x[i], offset=1 << shift)
                if upper == 1:
                    x[i] = other - x[i]
                else:
                    x[i] = x[i] + other
        amax = Float32(0.0)
        for i in cutlass.range_constexpr(4):
            x[i] = _round_bf16(x[i] * Float32(INV_SQRT_DIM))
            amax = cute.arch.fmax(amax, cute.arch.fmax(x[i], -x[i]))
        for shift in cutlass.range_constexpr(5):
            amax = cute.arch.fmax(
                amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift)
            )
        amax = cute.arch.fmax(amax, Float32(1e-4))
        exponent = Int32(
            _ceil(cute.math.log2(amax * Float32(1.0 / FP8_MAX), fastmath=False))
        )
        scale = bits_float(Uint32(exponent + 127) << 23)
        y = [
            cute.arch.fmin(
                cute.arch.fmax(_div(x[i], scale), Float32(-FP8_MAX)), Float32(FP8_MAX)
            )
            for i in range(4)
        ]
        loc64 = Int64(loc)
        page_base = cache.iterator + (loc64 // self.page) * self.page_bytes
        in_page = loc64 % self.page
        store_words(page_base + in_page * DIM + dim, (pack_fp8(tuple(y)),))
        if lane == 0:
            store_words(
                page_base + self.page * DIM + in_page * 4,
                (Uint32(exponent + 127) * Uint32(0x01010101),),
            )


_compiled: dict = {}


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1)


def kpool_update(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
) -> None:
    """``kpool_decode_update_and_maybe_write_cache_batched`` with power-of-two
    UE8M0 (MX) scales; the same arguments, ``head_dim`` 128."""
    requests, tokens = key.shape[0], key.shape[1]
    if requests == 0 or tokens == 0:
        return
    ring = tail_kv_cache.shape[2]
    assert ring % pool_size == 0 and tail_kv_cache.shape[3] == DIM
    assert key.shape[2] == DIM and key.stride(2) == 1 and slot_score.stride(2) == 1
    assert kv_cache.dtype == torch.uint8 and kv_cache.stride(0) % 4 == 0
    assert tail_kv_cache.stride(3) == 1 and tail_kv_cache.stride(2) == DIM
    page = kv_cache.shape[1]
    args = (
        kv_cache, tail_kv_cache, _flat(tail_slot_mapping), key, slot_score,
        _flat(ape), _flat(slot_mapping), _flat(positions),
    )  # fmt: skip
    scalars = (
        Int32(requests), Int32(tokens), Int32(key.stride(0)), Int32(key.stride(1)),
        Int32(slot_score.stride(0)), Int32(slot_score.stride(1)),
    )  # fmt: skip
    st = stream(key.device)
    config = (
        pool_size, ring, page, kv_cache.stride(0), tail_kv_cache.stride(0),
        tail_kv_cache.stride(1),
    )  # fmt: skip
    compile_key = (config, positions.dtype, key.device.index)
    cute_args = [to_cute(t, 8) for t in args]
    if compile_key not in _compiled:
        _compiled[compile_key] = cute.compile(
            _UpdateKernel(*config), *cute_args, *scalars, st
        )
    _compiled[compile_key](*cute_args, *scalars, st)


class _UpcastKernel:
    """CTA ``(row, split)`` converts the row's 8-element chunks ``split``,
    ``split + splits``, ... below the row end; one chunk per thread."""

    def __init__(self, threads: int, pdl: bool = True):
        self.threads, self.pdl = threads, pdl

    @cute.jit
    def __call__(
        self, src: cute.Tensor, dst: cute.Tensor, ends: cute.Tensor, rows: Int32,
        splits: Int32, stride: Int32, st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(src, dst, ends, stride).launch(
            grid=(rows, splits, 1), block=(self.threads, 1, 1), stream=st,
            use_pdl=self.pdl,
        )  # fmt: skip

    @cute.kernel
    def kernel(self, src: cute.Tensor, dst: cute.Tensor, ends: cute.Tensor, stride):
        row, split, _ = cute.arch.block_idx()
        _, splits, _ = cute.arch.grid_dim()
        tid, _, _ = cute.arch.thread_idx()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        end = ends[row]
        offset = Int64(row) * stride
        step = splits * self.threads * 8
        for first in cutlass.range((split * self.threads + tid) * 8, end, step):
            if first + 8 <= end:
                words = load_words(
                    src.iterator + offset + first, count=4, word_type="u32",
                    mutable=True,
                )  # fmt: skip
                values = []
                for w in cutlass.range_constexpr(4):
                    lo, hi = unpack_bf16(words[w])
                    values += [Uint32(float_bits(lo)), Uint32(float_bits(hi))]
                store_words(dst.iterator + offset + first, values[:4])
                store_words(dst.iterator + offset + first + 4, values[4:])
            else:
                for e in cutlass.range(first, end, unroll=1):
                    dst[offset + e] = Float32(src[offset + e])


UPCAST_THREADS = 128


def upcast_logits(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    threads=UPCAST_THREADS,
    pdl=True,
    cap=1024,
) -> torch.Tensor:
    """``as_fp32_logits`` for BF16 logits whose row stride is 16-byte aligned."""
    out = torch.empty_strided(
        logits.shape, logits.stride(), dtype=torch.float32, device=logits.device
    )
    rows, width = logits.shape
    if rows == 0 or width == 0:
        return out
    assert logits.dtype == torch.bfloat16 and row_ends.dtype == torch.int32
    stride = logits.stride(0)
    assert logits.stride(1) == 1 and stride % 8 == 0
    chunk = threads * 8
    splits = min((width + chunk - 1) // chunk, max(1, cap // rows))
    numel = (rows - 1) * stride + width
    flat = (logits.as_strided((numel,), (1,)), out.as_strided((numel,), (1,)))
    args = [to_cute(t, 16) for t in flat] + [to_cute(row_ends.contiguous(), 4)]
    scalars = (Int32(rows), Int32(splits), Int32(stride))
    st = stream(logits.device)
    key = ("upcast", threads, pdl, logits.device.index)
    if key not in _compiled:
        _compiled[key] = cute.compile(_UpcastKernel(threads, pdl), *args, *scalars, st)
    _compiled[key](*args, *scalars, st)
    return out


class _ExpandKernel:
    """Thread ``g`` of row ``r`` writes the four token ids of pool
    ``pool_ids[r, g]`` with one 16-byte store; CTA 0's first ``pool - 1``
    threads write the trailing incomplete pool."""

    def __init__(self, pool: int, groups: int, threads: int, pdl: bool):
        assert pool == 4
        self.pool, self.groups, self.threads, self.pdl = pool, groups, threads, pdl

    @cute.jit
    def __call__(
        self, pool_ids: cute.Tensor, positions: cute.Tensor, out: cute.Tensor,
        rows: Int32, ids_stride: Int32, out_stride: Int32, seq_offset: Int32,
        st: cuda.CUstream,
    ):  # fmt: skip
        ctas = (self.groups + self.threads - 1) // self.threads
        args = (pool_ids, positions, out, ids_stride, out_stride, seq_offset)
        self.kernel(*args).launch(
            grid=(rows, ctas, 1), block=(self.threads, 1, 1), stream=st,
            use_pdl=self.pdl,
        )  # fmt: skip

    @cute.kernel
    def kernel(self, pool_ids, positions, out, ids_stride, out_stride, seq_offset):
        row, cta, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        g = cta * self.threads + tid
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        base = out.iterator + Int64(row) * out_stride
        if g < self.groups:
            pid = pool_ids[row * ids_stride + g]
            first = pid * self.pool
            words = [Uint32(first + o) for o in range(self.pool)]
            if pid < 0:
                words = [Uint32(0xFFFFFFFF)] * self.pool
            store_words(base + g * self.pool, words)
        if cta == 0 and tid < self.pool - 1:
            seq_len = Int32(positions[row]) + seq_offset
            tail_start = (seq_len // self.pool) * self.pool
            value = Int32(-1)
            if tid < seq_len - tail_start:
                value = tail_start + tid
            out[Int64(row) * out_stride + self.groups * self.pool + tid] = value


EXPAND_THREADS = 128


def expand_pools(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    *,
    seq_offset: int = 0,
    out: torch.Tensor,
    threads: int = EXPAND_THREADS,
    pdl: bool = True,
) -> torch.Tensor:
    """``expand_pools_and_append_tail`` into ``out`` for int32 pool ids."""
    rows, groups = pool_ids.shape
    out_cols = groups * pool_size + pool_size - 1
    assert pool_ids.dtype == torch.int32 and pool_ids.stride(1) == 1
    assert out.dtype == torch.int32 and out.stride(1) == 1 and out.shape[1] >= out_cols
    assert out.stride(0) % 4 == 0 and out.data_ptr() % 16 == 0
    assert seq_lens.ndim == 1 and seq_lens.stride(0) == 1
    if rows:
        out_flat = out.as_strided(((rows - 1) * out.stride(0) + out_cols,), (1,))
        ids_flat = pool_ids.as_strided(
            ((rows - 1) * pool_ids.stride(0) + groups,), (1,)
        )
        args = [to_cute(ids_flat, 4), to_cute(seq_lens, 4), to_cute(out_flat, 16)]
        scalars = (
            Int32(rows), Int32(pool_ids.stride(0)), Int32(out.stride(0)),
            Int32(seq_offset),
        )  # fmt: skip
        st = stream(out.device)
        key = (
            "expand",
            pool_size,
            groups,
            threads,
            pdl,
            seq_lens.dtype,
            out.device.index,
        )
        if key not in _compiled:
            _compiled[key] = cute.compile(
                _ExpandKernel(pool_size, groups, threads, pdl), *args, *scalars, st
            )
        _compiled[key](*args, *scalars, st)
    return out[:rows, :out_cols]


class _RemapKernel:
    """``expand_pools_and_append_tail`` and the sparse MLA index remap with
    COMPACT_TO_FRONT in one CTA per query row.

    Thread ``g`` maps pool ``pool_ids[row, g]``: its tokens share one KV block,
    so the pool is valid or invalid as a whole and every compacted offset is a
    multiple of 4 (one 16-byte store). The column order of the stock compaction
    is kept: history pools by rank, then the trailing incomplete pool, then -1.
    Rows from ``expanded`` on hold no indices (the stock path filled them with
    -1).
    """

    def __init__(self, pool, groups, width, block_size, stride_rows):
        assert pool == 4 and block_size % pool == 0 and groups % 32 == 0
        assert groups * pool + pool - 1 <= width
        self.pool, self.groups, self.width = pool, groups, width
        self.block_size, self.stride_rows = block_size, stride_rows
        self.warps = groups // 32

    @cute.jit
    def __call__(
        self, pool_ids: cute.Tensor, positions: cute.Tensor, req_ids: cute.Tensor,
        block_table: cute.Tensor, out: cute.Tensor, counts: cute.Tensor,
        rows: Int32, expanded: Int32, ids_stride: Int32, bt_stride: Int32,
        max_blocks: Int32, seq_offset: Int32, st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            pool_ids,
            positions,
            req_ids,
            block_table,
            out,
            counts,
            expanded,
            ids_stride,
            bt_stride,
            max_blocks,
            seq_offset,
        ).launch(grid=(rows, 1, 1), block=(self.groups, 1, 1), stream=st, use_pdl=True)

    @cute.kernel
    def kernel(
        self, pool_ids, positions, req_ids, block_table, out, counts, expanded,
        ids_stride, bt_stride, max_blocks, seq_offset,
    ):  # fmt: skip
        row, _, _ = cute.arch.block_idx()
        g, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        smem = cutlass.memory.SmemAllocator()
        warp_counts = smem.allocate_array(Int32, self.warps + 1)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        req = req_ids[row]
        live = Int32(0)
        first = Int32(0)
        if row < expanded:
            pid = pool_ids[row * ids_stride + g]
            token = pid * self.pool
            block = token // self.block_size
            if pid >= 0 and block < max_blocks:
                live = Int32(1)
                first = (
                    block_table[req * bt_stride + block] * self.stride_rows
                    + token % self.block_size
                )
        ballot = cute.arch.vote_ballot_sync(live == 1)
        below = cute.arch.popc(ballot & ((Uint32(1) << lane) - Uint32(1)))
        if lane == 0:
            warp_counts[warp] = Int32(cute.arch.popc(ballot))
        cute.arch.barrier()
        prior = Int32(0)
        for w in cutlass.range(warp, unroll=1):
            prior += warp_counts[w]
        row_base = out.iterator + Int64(row) * self.width
        if live == 1:
            offset = (prior + Int32(below)) * self.pool
            store_words(
                row_base + offset,
                [Uint32(first + o) for o in range(self.pool)],
            )
        if g == self.groups - 1:
            # The last thread knows the history total; it appends the tail.
            total = (prior + Int32(cute.arch.popc(ballot))) * self.pool
            if row < expanded:
                seq_len = Int32(positions[row]) + seq_offset
                tail_start = (seq_len // self.pool) * self.pool
                for j in cutlass.range_constexpr(self.pool - 1):
                    token = tail_start + j
                    block = token // self.block_size
                    if j < seq_len - tail_start and block < max_blocks:
                        out[Int64(row) * self.width + total] = (
                            block_table[req * bt_stride + block] * self.stride_rows
                            + token % self.block_size
                        )
                        total += 1
            counts[row] = total
            warp_counts[self.warps] = total
        cute.arch.barrier()
        valid = warp_counts[self.warps]
        for c in cutlass.range(valid + g, self.width, self.groups, unroll=1):
            out[Int64(row) * self.width + c] = Int32(-1)


def expand_and_remap(
    pool_ids: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    *,
    block_size: int,
    stride_rows: int,
    seq_offset: int = 1,
) -> None:
    """``triton_convert_req_index_to_global_index(..., return_valid_counts=True)``
    of the ``expand_pools_and_append_tail`` rows of ``pool_ids``, the rest of the
    ``out.shape[0]`` rows read as -1. Writes ``out`` and ``counts``."""
    expanded, groups = pool_ids.shape
    rows, width = out.shape
    assert rows == req_ids.shape[0] and counts.shape[0] >= rows
    assert pool_ids.dtype == req_ids.dtype == block_table.dtype == torch.int32
    assert out.dtype == counts.dtype == torch.int32 and out.is_contiguous()
    assert out.data_ptr() % 16 == 0 and width % 4 == 0
    assert pool_ids.stride(1) == 1 and block_table.stride(1) == 1
    assert positions.ndim == 1 and positions.stride(0) == 1
    if rows == 0:
        return
    ids_flat = pool_ids.as_strided(
        (max(expanded - 1, 0) * pool_ids.stride(0) + groups,), (1,)
    )
    bt = block_table.as_strided(
        ((block_table.shape[0] - 1) * block_table.stride(0) + block_table.shape[1],),
        (1,),
    )
    args = [
        to_cute(ids_flat, 4), to_cute(positions, 4), to_cute(req_ids, 4),
        to_cute(bt, 4), to_cute(out.view(-1), 16), to_cute(counts, 4),
    ]  # fmt: skip
    scalars = (
        Int32(rows), Int32(expanded), Int32(pool_ids.stride(0)),
        Int32(block_table.stride(0)), Int32(block_table.shape[1]), Int32(seq_offset),
    )  # fmt: skip
    st = stream(out.device)
    config = (pool_size, groups, width, block_size, stride_rows)
    key = ("remap", config, positions.dtype, out.device.index)
    if key not in _compiled:
        _compiled[key] = cute.compile(_RemapKernel(*config), *args, *scalars, st)
    _compiled[key](*args, *scalars, st)


class FusedRemap:
    """A sparse MLA index group's conversion that expands a deferred pool top-k
    and remaps it in one kernel, on the group's side stream as the stock
    conversion. Installed as the group's ``_convert_once``; without a deferred
    top-k it runs the stock conversion."""

    def __init__(self, group, stage: "KpoolStage"):
        self.group, self.stage = group, stage
        self.stock = group._convert_once
        group._convert_once = self

    def __call__(
        self, layer_index, logical, req_ids, block_table, block_size, *,
        block_stride_rows, return_valid_counts,
    ):  # fmt: skip
        pending, self.stage.pending = self.stage.pending, None
        kwargs = dict(
            block_stride_rows=block_stride_rows,
            return_valid_counts=return_valid_counts,
        )
        if pending is None:
            return self.stock(
                layer_index, logical, req_ids, block_table, block_size, **kwargs
            )
        group = self.group
        rows = logical.shape[0]
        if (
            layer_index != 0
            or not return_valid_counts
            or rows > group.physical_topk_indices.shape[0]
        ):
            self.stage.materialize(pending, logical)
            return self.stock(
                layer_index, logical, req_ids, block_table, block_size, **kwargs
            )
        out = group.physical_topk_indices[:rows]
        counts = group.valid_topk_counts[:rows]
        compute_stream = current_stream()
        if not in_piecewise_cudagraph():
            group.side_stream.wait_event(group.logical_topk_ready)
        else:
            group.side_stream.wait_stream(compute_stream)
        pool_ids, positions, pool_size = pending
        with group.side_stream:
            expand_and_remap(
                pool_ids, positions, pool_size, req_ids, block_table, out, counts,
                block_size=block_size,
                stride_rows=block_stride_rows or block_size,
            )  # fmt: skip
            group.physical_topk_ready.record(group.side_stream)
        compute_stream.wait_event(group.physical_topk_ready)
        return out, counts


class KpoolStage:
    """The kpool indexer decode kernels inside planned steps (stage
    ``dsa_kpool``), reached from the stock indexer through
    ``sparse_indexer.SPECIALIZED_KPOOL``.

    With ``fuse`` every DSA layer's index group converts through ``FusedRemap``:
    the indexer then leaves its uniform decode rows as pool ids (``defer``) and
    skips filling the logical top-k buffer, which no consumer reads.
    """

    def __init__(self, attentions, max_rows: int, device):
        self.pending: tuple[torch.Tensor, torch.Tensor, int] | None = None
        self.fuse = False
        # Indexer caches of the fused layers: others (the MTP draft's) never
        # reach a FusedRemap, so they must not defer.
        self.layers: set[str] = set()
        self.max_rows = max_rows
        self.pool_buffer = torch.empty(
            (max_rows, 2048), dtype=torch.int32, device=device
        )
        impls = [attn.mla_attn.mla_attn.impl for attn in attentions]
        groups = [getattr(impl, "index_group", None) for impl in impls]
        if all(
            type(group) is SparseMLAIndexGroup
            and group.has_indexer
            and group.num_layers == 1
            and not getattr(impl, "group_conversion_decode_only", True)
            and getattr(impl, "dcp_world_size", 1) == 1
            for impl, group in zip(impls, groups)
        ):
            self.fuse = True
            self.layers = {str(attn.indexer.k_cache.prefix) for attn in attentions}
            for group in groups:
                FusedRemap(group, self)

    @staticmethod
    def active() -> bool:
        return planned()

    def defers(self, layer, rows: int, topk: int, pool_size: int, width: int) -> bool:
        return (
            str(layer) in self.layers
            and pool_size == 4
            and rows <= self.max_rows
            and topk // pool_size <= self.pool_buffer.shape[1]
            and topk + pool_size - 1 <= width
            and (topk // pool_size) % 32 == 0
            and topk // pool_size <= 1024
        )

    def pool_ids(self, rows: int, groups: int) -> torch.Tensor:
        return self.pool_buffer.view(-1)[: rows * groups].view(rows, groups)

    def defer(self, pool_ids, positions, pool_size: int) -> None:
        """Leaves the expansion of ``pool_ids`` to the layer's index remap."""
        assert self.pending is None, "a deferred kpool top-k was not consumed"
        assert pool_ids.data_ptr() == self.pool_buffer.data_ptr()
        self.pending = (pool_ids, positions, pool_size)

    @staticmethod
    def materialize(pending, logical: torch.Tensor) -> None:
        """The logical top-k rows the stock indexer would have written."""
        pool_ids, positions, pool_size = pending
        logical.fill_(-1)
        expand_pools(pool_ids, positions, pool_size, seq_offset=1, out=logical)

    @staticmethod
    def update(
        kv_cache, tail_kv_cache, tail_slot_mapping, key, slot_score, ape,
        slot_mapping, positions, pool_size, head_dim=DIM, round_scale=True,
    ):  # fmt: skip
        if head_dim != DIM or not mqa_logits_use_mx_scales():
            return kpool_decode_update_and_maybe_write_cache_batched(
                kv_cache, tail_kv_cache, tail_slot_mapping, key, slot_score, ape,
                slot_mapping, positions, pool_size, head_dim, round_scale=round_scale,
            )  # fmt: skip
        kpool_update(
            kv_cache, tail_kv_cache, tail_slot_mapping, key, slot_score, ape,
            slot_mapping, positions, pool_size,
        )  # fmt: skip

    @staticmethod
    def upcast(logits: torch.Tensor, row_ends: torch.Tensor) -> torch.Tensor:
        if (
            logits.dtype != torch.bfloat16
            or logits.stride(1) != 1
            or logits.stride(0) % 8
            or logits.data_ptr() % 16
        ):
            return as_fp32_logits(logits, row_ends)
        return upcast_logits(logits, row_ends)

    @staticmethod
    def expand(pool_ids, seq_lens, pool_size, *, seq_offset=0, out=None):
        if (
            out is None
            or pool_size != 4
            or pool_ids.dtype != torch.int32
            or pool_ids.stride(1) != 1
            or out.stride(0) % 4
            or out.data_ptr() % 16
            or seq_lens.stride(0) != 1
        ):
            return expand_pools_and_append_tail(
                pool_ids, seq_lens, pool_size, seq_offset=seq_offset, out=out
            )
        return expand_pools(
            pool_ids, seq_lens, pool_size, seq_offset=seq_offset, out=out
        )
