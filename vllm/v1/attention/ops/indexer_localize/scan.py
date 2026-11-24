# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local decode scan, top-k and merge of the sparse-attention indexer.

Each locality domain's green-context stream runs the unchanged DeepGEMM
`fp8_fp4_paged_mqa_logits` over the indexer pages in its own memory (the
compact block table, context lengths and schedule `build.py` made for the
domain), with DeepGEMM's programmatic dependent launch off for these
launches. Then either:

* `localized_paged_mqa_topk`: each domain runs the top-k the stock selector
  picks (`cooperative_topk`, or `persistent_topk` above 64 rows) on its
  compact scores, and one kernel (`merge.cu`) merges the 2 x K candidates into
  the K request-local token positions: the same selected score multiset as the
  single-launch scan and top-k (ties may pick other tokens); or
* `localized_paged_mqa_logits`: one kernel joins both domains' compact scores
  into the full-length score array with the stock layout, bit-identical to
  the single-launch scan, for consumers that need the whole array.
"""

import ctypes
from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers import indexer_overlap
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.indexer_localize.build import (
    LocalizedDecodeMetadata,
    ensure_built,
)
from vllm.v1.attention.ops.indexer_localize.kernels import CudaKernel, get_kernel
from vllm.v1.attention.ops.indexer_localize.placement import find_buffer

logger = init_logger(__name__)


def _view_matches(loc: LocalizedDecodeMetadata, kv_cache: torch.Tensor) -> bool:
    """Whether the layer's cache view has the group's page geometry.

    The page count may differ from the builder's view (the extensible KV
    cache grows after capture and some models keep the view they first saw);
    page `p` starts at `data_ptr + p * stride` either way and block ids stay
    below the live page count, so only the rows, the stride and the
    allocation matter.
    """
    return (
        kv_cache.shape[1] == loc.page_rows
        and kv_cache.stride(0) * kv_cache.element_size() == loc.page_stride
        and find_buffer(kv_cache.data_ptr()) is not None
    )


def _scan(
    loc: LocalizedDecodeMetadata,
    domain: int,
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    max_model_len: int,
    logits_dtype: torch.dtype,
    indices: torch.Tensor | None,
) -> torch.Tensor:
    """One domain's DeepGEMM scan, on the current (the domain's) stream."""
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_num_sms,
        get_pdl,
        set_num_sms,
        set_pdl,
    )

    num_sms = get_num_sms()
    set_num_sms(loc.resources.sms[domain])
    pdl = get_pdl()
    if pdl:
        set_pdl(False)
    try:
        return fp8_fp4_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            loc.seq_lens[domain],
            loc.block_tables[domain],
            loc.schedules[domain],
            max_model_len=max_model_len,
            logits_dtype=logits_dtype,
            indices=indices,
        )
    finally:
        set_num_sms(num_sms)
        if pdl:
            set_pdl(True)


# ---------------------------------------------------------------------------
# Join of both domains' compact scores
# ---------------------------------------------------------------------------
@triton.jit
def _copy_pages(s0, off, m, drow, p, i, R: tl.constexpr):
    """out[p * R + i] = src0[off + i] for the pages of a tile; off is the
    per-page column, plus src1 - src0 for pages of domain 1."""
    v = tl.load(s0 + off[:, None] + i[None, :], mask=m[:, None])
    tl.store(
        drow + p[:, None] * R + i[None, :],
        v.to(drow.dtype.element_ty),
        mask=m[:, None],
    )


@triton.jit
def _join_kernel(
    src0_ptr,
    src1_ptr,
    src_stride,
    dst_ptr,
    dst_stride,
    row_ends_ptr,
    jmap_ptr,
    jmap_stride,
    num_rows,
    delta,  # src1 - src0 in elements
    R: tl.constexpr,
    PB: tl.constexpr,
    LV: tl.constexpr,  # log2 of the vector width in source elements
    ROWS_P2: tl.constexpr,
):
    """dst[row, p*R + i] = src_{dom(p)}[row, local_col(p) + i] for columns
    below the row end, whole pages in 16-byte vectors where aligned.

    Program `pid` copies an equal share of all rows' full pages laid end to
    end; the partial last page of each row is copied element by element. A
    tile whose source columns are not vector-aligned is copied element by
    element too.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    i = tl.arange(0, R)
    pb = tl.arange(0, PB)

    # Full pages: program pid copies global full pages [lo, hi).
    rows = tl.arange(0, ROWS_P2)
    ends = tl.load(row_ends_ptr + rows, mask=rows < num_rows, other=0)
    nfull = tl.maximum(ends, 0) // R
    cum = tl.cumsum(nfull, axis=0)  # inclusive prefix of full pages
    total = tl.sum(nfull, axis=0).to(tl.int64)
    lo = total * pid // nprog
    hi = total * (pid + 1) // nprog
    r = tl.sum((cum.to(tl.int64) <= lo).to(tl.int32), axis=0)  # row of page lo
    base = tl.sum(tl.where(rows < r, nfull, 0), axis=0).to(tl.int64)
    gp = lo
    while gp < hi:
        end_r = tl.load(row_ends_ptr + r)
        nf = (tl.maximum(end_r, 0) // R).to(tl.int64)
        p_lo = (gp - base).to(tl.int32)
        p_hi = tl.minimum(hi - base, nf).to(tl.int32)
        r64 = r.to(tl.int64)
        s0 = src0_ptr + r64 * src_stride
        drow = dst_ptr + r64 * dst_stride
        jrow = jmap_ptr + r64 * jmap_stride
        e_next = tl.load(jrow + p_lo + pb, mask=(p_lo + pb) < p_hi, other=0)
        for p0 in range(p_lo, p_hi, PB):
            p = p0 + pb
            mp = p < p_hi
            e = e_next
            e_next = tl.load(jrow + p + PB, mask=(p + PB) < p_hi, other=0)
            m = mp
            die = (e & 1) * delta  # keeps delta's divisibility, tl.where not
            # (e >> (LV + 1)) << LV equals e >> 1 when that is a multiple of
            # the vector, and the compiler can see it.
            bad = tl.max(tl.where(m, (e >> 1) & ((1 << LV) - 1), 0), axis=0)
            if bad == 0:
                _copy_pages(s0, die + ((e >> (LV + 1)) << LV), m, drow, p, i, R)
            else:
                _copy_pages(s0, die + (e >> 1), m, drow, p, i, R)
        gp = base + p_hi
        base = base + nf
        r = r + 1

    # The partial last page of rows pid, pid + nprog, ...
    for rr in range(pid, num_rows, nprog):
        end_r = tl.load(row_ends_ptr + rr)
        nf = tl.maximum(end_r, 0) // R
        rem = end_r - nf * R
        if rem > 0:
            r64 = rr.to(tl.int64)  # type: ignore[attr-defined]
            e = tl.load(jmap_ptr + r64 * jmap_stride + nf)
            m = i < rem
            sp = tl.where((e & 1) != 0, src1_ptr, src0_ptr) + r64 * src_stride
            v = tl.load(sp + (e >> 1) + i, mask=m)
            tl.store(
                dst_ptr + r64 * dst_stride + nf * R + i,
                v.to(dst_ptr.dtype.element_ty),
                mask=m,
            )


# Join geometry (tuned on Rubin): 6 programs per SM, 32 pages x 4 warps.
_JOIN_PROGRAMS_PER_SM = 6
_JOIN_PAGES_PER_TILE = 32
_JOIN_WARPS = 4


def localized_join_alloc(
    o0: torch.Tensor, want_fp32: Callable[[torch.Tensor], bool]
) -> torch.Tensor:
    """The full-length score array for these compact scores: DeepGEMM's
    shape and row stride, with storage for whole rows; FP32 when
    `want_fp32(o0)` says the consumer would upcast, else `o0`'s dtype."""
    dtype = torch.float32 if want_fp32(o0) else o0.dtype
    rows, width = o0.shape
    full = torch.empty((rows, max(o0.stride(0), width)), dtype=dtype, device=o0.device)
    return full[:, :width]


def localized_join_scores(
    loc: LocalizedDecodeMetadata,
    o0: torch.Tensor,
    o1: torch.Tensor,
    out: torch.Tensor,
    row_ends: torch.Tensor,
) -> None:
    """Join both domains' compact scores into `out`, on the current stream.

    Writes `out[r, c]` for `c < row_ends[r]` only (bit-identical to the
    single-launch scan, converted to `out`'s dtype); columns at or past a
    row's end stay unwritten, as DeepGEMM leaves them.
    """
    num_rows, width = o0.shape
    if num_rows == 0 or width == 0:
        return
    assert o0.dtype == o1.dtype and o0.stride() == o1.stride()
    assert o0.stride(1) == 1 and out.stride(1) == 1
    _join_kernel[(_JOIN_PROGRAMS_PER_SM * loc.resources.total_sms,)](
        o0,
        o1,
        o0.stride(0),
        out,
        out.stride(0),
        row_ends,
        loc.join_map,
        loc.join_map.stride(0),
        num_rows,
        (o1.data_ptr() - o0.data_ptr()) // o0.element_size(),
        R=loc.page_rows,
        PB=_JOIN_PAGES_PER_TILE,
        LV=3 if o0.element_size() == 2 else 2,
        ROWS_P2=max(16, triton.next_power_of_2(num_rows)),
        num_warps=_JOIN_WARPS,
    )


def localized_paged_mqa_logits(
    loc: LocalizedDecodeMetadata,
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    max_model_len: int,
    logits_dtype: torch.dtype,
    indices: torch.Tensor | None,
    want_fp32: Callable[[torch.Tensor], bool],
) -> torch.Tensor | None:
    """Die-split decode scan and join.

    Returns the scores with the stock layout (FP32 if `want_fp32(bf16)` says
    the consumer upcasts them), or None when this call cannot be localized
    (the caller runs the stock scan).
    """
    rows = q[0].shape[0] * q[0].shape[1]
    if not _view_matches(loc, kv_cache) or rows != loc.seq_lens[0].shape[0]:
        logger.warning_once(
            "Die-local indexer: a decode scan does not match its localized "
            "metadata; stock scan for it."
        )
        return None
    ensure_built(loc)
    o0, o1 = torch.cuda.execute_on_streams(
        loc.resources.streams,
        lambda d: _scan(
            loc, d, q, kv_cache, weights, max_model_len, logits_dtype, indices
        ),
        range(len(loc.resources.streams)),
    )
    out = localized_join_alloc(o0, want_fp32)
    localized_join_scores(loc, o0, o1, out, seq_lens.reshape(-1))
    return out


# ---------------------------------------------------------------------------
# Per-domain top-k and merge
# ---------------------------------------------------------------------------
# CTA size of the CUDA merge per top-k.
_MERGE_THREADS = {512: 512, 1024: 512, 2048: 1024}
_MERGE_PAGE_ROWS = (64, 128)


def _merge_smem_bytes(topk: int, threads: int, score_f32: bool) -> int:
    """`SMEM_TOTAL` of merge.cu."""
    hist = 4 * (1024 if score_f32 else 512)
    end = hist + 32 + 4 * (threads // 32) + 4 * topk
    return -(-end // 128) * 128


def _merge_kernel(
    topk: int, score_f32: bool, page_rows: int, device_index: int
) -> CudaKernel | None:
    """The CUDA merge for this shape, compiled at its first use outside a
    CUDA graph capture; None for shapes it does not cover, when it cannot be
    built, and inside a capture before its first build."""
    if topk not in _MERGE_THREADS or page_rows not in _MERGE_PAGE_ROWS:
        return None
    defines = (
        ("TOP_K", topk),
        ("THREADS", _MERGE_THREADS[topk]),
        ("SCORE_F32", int(score_f32)),
        ("LOG2_PAGE_ROWS", page_rows.bit_length() - 1),
    )
    return get_kernel(
        "merge.cu",
        "indexer_localize_merge",
        defines,
        device_index,
        build=not torch.cuda.is_current_stream_capturing(),
    )


def prepare_merge(
    topk: int, logits_dtype: torch.dtype, page_rows: int, device_index: int
) -> None:
    """Compile the CUDA merge for a shape ahead of its first use, outside any
    CUDA graph capture."""
    if not torch.cuda.is_current_stream_capturing():
        _merge_kernel(topk, logits_dtype == torch.float32, page_rows, device_index)


def _merge_fits(
    idx: list[torch.Tensor], inv: list[torch.Tensor], out: torch.Tensor
) -> bool:
    """The CUDA merge's index buffer requirements: one row stride per pair,
    unit inner strides, 16-byte aligned rows for its vector loads (candidate
    columns) and stores (output), 32-bit strides."""
    return (
        idx[0].stride(0) == idx[1].stride(0)
        and inv[0].stride(0) == inv[1].stride(0)
        and all(t.stride(-1) == 1 for t in (idx[0], idx[1], inv[0], inv[1], out))
        and idx[0].stride(0) % 4 == 0
        and out.stride(0) % 4 == 0
        and all(t.data_ptr() % 16 == 0 for t in (idx[0], idx[1], out))
        and max(inv[0].stride(0), out.stride(0)) < 2**31
    )


def merge_candidates(
    loc: LocalizedDecodeMetadata,
    o0: torch.Tensor,
    o1: torch.Tensor,
    idx: list[torch.Tensor],
    topk_tokens: int,
    topk_out: torch.Tensor,
) -> None:
    """Merge both domains' top-k candidates into `topk_out`, on the current
    stream. The CUDA merge must be available for this shape and buffers."""
    score_f32 = o0.dtype == torch.float32
    kernel = _merge_kernel(topk_tokens, score_f32, loc.page_rows, loc.resources.device)
    assert kernel is not None and _merge_fits(idx, loc.inv_maps, topk_out)
    assert o0.stride() == o1.stride() and o0.stride(1) == 1 and o0.stride(0) < 2**31
    threads = _MERGE_THREADS[topk_tokens]

    def ptr(t: torch.Tensor) -> ctypes.c_uint64:
        return ctypes.c_uint64(t.data_ptr())

    kernel.launch(
        topk_out.shape[0],
        threads,
        _merge_smem_bytes(topk_tokens, threads, score_f32),
        [
            ptr(o0),
            ptr(o1),
            ctypes.c_int32(o0.stride(0)),
            ptr(idx[0]),
            ptr(idx[1]),
            ctypes.c_int32(idx[0].stride(0)),
            ptr(loc.inv_maps[0]),
            ptr(loc.inv_maps[1]),
            ctypes.c_int32(loc.inv_maps[0].stride(0)),
            ptr(topk_out),
            ctypes.c_int32(topk_out.stride(0)),
        ],
        torch.cuda.current_stream(),
    )


def localized_topk_backend(
    selector: Any, rows: int, max_model_len: int, topk_tokens: int
) -> str | None:
    """The top-k each domain runs on its scores: the kernel the stock selector
    would run on these BF16 scores, when it can run on a domain's SMs (no CTA
    waits for CTAs it cannot see). That is `cooperative_topk` (clusters, no
    grid sync) and, above 64 rows, `persistent_topk`, which then launches one
    CTA per row. None for any other kernel."""
    from vllm.model_executor.layers.indexer_topk import AUTO_COOPERATIVE_MAX_ROWS

    probe = torch.empty_strided(
        (rows, max_model_len),
        (-(-max_model_len // 64) * 64, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    try:
        backend = selector.resolve_backend(probe, topk_tokens, rows)
    except Exception:  # noqa: BLE001 - an unsupported shape
        return None
    if backend == "cooperative" or (
        backend == "persistent" and rows > AUTO_COOPERATIVE_MAX_ROWS
    ):
        return backend
    return None


# Per-domain top-k scratch, per (device, domain) and (device, domain, topk).
_TOPK_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}
_TOPK_INDICES: dict[tuple[int, int, int], torch.Tensor] = {}


def _topk_scratch(
    loc: LocalizedDecodeMetadata, topk_tokens: int, rows: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE

    device = loc.resources.device
    workspaces, indices = [], []
    for d in range(len(loc.resources.streams)):
        if (device, d) not in _TOPK_WORKSPACES:
            _TOPK_WORKSPACES[device, d] = torch.empty(
                RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=device
            )
        workspaces.append(_TOPK_WORKSPACES[device, d])
        key = (device, d, topk_tokens)
        if key not in _TOPK_INDICES or _TOPK_INDICES[key].shape[0] < rows:
            _TOPK_INDICES[key] = torch.empty(
                (max(rows, loc.capacity), topk_tokens), dtype=torch.int32, device=device
            )
        indices.append(_TOPK_INDICES[key][:rows])
    return workspaces, indices


def localized_paged_mqa_topk(
    loc: LocalizedDecodeMetadata,
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    max_model_len: int,
    logits_dtype: torch.dtype,
    indices: torch.Tensor | None,
    topk_tokens: int,
    max_seq_len: int,
    topk_out: torch.Tensor,
    keep: list[torch.Tensor] | None = None,
    topk_backend: str = "cooperative",
) -> bool:
    """Per-domain scan and top-k, then the merge, into `topk_out`.

    `topk_out` is the `[rows, K]` int32 view of the top-k buffer, and
    `topk_backend` the per-domain top-k (`localized_topk_backend`). With
    `keep`, both domains' compact scores are appended to it (for a join the
    caller runs later; such a call never defers). When the caller requested
    a deferred top-k (`indexer_overlap.request_deferred_topk`), the call
    returns without joining: domain 0's stream waits for domain 1 and merges.
    Returns False, without launching anything, when this call cannot be
    localized.
    """
    rows = q[0].shape[0] * q[0].shape[1]
    if (
        not _view_matches(loc, kv_cache)
        or logits_dtype not in (torch.bfloat16, torch.float32)
        or rows != loc.seq_lens[0].shape[0]
        or topk_out.shape[0] != rows
    ):
        logger.warning_once(
            "Die-local indexer: a decode top-k does not match its localized "
            "metadata; stock scan and top-k for it."
        )
        return False
    workspaces, idx = _topk_scratch(loc, topk_tokens, rows)
    merge = _merge_kernel(
        topk_tokens, logits_dtype == torch.float32, loc.page_rows, loc.resources.device
    )
    if merge is None or not _merge_fits(idx, loc.inv_maps, topk_out):
        logger.warning_once(
            "Die-local indexer: no CUDA merge for top-k %d, %d-row pages and "
            "these buffers; stock top-k for it.",
            topk_tokens,
            loc.page_rows,
        )
        return False
    ensure_built(loc)
    streams = loc.resources.streams
    topk_op = (
        torch.ops._C.persistent_topk
        if topk_backend == "persistent"
        else torch.ops._C.cooperative_topk
    )

    def scan_topk(d: int) -> torch.Tensor:
        o = _scan(loc, d, q, kv_cache, weights, max_model_len, logits_dtype, indices)
        topk_op(o, loc.seq_lens[d], idx[d], workspaces[d], topk_tokens, max_seq_len)
        return o

    if keep is None and indexer_overlap.consume_deferred_topk_request():
        # Fork without joining: each domain stream waits for the caller's
        # work so far, scans and selects; domain 0 then waits for domain 1
        # and merges. The caller joins on domain 0's stream.
        start = torch.cuda.Event()
        start.record(torch.cuda.current_stream())
        outs, done = [], []
        for d, stream in enumerate(streams):
            stream.wait_event(start)
            with torch.cuda.stream(stream):
                outs.append(scan_topk(d))
                event = torch.cuda.Event()
                event.record(stream)
                done.append(event)
        merge_stream = streams[0]
        for event in done[1:]:
            merge_stream.wait_event(event)
        with torch.cuda.stream(merge_stream):
            merge_candidates(loc, outs[0], outs[1], idx, topk_tokens, topk_out)
        # The domain streams read q, weights and indices, allocated on the
        # caller's stream, and domain 0 reads domain 1's scores.
        indexer_overlap.set_deferred_topk(merge_stream, (q, weights, indices, outs))
        return True
    o0, o1 = torch.cuda.execute_on_streams(streams, scan_topk, range(len(streams)))
    if keep is not None:
        keep.extend((o0, o1))
    merge_candidates(loc, o0, o1, idx, topk_tokens, topk_out)
    return True
