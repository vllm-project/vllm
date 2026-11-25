# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local decode scan, top-k and merge of the sparse-attention indexer.

Each locality domain's green-context stream runs DeepGEMM's
`fp8_fp4_paged_mqa_logits` over the indexer pages in its own memory (the
inputs `build.py` made for the domain). Then either:

* `localized_paged_mqa_topk`: each domain runs the stock selector's top-k on
  its scores, and `merge.cu` merges the 2 x K candidates into the K
  request-local token positions (the stock top-k's score multiset; ties may
  pick other tokens); or
* `localized_paged_mqa_logits`: one kernel joins both domains' scores into the
  stock full-length score array, bit-identical to the single-launch scan.
"""

import ctypes
from collections.abc import Callable
from typing import Any

import torch

from vllm.model_executor.layers import indexer_overlap
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.v1.attention.ops.indexer_localize.build import (
    LocalizedDecodeMetadata,
    ensure_built,
)
from vllm.v1.attention.ops.indexer_localize.kernels import CudaKernel, get_kernel


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
    """One domain's DeepGEMM scan on the current (the domain's) stream, with
    the domain's SM count and programmatic dependent launch off."""
    from vllm.utils import deep_gemm as dg

    num_sms = dg.get_num_sms()
    dg.set_num_sms(loc.resources.sms[domain])
    pdl = dg.get_pdl()
    if pdl:
        dg.set_pdl(False)
    try:
        return dg.fp8_fp4_paged_mqa_logits(
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
        dg.set_num_sms(num_sms)
        if pdl:
            dg.set_pdl(True)


# ---------------------------------------------------------------------------
# Join of both domains' compact scores
# ---------------------------------------------------------------------------
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
    ROWS_P2: tl.constexpr,
):
    """dst[row, p*R + i] = src_{dom(p)}[row, local_page(p)*R + i] for columns
    below the row end, whole pages in 16-byte vectors.

    Program `pid` copies an equal share of all rows' full pages laid end to
    end; the partial last page of each row is copied element by element.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    i = tl.arange(0, R)
    pb = tl.arange(0, PB)

    # Full pages: program pid copies global full pages [lo, hi).
    rows = tl.arange(0, ROWS_P2)
    nfull = tl.load(row_ends_ptr + rows, mask=rows < num_rows, other=0) // R
    cum = tl.cumsum(nfull, axis=0)  # inclusive prefix of full pages
    total = tl.sum(nfull, axis=0).to(tl.int64)
    lo = total * pid // nprog
    hi = total * (pid + 1) // nprog
    r = tl.sum((cum.to(tl.int64) <= lo).to(tl.int32), axis=0)  # row of page lo
    base = tl.sum(tl.where(rows < r, nfull, 0), axis=0).to(tl.int64)
    gp = lo
    while gp < hi:
        nf = (tl.load(row_ends_ptr + r) // R).to(tl.int64)
        p_lo = (gp - base).to(tl.int32)
        p_hi = tl.minimum(hi - base, nf).to(tl.int32)
        r64 = r.to(tl.int64)
        s0 = src0_ptr + r64 * src_stride
        drow = dst_ptr + r64 * dst_stride
        jrow = jmap_ptr + r64 * jmap_stride
        e_next = tl.load(jrow + p_lo + pb, mask=(p_lo + pb) < p_hi, other=0)
        for p0 in range(p_lo, p_hi, PB):
            p = p0 + pb
            m = p < p_hi
            e = e_next
            e_next = tl.load(jrow + p + PB, mask=(p + PB) < p_hi, other=0)
            # A product keeps delta's divisibility (tl.where would not); 64-bit
            # because delta plus a column can pass 2**31.
            off = (e & 1).to(tl.int64) * delta + (e >> 1) * R
            v = tl.load(s0 + off[:, None] + i[None, :], mask=m[:, None])
            tl.store(
                drow + p[:, None] * R + i[None, :],
                v.to(drow.dtype.element_ty),
                mask=m[:, None],
            )
        gp = base + p_hi
        base = base + nf
        r = r + 1

    # The partial last page of rows pid, pid + nprog, ...
    for rr in range(pid, num_rows, nprog):
        end_r = tl.load(row_ends_ptr + rr)
        nf = end_r // R
        rem = end_r - nf * R
        if rem > 0:
            r64 = rr.to(tl.int64)  # type: ignore[attr-defined]
            e = tl.load(jmap_ptr + r64 * jmap_stride + nf)
            m = i < rem
            sp = tl.where((e & 1) != 0, src1_ptr, src0_ptr) + r64 * src_stride
            v = tl.load(sp + (e >> 1) * R + i, mask=m)
            tl.store(
                dst_ptr + r64 * dst_stride + nf * R + i,
                v.to(dst_ptr.dtype.element_ty),
                mask=m,
            )


def localized_join_alloc(
    o0: torch.Tensor, want_fp32: Callable[[torch.Tensor], bool]
) -> torch.Tensor:
    """The full-length score array for these compact scores: DeepGEMM's
    shape and row stride, with storage for whole rows; FP32 when
    `want_fp32(o0)` says the consumer would upcast, else `o0`'s dtype."""
    dtype = torch.float32 if want_fp32(o0) else o0.dtype
    full = torch.empty((o0.shape[0], o0.stride(0)), dtype=dtype, device=o0.device)
    return full[:, : o0.shape[1]]


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
    num_rows = o0.shape[0]
    # Tuned on Rubin: 6 programs per SM, tiles of 32 pages on 4 warps.
    _join_kernel[(6 * loc.resources.total_sms,)](
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
        PB=32,
        ROWS_P2=max(16, triton.next_power_of_2(num_rows)),
        num_warps=4,
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
) -> torch.Tensor:
    """Die-split decode scan and join: the joined scores, laid out and typed
    as `localized_join_alloc` makes them."""
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
# CTA size of the CUDA merge per top-k: every top-k `cooperative_topk` and
# `persistent_topk` take, so each per-domain top-k has a merge.
_MERGE_THREADS = {512: 512, 1024: 512, 2048: 1024}


def _merge_kernel(
    loc: LocalizedDecodeMetadata, topk: int, score_f32: bool
) -> CudaKernel:
    """The CUDA merge for this shape (top-k, score dtype and `loc`'s page
    rows), compiled at its first use."""
    defines = (
        ("TOP_K", topk),
        ("THREADS", _MERGE_THREADS[topk]),
        ("SCORE_F32", int(score_f32)),
        ("LOG2_PAGE_ROWS", loc.page_rows.bit_length() - 1),
    )
    return get_kernel(
        "merge.cu", "indexer_localize_merge", defines, loc.resources.device
    )


def prepare_merge(
    loc: LocalizedDecodeMetadata, topk: int, logits_dtype: torch.dtype
) -> None:
    """Compile the CUDA merge for a shape ahead of its first use, which may
    be inside a CUDA graph capture, where modules cannot be loaded."""
    _merge_kernel(loc, topk, logits_dtype == torch.float32)


def merge_candidates(
    loc: LocalizedDecodeMetadata,
    o0: torch.Tensor,
    o1: torch.Tensor,
    idx: list[torch.Tensor],
    topk_tokens: int,
    topk_out: torch.Tensor,
) -> None:
    """Merge both domains' top-k candidates into `topk_out`, on the current
    stream."""
    kernel = _merge_kernel(loc, topk_tokens, o0.dtype == torch.float32)
    # Per array: both domains' pointers (dst has one) and the row stride.
    args: list[ctypes.c_uint64 | ctypes.c_int32] = []
    for ts in ((o0, o1), idx, loc.inv_maps, (topk_out,)):
        args += [ctypes.c_uint64(t.data_ptr()) for t in ts]
        args.append(ctypes.c_int32(ts[0].stride(0)))
    kernel.launch(topk_out.shape[0], _MERGE_THREADS[topk_tokens], 0, args)


def localized_topk_backend(
    selector: Any, rows: int, max_model_len: int, topk_tokens: int
) -> str | None:
    """The top-k each domain runs on its scores: the kernel the stock selector
    would run on these BF16 scores, if it runs on a domain's SMs (no CTA waits
    for CTAs it cannot see): `cooperative_topk` (clusters, no grid sync) or,
    above 64 rows, `persistent_topk` (one CTA per row). Else None."""
    from vllm.model_executor.layers.indexer_topk import AUTO_COOPERATIVE_MAX_ROWS

    probe = torch.empty_strided(
        (rows, max_model_len),
        (round_up(max_model_len, 64), 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    try:
        backend = selector.resolve_backend(probe, topk_tokens, rows)
    except RuntimeError:  # an explicit backend that rejects these logits
        return None
    if backend == "cooperative" or (
        backend == "persistent" and rows > AUTO_COOPERATIVE_MAX_ROWS
    ):
        return backend
    return None


# Per-domain top-k scratch, per (device, domain) and (device, domain, topk,
# localizer rows).
_TOPK_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}
_TOPK_INDICES: dict[tuple[int, int, int, int], torch.Tensor] = {}


def _topk_scratch(
    loc: LocalizedDecodeMetadata, topk_tokens: int, rows: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Each domain's top-k workspace and `[rows, K]` int32 output, kept per
    process and sized for the localizer's largest batch."""
    from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE

    device = loc.resources.device
    workspaces, indices = [], []
    for d in range(len(loc.resources.streams)):
        if (device, d) not in _TOPK_WORKSPACES:
            _TOPK_WORKSPACES[device, d] = torch.empty(
                RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=device
            )
        workspaces.append(_TOPK_WORKSPACES[device, d])
        key = (device, d, topk_tokens, loc.max_rows)
        if key not in _TOPK_INDICES:
            _TOPK_INDICES[key] = torch.empty(
                (loc.max_rows, topk_tokens), dtype=torch.int32, device=device
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
    topk_backend: str,
    scores_out: list[torch.Tensor] | None = None,
) -> None:
    """Per-domain scan and top-k, then the merge, into `topk_out`.

    `topk_out` is the `[rows, K]` int32 view of the top-k buffer, and
    `topk_backend` the per-domain top-k (`localized_topk_backend`). With
    `scores_out`, both domains' compact scores are appended to it (for a join
    the caller runs later; such a call never defers). When the caller
    requested a deferred top-k (`indexer_overlap.request_deferred_topk`), the
    call returns without joining: domain 0's stream waits for domain 1 and
    merges.
    """
    workspaces, idx = _topk_scratch(loc, topk_tokens, topk_out.shape[0])
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

    if scores_out is None and indexer_overlap.consume_deferred_topk_request():
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
        return
    o0, o1 = torch.cuda.execute_on_streams(streams, scan_topk, range(len(streams)))
    if scores_out is not None:
        scores_out.extend((o0, o1))
    merge_candidates(loc, o0, o1, idx, topk_tokens, topk_out)
