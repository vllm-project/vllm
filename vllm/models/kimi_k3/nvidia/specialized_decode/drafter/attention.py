# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense MLA decode of the DSpark draft block with the query tokens folded into heads.

A DSpark draft step runs ``k`` query tokens per request (anchor + ``k - 1`` noise
tokens) through each dense MLA draft layer. The block is non-causal: every query
token of a request attends to exactly the same keys (the whole context plus the
block itself, already written to the cache). The stock ``FLASHINFER_MLA`` decode
has no non-causal flag, so it flattens the block into ``R * k`` independent
single-token rows, each with its own copy of the request's block table. Every row
then streams the request's whole latent cache again: ``k`` times the KV bytes, and
each CTA's MMA sees only the ``H`` local heads of one token (8 at TP8).

Absorbed MLA decode is MQA (one latent KV head), so query heads are independent
rows over a shared key set. Folding the ``k`` tokens of a request into the head
axis, ``[R * k, H, D] -> [R, 1, k * H, D]``, gives one decode row per request with
``k * H`` heads over the same keys. That is the same attention, exactly: every
(token, head) pair still gets its own full softmax over the identical key set, at
the same precision (FP8 Q and KV, the stock bmm scales); only the kernel's tiling
(which CTA holds which head, the KV split) can differ. The KV is read once per
request instead of ``k`` times, and the MMA M grows ``k``-fold (32-56 heads at TP8,
256-448 at TP1 / DCP).

The output ``[R, 1, k * H, L]`` and LSE ``[R, k * H]`` are the stock
``[R * k, H, L]`` / ``[R * k, H]`` in the same memory order (token-major,
then head), so the DCP combine and the ``W_UV`` up-projection are unchanged.
"""

import functools
from typing import Any

import torch

_COUNTER: dict[torch.device, torch.Tensor] = {}
_WORKSPACE: dict[torch.device, torch.Tensor] = {}
# CAKE keeps its split-KV partials in the workspace: rows x splits x 1 KiB. Its
# own buffer, so the stock decode's shared workspace is never regrown under a
# captured graph.
CAKE_WORKSPACE_BYTES = 256 << 20
# Decode shapes whose CAKE plan needs more workspace than CAKE_WORKSPACE_BYTES.
_CAKE_WORKSPACE_TOO_SMALL: set[tuple[int, int, int, int]] = set()


# The most the workspace is sized to at init; larger plans run the trtllm-gen fold.
CAKE_WORKSPACE_MAX_BYTES = 2 << 30


def _cake_workspace(device: torch.device) -> torch.Tensor:
    buf = _WORKSPACE.get(device)
    if buf is None:
        buf = torch.zeros(CAKE_WORKSPACE_BYTES, dtype=torch.int8, device=device)
        _WORKSPACE[device] = buf
    return buf


def cake_workspace_bytes(
    max_reqs: int, k: int, heads: int, max_seq_len: int, sm_count: int
) -> int:
    """The largest CAKE split-KV workspace any folded draft batch of up to
    ``max_reqs`` requests needs (CUDA graphs plan at ``max_seq_len``)."""
    from flashinfer.mla.cake_kimi_k3_mla import plan_attention, workspace_bytes

    need = 0
    for batch in range(1, max_reqs + 1):
        plan = plan_attention(
            batch=batch,
            max_q_len=1,
            num_heads=k * heads,
            max_seq_len=max_seq_len,
            sm_count=sm_count,
        )
        need = max(need, workspace_bytes(plan.rows_max, plan.num_split))
    return need


def reserve_cake_workspace(
    device: torch.device, max_reqs: int, k: int, heads: int, max_seq_len: int
) -> int:
    """Allocate the CAKE workspace for the deployment's largest draft batch now,
    at model load (before the KV-cache memory profile and every graph capture,
    which bind its address), at most ``CAKE_WORKSPACE_MAX_BYTES``. Returns its
    size. (GB300, 1M plan length: TP8 64 x k5 6 MiB; DEP 128 x k4 324 MiB,
    128 x k5 589 MiB, 256 x k5 4.9 GiB -> capped, its largest batches fold on
    trtllm-gen.) An eager batch planned at a shorter length can need more; the
    call then takes the same trtllm-gen fold. Without FlashInfer's CAKE planner
    (capability fallback) nothing is reserved and the default buffer is used."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    try:
        need = cake_workspace_bytes(max_reqs, k, heads, max_seq_len, sms)
    except ImportError:
        return 0
    nbytes = min(max(need, 1 << 20), CAKE_WORKSPACE_MAX_BYTES)
    buf = _WORKSPACE.get(device)
    if buf is None or buf.numel() < nbytes:
        _WORKSPACE[device] = torch.zeros(nbytes, dtype=torch.int8, device=device)
    return nbytes


def foldable(attn_metadata, num_rows: int) -> int:
    """Query tokens per request when the decode batch can be folded, else 0.

    Folding needs a non-causal decode batch in which every request has the same
    number of query tokens (a DSpark draft block always does)."""
    if getattr(attn_metadata, "causal", True):
        return 0
    decode = attn_metadata.decode
    if decode is None:
        return 0
    num_reqs = attn_metadata.num_decodes
    if num_reqs <= 0 or num_rows != attn_metadata.num_decode_tokens:
        return 0
    if num_rows % num_reqs != 0:
        return 0
    k = num_rows // num_reqs
    # One token per request: nothing to fold (the stock rows are already per request).
    if k < 2 or decode.block_table.shape[0] < num_reqs:
        return 0
    # A total divisible by the request count is not proof of uniform blocks
    # (6 + 2 tokens over two requests); with every block at most
    # ``max_query_len`` long, ``max_query_len == k`` is.
    if getattr(decode, "max_query_len", k) != k:
        return 0
    return k


def _counter_buffer(max_reqs: int, num_heads: int, device: torch.device):
    """trtllm-gen's multi-CTA-KV semaphores (it resets them after each launch, so
    they are zeroed once). Sized for the deployment's largest batch and kept apart
    from the stock decode's buffer, so neither ever reallocates the other's under a
    captured CUDA graph."""
    from flashinfer.utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )

    nbytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        max_reqs, num_heads, get_device_sm_count(device)
    )
    buf = _COUNTER.get(device)
    if buf is None or buf.numel() < nbytes:
        buf = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        _COUNTER[device] = buf
    return buf


# FlashInfer's cute-dsl MLA decode (the LSE-returning DCP kernel) tiles at most 128
# query heads per decode row.
CUTE_DSL_MAX_HEADS = 128


def fold_plan(k: int, heads: int, need_lse: bool, cake_ok: bool) -> tuple[int, str]:
    """(query tokens folded into one decode row, FlashInfer MLA backend).

    Without an LSE (TP / DEP) all ``k`` tokens fold into one row per request and
    run the Kimi K3 CAKE kernel, whose two-CTA wide route shares each 128-key tile
    across 128 packed (token, head) rows (``auto`` = trtllm-gen when CAKE cannot
    run: it reads the tile once per 64-head group). With an LSE (DCP) the cute-dsl
    kernel takes at most 128 heads per row: fold the largest divisor ``f`` of
    ``k`` with ``f * heads <= 128`` (k=4 at 64 heads: pairs, half the KV reads)."""
    if not need_lse:
        return k, "cake" if cake_ok else "auto"
    f = max(
        d for d in range(1, k + 1) if k % d == 0 and d * heads <= CUTE_DSL_MAX_HEADS
    )
    return f, "cute-dsl"


def cake_supported(
    q: torch.Tensor, kv_cache: torch.Tensor, kv_lora_rank: int, rope: int
) -> bool:
    """CAKE's Kimi K3 MLA kernel: SM100 family, FP8 query and cache, page 64,
    kv_lora_rank 512 / rope 64."""
    from vllm.platforms import current_platform

    return (
        current_platform.is_device_capability_family(100)
        and q.dtype == torch.float8_e4m3fn
        and kv_cache.dtype == torch.float8_e4m3fn
        and kv_cache.shape[-2] == 64
        and kv_lora_rank == 512
        and rope == 64
    )


# Length-balanced LSE (DCP) draft decode. FlashInfer's cute-dsl MLA decode splits
# every decode row's KV into the same number of splits, picked from the row count
# alone: from 39 rows on (152 SMs) one split, so each row streams its whole local
# context on one SM pair and the launch lasts as long as its longest row. GB300,
# HIGH hybrid decode (124 rows = 62 requests x 2 fold groups, DCP8, random KV):
# 486 us at the production context mix vs 293 us at a uniform context of the same
# mean (the same FP8 MMA work; 2.6 PF is the kernel's rate at full occupancy).
#
# Here the TIERS[0] longest rows run as their own launch, which the same heuristic
# splits sms // (2 x 4) = 19 ways, the next TIERS[1] 9 ways and the next TIERS[2]
# 4 ways (the kernel's own split-KV: FP32 partials and its reducer); every other
# row runs in the one unsplit launch, in place. A tier row moves only when it is
# longer than a pair's share of the batch's keys (``MOVE``): one SM pair alone
# would hold it past the rest; otherwise its tier row is empty (length 0, its
# CTAs return at once) and it stays in the unsplit launch. The launches run on
# their own streams, the unsplit one (the longest work items) at the highest
# priority, so the split tiers fill the SM pairs it leaves idle. Only the KV
# split points of the moved rows differ from the one launch: the same kernel,
# query, scales and BF16 output + LSE (486 -> 287 us at
# the production mix, -176..-232 us on random production batches, +6..9 us at
# uniform contexts, where nothing moves).
TIERS = (4, 8, 16)
# A row moves to a split tier when longer than MOVE (num, den) of a pair's share of
# the batch's keys (sum / pairs). Swept on GB300 at 1/2, 3/4, 1: 1 is best.
MOVE = (1, 1)
_STREAMS: dict[torch.device, list[torch.cuda.Stream]] = {}
_TIER_WORKSPACE: dict[torch.device, torch.Tensor] = {}
_SMS: dict[torch.device, int] = {}


def _sm_count(device: torch.device) -> int:
    sms = _SMS.get(device)
    if sms is None:
        sms = _SMS[device] = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
    return sms


def balanced_tiers(device: torch.device, rows: int) -> tuple[int, ...]:
    """The split tiers (rows each, longest first) of a cute-dsl LSE draft decode of
    ``rows`` decode rows; () runs the one launch. Balanced from the row count on
    which the one launch runs unsplit (``sms // (2 * rows) < 2``: below it the
    kernel splits every row already), keeping at least half the rows unsplit."""
    if rows * 4 <= _sm_count(device):
        return ()
    tiers: list[int] = []
    for size in TIERS:
        if sum(tiers) + size > rows // 2:
            break
        tiers.append(size)
    return tuple(tiers)


def tier_workspace_bytes(rows: int, splits: int, heads: int, lora: int) -> int:
    """The cute-dsl decode's split-KV workspace: FP32 partials + LSE per split
    (``mla_decode.get_workspace_size``)."""
    return rows * splits * -(-heads // 128) * 128 * (lora + 1) * 4


def _streams(device: torch.device, count: int) -> list[torch.cuda.Stream]:
    """``count`` streams, the first the highest priority, each next one lower."""
    streams = _STREAMS.setdefault(device, [])
    low, high = torch.cuda.Stream.priority_range()
    while len(streams) < count:
        priority = min(low, high + len(streams))
        streams.append(torch.cuda.Stream(device, priority=priority))
    return streams[:count]


@functools.cache
def _row_copy_kernel():
    """``dst[i] = src[i]`` row by row (int32 words), for the rows whose ``target``
    is below ``bound`` only: gathers (``gather``: dst row i <- src row target[i])
    or scatters (dst row target[i] <- src row i). Cost: the moved rows."""
    from vllm.triton_utils import tl, triton

    @triton.jit
    def row_copy(
        src, dst, target, bound, words, src_pitch, dst_pitch,
        GATHER: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        i = tl.program_id(0)
        t = tl.load(target + i)
        if t < bound:
            cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            mask = cols < words
            if GATHER:
                s_row, d_row = t, i
            else:
                s_row, d_row = i, t
            v = tl.load(src + s_row.to(tl.int64) * src_pitch + cols, mask=mask)
            tl.store(dst + d_row.to(tl.int64) * dst_pitch + cols, v, mask=mask)

    return row_copy


def _row_copy(src, dst, target, bound, gather):
    """See ``_row_copy_kernel``; ``src`` / ``dst``: [rows, ...] contiguous rows."""
    s = src.reshape(src.shape[0], -1).view(torch.int32)
    d = dst.reshape(dst.shape[0], -1).view(torch.int32)
    words = s.shape[1]
    block = 4096
    grid = (target.shape[0], -(-words // block))
    _row_copy_kernel()[grid](
        s, d, target, bound, words, s.stride(0), d.stride(0), gather, block
    )


@functools.cache
def _plan_kernel():
    """One CTA: which rows the split tiers take (see ``_balanced_tables``)."""
    from vllm.triton_utils import tl, triton

    @triton.jit
    def plan(
        sl, tot, rest, dst, tier_sl, tier_tot, rows, split_rows, pairs,
        MOVE_NUM: tl.constexpr, MOVE_DEN: tl.constexpr, BLOCK: tl.constexpr,
    ):  # fmt: skip
        offs = tl.arange(0, BLOCK)
        live = offs < rows
        length = tl.load(sl + offs, mask=live, other=0)
        total = tl.sum(length.to(tl.int64), axis=0)
        # Rank by length, longest first (ties: the lower row first).
        rank = tl.zeros([BLOCK], dtype=tl.int32)
        for j in range(0, rows):
            other = tl.load(sl + j)
            ahead = (other > length) | ((other == length) & (j < offs))
            rank += ahead.to(tl.int32)
        split = live & (rank < split_rows)
        share = length.to(tl.int64) * (MOVE_DEN * pairs) > MOVE_NUM * total
        move = split & share
        tl.store(rest + offs, tl.where(move, 0, length), mask=live)
        tl.store(dst + rank, tl.where(move, offs, rows), mask=split)
        tl.store(tier_sl + rank, tl.where(move, length, 0), mask=split)
        tl.store(tier_tot + rank, tl.load(tot + offs, mask=live, other=0), mask=split)

    return plan


def _balanced_tables(cache: dict | None, bt, sl, tot, split_rows, pairs):
    """Which rows the split tiers take, cached per batch like ``_group_tables``
    (every draft layer of a step sees the same lengths): of the ``split_rows``
    longest (longest first), those longer than ``MOVE`` of a pair's share of the
    batch's keys (``sum / pairs``). Returns (each tier row's batch row, or
    ``rows`` when it stays in the unsplit launch; every row's length in the
    unsplit launch, 0 for the moved ones; the tier rows' block table, lengths
    (0 when not moved) and global lengths)."""
    rows = sl.shape[0]
    key = ("draft_balanced", rows, split_rows)
    if cache is not None and key in cache:
        return cache[key]
    i32 = dict(dtype=torch.int32, device=sl.device)
    rest = torch.empty(rows, **i32)
    dst = torch.empty(split_rows, **i32)
    tier_sl = torch.empty(split_rows, **i32)
    tier_tot = torch.empty(split_rows, **i32)
    block = max(32, 1 << (rows - 1).bit_length())
    _plan_kernel()[(1,)](
        sl, tot, rest, dst, tier_sl, tier_tot, rows, split_rows, pairs,
        MOVE[0], MOVE[1], block,
    )  # fmt: skip
    tier_bt = bt.new_empty((split_rows, bt.shape[1]))
    _row_copy(bt, tier_bt, dst, rows, gather=True)  # (unmoved rows: unread)
    out = (dst, rest, tier_bt, tier_sl, tier_tot)
    if cache is not None:
        cache[key] = out
    return out


def balanced_decode(
    common: dict[str, Any],
    table_cache: dict | None,
    kv_lora_rank: int,
    tiers: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The cute-dsl LSE decode of ``common`` (``folded_decode``'s FlashInfer
    arguments) as the split ``tiers`` of the longest rows and one unsplit launch
    of the rest (see ``TIERS``). Returns ``[rows, 1, heads, kv_lora_rank]`` BF16
    and ``[rows, heads]`` FP32, as the one launch."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    query = common["query"]
    rows, _, heads, _ = query.shape
    device = query.device
    tot = common.get("causal_seqlens_kv_global")
    split_rows = sum(tiers)
    sms = _sm_count(device)
    dst, rest, bt, sl, tot_s = _balanced_tables(
        table_cache,
        common["block_tables"],
        common["seq_lens"],
        common["seq_lens"] if tot is None else tot,
        split_rows,
        sms // 2,
    )
    # One workspace slice per concurrent split launch.
    sizes = [
        tier_workspace_bytes(m, max(1, sms // (2 * m)), heads, kv_lora_rank)
        for m in tiers
    ]
    ws = _TIER_WORKSPACE.get(device)
    if ws is None or ws.numel() < sum(sizes):
        ws = _TIER_WORKSPACE[device] = torch.empty(
            sum(sizes), dtype=torch.int8, device=device
        )
    out = torch.empty(
        (rows, 1, heads, kv_lora_rank), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty((rows, heads), dtype=torch.float32, device=device)
    q = query.new_empty((split_rows, *query.shape[1:]))
    out_s = out.new_empty((split_rows, *out.shape[1:]))
    lse_s = lse.new_empty((split_rows, heads))
    main = torch.cuda.current_stream(device)
    streams = _streams(device, len(tiers) + 1)
    # The rest in place, in the batch's order (the split rows' lengths zeroed:
    # their CTAs return at once), on the highest-priority stream.
    first = streams[0]
    first.wait_stream(main)
    with torch.cuda.stream(first):
        trtllm_batch_decode_with_kv_cache_mla(
            **dict(common, seq_lens=rest, out=out, lse=lse, return_lse=True)
        )
    # The moved rows' queries (the others' tier rows are empty).
    _row_copy(query, q, dst, rows, gather=True)
    # The tiers, from the coarsest split to the finest (lower priority each).
    start, offset, launches = 0, 0, []
    for m, size in zip(tiers, sizes):
        launches.append((start, m, ws[offset : offset + size]))
        start, offset = start + m, offset + size
    for stream, (begin, count, workspace) in zip(streams[1:], reversed(launches)):
        stream.wait_stream(main)
        rows_ = slice(begin, begin + count)
        args = dict(
            common,
            query=q[rows_],
            block_tables=bt[rows_],
            seq_lens=sl[rows_],
            out=out_s[rows_],
            lse=lse_s[rows_],
            return_lse=True,
            workspace_buffer=workspace,
        )
        if tot is not None:
            args["causal_seqlens_kv_global"] = tot_s[rows_]
        with torch.cuda.stream(stream):
            trtllm_batch_decode_with_kv_cache_mla(**args)
    for stream in streams:
        main.wait_stream(stream)
    _row_copy(out_s, out, dst, rows, gather=False)
    _row_copy(lse_s, lse, dst, rows, gather=False)
    return out, lse


def _group_tables(cache: dict | None, block_table, seq_lens, tot, num_reqs, groups):
    """Per decode-row tables when each request spans ``groups`` rows (cached per
    batch: every draft layer of a step sees the same tables)."""
    if groups == 1:
        return (
            block_table[:num_reqs],
            seq_lens[:num_reqs],
            None if tot is None else tot[:num_reqs],
        )
    key = ("draft_group_tables", groups, num_reqs)
    if cache is not None and key in cache:
        return cache[key]
    row_req = torch.arange(num_reqs * groups, device=seq_lens.device) // groups
    out = (
        block_table[row_req],
        seq_lens[row_req],
        None if tot is None else tot[row_req],
    )
    if cache is not None:
        cache[key] = out
    return out


def folded_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    tokens_per_request: int,
    *,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    bmm1_scale: float,
    bmm2_scale: float,
    workspace: torch.Tensor,
    return_lse: bool = False,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    dcp_tot_seq_lens: torch.Tensor | None = None,
    max_reqs: int = 0,
    backend: str | None = None,
    table_cache: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Non-causal dense MLA decode of ``R`` requests x ``k`` query tokens.

    ``q``: ``[R * k, H, kv_lora_rank + rope]`` (token-major per request);
    ``kv_cache``: ``[pages, page_size, kv_lora_rank + rope]``; ``block_table`` /
    ``seq_lens`` (/ ``dcp_tot_seq_lens``): per request. Returns ``[R * k, H,
    kv_lora_rank]`` and, with ``return_lse``, ``[R * k, H]``. ``backend`` forces
    a FlashInfer backend (benchmarks); by default :func:`fold_plan` picks."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    rows, heads, dim = q.shape
    k = tokens_per_request
    num_reqs = rows // k
    assert num_reqs * k == rows
    need_lse = return_lse or dcp_world_size > 1
    fold, planned = fold_plan(
        k,
        heads,
        need_lse,
        cake_supported(q, kv_cache, kv_lora_rank, qk_rope_head_dim),
    )
    if backend is not None and not need_lse:
        fold, planned = k, backend
    groups = k // fold
    query = q.reshape(num_reqs * groups, 1, fold * heads, dim)
    bt, sl, tot = _group_tables(
        table_cache, block_table, seq_lens, dcp_tot_seq_lens, num_reqs, groups
    )
    common: dict[str, Any] = dict(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=bt,
        seq_lens=sl,
        max_seq_len=max_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        return_lse=return_lse,
    )
    if dcp_world_size > 1:
        assert tot is not None
        common.update(
            enable_dcp=True,
            cp_world=dcp_world_size,
            cp_rank=dcp_rank,
            causal_seqlens_kv_global=tot,
        )
    cake_key = (num_reqs * groups, fold * heads, max_seq_len, bt.shape[1])
    if planned == "cake" and cake_key in _CAKE_WORKSPACE_TOO_SMALL:
        planned = "auto"
    tiers = balanced_tiers(q.device, num_reqs * groups) if return_lse else ()
    if planned == "cute-dsl" and tiers:
        common["backend"] = planned
        o, lse = balanced_decode(common, table_cache, kv_lora_rank, tiers)
        return o.reshape(rows, heads, o.shape[-1]), lse.reshape(rows, heads)
    out = None
    if planned == "cake":
        try:
            out = trtllm_batch_decode_with_kv_cache_mla(backend="cake", **common)
        except ValueError as e:
            # CAKE carves its split-KV partials (rows x splits x ~1 KiB) out of
            # the fixed workspace and validates the size on the host before it
            # launches anything. Large batches (e.g. 128 requests x 5 tokens x 64
            # heads) can need more than the buffer holds; growing it would free
            # memory that earlier captured graphs still point at, so such shapes
            # run the trtllm-gen fold instead.
            if "workspace_buffer needs" not in str(e):
                raise
            _CAKE_WORKSPACE_TOO_SMALL.add(cake_key)
            planned = "auto"
    if out is None:
        extra: dict[str, Any] = {}
        if planned == "auto":
            from vllm.v1.attention.backends.mla.flashinfer_mla import (
                _select_mla_decode_backend,
            )

            sel = _select_mla_decode_backend(fold * heads)
            if sel is not None:
                extra["backend"] = sel
            elif kv_cache.shape[-2] in (32, 64):
                extra["multi_ctas_kv_counter_buffer"] = _counter_buffer(
                    max(max_reqs, num_reqs * groups), fold * heads, q.device
                )
        else:
            extra["backend"] = planned
        out = trtllm_batch_decode_with_kv_cache_mla(**common, **extra)
    if return_lse:
        o, lse = out
        lse = lse.reshape(rows, heads)
    else:
        o, lse = out, None
    return o.reshape(rows, heads, o.shape[-1]), lse


def draft_forward_mqa(
    impl, q: torch.Tensor, kv_cache: torch.Tensor, attn_metadata, layer
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """``FlashInferMLAImpl.forward_mqa`` for a DSpark draft block, folded.

    Returns None (the caller runs the stock ``forward_mqa``) for a batch that
    cannot be folded or a backend other than FLASHINFER_MLA."""
    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        FlashInferMLAImpl,
        _get_workspace_buffer,
    )
    from vllm.v1.kv_cache_interface import is_quantized_kv_cache

    if not isinstance(impl, FlashInferMLAImpl) or isinstance(q, tuple):
        return None
    k = foldable(attn_metadata, q.shape[0])
    if k == 0:
        return None
    if impl.bmm1_scale is None:
        impl.bmm1_scale = impl.scale
        if is_quantized_kv_cache(impl.kv_cache_dtype):
            impl.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
    if impl.bmm2_scale is None:
        impl.bmm2_scale = 1.0
        if is_quantized_kv_cache(impl.kv_cache_dtype):
            impl.bmm2_scale *= layer._k_scale_float
    return_lse = impl.need_to_return_lse_for_decode
    decode = attn_metadata.decode
    cake = not return_lse and cake_supported(
        q, kv_cache, impl.kv_lora_rank, impl.qk_rope_head_dim
    )
    return folded_decode(
        q,
        kv_cache,
        decode.block_table,
        decode.seq_lens,
        attn_metadata.max_seq_len,
        k,
        kv_lora_rank=impl.kv_lora_rank,
        qk_nope_head_dim=impl.qk_nope_head_dim,
        qk_rope_head_dim=impl.qk_rope_head_dim,
        bmm1_scale=impl.bmm1_scale,
        bmm2_scale=impl.bmm2_scale,
        workspace=_cake_workspace(q.device)
        if cake
        else _get_workspace_buffer(return_lse),
        return_lse=return_lse,
        dcp_world_size=impl.dcp_world_size,
        dcp_rank=impl.dcp_rank,
        dcp_tot_seq_lens=decode.dcp_tot_seq_lens,
        max_reqs=impl._mla_counter_max_batch,
        table_cache=decode.__dict__.setdefault("_draft_fold_tables", {}),
    )


def _scales(impl, layer):
    """The stock decode's bmm1 / bmm2 scales (``FlashInferMLAImpl.forward_mqa``)."""
    from vllm.v1.kv_cache_interface import is_quantized_kv_cache

    if impl.bmm1_scale is None:
        impl.bmm1_scale = impl.scale
        if is_quantized_kv_cache(impl.kv_cache_dtype):
            impl.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
    if impl.bmm2_scale is None:
        impl.bmm2_scale = 1.0
        if is_quantized_kv_cache(impl.kv_cache_dtype):
            impl.bmm2_scale *= layer._k_scale_float
    return impl.bmm1_scale, impl.bmm2_scale


LOG2E = 1.4426950408889634


class DraftDcp:
    """The draft block's MLA decode under DCP (DCP = TP: rank d owns head block d)
    on the specialized decode's DCP mailboxes, in three launches per layer instead of
    six (the direct-DCP epoch + query all-gather, the folded decode's split-KV +
    reducer, the LSE all-to-all push + combine) plus the W_UV bmm:

      query      every rank multicasts its FP8 query rows [T, H, 576] into head
                 block ``rank`` of every rank's ``DcpBox`` q mailbox [T, DCP*H, 576]
                 (``comm.Gather``, in place) and polls until all landed
      decode     the stock folded draft decode (``folded_decode``: the k tokens of a
                 request folded into heads, every DCP*H head over the local KV
                 shard, rank-local BF16 rows + LSE), reading the mailbox
      value      ``mla.dcp.DcpValue``: re-arms the q mailbox, pushes each row to its
                 head owner, the owner merges its H heads over the DCP sources
                 (exp2(lse - max) / sum in FP32, one BF16 rounding, as the stock
                 combine) and applies W_UV (BF16 MMA, FP32 accumulation) into the
                 attention output. The draft layers have no output gate: the gate
                 operand is +inf (sigmoid exactly 1).

    Same attention as the stock DCP decode at every stage: the decode kernel and its
    query are the stock ones; only the transport and the merge order differ."""

    def __init__(self, group, device, size, rank, heads, cap):
        from ..cutedsl.mla.dcp import DcpBox

        self.size, self.rank, self.heads, self.cap = size, rank, heads, cap
        self.device = device
        self.box = DcpBox(group, device, size, rank, heads, cap=cap, partials="bf16")
        inf = torch.full((cap, heads * 128), float("inf"), dtype=torch.bfloat16)
        self.gate = inf.to(device)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.values: dict = {}

    @staticmethod
    def eligible(attn, heads) -> bool:
        """The draft layers this path serves: the stock folded draft decode
        (``draft_fold`` on FlashInfer MLA, which takes the FP8 query this path
        gathers), FP8 per-tensor KV, the q-LoRA path without output gate or fused
        GEMM-RS-AR, H local heads of 576 / 512."""
        from vllm.v1.attention.backends.mla.flashinfer_mla import FlashInferMLAImpl

        return (
            getattr(attn, "draft_fold", None) is not None
            and isinstance(getattr(attn, "impl", None), FlashInferMLAImpl)
            and getattr(attn.impl, "supports_quant_query_input", False)
            and attn.q_lora_rank is not None
            and attn.fused_qkv_a_proj is not None
            and not attn.use_output_gate
            and getattr(attn, "gemm_rs_ar", None) is None
            and attn.num_local_heads == heads
            and attn.kv_lora_rank == 512
            and attn.qk_rope_head_dim == 64
            and attn.v_head_dim == 128
            and str(attn.kv_cache_dtype).startswith("fp8")
            and attn.kv_cache_dtype != "fp8_ds_mla"
        )

    @staticmethod
    def latent_major(w_uv):
        """W_UV [H, 512, 128] latent-contiguous, as DcpValue's MMA fragments load it
        (16 B along the latent); the module's is a transposed view."""
        return w_uv.permute(0, 2, 1).contiguous().permute(0, 2, 1)

    def prepare(self, attentions):
        """Each layer's latent-contiguous W_UV (1 MiB per draft layer), at load."""
        self.uv = {id(a): self.latent_major(a.W_UV.detach()) for a in attentions}

    def value(self, batch, q, w_uv):
        key = (batch, q)
        compiled = self.values.get(key)
        if compiled is None:
            compiled = self.values[key] = _compile_value(
                self.heads, self.size, batch, self.cap, w_uv, self.device, self.sms, q
            )
        return compiled

    def forward(self, attn, positions, hidden, parity):
        """``attn(positions, hidden)`` (a draft layer's MultiHeadLatentAttention) on
        this path, or None when this batch takes the stock module."""
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        by_layer = context.attn_metadata
        if not isinstance(by_layer, dict) or attn.layer_name not in by_layer:
            return None
        metadata = by_layer[attn.layer_name]
        rows = hidden.shape[0]
        n = metadata.num_actual_tokens
        if metadata.num_decode_tokens != n or not 1 <= n <= self.cap or n > rows:
            return None
        k = foldable(metadata, n)
        if k == 0 or metadata.decode.dcp_tot_seq_lens is None:
            return None
        if torch.cuda.is_current_stream_capturing() and (n, k) not in self.values:
            return None  # compiled at the eager warmup of every captured size
        return self._forward(attn, positions, hidden, metadata, context, n, k, parity)

    def _forward(self, attn, positions, hidden, metadata, context, n, k, parity):
        from vllm.models.common.ops import fused_q_kv_rmsnorm

        from ..cutedsl.common.host import stream, to_cute
        from .comm import Q_SENT, compile_gather, gather_ctas

        qkv = attn.fused_qkv_a_proj(hidden)[0]
        q_c, kv_c, k_pe = qkv.split(
            [attn.q_lora_rank, attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
        )
        q_c, kv_c_normed = fused_q_kv_rmsnorm(
            q_c,
            kv_c,
            attn.q_a_layernorm.weight.data,
            attn.kv_a_layernorm.weight.data,
            attn.rms_norm_eps,
        )
        q = attn.q_b_proj(q_c)[0].view(-1, attn.num_local_heads, attn.qk_head_dim)
        rows, H = hidden.shape[0], attn.num_local_heads
        attn_out = torch.empty(
            (rows, H * attn.v_head_dim), dtype=hidden.dtype, device=hidden.device
        )
        slot_mapping = context.slot_mapping[attn.layer_name]
        cos_sin = None if attn.rotary_emb is None else attn.rotary_emb.cos_sin_cache
        rope_positions = None if attn.rotary_emb is None else positions[:n]
        q_nope, q_pe = q[:n].split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], -1)
        ql_nope = torch.bmm(q_nope.transpose(0, 1), attn.W_UK_T).transpose(0, 1)
        mqa_q = attn._decode_concat_cache(
            ql_nope,
            q_pe,
            kv_c_normed[:n],
            k_pe.unsqueeze(1)[:n],
            rope_positions,
            cos_sin,
            slot_mapping[:n],
        )
        # Query rows -> head block ``rank`` of every rank's q mailbox (in place).
        box = self.box
        mailbox = box.q[parity]
        words = H * 576 // 4
        gather = compile_gather(
            self.size, words, self.cap, Q_SENT, False, False, self.device
        )
        mc = box.q_mc + parity * mailbox.numel()
        flat = mailbox.view(torch.int32).view(self.cap, -1)
        shard = mqa_q.contiguous().view(torch.uint8).view(n, -1).view(torch.int32)
        gather(
            shard, flat[:n], flat, mc, self.rank, gather_ctas(n, words * self.size // 4)
        )
        query = mailbox[:n].view(torch.float8_e4m3fn)
        impl, decode = attn.impl, metadata.decode
        bmm1, bmm2 = _scales(impl, attn)
        from vllm.v1.attention.backends.mla.flashinfer_mla import _get_workspace_buffer

        o, lse = folded_decode(
            query,
            attn._attn_read_kv_cache(),
            decode.block_table,
            decode.seq_lens,
            metadata.max_seq_len,
            k,
            kv_lora_rank=attn.kv_lora_rank,
            qk_nope_head_dim=attn.qk_nope_head_dim,
            qk_rope_head_dim=attn.qk_rope_head_dim,
            bmm1_scale=bmm1,
            bmm2_scale=bmm2,
            workspace=_get_workspace_buffer(True),
            return_lse=True,
            dcp_world_size=self.size,
            dcp_rank=self.rank,
            dcp_tot_seq_lens=decode.dcp_tot_seq_lens,
            table_cache=decode.__dict__.setdefault("_draft_fold_tables", {}),
        )
        reqs = n // k
        w_uv = self.uv[id(attn)]
        value = self.value(n, k, w_uv)
        mail = box.mailbox(n)
        value(
            to_cute(o.view(n, 1, self.size * H, 512)),
            to_cute(lse.view(n, 1, self.size * H), 4),
            to_cute(decode.seq_lens[:reqs], 4),
            to_cute(mail.partial[parity]),
            to_cute(mail.lse[parity], 4),
            to_cute(mail.peers, 8),
            to_cute(mail.lse_peers, 8),
            to_cute(mailbox),
            to_cute(w_uv),
            to_cute(self.gate[:n]),
            to_cute(decode.dcp_tot_seq_lens[:reqs], 4),
            to_cute(attn_out[:n]),
            cutlass_int(parity),
            cutlass_int(self.rank),
            stream(self.device),
        )
        return attn.o_proj(attn_out)[0]


def cutlass_int(value):
    import cutlass

    return cutlass.Int32(value)


def _compile_value(heads, size, batch, cap, w_uv, device, sms, q):
    """``mla.dcp.compile_value``'s BF16 DcpValue of ``batch`` rows (``q`` per
    request) for flashinfer's natural-log LSE (``lse_scale`` log2(e))."""
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import make_fake_stream

    from ..cutedsl.common.host import fake, layout
    from ..cutedsl.mla.dcp import DcpValue

    f32, i32, width = torch.float32, torch.int32, heads * 128
    value = DcpValue(heads, size, batch, sms, q, 36, 0, LOG2E, False, "bf16")
    args: tuple = (
        fake((batch, 1, size * heads, 512)),
        fake((batch, 1, size * heads), None, f32, 4),
        fake((batch // q,), None, i32, 4),
        fake((size, cap, heads, 512), None, torch.bfloat16),
        fake((size, cap, heads), None, f32, 4),
        fake((size,), None, torch.int64, 8),
        fake((size,), None, torch.int64, 8),
        fake((cap, size * heads, 576), None, torch.uint8),
        fake(*layout(w_uv)),
        fake((batch, width), (width, 1)),
        fake((batch // q,), None, i32, 4),
        fake((batch, width), (width, 1)),
        Int32(0),
        Int32(0),
        make_fake_stream(),
    )
    with torch.accelerator.device_index(device.index):
        return cute.compile(value, *args)
