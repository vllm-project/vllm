# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 MoE kernels: top-k gate, the tile-padded expert row layout of the
expert GEMMs (`trtllm_bmm`) with the activations quantized into it, and the
top-k combine.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton

from .block_scales import _scale_offsets
from .linear import _k_group_tile, _prefetch_k_group_tile, k_group_partials
from .mxfp8 import _mxfp8_block, _store_mxfp8_block
from .nvfp4 import FP32_MAX, _quantize_block, _quantize_columns, _store_block
from .seq_parallel import HALO_ROWS, SeqParallel
from .trtllm_bmm import ExpertPlan


@triton.jit
def _sum8(x):
    # ((x0 + x2) + (x1 + x3)) + ((x4 + x6) + (x5 + x7)) of each row of [R, 8]:
    # the summation order of the reference renormalization kernel, so routing
    # weights match bitwise.
    low, high = tl.split(tl.permute(tl.reshape(x, (x.shape[0], 2, 2, 2)), (0, 1, 3, 2)))
    low, high = tl.split(low + high)
    low, high = tl.split(low + high)
    return low + high


@triton.jit
def _selection_keys(logits, bias, experts, E: tl.constexpr):
    """Sort keys of the experts `experts` (padding: 0): order-preserving
    bits of sigmoid(logit) + bias in the high word and the reversed expert
    index in the low bits, so ties prefer the lower expert index."""
    sigmoid = tl.div_rn(1.0, 1.0 + tldevice.exp(-logits))
    selection = sigmoid + bias
    bits = selection.to(tl.uint32, bitcast=True)
    bits = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)
    keys = (bits.to(tl.uint64) << 16) | (E - experts).to(tl.uint64)
    return tl.where(experts < E, keys, 0)


@triton.jit
def _key_values(values, keys, top):
    """`values` [R, C] where `keys` [R, C] equals each of `top` [R, K]."""
    hit = keys[:, None, :] == top[:, :, None]
    return tl.max(tl.where(hit, values[:, None, :], float("-inf")), 2)


@triton.jit
def _log_sigmoid(logits):
    return tl.minimum(logits, 0.0) - tldevice.log1p(tldevice.exp(-tl.abs(logits)))


@triton.jit
def _routing_weights(log_probs, route_scale, TOPK: tl.constexpr):
    """Renormalized sigmoid probabilities [R, TOPK] of the selected experts
    from their log-sigmoids."""
    max_log_probs = tl.max(log_probs, 1)
    shifted = tldevice.exp(log_probs - max_log_probs[:, None])
    total = _sum8(shifted) if TOPK == 8 else tl.sum(shifted, 1)
    logsumexp = max_log_probs + tldevice.log(total)
    return tldevice.exp(log_probs - logsumexp[:, None]) * route_scale


@triton.jit
def _group_sum(Partials, rows, cols, mask, T, E: tl.constexpr, GROUPS: tl.constexpr):
    """In-order sum of the K-group partials `[GROUPS, T, E]` at `(rows,
    cols)`."""
    total = tl.load(Partials + rows * E + cols, mask, other=0.0, cache_modifier=".cg")
    for group in tl.static_range(1, GROUPS):
        total = total + tl.load(
            Partials + (group * T + rows) * E + cols,
            mask,
            other=0.0,
            cache_modifier=".cg",
        )
    return total


@triton.jit
def _load_bias(Bias, E: tl.constexpr, E_POW2: tl.constexpr):
    """The selection bias [1, E_POW2] (padding: 0) for `_select_experts`; a
    weight, so it may load before `gdc_wait`."""
    offs = tl.arange(0, E_POW2)[None, :]
    return tl.load(Bias + offs, offs < E, other=0.0)


@triton.jit
def _top_keys(
    Partials,
    bias,
    tokens,
    T,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
):
    """The logits and `_selection_keys` [R, E_POW2] of tokens `tokens` [R]
    and their top-k keys [R, TOPK] (`_select_experts`)."""
    offs = tl.arange(0, E_POW2)[None, :]
    logits = _group_sum(Partials, tokens[:, None], offs, offs < E, T, E, GROUPS)
    keys = _selection_keys(logits, bias, offs, E)
    return logits, keys, tl.topk(keys, TOPK)


@triton.jit
def _select_experts(
    Partials,
    bias,
    tokens,
    T,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
):
    """Top-k ids and routing weights [R, TOPK] of tokens `tokens` [R] from
    the K-group logit partials `[GROUPS, T, E]` and the `_load_bias`
    `bias`, by `tl.topk` (`SORT`) or k max passes."""
    rows = tokens[:, None]
    offs = tl.arange(0, E_POW2)[None, :]
    if SORT:
        logits, keys, top = _top_keys(
            Partials, bias, tokens, T, E, E_POW2, GROUPS, TOPK
        )
        ids = E - (top & 0xFFFF).to(tl.int32)
        selected = _key_values(logits, keys, top)
    else:
        logits = _group_sum(Partials, rows, offs, offs < E, T, E, GROUPS)
        keys = _selection_keys(logits, bias, offs, E)
        # Fewer instructions than the bitonic top-k: better for large batches.
        slots = tl.arange(0, TOPK)[None, :]
        ids = tl.zeros((tokens.shape[0], TOPK), tl.int32)
        for k in tl.static_range(TOPK):
            best = tl.max(keys, 1)[:, None]
            ids = tl.where(slots == k, E - (best & 0xFFFF).to(tl.int32), ids)
            keys = tl.where(keys == best, 0, keys)
        # The selected logits, summed again in the same order.
        selected = _group_sum(Partials, rows, ids, ids >= 0, T, E, GROUPS)
    return ids, _routing_weights(_log_sigmoid(selected), route_scale, TOPK)


# Completion counters of the routing kernels, one 128-byte line apart.
_ROUTERS_DONE = tl.constexpr(0)
_TOKENS_DONE = tl.constexpr(32)
NUM_COUNTERS = 64


@triton.jit
def _spin_until(Counter, value):
    while tl.atomic_add(Counter, 0, sem="acquire") < value:
        pass


@triton.jit
def _local_experts(ids, lo, hi, padding):
    """Local index of the experts `ids` [R, TOPK] among the local experts
    `[lo, hi)` (-1: not local, or the token is `padding`; None: none
    is)."""
    local = tl.where((ids >= lo) & (ids < hi), ids - lo, -1)
    if padding is not None:
        local = tl.where(padding[:, None], -1, local)
    return local


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _place_experts(
    Counts,
    Starts,
    Tiles,
    NUM_LOCAL: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    TILE: tl.constexpr,
    MAX_CTAS: tl.constexpr,
):
    """Lays out the experts' rows (`Counts`, then reset) in `TILE`-row
    tiles: stores each expert's first layout row, first routed row and end
    row (`Starts`) and the `Tiles` metadata."""
    experts = tl.arange(0, LOCAL_POW2)
    mask = experts < NUM_LOCAL
    counts = tl.load(Counts + experts, mask, other=0, cache_modifier=".cg")
    tiles = tl.cdiv(counts, TILE)
    tile_ends = tl.cumsum(tiles, 0)
    starts = (tile_ends - tiles) * TILE
    tl.store(Starts + experts, starts, mask)
    tl.store(Starts + LOCAL_POW2 + experts, tl.cumsum(counts, 0) - counts, mask)
    tl.store(Starts + 2 * LOCAL_POW2 + experts, starts + counts)
    tl.store(Counts + experts, 0, mask)
    num_tiles = tl.max(tile_ends, 0)
    tl.store(Tiles + 2 * MAX_CTAS, num_tiles)
    tl.store(Tiles + 2 * MAX_CTAS + 1, num_tiles * TILE)
    tl.store(Tiles + 2 * MAX_CTAS + 2, tl.sum(counts, 0))
    # The expert of each tile: mark each expert's first tile, then take the
    # running maximum. Tiles past the last are left as they are.
    ctas = tl.arange(0, triton.next_power_of_2(MAX_CTAS))
    tl.store(Tiles + ctas, 0, ctas < num_tiles)
    tl.debug_barrier()
    tl.store(Tiles + tile_ends - tiles, experts, tiles > 0)
    tl.debug_barrier()
    expert = tl.load(Tiles + ctas, ctas < num_tiles, cache_modifier=".cg")
    expert = tl.associative_scan(expert, 0, _maximum)
    end = tl.load(
        Starts + 2 * LOCAL_POW2 + expert, ctas < num_tiles, cache_modifier=".cg"
    )
    tl.store(Tiles + ctas, expert, ctas < num_tiles)
    tl.store(
        Tiles + MAX_CTAS + ctas, tl.minimum((ctas + 1) * TILE, end), ctas < num_tiles
    )


@triton.jit
def _place_rows(
    Starts,
    Ids,
    Ranks,
    Weights,
    Src2Dst,
    Gammas,
    Rows,
    token,
    TOPK: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
):
    """Places a token's local rows (`_place_experts`); returns their layout
    rows (-1: not local) and local experts."""
    slots = token * TOPK + tl.arange(0, TOPK)
    local = tl.load(Ids + slots)
    rank = tl.load(Ranks + slots)
    routed = local >= 0
    dst = tl.where(routed, tl.load(Starts + local, routed) + rank, -1)
    tl.store(Src2Dst + slots, dst)
    tl.store(Gammas + dst, tl.load(Weights + slots), routed)
    tl.store(Rows + tl.load(Starts + LOCAL_POW2 + local, routed) + rank, dst, routed)
    return dst, local


@triton.jit
def _place_token(
    masks,
    mine,
    local,
    weights,
    Src2Dst,
    Gammas,
    Rows,
    Tiles,
    token,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
    MAX_CTAS: tl.constexpr,
):
    """Places the local rows of token `token` (`_place_rows`) for batches
    of at most `TILE` tokens, whose experts take one tile each, from the
    tokens of each local expert (`masks`, a bit per token; `mine`: those
    of the token's experts `local`); stores the token's part of the
    `route` outputs and returns its rows (-1: not local)."""
    experts = tl.arange(0, masks.shape[0])
    sizes = tldevice.popc(masks)
    below = experts[None, :] < local[:, None]
    tile = tl.sum((below & (masks[None, :] != 0)).to(tl.int32), 1)
    rank = tldevice.popc(mine & ((1 << token) - 1))
    count = tldevice.popc(mine)
    dst = tl.where(local >= 0, tile * TILE + rank, -1)
    slots = token * TOPK + tl.arange(0, TOPK)
    tl.store(Src2Dst + slots, dst)
    tl.store(Gammas + dst, weights, dst >= 0)
    tl.store(Rows + tl.sum(tl.where(below, sizes[None, :], 0), 1) + rank, dst, dst >= 0)
    expert_first = (rank == 0) & (local >= 0)
    tl.store(Tiles + tile, local, expert_first)
    tl.store(Tiles + MAX_CTAS + tile, tile * TILE + count, expert_first)
    if token == 0:
        num_tiles = tl.sum((masks != 0).to(tl.int32), 0)
        tl.store(Tiles + 2 * MAX_CTAS, num_tiles)
        tl.store(Tiles + 2 * MAX_CTAS + 1, num_tiles * TILE)
        tl.store(Tiles + 2 * MAX_CTAS + 2, tl.sum(sizes, 0))
    return dst


@triton.jit
def _pick(values, index):
    """`values[index]` for a small vector `values` and a scalar `index`."""
    return tl.sum(tl.where(tl.arange(0, values.shape[0]) == index, values, 0), 0)


@triton.jit
def _load_blocks(X, BLOCK: tl.constexpr):
    """fp32 `X[:BLOCK]` (bf16), loaded so that each thread holds whole
    32-column blocks."""
    i = tl.arange(0, 4)[:, None, None]
    b = tl.arange(0, BLOCK // 32)[None, :, None]
    j = tl.arange(0, 8)[None, None, :]
    x = tl.load(X + 32 * b + 8 * i + j)
    return tl.reshape(tl.permute(x, (1, 0, 2)), (BLOCK,)).to(tl.float32)


@triton.jit
def _quantize_x(
    X,
    token,
    H: tl.constexpr,
    NVFP4: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    """Quantizes row `token` of `X` (columns `[0, HEAD)` and `[HEAD,
    HEAD + TAIL)`); returns the values and block scales of both and the row
    amax (NVFP4)."""
    head = _load_blocks(X + token * H, HEAD)
    tail = head
    if TAIL > 0:
        tail = _load_blocks(X + token * H + HEAD, TAIL)
    amax = 0.0
    if NVFP4:
        amax = tl.max(tl.abs(head), 0)
        if TAIL > 0:
            amax = tl.maximum(amax, tl.max(tl.abs(tail), 0))
        encoder_scale = tl.minimum(tl.div_rn(1536.0, amax), FP32_MAX)
        decoder_scale = tl.div_rn(1.0, encoder_scale)
        outer_scale = amax * (1.0 / 1536.0)
        head_values, head_scales = _quantize_block(
            head, encoder_scale, decoder_scale, outer_scale, HEAD
        )
        tail_values, tail_scales = head_values, head_scales
        if TAIL > 0:
            tail_values, tail_scales = _quantize_block(
                tail, encoder_scale, decoder_scale, outer_scale, TAIL
            )
    else:
        head_values, head_scales = _mxfp8_block(tl.reshape(head, (HEAD // 32, 32)))
        tail_values, tail_scales = head_values, head_scales
        if TAIL > 0:
            tail_values, tail_scales = _mxfp8_block(tl.reshape(tail, (TAIL // 32, 32)))
    return head_values, head_scales, tail_values, tail_scales, amax


@triton.jit
def _row_scales(amax, WAmax, WDenominator, local, NVFP4: tl.constexpr):
    """The gate/up output scales of a row with amax `amax` in experts
    `local` (NVFP4: (x_amax * w_amax) * rcp(1536 * denominator))."""
    if NVFP4:
        w_amax = tl.load(WAmax + local, local >= 0, other=0.0)
        rcp_denominator = tldevice.rcp_rn(1536.0 * tl.load(WDenominator))
        return (amax * w_amax) * rcp_denominator
    return tl.zeros(local.shape, tl.float32)


@triton.jit
def _unpack_bf16(words):
    """The 4 bf16 of each uint64 of `words` as fp32, lowest first."""
    low = words.to(tl.uint32)
    high = (words >> 32).to(tl.uint32)
    return (
        (low << 16).to(tl.float32, bitcast=True),
        (low & 0xFFFF0000).to(tl.float32, bitcast=True),
        (high << 16).to(tl.float32, bitcast=True),
        (high & 0xFFFF0000).to(tl.float32, bitcast=True),
    )


@triton.jit
def _load_columns(X, blocks, mask):
    """The 16 columns (a tuple of fp32 [32]) of the 16-column blocks
    `blocks` of the bf16 row `X`, 16 bytes per load."""
    words = X.to(tl.pointer_type(tl.uint64), bitcast=True) + blocks[:, None] * 4
    pair = tl.arange(0, 2)[None, :]
    a, b = tl.split(tl.load(words + pair, mask[:, None], other=0))
    c, d = tl.split(tl.load(words + 2 + pair, mask[:, None], other=0))
    return _unpack_bf16(a) + _unpack_bf16(b) + _unpack_bf16(c) + _unpack_bf16(d)


@triton.jit
def _row_amax(X, H: tl.constexpr):
    """Max |X[:H]| of a bf16 row, in a single-warp program."""
    lanes = tl.arange(0, 32)
    amax = tl.zeros((32,), tl.float32)
    for step in range((H // 16 + 31) // 32):
        blocks = step * 32 + lanes
        x = _load_columns(X, blocks, blocks < H // 16)
        for j in tl.static_range(16):
            amax = tl.maximum(amax, tl.abs(x[j]))
    return tl.max(amax, 0)


@triton.jit
def _quantize_row(X, amax, QValues, QScales, rows, H: tl.constexpr, TILE: tl.constexpr):
    """Quantizes the bf16 row `X` with row amax `amax` to NVFP4 into each
    of the rows `rows` (a tuple; negative: skipped), in a single-warp
    program whose lanes each take whole blocks."""
    encoder_scale = tl.minimum(tl.div_rn(1536.0, amax), FP32_MAX)
    decoder_scale = tl.div_rn(1.0, encoder_scale)
    outer_scale = amax * (1.0 / 1536.0)
    words = QValues.to(tl.pointer_type(tl.uint64), bitcast=True)
    lanes = tl.arange(0, 32)
    for step in range((H // 16 + 31) // 32):
        blocks = step * 32 + lanes
        mask = blocks < H // 16
        values, scales = _quantize_columns(
            _load_columns(X, blocks, mask),
            encoder_scale,
            decoder_scale,
            outer_scale,
        )
        for i in tl.static_range(len(rows)):
            if rows[i] >= 0:
                tl.store(words + rows[i] * (H // 16) + blocks, values, mask)
                tl.store(
                    QScales + _scale_offsets(rows[i], blocks, H // 16, TILE),
                    scales,
                    mask,
                )


@triton.jit
def _store_rows(
    Values,
    Scales,
    rows,
    start,
    values,
    scales,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    PACK: tl.constexpr,
    GROUP: tl.constexpr,
    TILE: tl.constexpr,
):
    """Stores the quantized columns `[start, start + BLOCK)` (`PACK` per
    value, one scale per `GROUP`) to each of `rows` (negative: skipped)."""
    valid = rows[:, None] >= 0
    cols = start // PACK + tl.arange(0, BLOCK // PACK)
    tl.store(
        Values + rows[:, None] * (K // PACK) + cols[None, :],
        tl.broadcast_to(
            tl.reshape(values, (BLOCK // PACK,))[None, :],
            (rows.shape[0], BLOCK // PACK),
        ),
        valid,
    )
    groups = start // GROUP + tl.arange(0, BLOCK // GROUP)
    tl.store(
        Scales + _scale_offsets(rows[:, None], groups[None, :], K // GROUP, TILE),
        tl.broadcast_to(scales[None, :], (rows.shape[0], BLOCK // GROUP)),
        valid,
    )


@triton.jit
def _store_x(
    QValues,
    QScales,
    QRowScales,
    quantized,
    dst,
    row_scales,
    H: tl.constexpr,
    NVFP4: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
    TILE: tl.constexpr,
    BROADCAST: tl.constexpr,
):
    """Stores a `_quantize_x` row to rows `dst` (negative: skipped), with
    their `row_scales` (NVFP4). `BROADCAST` stores to all rows at once
    (lower latency); otherwise to one local row after another (fewer
    instructions)."""
    head_values, head_scales, tail_values, tail_scales, amax = quantized
    PACK: tl.constexpr = 2 if NVFP4 else 1
    GROUP: tl.constexpr = 16 if NVFP4 else 32
    if BROADCAST:
        rows = dst.to(tl.int64)
        if NVFP4:
            tl.store(QRowScales + rows, row_scales, rows >= 0)
        _store_rows(
            QValues,
            QScales,
            rows,
            0,
            head_values,
            head_scales,
            H,
            HEAD,
            PACK,
            GROUP,
            TILE,
        )
        if TAIL > 0:
            _store_rows(
                QValues,
                QScales,
                rows,
                HEAD,
                tail_values,
                tail_scales,
                H,
                TAIL,
                PACK,
                GROUP,
                TILE,
            )
    else:
        for slot in range(dst.shape[0]):
            row = _pick(dst, slot).to(tl.int64)
            if row >= 0:
                if NVFP4:
                    tl.store(QRowScales + row, _pick(row_scales, slot))
                    _store_block(
                        QValues,
                        QScales,
                        row,
                        0,
                        head_values,
                        head_scales,
                        H,
                        HEAD,
                        TILE,
                    )
                    if TAIL > 0:
                        _store_block(
                            QValues,
                            QScales,
                            row,
                            HEAD,
                            tail_values,
                            tail_scales,
                            H,
                            TAIL,
                            TILE,
                        )
                else:
                    _store_mxfp8_block(
                        QValues,
                        QScales,
                        row,
                        0,
                        head_values,
                        head_scales,
                        H,
                        HEAD,
                        TILE,
                    )
                    if TAIL > 0:
                        _store_mxfp8_block(
                            QValues,
                            QScales,
                            row,
                            HEAD,
                            tail_values,
                            tail_scales,
                            H,
                            TAIL,
                            TILE,
                        )


# Batches up to this many tokens route and quantize in one launch.
FUSED_ROUTING_MAX_TOKENS = 16
# From this many tokens, NVFP4 rows are quantized by single-warp programs.
NVFP4_SCATTER_MIN_TOKENS = 2048
# Up to this many tokens, kernels let the next kernel launch (and wait) as soon
# as they start, which hides its launch; with more, its waiting programs would
# hold back theirs.
LAUNCH_DEPENDENTS_MAX_TOKENS = 256


@triton.jit
def _fused_route_kernel(
    X,
    GateWeight,
    Bias,
    Partials,
    ExpertBounds,
    Padding,
    Counters,
    Masks,
    Src2Dst,
    Gammas,
    Rows,
    Tiles,
    QValues,
    QScales,
    QRowScales,
    WAmax,
    WDenominator,
    T,
    route_scale,
    H: tl.constexpr,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_LOCAL: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    TILE: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    NVFP4: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    # Programs [0, ROUTERS) compute the router K-group partials. Program
    # ROUTERS + t quantizes token t meanwhile, waits for them, selects the
    # token's experts and marks it in the token mask of each of its local
    # experts (``Masks``; a single token needs no masks); once all tokens are
    # marked, it stores the token into its place. The next kernel launches at
    # exit: the expert GEMM reads the number of tiles before it waits. The
    # router weights are prefetched before the wait.
    pid = tl.program_id(0)
    N_TILES: tl.constexpr = (E + BN - 1) // BN
    ROUTERS: tl.constexpr = N_TILES * GROUPS
    if pid < ROUTERS:
        _prefetch_k_group_tile(GateWeight, E, H, BN, pid % N_TILES, pid // N_TILES)
    bias = _load_bias(Bias, E, E_POW2)
    lo = tl.load(ExpertBounds)
    hi = tl.load(ExpertBounds + 1)
    tl.extra.cuda.gdc_wait()
    if pid < ROUTERS:
        _k_group_tile(
            X,
            GateWeight,
            Partials,
            T,
            E,
            H,
            BM,
            BN,
            True,
            pid % N_TILES,
            0,
            pid // N_TILES,
        )
        tl.debug_barrier()
        tl.atomic_add(Counters + _ROUTERS_DONE, 1, sem="release")
    else:
        token = pid - ROUTERS
        tokens = tl.full((1,), token, tl.int64)
        padding = None
        if Padding is not None:
            padding = tl.load(Padding + tokens)
        quantized = _quantize_x(X, token.to(tl.int64), H, NVFP4, HEAD, TAIL)
        _spin_until(Counters + _ROUTERS_DONE, ROUTERS)
        logits, keys, top = _top_keys(
            Partials, bias, tokens, T, E, E_POW2, GROUPS, TOPK
        )
        ids = E - (top & 0xFFFF).to(tl.int32)
        local = tl.reshape(_local_experts(ids, lo, hi, padding), (TOPK,))
        experts = tl.arange(0, LOCAL_POW2)
        if T > 1:
            tl.atomic_or(Masks + local, 1 << token, local >= 0, sem="relaxed")
            tl.debug_barrier()
            tl.atomic_add(Counters + _TOKENS_DONE, 1, sem="release")
        # The routing weights, while the other tokens are marked.
        weights = _routing_weights(
            _log_sigmoid(_key_values(logits, keys, top)), route_scale, TOPK
        )
        if T > 1:
            _spin_until(Counters + _TOKENS_DONE, T)
            masks = tl.load(
                Masks + experts, experts < NUM_LOCAL, other=0, cache_modifier=".cg"
            )
            mine = tl.load(Masks + local, local >= 0, other=0, cache_modifier=".cg")
        else:
            masks = tl.max((experts[:, None] == local[None, :]).to(tl.int32), 1)
            mine = (local >= 0).to(tl.int32)
        dst = _place_token(
            masks,
            mine,
            local,
            tl.reshape(weights, (TOPK,)),
            Src2Dst,
            Gammas,
            Rows,
            Tiles,
            token,
            TOPK,
            TILE,
            MAX_CTAS,
        )
        # The last program to get here resets the masks and counters: the
        # others have used them (the barrier follows the stores of the
        # placement).
        last = T == 1
        if T > 1:
            tl.debug_barrier()
            last = tl.atomic_add(Counters + _TOKENS_DONE, 1, sem="relaxed") == 2 * T - 1
        _store_x(
            QValues,
            QScales,
            QRowScales,
            quantized,
            dst,
            _row_scales(quantized[4], WAmax, WDenominator, local, NVFP4),
            H,
            NVFP4,
            HEAD,
            TAIL,
            TILE,
            True,
        )
        if last:
            if T > 1:
                tl.store(Masks + experts, 0, experts < NUM_LOCAL)
            tl.store(Counters + _ROUTERS_DONE, 0)
            tl.store(Counters + _TOKENS_DONE, 0)


@triton.jit
def _route_tokens_kernel(
    Partials,
    Bias,
    ExpertBounds,
    Padding,
    Counts,
    Counters,
    Ids,
    Ranks,
    Weights,
    Starts,
    Tiles,
    T,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
    NUM_LOCAL: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    TILE: tl.constexpr,
    MAX_CTAS: tl.constexpr,
):
    # Selects a token's experts (stores their local index, -1: not local or
    # padding) and ranks its local rows within their experts (in ``Counts``,
    # zeroed beforehand). The last program to finish lays out the experts'
    # rows. The scatter kernel may start meanwhile: it reads the tokens before
    # it waits.
    bias = _load_bias(Bias, E, E_POW2)
    lo = tl.load(ExpertBounds)
    hi = tl.load(ExpertBounds + 1)
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    token = tl.full((1,), tl.program_id(0), tl.int64)
    padding = None
    if Padding is not None:
        padding = tl.load(Padding + token)
    ids, weights = _select_experts(
        Partials, bias, token, T, route_scale, E, E_POW2, GROUPS, TOPK, SORT
    )
    local = _local_experts(ids, lo, hi, padding)
    slots = token[:, None] * TOPK + tl.arange(0, TOPK)[None, :]
    rank = tl.atomic_add(Counts + local, 1, mask=local >= 0, sem="relaxed")
    tl.store(Ranks + slots, rank)
    tl.store(Ids + slots, local)
    tl.store(Weights + slots, weights)
    tl.debug_barrier()
    if tl.atomic_add(Counters + _TOKENS_DONE, 1, sem="acq_rel") == T - 1:
        _place_experts(Counts, Starts, Tiles, NUM_LOCAL, LOCAL_POW2, TILE, MAX_CTAS)
        tl.store(Counters + _TOKENS_DONE, 0)


@triton.jit
def _scatter_kernel(
    X,
    Starts,
    Ids,
    Ranks,
    Weights,
    Src2Dst,
    Gammas,
    Rows,
    QValues,
    QScales,
    QRowScales,
    WAmax,
    WDenominator,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    TILE: tl.constexpr,
    NVFP4: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
    BROADCAST: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    # The tokens are complete before the routing kernel starts.
    quantized = _quantize_x(X, token, H, NVFP4, HEAD, TAIL)
    tl.extra.cuda.gdc_wait()
    if LAUNCH_DEPENDENTS:
        tl.extra.cuda.gdc_launch_dependents()
    dst, local = _place_rows(
        Starts, Ids, Ranks, Weights, Src2Dst, Gammas, Rows, token, TOPK, LOCAL_POW2
    )
    _store_x(
        QValues,
        QScales,
        QRowScales,
        quantized,
        dst,
        _row_scales(quantized[4], WAmax, WDenominator, local, NVFP4),
        H,
        NVFP4,
        HEAD,
        TAIL,
        TILE,
        BROADCAST,
    )


@triton.jit
def _scatter_nvfp4_kernel(
    X,
    Starts,
    Ids,
    Ranks,
    Weights,
    Src2Dst,
    Gammas,
    Rows,
    QValues,
    QScales,
    QRowScales,
    WAmax,
    WDenominator,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    TILE: tl.constexpr,
):
    # One token per single-warp program, each lane quantizing whole blocks.
    token = tl.program_id(0).to(tl.int64)
    x = X + token * H
    # The tokens are complete before the routing kernel starts.
    amax = _row_amax(x, H)
    tl.extra.cuda.gdc_wait()
    dst, local = _place_rows(
        Starts, Ids, Ranks, Weights, Src2Dst, Gammas, Rows, token, TOPK, LOCAL_POW2
    )
    row_scales = _row_scales(amax, WAmax, WDenominator, local, True)
    tl.store(QRowScales + dst, row_scales, dst >= 0)
    tl.static_assert(TOPK == 8)
    rows = (
        _pick(dst, 0).to(tl.int64),
        _pick(dst, 1).to(tl.int64),
        _pick(dst, 2).to(tl.int64),
        _pick(dst, 3).to(tl.int64),
        _pick(dst, 4).to(tl.int64),
        _pick(dst, 5).to(tl.int64),
        _pick(dst, 6).to(tl.int64),
        _pick(dst, 7).to(tl.int64),
    )
    _quantize_row(x, amax, QValues, QScales, rows, H, TILE)


def route(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    counters: torch.Tensor,
    counts: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    plan: ExpertPlan,
    weight_amax: torch.Tensor | None,
    weight_denominator: torch.Tensor | None,
    is_padding: torch.Tensor | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
]:
    """Router, top-k and the tile-padded layout of the rows of the local
    experts `expert_bounds`, with `x` quantized (NVFP4 with
    `weight_amax`, else MXFP8) into it.

    Returns `(src2dst, gammas, rows, tiles)`: the layout row of each flat
    (token * topk + slot) row (-1: not local), the routing weight of each
    layout row, the routed layout rows in expert order and the `tiles`
    metadata of `trtllm_bmm.bmm` followed by the number of routed rows; and
    the quantized rows, their block scales and, for NVFP4, their gate/up
    projection output scales. `counters` (`NUM_COUNTERS`) and `counts`
    (one per local expert) are zeroed int32s that the kernels reset. Tokens
    marked in the bool `is_padding` have no local rows on any rank: their
    `src2dst` is -1, so they combine to 0.
    """
    tokens, hidden = x.shape
    assert is_padding is None or (
        is_padding.shape == (tokens,) and is_padding.dtype == torch.bool
    )
    experts = gate_weight.shape[0]
    assert x.is_contiguous() and gate_weight.is_contiguous()
    nvfp4 = weight_amax is not None
    num_local = counts.numel()
    n = tokens * topk
    device = x.device
    rows = plan.rows
    src2dst = torch.empty(n, dtype=torch.int32, device=device)
    gammas = torch.empty(rows, dtype=torch.float32, device=device)
    routed = torch.empty(n, dtype=torch.int32, device=device)
    tiles = torch.empty(2 * plan.max_ctas + 3, dtype=torch.int32, device=device)
    ids = torch.empty(n, dtype=torch.int32, device=device)
    ranks = torch.empty(n, dtype=torch.int32, device=device)
    weights = torch.empty(n, dtype=torch.float32, device=device)
    if nvfp4:
        quantized = (
            torch.empty((rows, hidden // 2), dtype=torch.uint8, device=device),
            torch.empty((rows, hidden // 16), dtype=torch.float8_e4m3fn, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
        )
    else:
        quantized = (
            torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((rows, hidden // 32), dtype=torch.uint8, device=device),
            None,
        )
    outputs = (src2dst, gammas, routed, tiles)
    # Rows are quantized in a power-of-two head and tail.
    head = 1 << (hidden.bit_length() - 1)
    tail = hidden - head
    assert tail & (tail - 1) == 0 and (tail == 0 or tail >= 32)
    if tokens <= FUSED_ROUTING_MAX_TOKENS:
        assert plan.tile >= tokens
        groups = triton.cdiv(hidden, 1024)
        block_n = 32
        partials = torch.empty(
            (groups, tokens, experts), dtype=torch.float32, device=device
        )
        routers = triton.cdiv(experts, block_n) * groups
        # Token programs wait for the router programs: launch all co-resident.
        _fused_route_kernel[(routers + tokens,)](
            x,
            gate_weight,
            bias,
            partials,
            expert_bounds,
            is_padding,
            counters,
            counts,
            *outputs,
            *quantized,
            weight_amax,
            weight_denominator,
            tokens,
            route_scale,
            H=hidden,
            E=experts,
            E_POW2=triton.next_power_of_2(experts),
            GROUPS=groups,
            HEAD=head,
            TAIL=tail,
            BM=16,
            BN=block_n,
            TOPK=topk,
            NUM_LOCAL=num_local,
            LOCAL_POW2=triton.next_power_of_2(num_local),
            TILE=plan.tile,
            MAX_CTAS=plan.max_ctas,
            NVFP4=nvfp4,
            num_warps=4,
            num_stages=4,
            enable_fp_fusion=False,
            launch_cooperative_grid=True,
            launch_pdl=True,
        )
        return outputs, quantized

    partials = k_group_partials(x, gate_weight)
    local_pow2 = triton.next_power_of_2(num_local)
    starts = torch.empty(3 * local_pow2, dtype=torch.int32, device=device)
    _route_tokens_kernel[(tokens,)](
        partials,
        bias,
        expert_bounds,
        is_padding,
        counts,
        counters,
        ids,
        ranks,
        weights,
        starts,
        tiles,
        tokens,
        route_scale,
        E=experts,
        E_POW2=triton.next_power_of_2(experts),
        GROUPS=partials.shape[0],
        TOPK=topk,
        SORT=tokens <= 128,
        NUM_LOCAL=num_local,
        LOCAL_POW2=local_pow2,
        TILE=plan.tile,
        MAX_CTAS=plan.max_ctas,
        num_warps=4 if tokens <= 128 else 1,
        launch_pdl=True,
    )
    if nvfp4 and tokens >= NVFP4_SCATTER_MIN_TOKENS:
        _scatter_nvfp4_kernel[(tokens,)](
            x,
            starts,
            ids,
            ranks,
            weights,
            src2dst,
            gammas,
            routed,
            *quantized,
            weight_amax,
            weight_denominator,
            H=hidden,
            TOPK=topk,
            LOCAL_POW2=local_pow2,
            TILE=plan.tile,
            num_warps=1,
            enable_fp_fusion=False,
            launch_pdl=True,
        )
        return outputs, quantized
    _scatter_kernel[(tokens,)](
        x,
        starts,
        ids,
        ranks,
        weights,
        src2dst,
        gammas,
        routed,
        *quantized,
        weight_amax,
        weight_denominator,
        H=hidden,
        TOPK=topk,
        LOCAL_POW2=local_pow2,
        TILE=plan.tile,
        NVFP4=nvfp4,
        HEAD=head,
        TAIL=tail,
        BROADCAST=tokens < (4096 if nvfp4 else 512),
        LAUNCH_DEPENDENTS=tokens <= LAUNCH_DEPENDENTS_MAX_TOKENS,
        num_warps=2,
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    return outputs, quantized


@triton.jit
def _combine_kernel(
    Down,
    Src2Dst,
    Out,
    SlotBases,
    tokens,
    shard,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
    PUSH: tl.constexpr,
    HALO_ROWS: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    if LAUNCH_DEPENDENTS:
        tl.extra.cuda.gdc_launch_dependents()
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(TOPK):
        row = tl.load(Src2Dst + token * TOPK + slot).to(tl.int64)
        acc += tl.load(Down + row * H + cols, row >= 0, other=0).to(tl.float32)
    if PUSH:
        # Into the owner's slot of this rank, after its halo rows; the last
        # halo rows of a shard also into the next owner's.
        owner = token // shard
        local = token - owner * shard
        dst = tl.load(SlotBases + owner).to(tl.pointer_type(tl.bfloat16))
        tl.store(dst + (local + HALO_ROWS) * H + cols, acc.to(tl.bfloat16))
        if local >= shard - HALO_ROWS and (owner + 1) * shard < tokens:
            dst = tl.load(SlotBases + owner + 1).to(tl.pointer_type(tl.bfloat16))
            tl.store(dst + (local - shard + HALO_ROWS) * H + cols, acc.to(tl.bfloat16))
    else:
        tl.store(Out + token * H + cols, acc)


def combine(
    down: torch.Tensor,
    src2dst: torch.Tensor,
    topk: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Sum each token's rows of `down` (rows -1 count 0)."""
    tokens = src2dst.numel() // topk
    out = torch.empty((tokens, down.shape[1]), dtype=out_dtype, device=down.device)
    _combine(down, src2dst, topk, out, None)
    return out


def combine_push(
    down: torch.Tensor, src2dst: torch.Tensor, topk: int, sp: SeqParallel
) -> None:
    """`combine` into this rank's slots of the token shard owners
    (`SeqParallel.pushed`), in bf16."""
    tokens = src2dst.numel() // topk
    assert down.shape[1] == sp.hidden_size and tokens <= sp.max_tokens
    # Every shard is longer than the halo.
    assert tokens - (sp.world_size - 1) * sp.shard(tokens)[2] > HALO_ROWS
    _combine(down, src2dst, topk, None, sp)


def _combine(
    down: torch.Tensor,
    src2dst: torch.Tensor,
    topk: int,
    out: torch.Tensor | None,
    sp: SeqParallel | None,
) -> None:
    hidden = down.shape[1]
    tokens = src2dst.numel() // topk
    block = 512
    assert hidden % block == 0
    _combine_kernel[(tokens, hidden // block)](
        down,
        src2dst,
        out,
        None if sp is None else sp.slot_bases,
        tokens,
        1 if sp is None else sp.shard(tokens)[2],
        hidden,
        topk,
        block,
        tokens <= LAUNCH_DEPENDENTS_MAX_TOKENS,
        sp is not None,
        HALO_ROWS,
        num_warps=2,
        launch_pdl=True,
    )
