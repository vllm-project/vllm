# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark routing for decode batches of at most 8 tokens: sigmoid of the
FP32 router logits plus the correction bias, top-8 of 256 experts, the
selected sigmoid scores renormalized and scaled. Writes the global expert ids
(int32 [b, 8]) and the routing weights (FP32 [b, 8]) that the routed MoE op
takes.

The result is bitwise equal to ``fused_musespark1x_routing`` (the Triton kernel
in vllm/models/muse_spark/common/ops/fused_routing.py) as Muse Spark calls it:
sigmoid scores, no output multiplier (it is folded into the router weight after
loading) and router_score_division_eps = 1e-15. With renormalize the weights
are score * (scale / (sum + 1e-15)), the sum taken in top-k order from 0.0;
without it they are score * scale.

Optionally the same launch also writes the TRT-LLM-gen routing metadata of this
rank's experts for one GEMM tile, as FlashInfer's routingIndicesDynBlockKernel
(the routed MoE op's own routing kernel) writes it, so the routed MoE GEMMs need no
routing kernel of their own: local expert e takes ceil(count_e / tile) CTAs at
the exclusive prefix sum of the CTA counts in expert order, and its routes take
the permuted rows from that CTA offset times tile, in token order. A route to
another rank's expert and the padding rows of each expert's last CTA map to -1.
The histogram and per-expert token counts are not written, nor is any element
past what these fields take (``metadata_size`` leaves a guard element).

Launch: one CTA of 256 threads, one warp per token row, with programmatic
dependent launch: the bias (a weight) is read before ``griddepcontrol.wait``,
the logits after it; the dependents launch at the grid's end. Without metadata
the warps past the last token row exit at once.

Selection: lane l holds experts 8l .. 8l + 7 and sorts them once by a key that
orders as score + bias (NaN is outside the domain; -0.0 keys as +0.0). Each of
the 8 rounds takes the warp maximum of the lanes' head keys and the lowest lane
holding it, matching the Triton argmax (the lowest expert among equal values);
the winner pops its head.

Arithmetic: every float operation is the PTX instruction Triton emits for the
same expression of ``_routing_kernel`` (``primitives.sigmoid_f32``,
``add_rn``, ``mul_rn``, ``div_full_f32``), as inline assembly, so the compiler
can neither contract nor reorder them.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64, const_expr
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..common.host import assert_not_capturing, fp32
from ..common.primitives import (
    add_rn,
    div_full_f32,
    f32_bits,
    load_global_f32x4_address,
    lowest_set_bit,
    mul_rn,
    sigmoid_f32,
    store_global_u32x4,
)

EXPERTS = 256
TOP_K = 8
MAX_TOKENS = 8
WARP = 32
# One warp per token row.
THREADS = WARP * MAX_TOKENS
EXPERTS_PER_LANE = EXPERTS // WARP
WARPS = THREADS // WARP
ROUTES = MAX_TOKENS * TOP_K
# Muse Spark's router_score_division_eps (MuseSpark1xMoE builds its router
# with 1e-15), rounded to FP32 as the Triton kernel receives it.
DIVISION_EPS = 1e-15
_DIVISION_EPS_F32 = fp32(DIVISION_EPS)

# The int32 metadata buffer: offsets of its fields. A CTA holds at least one
# route, so 8 tokens take at most ROUTES CTAs. The offsets keep every field
# 16-byte aligned.
MAX_CTAS = ROUTES
META_PADDED_TOKENS = 0  # [1] CTAs * tile
META_NUM_CTAS = 1  # [1] CTAs
META_EXPANDED_TO_PERMUTED = 4  # [8 b]
META_CTA_BATCH = META_EXPANDED_TO_PERMUTED + ROUTES  # [CTAs]
META_CTA_LIMIT = META_CTA_BATCH + MAX_CTAS  # [CTAs]
META_PERMUTED_TO_TOKEN = META_CTA_LIMIT + MAX_CTAS  # [CTAs * tile]
MAX_TILE = 256
assert EXPERTS == WARP * EXPERTS_PER_LANE == THREADS  # a thread per local expert
assert ROUTES <= THREADS  # a thread per route
assert META_PERMUTED_TO_TOKEN % 4 == 0


def metadata_size(tile_n: int) -> int:
    """int32 elements of the metadata for GEMM tile ``tile_n``: the fields
    above and the permuted-to-token map (MAX_CTAS CTAs of tile_n rows), plus a
    guard element."""
    return META_PERMUTED_TO_TOKEN + MAX_CTAS * tile_n + 1


def _load_row(address: Int64) -> cute.Tensor:
    """A lane's 8 consecutive FP32 at the 16-byte aligned ``address``: two
    16-byte loads, kept in program order around the PDL wait."""
    row = cute.make_rmem_tensor((EXPERTS_PER_LANE,), Float32)
    values = (
        *load_global_f32x4_address(address, ordered=True),
        *load_global_f32x4_address(address + 16, ordered=True),
    )
    for j in range(EXPERTS_PER_LANE):
        row[j] = values[j]
    return row


def _select(cond, if_value, else_value):
    return type(if_value)(cutlass.select_(cond, if_value, else_value))


def _sort_word(value: Float32, j: int) -> Uint64:
    """A word that orders as ``value`` (non-NaN), then the lower index ``j``:
    the order-preserving unsigned key of the value in bits 3..34, 7 - j in
    bits 0..2. -0.0 takes the key of +0.0, as the two compare equal; the key of
    any non-NaN value (-inf included) is above 0, the word of a popped slot.
    Integer operations only, so subnormals keep their order."""
    bits = f32_bits(value)
    flipped = _select(bits < Int32(0), bits ^ Int32(0x7FFFFFFF), bits)
    key = _select((bits & Int32(0x7FFFFFFF)) == Int32(0), Int32(0), flipped)
    unsigned = Uint32(key) ^ Uint32(0x80000000)
    return (Uint64(unsigned) << Uint64(3)) | Uint64(7 - j)


def _compare_exchange(c: list, a: int, b: int) -> None:
    """``c[a], c[b] = max, min``."""
    c[a], c[b] = cutlass.max(c[a], c[b]), cutlass.min(c[a], c[b])


# Batcher's 19-comparator network for 8 values.
_SORT8 = (
    (0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (1, 2),
    (5, 6), (0, 4), (1, 5), (2, 6), (3, 7), (2, 4), (3, 5), (1, 2), (3, 4),
    (5, 6),
)  # fmt: skip


class _RoutingKernel:
    """The routing of up to 8 token rows; ``metadata`` and ``renormalize``
    are compile-time."""

    def __init__(self, metadata: bool, renormalize: bool) -> None:
        self.metadata = metadata
        self.renormalize = renormalize

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        meta: cute.Tensor | None,
        scale: Float32,
        local_offset: Int32,
        num_local: Int32,
        tile: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.routing_shorttail_kernel(
            logits, bias, ids, weights, meta, scale, local_offset, num_local, tile
        ).launch(grid=(1, 1, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=True)

    @cute.kernel
    def routing_shorttail_kernel(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        meta: cute.Tensor | None,
        scale: Float32,
        local_offset: Int32,
        num_local: Int32,
        tile: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        num_tokens = Int32(logits.shape[0])
        # The bias does not change between calls: every warp that routes reads
        # it before the PDL wait, the logits after it.
        bias_at = Int64((bias.iterator + lane * EXPERTS_PER_LANE).toint())

        if const_expr(self.metadata):
            smem = cutlass.utils.SmemAllocator()
            # tokens[e]: bit t set when token t routes to local expert e;
            # cta_offset[e]: its first CTA; warp_ctas[w]: the CTAs of warp
            # w's experts; routes: the selected ids.
            tokens = smem.allocate_tensor(Uint32, cute.make_layout(EXPERTS), 16)
            cta_offset = smem.allocate_tensor(Int32, cute.make_layout(EXPERTS), 16)
            warp_ctas = smem.allocate_tensor(Int32, cute.make_layout(WARPS), 16)
            routes = smem.allocate_tensor(Int32, cute.make_layout(ROUTES), 16)
            if tidx < num_local:
                tokens[tidx] = Uint32(0)
            bias_row = _load_row(bias_at)
            cute.arch.griddepcontrol_wait()
            # tokens[] is zero before any warp sets a bit.
            cute.arch.sync_threads()
            if warp < num_tokens:
                picked_id = self.route_row(
                    logits, bias_row, ids, weights, scale, warp, lane
                )
                if lane < TOP_K:
                    routes[warp * TOP_K + lane] = picked_id
                    expert = picked_id - local_offset
                    # 0 <= expert < num_local, one unsigned compare.
                    if Uint32(expert) < Uint32(num_local):
                        cute.arch.atomic_or(
                            tokens.iterator + expert,
                            Uint32(1) << Uint32(warp),
                            scope="cta",
                        )
            # The selections (tokens[], routes[]) are complete.
            cute.arch.sync_threads()
            self.write_expert_metadata(
                meta, tokens, cta_offset, warp_ctas, num_local, tile, tidx, warp, lane
            )
            # cta_offset[] is complete.
            cute.arch.sync_threads()
            self.write_route_metadata(
                meta, tokens, cta_offset, routes, local_offset, num_local, tile,
                num_tokens, tidx,
            )  # fmt: skip
        else:
            # The warps past the last row exit at once, so that no branch
            # follows the PDL wait (the bias loads stay before it).
            if warp < num_tokens:
                bias_row = _load_row(bias_at)
                cute.arch.griddepcontrol_wait()
                self.route_row(logits, bias_row, ids, weights, scale, warp, lane)

    @cute.jit
    def route_row(
        self,
        logits: cute.Tensor,
        bias_row: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        scale: Float32,
        warp: Int32,
        lane: Int32,
    ) -> Int32:
        """Warp ``warp``: the top-8 of logits row ``warp`` into its ids and
        weights rows; lanes 0..7 return the global id of pick ``lane``."""
        row = logits.iterator + (
            Int64(warp) * logits.stride[0] + lane * EXPERTS_PER_LANE
        )
        g = _load_row(Int64(row.toint()))
        s = [sigmoid_f32(g[j]) for j in range(EXPERTS_PER_LANE)]
        c = [_sort_word(add_rn(s[j], bias_row[j]), j) for j in range(EXPERTS_PER_LANE)]
        # Each lane's 8 (key, local index) words sorted once, best first: a
        # round reads the heads, and only the winning lane pops its head.
        for pair in cutlass.range_constexpr(len(_SORT8)):
            _compare_exchange(c, *_SORT8[pair])

        picked_ids = []
        picked = []
        for _ in cutlass.range_constexpr(TOP_K):
            head = c[0]
            key = Uint32(head >> Uint64(3))
            slot = Int32(Uint32(head) & Uint32(7))  # 7 - local index
            score = s[0]
            for j in cutlass.range_constexpr(1, EXPERTS_PER_LANE):
                score = _select(slot == Int32(7 - j), s[j], score)
            top = cute.arch.warp_redux_sync(key, "umax")
            holders = Uint32(cute.arch.vote_ballot_sync(key == top))
            winner = lowest_set_bit(holders)
            winner_slot = cute.arch.shuffle_sync(slot, winner)
            picked.append(cute.arch.shuffle_sync(score, winner))
            picked_ids.append(winner * 8 + (Int32(7) - winner_slot))
            won = lane == winner
            for j in cutlass.range_constexpr(EXPERTS_PER_LANE - 1):
                c[j] = _select(won, c[j + 1], c[j])
            c[EXPERTS_PER_LANE - 1] = _select(won, Uint64(0), c[EXPERTS_PER_LANE - 1])

        inverse = scale
        if const_expr(self.renormalize):
            total = add_rn(Float32(0.0), picked[0])
            for k in cutlass.range_constexpr(1, TOP_K):
                total = add_rn(total, picked[k])
            inverse = div_full_f32(scale, add_rn(total, Float32(_DIVISION_EPS_F32)))
        mine = picked[0]
        mine_id = picked_ids[0]
        for k in cutlass.range_constexpr(1, TOP_K):
            mine = _select(lane == Int32(k), picked[k], mine)
            mine_id = _select(lane == Int32(k), picked_ids[k], mine_id)
        if lane < TOP_K:
            ids[warp, lane] = mine_id
            weights[warp, lane] = mul_rn(mine, inverse)
        return mine_id

    @cute.jit
    def write_expert_metadata(
        self,
        meta: cute.Tensor,
        tokens: cute.Tensor,
        cta_offset: cute.Tensor,
        warp_ctas: cute.Tensor,
        num_local: Int32,
        tile: Int32,
        expert: Int32,
        warp: Int32,
        lane: Int32,
    ) -> None:
        """Thread e (e < num_local): local expert e's CTAs (batch index and
        row limit) and its permuted rows (token ids in token order, then -1 to
        the end of its last CTA); thread 0 the padded size and CTA count.
        Every thread of the block calls it (it holds a barrier)."""
        mask = Uint32(0)
        if expert < num_local:
            mask = tokens[expert]
        count = Int32(cute.arch.popc(mask))
        ctas = Int32(Uint32(count + tile - 1) // Uint32(tile))
        # Block-wide exclusive prefix sum of ctas over the experts.
        inclusive = ctas
        for shift in cutlass.range_constexpr(5):
            below = cute.arch.shuffle_sync_up(inclusive, 1 << shift)
            inclusive = inclusive + _select(lane >= (1 << shift), below, Int32(0))
        if lane == WARP - 1:
            warp_ctas[warp] = inclusive
        cute.arch.sync_threads()
        before = Int32(0)
        total = Int32(0)
        for w in cutlass.range_constexpr(WARPS):
            ctas_w = warp_ctas[w]
            before = before + _select(warp > w, ctas_w, Int32(0))
            total = total + ctas_w
        cta0 = before + inclusive - ctas
        if expert < num_local:
            cta_offset[expert] = cta0
            row0 = cta0 * tile
            # ctas <= count <= 8: one bit per token.
            for j in cutlass.range_constexpr(MAX_TOKENS):
                if j < ctas:
                    meta[META_CTA_BATCH + cta0 + j] = expert
                    meta[META_CTA_LIMIT + cta0 + j] = cutlass.min(
                        (cta0 + j + 1) * tile, row0 + count
                    )
            for t in cutlass.range_constexpr(MAX_TOKENS):
                if ((mask >> Uint32(t)) & Uint32(1)) != Uint32(0):
                    rank = Int32(cute.arch.popc(mask & Uint32((1 << t) - 1)))
                    meta[META_PERMUTED_TO_TOKEN + row0 + rank] = Int32(t)
            # The padding rows [first, end): 16-byte stores of -1 between the
            # multiples of 4 (META_PERMUTED_TO_TOKEN is one and the buffer is
            # 16-byte aligned), single ones at either end.
            first = row0 + count
            end = (cta0 + ctas) * tile
            body = cutlass.min((first + 3) & Int32(-4), end)
            tail = cutlass.max(end & Int32(-4), body)
            for padded in cutlass.range(first, body, 1, unroll=1):
                meta[META_PERMUTED_TO_TOKEN + padded] = Int32(-1)
            for padded in cutlass.range(body, tail, 4, unroll=4):
                at = meta.iterator + (META_PERMUTED_TO_TOKEN + padded)
                store_global_u32x4(Int64(at.toint()), [Uint32(0xFFFFFFFF)] * 4)
            for padded in cutlass.range(tail, end, 1, unroll=1):
                meta[META_PERMUTED_TO_TOKEN + padded] = Int32(-1)
        if expert == 0:
            meta[META_PADDED_TOKENS] = total * tile
            meta[META_NUM_CTAS] = total

    @cute.jit
    def write_route_metadata(
        self,
        meta: cute.Tensor,
        tokens: cute.Tensor,
        cta_offset: cute.Tensor,
        routes: cute.Tensor,
        local_offset: Int32,
        num_local: Int32,
        tile: Int32,
        num_tokens: Int32,
        route: Int32,
    ) -> None:
        """Thread r (r < 8 b): route r's permuted row, or -1 for another
        rank's expert."""
        if route < num_tokens * TOP_K:
            expert = routes[route] - local_offset
            permuted = Int32(-1)
            if Uint32(expert) < Uint32(num_local):
                earlier = (Uint32(1) << Uint32(route // TOP_K)) - Uint32(1)
                permuted = cta_offset[expert] * tile + Int32(
                    cute.arch.popc(tokens[expert] & earlier)
                )
            meta[META_EXPANDED_TO_PERMUTED + route] = permuted


def compile_routing(metadata: bool, renormalize: bool, options: str = ""):
    """``cute.compile`` the routing kernel of one variant, with these extra
    compile ``options``. Call the result with the tensors of ``route``
    (``meta`` None without metadata), then scale, local offset, local count and
    tile; it launches on the current stream."""
    assert_not_capturing("the routing short tail")
    rows = cute.sym_int32()
    meta = None
    if metadata:
        meta = make_fake_tensor(Int32, (cute.sym_int32(),), (1,), assumed_align=16)
    return cute.compile(
        _RoutingKernel(metadata, renormalize),
        make_fake_tensor(
            Float32,
            (rows, EXPERTS),
            (cute.sym_int64(divisibility=4), 1),
            assumed_align=16,
        ),
        make_fake_tensor(Float32, (EXPERTS,), (1,), assumed_align=16),
        make_fake_tensor(Int32, (rows, TOP_K), (TOP_K, 1), assumed_align=4),
        make_fake_tensor(Float32, (rows, TOP_K), (TOP_K, 1), assumed_align=4),
        meta,
        Float32(1.0),
        Int32(0),
        Int32(EXPERTS),
        Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi {options}".strip(),
    )


@functools.cache
def compiled_routing(metadata: bool, renormalize: bool):
    """``compile_routing`` of one variant, once per process (before CUDA graph
    capture)."""
    return compile_routing(metadata, renormalize)


def prepare(renormalize: bool = True) -> None:
    """Compile the variants ``route`` takes with and without metadata (before
    CUDA graph capture)."""
    for metadata in (False, True):
        compiled_routing(metadata, renormalize)


def route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    scale: float,
    renormalize: bool = True,
    metadata: torch.Tensor | None = None,
    local_expert_offset: int = 0,
    num_local_experts: int = 0,
    tile_n: int = 0,
) -> None:
    """``ids`` [b, 8] (global expert ids) and ``weights`` [b, 8] of the top-8
    of sigmoid(logits) + bias for FP32 ``logits`` [b, 256] (1 <= b <= 8, unit
    column stride, 16-byte aligned rows) and ``bias`` (contiguous, 16-byte
    aligned FP32 [256]), on the current stream of the logits' CUDA device (the
    compiled call switches to it and back). With ``metadata`` (a contiguous,
    16-byte aligned int32 vector of at least ``metadata_size(tile_n)``
    elements, aliasing neither ids nor weights), also the routing metadata of
    the local experts ``[local_expert_offset, local_expert_offset +
    num_local_experts)`` for GEMM tile ``tile_n`` (1 to 256), at the
    ``META_*`` offsets.

    Raises ValueError on arguments outside that contract (some through the
    compiled call's own checks: dtypes, shapes, strides, alignment, devices).
    Allocates nothing; graph-capturable once ``prepare`` compiled the variant.
    """
    num_tokens = logits.shape[0] if logits.dim() == 2 else 0
    if not 1 <= num_tokens <= MAX_TOKENS:
        raise ValueError(
            f"the routing short tail takes fp32 [b, {EXPERTS}] logits with b from 1 "
            f"to {MAX_TOKENS}, got {tuple(logits.shape)}"
        )
    if logits.stride(0) % 4:
        raise ValueError("logits rows must be 16-byte aligned")
    if ids.dtype != torch.int32:
        raise ValueError(f"ids must be int32 [b, {TOP_K}] and contiguous")
    if metadata is not None:
        if not 1 <= tile_n <= MAX_TILE:
            raise ValueError(f"tile_n must be 1 to {MAX_TILE}, got {tile_n}")
        if not (
            num_local_experts >= 1
            and local_expert_offset >= 0
            and local_expert_offset + num_local_experts <= EXPERTS
        ):
            raise ValueError(
                f"local experts [{local_expert_offset}, "
                f"{local_expert_offset + num_local_experts}) outside the "
                f"{EXPERTS} experts"
            )
        if metadata.dtype != torch.int32 or metadata.dim() != 1:
            raise ValueError(
                "metadata must be a contiguous, 16-byte aligned int32 vector"
            )
        if metadata.shape[0] < metadata_size(tile_n):
            raise ValueError(
                f"metadata needs {metadata_size(tile_n)} elements for tile_n "
                f"{tile_n}, got {metadata.shape[0]}"
            )
    compiled_routing(metadata is not None, bool(renormalize))(
        logits,
        bias,
        ids,
        weights,
        metadata,
        scale,
        local_expert_offset,
        num_local_experts,
        tile_n,
    )
