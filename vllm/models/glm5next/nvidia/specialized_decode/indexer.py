# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA indexer stage: the indexer's input projections and per-token query
epilogue in one kernel instead of eleven.

Stock decode runs the indexer key (``wk``), its FP32 head weights (an FP32 copy
of ``weights_proj`` against an FP32 copy of the input) and the kpool gate as a
skinny GEMM, an upcast, a cuBLAS dot with its reduce and a split-K GEMM with its
reduce. Here one GEMM reads the concatenated [wk; weights_proj; gate] rows once and
writes each part in its stock dtype: the products of BF16 values are exact in
FP32, so only the summation order differs. The same kernel then does the work
of ``indexer_query`` (key LayerNorm, Hadamard quant, weight scaling and head
padding; see ``_ProjectionKernel.fused_kernel``); token counts where that is
slower run the projection and ``indexer_query`` as two kernels.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    load_vector,
    load_words,
    mma_into,
    push_float,
    shared_address,
    wait,
)
from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

from .indexer_query import DIM, hadamard_quant, indexer_query, store_weight
from .linear import planned
from .mega_moe import _store_bf16
from .shared_expert import _round_bf16

# DeepGEMM's MQA logits take 32 or 64 heads; fewer are zero-padded.
MQA_MIN_HEADS = 32

ROWS = 8  # weight rows per CTA: the n8 of an m16n8k16 mma
TILE = 16  # tokens per mma: its m16


def _tactic(tokens: int) -> tuple[int, int]:
    """(warps per CTA, CTAs per cluster splitting K) from a GB300 sweep."""
    return (16, 1) if tokens <= 16 else (8, 4)


def _fuse(tokens: int, heads: int) -> bool:
    """Whether the query epilogue runs inside the projection kernel: where the
    projection's own tactic is the fused kernel's (no K split, so the weights
    stay bit-identical) and the GB300 chain after wq_b measured faster."""
    return tokens <= 2 or (heads >= 32 and tokens <= 16)


FUSED_WARPS = 16  # the fused kernel never splits K: one CTA per eight rows
_KEY, _HEAD, _GATE, _NONE = 0, 1, 2, 3  # slot kinds of the fused row order


@asm(Float32)
def _remote_f32(address, rank):
    code = "{ .reg .b32 a; mapa.shared::cluster.u32 a, $1, $2; "
    code += "ld.shared::cluster.f32 $0, [a]; }"
    return code, "=f,r,r", (Uint32(address), Uint32(rank))


class _ProjectionKernel:
    """``x @ weight.T`` on mma.sync: tokens are the m16, weight rows the n8.

    A cluster of ``split`` CTAs owns 8 weight rows, each CTA a K part and each
    warp a ``span`` of it. The dot product does not care which K a fragment
    slot holds, so lane (g, q) keeps the contiguous 16-byte chunks
    ``q * 8 + 32 p`` of the span for weight row g and tokens g and g + 8, and
    the loaded words are the mma fragments as they are. The weights are loaded
    into registers before the PDL wait; the warps' and the cluster's partial
    sums are added in shared memory.
    """

    def __init__(self, rows, k, key, heads, tokens, warps, split, fused=None):
        assert rows % ROWS == 0 and k % (32 * warps * split) == 0
        self.rows, self.k, self.key, self.heads = rows, k, key, heads
        self.gate = rows - key - heads
        self.tokens, self.warps, self.split = tokens, warps, split
        self.span = k // (warps * split)
        self.tiles = (tokens + TILE - 1) // TILE
        self.smem = (warps * self.tiles * 32 * 4 + tokens * ROWS) * 4
        if fused is not None:
            # (padded heads, CTAs incl. the cluster padding, first head CTA,
            # BF16 head weights)
            self.padded, self.ctas, self.first_head, self.bf16 = fused
            self.cluster = key // ROWS  # the key rows' CTAs: one cluster
            assert split == 1 and self.ctas % self.cluster == 0
            self.fused_smem = (
                tokens * (ROWS + 5 + 2 * key // ROWS) + 3 * ROWS
            ) * 4 + 128

    @cute.jit
    def __call__(
        self, w: cute.Tensor, x: cute.Tensor, key: cute.Tensor,
        weights: cute.Tensor, gate: cute.Tensor, st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(w, x, key, weights, gate).launch(
            grid=(self.rows // ROWS * self.split, 1, 1),
            block=(32 * self.warps, 1, 1),
            cluster=(self.split, 1, 1),
            smem=self.smem + 128,
            stream=st,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self, w: cute.Tensor, x: cute.Tensor, key: cute.Tensor,
        weights: cute.Tensor, gate: cute.Tensor,
    ):  # fmt: skip
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        block, _, _ = cute.arch.block_idx()
        cta = block // self.split
        rank = block % self.split
        b = cute.make_rmem_tensor(cute.make_layout((self.span // 8,)), Uint32)
        self.load_weights(w, b, cta, rank, warp, lane)
        smem = cutlass.utils.SmemAllocator()
        partial = alloc(smem, Float32, (self.warps, self.tiles, 32, 4))
        sums = alloc(smem, Float32, (self.tokens * ROWS,))

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        self.gemv(x, b, partial, rank, warp, lane)
        cute.arch.sync_threads()

        for e in cutlass.range(tid, self.tokens * ROWS, 32 * self.warps, unroll=1):
            total = self.warp_total(partial, e)
            if cutlass.const_expr(self.split == 1):
                self.store(key, weights, gate, e // ROWS, cta * ROWS + e % ROWS, total)
            else:
                sums[e] = total
        if cutlass.const_expr(self.split > 1):
            # Rank c adds elements c, c + split, ... over the cluster's K parts.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            base = shared_address(sums.iterator)
            for e in cutlass.range(
                tid * self.split + rank, self.tokens * ROWS,
                32 * self.warps * self.split, unroll=1,
            ):  # fmt: skip
                total = Float32(0.0)
                for c in cutlass.range_constexpr(self.split):
                    total = total + _remote_f32(base + e * 4, c)
                self.store(key, weights, gate, e // ROWS, cta * ROWS + e % ROWS, total)
            # Keep this CTA's sums alive until the cluster has read them.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def load_weights(self, w, b, cta, rank, warp, lane):
        """Lane (g, q)'s weight fragments of row g, loaded before the PDL wait."""
        first = (rank * self.warps + warp) * self.span + lane % 4 * 8
        for p in cutlass.range_constexpr(self.span // 32):
            got = load_vector(
                w.iterator + Int64(cta * ROWS + lane // 4) * self.k + first + p * 32,
                cache="weight",
            )
            for i in cutlass.range_constexpr(4):
                b[p * 4 + i] = got[i]

    @cute.jit
    def gemv(self, x, b, partial, rank, warp, lane):
        """Each warp's partial C tiles of its K span, into ``partial``."""
        g = lane // 4
        pairs = self.span // 32
        first = (rank * self.warps + warp) * self.span + lane % 4 * 8
        both = self.tokens > 8
        for tile in cutlass.range(self.tiles, unroll=1):
            # Padding tokens repeat the last one; their sums are not stored.
            a = cute.make_rmem_tensor(cute.make_layout((2, pairs * 4)), Uint32)
            for h in cutlass.range_constexpr(2 if both else 1):
                t = tile * TILE + h * 8 + g
                if t >= self.tokens:
                    t = self.tokens - 1
                for p in cutlass.range_constexpr(pairs):
                    got = load_words(
                        x.iterator + t * self.k + first + p * 32,
                        count=4, word_type="u32", mutable=True,
                    )  # fmt: skip
                    for i in cutlass.range_constexpr(4):
                        a[h, p * 4 + i] = got[i]
            # Two accumulators halve the dependent mma chain.
            acc = cute.make_rmem_tensor(cute.make_layout((2, 4)), Float32)
            acc.fill(0.0)
            high = 1 if both else 0
            for j in cutlass.range_constexpr(0, pairs * 4, 2):
                mma_into(
                    acc[j // 2 % 2, None],
                    (a[0, j], a[high, j], a[0, j + 1], a[high, j + 1]),
                    (b[j], b[j + 1]), bf16=True,
                )  # fmt: skip
            for i in cutlass.range_constexpr(4):
                partial[warp, tile, lane, i] = acc[0, i] + acc[1, i]

    @cute.jit
    def warp_total(self, partial, e):
        """Element e = (token, row) of C, summed over the warps: it sits in lane
        (token % 8) * 4 + row / 2."""
        t = e // ROWS
        r = e % ROWS
        src = t % 8 * 4 + r // 2
        idx = t % TILE // 8 * 2 + r % 2
        total = Float32(0.0)
        for v in cutlass.range_constexpr(self.warps):
            total = total + partial[v, t // TILE, src, idx]
        return total

    @cute.jit
    def fused_call(
        self, w: cute.Tensor, x: cute.Tensor, gate: cute.Tensor, q: cute.Tensor,
        q_out: cute.Tensor, w_out: cute.Tensor, key: cute.Tensor,
        norm_w: cute.Tensor, norm_b: cute.Tensor, slots: cute.Tensor,
        scale: Float32, eps: Float32, st: cuda.CUstream,
    ):  # fmt: skip
        self.fused_kernel(
            w, x, gate, q, q_out, w_out, key, norm_w, norm_b, slots, scale, eps
        ).launch(
            grid=(self.ctas, 1, 1),
            block=(32 * self.warps, 1, 1),
            cluster=(self.cluster, 1, 1),
            smem=self.smem + self.fused_smem + 128,
            stream=st,
            use_pdl=True,
        )

    @cute.kernel
    def fused_kernel(
        self, w: cute.Tensor, x: cute.Tensor, gate: cute.Tensor, q: cute.Tensor,
        q_out: cute.Tensor, w_out: cute.Tensor, key: cute.Tensor,
        norm_w: cute.Tensor, norm_b: cute.Tensor, slots: cute.Tensor,
        scale: Float32, eps: Float32,
    ):  # fmt: skip
        """The projection in the fused row order (``_fused_layout``) with the
        query epilogue: CTA ``first_head + h`` owns head h's weight row and its
        query (warp w the tokens w, w + warps, ...), and the first cluster holds
        the key rows, whose LayerNorm statistics it combines over DSMEM."""
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        threads = 32 * self.warps
        b = cute.make_rmem_tensor(cute.make_layout((self.span // 8,)), Uint32)
        self.load_weights(w, b, cta, 0, warp, lane)
        smem = cutlass.utils.SmemAllocator()
        partial = alloc(smem, Float32, (self.warps, self.tiles, 32, 4))
        scales = alloc(smem, Float32, (self.tokens,))
        values = alloc(smem, Float32, (self.tokens * ROWS,))
        stats = alloc(smem, Float32, (self.tokens, 2))
        affine = alloc(smem, Float32, (2, ROWS))
        codes = alloc(smem, Int32, (ROWS,))
        peers = alloc(smem, Float32, (self.cluster, self.tokens, 2))
        bar = alloc(smem, Int64, (1,))
        if cta < self.cluster:
            # Armed for every key CTA's (mean, M2) pushes, before the PDL wait.
            if tid == 0:
                cute.arch.mbarrier_init(bar.iterator, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar.iterator, self.cluster * self.tokens * 8
                )
                cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
        if tid < ROWS:
            code = slots[cta * ROWS + tid]
            codes[tid] = code
            if cta < self.cluster:
                affine[0, tid] = norm_w[code & 0xFFFF]
                affine[1, tid] = norm_b[code & 0xFFFF]

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.padded > self.heads):
            spare = self.padded - self.heads
            for i in cutlass.range(
                cta * self.warps + warp, self.tokens * spare,
                self.ctas * self.warps, unroll=1,
            ):  # fmt: skip
                q_out[i // spare, self.heads + i % spare, lane] = Int32(0)
                if lane == 0:
                    store_weight(
                        w_out, i // spare, self.heads + i % spare, Float32(0.0),
                        self.padded, self.bf16,
                    )  # fmt: skip
        rounds = (self.tokens + self.warps - 1) // self.warps
        mine = [j * self.warps + warp for j in range(rounds)]
        head = min(max(cta - self.first_head, 0), self.heads - 1)
        # Issued before the GEMV, so the query loads hide behind it.
        got = [
            load_words(
                q.iterator
                + (min(t, self.tokens - 1) * self.heads + head) * DIM
                + lane * 4,
                count=2,
                word_type="u32",
                mutable=True,
            )  # fmt: skip
            for t in mine
        ]
        self.gemv(x, b, partial, 0, warp, lane)
        if Uint32(cta - self.first_head) < self.heads:
            for j in cutlass.range_constexpr(rounds):
                if mine[j] < self.tokens:
                    word, q_scale = hadamard_quant(got[j], lane)
                    q_out[mine[j], head, lane] = word
                    if lane == 0:
                        scales[mine[j]] = q_scale
        cute.arch.sync_threads()

        for e in cutlass.range(tid, self.tokens * ROWS, threads, unroll=1):
            t = e // ROWS
            code = codes[e % ROWS]
            index = code & 0xFFFF
            kind = code >> 16
            total = self.warp_total(partial, e)
            if kind == _HEAD:
                value = total * scales[t] * scale
                store_weight(w_out, t, index, value, self.padded, self.bf16)
            elif kind == _GATE:
                _store_bf16(gate.iterator + t * self.gate + index, total)
            elif kind == _KEY:
                values[e] = _round_bf16(total)
        if cta < self.cluster:
            self.key_norm(key, values, peers, bar, stats, affine, codes, eps, cta, tid)

    @cute.jit
    def key_norm(self, key, values, peers, bar, stats, affine, codes, eps, rank, tid):
        """Each key CTA pushes its eight rows' (mean, M2) per token to every CTA
        of the cluster, which combine them in rank order (equal counts)."""
        threads = 32 * self.warps
        cute.arch.sync_threads()
        cute.arch.cluster_wait()
        for t in cutlass.range(tid, self.tokens, threads, unroll=1):
            total = Float32(0.0)
            for r in cutlass.range_constexpr(ROWS):
                total = total + values[t * ROWS + r]
            mean = total * Float32(1.0 / ROWS)
            m2 = Float32(0.0)
            for r in cutlass.range_constexpr(ROWS):
                d = values[t * ROWS + r] - mean
                m2 = m2 + d * d
            dst = peers.iterator + (rank * self.tokens + t) * 2
            for peer in cutlass.range_constexpr(self.cluster):
                push_float(dst, (mean, m2), bar.iterator, peer)
        wait(bar.iterator)
        for t in cutlass.range(tid, self.tokens, threads, unroll=1):
            means = [peers[c, t, 0] for c in range(self.cluster)]
            mean = means[0]
            for c in cutlass.range_constexpr(1, self.cluster):
                mean = mean + means[c]
            mean = mean * Float32(1.0 / self.cluster)
            m2 = Float32(0.0)
            for c in cutlass.range_constexpr(self.cluster):
                d = means[c] - mean
                m2 = m2 + peers[c, t, 1] + Float32(ROWS) * d * d
            stats[t, 0] = mean
            stats[t, 1] = cute.math.rsqrt(m2 / Float32(self.key) + eps)
        cute.arch.sync_threads()
        for e in cutlass.range(tid, self.tokens * ROWS, threads, unroll=1):
            t = e // ROWS
            r = e % ROWS
            _store_bf16(
                key.iterator + t * self.key + (codes[r] & 0xFFFF),
                (values[e] - stats[t, 0]) * stats[t, 1] * affine[0, r] + affine[1, r],
            )

    @cute.jit
    def store(self, key, weights, gate, t, row, total):
        if row < self.key:
            _store_bf16(key.iterator + t * self.key + row, total)
        elif row < self.key + self.heads:
            weights[t, row - self.key] = total
        else:
            _store_bf16(
                gate.iterator + t * self.gate + row - self.key - self.heads, total
            )


_compiled: dict = {}


def _fused_layout(key: int, heads: int, gate: int) -> tuple[list[int], int]:
    """(source row of every fused slot, -1 for a zero row; first head CTA).

    The key rows fill the first CTAs (one cluster), then CTA ``first + h`` holds
    head h and up to seven gate rows; the CTAs pad to whole clusters."""
    order = list(range(key))
    first = key // ROWS
    gates = list(range(key + heads, key + heads + gate))
    for h in range(heads):
        order += [key + h] + gates[: ROWS - 1]
        gates = gates[ROWS - 1 :]
        order += [-1] * (-len(order) % ROWS)
    order += gates
    order += [-1] * (-len(order) % (ROWS * first))
    return order, first


class IndexerProjection:
    """(key BF16, head weights FP32, kpool gate BF16) of the layer input, or with
    ``fused`` also the query epilogue of ``indexer_query`` in the same kernel."""

    def __init__(self, indexer):
        self.key = indexer.head_dim
        self.heads = indexer.n_head
        gate = indexer.index_kpool_compress_gate.data
        self.gate = gate.shape[0]
        self.weight = torch.cat([indexer.wk_weights_proj.weight.data, gate])
        rows, k = self.weight.shape
        if self.key % ROWS or self.heads % ROWS or rows % ROWS or k % 1024:
            raise ValueError("indexer projection rows or K do not tile")
        order, self.first_head = _fused_layout(self.key, self.heads, self.gate)
        source = torch.tensor([max(r, 0) for r in order], device=gate.device)
        self.fused_weight = self.weight[source].contiguous()
        self.fused_weight[torch.tensor(order, device=gate.device) < 0] = 0

        def code(row):
            if row < 0:
                return _NONE << 16
            if row < self.key:
                return _KEY << 16 | row
            if row < self.key + self.heads:
                return _HEAD << 16 | row - self.key
            return _GATE << 16 | row - self.key - self.heads

        self.slots = torch.tensor(
            [code(r) for r in order], dtype=torch.int32, device=gate.device
        )

    def __call__(self, x: torch.Tensor):
        m, k = x.shape
        assert x.is_contiguous()
        key = torch.empty((m, self.key), dtype=torch.bfloat16, device=x.device)
        weights = torch.empty((m, self.heads), dtype=torch.float32, device=x.device)
        gate = torch.empty((m, self.gate), dtype=torch.bfloat16, device=x.device)
        args = (self.weight, x, key, weights, gate)
        st = stream(x.device)
        rows = self.weight.shape[0]
        cache = (rows, k, self.heads, m, x.device.index)
        if cache not in _compiled:
            kernel = _ProjectionKernel(rows, k, self.key, self.heads, m, *_tactic(m))
            _compiled[cache] = cute.compile(kernel, *(to_cute(t, 16) for t in args), st)
        _compiled[cache](*(to_cute(t, 16) for t in args), st)
        return key, weights, gate

    def fused(self, x, q, norm, scale: float, padded: int, weights_dtype=torch.float32):
        """(q_fp8 [M, padded, 128], scaled head weights [M, padded], normed key
        [M, 128] BF16, kpool gate [M, 128] BF16), as ``__call__`` followed by
        ``indexer_query``."""
        m, k = x.shape
        assert x.is_contiguous() and q.is_contiguous()
        dev = x.device
        q_out = torch.empty((m, padded, DIM), dtype=torch.float8_e4m3fn, device=dev)
        w_out = torch.empty((m, padded), dtype=weights_dtype, device=dev)
        key = torch.empty((m, self.key), dtype=torch.bfloat16, device=dev)
        gate = torch.empty((m, self.gate), dtype=torch.bfloat16, device=dev)
        words = q_out.view(torch.int32).view(m, padded, DIM // 4)
        args = (self.fused_weight, x, gate, q, words, w_out, key, norm.weight,
                norm.bias, self.slots)  # fmt: skip
        st = stream(dev)
        rows = self.weight.shape[0]
        bf16 = weights_dtype == torch.bfloat16
        cache = ("fused", rows, k, self.heads, padded, bf16, m, dev.index)
        if cache not in _compiled:
            kernel = _ProjectionKernel(
                rows, k, self.key, self.heads, m, FUSED_WARPS, 1,
                fused=(padded, self.slots.numel() // ROWS, self.first_head, bf16),
            )  # fmt: skip
            _compiled[cache] = cute.compile(
                kernel.fused_call, *(to_cute(t, 16) for t in args),
                Float32(scale), Float32(norm.eps), st,
            )  # fmt: skip
        _compiled[cache](
            *(to_cute(t, 16) for t in args), Float32(scale), Float32(norm.eps), st
        )
        return q_out, w_out, key, gate


class IndexerStage:
    """``Indexer.forward`` with the fused projection inside planned steps.

    Installed as the indexer's ``forward``: the DSA attention calls it from its
    stock module, and every call outside a planned step runs the stock forward.
    """

    def __init__(self, indexer):
        if indexer.rope_dim != 0:
            raise ValueError("the fused indexer path assumes a NoPE indexer")
        self.indexer = indexer
        self.stock = indexer.forward
        self.projection = IndexerProjection(indexer)
        # The SM100 MQA logits take BF16 head weights: writing them in BF16
        # saves the stock FP32 -> BF16 cast kernel, bit for bit.
        self.weights_dtype = (
            torch.bfloat16 if mqa_logits_use_mx_scales() else torch.float32
        )
        indexer.forward = self.forward

    def forward(self, hidden_states, qr, positions, rotary_emb):
        if not planned():
            return self.stock(hidden_states, qr, positions, rotary_emb)
        ind = self.indexer
        q, _ = ind.wq_b(qr)
        scale = ind.softmax_scale * ind.n_head**-0.5
        padded = max(ind.n_head, MQA_MIN_HEADS)
        dtype = self.weights_dtype
        if _fuse(hidden_states.shape[0], ind.n_head):
            q_fp8, weights, k, gate_score = self.projection.fused(
                hidden_states, q, ind.k_norm, scale, padded, dtype
            )
        else:
            k, weights, gate_score = self.projection(hidden_states)
            q_fp8, weights, k = indexer_query(
                q, k, weights, ind.k_norm, scale, padded, dtype
            )
        return ind.indexer_op(
            hidden_states,
            q_fp8,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=ind.index_kpool_compress_ape,
            index_kpool=ind.index_kpool,
            positions=positions,
        )
