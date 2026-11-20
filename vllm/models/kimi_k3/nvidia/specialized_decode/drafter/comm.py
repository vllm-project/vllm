# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's TP collectives on the specialized decode's Lamport mailboxes.

The stock drafter step at TP8 / DCP8 runs eleven flashinfer one-shot all-reduces
(the vocab-parallel embedding, two per draft layer fused with the following RMSNorm,
the final norm: 6-9 us each, their own launch + cluster setup and a clear of the
previous buffer) and two NCCL ring all-gathers (the column-parallel context
projection and the vocab-parallel lm_head logits: 16-20 us each plus a
[TP, M, N / TP] -> [M, N] copy). Every one of them moves a few KB to 1.3 MB.

Here they are one kernel each over symmetric Lamport buffers (`common.mailbox`:
a 16 B fragment is published once none of its 32-bit words is the sentinel):

- `ArNorm` (one 8-CTA cluster per row, 112 threads x 8 columns each): the rank's
  partial row is
  multicast into slot [rank] of every rank's [TP, cap, 7168] mailbox (one
  `multimem.st` reaches every GPU), the TP slots are polled, summed in FP32 in
  rank order 0 .. TP-1 and rounded once to BF16 (flashinfer's deterministic
  one-shot order), the residual is added (one BF16 rounding) and the RMSNorm of the
  new residual follows (FP32 sum of squares, vllm_c's `rsqrt(sum / 7168 + eps)`,
  `x * rstd * w` rounded once). The reader re-arms its slots. Layer parity: the
  call sites alternate between two mailboxes, so a rank can never write a mailbox
  its peer has not read yet (two uses of one mailbox are separated by a use of the
  other, which needs this rank's push, made after its re-arm).
- `Gather`: every rank's [M, S] shard multicast into the [cap, TP * S] row of every
  rank (rank r's at columns r S ..), polled; `copy`: copied to a dense output and
  re-armed, else left in place for a reader that re-arms it (the DCP query mailbox,
  re-armed by `DcpValue`).

Exactness: the reductions keep every operand at stock precision (BF16 partials,
FP32 accumulation, one BF16 rounding; only the order may differ from flashinfer's
cluster-tree RMS sum). The gathers move bytes: a 32-bit word equal to the BF16
sentinel pair (+0, -0) is sent as (+0, +0), the only change, and no consumer can
tell a negative zero from a positive one (sums, products, comparisons, exp).
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.logger import init_logger

from ...specialized_decode.symm import multicast_rendezvous
from ..cutedsl.common.host import architecture, fake
from ..cutedsl.common.primitives import (
    add_rn,
    alloc,
    asm,
    at,
    bf16x2,
    launch,
    load_volatile,
    load_words,
    mul_rn,
    multicast_words,
    published,
    rms_norm_rstd,
    store_relaxed,
    store_words,
    unpack_bf16,
    warp_sum,
    zeros,
)
from ..cutedsl.common.primitives import fma as fma_rn

logger = init_logger(__name__)

HIDDEN = 7168
PAIR_SENT = 0x80000000  # BF16 pair (+0, -0): the AR / gather mailboxes' sentinel
Q_SENT = 0xFFFFFFFF  # four FP8 0xFF (NaN) bytes: the DCP query mailbox's (mla.dcp)
AR_CTAS = 8  # CTAs (one cluster) per all-reduced row
AR_FRAGS = HIDDEN // 8 // AR_CTAS  # 112 16 B fragments per CTA
GATHER_THREADS = 256
# Rows every drafter collective serves (R * (k + 1) context rows, R * k draft rows):
# R <= 16 at k = 4 (64 / 80 rows), R <= 9 at k = 7. Wider batches keep the stock
# collectives (their time is not latency).
ROWS_CAP = 64
CONTEXT_ROWS_CAP = 80


@cute.jit
def clean(word):
    """`word` unless it is the sentinel pair (+0, -0), then (+0, +0)."""
    result = Uint32(word)
    if result == Uint32(PAIR_SENT):
        result = Uint32(0)
    return result


@asm()
def store_peer_float(pointer, value, peer):
    """DSMEM store of an FP32 `value` into cluster CTA `peer`'s `pointer`."""
    code = "{ .reg .b32 dst; mapa.shared::cluster.u32 dst,$0,$2; "
    code += "st.shared::cluster.f32 [dst],$1; }"
    return code, "r,f,r,~{memory}", (pointer, Float32(value), Int32(peer))


class ArNorm:
    """All-reduce of BF16 rows [M, 7168] + residual + RMSNorm (arguments: partial,
    residual, gamma, out, residual out, mailbox [TP, cap, 7168], its multicast
    address, rank, eps, stream). One cluster of 8 CTAs per row, CTA c owning columns
    896 c .. + 895 (thread t < 112: one 16 B fragment): the pushes and polls spread
    over 8 SMs, the row's sum of squares is the 8 CTA sums exchanged through
    distributed shared memory, added in CTA order. `residual` False: the new
    residual is the all-reduced row itself (the embedding: residual = AR(x))."""

    def __init__(self, tp, residual):
        self.tp, self.residual = tp, residual

    @cute.jit
    def __call__(self, *args):
        rows = args[0].shape[0]
        launch(self.kernel, args, (AR_CTAS, rows, 1), 128, cluster=(AR_CTAS, 1, 1))

    @cute.kernel
    def kernel(self, args):
        x, res, gamma, out, res_out, box, mc, rank, eps = args
        tp: cutlass.Constexpr = self.tp
        tid, _, _ = cute.arch.thread_idx()
        cta, row, _ = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        active = tid < AR_FRAGS
        col = (cta * AR_FRAGS + tid) * 8
        smem = cutlass.utils.SmemAllocator()
        sums = alloc(smem, Float32, 4)
        parts = alloc(smem, Float32, AR_CTAS)
        square = Float32(0)
        weight = (Uint32(0),) * 4
        if active:
            weight = load_words(at(gamma, col), count=4)
        # The next kernel (a draft GEMM) launches now: its pre-wait weight loads
        # overlap this all-reduce's NVLink round trip (its reads of our outputs wait
        # for our completion; every CTA here is resident once it triggers).
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        values = cute.make_rmem_tensor(8, Float32)
        if active:
            mine = load_words(at(x, row, col), count=4, mutable=True)
            offset = cute.crd2idx((rank, row, col), box.layout)
            multicast_words(mc + Int64(offset) * 2, tuple(clean(w) for w in mine))
            old = mine
            if cutlass.const_expr(self.residual):
                old = load_words(at(res, row, col), count=4, mutable=True)
            raw = zeros((tp, 4), Uint32)
            pending = Int32((1 << tp) - 1)
            while pending != 0:
                for slot in cutlass.range_constexpr(tp):
                    if (pending & (1 << slot)) != 0:
                        words = load_volatile(at(box, slot, row, col), count=4)
                        for j in cutlass.range_constexpr(4):
                            raw[slot, j] = words[j]
                        if published(words):
                            pending &= ~(1 << slot)
            for slot in cutlass.range_constexpr(tp):
                store_relaxed(at(box, slot, row, col), (Uint32(PAIR_SENT),) * 4)
            new = []
            for j in cutlass.range_constexpr(4):
                lo, hi = unpack_bf16(raw[0, j])
                for slot in cutlass.range_constexpr(1, tp):
                    a, b = unpack_bf16(raw[slot, j])
                    lo, hi = add_rn(lo, a), add_rn(hi, b)
                word = bf16x2(lo, hi)  # the all-reduced BF16 pair
                if cutlass.const_expr(self.residual):
                    a, b = unpack_bf16(word)
                    c, d = unpack_bf16(old[j])
                    word = bf16x2(add_rn(a, c), add_rn(b, d))
                new.append(word)
                values[2 * j], values[2 * j + 1] = unpack_bf16(word)
            store_words(at(res_out, row, col), tuple(new))
            for i in cutlass.range_constexpr(8):
                square = fma_rn(values[i], values[i], square)
        square = warp_sum(square)
        if lane == 0:
            sums[warp] = square
        cute.arch.sync_threads()
        if tid == 0:
            block = add_rn(add_rn(sums[0], sums[1]), add_rn(sums[2], sums[3]))
            for peer in cutlass.range_constexpr(AR_CTAS):
                store_peer_float(parts.iterator + cta, block, peer)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        total = Float32(0)
        for c in cutlass.range_constexpr(AR_CTAS):
            total = add_rn(total, parts[c])
        rstd = rms_norm_rstd(total, eps)
        if active:
            normed = []
            for j in cutlass.range_constexpr(4):
                g0, g1 = unpack_bf16(weight[j])
                normed.append(
                    bf16x2(
                        mul_rn(mul_rn(values[2 * j], rstd), g0),
                        mul_rn(mul_rn(values[2 * j + 1], rstd), g1),
                    )
                )
            store_words(at(out, row, col), tuple(normed))


class Gather:
    """Every rank's [M, S] 32-bit-word shard -> the [cap, TP * S] mailbox rows
    (arguments: shard, out, mailbox, its multicast address, rank, stream). Grid
    (CTAs per row, M): each CTA multicasts its share of the row's shard fragments,
    then polls its share of the gathered row (all pushes precede every poll and the
    grid is at most a few CTAs per SM, so each CTA a poll waits on is resident or
    done). `copy`: the row is copied to `out` and re-armed (`out` is unused
    otherwise)."""

    def __init__(self, tp, words, sentinel, copy, clean):
        assert words % 4 == 0, words
        self.tp, self.frags, self.sentinel = tp, words // 4, sentinel
        self.copy, self.clean = copy, clean

    @cute.jit
    def __call__(self, shard, out, box, mc, rank, ctas, stream):
        args = (shard, out, box, mc, rank, stream)
        launch(self.kernel, args, (ctas, shard.shape[0], 1), GATHER_THREADS)

    @cute.kernel
    def kernel(self, args):
        shard, out, box, mc, rank = args
        tid, _, _ = cute.arch.thread_idx()
        cta, row, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        me, step = cta * GATHER_THREADS + tid, ctas * GATHER_THREADS
        F: cutlass.Constexpr = self.frags
        cute.arch.griddepcontrol_wait()
        for f in range(me, F, step):
            words = load_words(at(shard, row, 4 * f), count=4, mutable=True)
            if cutlass.const_expr(self.clean):
                words = tuple(clean(w) for w in words)
            offset = cute.crd2idx((row, rank * (4 * F) + 4 * f), box.layout)
            multicast_words(mc + Int64(offset) * 4, words)
        cute.arch.griddepcontrol_launch_dependents()
        for f in range(me, F * self.tp, step):
            source = at(box, row, 4 * f)
            words = load_volatile(source, count=4)
            while not published(words, sentinel=self.sentinel):
                words = load_volatile(source, count=4)
            if cutlass.const_expr(self.copy):
                store_words(at(out, row, 4 * f), words)
                store_words(source, (Uint32(self.sentinel),) * 4)


def _compile_options(device):
    return f"--gpu-arch={architecture(device)} --enable-tvm-ffi"


@cache
def compile_ar_norm(tp, residual, cap, device):
    rows = cute.sym_int32()
    row = fake((rows, HIDDEN))
    args: tuple = (row, row, fake((HIDDEN,)), row, row, fake((tp, cap, HIDDEN)))
    args += (Int64(0), Int32(0), Float32(1e-5))
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    with torch.accelerator.device_index(device.index):
        return cute.compile(
            ArNorm(tp, residual), *args, stream, options=_compile_options(device)
        )


@cache
def compile_gather(tp, words, cap, sentinel, copy, clean, device):
    rows = cute.sym_int32()
    i32 = torch.int32
    shard = fake((rows, words), None, i32)
    out = fake((rows, words * tp), None, i32)
    box = fake((cap, words * tp), None, i32)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = Gather(tp, words, sentinel, copy, clean)
    args = (shard, out, box, Int64(0), Int32(0), Int32(1), stream)
    with torch.accelerator.device_index(device.index):
        return cute.compile(kernel, *args, options=_compile_options(device))


def gather_ctas(rows, frags_per_row):
    """CTAs per gathered row: about one fragment per thread, at most ~1.5 waves of
    256-thread CTAs in all (every CTA of the grid must be resident or done)."""
    want = -(-frags_per_row // GATHER_THREADS)
    return max(1, min(want, 32, 192 // max(1, rows)))


class LamportRows:
    """A sentinel-filled symmetric [*shape] int32 buffer with its multicast address."""

    def __init__(self, group, device, shape, sentinel):
        self.buffer = symm_mem.empty(shape, dtype=torch.int32, device=device)
        _, self.mc = multicast_rendezvous(self.buffer, group, "DSpark drafter mailbox")
        if not self.mc:
            raise RuntimeError("the drafter collectives need multicast memory")
        self.buffer.fill_(sentinel - (1 << 32) if sentinel >= 1 << 31 else sentinel)

    def view(self, index=None):
        """(the [index] sub-buffer, its multicast address)."""
        if index is None:
            return self.buffer, self.mc
        sub = self.buffer[index]
        return sub, self.mc + index * sub.numel() * 4


class DrafterComm:
    """The drafter's TP all-reduces and gathers over `group` (the TP device group):
    two parities of the [TP, ROWS_CAP, 7168] all-reduce mailbox, the context-
    projection gather row [CONTEXT_ROWS_CAP, 7168] and the lm_head logits row
    [ROWS_CAP, vocab_padded] (BF16). Built at model load, before the KV-cache memory
    profile and every capture. Methods return None for a shape they do not serve
    (the caller runs the stock collective)."""

    def __init__(self, group, device, tp, rank, vocab_shard=None):
        self.group, self.device, self.tp, self.rank = group, device, tp, rank
        bf16_words = HIDDEN // 2
        self.ar = LamportRows(group, device, (2, tp, ROWS_CAP, bf16_words), PAIR_SENT)
        self.context = LamportRows(
            group, device, (CONTEXT_ROWS_CAP, bf16_words), PAIR_SENT
        )
        self.logits = None
        if vocab_shard and vocab_shard % 8 == 0:
            words = vocab_shard * tp // 2
            self.logits = LamportRows(group, device, (ROWS_CAP, words), PAIR_SENT)
        self.vocab_shard = vocab_shard
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)

    # -- all-reduce + residual + RMSNorm -----------------------------------------

    def ar_norm(self, x, residual, norm, parity):
        """`fused_allreduce_rms_norm(x, residual, norm)` (`residual` None: the
        new residual is AR(x)); None when this shape is not served."""
        rows = x.shape[0]
        if (
            x.dim() != 2
            or x.shape[1] != HIDDEN
            or not 1 <= rows <= ROWS_CAP
            or x.dtype != torch.bfloat16
            or norm.weight.dtype != torch.bfloat16
            or (residual is not None and residual.dtype != torch.bfloat16)
        ):
            return None
        kernel = compile_ar_norm(self.tp, residual is not None, ROWS_CAP, self.device)
        box, mc = self.ar.view(parity)
        x = x.contiguous()
        res = x if residual is None else residual.contiguous()
        out, res_out = torch.empty_like(x), torch.empty_like(x)
        kernel(
            x,
            res,
            norm.weight,
            out,
            res_out,
            box.view(torch.bfloat16),
            mc,
            self.rank,
            float(norm.variance_epsilon),
        )
        return out, res_out

    # -- gathers ------------------------------------------------------------------

    def _gather(self, lamport, shard, cap, out_shape):
        rows, width = shard.shape
        words = width * shard.element_size() // 4
        if not 1 <= rows <= cap or not shard.is_contiguous():
            return None
        kernel = compile_gather(self.tp, words, cap, PAIR_SENT, True, True, self.device)
        box, mc = lamport.view()
        out = shard.new_empty(out_shape)
        frags = words * self.tp // 4
        kernel(
            shard.view(torch.int32),
            out.view(torch.int32),
            box,
            mc,
            self.rank,
            gather_ctas(rows, frags),
        )
        return out

    def gather_context(self, shard):
        """The column-parallel context projection's [M, 7168 / TP] -> [M, 7168]."""
        if shard.dtype != torch.bfloat16 or shard.shape[1] * self.tp != HIDDEN:
            return None
        return self._gather(
            self.context, shard, CONTEXT_ROWS_CAP, (shard.shape[0], HIDDEN)
        )

    def gather_logits(self, shard):
        """The vocab-parallel lm_head's [M, V / TP] -> [M, V] (BF16)."""
        if (
            self.logits is None
            or shard.dtype != torch.bfloat16
            or shard.shape[1] != self.vocab_shard
        ):
            return None
        rows = shard.shape[0]
        return self._gather(
            self.logits, shard, ROWS_CAP, (rows, shard.shape[1] * self.tp)
        )


def build(model, tp_group) -> DrafterComm | None:
    """The drafter's collectives over the TP group (capability fallback: None when
    the platform has no multicast symmetric memory, or at TP 1)."""
    if tp_group.world_size <= 1:
        return None
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        DEFAULT_VOCAB_PADDING_SIZE,
        pad_vocab_size,
    )

    device = next(model.parameters()).device
    # The lm_head (the target's, aliased after this load) is vocab-parallel over the
    # padded draft vocabulary; a different shard at run time takes NCCL's gather.
    config = model.config
    vocab = getattr(config, "draft_vocab_size", None) or config.vocab_size
    padded = pad_vocab_size(vocab, DEFAULT_VOCAB_PADDING_SIZE)
    vocab_shard = padded // tp_group.world_size
    try:
        return DrafterComm(
            tp_group.device_group,
            device,
            tp_group.world_size,
            tp_group.rank_in_group,
            vocab_shard,
        )
    except Exception as exc:  # capability fallback: the stock collectives
        logger.warning_once("DSpark drafter collectives stay stock: %s", exc)
        return None
