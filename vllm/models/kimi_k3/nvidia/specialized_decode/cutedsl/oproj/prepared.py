# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output projection fused with its TP all-reduce and post-attention AttnRes: rank
partials go into Lamport slots of a multicast [TP, 64, 7168] mailbox (parity alternates
by layer) and the consumer sums and re-arms them. TP16 B<=8 fuses producer and consumer
(MLA layers below B5 keep them split); above B8 the producer runs 16-token groups.
At `twoshot_batches`, given `TwoShot` buffers, the all-reduce is `common.twoshot`
(chunks pushed to their column owner, owner sums + multicasts, all Lamport-polled), so
each GPU receives ~2 x 14 KB per token instead of TP x 14 KB. `bind_ahead`: fused
rows that also compute the next layer's AttnRes statistics (`consumer` ahead)."""

from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import make_fake_stream

from ....specialized_decode.symm import multicast_rendezvous, peer_rendezvous
from ..common.host import dynamic_fake, dynamic_view, fake, stream, to_cute
from ..common.policy import batch_table
from .consumer import OprojConsumer, OprojFused
from .producer import OprojProducer, OprojUmma


class OprojMailbox:
    """Sentinel-armed multicast mailbox; defaults to [parity, TP, 64, 7168] BF16.
    `multicast=False`: a peer-store mailbox (`handle.buffer_ptrs` only, `mc`
    0), which spends no NVLink multicast object."""

    def __init__(self, group, device, tp, rank, shape=None, multicast=True):
        self.tp, self.rank = tp, rank
        shape = (2, tp, 64, 7168) if shape is None else shape
        self.buffer = symm_mem.empty(shape, dtype=torch.bfloat16, device=device)
        if multicast:
            self.handle, self.mc = multicast_rendezvous(
                self.buffer, group, "Kimi K3 o-projection mailbox"
            )
        else:
            self.handle, self.mc = peer_rendezvous(self.buffer, group), 0
        if multicast and not self.mc:  # slot() offsets this address
            raise RuntimeError(
                "Kimi K3 specialized o-projection needs multicast memory"
            )
        self.buffer.view(torch.int32).fill_(-0x80000000)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)

    def slot(self, parity):
        return self.buffer[parity], self.mc + parity * self.buffer[0].numel() * 2


# Two-shot all-reduce batches per TP (2-/4-node GB300, 32 MLA layers, us/layer, max over
# ranks, multicast vs two-shot: TP8 B16 8.85/11.40, B32 10.63/11.32, B64 18.06/13.18;
# TP16 B16 9.78/12.28, B32 14.96/13.05, B64 28.06/14.07). TP8 keeps multicast up to 40
# rows (eight requests at k = 4): decode-step A/B at 131K context, per token, -2.9% /
# -0.6% at 35 / 40 rows.
TWOSHOT = {8: "41-64:1", 16: "17-64:1"}


def twoshot_batches(tp):
    table = batch_table(TWOSHOT.get(tp, ""))
    return {batch for batch, on in table.items() if on}


def engine(batch, tp):
    """Producer engine above B8 (GB300): grouped mma.sync for TP8 up to B24, else
    tcgen05. TP8 verify steps at k = 4, decode-step A/B at 131K context, per token:
    mma.sync -2.3% at 20 rows (four requests), +2.1% at 25-30 rows."""
    default = "mma" if tp == 8 and batch <= 24 else "umma"
    return default


def producer(k, batch, tp, mla, twoshot=False, kind=None):
    """B<=8: one mma.sync pass, TP8 stages 8 (MLA 4) of 12 K blocks per warp. Above
    B8 (`kind` overrides `engine`): the tcgen05 engine, or 2-tile warps over 8 K
    parts in 16-token groups, TP8 staging 2 of 6 K blocks per warp."""
    if batch <= 8:
        staged = 0 if tp == 16 else 4 if mla else 8
        return OprojProducer(k, staged_blocks=staged, twoshot=twoshot)
    if (kind or engine(batch, tp)) == "umma":
        return OprojUmma(k, twoshot=twoshot)
    grouped = dict(rows=2, group_tiles=2, twoshot=twoshot)
    return OprojProducer(k, staged_blocks=0 if tp == 16 else 2, **grouped)


def compile_oproj(
    kind, k, batch, tp, nb, has_delta, mla, front=False, cap=64, twoshot=False, ahead=0
):
    """Producer, consumer or fused kernel over a [TP, cap, 7168] parity view;
    `front`: the consumer also publishes x_moe [1, 4, 7168] and the epoch hint.
    `twoshot`: the two-shot pair (arguments as `TwoShot.views`). `ahead`: the
    fused kernel also stores the next layer's AttnRes statistics into [ahead, 7168] /
    [ahead, 2] FP32 rows (after the AttnRes rows: next norm, next query, the two
    rows, next eps). Each kind is cached on the arguments it reads, so layers that
    differ only in the others share one compiled kernel."""
    if kind.startswith("producer"):  # producer_mma / producer_umma: engine A/B
        return _compile_producer(kind, k, batch, tp, mla, cap, twoshot)
    if kind == "consumer":  # B64: one 128-thread group per CTA polls every rank
        groups = 4 if batch <= 32 and not twoshot else 1
        return _compile_consumer(tp, nb, has_delta, front, cap, twoshot, groups)
    return _compile_fused(k, batch, tp, nb, has_delta, front, cap, ahead)


def _box(tp, cap, twoshot):
    return fake((1 if twoshot else tp, cap, 7168), (cap * 7168, 7168, 1))


def _peers(tp, cap):
    return fake((tp, cap, 7168 // tp)), fake((tp,), None, torch.int64)


def _rows(front):
    row, weight = dynamic_fake(2), fake((7168,), (1,))
    xmoe = fake((1, 4, 7168), (4 * 7168, 7168, 1)) if front else None
    hints = fake((224,), (1,), torch.int32) if front else None
    eps = cutlass.Float32(1e-6)
    return (row, dynamic_fake(3), weight, weight, weight, row, xmoe, hints, eps, eps)


@cache
def _compile_producer(kind, k, batch, tp, mla, cap, twoshot):
    x, w = fake((batch, k), (k, 1)), fake((7168, k), (k, 1))
    kernel = producer(k, batch, tp, mla, twoshot, kind[9:] or None)
    if twoshot:
        tail = (*_peers(tp, cap), cutlass.Int32(0))
    else:
        tail = (_box(tp, cap, twoshot), cutlass.Int64(0), cutlass.Int32(0))
    return cute.compile(kernel, x, w, *tail, make_fake_stream())


@cache
def _compile_consumer(tp, nb, has_delta, front, cap, twoshot, groups):
    kernel = OprojConsumer(nb, groups=groups, twoshot=twoshot, has_delta=has_delta)
    extra = (_peers(tp, cap)[0], cutlass.Int64(0), cutlass.Int32(0)) if twoshot else ()
    box = _box(tp, cap, twoshot)
    return cute.compile(kernel, box, *_rows(front), *extra, make_fake_stream())


@cache
def _compile_fused(k, batch, tp, nb, has_delta, front, cap, ahead):
    x, w = fake((batch, k), (k, 1)), fake((7168, k), (k, 1))
    kernel = OprojFused(nb, has_delta=has_delta, ahead=ahead > 0)
    rows = _rows(front)
    if ahead:
        weight, eps = fake((7168,), (1,)), cutlass.Float32(1e-6)
        stats = (fake((ahead, n), None, torch.float32) for n in (7168, 2))
        rows += (weight, weight, *stats, eps)
    box, mc, slot = _box(tp, cap, False), cutlass.Int64(0), cutlass.Int32(0)
    return cute.compile(kernel, x, w, box, mc, slot, *rows, make_fake_stream())


class PreparedOproj:
    """One decoder layer's o-projection + all-reduce + post-attention AttnRes."""

    def __init__(self, decoder, mailbox, batches, mla, front=None, twoshot=None):
        weight = decoder.self_attn.o_proj.weight
        norm, out_norm = decoder.mlp_res_norm, decoder.post_attention_layernorm
        weights = (norm.weight, decoder.mlp_res_proj.weight.squeeze(0), out_norm.weight)
        self.views = tuple(to_cute(w.reshape(-1)) for w in weights)
        self.eps = tuple(cutlass.Float32(n.variance_epsilon) for n in (norm, out_norm))
        parity, rank = decoder.layer_idx % 2, cutlass.Int32(mailbox.rank)
        local_box, mc = mailbox.slot(parity)
        self.device, self.box = weight.device, to_cute(local_box)
        self.head = (self.box, cutlass.Int64(mc), rank)
        self.twoshot: Any = None
        if twoshot is not None:
            recv, peers, gather, gather_mc = twoshot.views(parity)
            recv = to_cute(recv)
            self.twoshot = (
                (recv, to_cute(peers), rank),
                to_cute(gather),
                (recv, cutlass.Int64(gather_mc), rank),
            )
        self.wview, tp, k = to_cute(weight), mailbox.tp, weight.shape[1]
        nb = decoder.prev_valid_blocks + decoder.is_block_write_layer
        delta = not decoder.is_block_write_layer
        self.front = None if front is None else (*map(to_cute, front[:2]), front[2])
        self.codes, self.twoshots = {}, twoshot_batches(tp) if self.twoshot else set()
        self.shapes: dict[int, tuple] = {}  # fused rows: compile_oproj arguments
        self.ahead: dict[int, Any] = {}
        self.ahead_views: tuple = ()
        for b in batches:
            publish = front is not None and b in front[2]
            fused = tp == 16 and (not mla or b >= 5) and b <= 8
            kinds = ("fused",) if fused else ("producer", "consumer")
            twoshot = b in self.twoshots and not fused
            shape = (k, b, tp, nb, delta, mla)
            self.codes[b] = tuple(
                compile_oproj(kind, *shape, publish and kind != "producer", 64, twoshot)
                for kind in kinds
            )
            if fused:
                self.shapes[b] = (*shape, publish)

    def bind_ahead(self, target, mixture, scal, rows):
        """The fused `rows` also compute `target`'s AttnRes statistics into
        `mixture` [cap, 7168] / `scal` [cap, 2] FP32; `target` must read the
        saved blocks of this kernel's AttnRes. Returns the rows bound."""
        norm, query = target.self_attention_res_norm, target.self_attention_res_proj
        weights = (to_cute(w.reshape(-1)) for w in (norm.weight, query.weight))
        stats = to_cute(mixture), to_cute(scal)
        eps = cutlass.Float32(norm.variance_epsilon)
        self.ahead_views = (*weights, *stats, eps)
        cap = mixture.shape[0]
        for b in sorted(rows):
            shape = self.shapes.get(b)
            if shape is None or b > cap or shape[3] == 0:
                continue
            k, batch, tp, nb, delta, mla, publish = shape
            self.ahead[b] = compile_oproj(
                "fused", k, batch, tp, nb, delta, mla, publish, 64, False, cap
            )
        return tuple(sorted(self.ahead))

    def run(self, local, prefix, blocks, out):
        batch, s = local.shape[0], stream(self.device)
        codes, x = self.codes[batch], to_cute(local)
        front = (None, None)
        if self.front is not None and batch in self.front[2]:
            front = self.front[:2]
        rows = (dynamic_view(prefix), dynamic_view(blocks), *self.views)
        rows += (dynamic_view(out), *front, *self.eps, s)
        if batch in self.ahead:
            stats = (*rows[:-1], *self.ahead_views, s)
            self.ahead[batch](x, self.wview, *self.head, *stats)
        elif len(codes) == 1:
            codes[0](x, self.wview, *self.head, *rows)
        elif batch in self.twoshots:
            head, gather, tail = self.twoshot
            codes[0](x, self.wview, *head, s)
            codes[1](gather, *rows[:-1], *tail, s)
        else:
            codes[0](x, self.wview, *self.head, s)
            codes[1](self.box, *rows)
        return out, prefix, blocks
