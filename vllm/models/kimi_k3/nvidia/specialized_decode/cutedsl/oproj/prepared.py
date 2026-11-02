# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output projection fused with its TP all-reduce and post-attention AttnRes: rank
partials go into Lamport slots of a multicast [TP, 64, 7168] mailbox (parity alternates
by layer) and the consumer sums and re-arms them. TP16 B<=8 fuses producer and consumer
(MLA layers below B5 keep them split); above B8 the producer runs 16-token groups.
At ``twoshot_batches``, given ``TwoShot`` buffers, the all-reduce is ``common.twoshot``
(chunks pushed to their column owner, owner sums + multicasts, all Lamport-polled), so
each GPU receives ~2 x 14 KB per token instead of TP x 14 KB."""

from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import make_fake_stream

from ..common.host import dynamic_fake, dynamic_view, fake, stream, to_cute
from ..common.policy import batch_table
from .consumer import OprojConsumer, OprojFused
from .producer import OprojProducer, OprojUmma


class OprojMailbox:
    """Sentinel-armed multicast mailbox; defaults to [parity, TP, 64, 7168] BF16."""

    def __init__(self, group, device, tp, rank, shape=None):
        self.tp, self.rank = tp, rank
        shape = (2, tp, 64, 7168) if shape is None else shape
        self.buffer = symm_mem.empty(shape, dtype=torch.bfloat16, device=device)
        self.handle = handle = symm_mem.rendezvous(self.buffer, group)
        self.mc = int(handle.multicast_ptr or 0)
        if not self.mc:  # slot() offsets this address
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
# TP16 B16 9.78/12.28, B32 14.96/13.05, B64 28.06/14.07).
TWOSHOT = {8: "33-64:1", 16: "17-64:1"}


def twoshot_batches(tp):
    table = batch_table(TWOSHOT.get(tp, ""))
    return {batch for batch, on in table.items() if on}


def engine(batch, tp):
    """Producer engine above B8 (GB300): grouped mma.sync for TP8 up to B16, else
    tcgen05."""
    default = "mma" if tp == 8 and batch <= 16 else "umma"
    return default


def producer(k, batch, tp, mla, twoshot=False, kind=None):
    """B<=8: one mma.sync pass, TP8 stages 8 (MLA 4) of 12 K blocks per warp. Above
    B8 (``kind`` overrides ``engine``): the tcgen05 engine, or 2-tile warps over 8 K
    parts in 16-token groups, TP8 staging 2 of 6 K blocks per warp."""
    if batch <= 8:
        staged = 0 if tp == 16 else 4 if mla else 8
        return OprojProducer(k, staged_blocks=staged, twoshot=twoshot)
    if (kind or engine(batch, tp)) == "umma":
        return OprojUmma(k, twoshot=twoshot)
    grouped = dict(rows=2, group_tiles=2, twoshot=twoshot)
    return OprojProducer(k, staged_blocks=0 if tp == 16 else 2, **grouped)


@cache
def compile_oproj(
    kind, k, batch, tp, nb, has_delta, mla, front=False, cap=64, twoshot=False
):
    """Producer, consumer or fused kernel over a [TP, cap, 7168] parity view;
    ``front``: the consumer also publishes x_moe [1, 4, 7168] and the epoch hint.
    ``twoshot``: the two-shot pair (arguments as ``TwoShot.views``)."""
    box = fake((1 if twoshot else tp, cap, 7168), (cap * 7168, 7168, 1))
    x, w = fake((batch, k), (k, 1)), fake((7168, k), (k, 1))
    mc, slot, eps = cutlass.Int64(0), cutlass.Int32(0), cutlass.Float32(1e-6)
    recv, peers = fake((tp, cap, 7168 // tp)), fake((tp,), None, torch.int64)
    if kind.startswith("producer"):  # producer_mma / producer_umma: engine A/B
        kernel = producer(k, batch, tp, mla, twoshot, kind[9:] or None)
        tail = (recv, peers, slot) if twoshot else (box, mc, slot)
        return cute.compile(kernel, x, w, *tail, make_fake_stream())
    row, weight = dynamic_fake(2), fake((7168,), (1,))
    xmoe = fake((1, 4, 7168), (4 * 7168, 7168, 1)) if front else None
    hints = fake((224,), (1,), torch.int32) if front else None
    rows = (row, dynamic_fake(3), weight, weight, weight, row, xmoe, hints, eps, eps)
    if kind == "consumer":  # B64: one 128-thread group per CTA polls every rank
        options = dict(has_delta=has_delta)
        groups = 4 if batch <= 32 and not twoshot else 1
        kernel = OprojConsumer(nb, groups=groups, twoshot=twoshot, **options)
        extra = (recv, mc, slot) if twoshot else ()
        return cute.compile(kernel, box, *rows, *extra, make_fake_stream())
    kernel = OprojFused(nb, has_delta=has_delta)
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

    def run(self, local, prefix, blocks, out):
        batch, s = local.shape[0], stream(self.device)
        codes, x = self.codes[batch], to_cute(local)
        front = (None, None)
        if self.front is not None and batch in self.front[2]:
            front = self.front[:2]
        rows = (dynamic_view(prefix), dynamic_view(blocks), *self.views)
        rows += (dynamic_view(out), *front, *self.eps, s)
        if len(codes) == 1:
            codes[0](x, self.wview, *self.head, *rows)
        elif batch in self.twoshots:
            head, gather, tail = self.twoshot
            codes[0](x, self.wview, *head, s)
            codes[1](gather, *rows[:-1], *tail, s)
        else:
            codes[0](x, self.wview, *self.head, s)
            codes[1](self.box, *rows)
        return out, prefix, blocks
