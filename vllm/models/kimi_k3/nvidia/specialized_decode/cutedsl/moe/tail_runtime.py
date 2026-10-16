# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed MoE tail: one fused kernel finalizes this rank's top-16 rows, multicasts the
latent partial and peer-stores the shared shard (three rotating parities), all-reduces
and RMS-normalizes per K-slice, runs the K-split up projection and multicasts the output
row into the next AttnRes mailbox. At TP8 B5+ and TP16 B9+ the experts publish the
latent partial, WaveMoeTail.push the shared shard and WaveMoeTail does the rest. At
``twoshot_batches`` the latent all-reduce is ``common.twoshot``: the experts push each
partial chunk to its column owner, whose WaveMoeTail CTAs sum and multicast it."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, prepare_shared_carveout, stream, tensor_maps, to_cute
from ..common.policy import batch_table
from ..common.twoshot import TwoShot
from ..oproj.prepared import OprojMailbox
from .tail import MoeTail, WaveMoeTail

# MoE tail mailbox tokens (padded decode batches); small-batch front_poll G/W/S rows per
# parity.
CAP, POLL_CAP = 64, 8
# Two-shot latent all-reduce batches per TP (B9+): 2-/4-node GB300, us/layer of RS push
# + experts + WaveMoeTail, max over ranks, multicast vs two-shot: TP8 B16 41.74/42.80,
# B32 52.20/53.18, B64 79.27/81.50; TP16 B16 33.98/34.49, B32 42.19/42.52, B64
# 65.05/60.51.
TWOSHOT = {8: "", 16: "33-64:1"}
# WaveMoeTail batches that release the next AttnRes at entry.
EARLY = {8: "9-64:1", 16: "9-64:1"}
# WaveMoeTail batches whose up projection runs on tcgen05 (B9+; 1-GPU WaveMoeTail us
# mma.sync vs tcgen05, B16/32/64: TP8 4.53/5.80/9.05 vs 4.45/5.26/7.80, TP16 two-shot
# 4.50/5.39/7.75 vs 4.52/5.46/7.21).
UMMA = {8: "9-64:1", 16: "33-64:1"}


def flat(t):
    return to_cute(t.reshape(-1), 16)


def published(tp, batch):
    """The experts publish the latent and WaveMoeTail runs the tail (else MoeTail
    front)."""
    return batch > (4 if tp == 8 else 8)


def twoshot_batches(tp):
    table = batch_table(TWOSHOT.get(tp, ""))
    return {batch for batch, on in table.items() if on and batch > 8}


def early_batches(tp):
    table = batch_table(EARLY.get(tp, ""))
    return {batch for batch, on in table.items() if on and published(tp, batch)}


def umma_batches(tp):
    table = batch_table(UMMA.get(tp, ""))
    return {batch for batch, on in table.items() if on and batch > 8}


def shares(tp, batch, twoshot=False):
    """WaveMoeTail clusters share reduced latent rows (all 112 CTAs must be
    co-resident)."""
    return published(tp, batch) and WaveMoeTail(batch, tp, twoshot).reduce


@cache
def resident_clusters(device):
    """Co-resident 16-CTA clusters at one CTA per SM."""
    with torch.accelerator.device_index(device):
        return cutlass.utils.HardwareInfo(device).get_max_active_clusters(16)


class MoeTailBox:
    """Model-owned MoE tail mailboxes: latent [3, TP, CAP, 3584] (multicast), shared
    reduce-scatter [3, TP, CAP, 7168/TP] (peer stores), output [1, CAP, 7168], the
    local all-reduced latent rows [3, CAP, 3584] that WaveMoeTail clusters share and the
    two-shot latent buffers (``twoshot_args``/``twoshot_tail`` replace args/tail)."""

    def __init__(self, group, device, tp, rank):
        self.rank = rank
        self.latent = OprojMailbox(group, device, tp, rank, (3, tp, CAP, 3584))
        self.rs = OprojMailbox(group, device, tp, rank, (3, tp, CAP, 7168 // tp))
        self.out = OprojMailbox(group, device, tp, rank, (1, CAP, 7168))
        ptrs = self.rs.handle.buffer_ptrs
        self.peers = torch.tensor(ptrs, dtype=torch.int64, device=device)
        self.args = (flat(self.latent.buffer), flat(self.rs.buffer))
        self.tail: tuple = (to_cute(self.peers, 16), flat(self.out.buffer))
        self.tail += (Int64(self.latent.mc), Int64(self.out.mc))
        self.reduced = torch.empty((3, CAP, 3584), dtype=torch.bfloat16, device=device)
        self.reduced.view(torch.int32).fill_(-0x80000000)
        self.twoshot = box = TwoShot(group, device, tp, rank, 3584, CAP, parities=3)
        self.twoshot_args = (flat(box.recv), self.args[1])
        self.twoshot_tail = (*self.tail[:2], Int64(box.gather_mc), self.tail[3])
        # Experts' push: every parity's receive slots [3 x TP, CAP, 3584/TP] and the
        # owners' buffer bases (slot parity x TP + rank).
        self.push_box = (
            to_cute(box.recv.view(3 * tp, CAP, -1)),
            to_cute(box.peers[0], 16),
        )


@cache
def compile_tail(tp, batch, device, mode="front", trigger=2, early=False, umma=False):
    """front: rows complete at the PDL wait; front_poll: three 0xffff-armed G/W/S parity
    slots polled per row and re-armed after use; publish (``published``): WaveMoeTail
    over the published mailboxes, reading the weight shard directly, not by TMA;
    twoshot: publish over the two-shot receive slots and gather."""
    m8, twoshot = mode in ("publish", "twoshot"), mode == "twoshot"
    vec = lambda n, dtype=BFloat16, align=16: fake((n,), None, dtype, align)
    slots = 3 * POLL_CAP if mode == "front_poll" else batch
    latent = 3 * tp * CAP * (3584 // tp if twoshot else 3584)
    tensors: tuple = (vec(latent), vec(3 * tp * CAP * (7168 // tp)))
    weights = vec(7168 // tp * 3584) if m8 and not umma else vec(128, Uint8, 64)
    tensors += (weights, vec(3584))
    tensors += (vec(slots * 16 * 3584), vec(slots * 16), vec(slots * 7168))
    tensors += (vec(tp, Int64), vec(CAP * 7168))
    scalars: tuple = (Int64(0), Int64(0), Int32(0), Int32(0), Int32(CAP), Float32(1e-6))
    scalars += (Int32(2), Int32(CAP))
    kernel: MoeTail
    if m8:
        kernel = wave = WaveMoeTail(batch, tp, twoshot, early, umma)
        scalars += (vec(3 * CAP * 3584),) if wave.reduce else ()
    else:
        kernel = MoeTail(tp, batch, mode=mode, trigger=trigger)
    options = "" if m8 else "--ptxas-options=--maxrregcount=120"
    with torch.accelerator.device_index(device):
        args = (*tensors, *scalars, make_fake_stream())
        code = cute.compile(kernel, *args, options=options)
        # MoE tail requests 100% shared carveout so the next large PDL kernel can
        # coexist.
        return prepare_shared_carveout(code, device, 100)


@cache
def compile_push(batch, device, tp=8):
    args = (fake((batch, 7168)), fake((tp,), None, Int64), Int32(0), Int32(0))
    with torch.accelerator.device_index(device):
        return cute.compile(
            WaveMoeTail(batch, tp).push, *args, Int32(CAP), make_fake_stream()
        )


class MoeTailWeights:
    """One layer's up-projection shard [7168/TP, 3584], RMS weight and TMA map."""

    def __init__(self, tp, weight, gamma, eps, ordinal, count, batches, twoshots=()):
        self.weight, self.gamma = weight, gamma
        # Three parities rotate over the model's MoE layers; each launch re-arms the
        # previous MoE layer's slots of its batch (wrapping across decode steps, where
        # the first layer re-arms the largest batch the last layer may have published).
        parity, rearm = ordinal % 3, (ordinal - 1) % count % 3
        if parity == rearm:  # a launch must never re-arm the slots it publishes to
            raise ValueError(f"MoE tail parities collide for {count} MoE layers")
        self.parity, rearm = Int32(parity), Int32(rearm)
        self.scalars = (Int32(CAP), Float32(eps), rearm)
        self.wrap = 0 if ordinal else max(batches)
        cols = 512 if tp == 8 else 448
        u16 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16
        dims = (cols // 2, 3584 // (cols // 2), 7168 // tp)
        spec: tuple = (weight.data_ptr(), u16, dims, (cols, 7168), (cols // 2, 2, 32))
        spec += (cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,)
        spec += (cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,)
        self.descriptor = tensor_maps([spec], weight.device)
        self.views = (to_cute(self.descriptor, 64), to_cute(gamma, 16))
        self.codes, device, early = {}, weight.device.index, early_batches(tp)
        umma = umma_batches(tp)
        if any(published(tp, b) and b in umma for b in batches):
            # WaveMoeTail tcgen05: [64 x lines, 64] boxes of the shard, 128B swizzle
            spec = (weight.data_ptr(), u16, (3584, 7168 // tp), (7168,))
            spec += ((64, 128 if tp == 8 else 64),)
            spec += (cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,)
            spec += (cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,)
            self.umma_map = tensor_maps([spec], weight.device)
            self.umma_views = (to_cute(self.umma_map, 64), self.views[1])
        for b in batches:  # WaveMoeTail reads the shard directly, not by TMA
            twoshot = b in twoshots
            mode = "twoshot" if twoshot else "publish" if published(tp, b) else "front"
            views = self.views if mode == "front" else (flat(weight), self.views[1])
            on = mode != "front" and b in umma
            views = self.umma_views if on else views
            code = compile_tail(tp, b, device, mode, early=b in early, umma=on)
            self.codes[b] = code, views, shares(tp, b, twoshot), twoshot
        shared = any(shares(tp, b, b in twoshots) for b in batches)
        if shared and resident_clusters(device) < 7:
            raise RuntimeError("WaveMoeTail needs seven co-resident 16-CTA clusters")

    def run(self, box, expanded, weights, shared, code=None, batch=None):
        batch = batch or weights.shape[0]
        if code is None:
            code, views, share, twoshot = self.codes[batch]
        else:
            views, share, twoshot = self.views, 0, False
        args, tail = box.args, box.tail
        if twoshot:
            args, tail = box.twoshot_args, box.twoshot_tail
        rows = (flat(expanded), flat(weights), flat(shared), *tail, self.parity)
        rows += (Int32(box.rank), *self.scalars, Int32(self.wrap or batch))
        if share:
            rows += (flat(box.twoshot.gather if twoshot else box.reduced),)
        code(*args, *views, *rows, stream(expanded.device))


class PreparedTail:
    """The layer's MoE tail; placement = (model MoeTailBox, MoE ordinal, MoE layer
    count). At WaveMoeTail batches the experts publish into latent slot ``publish[b]``
    (multicast, parity, rank; at ``twoshots`` the two-shot push box) and the side
    stream pushes the shared-expert output (``push``)."""

    def __init__(self, op, norm, up, batches, placement):
        tp, rank = op.contract.tp_size, op.rank
        local_up = up.narrow(0, rank * (7168 // tp), 7168 // tp)
        self.tail_box, ordinal, count = placement
        eps = op.contract.rms_eps
        m8 = [b for b in batches if published(tp, b)]
        self.twoshots = twoshot_batches(tp) & set(m8)
        args = (tp, local_up, norm, eps, ordinal, count, batches, self.twoshots)
        self.tail_weights = MoeTailWeights(*args)
        self.peers, _, mc, _ = self.tail_box.tail
        self.ring = (self.tail_weights.parity, Int32(rank))
        self.pushers = {b: compile_push(b, norm.device.index, tp) for b in m8}
        self.publish = {b: (mc, *self.ring) for b in m8}
        slot = Int32(ordinal % 3 * tp + rank)
        self.publish.update({b: (*self.tail_box.push_box, slot) for b in self.twoshots})

    def push(self, shared):
        if shared.shape[0] in self.pushers:
            args = (to_cute(shared), self.peers, *self.ring, Int32(CAP))
            self.pushers[shared.shape[0]](*args, stream(shared.device))

    def run(self, expanded, weights, shared, out, *, consume):
        self.tail_weights.run(self.tail_box, expanded, weights, shared)
        if consume(out) is not out:
            raise RuntimeError("The tail output must be consumed from its mailbox ring")
        return out
