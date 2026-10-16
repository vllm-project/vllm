# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch MoE chains: front -> routed experts -> (shared expert) -> polling tail.

TP8 B1: ``MoeFrontTP8`` -> ``ClusterExpertsTP8`` -> ``MoeSharedTP8`` -> ``MoeTail``
(front_poll, trigger 3). TP16 B1/B2: ``MoeFrontTP16`` (shared expert inside, on
designated SMs) -> ``ClusterExpertsTP16`` -> ``MoeTail``. The o-projection consumer
publishes the post-attention row into the x_moe mailbox (parity = layer index % 2) and
bumps an epoch hint; the co-resident front has staged its weights by then. MoE tail
polls the G/W/S slots (three parities, 0xffff-armed, re-armed after use) and publishes
the next AttnRes input."""

import functools

import cutlass.cute as cute
import torch
from cutlass import Float32, Float64, Int32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ...common.host import fake, layout, stream, to_cute
from ...oproj.prepared import OprojMailbox
from ..tail_runtime import POLL_CAP, compile_tail
from .experts import ClusterExpertsTP8, ClusterExpertsTP16
from .front_tp8 import MoeFrontTP8, MoeSharedTP8, SharedSituTables
from .front_tp16 import MoeFrontTP16, initialize_designated

# Kernel class and trailing scalar arguments of each small-batch launch.
_KERNELS = {
    "front": (MoeFrontTP8, (Int32,) * 3 + (Uint64, Float32)),
    "front16": (MoeFrontTP16, (Int32,) * 3 + (Uint64,) + (Float32,) * 4 + (Float64,)),
    "experts_tp16": (ClusterExpertsTP16, (Float32,) * 3),
    "experts_tp8": (ClusterExpertsTP8, (Float32,) * 3),
    "shared": (MoeSharedTP8, (Int32,) * 2),
    "tables": (SharedSituTables, (Float32,) * 2),
}


@functools.cache
def _compile(kind, specs, device, *args, **kwargs):
    kernel, scalars = _KERNELS[kind]
    operands = [fake(*spec) for spec in specs] + [t(0) for t in scalars]
    with torch.accelerator.device_index(device):
        return cute.compile(kernel(*args, **kwargs), *operands, make_fake_stream())


def _launch(kind, tensors, *args, **kwargs):
    specs = tuple(map(layout, tensors))
    code = _compile(kind, specs, tensors[0].device.index, *args, **kwargs)
    return code, tuple(map(to_cute, tensors))


def u32(t):
    return t.view(torch.uint32)


class SmallBatchBox:
    """Model-owned small-batch storage, shared serially by every MoE layer."""

    def __init__(self, group, device, tp, rank):
        def armed(shape, fill, dtype=torch.bfloat16):
            value = torch.empty(shape, dtype=dtype, device=device)
            value.view(torch.int32).fill_(fill)
            return value

        zeros = functools.partial(torch.zeros, dtype=torch.int32, device=device)
        self.tp, self.rank, self.device = tp, rank, device
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.xmoe = armed((2, 1, 4, 7168), -0x80000000)
        self.epoch = zeros(256)
        self.scores = armed((7168,), -0x80000000, torch.int32)
        self.ids = armed((16,), -1, torch.int32)
        self.w, self.s = armed((3, POLL_CAP, 16), -1), armed((3, POLL_CAP, 7168), -1)
        self.g = armed((3, POLL_CAP * 16, 3584), -1)
        self.h2 = OprojMailbox(group, device, tp, rank, (2, POLL_CAP, 3584))
        self.latent = zeros(3584, dtype=torch.bfloat16)
        self.hg, self.counter = zeros(6144), zeros(1)
        self.tables = {}
        self.batches = (1,) if tp == 8 else (1, 2)
        if tp == 16:  # resident-SM placement mask plus per-batch route/gate scratch
            self.designated = zeros(self.sms + 16)
            initialize_designated(self.designated)
            self.scratch, i32 = {}, torch.int32
            for b in self.batches:
                scores = armed((2 * b * 896 * 2,), -0x80000000, i32)
                self.scratch[b] = scores, armed((b * 16,), -1, i32), zeros(2 * b * 768)
        self.front_views = tuple(self.xmoe[p] for p in range(2))
        self.hint_views = tuple(self.epoch[p * 32 : p * 32 + 224] for p in range(2))

    def situ_tables(self, beta, linear_beta):
        key = float(beta), float(linear_beta)
        if key not in self.tables:
            table = torch.empty(131072, dtype=torch.float32, device=self.device)
            code, args = _launch("tables", (table,))
            code(*args, Float32(beta), Float32(linear_beta), stream(self.device))
            self.tables[key] = table
        return self.tables[key]


class SmallBatchChain:
    """One MoE layer's small-batch chain; MoE tail weights come from the layer's
    prepared tail."""

    def __init__(self, box, moe, experts, tail, layer_idx, beta, linear_beta, prefetch):
        self.box, self.tail_weights, self.tail_box = (
            box,
            tail.tail_weights,
            tail.tail_box,
        )
        self.prefetch, self.layer_idx = prefetch, layer_idx
        self.parity, self.slot = layer_idx % 2, int(tail.tail_weights.parity)
        self.batches, shared = box.batches, moe.shared_experts
        router, bias = moe.gate.weight, moe.gate.e_score_correction_bias
        gate_up, shared_down = shared.gate_up_proj.weight, shared.down_proj.weight
        rows = 3584 // box.tp
        down = moe.routed_expert_down_proj.weight.narrow(0, box.rank * rows, rows)
        scales = tuple(experts.weights[i].view(torch.uint8).reshape(-1) for i in (1, 3))
        scale, renorm = moe.routed_scaling_factor, moe.moe_renormalize
        self.scalars: tuple = (Int32(box.rank), Uint64(box.h2.mc), Float32(scale))
        head: tuple = (router, bias, box.xmoe, u32(box.epoch))
        if box.tp == 8:
            tensors = head + (u32(box.scores), box.ids, box.w, down, box.h2.buffer)
            tensors += (box.latent, gate_up, u32(box.hg))
            self.front = _launch("front", tensors, grid=box.sms - 8, renorm=renorm)
            tables = box.situ_tables(beta, linear_beta)
            tensors = (box.xmoe, shared_down, tables, u32(box.hg), box.s)
            self.shared = _launch("shared", tensors, grid=box.sms - 96)
            tensors = (box.latent, box.ids, experts.maps[256:640].view(3, 128), *scales)
            routed, args = _launch(
                "experts_tp8", tensors + (box.g[self.slot, :16], box.counter)
            )
            self.routed = routed, args + experts.scalars
            self.tail = compile_tail(8, 1, box.device.index, "front_poll", 3)
            return
        maps, self.codes = experts.maps[:256].view(2, 128), {}
        for b in self.batches:
            scores, ids, hg = box.scratch[b]
            tensors = head + (u32(scores), ids, box.w, down, box.h2.buffer, gate_up)
            tensors += (u32(hg), shared_down, box.s, box.designated)
            front = _launch("front16", tensors, b, grid=box.sms - 7 * b, renorm=renorm)
            latent = box.h2.buffer[self.parity, :b].reshape(-1)
            g = box.g[self.slot, : b * 16]
            routed = _launch("experts_tp16", (latent, ids, maps, *scales, g), b)
            tail = compile_tail(16, b, box.device.index, "front_poll", 2)
            self.codes[b] = front, routed, tail
        lb = linear_beta
        self.scalars += (Float32(beta), Float32(1 / beta), Float32(lb), Float64(1 / lb))
        self.experts = experts.scalars

    def run(self, out, consume):
        b, s = out.shape[0], stream(self.box.device)
        par, slot = Int32(self.parity), Int32(self.slot)
        if self.box.tp == 8:
            self.front[0](*self.front[1], par, slot, *self.scalars, s)
            self.routed[0](*self.routed[1], s)
            self.shared[0](*self.shared[1], par, slot, s)
            tail = self.tail
        else:
            (front, front_args), (routed, routed_args), tail = self.codes[b]
            front(*front_args, par, slot, *self.scalars, s)
            routed(*routed_args, *self.experts, s)
        if self.prefetch is not None:
            self.prefetch.fork("next_attention", self.layer_idx, b)
        self.tail_weights.run(
            self.tail_box, self.box.g, self.box.w, self.box.s, tail, b
        )
        if consume(out) is not out:
            raise RuntimeError("MoE tail output must be consumed from its mailbox ring")
        return out
