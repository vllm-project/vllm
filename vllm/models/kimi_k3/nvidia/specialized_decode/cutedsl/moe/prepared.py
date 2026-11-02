# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepared MoE provider: one model owner serializes its scratch, no tensor pointers
are rebound at execution, and the side stream joins before the tail."""

import torch

from ..common.policy import batch_table
from .experts_runtime import scratch as expert_scratch
from .shared import scratch as shared_scratch


class MoEScratch:
    def __init__(self, tp, device, batches):
        self.tp, self.device, self.batches = tp, torch.device(device), tuple(batches)
        cap, h = max(batches), 6144 // tp
        with torch.accelerator.device_index(self.device.index):
            self.stream = torch.cuda.Stream(device=self.device)
            self.begin, self.shared_done = torch.cuda.Event(), torch.cuda.Event()
            self.gate_done = torch.cuda.Event()  # the split shared expert's front half
            f32, i32, bf16 = torch.float32, torch.int32, torch.bfloat16
            empty = lambda s, t=bf16: torch.empty(s, dtype=t, device=self.device)
            self.latent = empty((cap, 3584))
            self.front: tuple = (empty((cap, 896), f32), empty((cap, 2 * h), f32))
            self.front += (empty((cap, 16), i32), empty((cap, 16)), empty((cap, h)))
            self.front += (empty((cap, 7168)), self.latent)
            self.expanded = empty((cap * 16, 3584))
            self.experts = expert_scratch(tp, self.device)
            # The experts' plan (MXFP8 x), written by the routing kernel at B>8.
            self.plan = self.experts["plan"]
            self.shared = shared_scratch(self.device)  # fused shared-expert counters
        self.views = {}
        for b in batches:
            self.views[b] = (self.latent[:b], self.expanded[: b * 16], self.experts)


# TP8 B16: the experts start beside the split gate/up half (+1% E2E).
JOIN = {8: "9-16:0", 16: ""}


class PreparedMoE:
    def __init__(self, front, experts, tail, state, *, prefetch=None, layer_idx=None):
        self.front, self.experts, self.tail, self.state = front, experts, tail, state
        self.prefetch, self.layer_idx = prefetch, layer_idx
        self.small_batch = None  # optional small-batch chain
        # Split batches whose routed experts launch beside the shared expert's
        # gate/up half instead of after it.
        self.join = batch_table(JOIN[state.tp])
        self.sms = torch.cuda.get_device_properties(state.device).multi_processor_count
        for b, sms in front.split.items():  # the split's down CTAs take the SMs left
            experts.grids[b] = min(experts.grid_for(b), self.sms - sms)
        for b in state.batches:
            _, expanded, buffers = state.views[b]
            publish, planned = tail.publish.get(b, ()), front.planned(b)
            twoshot = b in tail.twoshots
            experts.prepare(b, buffers, expanded, publish, planned, twoshot, True)

    def run(self, x, out, *, consume=None):
        state, b = self.state, x.shape[0]
        if self.small_batch is not None and b in self.small_batch.batches:
            return self.small_batch.run(out, consume)
        main = torch.cuda.current_stream(state.device)
        # The shared expert runs on the side stream beside the routed experts, or, when
        # the persistent expert grid spans every SM, beside the front only. Split: its
        # gate/up half beside the front, its down rows beside the experts.
        before = self.experts.grid_for(b) >= self.sms
        if self.front.merged_batch(b):
            # One kernel publishes the routing logits and leaves h; the down rows then
            # run beside the experts.
            ids, weights, latent = self.front.route(x)
            state.begin.record(main)
            with torch.cuda.stream(state.stream):
                state.stream.wait_event(state.begin)
                shared, beside = self.front.shared_expert(x, self.tail)
                beside()
                state.shared_done.record(state.stream)
            self.experts.group(ids)
            return self.finish(b, out, latent, ids, weights, shared, consume)
        state.begin.record(main)
        with torch.cuda.stream(state.stream):
            state.stream.wait_event(state.begin)
            shared, beside = self.front.shared_expert(x, self.tail)
            if beside is not None:
                state.gate_done.record(state.stream)
                beside()
            state.shared_done.record(state.stream)
        ids, weights, latent = self.front.route(x)
        self.experts.group(ids)  # pregrouped unit table, beside the shared expert
        if before or (beside is not None and self.join.get(b, 1)):
            main.wait_event(state.shared_done if before else state.gate_done)
        return self.finish(b, out, latent, ids, weights, shared, consume, before)

    def finish(self, b, out, latent, ids, weights, shared, consume, before=False):
        """The routed experts, the next layer's prefetch fork, then the tail once the
        shared expert's side stream is done."""
        main = torch.cuda.current_stream(self.state.device)
        expanded = self.experts.run(latent, ids, weights)
        if self.prefetch is not None:
            self.prefetch.fork("next_attention", self.layer_idx, b)
        if not before:
            main.wait_event(self.state.shared_done)
        return self.tail.run(expanded, weights, shared, out, consume=consume)
