# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wide MoE: the MoE layer of decode steps past the prepared batches (65 to
`WIDE_ROWS` rows, e.g. 13-204 DSpark verify requests of 5 tokens), at TP8/TP16.

One layer (spec engine, `ROUTING` "sharded"):
  side stream   shared expert (gate/up, SiTU, down; the stock module) -> BF16
                partial [b, 7168]
  front         one FP32-out GEMM over this rank's router rows and latent down shard
                (`providers.pack_front`: 1/TP of the replicated 51 MB latent down),
                `WidePublish` multicasts both into every rank's mailbox row, then
                `WideRouteSharded`: exact top16 from the polled logits, each pair
                appended to its expert's routing list, and the polled latent row
                written as the experts' MXFP8 plan
  experts       the listed wide grouped engine (unit windows built from the routing
                lists: every touched expert streams once per 8 of its tokens) and the
                routing-weighted top16 sum (`Finalize`); past `ENGINES`' spec
                range the stock TRTLLM-Gen MXFP4 experts (routing inside), finalized,
                behind the replicated router and latent down GEMMs
  tail          the latent all-reduce with the RMSNorm fused (the stock FlashInfer
                kernel), this rank's up-projection columns added into its shared
                partial, and the all-reduce of the hidden rows (the stock latent
                MoE's column-parallel tail)

Precision is the stock path's: BF16 linears and shared expert, FP32 router logits,
the latent rounded once to BF16 (as the stock GEMM epilogue), MXFP8 activations x
MXFP4 experts with the MXFP8 hidden, FP32 accumulation of the BF16 top16 rows, BF16
all-reduces; only accumulation orders differ.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import make_fake_stream

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)

from ..common.host import fake, stream, to_cute
from ..common.mailbox import poll, rearm
from ..common.policy import batch_table
from ..common.primitives import (
    bits_float,
    fma,
    launch,
    load_words,
    multicast_words,
    pack_pair,
    store_words,
    thread,
    unpack_bf16,
    view,
)
from .experts import (
    LIST_EXPERTS,
    LIST_PAIRS,
    LIST_WORDS,
    WIDE_ROWS,
    ExpertsGeometry,
    claim,
    plan_bytes,
    plan_chunk,
    plan_xsf,
)
from .experts_runtime import scratch as expert_scratch
from .front import WIDTH, RouteSitu, compile_route


def wide_batches(batches):
    """The wide MoE's batches among `batches`: past the prepared 64, at most
    WIDE_ROWS."""
    return tuple(b for b in batches if 64 < b <= WIDE_ROWS)


# Routed-expert engine per batch: whole wide MoE layer, TP8 (8 GB300 ranks, slowest
# rank, uniform top16 of 896, us per layer, specialized engine / TRTLLM-Gen experts):
# grouped front B192 381.6 / 396.0, B256 399.2 / 405.9, B320 439.2 / 433.7, B640
# 752.0 / 620.2 (8-token units re-read the weights of experts with > 8 tokens); with
# the sharded front B320 421.9 / 433.4. TRTLLM-Gen needs a finalized output (more rows
# than the stock deferred-finalize limit, 128).
ENGINES = "65-320:spec,321-1024:trtllm"
# Shared expert placement per batch (side stream): "front" from the layer start, beside
# the front GEMMs; "beside" after routing, beside the spec experts on the SMs their
# ``BESIDE_GRID`` CTAs leave; "tail" after the experts, beside Finalize and the latent
# all-reduce. TP8, sharded front, us per layer front / beside / tail: B80 290.1 /
# 293.4 / 294.5, B160 355.2 / 362.2 / 356.5, B320 421.9 / 430.6 / 426.9.
SHARED = "65-1024:front"
BESIDE_GRID = 144
# Spec-engine front per batch: "sharded" (one GEMM over this rank's router rows and
# latent down shard, ``WidePublish`` multicasts them into every rank's mailbox, then
# ``WideRouteSharded``), "listed" (replicated router and latent down GEMMs, then
# ``WideRoute``: both append the routing lists and write the plan, and the experts
# build their unit windows from the lists) or "grouped" (routing, then the plan and
# unit-table kernels).
ROUTING = "65-1024:sharded"


def engine(batch, defers):
    """'spec' or 'trtllm' for `batch` (`defers`: the stock experts would leave
    this batch unfinalized)."""
    name = batch_table(ENGINES).get(batch, "spec")
    return "spec" if defers else name


class WideRoute(RouteSitu):
    """The listed wide engine's front: CTA t routes token t (`RouteSitu`'s exact top16
    and renormalized weights) and appends its 16 (token, slot) pairs to their experts'
    routing lists (relaxed GPU-scope counter claims; this call's parity is the experts
    workspace's grouped-call generation & 1); CTA b + t writes token t's MXFP8 plan row
    from the BF16 latent (`plan_chunk`: bit-identical to `ExpertsPlanKernel`). The
    experts release at the end (late: they must not take SMs from routing)."""

    def __init__(self, batch, renorm, tp):
        super().__init__(batch, 1, renorm, False, (True, True, False), False)
        geometry = ExpertsGeometry(tp, WIDE_ROWS)
        self.calls, self.xsf = geometry.ws_calls, plan_xsf(WIDE_ROWS)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, 2 * self.batch, 256)

    @cute.kernel
    def kernel(self, args):
        logits, bias, ids, weights, latent, plan, lists, workspace, scale = args
        tid, _, _, cta = thread()
        if cta < self.batch:
            self.route(logits, bias, ids, weights, scale, cta, tid, None)
            if tid < 16:  # this thread wrote ids[cta, tid] in ``publish``
                expert = ids[cta, tid]
                generation = view(workspace.iterator, self.calls, cutlass.Int32, 1)[0]
                column = (generation & 1) * LIST_EXPERTS + expert
                slot = claim(lists.iterator + column, 1)
                row = LIST_PAIRS + column * WIDE_ROWS + cutlass.Int32(slot)
                lists[row] = cta * 16 + tid
        else:
            token = cta - self.batch
            cute.arch.griddepcontrol_wait()
            for i in cutlass.range_constexpr(2):
                chunk = tid + 256 * i
                if chunk < 448:  # warp-uniform: chunks 256-447 are warps 0-5
                    words = load_words(
                        latent.iterator + token * 3584 + chunk * 8, count=4,
                        mutable=True,
                    )  # fmt: skip
                    plan_chunk(plan, token, chunk, words, self.xsf)
        cute.arch.griddepcontrol_launch_dependents()


class WideRouteSharded(WideRoute):
    """`WideRoute` over the sharded front: every rank's `WidePublish` left its
    router logits (FP32 words) and latent shard (BF16) in this rank's mailbox row of
    each token. The routing CTAs poll the logits (`RouteSitu`'s polled routing), the
    plan CTAs poll the latent row and write the MXFP8 plan; both re-arm what they
    read (Lamport: no dependency wait on the data)."""

    def __init__(self, batch, renorm, tp):
        super().__init__(batch, renorm, tp)
        self.polled = True

    @cute.kernel
    def kernel(self, args):
        mailbox, bias, ids, weights, plan, lists, workspace, scale = args
        tid, _, _, cta = thread()
        if cta < self.batch:
            self.route(None, bias, ids, weights, scale, cta, tid, mailbox)
            if tid < 16:  # this thread wrote ids[cta, tid] in ``publish``
                cute.arch.griddepcontrol_wait()  # the last experts call's counts
                expert = ids[cta, tid]
                generation = view(workspace.iterator, self.calls, cutlass.Int32, 1)[0]
                column = (generation & 1) * LIST_EXPERTS + expert
                slot = claim(lists.iterator + column, 1)
                row = LIST_PAIRS + column * WIDE_ROWS + cutlass.Int32(slot)
                lists[row] = cta * 16 + tid
        else:
            token = cta - self.batch
            base = mailbox.iterator + token * mailbox.shape[2] + tid * 8
            if tid < 192:  # chunks tid and tid + 256 (448 chunks of 8)
                got = poll(base, 2048, 2)
                for i in cutlass.range_constexpr(2):
                    rearm(base + i * 2048)
                    words = [got[i, k] for k in range(4)]
                    plan_chunk(plan, token, tid + 256 * i, words, self.xsf)
            else:
                got = poll(base, 2048, 1)
                rearm(base)
                words = [got[0, k] for k in range(4)]
                plan_chunk(plan, token, tid, words, self.xsf)
        cute.arch.griddepcontrol_launch_dependents()


class WidePublish:
    """This rank's sharded front rows (FP32 [b, n]: its router logit rows [0, valid),
    then its latent down shard [n - shard, n)) multicast into every rank's mailbox
    row of each token: the logits as FP32 words at columns 3584 + 2 * valid * rank,
    the latent shard rounded once to BF16 (as the stock BF16 GEMM epilogue) at
    columns shard * rank. Negative zeros become +0 (the Lamport sentinel)."""

    def __init__(self, batch, n, valid, shard):
        self.batch, self.n, self.valid, self.shard = batch, n, valid, shard
        self.fragments = valid // 4 + shard // 8

    @cute.jit
    def __call__(self, *args):
        threads = (self.fragments + 31) // 32 * 32
        launch(self.kernel, args, self.batch, threads)

    @cute.kernel
    def kernel(self, args):
        front, mc, rank = args
        tid, _, _, token = thread()
        cute.arch.griddepcontrol_wait()
        row = cutlass.Int64(token) * (3584 + 1792)
        logits: cutlass.Constexpr = self.valid // 4
        source = front.iterator + token * self.n
        if tid < logits:
            got = load_words(source + 4 * tid, count=4, mutable=True)
            words = []
            for k in cutlass.range_constexpr(4):
                word = cutlass.Uint32(got[k])
                if word == cutlass.Uint32(0x80000000):
                    word = cutlass.Uint32(0)
                words.append(word)
            column = 3584 + 2 * self.valid * rank + 8 * tid
            multicast_words(mc + (row + column) * 2, words)  # bits, as FP32 lanes
        elif tid < self.fragments:
            fragment = tid - logits
            start = source + (self.n - self.shard) + 8 * fragment
            values = [*load_words(start, count=4, mutable=True)]
            values += [*load_words(start + 4, count=4, mutable=True)]
            packed = [
                pack_pair(bits_float(values[2 * i]), bits_float(values[2 * i + 1]),
                          policy="latent")
                for i in range(4)
            ]  # fmt: skip
            column = self.shard * rank + 8 * fragment
            multicast_words(mc + (row + column) * 2, packed)
        cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def compile_publish(batch, n, valid, shard, device):
    args = (fake((batch, n), None, cutlass.Float32), cutlass.Int64(0), cutlass.Int32(0))
    with torch.accelerator.device_index(device):
        return cute.compile(
            WidePublish(batch, n, valid, shard), *args, make_fake_stream()
        )


@functools.cache
def compile_sharded_route(batch, renorm, tp, device, rows):
    f32, i32, u8 = cutlass.Float32, cutlass.Int32, cutlass.Uint8
    args: tuple = (fake((1, rows, 3584 + 1792)), fake((896,), None, f32))
    args += (fake((batch, 16), None, i32), fake((batch, 16)))
    args += (fake((plan_bytes(WIDE_ROWS),), None, u8), fake((LIST_WORDS,), None, i32))
    args += (fake((ExpertsGeometry(tp, WIDE_ROWS).workspace_bytes,), None, u8),)
    args += (f32(1), make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(WideRouteSharded(batch, renorm, tp), *args)


@functools.cache
def compile_wide_route(batch, renorm, tp, device):
    f32, i32, u8 = cutlass.Float32, cutlass.Int32, cutlass.Uint8
    args: tuple = (fake((batch, 896), None, f32), fake((896,), None, f32))
    args += (fake((batch, 16), None, i32), fake((batch, 16)), fake((batch, 3584)))
    args += (fake((plan_bytes(WIDE_ROWS),), None, u8), fake((LIST_WORDS,), None, i32))
    args += (fake((ExpertsGeometry(tp, WIDE_ROWS).workspace_bytes,), None, u8),)
    args += (f32(1), make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(WideRoute(batch, renorm, tp), *args)


class Finalize:
    """out[t] = sum_k w[t, k] * rows[16 t + k] (FP32, k in order, one BF16 rounding):
    one CTA per token, 448 threads of 8 columns."""

    def __init__(self, batch):
        self.batch = batch

    @cute.jit
    def __call__(self, rows, weights, out, s):
        launch(self.kernel, (rows, weights, out, s), self.batch, 448)

    @cute.kernel
    def kernel(self, args):
        rows, weights, out = args
        tid = cute.arch.thread_idx()[0]
        token = cute.arch.block_idx()[0]
        cute.arch.griddepcontrol_wait()
        acc = cute.make_rmem_tensor(8, Float32)
        for i in cutlass.range_constexpr(8):
            acc[i] = Float32(0)
        words = cute.make_rmem_tensor((16, 4), cutlass.Uint32)
        for k in cutlass.range_constexpr(16):  # every load in flight first
            src = rows.iterator + (token * 16 + k) * 3584 + tid * 8
            got = load_words(src, count=4)
            for i in cutlass.range_constexpr(4):
                words[k, i] = got[i]
        for k in cutlass.range_constexpr(16):
            w = Float32(weights[token, k])
            for i in cutlass.range_constexpr(4):
                lo, hi = unpack_bf16(words[k, i])
                acc[2 * i] = fma(w, lo, acc[2 * i])
                acc[2 * i + 1] = fma(w, hi, acc[2 * i + 1])
        packed = [pack_pair(acc[2 * i], acc[2 * i + 1], policy=None) for i in range(4)]
        store_words(out.iterator + token * 3584 + tid * 8, packed)
        cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def compile_finalize(batch, device):
    args = (fake((batch * 16, 3584)), fake((batch, 16)), fake((batch, 3584)))
    with torch.accelerator.device_index(device):
        return cute.compile(Finalize(batch), *args, make_fake_stream())


class WideScratch:
    """Model-owned wide-MoE buffers at the widest batch (shared by every layer), the
    wide experts' own workspace and plan, and the side streams."""

    def __init__(self, tp, device, batches):
        from ..oproj.prepared import OprojMailbox

        self.batches = tuple(batches)
        cap = max(self.batches, default=0)
        self.device = torch.device(device)
        # The sharded front's mailbox rows [1, cap, latent | logits words].
        group = get_tp_group()
        owner = group.device_group, self.device, tp, group.rank_in_group
        self.mailbox = OprojMailbox(*owner, (1, max(cap, 1), 3584 + 2 * 896))
        empty = lambda s, t=torch.bfloat16: torch.empty(s, dtype=t, device=self.device)
        self.ids, self.weights = empty((cap, 16), torch.int32), empty((cap, 16))
        self.expanded = empty((0, 3584))  # the spec engine's rows (``reserve``)
        self.bound = False
        self.latent = empty((cap, 3584))  # this rank's routed latent partial
        self.experts = expert_scratch(tp, self.device, wide=True)
        h = 6144 // tp  # RouteSitu's (unused) SiTU-role arguments
        self.unused: tuple = (empty((1, 64, WIDTH)), empty((cap, 3584)))
        self.unused += (empty((cap, 2 * h), torch.float32), empty((cap, h)))
        self.empty = empty
        with torch.accelerator.device_index(self.device.index):
            high = torch.cuda.Stream.priority_range()[1]
            self.stream = torch.cuda.Stream(device=self.device)  # shared expert
            self.front = torch.cuda.Stream(device=self.device, priority=high)
            self.events = [torch.cuda.Event() for _ in range(4)]

    def reserve(self, batch):
        """Size the spec engine's expanded rows for `batch` (the widest batch the
        layers' spec engines take), before any engine binds them."""
        if self.expanded.shape[0] < batch * 16:
            if self.bound:
                raise RuntimeError("wide MoE rows grown after an engine bound them")
            self.expanded = self.empty((batch * 16, 3584))

    def rows(self, batch):
        self.bound = True
        return self.expanded[: batch * 16]


class WideMoE:
    """One MoE layer's wide provider over the stock module's weights and the layer's
    `PersistentExperts` (prepared here for its wide batches)."""

    def __init__(self, module, experts, state, beta, linear_beta, packed=None):
        """`packed`: this rank's [router rows; latent down shard] (`providers.
        pack_front`, shared with the narrow front), else packed here."""
        self.module, self.experts, self.state = module, experts, state
        h = module.shared_experts.gate_up_proj.weight.shape[0] // 2
        tp, rank = (
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
        up = module.routed_expert_up_proj.weight
        shard = up.shape[0] // tp
        self.columns, self.rank = (rank * shard, shard), rank
        self.up = up.narrow(0, rank * shard, shard)  # this rank's output columns
        self.norm = module.routed_expert_norm
        self.routed = routed = module.experts.routed_experts
        monolithic = getattr(routed.quant_method, "is_monolithic", False)
        bias = to_cute(module.gate.e_score_correction_bias)
        scale, renorm = module.routed_scaling_factor, module.moe_renormalize
        scalars = tuple(map(cutlass.Float32, (scale, beta, linear_beta)))
        dev = state.device.index
        self.plans: dict[int, tuple] = {}
        names = {}
        self.shared, self.routing = batch_table(SHARED), batch_table(ROUTING)
        for b in state.batches:
            defers = routed.moe_config.should_defer_moe_finalize(b)
            names[b] = engine(b, defers or not monolithic)
            if names[b] == "spec" and self.shared.get(b) == "beside":
                experts.grids[b] = BESIDE_GRID
        state.reserve(max((b for b, n in names.items() if n == "spec"), default=0))
        for b in state.batches:
            name = names[b]
            ids, weights = state.ids[:b], state.weights[:b]
            if name == "trtllm":
                self.plans[b] = (name,)
                continue
            expanded = state.rows(b)
            out = (to_cute(expanded), to_cute(weights), to_cute(state.latent[:b]))
            finalize = compile_finalize(b, dev)
            if self.routing.get(b) == "sharded":
                if packed is None:
                    from ...runtime.providers import pack_front

                    packed = pack_front(module, rank, tp)
                self.packed = packed
                n, valid = packed.shape[0], 896 // tp
                publish = compile_publish(b, n, valid, 3584 // tp, dev)
                rows = state.mailbox.buffer.shape[1]
                route = compile_sharded_route(b, renorm, tp, dev, rows)
                buffers = state.experts
                experts.prepare(b, buffers, expanded, planned=True, listed=True)
                tail = tuple(map(to_cute, (buffers["plan"], buffers["lists"])))
                tail += (to_cute(buffers["workspace"]), scalars[0])
                box = to_cute(state.mailbox.buffer), cutlass.Int64(state.mailbox.mc)
                args: tuple = (
                    box,
                    (bias, to_cute(ids), to_cute(weights)) + tail,
                    publish,
                )
                lists = buffers["lists"]
                self.plans[b] = "sharded", route, args, finalize, out, lists
                continue
            if self.routing.get(b) == "listed":
                route = compile_wide_route(b, renorm, tp, dev)
                buffers = state.experts
                experts.prepare(b, buffers, expanded, planned=True, listed=True)
                tail = tuple(map(to_cute, (buffers["plan"], buffers["lists"])))
                tail += (to_cute(buffers["workspace"]), scalars[0])
                args = (bias, to_cute(ids), to_cute(weights)), tail
                lists = buffers["lists"]
                self.plans[b] = "listed", route, args, finalize, out, lists
                continue
            route = compile_route(b, h, renorm, (True, False, False), dev)
            mailbox, latent, gate, hidden = state.unused
            args = (bias, to_cute(ids), to_cute(weights), to_cute(mailbox))
            args += (to_cute(latent[:b]), to_cute(gate[:b]), to_cute(hidden[:b]))
            experts.prepare(b, state.experts, expanded)
            self.plans[b] = name, route, args + scalars, finalize, out, ids

    def serves(self, batch):
        """Whether this layer's wide MoE is prepared for `batch` rows."""
        return batch in self.plans

    def run(self, x):
        """[b, 7168] BF16 MoE output of the hidden rows `x` (all-reduced)."""
        state, b, module = self.state, x.shape[0], self.module
        plan = self.plans[b]
        main = torch.cuda.current_stream(state.device)
        fork, down_done, shared_done, late = state.events
        where = self.shared.get(b, "front")
        if where == "beside" and plan[0] == "trtllm":
            where = "front"

        def shared_expert(event, after=main):
            event.record(after)
            with torch.cuda.stream(state.stream):
                state.stream.wait_event(event)
                rows = module.shared_experts(x)
                rows.record_stream(main)
                shared_done.record(state.stream)
            return rows

        fork.record(main)
        if where == "front":
            shared = shared_expert(fork)
        if plan[0] == "sharded":
            # The front GEMM, publish, routing, experts and Finalize run on the
            # high-priority stream: their CTAs go before the shared expert's.
            _, route, (box, route_args, publish), finalize, out, lists = plan
            with torch.cuda.stream(state.front):
                state.front.wait_event(fork)
                device = stream(state.device)
                front = torch.mm(x, self.packed.t(), out_dtype=torch.float32)
                publish(to_cute(front), box[1], cutlass.Int32(self.rank), device)
                route(box[0], *route_args, device)
                if where == "beside":
                    shared = shared_expert(late, state.front)
                self.experts.run(state.latent[:b], lists)
                if where == "tail":
                    shared = shared_expert(late, state.front)
                finalize(*out, device)
                down_done.record(state.front)
            main.wait_event(down_done)
            return self.tail(state.latent[:b], shared)
        with torch.cuda.stream(state.front):
            state.front.wait_event(fork)
            latent = module.routed_expert_down_proj(x)[0]
            latent.record_stream(main)
            down_done.record(state.front)
        logits = module.gate(x)[0]
        if plan[0] == "trtllm":
            main.wait_event(down_done)
            routed = self.routed.forward_monolithic(latent, logits)
            if where == "tail":
                shared = shared_expert(late)
        else:
            kind, route, route_args, finalize, out, ids = plan
            if kind == "listed":  # the plan is written from the latent: wait first
                main.wait_event(down_done)
                head, tail = route_args
                route(
                    to_cute(logits), *head, to_cute(latent), *tail, stream(state.device)
                )
            else:
                route(to_cute(logits), *route_args, stream(state.device))
                main.wait_event(down_done)
            if where == "beside":
                shared = shared_expert(late)
            self.experts.run(latent, ids)
            if where == "tail":
                shared = shared_expert(late)
            finalize(*out, stream(state.device))
            routed = state.latent[:b]
        return self.tail(routed, shared)

    def tail(self, routed, shared):
        """Latent all-reduce + RMSNorm, this rank's up-projection columns added into
        the shared expert's partial, then the all-reduce of the hidden rows."""
        module, main = self.module, torch.cuda.current_stream(self.state.device)
        normed = module.experts.allreduce_norm_latent_out(routed, self.norm)
        main.wait_event(self.state.events[2])  # the shared expert's rows
        start, width = self.columns
        shared.narrow(-1, start, width).addmm_(normed, self.up.t())
        return tensor_model_parallel_all_reduce(shared)
